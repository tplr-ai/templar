"""Templar Autonomous Model Converter Service

This script implements an autonomous service that continuously monitors the latest
model checkpoints, saves them with proper versioning, converts them to GGUF format,
and optionally uploads them to HuggingFace Hub and/or Ollama. It runs on a fixed interval
(default 10 minutes), downloads the latest model checkpoint, saves it with a versioned format
that combines the package version and the global step in a semver-compatible way:
{version}-alpha+{global_step}, and converts it to GGUF format.

Key Features:
    - Automatic checkpoint detection and versioned saving
    - GGUF format conversion for compatibility with efficient inference engines
    - Optional upload to HuggingFace Hub and Ollama
    - Resource management
    - Service-oriented design for continuous operation

Environment Requirements:
    - Registered Bittensor wallet
    - R2 Dataset access credentials
    - Python scripts for GGUF conversion (scripts/convert_hf_to_gguf.py)
    - Python package: gguf (will be automatically installed if missing)
    - Optional: huggingface_hub for HuggingFace uploads
    - Optional: Ollama for local model deployment

Required Environment Variables:
    R2_DATASET_ACCOUNT_ID: R2 dataset account identifier (see miner documentation)
    R2_DATASET_BUCKET_NAME: R2 storage bucket name (see miner documentation)
    R2_DATASET_READ_ACCESS_KEY_ID: R2 read access key (see miner documentation)
    R2_DATASET_READ_SECRET_ACCESS_KEY: R2 secret access key (see miner documentation)

Optional Environment Variables (for uploads):
    HF_TOKEN: HuggingFace authentication token for Hub uploads

Usage Examples:
    Basic run:
        $ uv run ./scripts/model_converter.py

    Custom configuration:
        $ uv run scripts/model_converter.py \
            --netuid 3 \
            --device cuda \
            --conversion_interval 300

    With upload capabilities:
        $ export HF_TOKEN=hf_your_token_here
        $ uv run scripts/model_converter.py \
            --netuid 3 \
            --device cuda \
            --upload-hf \
            --upload-ollama

For additional environment setup, refer to the miner documentation:
https://github.com/tplr-ai/templar/blob/main/docs/miner.md
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import time
import urllib.request
from typing import Optional, Tuple

import bittensor as bt
from transformers.models.llama import LlamaForCausalLM

import tplr

try:
    from huggingface_hub import HfApi, create_repo

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

CHECKPOINT_DEFAULT_DIR: str = "checkpoints/"
MODEL_PATH: str = "models/upload"
DEFAULT_CONVERSION_INTERVAL: int = 60 * 10  # 10 mins default interval
GGUF_SCRIPT_PATH: str = "scripts/convert_hf_to_gguf.py"
GGUF_SCRIPT_URL: str = (
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py"
)


def config() -> bt.Config:
    """
    Parse command-line arguments and return a configuration object.
    """

    parser = argparse.ArgumentParser(
        description="Model Converter script. Use --help to display options.",
        add_help=True,
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=3,
        help="Bittensor network UID.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for model loading",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to save/load checkpoints",
    )
    parser.add_argument(
        "--conversion_interval",
        type=int,
        default=DEFAULT_CONVERSION_INTERVAL,
        help="Global steps between model conversion checks",
    )
    parser.add_argument(
        "--uid",
        type=int,
        default=None,
        help="Override the wallet's UID",
    )

    parser.add_argument(
        "--upload-hf",
        action="store_true",
        help="Upload converted model to HuggingFace Hub",
    )
    parser.add_argument(
        "--upload-ollama",
        action="store_true",
        help="Upload GGUF model to Ollama",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default=None,
        help="HuggingFace repository ID (e.g., 'username/model-name'). Auto-generated if not provided.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for authentication (overrides HF_TOKEN env var)",
    )
    parser.add_argument(
        "--ollama_model_name",
        type=str,
        default=None,
        help="Custom Ollama model name (default: auto-generate from version)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private HuggingFace repository",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="Custom commit message for HuggingFace upload",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )

    bt.subtensor.add_args(parser)
    parser.parse_args()
    return bt.config(parser)


class ModelConverter:
    """Templar Model Converter Component

    The ModelConverter is responsible for monitoring model checkpoints, converting them
    to GGUF format, and optionally uploading them to deployment platforms. It continuously
    checks for new checkpoints by window number, downloads them, and saves them with a
    versioned format combining the package version and the global step in a semver-compatible
    way: {version}-alpha+{global_step}

    Key Features:
        - Automatic checkpoint detection by window number
        - Versioned model saving using semver-compatible format
        - GGUF format conversion
        - Optional upload to HuggingFace Hub and Ollama
        - Resource management

    Workflow:
        1. Monitor blockchain for new checkpoints by window number
        2. Download and load checkpoint when detected
        3. Save model with versioned format
        4. Convert model to GGUF format
        5. Upload to HuggingFace Hub and/or Ollama if configured
        6. Clean up previous versions
        7. Wait for next checkpoint

    Attributes:
        config (bt.Config): Configuration object containing CLI arguments
        netuid (int): Network UID for the subnet
        model (LlamaForCausalLM): The language model being converted
        last_converted_window (int): Last converted window number
        last_block_number (int): Last processed block number
    """

    def __init__(self) -> None:
        self.config = config()
        if self.config.netuid is None:
            raise ValueError("No netuid provided")
        if self.config.device is None:
            raise ValueError("No device provided")
        self.checkpoint_path: str = (
            self.config.checkpoint_path or CHECKPOINT_DEFAULT_DIR
        )
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        self.netuid = self.config.netuid
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.netuid)
        self.hparams = tplr.load_hparams()
        self.wallet = bt.wallet(config=self.config)

        self.version = tplr.__version__

        self.uid = 1

        self.model = LlamaForCausalLM(config=self.hparams.model_config)
        self.model.to("cpu")

        self.tokenizer = self.hparams.tokenizer
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location="/tmp",
            key_prefix="model",
            config=self.config,
            netuid=self.netuid,
            metagraph=self.metagraph,
            hparams=self.hparams,
            uid=self.uid,
        )

        self.buckets = self.comms.get_all_buckets()
        self.last_converted_window = 0
        self.stop_event = asyncio.Event()
        self.last_block_number = 0

        self.metrics_logger = tplr.metrics.MetricsLogger(
            prefix="C",
            uid=str(self.uid),
            config=self.config,
            role="converter",
            group="conversions",
            job_type="convert",
        )

    async def update_state(self) -> None:
        """
        Refresh the metagraph and bucket information.
        """
        self.metagraph = self.subtensor.metagraph(netuid=self.netuid)
        self.buckets = self.comms.get_all_buckets()

    async def load_latest_model(self) -> Tuple[bool, dict, int, int]:
        """Load and prepare the latest model checkpoint for conversion.

        This method:
        1. Fetches the latest checkpoint from storage
        2. Verifies checkpoint validity
        3. Loads model weights
        4. Updates internal state trackers

        Returns:
            Tuple containing:
            - success (bool): Whether loading succeeded
            - checkpoint_data (dict): Checkpoint metadata
            - checkpoint_window (int): Window number of checkpoint
            - global_step (int): Global training step
        """
        result = await self.comms.get_latest_checkpoint(version=self.version)  # type: ignore
        if not result:
            tplr.logger.error(
                f"No valid checkpoints found. Check bucket: {getattr(self.comms, 'bucket_name', 'unknown')}, "
                f"key_prefix: {self.comms.key_prefix}"
            )
            return (False, {}, 0, 0)

        tplr.logger.info(f"[DEBUG] get_latest_checkpoint() result: {result}")

        checkpoint_data, _ = result
        tplr.logger.info(f"[DEBUG] Checkpoint data: {checkpoint_data}")

        checkpoint_start_window = checkpoint_data.get("start_window")
        checkpoint_current_window = checkpoint_data.get("current_window", None)

        if checkpoint_start_window is None or checkpoint_current_window is None:
            tplr.logger.error("Checkpoint missing start_window or current_window info")
            return (False, checkpoint_data, 0, 0)

        if int(checkpoint_current_window) <= self.last_converted_window:
            tplr.logger.info(
                f"Checkpoint already converted (checkpoint window: {checkpoint_current_window}, "
                f"last converted: {self.last_converted_window})."
            )
            return (False, checkpoint_data, int(checkpoint_current_window), 0)

        tplr.logger.info(
            f"Loading model from checkpoint (window: {checkpoint_current_window})"
        )
        self.model.load_state_dict(
            {
                k: v.to("cpu")
                for k, v in checkpoint_data["model_state_dict"].items()  # type: ignore
            }
        )
        self.model.to("cpu")  # type: ignore

        self.momentum = checkpoint_data["momentum"]

        global_step = int(checkpoint_current_window) - int(checkpoint_start_window)

        tplr.logger.info(
            f"Loaded checkpoint (start_window={checkpoint_start_window}, "
            f"current_window={checkpoint_current_window}, global_step={global_step})"
        )

        return (True, checkpoint_data, int(checkpoint_current_window), global_step)

    def save_versioned_model(self, global_step: int) -> tuple[str, str]:
        """Save model with proper versioning.

        Creates a versioned format combining the package version and the global step
        in a semver-compatible way, following the format: {version}-alpha+{global_step}

        Args:
            global_step: Current global step for versioning

        Returns:
            tuple:
                - The version string used for saving
                - The full path to the saved model directory
        """
        version_string = f"{self.version}-alpha+{global_step}"

        version_dir = os.path.join(MODEL_PATH, version_string)
        os.makedirs(version_dir, exist_ok=True)

        self.model.save_pretrained(version_dir)
        self.tokenizer.save_pretrained(version_dir)

        tplr.logger.info(f"Saved model with version: {version_string} to {version_dir}")

        return version_string, version_dir

    def delete_previous_models(self, current_model_dir: str) -> None:
        """
        Delete all previous model versions from MODEL_PATH, except the current one.

        Args:
            current_model_dir: Path to the current model directory that should be kept
        """
        try:
            if not os.path.exists(MODEL_PATH):
                return

            model_dirs = [
                os.path.join(MODEL_PATH, d)
                for d in os.listdir(MODEL_PATH)
                if os.path.isdir(os.path.join(MODEL_PATH, d))
            ]

            dirs_to_delete = [d for d in model_dirs if d != current_model_dir]

            if not dirs_to_delete:
                tplr.logger.info("No previous model versions to delete")
                return

            for dir_path in dirs_to_delete:
                tplr.logger.info(f"Deleting previous model version: {dir_path}")
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path, ignore_errors=True)

            tplr.logger.info(f"Deleted {len(dirs_to_delete)} previous model versions")
        except Exception as e:
            tplr.logger.warning(f"Error deleting previous models: {e}")

    def _convert_hf_to_gguf(self, path: str) -> str:
        """
        Convert a HuggingFace model to GGUF format using the `convert_hf_to_gguf.py` script.

        Args:
            path (str): Path to the model directory.

        Returns:
            str: Path to the converted GGUF model file.

        Raises:
            RuntimeError: If the conversion fails for any reason.
        """
        # Ensure both the script and dependencies are available
        ensure_gguf_script_exists()
        ensure_gguf_dependencies()

        gguf_output = f"{path}/model.gguf"

        command = [
            "uv",
            "run",
            "python",
            GGUF_SCRIPT_PATH,
            f"{path}/",
            "--outfile",
            gguf_output,
        ]

        tplr.logger.info(
            f"Converting to GGUF format. Running command: {' '.join(command)}"
        )

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
            tplr.logger.info(result.stdout)
            tplr.logger.info(f"GGUF conversion successful: {gguf_output}")
        except subprocess.CalledProcessError as e:
            error_msg = f"GGUF conversion command failed: {e.stdout}, exit status: {e.returncode}"
            tplr.logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not os.path.exists(gguf_output):
            error_msg = (
                f"GGUF conversion file not found at expected location: {gguf_output}"
            )
            tplr.logger.error(error_msg)
            raise RuntimeError(error_msg)

        return gguf_output

    async def _convert(self) -> Optional[int]:
        """Execute model save, GGUF conversion, and optional upload process.

        Workflow:
        1. Load model from checkpoint
        2. Create versioned directory
        3. Save model and tokenizer to versioned location
        4. Convert model to GGUF format (raises error on failure)
        5. Upload to HuggingFace Hub and/or Ollama if configured
        6. Log metrics about the conversion and upload operations
        7. Clean up previous versions

        Returns:
            Optional[int]: Global step number if successful, None on failure

        Raises:
            RuntimeError: If GGUF conversion fails
        """
        self.comms.commitments = await self.comms.get_commitments()
        self.comms.update_peers_with_buckets()

        block_number = self.subtensor.get_current_block() - 1

        tplr.logger.info(f"Looking for new checkpoint (block: {block_number})")

        (
            success,
            checkpoint_data,
            checkpoint_window,
            global_step,
        ) = await self.load_latest_model()

        if not success:
            tplr.logger.info(
                f"No new checkpoint to convert (last converted window: {self.last_converted_window})"
            )
            return global_step

        tplr.logger.info(
            f"Starting model conversion at global step {global_step} (checkpoint window: {checkpoint_window})"
        )

        version_string, model_dir = self.save_versioned_model(global_step)

        tplr.logger.info(f"Converting model to GGUF format at {model_dir}")
        gguf_path = self._convert_hf_to_gguf(model_dir)

        if not gguf_path:
            error_msg = "GGUF conversion failed, aborting process"
            tplr.logger.error(error_msg)
            raise RuntimeError(error_msg)

        hf_success = False
        ollama_success = False
        hf_repo_id = None
        ollama_model_name = None

        if getattr(self.config, "upload_hf", False):
            hf_repo_id = generate_repo_id(
                version_string, getattr(self.config, "hf_repo_id", None)
            )
            tplr.logger.info(f"Uploading to HuggingFace: {hf_repo_id}")

            hf_success = upload_to_huggingface(
                model_path=model_dir,
                repo_id=hf_repo_id,
                version=version_string,
                private=getattr(self.config, "private", False),
                commit_message=getattr(self.config, "commit_message", None),
                dry_run=getattr(self.config, "dry_run", False),
                hf_token=getattr(self.config, "hf_token", None),
            )

        if getattr(self.config, "upload_ollama", False):
            ollama_model_name = (
                getattr(self.config, "ollama_model_name", None)
                or f"templar-{version_string}"
            )
            tplr.logger.info(f"Uploading to Ollama: {ollama_model_name}")

            ollama_success = upload_to_ollama(
                model_path=model_dir,
                model_name=ollama_model_name,
                dry_run=getattr(self.config, "dry_run", False),
            )

        self.metrics_logger.log(
            measurement="model_conversion",
            tags={
                "global_step": global_step,
                "window": checkpoint_window,
                "block": block_number,
                "version": version_string,
            },
            fields={
                "conversion_timestamp": time.time(),
                "gguf_converted": 1.0,
                "hf_uploaded": 1.0 if hf_success else 0.0,
                "ollama_uploaded": 1.0 if ollama_success else 0.0,
            },
        )

        self.last_converted_window = checkpoint_window
        self.last_block_number = block_number

        upload_info = ""
        if hf_success:
            upload_info += f", HF: {hf_repo_id}"
        if ollama_success:
            upload_info += f", Ollama: {ollama_model_name}"

        success_message = (
            f"Successfully converted model (window: {checkpoint_window}, "
            f"global_step: {global_step}, block: {block_number}, version: {version_string}"
            f", GGUF: {gguf_path}{upload_info})"
        )
        tplr.logger.info(success_message)

        tplr.logger.info("Cleaning up previous model versions...")
        self.delete_previous_models(model_dir)

        return global_step

    async def run(self) -> None:
        """Main conversion loop.

        Continuously:
        1. Check for new checkpoints by window number and block
        2. Trigger model conversion when new checkpoint detected
        3. Handle interrupts and errors
        4. Maintain conversion interval
        """
        try:
            self.comms.start_commitment_fetcher()
            self.comms.start_background_tasks()

            while not self.stop_event.is_set():
                await self.update_state()

                latest_block = self.subtensor.get_current_block()
                start_window = await self.comms.get_start_window()

                if start_window is not None and (
                    latest_block > self.last_block_number
                    or start_window > self.last_converted_window
                ):
                    tplr.logger.info(
                        f"New checkpoint detected (block: {latest_block}, window: {start_window}), executing conversion..."
                    )
                    await self._convert()
                else:
                    tplr.logger.info(
                        f"No new checkpoint available (block: {latest_block}/{self.last_block_number}, "
                        f"window: {start_window}/{self.last_converted_window})"
                    )
                await asyncio.sleep(self.config.conversion_interval)  # type: ignore
        except KeyboardInterrupt:
            tplr.logger.info("Conversion process interrupted by user")
            self.stop_event.set()
        except Exception as e:
            tplr.logger.error(f"Conversion process failed: {e}")

    def cleanup(self) -> None:
        """
        Cleanup resources before exit.
        """
        self.stop_event.set()


def ensure_model_paths_exist() -> None:
    """
    Ensure that the MODEL_PATH and scripts directories exist.

    This function creates the necessary directories for model storage and
    verifies that they are available before the service starts.
    """
    os.makedirs(MODEL_PATH, exist_ok=True)
    tplr.logger.info(f"Ensured model directory exists: {MODEL_PATH}")

    os.makedirs(os.path.dirname(GGUF_SCRIPT_PATH), exist_ok=True)


def ensure_gguf_dependencies() -> None:
    """
    Ensure that the gguf Python package is installed.

    Raises:
        RuntimeError: If unable to install the package.
    """
    tplr.logger.info("Checking for gguf Python package...")

    try:
        __import__("gguf")
        tplr.logger.info("gguf package is already installed")
        return
    except ImportError:
        tplr.logger.info("gguf package not found, installing...")

    try:
        result = subprocess.run(
            ["pip3", "install", "gguf"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        tplr.logger.info(f"Successfully installed gguf package: {result.stdout}")
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to install gguf package: {e.stdout}"
        tplr.logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        __import__("gguf")
        tplr.logger.info("Verified gguf package is now installed")
    except ImportError:
        error_msg = "Failed to import gguf package after installation"
        tplr.logger.error(error_msg)
        raise RuntimeError(error_msg)


def ensure_gguf_script_exists() -> None:
    """
    Check if the GGUF conversion script exists, and download it if not.

    Raises:
        RuntimeError: If unable to download the script.
    """
    if os.path.exists(GGUF_SCRIPT_PATH):
        tplr.logger.info(f"GGUF conversion script found at {GGUF_SCRIPT_PATH}")
        return

    tplr.logger.info(
        f"GGUF conversion script not found, downloading from {GGUF_SCRIPT_URL}"
    )

    try:
        urllib.request.urlretrieve(GGUF_SCRIPT_URL, GGUF_SCRIPT_PATH)
        tplr.logger.info(
            f"Successfully downloaded GGUF conversion script to {GGUF_SCRIPT_PATH}"
        )
    except Exception as e:
        error_msg = f"Failed to download GGUF conversion script: {e}"
        tplr.logger.error(error_msg)
        raise RuntimeError(error_msg)

    if not os.path.exists(GGUF_SCRIPT_PATH) or os.path.getsize(GGUF_SCRIPT_PATH) == 0:
        error_msg = f"Downloaded GGUF script at {GGUF_SCRIPT_PATH} is missing or empty"
        tplr.logger.error(error_msg)
        raise RuntimeError(error_msg)


def generate_repo_id(version: str, custom_repo_id: Optional[str] = None) -> str:
    """Generate HuggingFace repository ID.

    Args:
        version: Version string
        custom_repo_id: Custom repo ID if provided

    Returns:
        Repository ID string
    """
    if custom_repo_id:
        return custom_repo_id
    safe_version = version.replace("+", "-").replace(".", "-")
    return f"templar-model-{safe_version}"


def check_huggingface_auth(hf_token: Optional[str] = None) -> bool:
    """Check if HuggingFace authentication is available.

    Args:
        hf_token: Optional HF token to use instead of environment variable

    Returns:
        True if authentication is successful
    """
    if not HF_AVAILABLE:
        return False
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        return False

    try:
        api = HfApi(token=token)
        api.whoami()
        return True
    except Exception:
        return False


def upload_to_huggingface(
    model_path: str,
    repo_id: str,
    version: str,
    private: bool = False,
    commit_message: Optional[str] = None,
    dry_run: bool = False,
    hf_token: Optional[str] = None,
) -> bool:
    """Upload model to HuggingFace Hub.

    Args:
        model_path: Path to the model directory
        repo_id: HuggingFace repository ID
        version: Version tag
        private: Whether to create private repository
        commit_message: Custom commit message
        dry_run: Whether to perform dry run
        hf_token: Optional HF token for authentication

    Returns:
        Success status
    """
    if not HF_AVAILABLE:
        tplr.logger.error(
            "HuggingFace Hub library not available. Install with: pip install huggingface_hub"
        )
        return False

    if not check_huggingface_auth(hf_token):
        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            tplr.logger.error(
                "HuggingFace token not found. Provide via --hf_token or set HF_TOKEN environment variable"
            )
        else:
            tplr.logger.error(
                "HuggingFace authentication failed. Verify your token or run: huggingface-cli login"
            )
        return False

    if dry_run:
        tplr.logger.info(
            f"[DRY RUN] Would upload {model_path} to HuggingFace repo: {repo_id}"
        )
        return True

    try:
        token = hf_token or os.getenv("HF_TOKEN")
        api = HfApi(token=token)
        try:
            create_repo(repo_id, private=private, exist_ok=True, token=token)
            tplr.logger.info(f"Repository ready: {repo_id}")
        except Exception as e:
            tplr.logger.error(f"Failed to create repository: {e}")
            return False
        if not commit_message:
            commit_message = f"Upload model version {version}"

        tplr.logger.info(f"Uploading model to HuggingFace: {repo_id}")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            commit_message=commit_message,
            ignore_patterns=["*.gguf"],
        )

        tplr.logger.info(
            f"Successfully uploaded to HuggingFace: https://huggingface.co/{repo_id}"
        )
        return True

    except Exception as e:
        tplr.logger.error(f"HuggingFace upload failed: {e}")
        return False


def check_ollama_available() -> bool:
    """Check if Ollama is available."""
    try:
        subprocess.run(
            ["ollama", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def create_ollama_modelfile(gguf_path: str, model_name: str, temp_dir: str) -> str:
    """Create Ollama Modelfile for GGUF model.

    Args:
        gguf_path: Path to GGUF file
        model_name: Name for the Ollama model
        temp_dir: Temporary directory for Modelfile

    Returns:
        Path to created Modelfile
    """
    modelfile_path = os.path.join(temp_dir, "Modelfile")

    modelfile_content = f"""FROM {gguf_path}

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40

SYSTEM \"\"\"You are a helpful AI assistant based on the Templar model.\"\"\"
"""

    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    return modelfile_path


def upload_to_ollama(
    model_path: str,
    model_name: str,
    dry_run: bool = False,
) -> bool:
    """Upload GGUF model to Ollama.

    Args:
        model_path: Path to the model directory
        model_name: Name for the Ollama model
        dry_run: Whether to perform dry run

    Returns:
        Success status
    """
    if not check_ollama_available():
        tplr.logger.error("Ollama not found. Install from: https://ollama.ai")
        return False

    gguf_path = os.path.join(model_path, "model.gguf")
    if not os.path.exists(gguf_path):
        tplr.logger.error(f"GGUF file not found: {gguf_path}")
        return False

    if dry_run:
        tplr.logger.info(
            f"[DRY RUN] Would upload {gguf_path} to Ollama as: {model_name}"
        )
        return True

    try:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            modelfile_path = create_ollama_modelfile(gguf_path, model_name, temp_dir)
            tplr.logger.info(f"Creating Ollama model: {model_name}")
            subprocess.run(
                ["ollama", "create", model_name, "-f", modelfile_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            tplr.logger.info(f"Successfully created Ollama model: {model_name}")
            tplr.logger.info(f"Test with: ollama run {model_name}")
            return True

    except subprocess.CalledProcessError as e:
        tplr.logger.error(f"Ollama upload failed: {e.stderr}")
        return False
    except Exception as e:
        tplr.logger.error(f"Ollama upload failed: {e}")
        return False


def main() -> None:
    """
    Entry point for the model converter.
    """
    ensure_model_paths_exist()
    ensure_gguf_script_exists()
    ensure_gguf_dependencies()

    converter = ModelConverter()
    try:
        asyncio.run(converter.run())
    except Exception as e:
        tplr.logger.error(f"Model converter terminated with error: {e}")
    finally:
        converter.cleanup()


if __name__ == "__main__":
    main()
