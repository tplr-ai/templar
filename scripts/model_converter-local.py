"""Local Model Converter

This script converts model checkpoints from local files to GGUF format and optionally uploads
them to HuggingFace Hub and/or Ollama. Instead of pulling checkpoints from the Bittensor network,
it accepts checkpoint paths as arguments.

Usage:
    python scripts/model_converter-local.py --checkpoint_path model_checkpoint.pt
    python scripts/model_converter-local.py --checkpoint_path model_checkpoint.pt --output_dir ./converted_models
    python scripts/model_converter-local.py --checkpoint_path model_checkpoint.pt --upload-hf --upload-ollama
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Optional

import torch
from transformers import LlamaForCausalLM

import tplr

from .model_converter import ( 
    upload_to_huggingface, 
    ensure_gguf_script_exists, 
    ensure_gguf_dependencies,
    generate_repo_id,
    upload_to_ollama,
    HF_AVAILABLE,
)

MODEL_PATH: str = "models/upload"
GGUF_SCRIPT_PATH: str = "scripts/convert_hf_to_gguf.py"
GGUF_SCRIPT_URL: str = (
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Local model converter script for GGUF conversion"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for model loading",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="converted_models",
        help="Directory to save converted models",
    )
    parser.add_argument(
        "--version_tag",
        type=str,
        default=None,
        help="Custom version tag for the converted model (default: auto-generate)",
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

    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: str) -> tuple[LlamaForCausalLM, dict]:
    """Load model from checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Tuple of (model, metadata)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tplr.logger.info(f"Loading checkpoint from {checkpoint_path}")

    hparams = tplr.load_hparams()
    model = LlamaForCausalLM(config=hparams.model_config)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        model_state = checkpoint
        metadata = {}

    model.load_state_dict(
        {
            k: v.to("cpu")
            for k, v in model_state.items()  # type: ignore
        }
    )
    model.to("cpu")  # type: ignore

    tplr.logger.info("Model loaded successfully")
    return model, metadata


def generate_version_tag(metadata: dict, custom_tag: Optional[str] = None) -> str:
    """Generate a version tag for the model.

    Args:
        metadata: Checkpoint metadata
        custom_tag: Custom version tag if provided

    Returns:
        Version string for the model
    """
    if custom_tag:
        return custom_tag

    version = tplr.__version__
    if "current_window" in metadata and "start_window" in metadata:
        global_step = int(metadata["current_window"]) - int(metadata["start_window"])
        return f"{version}+{global_step}"
    elif "global_step" in metadata:
        return f"{version}+{metadata['global_step']}"
    else:
        timestamp = int(time.time())
        return f"{version}+{timestamp}"


def save_versioned_model(
    model: LlamaForCausalLM, version_tag: str, output_dir: str
) -> str:
    """Save model with proper versioning.

    Args:
        model: The model to save
        version_tag: Version string for the model
        output_dir: Base output directory

    Returns:
        Path to the saved model directory
    """
    hparams = tplr.load_hparams()
    model_dir = os.path.join(output_dir, version_tag)
    os.makedirs(model_dir, exist_ok=True)

    model.save_pretrained(model_dir)
    hparams.tokenizer.save_pretrained(model_dir)

    tplr.logger.info(f"Saved model with version: {version_tag} to {model_dir}")
    return model_dir


def convert_hf_to_gguf(model_dir: str) -> str:
    """Convert a HuggingFace model to GGUF format.

    Args:
        model_dir: Path to the HuggingFace model directory

    Returns:
        Path to the converted GGUF model file
    """
    ensure_gguf_script_exists()
    ensure_gguf_dependencies()

    gguf_output = f"{model_dir}/model.gguf"

    command = [
        "uv",
        "run",
        "python",
        GGUF_SCRIPT_PATH,
        f"{model_dir}/",
        "--outfile",
        gguf_output,
    ]

    tplr.logger.info(f"Converting to GGUF format. Running command: {' '.join(command)}")

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
        error_msg = (
            f"GGUF conversion command failed: {e.stdout}, exit status: {e.returncode}"
        )
        tplr.logger.error(error_msg)
        raise RuntimeError(error_msg)

    if not os.path.exists(gguf_output):
        error_msg = (
            f"GGUF conversion file not found at expected location: {gguf_output}"
        )
        tplr.logger.error(error_msg)
        raise RuntimeError(error_msg)

    return gguf_output





# def check_huggingface_auth(hf_token: Optional[str] = None) -> bool:
#     """Check if HuggingFace authentication is available.

#     Args:
#         hf_token: Optional HF token to use instead of environment variable

#     Returns:
#         True if authentication is successful
#     """
#     if not HF_AVAILABLE:
#         return False
#     token = hf_token or os.getenv("HF_TOKEN")
#     if not token:
#         return False

#     try:
#         api = HfApi(token=token)
#         api.whoami()
#         return True
#     except Exception:
#         return False




def print_results(
    version_tag: str,
    model_dir: str,
    gguf_path: str,
    runtime: float,
    hf_success: bool = False,
    hf_repo_id: Optional[str] = None,
    ollama_success: bool = False,
    ollama_model_name: Optional[str] = None,
) -> None:
    """Print conversion and upload results in a readable format."""
    print("\n" + "=" * 50)
    print("CONVERSION & UPLOAD RESULTS")
    print("=" * 50)

    print(f"\nVersion: {version_tag}")
    print(f"Model Directory: {model_dir}")
    print(f"GGUF File: {gguf_path}")
    print(f"Total Time: {runtime:.2f} seconds")
    if os.path.exists(gguf_path):
        gguf_size = os.path.getsize(gguf_path) / (1024**3)  # GB
        print(f"GGUF File Size: {gguf_size:.2f} GB")
    if hf_success or ollama_success:
        print("\nUploads:")
        if hf_success and hf_repo_id:
            print(f"  ✅ HuggingFace: https://huggingface.co/{hf_repo_id}")
        elif hf_repo_id:
            print("  ❌ HuggingFace: Failed")

        if ollama_success and ollama_model_name:
            print(f"  ✅ Ollama: {ollama_model_name}")
            print(f"     Test with: ollama run {ollama_model_name}")
        elif ollama_model_name:
            print("  ❌ Ollama: Failed")

    print("=" * 50 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    try:
        start_time = time.time()
        model, metadata = load_checkpoint(args.checkpoint_path, args.device)

        if metadata:
            tplr.logger.info(f"Checkpoint metadata: {metadata}")
        version_tag = generate_version_tag(metadata, args.version_tag)

        model_dir = save_versioned_model(model, version_tag, args.output_dir)
        tplr.logger.info("Starting GGUF conversion...")
        gguf_path = convert_hf_to_gguf(model_dir)
        hf_success = False
        hf_repo_id = None
        ollama_success = False
        ollama_model_name = None
        if getattr(args, "upload_hf", False):
            hf_repo_id = generate_repo_id(version_tag, args.hf_repo_id)
            tplr.logger.info(f"Uploading to HuggingFace: {hf_repo_id}")

            hf_success = upload_to_huggingface(
                model_path=model_dir,
                repo_id=hf_repo_id,
                version=version_tag,
                private=args.private,
                commit_message=args.commit_message,
                dry_run=args.dry_run,
                hf_token=args.hf_token,
            )
        if getattr(args, "upload_ollama", False):
            ollama_model_name = args.ollama_model_name or f"templar-{version_tag}"
            tplr.logger.info(f"Uploading to Ollama: {ollama_model_name}")

            ollama_success = upload_to_ollama(
                model_path=model_dir,
                model_name=ollama_model_name,
                dry_run=args.dry_run,
            )

        runtime = time.time() - start_time
        print_results(
            version_tag=version_tag,
            model_dir=model_dir,
            gguf_path=gguf_path,
            runtime=runtime,
            hf_success=hf_success,
            hf_repo_id=hf_repo_id,
            ollama_success=ollama_success,
            ollama_model_name=ollama_model_name,
        )
        tplr.logger.info("Process completed successfully. All files preserved:")
        tplr.logger.info(f"  HuggingFace model: {model_dir}")
        tplr.logger.info(f"  GGUF model: {gguf_path}")
        if getattr(args, "upload_hf", False) and not hf_success:
            tplr.logger.error("HuggingFace upload failed")
            sys.exit(1)

        if getattr(args, "upload_ollama", False) and not ollama_success:
            tplr.logger.error("Ollama upload failed")
            sys.exit(1)

    except Exception as e:
        tplr.logger.error(f"Process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
