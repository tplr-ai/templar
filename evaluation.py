import os
import sys
import json
import uuid
import torch
import shutil
import argparse
import asyncio
import traceback
import bittensor as bt
import wandb
<<<<<<< HEAD
import tplr
from tplr.hparams import load_hparams
from transformers import LlamaForCausalLM

# Verify environment variables are loaded
=======

from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 1. Load environment variables from .env
# ------------------------------------------------------------------------------
try:
    load_dotenv(dotenv_path=".env")  # Ensure the file name and path are correct
    print("Environment variables loaded from .env file.")
except Exception as e:
    print(f"Error loading environment variables: {e}")

>>>>>>> 0725e17 (making it compatible with main)
required_keys = [
    "R2_ACCOUNT_ID", 
    "R2_READ_ACCESS_KEY_ID", 
    "R2_READ_SECRET_ACCESS_KEY", 
    "R2_WRITE_ACCESS_KEY_ID", 
    "R2_WRITE_SECRET_ACCESS_KEY"
]
for key in required_keys:
    value = os.getenv(key)
    if value:
        print(f"{key} is set.")
    else:
        print(f"{key} is missing or not set.")

<<<<<<< HEAD

async def evaluate_latest_checkpoint(config):
    """
    Calls comms.get_latest_checkpoint(), which enumerates checkpoint files
    in the R2 bucket, selects the most recently modified, and returns
    (loaded_data, window).
    This approach matches the internal logic used by your comms.py
    and avoids the single-window limitation of comms.get(...).
=======
# ------------------------------------------------------------------------------
# 2. Add templar/src to sys.path & import tplr
# ------------------------------------------------------------------------------
templar_src_path = "/root/templar/src"
if templar_src_path not in sys.path:
    sys.path.insert(0, templar_src_path)
    print(f"Added '{templar_src_path}' to sys.path.")

try:
    import tplr
    print("tplr imported successfully.")
except Exception as e:
    print(f"Error during tplr import: {e}")
    traceback.print_exc()

from tplr.hparams import load_hparams
from transformers import LlamaForCausalLM

# If your new comms.py or its submodules rely on __version__ matching the checkpoint naming,
# ensure you import or define it here. For example:
try:
    from tplr import __version__  # or wherever __version__ is defined
except ImportError:
    # fallback if needed
    __version__ = "0.1.0"

# ------------------------------------------------------------------------------
# 3. Main async evaluation function
# ------------------------------------------------------------------------------
async def evaluate_latest_checkpoint(config):
    """
    Adapted for your new comms.py approach, which uses:
    async def get_latest_checkpoint(self) -> Optional[tuple[dict, int]]
    internally calling self._get_highest_stake_validator_bucket() 
    and scanning for files named checkpoint-<window>-<validator_uid>-v<__version__>.pt
>>>>>>> 0725e17 (making it compatible with main)
    """
    # 1. Load templar hparams
    hparams = load_hparams()
    if not config.tasks:
        tplr.logger.error("No tasks provided for evaluation. Exiting.")
        return
<<<<<<< HEAD
    # 2. Subtensor and metagraph
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    # 3. Create comms object (matching how get_miner_details.py instantiates comms)
=======

    # 2. Create / sync metagraph for up-to-date staking info
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    tplr.logger.info("Syncing metagraph...")
    metagraph.sync()  # Ensure .S is up-to-date (for argmax on stake)

    # 3. Create comms object
>>>>>>> 0725e17 (making it compatible with main)
    comms = tplr.comms.Comms(
        wallet=config.wallet,
        save_location="/tmp",
        key_prefix="model",
        config=config,
        netuid=config.netuid,
        metagraph=metagraph,
        hparams=hparams,
<<<<<<< HEAD
        uid=config.uid_for_eval,  # We can still pass this UID if your code references it
    )
    # 4. Sync commitments and update S3 buckets
    comms.commitments = comms.get_commitments_sync()
    comms.update_peers_with_buckets()

    # 5. Use the pre-existing “get_latest_checkpoint” method
    tplr.logger.info("Attempting to fetch the latest checkpoint via comms.get_latest_checkpoint()...")
    result = await comms.get_latest_checkpoint()
    if not result:
        tplr.logger.info("No valid checkpoint returned by comms.get_latest_checkpoint(). Exiting.")
        return

    checkpoint_data, checkpoint_window = result  # Typically a dict and an integer
    if not checkpoint_data or "model_state_dict" not in checkpoint_data:
        tplr.logger.error("Invalid or incomplete checkpoint data. Missing 'model_state_dict'. Exiting.")
        return
    tplr.logger.debug(f"Loaded checkpoint: {checkpoint_data.keys()}")
    tplr.logger.debug(f"Checkpoint window: {checkpoint_window}")
    global_step = checkpoint_data.get("global_step", 0)
    tplr.logger.info(
        f"Successfully fetched checkpoint from window={checkpoint_window}, global_step={global_step}"
=======
        uid=config.uid_for_eval,  # Not strictly used by get_latest_checkpoint
    )

    # 4. Sync commitments & set up R2 bucket references
    comms.commitments = comms.get_commitments_sync()
    comms.update_peers_with_buckets()

    # 5. Attempt to fetch the latest checkpoint
    tplr.logger.info("Attempting to fetch the latest checkpoint via comms.get_latest_checkpoint()...")
    result = await comms.get_latest_checkpoint()

    if not result:
        # Summarize. By now, comms.py has likely logged the detailed reason.
        tplr.logger.error(
            "comms.get_latest_checkpoint() returned None. Either no checkpoint was found "
            "or a download failure (OOM, timeout, etc.) occurred. See logs above for details."
        )
        return

    checkpoint_data, checkpoint_window = result  # Typically a dict + int

    # Double-check the loaded data
    if not checkpoint_data or "model_state_dict" not in checkpoint_data:
        tplr.logger.error(
            "We downloaded a checkpoint file, but it's missing 'model_state_dict'. "
            "Aborting evaluation."
        )
        return

    global_step = checkpoint_data.get("global_step", 0)
    tplr.logger.info(
        f"Successfully fetched checkpoint from window={checkpoint_window}, "
        f"global_step={global_step}"
>>>>>>> 0725e17 (making it compatible with main)
    )

    # 6. Load the model weights into a LlamaForCausalLM
    model = LlamaForCausalLM(config=hparams.model_config)
    try:
        model.load_state_dict(
            {k: v.cpu() for k, v in checkpoint_data["model_state_dict"].items()},
            strict=False
        )
    except KeyError as e:
        tplr.logger.error(f"Missing key in checkpoint data: {e}")
        return
    except Exception as e:
        tplr.logger.error(f"Error loading model weights: {e}")
        return

    model.to(config.device)

    # 7. Prepare local folder to run lm-eval out of
    eval_id = str(uuid.uuid4())[:8]
    model_dir = f"eval_{eval_id}_model"
    model_dir = os.path.join(os.getcwd(), model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)

    # 8. Save tokenizer
    hparams.tokenizer.save_pretrained(model_dir)

    # 9. Create local results path
    results_dir = os.path.join(model_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
<<<<<<< HEAD
    print(os.getcwd())
    print(results_dir)
=======
>>>>>>> 0725e17 (making it compatible with main)
    if not os.path.exists(results_dir):
        tplr.logger.error(f"Expected results folder not found: {results_dir}")
        return

    # 10. Construct lm-eval command
    tasks_arg = config.tasks
    lm_eval_cmd = (
        f"lm-eval "
        f"--model hf "
        f"--model_args pretrained={model_dir},tokenizer={model_dir} "
        f"--tasks {tasks_arg} "
        f"--device {config.device} "
        f"--batch_size {config.actual_batch_size} "
        f"--output_path {os.path.normpath(results_dir)}"
    )
    tplr.logger.info(f"Running lm-eval command: {lm_eval_cmd}")

    exit_code = os.system(lm_eval_cmd)
    if exit_code != 0:
        tplr.logger.error(f"lm-eval failed (exit code={exit_code}). Cleaning up and exiting.")
        shutil.rmtree(model_dir)
        return

    # 11. Parse results and log metrics
    try:
<<<<<<< HEAD
        # Find the most recent subfolder inside results_dir
=======
>>>>>>> 0725e17 (making it compatible with main)
        subfolders = [
            os.path.join(results_dir, f) for f in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, f))
        ]
        
        # Get the most recently modified subfolder
        latest_folder = max(subfolders, key=os.path.getctime)
        
<<<<<<< HEAD
        # Find the latest file inside that subfolder (which should be the result JSON file)
=======
        # Find the latest file inside that subfolder (likely the JSON with results)
>>>>>>> 0725e17 (making it compatible with main)
        latest_file = max(
            (os.path.join(latest_folder, f) for f in os.listdir(latest_folder)),
            key=os.path.getctime
        )
        with open(latest_file, "r") as f:
            eval_json = json.load(f)

        # Log or print the metrics
        for task_name, task_metrics in eval_json.get("results", {}).items():
<<<<<<< HEAD
            # Use appropriate metric key for evaluation (e.g., acc_norm,none or acc,none)
            metric_key = "acc,none" if task_name.lower() == "winogrande" else "acc_norm,none"
=======
            # Example metric key usage
            metric_key = "acc,none"
>>>>>>> 0725e17 (making it compatible with main)
            metric_val = task_metrics.get(metric_key)
            if metric_val is not None:
                metric_val = float(metric_val)
                tplr.logger.info(f"[{task_name}]: {metric_val}")
                if config.use_wandb:
                    wandb.log({f"eval/{task_name}": metric_val, "global_step": global_step})
            else:
                tplr.logger.warning(f"Metric {metric_key} missing in {task_name} results.")
<<<<<<< HEAD
=======

>>>>>>> 0725e17 (making it compatible with main)
        tplr.logger.info("Evaluation completed successfully.")
    except Exception as e:
        tplr.logger.error(f"Error reading or processing evaluation results: {e}")
        traceback.print_exc()
    finally:
<<<<<<< HEAD
        # shutil.rmtree(model_dir)
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Templar evaluation script via get_latest_checkpoint().")
    parser.add_argument("--project", type=str, default="templar", help="Wandb project name.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--tasks", type=str, default="arc_challenge,arc_easy,openbookqa,winogrande,piqa,hellaswag",
                        help="Comma-separated tasks for lm-eval.")
    parser.add_argument("--actual_batch_size", type=int, default=8, help="Batch size for lm-eval.")
    parser.add_argument("--uid_for_eval", type=int, default=1, help="UID of the miner to fetch checkpoint.")
    parser.add_argument("--netuid", type=int, default=3, help="Netuid to use.")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

=======
        # Optionally clean up the model dir.
        # shutil.rmtree(model_dir)
        torch.cuda.empty_cache()

# ------------------------------------------------------------------------------
# 4. Main CLI entry point
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Templar evaluation script via new comms.get_latest_checkpoint().")
    parser.add_argument("--project", type=str, default="templar", help="Wandb project name.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--tasks", type=str,
        default="arc_challenge,arc_easy,openbookqa,winogrande,piqa,hellaswag",
        help="Comma-separated tasks for lm-eval."
    )
    parser.add_argument("--actual_batch_size", type=int, default=8, help="Batch size for lm-eval.")
    parser.add_argument("--uid_for_eval", type=int, default=1, help="UID (not strictly needed by new approach).")
    parser.add_argument("--netuid", type=int, default=3, help="Netuid to use.")

    # Bittensor config
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
>>>>>>> 0725e17 (making it compatible with main)
    config = bt.config(parser)

    # Possibly override or define defaults
    config.netuid = getattr(config, "netuid", 3)
    config.subtensor.network = getattr(config.subtensor, "network", "finney")
    config.subtensor.chain_endpoint = getattr(
        config.subtensor, "chain_endpoint",
        "wss://entrypoint-finney.opentensor.ai:443"
    )

    config.wallet = bt.wallet(config=config)
<<<<<<< HEAD
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
=======
    config.device = "cuda:4" if torch.cuda.is_available() else "cpu"
>>>>>>> 0725e17 (making it compatible with main)

    init_config = {
        "project": config.project,
        "tasks": config.tasks,
        "uid_for_eval": config.uid_for_eval,
        "netuid": config.netuid,
        "device": config.device,
    }

    print(f"Network: {config.subtensor.network}")
    print(f"Endpoint: {config.subtensor.chain_endpoint}")
    print(f"NetUID: {config.netuid}")
    print(f"Device: {config.device}")
<<<<<<< HEAD
    # We do NOT print wallet hotkey to avoid KeyFileError if the file is missing
=======
>>>>>>> 0725e17 (making it compatible with main)

    if config.use_wandb:
        wandb.init(project=config.project, name="eval-run", config=init_config)

    try:
<<<<<<< HEAD
        tplr.logger.setLevel("DEBUG")  # Ensure all debug logs are shown
=======
        tplr.logger.setLevel("DEBUG")
>>>>>>> 0725e17 (making it compatible with main)
        asyncio.run(evaluate_latest_checkpoint(config))
    except KeyboardInterrupt:
        print("Evaluation aborted by user.")
    except Exception as e:
<<<<<<< HEAD
        print(f"Fatal error in eval1: {e}")
=======
        print(f"Fatal error in eval script: {e}")
>>>>>>> 0725e17 (making it compatible with main)
        traceback.print_exc()
    finally:
        if config.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    print("Debug: __main__ is being executed")
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 0725e17 (making it compatible with main)
