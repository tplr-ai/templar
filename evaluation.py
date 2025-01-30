# The MIT License (MIT)
# © 2024 templar.tech

import os
import json
import uuid
import torch
import shutil
import argparse
import asyncio
import traceback
import bittensor as bt
import wandb
from transformers import LlamaForCausalLM

import tplr
from tplr.hparams import load_hparams

# Constants
REQUIRED_ENV_KEYS = [
    "R2_ACCOUNT_ID",
    "R2_READ_ACCESS_KEY_ID",
    "R2_READ_SECRET_ACCESS_KEY",
    "R2_WRITE_ACCESS_KEY_ID",
    "R2_WRITE_SECRET_ACCESS_KEY",
]

DEFAULT_TASKS = "arc_challenge,arc_easy,openbookqa,hellaswag"


async def evaluate_latest_checkpoint(config):
    """Evaluates the latest checkpoint from the specified UID."""
    model_dir = None
    try:
        # Initialize components
        hparams = load_hparams()
        if not config.tasks:
            tplr.logger.error("No tasks provided for evaluation")
            return

        # Setup metagraph
        subtensor = bt.subtensor(config=config)
        metagraph = subtensor.metagraph(netuid=config.netuid)
        tplr.logger.debug("Syncing metagraph...")
        metagraph.sync()

        # Initialize comms
        comms = tplr.comms.Comms(
            wallet=config.wallet,
            save_location="/tmp",
            key_prefix="model",
            config=config,
            netuid=config.netuid,
            metagraph=metagraph,
            hparams=hparams,
            uid=config.uid_for_eval,
        )

        # Get latest checkpoint directly from the UID's bucket
        tplr.logger.debug(
            f"Fetching latest checkpoint for UID {config.uid_for_eval}..."
        )
        result = await comms._get_bucket_checkpoint(config.uid_for_eval)
        if not result:
            tplr.logger.info(f"No valid checkpoint found for UID {config.uid_for_eval}")
            return

        checkpoint_data, checkpoint_window = result
        if "model_state_dict" not in checkpoint_data:
            tplr.logger.error("Invalid checkpoint: missing model_state_dict")
            return

        global_step = checkpoint_data.get("global_step", 0)
        tplr.logger.info(
            f"Found checkpoint for UID {config.uid_for_eval}: window={checkpoint_window}, global_step={global_step}"
        )

        # Load model
        model = LlamaForCausalLM(config=hparams.model_config)
        model.load_state_dict(
            {k: v.cpu() for k, v in checkpoint_data["model_state_dict"].items()},
            strict=False,
        )
        model.to(config.device)

        # Setup evaluation directory
        eval_id = str(uuid.uuid4())[:8]
        model_dir = os.path.join(os.getcwd(), f"eval_{eval_id}_model")
        results_dir = os.path.join(model_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Save model and tokenizer
        model.save_pretrained(model_dir)
        hparams.tokenizer.save_pretrained(model_dir)

        # Run evaluation
        lm_eval_cmd = (
            f"lm-eval "
            f"--model hf "
            f"--model_args pretrained={model_dir},tokenizer={model_dir} "
            f"--tasks {config.tasks} "
            f"--device {config.device} "
            f"--batch_size {config.actual_batch_size} "
            f"--output_path {os.path.normpath(results_dir)}"
        )

        tplr.logger.debug(f"Running: {lm_eval_cmd}")
        if os.system(lm_eval_cmd) != 0:
            raise RuntimeError("lm-eval command failed")

        # Process results
        await _process_eval_results(results_dir, global_step, config.use_wandb)

    except Exception as e:
        tplr.logger.error(f"Evaluation failed: {str(e)}")
        tplr.logger.debug(traceback.format_exc())
    finally:
        # Add delay to ensure wandb has time to sync
        if config.use_wandb:
            tplr.logger.info("Waiting for WandB to sync...")
            await asyncio.sleep(5)  # Give wandb time to sync

        # Cleanup
        if model_dir and os.path.exists(model_dir):
            tplr.logger.debug(f"Cleaning up {model_dir}")
            shutil.rmtree(model_dir)
        torch.cuda.empty_cache()


async def _process_eval_results(results_dir: str, global_step: int, use_wandb: bool):
    """Helper to process and log evaluation results."""
    try:
        subfolders = [
            f
            for f in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, f))
        ]
        latest_folder = max(
            (os.path.join(results_dir, f) for f in subfolders), key=os.path.getctime
        )
        latest_file = max(
            (os.path.join(latest_folder, f) for f in os.listdir(latest_folder)),
            key=os.path.getctime,
        )

        with open(latest_file, "r") as f:
            results = json.load(f)

        for task_name, metrics in results.get("results", {}).items():
            tplr.logger.info(f"{task_name} metrics:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):  # Only log numeric values
                    formatted_value = (
                        f"{value:.4f}" if isinstance(value, float) else str(value)
                    )
                    tplr.logger.info(f"  {metric_name}: {formatted_value}")

                    if use_wandb:
                        wandb.log(
                            {
                                f"eval/{task_name}/{metric_name}": value,
                                "global_step": global_step,
                            }
                        )

    except Exception as e:
        tplr.logger.error(f"Failed to process results: {str(e)}")


def main():
    """Entry point for evaluation script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="eval")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS)
    parser.add_argument("--actual_batch_size", type=int, default=8)
    parser.add_argument("--uid_for_eval", type=int, default=1)
    parser.add_argument("--netuid", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Add bittensor args
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

    # Parse args directly first
    args = parser.parse_args()
    config = bt.config(parser)

    # Transfer our custom args to config
    config.use_wandb = args.use_wandb
    config.project = args.project
    config.device = args.device
    config.tasks = args.tasks
    config.actual_batch_size = args.actual_batch_size
    config.uid_for_eval = args.uid_for_eval

    # Set defaults
    config.netuid = getattr(config, "netuid", 3)
    config.subtensor.network = getattr(config.subtensor, "network", "finney")
    config.subtensor.chain_endpoint = getattr(
        config.subtensor, "chain_endpoint", "wss://entrypoint-finney.opentensor.ai:443"
    )

    config.wallet = bt.wallet(config=config)

    # Verify device is available
    if config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            tplr.logger.warning("CUDA not available, falling back to CPU")
            config.device = "cpu"
        else:
            gpu_id = int(config.device.split(":")[1])
            if gpu_id >= torch.cuda.device_count():
                tplr.logger.warning(
                    f"GPU {gpu_id} not available, falling back to cuda:0"
                )
                config.device = "cuda:0"

    tplr.logger.info(f"Using device: {config.device}")

    if config.use_wandb:
        tplr.logger.info("Initializing WandB...")
        run = wandb.init(
            project=config.project,
            name="eval-run",
            config={
                "project": config.project,
                "tasks": config.tasks,
                "uid_for_eval": config.uid_for_eval,
                "netuid": config.netuid,
                "device": config.device,
                "batch_size": config.actual_batch_size,
            },
        )
        tplr.logger.info(f"WandB run initialized: {run.name}")

    try:
        tplr.logger.setLevel("DEBUG")
        asyncio.run(evaluate_latest_checkpoint(config))
    except KeyboardInterrupt:
        tplr.logger.info("Evaluation interrupted")
    except Exception as e:
        tplr.logger.error(f"Fatal error: {str(e)}")
        tplr.logger.debug(traceback.format_exc())
    finally:
        if config.use_wandb:
            tplr.logger.info("Finishing WandB run...")
            wandb.finish()


if __name__ == "__main__":
    main()
