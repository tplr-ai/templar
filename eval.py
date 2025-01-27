import os
import sys
import json
import uuid
import time
import torch
import shutil
import argparse
import asyncio
import traceback

import bittensor as bt
import wandb

# Templar imports
import tplr
from tplr.hparams import load_hparams
from transformers import LlamaForCausalLM

async def evaluate_latest_checkpoint(config):
    """
    Fetch the latest checkpoint via templar.comms, run eval using lm-eval,
    and optionally log to wandb.
    """
    # 1. Load base Templar hparams
    hparams = load_hparams()

    # 2. Initialize subtensor & metagraph
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)

    # 3. Create comms object
    comms = tplr.comms.Comms(
        wallet=config.wallet,
        save_location="/tmp",
        key_prefix="model",
        config=config,
        netuid=config.netuid,
        metagraph=metagraph,
        hparams=hparams,
        uid=999999,  # Dummy UID for the evaluation node
    )

    # 4. Sync commitments and update buckets (mimics get_miner_details.py)
    comms.commitments = comms.get_commitments_sync()
    comms.update_peers_with_buckets()

    # 5. Optionally set the current window (helpful if your code needs it)
    comms.current_window = int(subtensor.block / hparams.blocks_per_window)

    # 6. Fetch the latest checkpoint
    tplr.logger.info("Fetching latest checkpoint from the highest-stake validator ...")
    checkpoint_result = await comms.get_latest_checkpoint()
    if not checkpoint_result:
        tplr.logger.info("No valid checkpoint found. Exiting.")
        return

    checkpoint_data, checkpoint_window = checkpoint_result
    if not checkpoint_data or "model_state_dict" not in checkpoint_data:
        tplr.logger.error("Invalid checkpoint data. Missing 'model_state_dict'. Exiting.")
        return

    global_step = checkpoint_data.get("global_step", 0)
    tplr.logger.info(f"Fetched checkpoint from window {checkpoint_window}, global_step={global_step}.")

    # 7. Load the model from the checkpoint
    model = LlamaForCausalLM(config=hparams.model_config)
    try:
        model.load_state_dict({k: v.cpu() for k, v in checkpoint_data["model_state_dict"].items()}, strict=False)
    except Exception as e:
        tplr.logger.error(f"Error loading model weights: {e}")
        return
    model.to(config.device)

    # 8. Save model locally for lm-eval
    eval_id = str(uuid.uuid4())[:8]
    model_dir = f"eval_{eval_id}_model"
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)

    # 9. Save tokenizer
    hparams.tokenizer.save_pretrained(model_dir)

    # 10. Construct results dir
    results_dir = os.path.join(model_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 11. Build and run the lm-eval command
    tasks_arg = config.tasks
    lm_eval_cmd = (
        f"lm-eval "
        f"--model hf "
        f"--model_args pretrained={model_dir},tokenizer={model_dir} "
        f"--tasks {tasks_arg} "
        f"--device {config.device} "
        f"--batch_size {config.actual_batch_size} "
        f"--output_path {results_dir}"
    )
    tplr.logger.info(f"Running lm-eval command: {lm_eval_cmd}")
    exit_code = os.system(lm_eval_cmd)
    if exit_code != 0:
        tplr.logger.error(f"lm-eval failed with exit code {exit_code}. Cleanup and return.")
        shutil.rmtree(model_dir)
        return

    # 12. Parse results
    try:
        results_subdir = os.path.join(results_dir, "models__eval")
        if not os.path.exists(results_subdir):
            tplr.logger.error(f"Results folder {results_subdir} not found.")
            shutil.rmtree(model_dir)
            return

        latest_file = max([os.path.join(results_subdir, f) for f in os.listdir(results_subdir)], key=os.path.getctime)
        with open(latest_file, "r") as f:
            eval_json = json.load(f)

        for task_name, task_metrics in eval_json.get("results", {}).items():
            if task_name.lower() == "winogrande":
                metric_key = "acc,none"
            else:
                metric_key = "acc_norm,none"

            metric_val = task_metrics.get(metric_key)
            if metric_val is not None:
                metric_val = float(metric_val)
                tplr.logger.info(f"[{task_name}]: {metric_val}")
                if config.use_wandb:
                    wandb.log({f"eval/{task_name}": metric_val, "global_step": global_step})
            else:
                tplr.logger.warning(f"Metric {metric_key} not found in {task_name} results.")

        tplr.logger.info("Evaluation completed successfully.")
    except Exception as e:
        tplr.logger.error(f"Error reading/processing evaluation results: {e}")
        traceback.print_exc()
    finally:
        shutil.rmtree(model_dir)
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Templar evaluation script.")
    parser.add_argument("--project", type=str, default="templar", help="Wandb project name.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--tasks", type=str, default="arc_challenge,arc_easy,openbookqa,winogrande,piqa,hellaswag",
                        help="Comma-separated list of tasks for lm-eval.")
    parser.add_argument("--actual_batch_size", type=int, default=8, help="Batch size for lm-eval tasks.")
    parser.add_argument("--netuid", type=int, default=3, help="Default netuid: 3.")
    bt.subtensor.add_args(parser)
    config = bt.config(parser)

    # Hard-code or set defaults
    config.netuid = getattr(config, "netuid", 3)
    config.subtensor.network = getattr(config.subtensor, "network", "finney")
    config.subtensor.chain_endpoint = getattr(config.subtensor, "chain_endpoint", "wss://entrypoint-finney.opentensor.ai:443")
    config.wallet = bt.wallet(config=config)
    config.device = "cuda:4" if torch.cuda.is_available() else "cpu"

    init_config = {
        "project": config.project,
        "tasks": config.tasks,
        "netuid": config.netuid,
        "device": config.device,
    }

    print(f"Configured network: {config.subtensor.network}")
    print(f"Chain endpoint: {config.subtensor.chain_endpoint}")
    print(f"NetUID: {config.netuid}")
    print(f"Device: {config.device}")
    print(f"Wallet hotkey: {config.wallet.hotkey.ss58_address}")

    if config.use_wandb:
        wandb.init(
            project=config.project,
            name="eval-run",
            config=init_config
        )

    try:
        asyncio.run(evaluate_latest_checkpoint(config))
    except KeyboardInterrupt:
        print("Evaluation aborted by user.")
    except Exception as e:
        print(f"Fatal error in eval: {e}")
        traceback.print_exc()
    finally:
        if config.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()