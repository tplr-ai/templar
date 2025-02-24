import os
import json
import shutil
import sys
import torch
import asyncio
import argparse
import wandb
import time
import tplr
import bittensor as bt

from typing import Any, Optional, Tuple
from tplr.metrics import MetricsLogger
from tplr.chain import ChainManager
from transformers.models.llama import LlamaForCausalLM
from tplr import __version__
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)

CHECKPOINT_DEFAULT_DIR: str = "checkpoints/"
MODEL_PATH: str = "models/eval"
DEFAULT_EVAL_INTERVAL: int = 60 * 10  # 10 mins default interval


def config() -> bt.Config:
    """
    Parse command-line arguments and return a configuration object.
    """

    parser = argparse.ArgumentParser(
        description="Evaluator script. Use --help to display options.",
        add_help=True,
    )
    parser.add_argument(
        "--project",
        type=str,
        default="templar",
        help="Optional wandb project name",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=3,
        help="Bittensor network UID.",
    )
    parser.add_argument(
        "--actual_batch_size",
        type=int,
        default=8,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_challenge,arc_easy,openbookqa,winogrande,piqa,hellaswag",
        help="Comma-separated list of tasks to evaluate",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to save/load checkpoints",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=DEFAULT_EVAL_INTERVAL,
        help="Global steps between evaluations",
    )
    parser.add_argument(
        "--uid",
        type=int,
        default=None,
        help="Override the wallet's UID",
    )

    bt.subtensor.add_args(parser)
    parser.parse_args()
    return bt.config(parser)


class Evaluator:
    """
    Evaluator periodically checks for new checkpoints, runs benchmark evaluations, and logs results.
    Also provides a load_model method that uses the chain to fetch the most recent global_step and block_number.
    """

    def __init__(self) -> None:
        self.config = config()
        if self.config.netuid is None:
            raise ValueError("No netuid provided")
        if self.config.device is None:
            raise ValueError("No device provided")
        # Use constant for default checkpoint directory.
        self.checkpoint_path: str = (
            self.config.checkpoint_path or CHECKPOINT_DEFAULT_DIR
        )
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        self.netuid = self.config.netuid
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.netuid)
        self.hparams = tplr.load_hparams()
        self.wallet = bt.wallet(config=self.config)

        self.uid = (
            self.config.uid
            if self.config.uid is not None
            else self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        )

        self.model = LlamaForCausalLM(config=self.hparams.model_config)
        self.model.to(self.config.device)

        self.tokenizer = self.hparams.tokenizer
        self.compressor = tplr.compress.CompressDCT()
        self.optimizer = SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        self.transformer = tplr.compress.TransformDCT(
            self.model, target_chunk=self.hparams.target_chunk
        )

        self.warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=250,
        )
        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10000,
            T_mult=2,
            eta_min=self.hparams.learning_rate * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[250],
        )

        # Init compression
        self.transformer = tplr.compress.TransformDCT(
            self.model,
            target_chunk=self.hparams.target_chunk,
        )

        self.momentum = {}
        self.totalks = {}
        self.xshapes = {}
        for n, p in self.model.named_parameters():
            self.momentum[n] = torch.zeros_like(p)
            _, _, xshape, totalk = self.compressor.compress(
                self.transformer.encode(self.momentum[n]), self.hparams.topk_compression
            )
            self.xshapes[n] = xshape
            self.totalks[n] = totalk

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
        self.last_eval_step = 0
        self.stop_event = asyncio.Event()
        self.last_block_number = 0

        # Metrics logging - see neurons/validator.py & miner.py as it needs to be the same
        self.metrics_logger = tplr.metrics.MetricsLogger(
            host="uaepr2itgl-tzxeth774u3fvf.timestream-influxdb.us-east-2.on.aws",
            port=8086,
            database="tplr",
            token=os.environ.get("INFLUXDB_TOKEN"),
            org="templar",
        )
        self.wandb_run = wandb.init(project=self.config.project)

    async def update_state(self) -> None:
        """
        Refresh the metagraph and bucket information.
        """
        self.metagraph = self.subtensor.metagraph(netuid=self.netuid)
        self.buckets = self.comms.get_all_buckets()

    async def load_model(
        self,
        block_number: int,
        current_window: int,
    ) -> Tuple[bool, dict, int, int]:
        """Load the most recent model checkpoint from the chain.

        Returns:
            Tuple[bool, dict, int, int, int]: A tuple containing the following:
                - success: A boolean indicating whether the model was successfully loaded.
                - checkpoint_data: The checkpoint data dictionary.
                - checkpoint_current_window: The current window of the checkpoint.
                - global_step: The global step of the checkpoint.
                - block_number: The block number of the checkpoint.
        """
        result = await self.comms.get_latest_checkpoint()

        if not result:
            tplr.logger.error("No valid checkpoints found")
            return (False, {}, 0, 0)

        checkpoint_data, _ = result

        if block_number <= self.last_block_number:
            tplr.logger.info(
                f"No new checkpoint available (current block: {block_number}, last: {self.last_block_number})."
            )
            return (False, checkpoint_data, 0, 0)

        batch_size = int(self.config.actual_batch_size)  # type: ignore
        windows_to_catch_up = range(
            checkpoint_data.get("current_window"), current_window + 1
        )

        for i in range(0, len(windows_to_catch_up), batch_size):
            batch_windows = list(windows_to_catch_up)[i : i + batch_size]

            # Launch gathers in parallel
            tasks = [
                self.comms.gather(
                    my_uid=self.uid,
                    uids=[str(uid) for uid in self.comms.peers if uid != self.uid],
                    window=w,
                    key="gradient",
                    timeout=72,
                    device="cpu",
                    local=False,
                    stale_retention=100,
                    totalks=self.totalks,
                )
                for w in batch_windows
            ]
            await asyncio.gather(*tasks)

        tplr.logger.info("Loaded commitments")

        self.model.load_state_dict(
            {
                k: v.to(self.config.device)
                for k, v in checkpoint_data["model_state_dict"].items()
            }
        )
        self.model.to(self.config.device)  # type: ignore

        self.momentum = checkpoint_data["momentum"]

        checkpoint_start_window = checkpoint_data.get("start_window")
        checkpoint_current_window = checkpoint_data.get("current_window")
        if checkpoint_start_window is None or checkpoint_current_window is None:
            tplr.logger.error("Checkpoint missing start_window or current_window info")
            return (False, checkpoint_data, checkpoint_current_window, 0)

        # Instead of computing global_step from the entire training period,
        # compute window_difference as the number of missed windows.
        window_difference = current_window - checkpoint_current_window
        global_step = current_window - checkpoint_start_window

        tplr.logger.info(
            f"Checkpoint windows (start={checkpoint_start_window}, checkpoint_current={checkpoint_current_window}), "
            f"local_current={current_window}, window_diff={window_difference}, global_step={global_step}"
        )

        if window_difference > 0:
            tplr.logger.error(
                f"Missed {window_difference} windows since last checkpoint"
            )
            return (False, checkpoint_data, checkpoint_current_window, 0)

        return (
            True,
            checkpoint_data,
            checkpoint_current_window,
            global_step,
        )

    async def _evaluate(self) -> Optional[int]:
        """
        Run benchmark evaluation with the highest stake model checkpoint and log the results.
        """
        await self.update_state()
        self.comms.commitments = await self.comms.get_commitments()
        self.comms.update_peers_with_buckets()

        start_window = await self.comms.get_start_window()
        block_number = self.subtensor.get_current_block() - 1

        tplr.logger.info(
            f"Using start_window: {start_window} and block: {block_number}"
        )

        (
            success,
            checkpoint_data,
            checkpoint_window,
            global_step,
        ) = await self.load_model(block_number, start_window)

        if not success:
            tplr.logger.error(
                f"Failed to load model checkpoint  (current block: {block_number}, last: {self.last_block_number})."
            )

            return global_step

        tplr.logger.info(
            f"Starting benchmark run at global step {global_step} (block {block_number})"
        )
        os.makedirs(MODEL_PATH, exist_ok=True)
        self.model.save_pretrained(MODEL_PATH)
        self.hparams.tokenizer.save_pretrained(MODEL_PATH)

        results_dir = os.path.join(MODEL_PATH, "results")
        os.makedirs(results_dir, exist_ok=True)
        # Start benchmark timer
        start_time = time.time()
        lm_eval_command = (
            f"lm-eval "
            f"--model hf "
            f"--model_args pretrained={MODEL_PATH},tokenizer={MODEL_PATH} "
            f"--tasks {self.config.tasks} "
            f"--device {self.config.device} "
            f"--batch_size {self.config.actual_batch_size} "
            f"--output_path {results_dir}"
        )
        tplr.logger.info(f"Running benchmark command: {lm_eval_command}")
        exit_code = os.system(lm_eval_command)
        runtime = time.time() - start_time
        self.metrics_logger.log(
            measurement="templar_benchmark_metrics",
            tags={
                "role": "evaluator",
                "global_step": global_step,
                "window": checkpoint_window,
                "block": block_number,
                "version": __version__,
            },
            fields={
                "lm_eval_exit_code": float(exit_code),
                "benchmark_runtime_s": runtime,
            },
        )
        wandb.log(
            {
                "evaluator/lm_eval_exit_code": exit_code,
                "evaluator/benchmark_runtime_s": runtime,
            }
        )
        if exit_code != 0:
            tplr.logger.error("Benchmarking command failed")
            return global_step

        eval_results_dir = os.path.join(results_dir, "models__eval")
        if not os.path.exists(eval_results_dir):
            tplr.logger.error(f"Results directory not found: {eval_results_dir}")
            return global_step

        latest_file = max(
            [os.path.join(eval_results_dir, f) for f in os.listdir(eval_results_dir)],
            key=os.path.getctime,
        )
        with open(latest_file, "r") as f:
            results = json.load(f)

        for task_name, task_results in results["results"].items():
            metric_name = "acc_norm,none" if task_name != "winogrande" else "acc,none"
            if (metric_value := task_results.get(metric_name)) is not None:
                tplr.logger.info(f"Benchmark for {task_name}: {metric_value}")
                self.metrics_logger.log(
                    measurement="templar_benchmark",
                    tags={
                        "role": "evaluator",
                        "task": task_name,
                        "global_step": global_step,
                        "block": block_number,
                        "window": checkpoint_window,
                        "version": __version__,
                    },
                    fields={"score": float(metric_value)},
                )
                wandb.log(
                    {
                        "task": task_name,
                        "global_step": global_step,
                        "block": block_number,
                        "score": float(metric_value),
                    }
                )

        overall_benchmark = {
            "num_tasks": len(results["results"]),
            "global_step": global_step,
            "block_number": block_number,
        }
        self.metrics_logger.log(
            measurement="templar_benchmark_summary",
            tags={
                "role": "evaluator",
                "global_step": global_step,
                "window": checkpoint_window,
                "block": block_number,
                "version": __version__,
            },
            fields=overall_benchmark,
        )
        wandb.log(
            {
                "overall_num_tasks": overall_benchmark["num_tasks"],
                "overall_global_step": overall_benchmark["global_step"],
                "overall_block_number": overall_benchmark["block_number"],
            }
        )

        shutil.rmtree(MODEL_PATH)
        torch.cuda.empty_cache()
        self.last_eval_step = global_step
        self.last_block_number = block_number
        return global_step

    async def run(self) -> None:
        """
        Main loop: periodically update state and run evaluations if new checkpoints exist.
        """
        try:
            self.comms.start_commitment_fetcher()
            self.comms.start_background_tasks()

            while not self.stop_event.is_set():
                await self.update_state()

                latest_block = self.subtensor.get_current_block()

                if latest_block > self.last_block_number:
                    tplr.logger.info(
                        f"New checkpoint detected at block {latest_block}, executing benchmark..."
                    )
                    await self._evaluate()
                else:
                    tplr.logger.info(
                        f"No new checkpoint available (current: {latest_block}, last: {self.last_block_number})"
                    )
                await asyncio.sleep(self.config.eval_interval)  # type: ignore
        except KeyboardInterrupt:
            tplr.logger.info("Benchmark run interrupted by user")
            self.stop_event.set()
        except Exception as e:
            tplr.logger.error(f"Benchmark run failed: {e}")

    def cleanup(self) -> None:
        """
        Cleanup resources before exit.
        """
        self.stop_event.set()


def main() -> None:
    """
    Entry point for the evaluator.
    """
    evaluator = Evaluator()
    try:
        asyncio.run(evaluator.run())
    except Exception as e:
        tplr.logger.error(f"Evaluator terminated with error: {e}")
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main()
