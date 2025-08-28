"""Templar Autonomous Model Evaluator Service

This script implements an autonomous service that continuously evaluates the latest
model checkpoints using standardized benchmark tasks. It runs on a fixed interval
(default 10 minutes), downloads the latest model checkpoint, executes evaluations,
and logs results to InfluxDB.

Key Features:
    - Automatic checkpoint detection and evaluation
    - Multiple benchmark task support (arc_challenge, winogrande, etc.)
    - Distributed metrics logging
    - Resource management and cleanup
    - Service-oriented design for continuous operation

Environment Requirements:
    - Registered Bittensor wallet
    - InfluxDB API access
    - R2 Dataset access credentials

Required Environment Variables:
    R2_DATASET_ACCOUNT_ID: R2 dataset account identifier (see miner documentation)
    R2_DATASET_BUCKET_NAME: R2 storage bucket name (see miner documentation)
    R2_DATASET_READ_ACCESS_KEY_ID: R2 read access key (see miner documentation)
    R2_DATASET_READ_SECRET_ACCESS_KEY: R2 secret access key (see miner documentation)
    INFLUXDB_TOKEN: InfluxDB API token (optional, uses default if not provided)

Usage Examples:
    Basic run:
        $ WALLET_NAME=...
        $ WALLET_HOTKEY=...
        $ torchrun --standalone --nnodes 1 --nproc_per_node 8 neurons/evaluator.py \
            --device cuda \
            --netuid 3 \
            --wallet.name $WALLET_NAME  \
            --wallet.hotkey $WALLET_HOTKEY

    Custom configuration:
        $ torchrun --standalone --nnodes 1 --nproc_per_node 8 neurons/evaluator.py \
            --device cuda \
            --netuid 3 \
            --wallet.name $WALLET_NAME  \
            --wallet.hotkey $WALLET_HOTKEY \
            --tasks "arc_challenge,winogrande" \
            --eval_interval 300 \
            --custom_eval_path eval 

For additional environment setup, refer to the miner documentation:
https://github.com/tplr-ai/templar/blob/main/docs/miner.md
"""

import argparse
import asyncio
import json
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import bittensor as bt
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchtitan.components.loss import cross_entropy_loss
from tqdm.auto import tqdm

import tplr
from lm_eval import simple_evaluate
from tplr import decos
from tplr.model_factory import initialize_torchtitan_model
from tplr.test_lm_eval_direct import TitanLlamaLM

CHECKPOINT_DEFAULT_DIR: str = "checkpoints/"
MODEL_PATH: str = "models/eval"
DEFAULT_EVAL_INTERVAL: int = 60 * 10  # 10 mins default interval


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class NullMetricsLogger:
    def log(self, *args, **kwargs) -> None:
        return


class Evaluator:
    """Templar Model Evaluator Component

    The Evaluator is responsible for automated benchmark evaluation of model checkpoints.
    It continuously monitors for new checkpoints by window number, downloads them, runs a
    comprehensive suite of language model evaluations, and logs results to InfluxDB.
    """

    @staticmethod
    def evaluator_config() -> bt.config:
        """
        Parse command-line arguments and return a configuration object.
        """

        parser = argparse.ArgumentParser(
            description="Evaluator script. Use --help to display options.",
            add_help=True,
        )
        parser.add_argument(
            "--netuid",
            type=int,
            default=3,
            help="Bittensor network UID.",
        )
        parser.add_argument(
            "--actual_batch_size",
            type=str,
            default="auto",
            help="Evaluation batch size.",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda:0",
            help="Device to use for evaluation",
        )
        parser.add_argument(
            "--tasks",
            type=str,
            default="arc_challenge,arc_easy,openbookqa,winogrande,piqa,hellaswag,mmlu",
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
        parser.add_argument(
            "--skip-gaps",
            type=bool,
            default=False,
            help="Skip gaps in the evaluation process",
        )
        parser.add_argument(
            "--custom_eval_path",
            type=str,
            default=None,
            help="Path to the custom evaluation dataset bins for perplexity calculation.",
        )
        parser.add_argument(
            "--limit",
            type=float,
            default=1.0,
            help="Fraction of dataset to evaluate (0.0-1.0)",
        )
        parser.add_argument(
            "--num_fewshot",
            type=int,
            default=0,
            help="Number of few-shot examples",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="evaluation_results",
            help="Directory to save evaluation results",
        )
        parser.add_argument(
            "--cleanup",
            action="store_true",
            help="Clean up model files after evaluation",
        )
        parser.add_argument(
            "--version",
            type=str,
            default=None,
            help="Model version to evaluate",
        )

        bt.subtensor.add_args(parser)
        bt.wallet.add_args(parser)

        config = bt.config(parser)
        return config

    def __init__(self) -> None:
        self.config = self.evaluator_config()

        if self.config.netuid is None:
            raise ValueError("No netuid provided")
        if self.config.device is None:
            raise ValueError("No device provided")

        if self.config.debug:
            tplr.debug()
        if self.config.trace:
            tplr.trace()

        # Distributed training setup
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.is_master = self.rank == 0

        tplr.logger.info(
            f"[Init] rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank}"
        )

        if self.world_size > 1:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=60 * 10),
                    rank=self.rank,
                    world_size=self.world_size,
                )
                torch.cuda.set_device(self.local_rank)
            self.device: str = f"cuda:{self.local_rank}"
            tplr.logger.info(
                f"Distributed evaluation: rank {self.rank}/{self.world_size}, "
                f"local_rank {self.local_rank}, is_master: {self.is_master}"
            )
        else:
            self.device = self.config.device

        tplr.logger.info(f"[Init] device set → {self.device}")

        # Bittensor objects
        self.wallet = bt.wallet(config=self.config)
        self.netuid = self.config.netuid
        self.version = self.config.version or tplr.__version__
        
        # Model and tokenizer
        self.hparams = tplr.load_hparams()
        self.model = initialize_torchtitan_model(
            hparams=self.hparams,
            role="evaluator",
            device=self.device,
            world_size=self.world_size,
        )
        self.tokenizer = self.hparams.tokenizer
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Comms and logging
        self.uid = 1 # Mock UID
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location="/tmp",
            key_prefix="checkpoint",
            config=self.config,
            hparams=self.hparams,
            uid=self.uid,
        )

        if self.is_master:
            self.buckets = self.comms.get_all_buckets()
            self.metrics_logger = tplr.metrics.MetricsLogger(
                prefix="E",
                uid=str(self.uid),
                config=self.config,
                role="evaluator",
                group="evaluations",
                job_type="eval",
                version=self.version,
            )
        else:
            self.buckets = None
            self.metrics_logger = NullMetricsLogger()

        # Evaluation state
        self.last_eval_window = 0
        self.stop_event = asyncio.Event()
        self.last_block_number = 0
        self.eval_counter = 0
        self.evaluated_checkpoints = []
        self.task_list = self.config.tasks.split(",") if self.config.tasks else []

        if dist.is_available() and dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank])

    async def update_state(self) -> None:
        """
        Refresh the metagraph and bucket information.
        """
        self.comms.metagraph = self.comms.subtensor.metagraph(netuid=self.netuid)
        self.buckets = self.comms.get_all_buckets()

    async def load_latest_model(self) -> tuple[bool, int]:
        """Load and prepare the latest model checkpoint for evaluation."""
        current_window = (
            self.comms.subtensor.get_current_block() // self.hparams.blocks_per_window
        )

        success, checkpoint_window = await self.comms.load_checkpoint(
            model=self.model,
            current_window=current_window,
            init_version=self.version,
            is_master=self.is_master,
        )

        if not success:
            if self.is_master:
                tplr.logger.error(
                    f"No valid checkpoints found. Check bucket: {getattr(self.comms, 'bucket', 'unknown')}, "
                    f"key_prefix: {self.comms.key_prefix}"
                )
            return (False, 0)

        if checkpoint_window <= self.last_eval_window:
            if self.is_master:
                tplr.logger.info(
                    f"Checkpoint already evaluated (checkpoint window: {checkpoint_window}, "
                    f"last evaluated: {self.last_eval_window})."
                )
            return (False, checkpoint_window)

        if self.is_master:
            tplr.logger.info(f"Loaded checkpoint (window={checkpoint_window})")

        if dist.is_available() and dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank])

        return (True, checkpoint_window)

    @decos.async_evaluator_exception_catcher()
    @decos.master_only
    async def _process_results(
        self,
        results: dict[str, Any] | None,
        global_step: int,
        checkpoint_window: int,
        block_number: int,
    ) -> None:
        """Process results from the lm-eval benchmark."""

        tplr.logger.info(
            f"Processing metrics for global step {global_step} (block: {block_number}, window: {checkpoint_window})"
        )

        if not results:
            tplr.logger.warning("No results to process.")
            return

        for task_name, task_results in results.get("results", {}).items():
            metric_names = ["acc_norm,none", "acc,none"]
            metric_value = None
            used_metric = None

            for metric_name in metric_names:
                if (value := task_results.get(metric_name)) is not None:
                    metric_value = value
                    used_metric = metric_name
                    break

            if metric_value is not None:
                tplr.logger.info(
                    f"Benchmark for {task_name} ({used_metric}): {metric_value}"
                )
                self.metrics_logger.log(
                    measurement="benchmark_task",
                    tags={
                        "task": task_name,
                        "metric": used_metric,
                        "global_step": global_step,
                        "block": block_number,
                        "window": checkpoint_window,
                    },
                    fields={"score": float(metric_value)},
                )

        self.metrics_logger.log(
            measurement="benchmark_summary",
            tags={
                "global_step": global_step,
                "window": checkpoint_window,
                "block": block_number,
            },
            fields={
                "num_tasks": len(results["results"]),
                "global_step": global_step,
                "block_number": block_number,
            },
        )
        tplr.logger.info(
            f"Reported summary for global step {global_step} (block: {block_number}, window: {checkpoint_window})"
        )
        return

    @decos.async_evaluator_exception_catcher()
    async def _evaluate(self) -> None:
        """Execute benchmark evaluation on the current model."""
        self.comms.commitments = await self.comms.get_commitments()
        self.comms.update_peers_with_buckets()
        start_window = await self.comms.get_start_window(version=self.version)

        block_number = self.comms.subtensor.get_current_block() - 1

        tplr.logger.info(f"Looking for new checkpoint (block: {block_number})")

        (success, checkpoint_window) = await self.load_latest_model()

        if not success and self.last_eval_window > 0:
            tplr.logger.info(
                f"No new checkpoint to evaluate (last evaluated window: {self.last_eval_window})"
            )
            return
        elif not success and self.last_eval_window == 0:
            tplr.logger.info(
                "Using initialized model for evaluation (no checkpoint available)"
            )
            checkpoint_window = 0
            global_step = 0
        else:
            self.evaluated_checkpoints.append(checkpoint_window)
            global_step = (
                max(0, checkpoint_window - start_window)
                if start_window is not None
                else checkpoint_window
            )

        if start_window is None:
            if self.is_master:
                tplr.logger.warning(
                    f"Start window not found for version {self.version}. Defaulting to 0."
                )
            start_window = 0       

        if self.is_master:
            tplr.logger.info(
                f"Starting benchmark run at global step {global_step} (checkpoint window: {checkpoint_window})"
            )

        self._evaluate_custom(
            global_step=global_step,
            checkpoint_window=checkpoint_window,
            block_number=block_number,
        )

        if self.task_list:
            torch.cuda.empty_cache()

            lm_eval_model = TitanLlamaLM(
                model=self.model,
                tokenizer=self.tokenizer,
                hparams=self.hparams,
                device=self.device,
                actual_batch_size=self.config.actual_batch_size,
            )

            if self.is_master:
                self.eval_counter += 1
                tasks_to_run = [t for t in self.task_list if t != "mmlu"]
                
                if tasks_to_run:
                    results = self._run_lm_eval_direct(
                        lm_eval_model,
                        tasks=tasks_to_run,
                        batch_size="auto",
                        limit=0.2,
                    )
                    if results:
                        await self._process_results(
                            results,
                            global_step=global_step,
                            checkpoint_window=checkpoint_window,
                            block_number=block_number,
                        )

            if dist.is_available() and dist.is_initialized():
                dist.barrier(device_ids=[self.local_rank])
            torch.cuda.empty_cache()

            if self.is_master:
                has_mmlu_task = "mmlu" in self.task_list
                should_run_mmlu = has_mmlu_task and (
                    self.config.skip_gaps or self.eval_counter % 4 == 0
                )

                if should_run_mmlu:
                    tplr.logger.info(f"Run #{self.eval_counter}: Running mmlu")
                    results = self._run_lm_eval_direct(
                        lm_eval_model,
                        tasks=["mmlu"],
                        limit=0.15,
                        num_fewshot=5,
                        batch_size="auto",
                    )
                    if results:
                        await self._process_results(
                            results,
                            global_step=global_step,
                            checkpoint_window=checkpoint_window,
                            block_number=block_number,
                        )
                elif has_mmlu_task:
                    tplr.logger.info(
                        f"Skipping mmlu (run #{self.eval_counter}, next at run #{(self.eval_counter // 4 + 1) * 4})"
                    )

        torch.cuda.empty_cache()
        if dist.is_available() and dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank])
        torch.cuda.empty_cache()

        self.last_eval_window = checkpoint_window
        self.last_block_number = block_number

        tplr.logger.info(
            f"Successfully evaluated checkpoint (window: {checkpoint_window}, "
            f"global_step: {global_step}, block: {block_number})"
        )
        self.model = self.model.to(self.device)

    @decos.evaluator_exception_catcher()
    def _evaluate_custom(
        self, global_step: int, checkpoint_window: int, block_number: int
    ) -> None:
        """Run evaluation on a custom dataset and log perplexity."""
        if not self.config.custom_eval_path:
            if self.is_master:
                tplr.logger.info("Custom evaluation path not provided, skipping.")
            return

        if self.is_master:
            tplr.logger.info(
                f"Starting custom evaluation on dataset: {self.config.custom_eval_path}"
            )
        os.environ["DATASET_BINS_PATH"] = self.config.custom_eval_path

        eval_path = Path(self.config.custom_eval_path)
        file_prefix = "train"
        if (eval_path / "val_000000.npy").exists():
            file_prefix = "val"
        elif (eval_path / "eval_000000.npy").exists():
            file_prefix = "eval"
        elif (eval_path / "test_000000.npy").exists():
            file_prefix = "test"

        if self.is_master:
            tplr.logger.info(f"Using file prefix: {file_prefix}")

        custom_dataset = tplr.SharedShardedDataset(
            shard_index=0,
            sequence_length=self.hparams.sequence_length,
            rank=self.rank if self.world_size > 1 else 0,
            world_size=self.world_size,
            file_prefix=file_prefix,
        )

        if self.world_size > 1:
            total_samples = min(1024, len(custom_dataset))
            indices = list(range(total_samples))
            samples_per_rank = total_samples // self.world_size
            start_idx = self.rank * samples_per_rank
            end_idx = (
                start_idx + samples_per_rank
                if self.rank < self.world_size - 1
                else total_samples
            )
            sampler = torch.utils.data.SubsetRandomSampler(indices[start_idx:end_idx])
        else:
            sampler = torch.utils.data.SubsetRandomSampler(
                range(min(1024, len(custom_dataset)))
            )

        dataloader = DataLoader(
            dataset=custom_dataset,
            batch_size=self.config.actual_batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
        )

        self.model.to(self.device)
        self.model.eval()

        local_loss = 0.0
        local_tokens = 0
        start_time = time.time()

        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Custom Eval"):
                input_ids = batch.to(self.device, dtype=torch.long, non_blocking=True)
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = self.tokenizer.pad_token_id
                labels = torch.where(
                    labels == self.tokenizer.pad_token_id, -100, labels
                )
                model_device = next(self.model.parameters()).device
                with torch.autocast(
                    device_type=model_device.type, dtype=torch.bfloat16
                ):
                    logits = self.model(input_ids)
                loss = cross_entropy_loss(logits, labels)

                num_tokens = (labels != -100).sum().item()
                if num_tokens > 0:
                    local_loss += loss.item() * num_tokens
                    local_tokens += num_tokens

        if self.world_size > 1:
            loss_tensor = torch.tensor([local_loss], device=self.device)
            tokens_tensor = torch.tensor([local_tokens], device=self.device)

            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)

            total_loss = loss_tensor.item()
            total_tokens = int(tokens_tensor.item())
        else:
            total_loss = local_loss
            total_tokens = local_tokens

        eval_runtime = time.time() - start_time
        average_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = (
            torch.exp(torch.tensor(average_loss)).item()
            if average_loss > 0
            else float("inf")
        )

        if self.is_master:
            tplr.logger.info(
                f"Custom evaluation finished. Perplexity: {perplexity:.4f}, "
                f"Avg Loss: {average_loss:.4f}, Runtime: {eval_runtime:.2f}s, "
                f"Total tokens: {total_tokens}"
            )
            self.metrics_logger.log(
                "custom_evaluation",
                tags={
                    "global_step": global_step,
                    "window": checkpoint_window,
                    "block": block_number,
                    "world_size": self.world_size,
                },
                fields={
                    "perplexity": perplexity,
                    "average_loss": average_loss,
                    "runtime_s": eval_runtime,
                    "total_tokens": total_tokens,
                },
            )

    @decos.async_evaluator_exception_catcher()
    async def run(self) -> None:
        """Main evaluation loop."""
        if self.is_master:
            self.comms.start_commitment_fetcher()
            self.comms.start_background_tasks()

        await self._evaluate()

        while not self.stop_event.is_set():
            latest_block = None
            start_window = None
            if self.is_master:
                await self.update_state()
                latest_block = self.comms.subtensor.get_current_block()
                start_window = await self.comms.get_start_window(version=self.version)

                should_evaluate = start_window is not None and (
                    latest_block > self.last_block_number
                    or start_window > self.last_eval_window
                )

                if should_evaluate:
                    tplr.logger.info(
                        f"New checkpoint detected (block: {latest_block}, window: {start_window}), executing benchmark..."
                    )
            else:
                should_evaluate = False

            if dist.is_available() and dist.is_initialized():
                should_eval_tensor = torch.tensor(
                    [1 if should_evaluate else 0],
                    dtype=torch.int32,
                    device=self.device,
                )
                dist.broadcast(should_eval_tensor, src=0)
                should_evaluate = bool(should_eval_tensor.item())

            if should_evaluate:
                await self._evaluate()
                if dist.is_available() and dist.is_initialized():
                    dist.barrier(device_ids=[self.local_rank])
            elif self.is_master:
                tplr.logger.info(
                    f"No new checkpoint available (block: {latest_block}/{self.last_block_number}, "
                    f"window: {start_window}/{self.last_eval_window})"
                )

            await asyncio.sleep(int(self.config.eval_interval))

    def cleanup(self) -> None:
        """
        Cleanup resources before exit.
        """
        self.stop_event.set()

    def _run_lm_eval_direct(
        self,
        lm_eval_model,
        tasks: list[str],
        batch_size: str = "auto",
        limit: float = 1.0,
        num_fewshot: int | None = None,
    ) -> dict[str, Any] | None:
        """Run lm-eval benchmarks directly in-process."""
        if not self.is_master:
            return None

        tplr.logger.info(
            f"Running in-process evaluation for tasks: {tasks} (limit: {limit}, fewshot: {num_fewshot})"
        )
        start_time = time.time()

        try:
            results: Dict[str, Any] | None = simple_evaluate(
                model=lm_eval_model,
                tasks=tasks,
                limit=limit,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
            )
            if results:
                results["benchmark_runtime_s"] = time.time() - start_time
            return results
        except Exception as e:
            tplr.logger.error(f"Error during simple_evaluate: {e}", exc_info=True)
            return None


@decos.evaluator_exception_catcher()
def main() -> None:
    """
    Entry point for the evaluator.
    """
    evaluator = Evaluator()
    try:
        asyncio.run(evaluator.run())
    except Exception as e:
        tplr.logger.exception(f"Evaluator encountered an error: {e}")
    finally:
        evaluator.cleanup()
        if hasattr(evaluator, 'model'):
            evaluator.model = evaluator.model.to("cpu")
        torch.cuda.empty_cache()

        if "DATASET_BINS_PATH" in os.environ:
            del os.environ["DATASET_BINS_PATH"]


if __name__ == "__main__":
    main()
