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
        $ torchrun --standalone --nnodes 1 --nproc_per_node 8 neurons/evaluator.py \\
            --device cuda \\
            --netuid 3 \\
            --wallet.name $WALLET_NAME  \\
            --wallet.hotkey $WALLET_HOTKEY

    Custom configuration:
        $ torchrun --standalone --nnodes 1 --nproc_per_node 8 neurons/evaluator.py \\
            --device cuda \\
            --netuid 3 \\
            --wallet.name $WALLET_NAME  \\
            --wallet.hotkey $WALLET_HOTKEY
            --tasks "arc_challenge,winogrande" \\
            --eval_interval 300 \\
            --custom_eval_path eval 

For additional environment setup, refer to the miner documentation:
https://github.com/tplr-ai/templar/blob/main/docs/miner.md
"""

import argparse
import asyncio
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import bittensor as bt
import torch
import torch.distributed as dist
from lm_eval import simple_evaluate
from torch.utils.data import DataLoader
from torchtitan.components.loss import cross_entropy_loss
from tqdm.auto import tqdm
from websockets.exceptions import ConcurrencyError

import tplr
from tplr import (
    SharedShardedDataset,
    decos,
)
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

    Key Features:
        - Automatic checkpoint detection by window number
        - Multi-task model evaluation
        - Distributed metrics logging
        - Progress tracking via grafana
        - Resource cleanup and management

    Evaluation Flow:
        1. Monitor blockchain for new checkpoints by window number
        2. Download and load checkpoint directly when detected
        3. Run benchmark suite using lm-eval
        4. Parse and log results
        5. Clean up resources
        6. Wait for next checkpoint

    Attributes:
        config (bt.Config): Configuration object containing CLI arguments
        netuid (int): Network UID for the subnet
        model (TitanLlama): The TorchTitan language model being evaluated
        metrics_logger (MetricsLogger): Logger for InfluxDB metrics
        last_eval_window (int): Last evaluated window number
        last_block_number (int): Last processed block number
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

        # For "local" runs:
        parser.add_argument(
            "--limit",
            type=float,
            default=0.2,
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

        parser.parse_args()
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

        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.is_master = self.rank == 0

        # Use constant for default checkpoint directory.
        self.checkpoint_path: str = (
            self.config.checkpoint_path or CHECKPOINT_DEFAULT_DIR
        )
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        self.netuid = self.config.netuid
        self.hparams = tplr.load_hparams()
        tplr.logger.info(
            f"Loaded hparams: hidden_size={self.hparams.hidden_size}, num_hidden_layers={self.hparams.num_hidden_layers}, num_key_value_heads={self.hparams.num_key_value_heads}"
        )
        self.wallet = bt.wallet(config=self.config)

        self.version = self.config.version or tplr.__version__

        # Mock for the comms class
        self.uid = 1
        # All ranks need comms for distributed checkpoint loading
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location="/tmp",
            key_prefix="checkpoint",
            config=self.config,
            hparams=self.hparams,
            uid=self.uid,
        )

        # Only master rank gets buckets and initializes metrics logger
        if self.is_master:
            self.buckets = self.comms.get_all_buckets()

            # Initialize metrics logger with consistent patterns
            self.metrics_logger = tplr.metrics.MetricsLogger(
                prefix="E",
                uid=str(self.uid),  # Is this intended to be user or default==1?
                config=self.config,
                role="evaluator",
                group="evaluations",
                job_type="eval",
                version=self.version,
            )
        else:
            self.buckets = None
            self.metrics_logger = NullMetricsLogger()

        # Initialize distributed training if available
        tplr.logger.info(
            f"[Init] rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank}"
        )
        if self.world_size < 2:
            raise ValueError(
                "Current models require distribution. Use DDP mode for smaller models."
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

            tplr.logger.info(
                f"[Init] NCCL process-group ready and GPU selected, {self.rank}"
            )
            self.device: str = f"cuda:{self.local_rank}"

            tplr.logger.info(
                f"Distributed evaluation: rank {self.rank}/{self.world_size}, "
                f"local_rank {self.local_rank}, is_master: {self.is_master}"
            )
        else:  # Single GPU or CPU mode
            if self.config.device:
                self.device: str = self.config.device
            elif torch.cuda.is_available():
                self.device: str = "cuda:0"  # Use cuda:0 for single GPU
            else:
                self.device: str = "cpu"
        tplr.logger.info(f"[Init] device set → {self.device}")

        # Initialize TorchTitan model using model factory
        self.model = initialize_torchtitan_model(
            hparams=self.hparams,
            role="evaluator",
            device=self.device,
            world_size=self.world_size,
        )
        self.ckpt = tplr.DCPCheckpointer(
            self.comms, uid=self.uid, version=tplr.__version__, repo_root="."
        )

        self.tokenizer = self.hparams.tokenizer
        # Ensure a pad token exists for loss masking/perplexity
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            # LLaMA tokenizers typically don't have PAD; use EOS
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.last_eval_window = 0
        self.stop_event = asyncio.Event()
        self.last_block_number = 0
        self.eval_counter = 0

        # Track evaluated checkpoints
        self.evaluated_checkpoints = []

        self.task_list = []
        if self.config.tasks:
            self.task_list = self.config.tasks.split(",")


    async def update_state(self) -> None:
        """
        Refresh the metagraph and bucket information.
        """
        self.comms.metagraph = self.comms.subtensor.metagraph(netuid=self.netuid)
        self.buckets = self.comms.get_all_buckets()

    async def load_latest_model(self) -> tuple[bool, int]:
        """Load and prepare the latest model checkpoint for evaluation.

        This method:
        1. Uses comms.load_checkpoint for distributed loading
        2. Verifies checkpoint validity
        3. Updates internal state trackers

        Returns:
            Tuple containing:
            - success (bool): Whether loading succeeded
            - checkpoint_window (int): Window number of checkpoint
        """
        # Get the current window to check against
        current_window = (
            self.comms.subtensor.get_current_block() // self.hparams.blocks_per_window
        )

        if self.is_master:
            tplr.logger.info("Attempting to load latest model checkpoint...")

        # Use load_checkpoint which handles distributed loading properly
        checkpoint_window = await self.ckpt.download_and_load(
            model=self.model,
            window=None,
            shared_fs=True,
            process_group=None,
            prefer_highest_staked=True,
        )
        success = checkpoint_window is not None

        if not success:
            if self.is_master:
                tplr.logger.info("No checkpoint found on the network.")
            return False, 0

        if checkpoint_window <= self.last_eval_window:
            if self.is_master:
                tplr.logger.info(
                    f"Latest checkpoint window ({checkpoint_window}) is not newer than "
                    f"last evaluated window ({self.last_eval_window}). Skipping."
                )
            return False, checkpoint_window

        if checkpoint_window in self.evaluated_checkpoints:
            if self.is_master:
                tplr.logger.info(
                    f"Checkpoint window {checkpoint_window} already evaluated. Skipping."
                )
            return False, checkpoint_window

        if self.is_master:
            tplr.logger.info(
                f"New checkpoint window detected: {checkpoint_window}. "
                "Downloading and loading model..."
            )

        return True, checkpoint_window

    def _run_lm_eval(
        self,
        tasks: str,
        output_dir: str,
        batch_size: str = "auto",
        model_args: str | None = None,
        limit: str | None = None,
        num_fewshot: int | None = None,
    ) -> tuple[int, float]:
        """Run lm-eval benchmark for specified tasks with custom configuration.

        Args:
            tasks: Comma-separated task list
            output_dir: Directory to save results
            model_args: Optional model arguments
            batch_size: Optional batch size
            limit: Optional dataset limit
            num_fewshot: Optional few-shot examples

        Returns:
            Tuple containing (exit_code, runtime)
        """
        default_model_args = [
            f"pretrained={MODEL_PATH}",
            f"tokenizer={MODEL_PATH}",
            "max_length=2048",
        ]

        extra = None
        device_arg = self.device
        if self.world_size > 1 and str(self.device).startswith("cuda"):
            # Harness path for single‑process multi‑GPU sharding (no DDP).
            # Uses all GPUs in CUDA_VISIBLE_DEVICES.
            extra = ["parallelize=True", "device_map=auto"]

            # For sharded runs, pass plain 'cuda' so lm‑eval doesn’t pin to ':0'.
            device_arg = "cuda"

        if extra and model_args is None:
            model_args = ",".join(default_model_args + extra)

        cmd_parts = [
            "accelerate launch",
            "--dynamo_backend inductor",
            "--multi_gpu",
            "--mixed_precision bf16",
            f"--num_processes {self.world_size}",
            "-m",
            "lm_eval",
            "--model hf",
            f"--model_args {model_args}",
            f"--tasks {tasks}",
            f"--device {device_arg}",
            f"--batch_size {batch_size}",
            f"--output_path {output_dir}",
            # '--gen_kwargs "max_gen_toks=2048",
        ]

        if limit:
            cmd_parts.append(f"--limit {limit}")
        if num_fewshot:
            cmd_parts.append(f"--num_fewshot {num_fewshot}")

        command = " ".join(cmd_parts)

        start_time = tplr.T()
        tplr.logger.info(f"Running benchmark command: {command}")
        exit_code = os.system(command)
        benchmark_runtime = tplr.T() - start_time

        return exit_code, benchmark_runtime

    def _load_latest_file(self, eval_results_dir: str) -> dict[str, dict[str, float]]:
        """Load the latest evaluation results file from the specified directory.

        Args:
            eval_results_dir: Directory containing evaluation results files

        Returns:
            Parsed JSON data from the latest results file
        """
        output = {}

        try:
            latest_file = max(
                [
                    os.path.join(eval_results_dir, f)
                    for f in os.listdir(eval_results_dir)
                ],
                key=os.path.getctime,
            )
            with open(latest_file, "r") as f:
                output = json.load(f)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            tplr.logger.error(
                f"Error loading latest evaluation file from {eval_results_dir}: {e}"
            )

        return output

    @decos.async_evaluator_exception_catcher()
    @decos.master_only
    async def _process_results(
        self,
        results: dict[str, Any] | None,
        global_step: int,
        checkpoint_window: int,
        block_number: int,
    ) -> None:
        """Process results from the lm-eval benchmark.

        Args:
            task_name: Name of the benchmark task
            results_dir: Directory containing results
            global_step: Current global step for logging
            checkpoint_window: Current window for logging
            block_number: Current block for logging
            benchmark_runtime: Runtime of the benchmark
            exit_code: Exit code of the benchmark command
        """

        tplr.logger.info(
            f"Processing metrics for global step {global_step} (block: {block_number}, window: {checkpoint_window})"
        )

        if not results:
            tplr.logger.warning("No results to process.")
            return

        for task_name, task_results in results.get("results", {}).items():
            # We need to try each metric in order until we find one that exists
            # Also we need to prioritise metrics in order of preference
            # see: https://github.com/EleutherAI/lm-evaluation-harness/blob/758c5ed891b1ca48acd8d3a0d309a827215796b7/scripts/regression.py#L115
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

    @decos.async_evaluator_exception_catcher(on_error_raise=True)
    async def _evaluate(self) -> None:
        """Execute benchmark evaluation on the current model.

        Workflow:
        1. Save model to temporary location
        2. Run lm-eval benchmark suite
        3. Parse results for each task
        4. Log metrics to InfluxDB
        5. Clean up temporary files
        """
        self.comms.commitments = await self.comms.get_commitments()
        self.comms.update_peers_with_buckets()
        start_window = (
            await self.comms.get_start_window(version=self.version) 
        )

        if self.is_master:
            block_number_list = []
            while not block_number_list:
                try:
                    # Master node determines the block number
                    block_number_list = [self.comms.subtensor.get_current_block() - 1]
                except ConcurrencyError:
                    pass
        else:
            # Other nodes have a placeholder
            block_number_list = [0]

        # Broadcast the block number from master to all other nodes
        dist.broadcast_object_list(block_number_list, src=0)
        block_number = block_number_list[0]
        plr.logger.info(f"Looking for new checkpoint (block: {block_number})")

        (success, checkpoint_window) = await self.load_latest_model()

        if not success and self.last_eval_window > 0:
            # We've already evaluated something and no new checkpoint is available
            tplr.logger.info(
                f"No new checkpoint to evaluate (last evaluated window: {self.last_eval_window})"
            )
            return
        elif not success and self.last_eval_window == 0:
            # First run with no checkpoint - use initialized model
            tplr.logger.info(
                "Using initialized model for evaluation (no checkpoint available)"
            )
            checkpoint_window = 0
            global_step = 0
        else:
            self.evaluated_checkpoints.append(checkpoint_window)

            # Calculate global step, ensuring it's a positive integer
            global_step = (
                max(0, checkpoint_window - start_window)
                if start_window is not None
                else checkpoint_window
            )

        # If start_window is not found, default to 0
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

        # Run custom perplexity evaluation first. It manages its own model device placement.
        # This runs on all ranks for distributed evaluation
        self._evaluate_custom(  # Commented out to match debug_evaluator_working.py
            global_step=global_step,
            checkpoint_window=checkpoint_window,
            block_number=block_number,
        )

        if self.task_list:
        
            # All ranks should clear cache
            tplr.logger.info(f"Clearing GPU cache, {self.local_rank}")
            torch.cuda.empty_cache()

            lm_eval_model = TitanLlamaLM(
                model=self.model,
                tokenizer=self.tokenizer,
                hparams=self.hparams,
                device=self.device,
                actual_batch_size="auto",  # self.config.actual_batch_size,
            )

            self.eval_counter += 1  # Increment eval_counter for all ranks
            tasks_to_run = [t for t in self.task_list if t != "mmlu"]

            if tasks_to_run:
                results = self._run_lm_eval_direct(
                    lm_eval_model,
                    tasks=tasks_to_run,
                    batch_size="auto",
                    limit=self.config.limit,
                )
                if self.is_master and results:
                    await self._process_results(
                        results,
                        global_step=global_step,
                        checkpoint_window=checkpoint_window,
                        block_number=block_number,
                    )

            # Synchronize all ranks after evaluation
            if dist.is_available() and dist.is_initialized():
                dist.barrier(device_ids=[self.local_rank])

            # Clear cache again for after the huggingface tasks
            tplr.logger.info(f"Clearing GPU cache, {self.local_rank}")
            torch.cuda.empty_cache()

            has_mmlu_task = "mmlu" in self.task_list
            should_run_mmlu = has_mmlu_task and (
                self.config.skip_gaps or self.eval_counter % 4 == 0
            )

            if should_run_mmlu:
                if self.is_master:
                    tplr.logger.info(f"Run #{self.eval_counter}: Running mmlu")
                results = self._run_lm_eval_direct(
                    lm_eval_model,
                    tasks=["mmlu"],
                    limit=self.config.limit,
                    num_fewshot=self.config.num_fewshot,
                    batch_size="auto",
                )
                if self.is_master and results:
                    await self._process_results(
                        results,
                        global_step=global_step,
                        checkpoint_window=checkpoint_window,
                        block_number=block_number,
                    )
            elif has_mmlu_task and self.is_master:
                tplr.logger.info(
                    f"Skipping mmlu (run #{self.eval_counter}, next at run #{(self.eval_counter // 4 + 1) * 4})"
                )

        # All ranks should clear cache
        tplr.logger.info(f"Clearing GPU cache, {self.local_rank}")
        torch.cuda.empty_cache()

        # Synchronize all ranks after evaluation
        if dist.is_available() and dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank])

        # Clear cache again for after the huggingface tasks
        tplr.logger.info(f"Clearing GPU cache, {self.local_rank}")
        torch.cuda.empty_cache()

        self.last_eval_window = checkpoint_window
        self.last_block_number = block_number

        tplr.logger.info(
            f"Successfully evaluated checkpoint (window: {checkpoint_window}, "
            f"global_step: {global_step}, block: {block_number})"
        )

        # Return to gpu for next cycle
        self.model = self.model.to(self.device)

    @decos.evaluator_exception_catcher(on_error_raise=True)
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

        # 1. Determine the file prefix by checking what exists
        eval_path = Path(self.config.custom_eval_path)
        file_prefix = "train"  # default

        # Check for different possible prefixes
        if (eval_path / "val_000000.npy").exists():
            file_prefix = "val"
        elif (eval_path / "eval_000000.npy").exists():
            file_prefix = "eval"
        elif (eval_path / "test_000000.npy").exists():
            file_prefix = "test"

        if self.is_master:
            tplr.logger.info(f"Using file prefix: {file_prefix}")

        # 2. Setup dataset and dataloader
        # For distributed evaluation, each rank processes different samples
        custom_dataset = tplr.SharedShardedDataset(
            shard_index=0,  # Use shard index 0 for evaluation dataset
            sequence_length=self.hparams.sequence_length,
            rank=self.rank if self.world_size > 1 else 0,
            world_size=self.world_size,
            file_prefix=file_prefix,  # Use detected or default prefix
        )

        # Create a distributed sampler if using multiple GPUs
        if self.world_size > 1:
            # Limit to 1024 samples total, distributed across ranks
            total_samples = min(1024, len(custom_dataset))
            indices = list(range(total_samples))
            # Each rank gets a subset
            samples_per_rank = total_samples // self.world_size
            start_idx = self.rank * samples_per_rank
            end_idx = (
                start_idx + samples_per_rank
                if self.rank < self.world_size - 1
                else total_samples
            )
            sampler = torch.utils.data.SubsetRandomSampler(indices[start_idx:end_idx])
        else:
            # Single GPU: use all samples up to 1024
            sampler = torch.utils.data.SubsetRandomSampler(
                range(min(1024, len(custom_dataset)))
            )

        if "auto" not in self.config.actual_batch_size:
            bs = int(self.config.actual_batch_size)
        else:
            bs = 4  # Set safe value heuristically
        tplr.logger.info(f"Using batch_size {bs} for _evaluate_custom")

        dataloader = DataLoader(
            dataset=custom_dataset,
            batch_size=bs,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
        )

        # 2. Prepare model for evaluation
        self.model.to(self.device)  # type: ignore
        self.model.eval()

        local_loss = 0.0
        local_tokens = 0
        start_time = time.time()

        # 3. Evaluation loop
        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Custom Eval")):
                input_ids = batch.to(self.device, dtype=torch.long, non_blocking=True)
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]  # shift left by one
                labels[:, -1] = self.tokenizer.pad_token_id
                labels = torch.where(
                    labels == self.tokenizer.pad_token_id, -100, labels
                )
                # Get device from model parameters (more robust for TorchTitan)
                model_device = next(self.model.parameters()).device
                with torch.autocast(
                    device_type=model_device.type, dtype=torch.bfloat16
                ):
                    # TorchTitan model returns logits directly
                    logits = self.model(input_ids)
                # Calculate loss using cross_entropy_loss from TorchTitan
                loss = cross_entropy_loss(logits, labels)

                # Accumulate loss, weighted by the number of tokens
                num_tokens = (labels != -100).sum().item()
                if num_tokens > 0:
                    local_loss += loss.item() * num_tokens
                    local_tokens += num_tokens

        # 4. Gather results from all ranks if distributed
        if self.world_size > 1:
            # Convert to tensors for all_reduce
            loss_tensor = torch.tensor([local_loss], device=self.device)
            tokens_tensor = torch.tensor([local_tokens], device=self.device)

            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)

            total_loss = loss_tensor.item()
            total_tokens = int(tokens_tensor.item())
        else:
            total_loss = local_loss
            total_tokens = local_tokens

        # 5. Calculate final metrics (only master logs)
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

            # 6. Log metrics (only master)
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

    @decos.async_evaluator_exception_catcher(on_error_raise=True)
    async def run(self) -> None:
        """Main evaluation loop.

        Continuously:
        1. Check for new checkpoints by window number and block
        2. Trigger evaluation when new checkpoint detected
        3. Handle interrupts and errors
        4. Maintain evaluation interval
        """
        # Only master starts background tasks
        if self.is_master:
            self.comms.start_commitment_fetcher()
            self.comms.start_background_tasks()

        await self._evaluate()

        while not self.stop_event.is_set():
            # Only master updates state and checks for new checkpoints
            latest_block = None
            start_window = None
            if self.is_master:
                await self.update_state()
                latest_block = self.comms.subtensor.get_current_block()
                start_window = (
                    await self.comms.get_start_window(version=self.version)
                )

                should_evaluate = start_window is not None and (
                    latest_block > self.last_block_number
                    or start_window > self.last_eval_window
                )

                if should_evaluate:
                    tplr.logger.info(
                        f"New checkpoint detected (block: {latest_block}, window: {start_window}), executing benchmark..."
                    )
            else:
                # Non-master ranks just wait for master's decision
                should_evaluate = False

            # Broadcast evaluation decision to all ranks
            if dist.is_available() and dist.is_initialized():
                should_eval_tensor = torch.tensor(
                    [1 if should_evaluate else 0],
                    dtype=torch.int32,
                    device=self.device if self.device != "cpu" else "cpu",
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
        # Removed dist.destroy_process_group() as it was causing issues
        # Removed model.to("cpu") and torch.cuda.empty_cache() as they are handled in main's finally block

    def _run_lm_eval_direct(
        self,
        lm_eval_model,
        tasks: list[str],
        batch_size: str = "auto",
        limit: float = 1.0,
        num_fewshot: int | None = 0,
    ) -> dict[str, Any]:
        """Run lm-eval benchmarks directly in-process."""
        tplr.logger.info(
            f"Running in-process evaluation for tasks: {tasks} (limit: {limit}, fewshot: {num_fewshot}) on rank {self.rank}"
        )
        start_time = time.time()

        kwargs = {}
        if num_fewshot:
            kwargs["num_fewshot"] = num_fewshot

        try:
            results: dict[str, Any] | None = simple_evaluate(
                model=lm_eval_model,
                tasks=tasks,
                limit=limit,
                batch_size=lm_eval_model.batch_size,
                device=self.device,  # Pass the device explicitly
            )
            if (
                results and self.is_master
            ):  # Only master rank should process and return results
                results["benchmark_runtime_s"] = time.time() - start_time
            return results
        except Exception as e:
            tplr.logger.exception(
                f"Error during simple_evaluate on rank {self.rank}: {e}", exc_info=True
            )
            raise e


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

        evaluator.model = evaluator.model.to("cpu")  # type: ignore
        torch.cuda.empty_cache()
        if dist.is_available() and dist.is_initialized():  # Added this block
            dist.destroy_process_group()  # Added this line

        if "DATASET_BINS_PATH" in os.environ:
            del os.environ["DATASET_BINS_PATH"]


if __name__ == "__main__":
    main()
