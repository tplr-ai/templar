# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Standard library
import argparse
import asyncio
import concurrent.futures
import copy
import os
import random
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from io import StringIO
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Dict, cast

import bittensor as bt
import numpy as np

# Third party
import torch
import uvloop
from openskill.models import PlackettLuce
from rich.console import Console
from rich.table import Table
from torch import autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from transformers import LlamaForCausalLM

# Local
import tplr

CPU_COUNT = os.cpu_count() or 4
CPU_MAX_CONNECTIONS = min(100, max(30, CPU_COUNT * 4))

# GPU optimizations.
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@contextmanager
def timer(name: str, wandb_obj=None, step=None, metrics_logger=None):
    start = perf_counter()
    yield
    duration = perf_counter() - start
    tplr.logger.debug(f"{name} took {duration:.2f}s")
    if wandb_obj and step is not None:
        wandb_obj.log({f"validator/{name}": duration}, step=step)
    if metrics_logger and step is not None:
        metrics_logger.log(
            measurement="timing", tags={"window": step}, fields={name: duration}
        )


class Validator:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description="Validator script")
        parser.add_argument(
            "--netuid", type=int, default=268, help="Bittensor network UID."
        )
        parser.add_argument(
            "--project", type=str, default="templar", help="Wandb project."
        )
        parser.add_argument(
            "--device", type=str, default="cuda", help="Device to use for training"
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument("--trace", action="store_true", help="Enable trace logging")
        parser.add_argument(
            "--test",
            action="store_true",
            help="Test mode - use all peers without filtering",
        )
        parser.add_argument(
            "--local",
            action="store_true",
            help="Local run - use toy model, small enough for a laptop.",
        )
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        config = bt.config(parser)
        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()

        return config

    def __init__(self):
        tplr.logger.info(
            "Starting initialization…"
        )  # This will log from all ranks initially

        # --------------------------------------------------------
        # 1. DDP bootstrap (must precede any CUDA work on a specific device)
        # --------------------------------------------------------
        self.config = Validator.config()  # All ranks parse CLI

        # WORLD_SIZE and LOCAL_RANK should be set by torchrun
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        # LOCAL_RANK is the rank of the process on the current node.
        # RANK is the global rank across all nodes. For single-node, RANK == LOCAL_RANK.
        # ddp_init will use RANK.
        local_rank = int(os.getenv("LOCAL_RANK", "0"))  # This is the device index

        # Pass local_rank for torch.cuda.set_device, global_rank for dist.init_process_group
        tplr.distrib.ddp_init(
            local_rank, world_size
        )  # Initializes DDP, sets device to local_rank

        self.rank = tplr.distrib.get_rank()  # Global rank
        self.world_size = tplr.distrib.get_world_size()
        self.device = (
            f"cuda:{local_rank}"
            if torch.cuda.is_available() and self.config.device == "cuda"
            else "cpu"
        )

        # Configure logging: Rank 0 full, others warning (after DDP init)
        if not tplr.distrib.is_rank0():
            tplr.logger.setLevel("WARNING")
        tplr.logger.info(
            f"[Rank {self.rank}/{self.world_size}] DDP initialized. Device: {self.device}"
        )

        # Set per-rank RNG *after* DDP init and device setting
        torch.manual_seed(42 + self.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                42 + self.rank
            )  # Seeds all GPUs, but current device is set
        np.random.seed(42 + self.rank)
        random.seed(42 + self.rank)

        # HParams: Rank 0 loads, then broadcasts
        if tplr.distrib.is_rank0():
            self.hparams = tplr.load_hparams(use_local_run_hparams=self.config.local)
        else:
            self.hparams = None  # Placeholder for non-rank0
        self.hparams = tplr.distrib.broadcast_object(self.hparams, src=0)
        tplr.logger.info(f"[Rank {self.rank}] HParams synchronized.")

        # Bittensor objects
        # Wallet and Subtensor can be initialized by all ranks
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)

        # Metagraph: fetch on every rank ( cheap ) and only broadcast the UID
        self.metagraph = self.subtensor.metagraph(cast(int, self.config.netuid))

        # Rank-0 validates registration
        if (
            tplr.distrib.is_rank0()
            and self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys
        ):
            tplr.logger.error(
                f"[Rank 0] Wallet {self.wallet} not registered on subnet {self.metagraph.netuid}"
            )
            tplr.distrib.broadcast_object({"error": "Wallet not registered"}, src=0)
            sys.exit(1)

        # Compute / broadcast UID so that all ranks are consistent
        uid_local = (
            self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            if self.wallet.hotkey.ss58_address in self.metagraph.hotkeys
            else -1
        )
        self.uid = tplr.distrib.broadcast_object(uid_local, src=0)

        # React to broadcasted error from rank-0 (if any)
        if self.uid == -1:
            tplr.logger.error(f"[Rank {self.rank}] Wallet not registered – exiting.")
            sys.exit(1)

        tplr.logger.info(
            f"[Rank {self.rank}] Metagraph and UID {self.uid} synchronized."
        )

        # --------------------------------------------------------
        # 2.  Model (no DDP wrapper for validator)
        # --------------------------------------------------------
        # All ranks create the model instance.
        self.model = LlamaForCausalLM(self.hparams.model_config).to(self.device)
        tplr.logger.info(f"[Rank {self.rank}] Model created on {self.device}.")

        # Tokenizer should be loaded consistently. Assuming hparams provide necessary info.
        self.tokenizer = self.hparams.tokenizer  # Assuming this is lightweight

        # Init compression (all ranks, based on common hparams and model structure)
        self.transformer = tplr.compress.TransformDCT(
            self.model, target_chunk=self.hparams.target_chunk
        )
        self.compressor = tplr.compress.CompressDCT()

        # Optimizer (all ranks)
        self.optimizer = SGD(self.model.parameters(), lr=self.hparams.learning_rate)

        # Momentum, xshapes, totalks (all ranks, derived from model parameters)
        self.momentum: Dict[str, torch.Tensor] = {}
        self.xshapes: Dict[str, Any] = {}
        self.totalks: Dict[str, Any] = {}

        for n, p in self.model.named_parameters():
            self.momentum[n] = torch.zeros_like(
                p, device=self.device
            )  # Ensure momentum is on correct device
            encoded_param = self.transformer.encode(self.momentum[n])
            _, _, xshape, totalk = self.compressor.compress(
                encoded_param, self.hparams.topk_compression
            )
            self.xshapes[n] = xshape
            self.totalks[n] = totalk

        # Scheduler (all ranks)
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=250
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.hparams.t_max,
            T_mult=2,
            eta_min=self.hparams.learning_rate * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[250],
        )

        self.bootstrap_version = getattr(self.hparams, "checkpoint_init_version", None)
        tplr.logger.info(
            f"[Rank {self.rank}] Validator code_version={tplr.__version__} "
            f"checkpoint_init_flag={self.bootstrap_version or '<none>'}"
        )

        # Init comms - all ranks need full Comms object
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location="/tmp",
            key_prefix="model",
            config=self.config,
            netuid=self.config.netuid,
            metagraph=self.metagraph,
            hparams=self.hparams,
            uid=self.uid,
        )

        # Only rank 0 handles bucket operations
        if tplr.distrib.is_rank0():
            self.bucket = self.comms.get_own_bucket("gradients", "read")
            self.comms.try_commit(self.wallet, self.bucket)

        # Barrier to ensure rank 0 completes bucket operations
        tplr.distrib.barrier()

        # Init state params (all ranks)
        self.stop_event = (
            asyncio.Event()
        )  # For async tasks, typically managed by rank 0 actions
        self.current_block = (
            self.subtensor.block
        )  # Will be updated by listener or broadcast
        self.current_window = int(
            self.current_block / self.hparams.blocks_per_window
        )  # Will be updated
        # start_window will be determined and synced in run()
        self.global_step = 0
        self.comms.current_window = self.current_window
        self.sync_window = self.current_window
        # self.comms.current_window needs to be updated based on synced current_window
        self.window_step = 0
        self.eval_count = 0

        # OpenSkill and scoring - all ranks need these
        self.openskill_model = PlackettLuce(
            beta=self.hparams.openskill_beta, tau=self.hparams.openskill_tau
        )
        self.openskill_ratings = {}
        self.current_window_scores = {}

        # Initialize scoring tensors - all ranks need these
        d = self.device
        self.gradient_scores = torch.zeros(256, dtype=torch.float32, device=d)
        self.sync_scores = torch.zeros(256, dtype=torch.float32, device=d)
        self.binary_indicator_scores = torch.zeros(256, dtype=torch.float32, device=d)
        self.final_scores = torch.zeros(256, dtype=torch.float32, device=d)
        self.binary_moving_averages = torch.zeros(256, dtype=torch.float32, device=d)
        self.weights = torch.zeros(256, dtype=torch.float32, device=d)
        self.evaluated_uids = set()

        # Load state if exists - only rank 0 loads, then broadcasts
        self.state_path = f"validator-state-{tplr.__version__}.pt"
        if tplr.distrib.is_rank0() and os.path.isfile(self.state_path):
            state_data = torch.load(self.state_path, map_location="cpu")
        else:
            state_data = None

        # Broadcast state data
        state_data = tplr.distrib.broadcast_object(state_data, src=0)
        if state_data is not None:
            # All ranks load the broadcast state
            for key, value in state_data.items():
                if hasattr(self, key) and isinstance(getattr(self, key), torch.Tensor):
                    getattr(self, key).copy_(value.to(self.device))
                else:
                    setattr(self, key, value)
        self.window_step = 0
        self.eval_count = 0

        # --------------------------------------------------------
        # 3. Logging / WANDB only on rank-0
        # --------------------------------------------------------
        self.wandb = None
        self.metrics_logger = None
        if tplr.distrib.is_rank0():
            self.wandb = tplr.initialize_wandb(
                run_prefix="V",
                uid=self.uid,
                config=self.config,
                group="validator",
                job_type="validation",
            )

            if self.wandb:
                tplr.logger.info(
                    f"[Rank 0 W&B] Run initialized: {self.wandb.url or 'local run'}"
                )
            else:
                tplr.logger.warning("[Rank 0 W&B] WandB initialization failed.")

            self.metrics_logger = tplr.metrics.MetricsLogger(
                prefix="V",
                uid=self.uid,
                config=self.config,
                role="validator",
                group="validator",
                job_type="validation",
            )

        tplr.distrib.barrier()  # Ensure rank 0 logging is set up before proceeding

        # Initialize remaining attributes
        self.eval_peers = defaultdict(lambda: 1)
        self.inactive_scores = {}
        self.inactivity_slash_rate = 0.25
        self.missing_gradient_slash_rate = 0.75
        self.sync_score_slash_rate = 0.75
        self.next_peers = None
        self.peers_update_window = -1
        self.peers_last_eval_window = {}

        tplr.logger.info(f"[Rank {self.rank}] Initialization complete.")

    def reset_peer(self, inactive_since: int, uid: int) -> bool:
        if self.current_window - inactive_since > self.hparams.reset_inactivity_windows:
            self.final_scores[uid] = 0.0
            self.weights[uid] = 0.0
            self.gradient_scores[uid] = 0.0
            self.binary_moving_averages[uid] = 0.0
            self.binary_indicator_scores[uid] = 0.0
            self.sync_scores[uid] = 0.0
            if uid in self.openskill_ratings:
                del self.openskill_ratings[uid]
            if uid in self.eval_peers:
                del self.eval_peers[uid]
            del self.inactive_scores[uid]
            tplr.log_with_context(
                level="info",
                message=f"UID {uid} fully reset after extended inactivity",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            return True
        return False

    def log_sync_score(
        self, eval_uid: int, sync_result: dict[str, bool | float | int | str]
    ) -> None:
        l2_norm = float(sync_result.get("l2_norm", 99.0))
        avg_l2_norm = float(sync_result.get("avg_l2_norm", 99.0))
        avg_abs_diff = float(sync_result.get("avg_abs_diff", 99.0))
        max_diff = float(sync_result.get("max_diff", 99.0))
        avg_steps_behind = float(sync_result.get("avg_steps_behind", 99.0))
        max_steps_behind = float(sync_result.get("max_steps_behind", 99.0))
        self.wandb.log(
            {
                f"validator/sync/l2_norm/{eval_uid}": l2_norm,
                f"validator/sync/avg_l2_norm/{eval_uid}": avg_l2_norm,
                f"validator/sync/avg_abs_diff/{eval_uid}": avg_abs_diff,
                f"validator/sync/sync_max_diff/{eval_uid}": max_diff,
                f"validator/sync/avg_steps_behind/{eval_uid}": avg_steps_behind,
                f"validator/sync/max_steps_behind/{eval_uid}": max_steps_behind,
            },
            step=self.global_step,
        )
        self.metrics_logger.log(
            measurement="validator_sync_score",
            tags={
                "uid": str(eval_uid),
                "window": int(self.sync_window),
                "global_step": int(self.global_step),
            },
            fields={
                "l2_norm": l2_norm,
                "avg_l2_norm": avg_l2_norm,
                "avg_abs_diff": avg_abs_diff,
                "max_diff": max_diff,
                "avg_steps_behind": avg_steps_behind,
                "max_steps_behind": max_steps_behind,
            },
            with_system_metrics=True,
            with_gpu_metrics=True,
        )

    def update_openskill_ratings(self):
        """
        Update OpenSkill ratings based on gradient scores and recalculate final scores.

        This method:
        1. Processes all peers evaluated in the current window
        2. Updates their OpenSkill ratings based on gradient performance
        3. Recalculates final scores using OpenSkill mu value combined with binary and sync scores
        4. Logs the updated ratings to monitoring systems

        The OpenSkill rating system provides a probabilistic skill rating that accounts for
        uncertainty and relative performance between peers. Ratings are updated using the
        PlackettLuce model where higher gradient scores indicate better performance.

        The final score calculation combines:
        - OpenSkill mu (mean skill estimate)
        - Binary moving average (filtered to non-negative values)
        - Sync score (model synchronization quality)
        """
        if (
            hasattr(self, "current_window_scores")
            and len(self.current_window_scores) > 1
        ):
            # Get UIDs and scores
            window_uids = list(self.current_window_scores.keys())

            # Store original ordinal values to calculate diff after update
            original_ordinals = {}
            for uid in window_uids:
                if uid in self.openskill_ratings:
                    original_ordinals[uid] = float(
                        self.openskill_ratings[uid].ordinal()
                    )
                else:
                    # For new peers without previous ratings
                    original_ordinals[uid] = 0.0

            # Calculate ranks based on gradient scores (lower rank = better performance)
            # In OpenSkill, ranks start at 1 (best) and increase for worse performers
            scores = [self.current_window_scores[uid] for uid in window_uids]

            # Create teams list for OpenSkill
            teams = [[self.openskill_ratings[uid]] for uid in window_uids]

            # Rate the teams using scores (higher score is better in OpenSkill)
            rated_teams = self.openskill_model.rate(teams, scores=scores)

            # Store updated ratings
            for i, uid in enumerate(window_uids):
                self.openskill_ratings[uid] = rated_teams[i][0]

                # Log updated OpenSkill values
                openskill_mu = float(self.openskill_ratings[uid].mu)
                openskill_sigma = float(self.openskill_ratings[uid].sigma)
                openskill_ordinal = float(self.openskill_ratings[uid].ordinal())

                sync_score = float(
                    self.sync_scores[uid].item() if uid in self.evaluated_uids else 0.0
                )

                self.final_scores[uid] = (
                    openskill_ordinal
                    * max(0, self.binary_moving_averages[uid].item())
                    * sync_score
                )
                tplr.log_with_context(
                    level="info",
                    message=f"Computed Final Score for UID {uid}: {self.final_scores[uid]}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                    eval_uid=uid,
                )

                # Log to WandB
                self.wandb.log(
                    {
                        f"validator/openskill/mu/{uid}": openskill_mu,
                        f"validator/openskill/sigma/{uid}": openskill_sigma,
                        f"validator/openskill/ordinal/{uid}": openskill_ordinal,
                    },
                    step=self.global_step,
                )

                # Log to InfluxDB
                self.metrics_logger.log(
                    measurement="validator_openskill",
                    tags={
                        "eval_uid": str(uid),
                        "window": int(self.sync_window),
                        "global_step": int(self.global_step),
                    },
                    fields={
                        "mu": openskill_mu,
                        "sigma": openskill_sigma,
                        "ordinal": openskill_ordinal,
                    },
                )

            # Create a ranking table to display current match rankings
            try:
                # Sort UIDs by current window gradient scores (descending)
                sorted_uids = sorted(
                    window_uids,
                    key=lambda uid: self.current_window_scores[uid],
                    reverse=True,
                )

                try:
                    width = os.get_terminal_size().columns
                except Exception:
                    width = 0
                os.environ["COLUMNS"] = str(max(200, width))

                rich_table = Table(
                    title=f"Current Match Rankings (Window {self.sync_window})"
                )
                rich_table.add_column("Match Rank")
                rich_table.add_column("UID")
                rich_table.add_column("Match Score")
                rich_table.add_column("OpenSkill μ (After)")
                rich_table.add_column("OpenSkill σ (After)")
                rich_table.add_column("Ordinal (After)")
                rich_table.add_column("Ordinal Δ")

                # Add rows to table
                for rank, uid in enumerate(sorted_uids, 1):
                    rating = self.openskill_ratings[uid]
                    ordinal_before = original_ordinals[uid]
                    ordinal_after = rating.ordinal()
                    ordinal_diff = ordinal_after - ordinal_before

                    # Format the diff with color indicators
                    diff_str = f"{ordinal_diff:+.4f}"

                    rich_table.add_row(
                        str(rank),
                        str(uid),
                        f"{self.current_window_scores[uid]:.6f}",
                        f"{rating.mu:.4f}",
                        f"{rating.sigma:.4f}",
                        f"{ordinal_after:.4f}",
                        diff_str,
                    )

                # Render table to string
                sio = StringIO()
                console = Console(file=sio, width=int(os.environ["COLUMNS"]))
                console.print(rich_table)
                table_str = sio.getvalue()

                tplr.log_with_context(
                    level="info",
                    message=f"Current Match Rankings (Window {self.sync_window}):\n{table_str}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
            except Exception as e:
                tplr.log_with_context(
                    level="warning",
                    message=f"Failed to create OpenSkill rankings table: {str(e)}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )

            tplr.log_with_context(
                level="info",
                message=f"Updated OpenSkill ratings for {len(window_uids)} peers based on gradient scores",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            # Clear the current window scores
            self.current_window_scores = {}

    def update_weights(self) -> None:
        """
        Update the weights for all evaluated peers using min power normalization.
        This method:
        1. Creates a mask for peers that have been evaluated
        2. Creates a mask for evaluated peers with positive scores
        3. Applies power normalization to only the positive scores
        4. Verifies that weights sum to approximately 1.0
        This approach only assigns weights to peers with positive scores.
        """
        self.weights = torch.zeros_like(self.final_scores)
        evaluated_mask = torch.zeros_like(self.final_scores, dtype=torch.bool)
        evaluated_mask[list(self.evaluated_uids)] = True

        # Create a mask for positive scores among evaluated peers
        positive_mask = evaluated_mask.clone()
        positive_mask[evaluated_mask] = self.final_scores[evaluated_mask] > 0

        # Only consider peers with positive scores
        positive_scores = self.final_scores[positive_mask]

        if len(positive_scores) > 0:
            # Apply power normalization to only the positive scores
            normalized_weights = min_power_normalization(
                positive_scores,
                power=self.hparams.power_normalisation,
            )

            # Assign weights only to peers with positive scores
            self.weights[positive_mask] = normalized_weights

            weight_sum = self.weights.sum().item()
            tplr.log_with_context(
                level="debug",
                message=f"Weight sum: {weight_sum}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            if abs(weight_sum - 1.0) > 1e-6:
                tplr.log_with_context(
                    level="warning",
                    message=f"Weights sum to {weight_sum}, expected close to 1.0",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
        else:
            tplr.log_with_context(
                level="warning",
                message="No positive scores found among evaluated peers. All weights set to zero.",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

    def evaluate_model_on_batches(
        self,
        model: torch.nn.Module,
        batches: list[list[int]],
        sampled_indices: list[int],
    ) -> tuple[float, int]:
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            model.eval()
            with autocast(device_type=self.model.device.type, dtype=torch.bfloat16):
                for i, batch in enumerate(batches):
                    if i not in sampled_indices:
                        continue
                    input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                    labels = input_ids.clone()
                    labels = torch.where(
                        labels == self.tokenizer.pad_token_id, -100, labels
                    )
                    outputs = model(input_ids=input_ids, labels=labels)
                    total_loss += outputs.loss.item()
                    n_batches += 1
                    del input_ids, labels, outputs
                    torch.cuda.empty_cache()

        return total_loss, n_batches

    async def run(self):
        # Start background block listener on rank 0 only
        self.loop = asyncio.get_running_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
        self.loop.set_default_executor(self.executor)

        if tplr.distrib.is_rank0():
            self.listener = threading.Thread(
                target=self.block_listener, args=(self.loop,), daemon=True
            ).start()

        # Use config peers if provided - only rank 0 handles this
        if tplr.distrib.is_rank0():
            if self.config.peers:
                self.comms.peers = self.config.peers

            self.comms.commitments = await self.comms.get_commitments()
            self.comms.update_peers_with_buckets()
            tplr.logger.info("Loaded commitments")

            # Prepare data for broadcast
            comms_data = {
                "peers": self.comms.peers,
                "commitments": self.comms.commitments,
                "eval_peers": self.comms.eval_peers,
            }
        else:
            comms_data = None

        # Broadcast comms data to all ranks
        comms_data = tplr.distrib.broadcast_object(comms_data, src=0)

        # Non-rank 0 processes update their local comms state
        if not tplr.distrib.is_rank0():
            self.comms.peers = comms_data["peers"]
            self.comms.commitments = comms_data["commitments"]
            self.comms.eval_peers = comms_data["eval_peers"]
            # Add other necessary attributes as needed
            self.comms.inactive_peers = []

        # Only post start window if you are the highest stake validator and rank 0
        if tplr.distrib.is_rank0() and self.uid == self.metagraph.S.argmax().item():
            # Check if an existing start window already exists
            try:
                existing_start_window = await self.comms.get_start_window(retries=2)
            except Exception as e:
                tplr.logger.warning(f"Error fetching existing start_window: {e}")
                existing_start_window = None

            if existing_start_window is not None:
                self.start_window = existing_start_window
                tplr.logger.info(
                    f"Highest staked validator found existing start_window: {self.start_window}"
                )
            else:
                # Initialize start_window before using it
                self.start_window = self.current_window
                # No existing start window, so post new start window to R2
                await self.comms.post_start_window(self.start_window)
                tplr.logger.info(
                    f"This validator is the highest staked. Posted start_window: {self.start_window}"
                )
        elif tplr.distrib.is_rank0():
            tplr.logger.info(
                "This validator is not the highest staked. Waiting to fetch start_window."
            )
            self.start_window = await self.comms.get_start_window()
        else:
            # Initialize start_window for non-rank0 processes
            self.start_window = None

        # Broadcast start_window from rank 0 to all ranks
        self.start_window = tplr.distrib.broadcast_object(
            self.start_window if tplr.distrib.is_rank0() else None, src=0
        )

        if self.start_window is None:
            raise RuntimeError(
                "Could not find a valid start window. This should not be possible."
            )

        self.global_step = self.current_window - self.start_window

        tplr.logger.info(
            f"Using start_window: {self.start_window}, global_step: {self.global_step}, sync_window: {self.sync_window}"
        )

        checkpoint_window_buffer = 5
        has_new_checkpoint = (
            self.global_step
            >= self.hparams.checkpoint_frequency + checkpoint_window_buffer
        )

        # Proceed to load checkpoint - rank 0 loads, then broadcasts to all ranks
        if tplr.distrib.is_rank0():
            # Rank 0 loads the checkpoint
            (
                success,
                loaded_momentum,
                loaded_checkpoint_window,
                loaded_optimizer,
                loaded_scheduler,
            ) = await self.comms.load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                current_window=self.current_window,
                device=self.config.device,
                init_version=tplr.__version__
                if has_new_checkpoint
                else self.bootstrap_version,
            )

            if success:
                # Prepare checkpoint data for broadcasting
                self.momentum = loaded_momentum
                self.optimizer = loaded_optimizer
                self.scheduler = loaded_scheduler

                # Create serializable checkpoint data
                checkpoint_data = {
                    "model_state": {
                        k: v.cpu() for k, v in self.model.state_dict().items()
                    },
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "momentum": {n: m.cpu() for n, m in self.momentum.items()},
                    "checkpoint_window": loaded_checkpoint_window,
                }
            else:
                checkpoint_data = None
        else:
            success = None
            checkpoint_data = None
            loaded_checkpoint_window = None

        # Broadcast checkpoint loading result and data
        broadcast_result = tplr.distrib.broadcast_object(
            (success, checkpoint_data) if tplr.distrib.is_rank0() else (None, None),
            src=0,
        )

        success_on_rank0, checkpoint_data = broadcast_result

        # If checkpoint was loaded on rank 0, all ranks update their local state
        if success_on_rank0:
            if not tplr.distrib.is_rank0():
                # Non-rank 0 processes load the broadcast state
                self.model.load_state_dict(checkpoint_data["model_state"])
                self.optimizer.load_state_dict(checkpoint_data["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint_data["scheduler_state"])
                self.momentum = {
                    n: m.to(self.device) for n, m in checkpoint_data["momentum"].items()
                }

            loaded_checkpoint_window = checkpoint_data["checkpoint_window"]

            tplr.logger.info(
                f"Loaded checkpoint with global_step={self.global_step}, "
                f"optimizer_step={self.optimizer.state_dict()['state'].get(0, {}).get('step', 0)}, "
                f"scheduler_step={self.scheduler.last_epoch}"
            )

            # Only catch up if we're behind and we're rank 0
            if (
                tplr.distrib.is_rank0()
                and loaded_checkpoint_window < self.current_window
                and self.global_step > checkpoint_window_buffer
            ):
                tplr.logger.info(
                    f"Checkpoint is behind current window ({loaded_checkpoint_window} < {self.current_window}), starting catchup..."
                )
                await tplr.neurons.catchup_with_aggregation_server(
                    self, max(loaded_checkpoint_window, self.start_window)
                )

                # After catchup, broadcast updated model state to all ranks
                updated_model_state = {
                    k: v.cpu() for k, v in self.model.state_dict().items()
                }
                updated_model_state = tplr.distrib.broadcast_object(
                    updated_model_state if tplr.distrib.is_rank0() else None, src=0
                )

                if not tplr.distrib.is_rank0():
                    self.model.load_state_dict(updated_model_state)
            else:
                tplr.logger.info("Checkpoint is up-to-date, skipping catchup.")
        else:
            tplr.logger.info("Starting from scratch")
            self.momentum = {
                n: torch.zeros_like(p) for n, p in self.model.named_parameters()
            }
            self.model.to(self.config.device)

        if tplr.distrib.is_rank0():  # Commitment fetcher only on rank 0
            self.comms.start_commitment_fetcher()
            self.comms.start_background_tasks()
        time_min = None
        self.last_peer_update_window = None
        self.last_peer_post_window = None

        # Barrier to ensure all ranks are synchronized before entering main loop
        tplr.distrib.barrier()

        while True:
            try:
                # Synchronize window information across ranks
                if tplr.distrib.is_rank0():
                    # 1. Wait for the validator window offset - USE THE ORIGINAL LOGIC
                    while self.sync_window >= (
                        self.current_window - self.hparams.validator_offset
                    ):
                        tplr.log_with_context(
                            level="info",
                            message=f"Waiting for validator window offset, synced: {self.sync_window}, current:{self.current_window}, offset:{self.hparams.validator_offset}, gap:{self.current_window - self.sync_window}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                        )
                        await asyncio.sleep(12)

                    # 2. Increment sync window and update peer lists
                    self.sync_window += 1

                # Broadcast current_block, current_window, sync_window from rank 0
                sync_data = tplr.distrib.broadcast_object(
                    (self.current_block, self.current_window, self.sync_window)
                    if tplr.distrib.is_rank0()
                    else None,
                    src=0,
                )

                if not tplr.distrib.is_rank0():
                    # Update local values on non-rank 0 processes
                    self.current_block, self.current_window, self.sync_window = (
                        sync_data
                    )

                # Barrier to ensure all ranks are synchronized
                tplr.distrib.barrier()

                window_start = tplr.T()

                if tplr.distrib.is_rank0():
                    tplr.log_with_context(
                        level="info",
                        message=f"Sync Window: {self.sync_window}, Scheduler epoch: {self.scheduler.last_epoch}, Global step: {self.global_step}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )

                    tplr.log_with_context(
                        level="info",
                        message=f"Processing window: {self.sync_window} current: {self.current_window}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )

                # Save state
                if tplr.distrib.is_rank0():
                    self.save_state()

                # Create and post peers - only rank 0 handles peer selection and posting
                initial_selection = False
                if tplr.distrib.is_rank0():
                    if (
                        self.last_peer_update_window is None
                        or self.sync_window - self.last_peer_update_window
                        >= self.hparams.peer_replacement_frequency
                    ):
                        reason = (
                            f"{self.last_peer_update_window=}"
                            if self.last_peer_update_window is None
                            else f"{self.sync_window=}>="
                            f"{self.last_peer_update_window}+"
                            f"{self.hparams.peer_replacement_frequency}="
                            "self.last_peer_update_window+"
                            "self.hparams.peer_replacement_frequency"
                        )

                        tplr.log_with_context(
                            level="info",
                            message=f"Time to create and post a new peer list because {reason}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                        )
                        if self.last_peer_update_window is None:
                            selected_peers = self.select_initial_peers()
                            initial_selection = True
                        else:
                            selected_peers = self.select_next_peers()
                        if selected_peers is not None:
                            self.last_peer_update_window = self.sync_window
                            await self.comms.post_peer_list(
                                peers=selected_peers,
                                first_effective_window=self.current_window
                                + self.hparams.peer_list_window_margin,
                                sync_window=self.sync_window,
                                weights=self.weights,
                                initial_selection=initial_selection,
                            )

                # Update peers on all ranks
                self.comms.update_peers_with_buckets()
                peer_start = tplr.T()
                await tplr.neurons.update_peers(
                    instance=self, window=self.sync_window, peer_start=peer_start
                )

                # Broadcast eval_peers from rank 0 to ensure consistency
                if tplr.distrib.is_rank0():
                    self.eval_peers = self.comms.eval_peers
                    eval_peers_to_broadcast = self.eval_peers
                else:
                    eval_peers_to_broadcast = None

                self.eval_peers = tplr.distrib.broadcast_object(
                    eval_peers_to_broadcast if tplr.distrib.is_rank0() else None, src=0
                )

                if tplr.distrib.is_rank0():
                    tplr.log_with_context(
                        level="info",
                        message=f"{tplr.P(self.sync_window, tplr.T() - peer_start)} Updated peers - eval:{len(self.eval_peers)}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )

                    tplr.log_with_context(
                        level="info",
                        message=f"Current gather peers: {self.comms.peers}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                    tplr.log_with_context(
                        level="info",
                        message=f"Current evaluation peers: {self.eval_peers}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )

                # Check if we have any peers to work with
                peers_available = True
                if tplr.distrib.is_rank0():
                    try:
                        # Handle case where peers might be numpy array, tensor, or list
                        if hasattr(self.comms.peers, "__len__"):
                            peers_available = len(self.comms.peers) > 0
                        elif hasattr(self.comms.peers, "numel"):  # torch tensor
                            peers_available = self.comms.peers.numel() > 0
                        elif hasattr(self.comms.peers, "size"):  # numpy array
                            peers_available = self.comms.peers.size > 0
                        else:
                            peers_available = bool(self.comms.peers)
                    except Exception as e:
                        tplr.log_with_context(
                            level="warning",
                            message=f"Error checking peers availability: {e}. Assuming no peers.",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                        )
                        peers_available = False

                    if not peers_available:
                        tplr.log_with_context(
                            level="warning",
                            message="No peers available for gathering. Skipping this window.",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                        )

                    skip_window = not peers_available
                else:
                    skip_window = False

                # Broadcast skip decision to all ranks
                skip_window = tplr.distrib.broadcast_object(
                    skip_window if tplr.distrib.is_rank0() else None, src=0
                )

                if skip_window:
                    # All ranks skip this window together
                    self.global_step += 1
                    tplr.distrib.barrier()
                    continue

                # Process inactive peers - only rank 0 updates scores
                if tplr.distrib.is_rank0():
                    newly_inactive = self.comms.inactive_peers
                    current_window = self.sync_window

                    # Process inactive peers and apply penalties
                    for uid in newly_inactive:
                        if uid not in self.inactive_scores:
                            self.inactive_scores[uid] = (
                                current_window,
                                self.final_scores[uid].item(),
                            )
                            tplr.log_with_context(
                                level="info",
                                message=f"UID {uid} became inactive at window {current_window} with score {self.final_scores[uid].item():.4f}",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                            )

                    # Apply penalties to all inactive peers
                    for uid, (inactive_since, _) in list(self.inactive_scores.items()):
                        # If peer became active again, remove from inactive tracking
                        if uid in self.eval_peers.keys():
                            del self.inactive_scores[uid]
                            tplr.log_with_context(
                                level="info",
                                message=f"UID {uid} became active again",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                            )
                            continue

                        peer_reset = self.reset_peer(inactive_since, uid)
                        if peer_reset:
                            continue

                        # Apply flat 25% penalty instead of exponential decay
                        old_score = self.final_scores[uid].item()
                        new_score = (
                            old_score  # Initialize new_score with old_score value
                        )
                        if self.final_scores[uid] > 0:
                            self.final_scores[uid] *= (
                                0.75  # Apply flat 25% reduction for positive scores only
                            )

                            new_score = self.final_scores[uid].item()

                            tplr.log_with_context(
                                level="info",
                                message=f"UID {uid} penalized for inactivity: {old_score:.4f} -> {new_score:.4f}",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                            )

                        # Log slash metrics to WandB
                        self.wandb.log(
                            {
                                f"validator/inactivity/{uid}/score_before": old_score,
                                f"validator/inactivity/{uid}/score_after": new_score,
                            },
                            step=self.global_step,
                        )

                        # Log slash metrics to InfluxDB with primitive types
                        self.metrics_logger.log(
                            measurement="validator_inactivity",
                            tags={
                                "uid": str(uid),
                                "window": int(current_window),
                                "global_step": int(self.global_step),
                            },
                            fields={
                                "score_before": float(old_score),
                                "score_after": float(new_score),
                            },
                            with_system_metrics=True,
                            with_gpu_metrics=True,
                        )

                # Calculate time window for this sync window
                if tplr.distrib.is_rank0():
                   
                    sync_block = (self.sync_window + 1) * self.hparams.blocks_per_window
                    retries = 0
                    delay = 1
                    max_retries = 2
                    max_delay = 60
                    while True:
                        try:
                            # Query timestamp for target block instead of current block
                            response = self.subtensor.query_module("Timestamp", "Now", block=sync_block)
                            ts_value = response.value / 1000  # convert ms to seconds
                            tplr.log_with_context(
                                level="info", 
                                message=f"Queried timestamp for target block {sync_block}: {ts_value}",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                            )
                            break
                        except Exception as e:
                            tplr.log_with_context(
                                level="error",
                                message=f"Failed to query timestamp for block {sync_block}: {str(e)}. Retry {retries + 1}/{max_retries}",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                            )
                            retries += 1
                            if retries > max_retries:
                                tplr.log_with_context(
                                    level="error",
                                    message="Exceeded maximum retries for timestamp query. Falling back to current system time.",
                                    sync_window=self.sync_window,
                                    current_window=self.current_window,
                                )
                                ts_value = time.time()  # Fallback: use current system time
                                break
                            await asyncio.sleep(delay)
                            delay = min(delay * 2, max_delay)
                    time_min = datetime.fromtimestamp(ts_value, tz=timezone.utc)
                    time_max = time_min + timedelta(
                        seconds=self.hparams.time_window_delta_seconds
                    )

                    # Log the time window we're using
                    tplr.log_with_context(
                        level="info",
                        message=f"Using time window for gather: {time_min} to {time_max}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                    tplr.log_with_context(
                        level="info",
                        message=f"We are using peers {self.comms.peers}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                else:
                    time_min = None
                    time_max = None

                # Broadcast time window from rank 0 to all ranks
                time_window = tplr.distrib.broadcast_object(
                    (time_min, time_max) if tplr.distrib.is_rank0() else None, src=0
                )
                time_min, time_max = time_window

                # Refresh peers explicitly before starting gather to avoid missing updated active peers.
                if tplr.distrib.is_rank0():
                    tplr.log_with_context(
                        level="info",
                        message="Refreshing eval peers before gather task in validator...",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )

                    if self.config.test:
                        # In test mode, use all UIDs from metagraph except self
                        tplr.log_with_context(
                            level="info",
                            message="Test mode active: Using all peers from metagraph.",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                        )
                        all_uids = list(range(len(self.metagraph.S)))
                        self.comms.peers = [uid for uid in all_uids if uid != self.uid]

                        # For evaluation, also use all peers but track separately with equal initial weight
                        self.eval_peers = {uid: 1 for uid in self.comms.peers}
                    else:
                        # Normal operation - update and filter peers
                        self.comms.update_peers_with_buckets()
                        self.eval_peers = self.comms.eval_peers

                    tplr.log_with_context(
                        level="info",
                        message=f"Validator gather peers: {self.comms.peers}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )

                # Broadcast updated peers to all ranks
                peers_data = tplr.distrib.broadcast_object(
                    (self.comms.peers, self.eval_peers)
                    if tplr.distrib.is_rank0()
                    else None,
                    src=0,
                )
                self.comms.peers, self.eval_peers = peers_data

                # Gradient gathering/loading aggregation - only rank 0 handles this
                aggregation_result = None
                gather_result = None

                if tplr.distrib.is_rank0():
                    gather_start = tplr.T()

                    # Try to load from aggregation server first
                    aggregation_result = await self.comms.load_aggregation(
                        window=self.sync_window,
                    )

                    if aggregation_result is None:
                        gather_result = await self.comms.gather(
                            my_uid=self.uid,
                            uids=self.comms.peers,
                            window=self.sync_window,
                            key="gradient",
                            timeout=35,
                            local=False,
                            totalks=self.totalks,
                            time_min=time_min,
                            time_max=time_max,
                            device=self.device,  # Add the missing device parameter
                        )

                        if gather_result is None:
                            tplr.log_with_context(
                                level="error",
                                message="Failed to gather gradients from peers. Waiting for next window.",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                            )
                            self.global_step += 1
                            continue

                        skipped_uids = gather_result.skipped_uids
                        success_rate = gather_result.success_rate
                    else:
                        state_dict = cast(dict, aggregation_result.get("state_dict"))
                        skipped_uids = cast(
                            list[int], state_dict.get("skipped_uids", [])
                        )
                        success_rate = cast(float, state_dict.get("success_rate", 0.0))

                    gather_time = tplr.T() - gather_start

                    from_aggregator = 1 if aggregation_result is not None else 0
                    tplr.log_with_context(
                        level="info",
                        message=f"Using gradient source: {'aggregator' if from_aggregator else 'gather'} (took {gather_time:.2f}s)",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )

                    self.wandb.log(
                        {
                            "validator/aggregator_gradient": from_aggregator,
                            "validator/gather_time": gather_time,
                        },
                        step=self.global_step,
                    )

                    tplr.log_with_context(
                        level="info",
                        message=f"Skipped UIDs: {skipped_uids}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                else:
                    skipped_uids = None
                    success_rate = None

                # Broadcast gradient results to all ranks
                gradient_data = tplr.distrib.broadcast_object(
                    (aggregation_result, gather_result, skipped_uids, success_rate)
                    if tplr.distrib.is_rank0()
                    else None,
                    src=0,
                )
                aggregation_result, gather_result, skipped_uids, success_rate = (
                    gradient_data
                )

                # Miner evaluation workload distribution
                evaluation_uids = []
                if tplr.distrib.is_rank0():
                    # Determine evaluation UIDs using binning strategy
                    num_bins = self.hparams.num_evaluation_bins
                    bins = self.bin_evaluation_peers(num_bins)
                    selected_bin = self.select_next_bin_for_evaluation(num_bins)
                    evaluation_uids = self.select_evaluation_uids_from_bin(
                        bins, selected_bin
                    )

                    tplr.log_with_context(
                        level="info",
                        message=f"Selected {len(evaluation_uids)} UIDs for evaluation: {evaluation_uids}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )

                    # Split evaluation UIDs across ranks
                    if len(evaluation_uids) > 0:
                        uids_chunks = [
                            list(arr)
                            for arr in np.array_split(
                                evaluation_uids, tplr.distrib.get_world_size()
                            )
                        ]
                    else:
                        uids_chunks = [[] for _ in range(tplr.distrib.get_world_size())]
                else:
                    uids_chunks = None

                # Broadcast UID chunks to all ranks
                uids_chunks = tplr.distrib.broadcast_object(
                    uids_chunks if tplr.distrib.is_rank0() else None, src=0
                )

                # Each rank gets its subset of UIDs to evaluate
                uids_for_my_rank = uids_chunks[tplr.distrib.get_rank()]

                tplr.log_with_context(
                    level="info",
                    message=f"[Rank {self.rank}/{self.world_size}] Assigned {len(uids_for_my_rank)} UIDs for evaluation: {uids_for_my_rank}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )

                # Parallel miner evaluation - each rank processes its subset
                local_evaluation_results = {}

                # Generate common random seed for shared dataloader - rank 0 decides
                if tplr.distrib.is_rank0():
                    common_random_seed = random.randint(
                        256, 10000
                    )  # > 255 for random context
                else:
                    common_random_seed = None

                common_random_seed = tplr.distrib.broadcast_object(
                    common_random_seed if tplr.distrib.is_rank0() else None, src=0
                )

                # Each rank evaluates its assigned UIDs
                for eval_uid in uids_for_my_rank:
                    try:
                        # Create temporary model copy for this evaluation
                        model_eval_copy = copy.deepcopy(self.model).to(self.device)
                        self.log_gpu_memory_usage(f"rank_{self.rank}_after_model_copy")

                        optimizer_eval_copy = SGD(
                            model_eval_copy.parameters(), lr=self.hparams.learning_rate
                        )

                        # Preload dataloaders
                        loader_own_result = await self.preload_dataloader(seed=eval_uid)
                        loader_random_result = await self.preload_dataloader(
                            seed=common_random_seed
                        )

                        if loader_own_result is None or loader_random_result is None:
                            tplr.log_with_context(
                                level="warning",
                                message=f"[Rank {self.rank}] Failed to preload dataloaders for UID {eval_uid}",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                            )
                            continue

                        loader_own = loader_own_result["loader"]
                        loader_random = loader_random_result["loader"]
                        local_pages = loader_own_result["pages"]  # ← FIX: Add missing variable

                        # Add detailed logging for gradient fetching across all ranks
                        # Get miner's gradient
                        tplr.log_with_context(
                            level="info",
                            message=f"[Rank {self.rank}] Attempting to fetch gradient for UID {eval_uid} with time window {time_min} to {time_max}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                        gradient_result = await self.comms.get(
                            uid=str(eval_uid),
                            window=self.sync_window,
                            key="gradient",
                            local=False,
                            stale_retention=10,
                            time_max=time_max,  # ADD THIS
                            time_min=time_min,  # ADD THIS
                        )

                        tplr.log_with_context(
                            level="info",
                            message=f"[Rank {self.rank}] Gradient fetch result for UID {eval_uid}: {gradient_result is not None}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                        if gradient_result is None:
                            tplr.log_with_context(
                                level="warning",
                                message=f"[Rank {self.rank}] Failed to get gradient for UID {eval_uid} - no data returned",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                                eval_uid=eval_uid,
                            )
                            local_evaluation_results[eval_uid] = {
                                "gradient_score": 0.0,
                                "binary_indicator_score": 0.0,
                                "sync_score": 0.0,
                                "success": False,
                                "error": "No gradient data returned",
                            }
                            continue

                        # Check for status errors in gradient result
                        if isinstance(gradient_result, dict) and gradient_result.get("__status") in ["TOO_LATE", "TOO_EARLY"]:
                            tplr.log_with_context(
                                level="warning",
                                message=f"[Rank {self.rank}] Gradient fetch for UID {eval_uid} failed with status: {gradient_result.get('__status')}",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                                eval_uid=eval_uid,
                            )
                            local_evaluation_results[eval_uid] = {
                                "gradient_score": 0.0,
                                "binary_indicator_score": 0.0,
                                "sync_score": 0.0,
                                "success": False,
                                "error": f"Gradient fetch status: {gradient_result.get('__status')}",
                            }
                            continue

                        # Extract the gradient data (should be tuple: (state_dict, metadata))
                        if isinstance(gradient_result, (list, tuple)) and len(gradient_result) >= 1:
                            state_dict = gradient_result[0]
                            metadata = gradient_result[1] if len(gradient_result) > 1 else None
                        else:
                            state_dict = gradient_result
                            metadata = None

                        # Extract pages info from metadata
                        miner_pages = None
                        if metadata and isinstance(metadata, dict) and "pages_info" in metadata:
                            miner_pages = metadata["pages_info"]
                        elif isinstance(state_dict, dict) and "metadata" in state_dict:
                            miner_pages = state_dict["metadata"].get("pages_info")

                        # Verify pages match if miner sent them
                        if miner_pages is not None:
                            if local_pages != miner_pages:
                                tplr.log_with_context(
                                    level="warning",
                                    message=f"Pages mismatch for UID {eval_uid}: miner sent {miner_pages} vs local pages {local_pages}",
                                    sync_window=self.sync_window,
                                    current_window=self.current_window,
                                    eval_uid=eval_uid,
                                )
                            else:
                                tplr.log_with_context(
                                    level="info",
                                    message=f"Pages verified for UID {eval_uid}: pages match",
                                    sync_window=self.sync_window,
                                    current_window=self.current_window,
                                    eval_uid=eval_uid,
                                )
                        else:
                            tplr.log_with_context(
                                level="info",
                                message=f"Using local pages for UID {eval_uid} as miner metadata is missing",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                                eval_uid=eval_uid,
                            )

                        # Evaluate model performance on both dataloaders
                        eval_start = tplr.T()

                        # Collect all batches first
                        own_batches = []
                        for batch in loader_own:
                            own_batches.append(batch)

                        random_batches = []
                        for batch in loader_random:
                            random_batches.append(batch)

                        # Apply sampling rate logic 
                        total_batches_own = len(own_batches)
                        sample_size_own = max(
                            1,
                            int(total_batches_own * self.hparams.validator_sample_rate),
                        )
                        sampled_indices_own = random.sample(
                            range(total_batches_own), sample_size_own
                        )
                        sampled_indices_own = sorted(
                            sampled_indices_own
                        )  # Sort for sequential access

                        total_batches_random = len(random_batches)
                        sample_size_random = max(
                            1,
                            int(
                                total_batches_random
                                * self.hparams.validator_sample_rate
                            ),
                        )
                        sampled_indices_random = random.sample(
                            range(total_batches_random), sample_size_random
                        )
                        sampled_indices_random = sorted(sampled_indices_random)

                        tplr.log_with_context(
                            level="info",
                            message=f"Evaluating {sample_size_own}/{total_batches_own} own batches and {sample_size_random}/{total_batches_random} random batches ({self.hparams.validator_sample_rate * 100:.1f}%)",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                        )

                        # Evaluate before applying gradient
                        loss_before_own, n_batches_own = self.evaluate_model_on_batches(
                            model_eval_copy, own_batches, sampled_indices_own
                        )
                        loss_before_random, n_batches_random = (
                            self.evaluate_model_on_batches(
                                model_eval_copy, random_batches, sampled_indices_random
                            )
                        )

                        # Calculate per-batch averages
                        loss_before_own_per_batch = (
                            loss_before_own / n_batches_own if n_batches_own > 0 else 0
                        )
                        loss_before_random_per_batch = (
                            loss_before_random / n_batches_random
                            if n_batches_random > 0
                            else 0
                        )

                        tplr.log_with_context(
                            level="info",
                            message=f"[Rank {self.rank}] UID {eval_uid} Loss before (own data): {loss_before_own_per_batch:.6f}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                        tplr.log_with_context(
                            level="info",
                            message=f"[Rank {self.rank}] UID {eval_uid} Loss before (random data): {loss_before_random_per_batch:.6f}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                        # Apply miner's gradient to the temporary model
                        miner_gradient = gradient_result[0]
                        if miner_gradient is not None:
                            try:
                                # First validate all gradients before applying any
                                validation_step = "structure_check"
                                for name, param in model_eval_copy.named_parameters():
                                    if name in miner_gradient:
                                        compressed_grad = miner_gradient[name]

                                        # Validate compressed gradient structure
                                        if not isinstance(compressed_grad, dict):
                                            raise ValueError(
                                                f"[{validation_step}] Invalid gradient format for {name}: expected dict, got {type(compressed_grad)}"
                                            )

                                        validation_step = "indices_values_check"
                                        idxs = compressed_grad.get("idxs")
                                        vals = compressed_grad.get("vals")

                                        if idxs is None or vals is None:
                                            raise ValueError(
                                                f"[{validation_step}] Missing indices or values for {name}: idxs={idxs is not None}, vals={vals is not None}"
                                            )

                                        validation_step = "device_transfer"
                                        # Move to correct device
                                        idxs = idxs.to(self.device)
                                        vals = vals.to(self.device)

                                        validation_step = "bounds_check"
                                        # Validate indices are within bounds
                                        if self.totalks.get(name) is None:
                                            raise ValueError(
                                                f"[{validation_step}] Missing totalk for parameter {name}"
                                            )

                                        validation_step = "nan_inf_check"
                                        # Check for NaN or Inf values
                                        if (
                                            torch.isnan(vals).any()
                                            or torch.isinf(vals).any()
                                        ):
                                            nan_count = torch.isnan(vals).sum().item()
                                            inf_count = torch.isinf(vals).sum().item()
                                            raise ValueError(
                                                f"[{validation_step}] Invalid values in gradient for {name}: {nan_count} NaN, {inf_count} Inf values"
                                            )

                                # If all validations pass, apply the gradients using the OLD METHOD
                                for name, param in model_eval_copy.named_parameters():
                                    if name in miner_gradient:
                                        compressed_grad = miner_gradient[name]
                                        idxs = compressed_grad["idxs"].to(self.device)
                                        vals = compressed_grad["vals"].to(self.device)

                                        try:
                                            # Decompress the gradient
                                            decompressed_grad = self.compressor.decompress(
                                                compressed_grad,
                                                self.xshapes[name],
                                                self.totalks[name],
                                            )
                                            # Decode using transformer
                                            decoded_grad = self.transformer.decode(
                                                decompressed_grad, name
                                            )

                                            # Final safety check on the gradient
                                            if (
                                                torch.isnan(decoded_grad).any()
                                                or torch.isinf(decoded_grad).any()
                                            ):
                                                raise ValueError(
                                                    f"NaN/Inf in decompressed gradient for {name}"
                                                )

                                            # FIXED: Use the old code method instead of momentum/optimizer
                                            param.data.sub_(
                                                decoded_grad.sign(),
                                                alpha=self.scheduler.get_last_lr()[0] * self.hparams.eval_lr_factor,
                                            )

                                        except Exception as e:
                                            raise ValueError(
                                                f"Failed to decompress gradient for {name}: {e}"
                                            )

                                # Apply the gradients using optimizer step
                                optimizer_eval_copy.step()
                                optimizer_eval_copy.zero_grad()

                            except Exception as e:
                                tplr.log_with_context(
                                    level="warning",
                                    message=f"Invalid gradient data from UID {eval_uid}: {e}",
                                    sync_window=self.sync_window,
                                    current_window=self.current_window,
                                )

                                # Record failed evaluation with zero scores
                                local_evaluation_results[eval_uid] = {
                                    "gradient_score": 0.0,
                                    "binary_indicator_score": 0.0,
                                    "sync_score": 0.0,
                                    "success": False,
                                    "error": str(e),
                                }
                                continue  # Skip to next UID

                        # Evaluate after applying gradient
                        loss_after_own, _ = self.evaluate_model_on_batches(
                            model_eval_copy, own_batches, sampled_indices_own
                        )
                        loss_after_random, _ = self.evaluate_model_on_batches(
                            model_eval_copy, random_batches, sampled_indices_random
                        )

                        # Calculate per-batch averages
                        loss_after_own_per_batch = loss_after_own / n_batches_own if n_batches_own > 0 else 0
                        loss_after_random_per_batch = loss_after_random / n_batches_random if n_batches_random > 0 else 0

                        tplr.log_with_context(
                            level="info",
                            message=f"[Rank {self.rank}] UID {eval_uid} Loss after (own data): {loss_after_own_per_batch:.6f}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                        tplr.log_with_context(
                            level="info",
                            message=f"[Rank {self.rank}] UID {eval_uid} Loss after (random data): {loss_after_random_per_batch:.6f}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                        # Use per-batch values for score calculations
                        improvement_own = loss_before_own_per_batch - loss_after_own_per_batch
                        improvement_random = loss_before_random_per_batch - loss_after_random_per_batch

                        tplr.log_with_context(
                            level="info",
                            message=f"[Rank {self.rank}] UID {eval_uid} Loss improvement (own data): {improvement_own:.6f}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                        tplr.log_with_context(
                            level="info",
                            message=f"[Rank {self.rank}] UID {eval_uid} Loss improvement (random data): {improvement_random:.6f}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                        # Binary indicator score (1 if improvement on own data, 0 otherwise)
                        binary_indicator_score = 1.0 if improvement_own > 0 else 0.0

                        # FIXED: Use the old code gradient score calculation formula
                        gradient_score = (loss_before_random_per_batch - loss_after_random_per_batch) / loss_before_random_per_batch if loss_before_random_per_batch > 0 else 0.0

                        tplr.log_with_context(
                            level="info",
                            message=f"[Rank {self.rank}] UID {eval_uid} Gradient Score: {gradient_score:.6f}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                        tplr.log_with_context(
                            level="info",
                            message=f"[Rank {self.rank}] UID {eval_uid} Binary Indicator Score: {binary_indicator_score}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                        # Evaluate synchronization
                        sync_result = await self.evaluate_miner_sync(
                            eval_uid, model_eval_copy
                        )
                        sync_score = sync_result.get("sync_score", 0.0)

                        tplr.log_with_context(
                            level="info",
                            message=f"[Rank {self.rank}] UID {eval_uid} Sync Score: {sync_score:.6f}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                        eval_time = tplr.T() - eval_start

                        local_evaluation_results[eval_uid] = {
                            "gradient_score": float(gradient_score),
                            "binary_indicator_score": float(binary_indicator_score),
                            "sync_score": float(sync_score),
                            "loss_before_own": float(
                                loss_before_own_per_batch
                            ),  # Use per-batch values
                            "loss_after_own": float(loss_after_own_per_batch),
                            "loss_before_random": float(loss_before_random_per_batch),
                            "loss_after_random": float(loss_after_random_per_batch),
                            "improvement_own": float(improvement_own),
                            "improvement_random": float(improvement_random),
                            "eval_time": float(eval_time),  # Add timing
                            "success": True,
                        }

                        tplr.log_with_context(
                            level="info",
                            message=f"[Rank {self.rank}] Completed evaluation for UID {eval_uid}: gradient_score={gradient_score:.4f}, binary={binary_indicator_score}, sync={sync_score:.4f} (took {eval_time:.2f}s)",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                        # Memory monitoring before cleanup
                        self.log_gpu_memory_usage(f"rank_{self.rank}_before_cleanup")

                        # Cleanup
                        del model_eval_copy, optimizer_eval_copy
                        torch.cuda.empty_cache()
                        self.log_gpu_memory_usage(f"rank_{self.rank}_after_cleanup")

                    except Exception as e:
                        tplr.log_with_context(
                            level="error",
                            message=f"[Rank {self.rank}] Error evaluating UID {eval_uid}: {e}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                        )

                        local_evaluation_results[eval_uid] = {
                            "gradient_score": 0.0,
                            "binary_indicator_score": 0.0,
                            "sync_score": 0.0,
                            "success": False,
                            "error": str(e),
                            "loss_before_own": 0.0,
                            "loss_after_own": 0.0,
                            "loss_before_random": 0.0,
                            "loss_after_random": 0.0,
                            "improvement_own": 0.0,
                            "improvement_random": 0.0,
                        }

                # Gather evaluation results from all ranks
                list_of_local_results_dicts = tplr.distrib.all_gather_object(
                    local_evaluation_results
                )

                # Rank 0 aggregates all results and updates global scores
                if tplr.distrib.is_rank0():
                    aggregated_results = {}
                    for results_dict in list_of_local_results_dicts:
                        aggregated_results.update(results_dict)

                    # Initialize current_window_scores if needed
                    if not hasattr(self, "current_window_scores"):
                        self.current_window_scores = {}

                    # Update global scores and populate current_window_scores
                    for uid, results in aggregated_results.items():
                        if results["success"]:
                            self.gradient_scores[uid] = results["gradient_score"]
                            self.binary_indicator_scores[uid] = results[
                                "binary_indicator_score"
                            ]
                            self.sync_scores[uid] = results["sync_score"]

                            # Populate current window scores for OpenSkill
                            self.current_window_scores[uid] = results["gradient_score"]

                            # Initialize OpenSkill rating if not exists
                            if uid not in self.openskill_ratings:
                                self.openskill_ratings[uid] = (
                                    self.openskill_model.rating(name=str(uid))
                                )

                            # Log successful evaluation metrics
                            self.wandb.log(
                                {
                                    f"validator/miner_{uid}/gradient_score": results[
                                        "gradient_score"
                                    ],
                                    f"validator/miner_{uid}/binary_indicator_score": results[
                                        "binary_indicator_score"
                                    ],
                                    f"validator/miner_{uid}/sync_score": results[
                                        "sync_score"
                                    ],
                                    f"validator/miner_{uid}/loss_before_own": results.get(
                                        "loss_before_own", 0.0
                                    ),
                                    f"validator/miner_{uid}/loss_after_own": results.get(
                                        "loss_after_own", 0.0
                                    ),
                                    f"validator/miner_{uid}/improvement_own": results.get(
                                        "improvement_own", 0.0
                                    ),
                                },
                                step=self.global_step,
                            )

                        else:
                            # Handle failed evaluation - reset positive scores to zero
                            old_score = self.final_scores[uid].item()
                            if old_score > 0:
                                self.final_scores[uid] = 0.0
                                tplr.log_with_context(
                                    level="warning",
                                    message=f"Reset score of UID {uid} from {old_score:.4f} to 0.0 due to evaluation failure: {results.get('error', 'unknown')}",
                                    sync_window=self.sync_window,
                                    current_window=self.current_window,
                                )

                            # Log slashing metrics
                            self.wandb.log(
                                {
                                    f"validator/slash/{uid}/score_before": old_score,
                                    f"validator/slash/{uid}/score_after": 0.0,
                                    f"validator/slash/{uid}/reason": results.get(
                                        "error", "evaluation_failed"
                                    ),
                                },
                                step=self.global_step,
                            )

                            self.metrics_logger.log(
                                measurement="validator_slash",
                                tags={
                                    "eval_uid": str(uid),
                                    "window": int(self.sync_window),
                                    "global_step": int(self.global_step),
                                    "reason_code": "evaluation_failed",
                                },
                                fields={
                                    "score_before": float(old_score),
                                    "score_after": 0.0,
                                    "reason": str(
                                        results.get("error", "evaluation_failed")
                                    )[:255],
                                },
                                with_system_metrics=True,
                                with_gpu_metrics=True,
                            )

                        # Add to evaluated UIDs for consistent tracking
                        self.evaluated_uids.add(uid)

                # Apply gathered gradients to validator's main model - only rank 0
                model_was_updated = False
                if tplr.distrib.is_rank0():
                    if aggregation_result is not None:
                        self.apply_aggregated_gradients(aggregation_result)
                        model_was_updated = True
                        tplr.log_with_context(
                            level="info",
                            message="Applied aggregated gradients to validator model",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                        )
                    elif gather_result is not None:
                        self.apply_gathered_gradients(gather_result)
                        model_was_updated = True
                        tplr.log_with_context(
                            level="info",
                            message="Applied gathered gradients to validator model",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                        )

                    if model_was_updated:
                        self.scheduler.step()
                        # Prepare updated model state for broadcasting
                        updated_model_state_dict = {
                            k: v.cpu() for k, v in self.model.state_dict().items()
                        }
                    else:
                        updated_model_state_dict = None

                # Broadcast model updates to all ranks
                broadcast_data = tplr.distrib.broadcast_object(
                    (updated_model_state_dict, model_was_updated)
                    if tplr.distrib.is_rank0()
                    else (None, False),
                    src=0,
                )
                new_model_state_to_load, model_updated_on_rank0 = broadcast_data

                # All ranks update their model if rank 0 updated
                if model_updated_on_rank0 and not tplr.distrib.is_rank0():
                    self.model.load_state_dict(new_model_state_to_load)
                    self.scheduler.step()  # Keep scheduler in sync
                    tplr.log_with_context(
                        level="info",
                        message="Updated local model from rank 0 broadcast",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )

                # Barrier to ensure all models are synced before continuing
                tplr.distrib.barrier()

                # Update scores and weights - only rank 0
                if tplr.distrib.is_rank0():
                    # Update OpenSkill ratings
                    self.update_openskill_ratings()

                    # Update final scores and weights
                    self.update_weights()

                    # Set weights on subtensor
                    try:
                        result, message = self.subtensor.set_weights(
                            wallet=self.wallet,
                            netuid=self.config.netuid,
                            uids=self.metagraph.uids,
                            weights=self.weights,
                            wait_for_finalization=False,
                            wait_for_inclusion=True,
                            max_retries=3,
                        )
                        if result:
                            tplr.log_with_context(
                                level="info",
                                message=f"Successfully set weights on chain: {message}",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                            )
                        else:
                            tplr.log_with_context(
                                level="error",
                                message=f"Failed to set weights on chain: {message}",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                            )
                    except Exception as e:
                        tplr.log_with_context(
                            level="error",
                            message=f"Exception setting weights: {e}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                        )

                    # Log final scores table
                    self.log_final_scores_table()

                    # Store debug information - fix any tensor.items() calls
                    debug_info = {
                        "sync_window": self.sync_window,
                        "global_step": self.global_step,
                        # FIXED: Convert tensors to dictionaries properly
                        "gradient_scores": {
                            uid: float(self.gradient_scores[uid].item()) 
                            for uid in range(len(self.gradient_scores)) 
                            if uid in self.evaluated_uids
                        },
                        "binary_indicator_scores": {
                            uid: float(self.binary_indicator_scores[uid].item()) 
                            for uid in range(len(self.binary_indicator_scores)) 
                            if uid in self.evaluated_uids
                        },
                        "sync_scores": {
                            uid: float(self.sync_scores[uid].item()) 
                            for uid in range(len(self.sync_scores)) 
                            if uid in self.evaluated_uids
                        },
                        "final_scores": {
                            uid: float(self.final_scores[uid].item()) 
                            for uid in range(len(self.final_scores)) 
                            if uid in self.evaluated_uids or self.final_scores[uid] > 0
                        },
                        "weights": {
                            uid: float(self.weights[uid].item()) 
                            for uid in range(len(self.weights)) 
                            if uid in self.evaluated_uids or self.weights[uid] > 0
                        },
                        "skipped_uids": skipped_uids,
                        "success_rate": success_rate,
                    }

                    await self.comms.put(
                        key="debug",
                        data=debug_info,
                        window=self.sync_window,
                        uid=str(self.uid),
                    )

                    # Store checkpoint
                    checkpoint_data = {
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.scheduler.state_dict(),
                        "momentum": self.momentum,
                        "checkpoint_window": self.sync_window,
                    }

                    await self.comms.put(
                        key="checkpoint",
                        data=checkpoint_data,
                        window=self.sync_window,
                        uid=str(self.uid),
                    )

                # Increment global step and log window completion
                self.global_step += 1

                if tplr.distrib.is_rank0():
                    window_time = tplr.T() - window_start
                    tplr.log_with_context(
                        level="info",
                        message=f"Completed window {self.sync_window} in {window_time:.2f}s",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )

                    # Log window timing metrics
                    self.wandb.log(
                        {
                            "validator/window_time": window_time,
                            "validator/global_step": self.global_step,
                            "validator/sync_window": self.sync_window,
                        },
                        step=self.global_step,
                    )

                # Barrier to ensure all ranks complete the window together
                tplr.distrib.barrier()

                # Add summary logging after result aggregation:
                if tplr.distrib.is_rank0():
                    # ... existing aggregation code ...

                    # Add distributed evaluation summary
                    successful_evaluations = sum(
                        1
                        for results in aggregated_results.values()
                        if results["success"]
                    )
                    failed_evaluations = (
                        len(aggregated_results) - successful_evaluations
                    )

                    # Log per-rank breakdown
                    rank_breakdown = {}
                    for rank_id, results_dict in enumerate(list_of_local_results_dicts):
                        rank_breakdown[rank_id] = {
                            "total": len(results_dict),
                            "successful": sum(
                                1 for r in results_dict.values() if r["success"]
                            ),
                            "failed": sum(
                                1 for r in results_dict.values() if not r["success"]
                            ),
                        }

                    # Create summary message
                    breakdown_str = ", ".join(
                        [
                            f"Rank {rank}: {stats['successful']}/{stats['total']} success"
                            for rank, stats in rank_breakdown.items()
                        ]
                    )

                    tplr.log_with_context(
                        level="info",
                        message=f"Distributed evaluation summary: {successful_evaluations}/{len(aggregated_results)} total successful. Per-rank: {breakdown_str}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )

                    # Log to WandB for monitoring
                    self.wandb.log(
                        {
                            "validator/distributed/total_evaluations": len(
                                aggregated_results
                            ),
                            "validator/distributed/successful_evaluations": successful_evaluations,
                            "validator/distributed/failed_evaluations": failed_evaluations,
                            "validator/distributed/success_rate": successful_evaluations
                            / len(aggregated_results)
                            if len(aggregated_results) > 0
                            else 0.0,
                        },
                        step=self.global_step,
                    )

                # Add logging for the distributed evaluation summary on all ranks
                # After gathering results from all ranks, log summary on each rank
                if not tplr.distrib.is_rank0():
                    # Non-rank 0 processes log their local results
                    successful_local = sum(1 for r in local_evaluation_results.values() if r["success"])
                    failed_local = len(local_evaluation_results) - successful_local
                    
                    tplr.log_with_context(
                        level="info",
                        message=f"[Rank {self.rank}] Local evaluation results: {successful_local}/{len(local_evaluation_results)} successful, {failed_local} failed",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                    
                    # Log individual results
                    for uid, result in local_evaluation_results.items():
                        if result["success"]:
                            tplr.log_with_context(
                                level="info",
                                message=f"[Rank {self.rank}] UID {uid} final scores: gradient={result['gradient_score']:.6f}, binary={result['binary_indicator_score']}, sync={result['sync_score']:.6f}",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                                eval_uid=uid,
                            )
                        else:
                            tplr.log_with_context(
                                level="warning",
                                message=f"[Rank {self.rank}] UID {uid} evaluation failed: {result.get('error', 'unknown error')}",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                                eval_uid=uid,
                            )

            except Exception as e:
                if tplr.distrib.is_rank0():
                    tplr.log_with_context(
                        level="error",
                        message=f"Error in main validator loop: {str(e)}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )

                # Broadcast error to all ranks to keep them synchronized
                error_occurred = True
                error_occurred = tplr.distrib.broadcast_object(
                    error_occurred if tplr.distrib.is_rank0() else None, src=0
                )

                # All ranks sleep and continue together
                await asyncio.sleep(30)
                continue

    def select_initial_peers(self) -> list[int] | None:
        """
        Simple initial peer selection based on incentive.
        1) Select peers with highest incentive
        2) If needed, fill remaining slots with active peers
        3) Ensure we have minimum number of peers
        """
        try:
            tplr.log_with_context(
                level="info",
                message="Starting selection of initial gather peers",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            # 1. Create a dictionary of active peers with non-zero incentive
            uid_to_incentive = {}
            for uid, incentive in zip(
                self.metagraph.uids.tolist(), self.metagraph.I.tolist()
            ):
                if incentive > 0 and uid in self.comms.active_peers:
                    uid_to_incentive[uid] = float(incentive)

            # Sort by incentive (highest first) and take top peers
            top_incentive_peers = sorted(
                uid_to_incentive.keys(),
                key=lambda uid: uid_to_incentive[uid],
                reverse=True,
            )[: self.hparams.max_topk_peers]

            # If we have enough peers with incentive, return them
            if len(top_incentive_peers) == self.hparams.max_topk_peers:
                tplr.log_with_context(
                    level="info",
                    message=f"Selected {len(top_incentive_peers)} initial peers purely based on incentive",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                return top_incentive_peers

            # 2. If needed, fill up with active peers that don't have incentive
            remaining_active_peers = [
                int(peer)
                for peer in self.comms.active_peers
                if peer not in top_incentive_peers
            ]

            # Calculate how many more peers we need
            needed_peers = self.hparams.max_topk_peers - len(top_incentive_peers)

            # Randomly select from remaining active peers
            additional_peers = random.sample(
                remaining_active_peers, min(len(remaining_active_peers), needed_peers)
            )

            # Combine the lists
            selected_peers = top_incentive_peers + additional_peers

            # Ensure we have enough peers
            if len(selected_peers) >= self.hparams.minimum_peers:
                tplr.log_with_context(
                    level="info",
                    message=f"Selected {len(selected_peers)} initial peers: "
                    f"{len(top_incentive_peers)} with incentive and "
                    f"{len(additional_peers)} without incentive",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                # No need to remove duplicates as we ensured lists don't overlap
                return selected_peers

            # 3. If we don't have enough peers, give up
            tplr.log_with_context(
                level="info",
                message=f"Failed to select at least {self.hparams.minimum_peers} initial gather "
                f"peers. Found only {len(selected_peers)} active peers, of which "
                f"{len(top_incentive_peers)} had incentive and "
                f"{len(additional_peers)} were incentiveless active peers.",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            return None

        except Exception as e:
            tplr.log_with_context(
                level="error",
                message=f"Failed to create new peer list: {e}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            return None

    def select_next_peers(self) -> list[int] | None:
        """
        Simple peer selection that prioritizes the highest weights.
        1) Get all active peers
        2) Sort them by weight (highest first)
        3) Select up to max_topk_peers
        4) If not enough high-weight peers, fill remaining with random active peers
        """
        # Get all active peers as a list
        active_peers = [int(peer) for peer in self.comms.active_peers]

        # Check if we have enough active peers
        if len(active_peers) < self.hparams.minimum_peers:
            tplr.log_with_context(
                level="info",
                message=f"Not enough active peers ({len(active_peers)}) to meet minimum requirement ({self.hparams.minimum_peers})",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            return None

        # Create list of (peer_id, weight) tuples
        peer_weights = []
        for peer_id in active_peers:
            weight = float(self.weights.cpu()[peer_id])
            peer_weights.append((peer_id, weight))

        # Sort by weight (highest first)
        peer_weights.sort(key=lambda x: x[1], reverse=True)

        # Take peers with highest weights first
        highest_weight_count = min(len(peer_weights), self.hparams.max_topk_peers)
        selected_peers = [peer_id for peer_id, _ in peer_weights[:highest_weight_count]]

        tplr.log_with_context(
            level="info",
            message=f"Selected {len(selected_peers)} peers based on highest weights",
            sync_window=self.sync_window,
            current_window=self.current_window,
        )

        return selected_peers

    async def evaluate_miner_sync(self, eval_uid, model_to_compare_against=None):
        """
        Evaluate miner synchronization by comparing model states.

        Args:
            eval_uid: UID of the miner to evaluate
            model_to_compare_against: Model to use for comparison (defaults to self.model)
        """
        if model_to_compare_against is None:
            model_to_compare_against = self.model

        try:
            # Get miner's debug information
            debug_result = await self.comms.get(
                uid=str(eval_uid),
                window=self.sync_window,
                key="debug",
                local=False,
                stale_retention=10,
            )

            if debug_result is None:
                return {"sync_score": 0.0, "reason": "no_debug_data"}

            debug_dict = debug_result[0]
            if debug_dict is None:
                return {"sync_score": 0.0, "reason": "empty_debug_data"}

            # Compare model with debug dict using the provided model
            sync_result = tplr.neurons.compare_model_with_debug_dict(
                model=model_to_compare_against,
                debug_dict=debug_dict,
                tolerance=self.hparams.sync_tolerance,
            )

            return {
                "sync_score": sync_result.get("sync_score", 0.0),
                "reason": sync_result.get("reason", "unknown"),
            }

        except Exception as e:
            tplr.log_with_context(
                level="warning",
                message=f"Error evaluating sync for UID {eval_uid}: {e}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            return {"sync_score": 0.0, "reason": f"error: {str(e)}"}

    def log_final_scores_table(self):
        """Log a table of final scores for all miners."""
        if not tplr.distrib.is_rank0():
            return  # Only rank 0 logs tables

        try:
            # Create rich table
            table = Table(title=f"Final Scores - Window {self.sync_window}")
            table.add_column("UID", style="cyan")
            table.add_column("Gradient", style="green")
            table.add_column("Binary", style="yellow")
            table.add_column("Sync", style="blue")
            table.add_column("Final", style="red")
            table.add_column("Weight", style="magenta")

            # Add rows for miners with scores
            for uid in range(len(self.final_scores)):
                if (
                    uid in self.evaluated_uids
                    or self.final_scores[uid] > 0
                    or self.weights[uid] > 0
                ):
                    # Use tensor indexing instead of .get() method
                    gradient_score = float(self.gradient_scores[uid].item()) if uid < len(self.gradient_scores) else 0.0
                    binary_score = float(self.binary_indicator_scores[uid].item()) if uid < len(self.binary_indicator_scores) else 0.0
                    sync_score = float(self.sync_scores[uid].item()) if uid < len(self.sync_scores) else 0.0
                    final_score = float(self.final_scores[uid].item())
                    weight = float(self.weights[uid].item())

                    table.add_row(
                        str(uid),
                        f"{gradient_score:.4f}",
                        f"{binary_score:.1f}",
                        f"{sync_score:.4f}",
                        f"{final_score:.4f}",
                        f"{weight:.6f}",
                    )

            # Log table using rich console
            console = Console(file=StringIO(), width=120)
            console.print(table)
            table_str = console.file.getvalue()

            tplr.log_with_context(
                level="info",
                message=f"Final Scores Table:\n{table_str}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

        except Exception as e:
            tplr.log_with_context(
                level="warning",
                message=f"Error creating scores table: {e}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

    def apply_aggregated_gradients(self, aggregation_result: dict):
        """
        Apply aggregated gradients from the aggregation server.
        Args:
            aggregation_result: Pre-loaded aggregation data from the aggregation server.
        Returns:
            bool: True if aggregation was successfully applied, False otherwise
        """
        try:
            update_start = time.time()

            state_dict = aggregation_result.get("state_dict")
            if state_dict is None:
                tplr.log_with_context(
                    level="warning",
                    message="No state_dict found in aggregation result",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                return False

            tensors_applied = 0

            for name, param in self.model.named_parameters():
                if name in state_dict:
                    packed_tensor = state_dict[name]
                    if packed_tensor is None:
                        continue

                    # Unpack binary tensor
                    unpacked_tensor = tplr.neurons.unpack_binary_tensor(
                        packed_tensor, param.shape
                    )

                    # Move to appropriate device
                    unpacked_tensor = unpacked_tensor.to(self.config.device)

                    # Set as gradient for optimizer
                    if param.grad is None:
                        param.grad = unpacked_tensor
                    else:
                        param.grad.copy_(unpacked_tensor)

                    tensors_applied += 1

            if tensors_applied > 0:
                tplr.log_with_context(
                    level="info",
                    message=f"Set gradients for {tensors_applied} tensors in {time.time() - update_start:.2f}s",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )

                # Update parameters with optimizer
                self.optimizer.step()
                self.scheduler.step()
                torch.cuda.empty_cache()

                tplr.log_with_context(
                    level="info",
                    message="Successfully applied aggregation",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                return True
            else:
                tplr.log_with_context(
                    level="warning",
                    message="No tensors were applied during aggregation",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                return False

        except Exception as e:
            tplr.log_with_context(
                level="error",
                message=f"Error applying aggregated gradients: {e}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            return False

    def apply_gathered_gradients(self, gather_result: SimpleNamespace):
        """
        Apply gathered gradients from peers to the model.

        This method:
        1. Extracts the compressed gradients from the gather result
        2. Decompresses them using the DCT transformer and compressor
        3. Stores the gradients in momentum for checkpointing
        4. Applies sign operation for SignSGD optimization
        5. Updates the model using the optimizer and scheduler

        Args:
            gather_result: The result object from a gather operation containing
                          compressed gradients from peers
        """
        for n, p in self.model.named_parameters():
            idxs_key = n + "idxs"
            vals_key = n + "vals"
            idxs = getattr(gather_result.state_dict, idxs_key, None)
            vals = getattr(gather_result.state_dict, vals_key, None)
            if idxs is not None and vals is not None:
                if not isinstance(idxs, (list, tuple)):
                    idxs = [idxs]
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                new_grad = self.transformer.decode(
                    self.compressor.batch_decompress(
                        p.to(self.config.device),
                        idxs,
                        vals,
                        self.xshapes[n],
                        self.totalks[n],
                    )
                )
                # Store pre-sign gradient in momentum
                self.momentum[n] = new_grad.clone()
                if p.grad is None:
                    p.grad = new_grad
                else:
                    p.grad.copy_(new_grad)
                p.grad.sign_()
            else:
                tplr.log_with_context(
                    level="info",
                    message=f"Gradient data missing for parameter {n}, skipping.",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
        self.optimizer.step()
        self.scheduler.step()
        torch.cuda.empty_cache()

    # ------------- state helpers ----------------
    def _state_dict(self):
        # Only rank 0 needs to create the full state dict
        if not tplr.distrib.is_rank0():
            return None

        return {
            "global_step": self.global_step,
            "gradient_scores": self.gradient_scores.cpu(),
            "sync_scores": self.sync_scores.cpu(),
            "binary_indicator_scores": self.binary_indicator_scores.cpu(),
            "final_scores": self.final_scores.cpu(),
            "binary_moving_averages": self.binary_moving_averages.cpu(),
            "weights": self.weights.cpu(),
            # Store OpenSkill statistics per-uid so we can fully restore them.
            # ordinal is redundant (mu/sigma ⇒ ordinal) but handy for debugging.
            "openskill_ratings": {
                int(uid): {
                    "mu": float(r.mu),
                    "sigma": float(r.sigma),
                    "ordinal": float(r.ordinal()),
                }
                for uid, r in self.openskill_ratings.items()
            },
        }

    def save_state(self):
        # Only rank 0 saves state
        if not tplr.distrib.is_rank0():
            return

        try:
            tplr.log_with_context(
                level="info",
                message="Saving validator state",
            )
            torch.save(self._state_dict(), self.state_path)
        except Exception as e:
            tplr.log_with_context(
                level="warning",
                message=f"Failed to save validator state: {e}",
            )

    def load_state(self):
        # Rank 0 loads state from file, then broadcasts to other ranks
        if tplr.distrib.is_rank0():
            tplr.logger.info("Loading validator state")

            # ── stage 1: read file ─────────────────────────────────────────────
            try:
                state = torch.load(self.state_path, map_location=self.config.device)
            except FileNotFoundError:
                tplr.logger.warning(f"No validator state found at {self.state_path}")
                return
            except Exception as e:
                tplr.logger.warning(f"Failed to deserialize validator state: {e}")
                return

            # ── stage 2: selectively hydrate fields ────────────────────────────
            # NOTE: use `.get` so missing keys don't blow up wrong‑schema tests.
            self.global_step = int(
                state.get("global_step", getattr(self, "global_step", 0))
            )

            def _maybe(name):
                if name in state:
                    setattr(
                        self,
                        name,
                        state[name].float().to(self.config.device),
                    )

            for _tensor in (
                "gradient_scores",
                "sync_scores",
                "binary_indicator_scores",
                "final_scores",
                "binary_moving_averages",
                "weights",
            ):
                _maybe(_tensor)

            # ── OpenSkill ratings ──────────────────────────────────────────────
            try:
                saved_os = state.get("openskill_ratings", {})
                self.openskill_ratings = {
                    int(uid): self.openskill_model.rating(
                        mu=float(osd["mu"]), sigma=float(osd["sigma"]), name=str(uid)
                    )
                    for uid, osd in saved_os.items()
                }
                tplr.logger.info(
                    f"Restored OpenSkill ratings for {len(self.openskill_ratings)} peers"
                )
            except Exception as e:
                tplr.logger.warning(f"Failed to restore OpenSkill ratings: {e}")
        else:
            state_to_broadcast = None

            # Broadcast state from rank 0 to all ranks
            state = tplr.distrib.broadcast_object(state_to_broadcast, src=0)

            # All ranks update their local state from the broadcast
            if state is not None:
                self.global_step = state["global_step"]
                self.gradient_scores = state["gradient_scores"]
                self.binary_indicator_scores = state["binary_indicator_scores"]
                self.sync_scores = state["sync_scores"]
                self.final_scores = state["final_scores"]
                self.binary_moving_averages = state["binary_moving_averages"]
                self.weights = state["weights"]
                self.openskill_ratings = state["openskill_ratings"]

    async def preload_dataloader(self, seed: int):
        """
        Preload a dataloader using a seed value.

        Args:
            seed: Seed value for generating pages
                 Values > 255 are considered for random dataloaders

        Returns:
            Dictionary containing:
                - loader: The created dataloader
                - pages: The pages used
                - is_random: Whether this is considered a random loader (seed > 255)
        """
        # Determine if this is a random context based on seed value
        is_random = seed > 255
        context_id = "random" if is_random else f"UID: {seed}"

        try:
            # Generate pages based on seed
            tplr.log_with_context(
                level="info",
                message=f"Generating pages for {context_id}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            local_pages = await retry_call(
                tplr.r2_dataset.R2DatasetLoader.next_pages,
                offset=self.sync_window * self.hparams.pages_per_window,
                n_pages=self.hparams.pages_per_window,
                seed=seed,
                attempts=3,
                delay=1,
                context=f"pages for {context_id}",
                **{},
            )

            if local_pages is None:
                tplr.log_with_context(
                    level="error",
                    message=f"Failed to load pages for {context_id}. Cannot preload dataloader.",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                return None

            # Log which pages we're using
            page_ids = [p[1] for p in local_pages]
            tplr.log_with_context(
                level="info",
                message=f"Creating dataloader for {context_id} using pages: {page_ids}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            # Create the evaluation loader using the generated pages
            loader = await retry_call(
                tplr.r2_dataset.R2DatasetLoader.create,
                batch_size=self.hparams.batch_size,
                sequence_length=self.hparams.sequence_length,
                pages_info=local_pages,
                tokenizer=self.tokenizer,
                attempts=3,
                delay=1,
                context=f"loader for {context_id}",
                **{},
            )

            if loader is None:
                tplr.log_with_context(
                    level="error",
                    message=f"Failed to create loader for {context_id} with valid pages.",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                return None

            tplr.log_with_context(
                level="info",
                message=f"Successfully preloaded dataloader for {context_id} with pages: {page_ids}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            return {"loader": loader, "pages": local_pages, "is_random": is_random}
        except Exception as e:
            tplr.log_with_context(
                level="error",
                message=f"Error preloading data for {context_id}: {str(e)}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            return None

    def bin_evaluation_peers(self, num_bins: int) -> dict[int, list[int]]:
        """
        Bins evaluation peers based on their performance metrics.
        Peers are grouped into bins of similar performance.

        Args:
            num_bins (int): Number of bins to divide peers into

        Returns:
            dict: Dictionary mapping bin indices to lists of peer UIDs
        """
        # Get all active peers
        active_peers = list(self.eval_peers.keys())

        # If we don't have enough peers, return a single bin with all peers
        if len(active_peers) < num_bins * self.hparams.uids_per_window:
            tplr.log_with_context(
                level="info",
                message=f"Not enough active peers ({len(active_peers)}) for {num_bins} bins. Using single bin.",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            return {0: active_peers}

        # Collect performance metrics for binning
        peer_metrics = []
        for uid in active_peers:
            metric = (
                float(self.final_scores[uid].item())
                if uid < len(self.final_scores)
                else 0.0
            )
            peer_metrics.append((uid, metric))

        # Sort peers by metric (highest first)
        peer_metrics.sort(key=lambda x: x[1], reverse=True)
        sorted_peers = [uid for uid, _ in peer_metrics]

        total_peers = len(sorted_peers)
        peers_per_bin = total_peers // num_bins
        remainder = total_peers % num_bins

        # Create bins with all peers distributed
        bins = {}
        start_idx = 0

        for i in range(num_bins):
            # Add one extra peer to the first 'remainder' bins
            bin_size = peers_per_bin + (1 if i < remainder else 0)
            end_idx = start_idx + bin_size

            # Skip empty bins
            if start_idx >= len(sorted_peers):
                continue

            # Ensure we don't go beyond the array bounds
            end_idx = min(end_idx, len(sorted_peers))

            bins[i] = sorted_peers[start_idx:end_idx]
            start_idx = end_idx

        # Verify all peers are assigned
        total_assigned = sum(len(peers) for peers in bins.values())
        if total_assigned != total_peers:
            tplr.log_with_context(
                level="warning",
                message=f"Bin assignment error: {total_assigned} peers assigned out of {total_peers}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

        # Log the bins for debugging
        for bin_idx, peer_list in bins.items():
            tplr.log_with_context(
                level="info",
                message=f"Bin {bin_idx} (size {len(peer_list)}): {peer_list}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

        return bins

    def select_next_bin_for_evaluation(self, num_bins: int) -> int:
        """
        Determine which bin should be evaluated in the current window.
        We rotate through bins sequentially, keeping track of the last evaluated bin.

        Args:
            num_bins (int): Total number of bins

        Returns:
            int: The bin index to evaluate in this window
        """
        # Initialize bin rotation tracking if not present
        if not hasattr(self, "last_evaluated_bin"):
            self.last_evaluated_bin = -1

        # Rotate to the next bin
        next_bin = (self.last_evaluated_bin + 1) % num_bins

        # Update for next window
        self.last_evaluated_bin = next_bin

        tplr.log_with_context(
            level="info",
            message=f"Selected bin {next_bin} for evaluation in window {self.sync_window}",
            sync_window=self.sync_window,
            current_window=self.current_window,
        )

        return next_bin

    def select_evaluation_uids_from_bin(
        self, bins: dict[int, list[int]], bin_idx: int
    ) -> list[int]:
        """
        Selects peers for evaluation from a specific bin using weighted sampling.

        Args:
            bins (dict): Dictionary mapping bin indices to lists of peer UIDs
            bin_idx (int): The specific bin to select peers from

        Returns:
            list: Selected UIDs for evaluation
        """
        # Ensure the bin exists
        if bin_idx not in bins:
            tplr.log_with_context(
                level="warning",
                message=f"Bin {bin_idx} not found. Using bin 0 instead.",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            bin_idx = 0

        # Get peers in the selected bin
        bin_peers = bins[bin_idx]

        # Get weights for weighted sampling
        candidate_uids = list(bin_peers)
        candidate_weights = [self.eval_peers[uid] for uid in candidate_uids]

        # Determine how many peers to select (either all in the bin or uids_per_window)
        k = min(self.hparams.uids_per_window, len(candidate_uids))

        # Use weighted random sampling
        selected_uids = self.comms.weighted_random_sample_no_replacement(
            candidate_uids, candidate_weights, k
        )

        tplr.log_with_context(
            level="info",
            message=f"Selected {len(selected_uids)} evaluation UIDs from bin {bin_idx}",
            sync_window=self.sync_window,
            current_window=self.current_window,
        )

        return selected_uids

    # Listens for new blocks and sets self.current_block and self.current_window
    def block_listener(self, loop):
        import websockets.exceptions  # Ensure we catch websockets errors

        def handler(event):
            try:
                self.current_block = int(event["header"]["number"])
                new_window = int(self.current_block / self.hparams.blocks_per_window)
                if new_window != self.current_window:
                    self.current_window = new_window
                    self.comms.current_window = self.current_window
                    tplr.logger.info(
                        f"New block received. Current window updated to: {self.current_window}"
                    )
            except Exception as e:
                tplr.logger.error(f"Error processing block event: {e}")

        backoff = 1  # initial backoff in seconds
        max_backoff = 60  # maximum backoff limit

        while not self.stop_event.is_set():
            try:
                # This call subscribes to block headers and might throw keepalive errors
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(
                    handler
                )
                backoff = 1  # reset backoff if subscription exits without exception
            except websockets.exceptions.ConnectionClosedError as e:
                tplr.logger.warning(
                    f"Websocket ConnectionClosedError caught: {e}. Retrying in {backoff} seconds."
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            except Exception as e:
                tplr.logger.error(
                    f"Block subscription error: {e}. Retrying in {backoff} seconds."
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    def is_rank0(self) -> bool:
        """Returns True if this process is rank 0 in the distributed setup."""
        return tplr.distrib.get_rank() == 0

    def log_gpu_memory_usage(self, context: str):
        """Log current GPU memory usage with context."""
        if torch.cuda.is_available():  # Log from all ranks
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3  # GB

            tplr.log_with_context(
                level="debug",
                message=f"[Rank {self.rank}] GPU Memory [{context}]: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            # Only log to WandB from rank 0 to avoid duplicates
            if self.wandb and tplr.distrib.is_rank0():
                self.wandb.log(
                    {
                        f"validator/gpu_memory/rank_{self.rank}/allocated_gb": allocated,
                        f"validator/gpu_memory/rank_{self.rank}/reserved_gb": reserved,
                    },
                    step=self.global_step,
                )


def min_power_normalization(logits, power=2.0, epsilon=1e-8):
    """Normalizes logits using a minimum power normalization approach.

    This function applies power normalization to the input logits, raising them to a power
    and normalizing to create a probability distribution. If the sum is too small (below epsilon),
    returns zeros to avoid division by very small numbers.

    Args:
        logits (torch.Tensor): Input tensor to be normalized
        power (float, optional): Power to raise the logits to. Defaults to 2.0.
        epsilon (float, optional): Small value to prevent division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: Normalized probabilities
    """
    if logits.dim() == 0:
        logits = logits.unsqueeze(0)

    powered_logits = logits**power
    sum_powered = torch.sum(powered_logits)
    if sum_powered > epsilon:
        probabilities = powered_logits / sum_powered
    else:
        probabilities = torch.zeros_like(powered_logits)

    return probabilities


async def retry_call(func, *args, attempts=3, delay=1, context="", **kwargs):
    """
    Calls an async function with retries.

    Args:
        func (Callable): An async function.
        *args: Positional arguments to pass to func.
        attempts (int): Number of retries.
        delay (int): Delay between attempts in seconds.
        context (str): Context description for logging.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        The result of func(*args, **kwargs) or None if all attempts fail.
    """
    for attempt in range(attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            tplr.logger.error(
                f"Attempt {attempt + 1}/{attempts} failed for {context}: {e}"
            )
            await asyncio.sleep(delay)
    tplr.logger.error(f"Failed to complete {context} after {attempts} attempts.")
    return None


def sign_preserving_multiplication(a, b):
    return -abs(a) * abs(b) if a < 0 or b < 0 else a * b


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(Validator().run())
