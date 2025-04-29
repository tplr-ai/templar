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
from typing import cast

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
            "--store-gathers",
            action="store_true",
            help="Store gathered gradients in R2",
        )
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
        tplr.logger.debug("Starting initialization...")

        # Init config and load hparams
        self.config = Validator.config()
        self.hparams = tplr.load_hparams(use_local_run_hparams=self.config.local)

        # Init bittensor objects
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            tplr.logger.error(
                f"\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}[/bold]"
            )
            sys.exit()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        try:
            version = tplr.__version__
            tplr.logger = tplr.setup_loki_logger(
                service="validator", uid=str(self.uid), version=version
            )
            tplr.logger.info(f"Loki logging enabled for validator UID: {self.uid}")
        except Exception as e:
            tplr.logger.warning(f"Failed to initialize Loki logging: {e}")

        # Init model with hparams config
        self.model = LlamaForCausalLM(self.hparams.model_config)
        self.model.to(self.config.device)
        self.tokenizer = self.hparams.tokenizer

        # Init compression
        self.transformer = tplr.compress.TransformDCT(
            self.model, target_chunk=self.hparams.target_chunk
        )
        self.compressor = tplr.compress.CompressDCT()

        # Init optimizer and momentum
        self.optimizer = SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        self.momentum = {}
        self.xshapes = {}
        self.totalks = {}
        for n, p in self.model.named_parameters():
            self.momentum[n] = torch.zeros_like(p)
            _, _, xshape, totalk = self.compressor.compress(
                self.transformer.encode(self.momentum[n]), self.hparams.topk_compression
            )
            self.xshapes[n] = xshape
            self.totalks[n] = totalk

        # Set up scheduler setup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=250,
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

        self.openskill_model = PlackettLuce(
            beta=self.hparams.openskill_beta, tau=self.hparams.openskill_tau
        )
        self.openskill_ratings = {}  # Dictionary to store peer ratings

        self.bootstrap_version = getattr(self.hparams, "checkpoint_init_version", None)
        tplr.logger.info(
            f"[Miner] code_version={tplr.__version__} "
            f"checkpoint_init_flag={self.bootstrap_version or '<none>'}"
        )

        # Init comms
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

        self.bucket = self.comms.get_own_bucket("gradients", "read")
        self.comms.try_commit(self.wallet, self.bucket)
        # self.comms.fetch_commitments()

        # Init state params
        self.stop_event = asyncio.Event()
        self.current_block = self.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        self.start_window = self.current_window  # Record the start window
        self.global_step = 0  # Initialize global_step to zero
        self.comms.current_window = self.current_window
        self.sync_window = self.current_window

        # Init score tracking variables
        self.loss_before_per_batch_own = 0.0
        self.loss_after_per_batch_own = 0.0
        self.loss_before_per_batch_random = 0.0
        self.loss_after_per_batch_random = 0.0
        self.loss_improvement_own = 0.0
        self.loss_improvement_random = 0.0
        self.relative_improvement_own = 0.0
        self.relative_improvement_random = 0.0

        # For better looking graphs if no eval peers could be evaluated
        self.previous_avg_loss_before_own = 0.0
        self.previous_avg_loss_after_own = 0.0
        self.previous_avg_loss_before_random = 0.0
        self.previous_avg_loss_after_random = 0.0
        self.valid_score_indices = []

        # Caching
        self.state_path = f"validator-state-{tplr.__version__}.pt"
        if os.path.isfile(self.state_path):
            self.load_state()
        else:
            d = self.config.device
            self.gradient_scores = torch.zeros(256, dtype=torch.float32, device=d)
            self.sync_scores = torch.zeros(256, dtype=torch.float32, device=d)
            self.binary_indicator_scores = torch.zeros(
                256, dtype=torch.float32, device=d
            )
            self.final_scores = torch.zeros(256, dtype=torch.float32, device=d)
            self.binary_moving_averages = torch.zeros(
                256, dtype=torch.float32, device=d
            )
            self.weights = torch.zeros(256, dtype=torch.float32, device=d)
        self.evaluated_uids = set()

        # Add step tracking
        self.window_step = 0
        self.eval_count = 0

        # Initialize WandB
        self.wandb = tplr.initialize_wandb(
            run_prefix="V",
            uid=self.uid,
            config=self.config,
            group="validator",
            job_type="validation",
        )

        # Initialize metrics logger for InfluxDB
        self.metrics_logger = tplr.metrics.MetricsLogger(
            prefix="V",
            uid=self.uid,
            config=self.config,
            role="validator",
            group="validator",
            job_type="validation",
        )
        # Weighted selection counters for fair picking of eval peers
        self.eval_peers = defaultdict(lambda: 1)

        # Track inactive peer scores
        self.inactive_scores = {}  # {uid: (last_active_window, last_score)}
        self.inactivity_slash_rate = 0.25  # 25% slash per window
        self.missing_gradient_slash_rate = 0.75
        self.sync_score_slash_rate = 0.75

        # Initialize peer related attributes
        self.next_peers: tplr.comms.PeerArray | None = None
        self.peers_update_window = -1

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
        # Start background block listener
        self.loop = asyncio.get_running_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
        self.loop.set_default_executor(self.executor)

        self.listener = threading.Thread(
            target=self.block_listener, args=(self.loop,), daemon=True
        ).start()

        # Use config peers if provided
        if self.config.peers:
            self.comms.peers = self.config.peers

        self.comms.commitments = await self.comms.get_commitments()
        self.comms.update_peers_with_buckets()
        tplr.logger.info("Loaded commitments")

        # Only post start window if you are the highest stake validator
        if self.uid == self.metagraph.S.argmax().item():
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
                # No existing start window, so post new start window to R2
                await self.comms.post_start_window(self.start_window)
                tplr.logger.info(
                    f"This validator is the highest staked. Posted start_window: {self.start_window}"
                )
        else:
            tplr.logger.info(
                "This validator is not the highest staked. Waiting to fetch start_window."
            )
            self.start_window = await self.comms.get_start_window()

        if self.start_window is None:
            raise RuntimeError(
                "Could not find a valid start window. This should not be possible."
            )

        self.global_step = self.current_window - self.start_window
        tplr.logger.info(
            f"Using start_window: {self.start_window}, global_step: {self.global_step}"
        )

        checkpoint_window_buffer = 5
        has_new_checkpoint = (
            self.global_step
            >= self.hparams.checkpoint_frequency + checkpoint_window_buffer
        )
        # Proceed to load checkpoint
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
            self.momentum = loaded_momentum
            self.optimizer = loaded_optimizer
            self.scheduler = loaded_scheduler
            tplr.logger.info(
                f"Loaded checkpoint with global_step={self.global_step}, "
                f"optimizer_step={self.optimizer.state_dict()['state'].get(0, {}).get('step', 0)}, "
                f"scheduler_step={self.scheduler.last_epoch}"
            )
            # Only catch up if we're behind
            if (
                loaded_checkpoint_window < self.current_window
                and self.global_step > checkpoint_window_buffer
            ):
                tplr.logger.info(
                    f"Checkpoint is behind current window ({loaded_checkpoint_window} < {self.current_window}), starting catchup..."
                )
                await tplr.neurons.catchup_with_aggregation_server(
                    self, max(loaded_checkpoint_window, self.start_window)
                )
            else:
                tplr.logger.info("Checkpoint is up-to-date, skipping catchup.")

        else:
            tplr.logger.info("Starting from scratch")
            self.momentum = {
                n: torch.zeros_like(p) for n, p in self.model.named_parameters()
            }
            self.model.to(self.config.device)

        self.comms.start_commitment_fetcher()
        self.comms.start_background_tasks()
        time_min = None
        self.last_peer_update_window = None
        self.last_peer_post_window = None
        while True:
            # 1. Wait for the validator window offset
            while self.sync_window >= (
                self.current_window - self.hparams.validator_offset
            ):
                tplr.log_with_context(
                    level="info",
                    message=f"Waiting for validator window offset, synced: {self.sync_window}, current:{self.current_window}, offset:{self.hparams.validator_offset}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                await asyncio.sleep(12)

            # 2. Increment sync window and update peer lists
            window_start = tplr.T()
            self.sync_window += 1
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
            self.save_state()

            # Create and post peers
            initial_selection = False
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

            self.comms.update_peers_with_buckets()
            peer_start = tplr.T()
            await tplr.neurons.update_peers(
                instance=self, window=self.sync_window, peer_start=peer_start
            )

            self.eval_peers = self.comms.eval_peers
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

            newly_inactive = self.comms.inactive_peers
            current_window = self.sync_window

            # 3. Process inactive peers and apply penalties
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
                new_score = old_score  # Initialize new_score with old_score value
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

            sync_block = (self.sync_window + 1) * self.hparams.blocks_per_window
            retries = 0
            delay = 1
            max_retries = 2
            max_delay = 60
            while True:
                try:
                    response = self.subtensor.query_module(
                        "Timestamp", "Now", block=sync_block
                    )
                    ts_value = response.value / 1000  # convert ms to seconds
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
                        ts_value = (
                            time.time()
                        )  # Fallback: use current system time as timestamp
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

            # Refresh peers explicitly before starting gather to avoid missing updated active peers.
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

            gather_start = tplr.T()
            skipped_uids: list[int] = []
            success_rate = 0.0
            gather_result = None
            aggregation_result = await self.comms.load_aggregation(self.sync_window)
            if aggregation_result is None:
                gather_result = await self.comms.gather(
                    my_uid=self.uid,
                    uids=self.comms.peers,
                    window=self.sync_window,
                    key="gradient",
                    timeout=35,
                    device=self.config.device,
                    local=False,
                    totalks=self.totalks,
                    time_min=time_min,
                    time_max=time_max,
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
                skipped_uids = cast(list[int], state_dict.get("skipped_uids", []))
                success_rate = cast(float, state_dict.get("success_rate", 0.0))
            gather_time = tplr.T() - gather_start

            from_aggregator = 1 if aggregation_result is not None else 0
            tplr.log_with_context(
                level="info",
                message=f"Using gradient source: {'aggregator' if from_aggregator else 'gather'}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            self.wandb.log(
                {
                    "validator/aggregator_gradient": from_aggregator,
                },
                step=self.global_step,
            )

            tplr.log_with_context(
                level="info",
                message=f"Skipped UIDs: {skipped_uids}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            gather_sync_scores = await asyncio.gather(
                *(self.evaluate_miner_sync(uid) for uid in self.comms.peers)
            )

            for score_info, uid in zip(gather_sync_scores, self.comms.peers):
                avg_steps_behind = score_info.get("avg_steps_behind", 99.0)
                success = score_info.get("success", False)
                if not success or avg_steps_behind > self.hparams.sync_max_steps_behind:
                    tplr.log_with_context(
                        level="info",
                        message=f"Slashing {uid}: avg_steps_behind={avg_steps_behind:.2f} > max={self.hparams.sync_max_steps_behind}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                    if self.final_scores[uid] > 0:
                        self.final_scores[uid] *= self.sync_score_slash_rate
                        self.binary_moving_averages[uid] *= self.sync_score_slash_rate

            # Slash peers failing to submit gradients
            for uid in skipped_uids:
                tplr.log_with_context(
                    level="info",
                    message=f"No gradient gathered from UID {uid}. Slashing moving average score by {1 - self.missing_gradient_slash_rate:.2%}.",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                if 0 <= uid < self.final_scores.size(0):
                    old_score = self.final_scores[uid].item()

                    # Only reduce positive scores
                    if self.final_scores[uid] > 0:
                        self.final_scores[uid] *= self.missing_gradient_slash_rate
                        self.binary_moving_averages[uid] *= (
                            self.missing_gradient_slash_rate
                        )

                        new_score = self.final_scores[uid].item()
                        tplr.log_with_context(
                            level="info",
                            message=f"Reduced score of UID {uid} from {old_score:.4f} to {new_score:.4f} due to missing gradient in gather.",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                        )
                    else:
                        tplr.log_with_context(
                            level="info",
                            message=f"Skipped score of UID {uid} (current score: {old_score:.4f}) due to negative or zero value.",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                        )
                    self.evaluated_uids.add(uid)
                else:
                    tplr.log_with_context(
                        level="info",
                        message=f"UID {uid} not found in final_scores; skipping penalty.",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )

            # Add check for empty peers (evaluating all peer uids)
            if len(self.comms.eval_peers) == 0:
                tplr.log_with_context(
                    level="warning",
                    message=f"No peers available for evaluation in window {self.sync_window}. Waiting for next window.",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                self.global_step += 1
                continue

            # 5. Save original model state for evaluation
            eval_start = tplr.T()

            # 6. Select peers to evaluate
            candidate_uids = list(self.eval_peers.keys())
            candidate_weights = [self.eval_peers[uid] for uid in candidate_uids]
            k = min(self.hparams.uids_per_window, len(candidate_uids))
            evaluation_uids = self.comms.weighted_random_sample_no_replacement(
                candidate_uids, candidate_weights, k
            )

            # Reset counters for chosen peers
            for uid in evaluation_uids:
                self.eval_peers[uid] = 1

            # Increment counters for not chosen peers
            for uid in candidate_uids:
                if uid not in evaluation_uids:
                    self.eval_peers[uid] += 1
            self.comms.eval_peers = self.eval_peers

            tplr.log_with_context(
                level="info",
                message=f"Evaluating random subset of peers: {evaluation_uids}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            avg_loss_before_per_batch_own = 0.0
            avg_loss_after_per_batch_own = 0.0
            avg_loss_before_per_batch_random = 0.0
            avg_loss_after_per_batch_random = 0.0
            evaluated_peers = 0

            # Pre-load common random loader for all evaluated UIDs in this window.
            data_start_random = tplr.T()

            # Load the random loader directly
            random_seed = random.randint(
                1000, 10000000
            )  # Using high seed number for random context
            tplr.log_with_context(
                level="info",
                message=f"Loading common random dataloader with seed {random_seed}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            try:
                random_loader_data = await self.preload_dataloader(seed=random_seed)
                if random_loader_data:
                    common_loader_random = random_loader_data["loader"]
                    tplr.log_with_context(
                        level="info",
                        message=f"{tplr.P(self.sync_window, tplr.T() - data_start_random)} Loaded common random loader for evaluation.",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                else:
                    tplr.log_with_context(
                        level="error",
                        message="Random loader was None, cannot continue evaluation",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                    continue
            except Exception as e:
                tplr.log_with_context(
                    level="error",
                    message=f"Error loading random loader: {str(e)}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                continue

            # Setup for sliding window approach
            evaluation_uids_queue = list(
                evaluation_uids
            )  # Create a copy of the list to work with
            next_uid_dataloader_task = None
            next_uid = None

            # If we have at least one UID to evaluate, start loading the first one
            if evaluation_uids_queue:
                next_uid = evaluation_uids_queue.pop(0)
                tplr.log_with_context(
                    level="info",
                    message=f"Starting preload for first UID: {next_uid}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                next_uid_dataloader_task = asyncio.create_task(
                    self.preload_dataloader(seed=next_uid)
                )

            # Process each UID with sliding window loading
            while next_uid is not None:
                eval_uid = next_uid
                eval_uid_dataloader_task = next_uid_dataloader_task

                if eval_uid_dataloader_task is None:
                    tplr.log_with_context(
                        level="error",
                        message=f"Error loading data for UID {eval_uid}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )
                    continue

                # Start loading the next UID if there are more in the queue
                next_uid = None
                next_uid_dataloader_task = None
                if evaluation_uids_queue:
                    next_uid = evaluation_uids_queue.pop(0)
                    tplr.log_with_context(
                        level="info",
                        message=f"Starting preload for next UID: {next_uid}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                    next_uid_dataloader_task = asyncio.create_task(
                        self.preload_dataloader(seed=next_uid)
                    )

                tplr.log_with_context(
                    level="info",
                    message=f"Evaluating UID: {eval_uid}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                    eval_uid=eval_uid,
                )

                # Fetch gradient data for evaluation
                eval_result = await self.comms.get(
                    uid=str(eval_uid),
                    window=self.sync_window,
                    key="gradient",
                    local=False,
                    stale_retention=10,
                    time_max=time_max,
                    time_min=time_min,
                )

                scoring_start = tplr.T()

                # Wait for the current UID's data to be loaded
                data_start = tplr.T()
                try:
                    loader_data = await eval_uid_dataloader_task
                except Exception as e:
                    tplr.log_with_context(
                        level="error",
                        message=f"Error loading data for UID {eval_uid}: {str(e)}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )
                    loader_data = None

                if (
                    eval_result is not None
                    and not (
                        isinstance(eval_result, dict)
                        and eval_result.get("__status") in ["TOO_LATE", "TOO_EARLY"]
                    )
                    and eval_result[0] is not None
                    and loader_data is not None
                ):
                    state_dict, _ = eval_result

                    # Extract data from loader
                    loader_own = loader_data["loader"]
                    local_pages = loader_data["pages"]

                    # Pull miner-sent pages info from metadata
                    miner_pages = None
                    if (
                        "metadata" in state_dict
                        and "pages_info" in state_dict["metadata"]
                    ):
                        miner_pages = state_dict["metadata"]["pages_info"]
                    else:
                        tplr.log_with_context(
                            level="warning",
                            message=f"Missing pages info metadata from miner UID {eval_uid}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                    if local_pages is None or loader_own is None:
                        tplr.log_with_context(
                            level="warning",
                            message=f"Invalid loader data for UID {eval_uid}, skipping",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )
                        continue

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
                                message=f"Pages verified for UID {eval_uid}: pages match.",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                                eval_uid=eval_uid,
                            )
                    else:
                        tplr.log_with_context(
                            level="info",
                            message=f"Using local pages for UID {eval_uid} as miner metadata is missing.",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                    tplr.log_with_context(
                        level="info",
                        message=f"{tplr.P(self.sync_window, tplr.T() - data_start)} Loaded evaluation data using pages: {[p[1] for p in local_pages]}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )

                    state_dict, _ = eval_result
                    model_own_data_eval = copy.deepcopy(self.model)

                    # 9. Compute loss before applying gradient
                    self.optimizer.zero_grad()
                    model_own_data_eval.zero_grad()
                    with torch.no_grad():
                        model_own_data_eval.eval()
                        batches_own = []
                        for batch in loader_own:
                            batches_own.append(batch)

                        total_batches_own = len(batches_own)
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

                        tplr.log_with_context(
                            level="info",
                            message=f"Evaluating {sample_size_own}/{total_batches_own} batches ({self.hparams.validator_sample_rate * 100:.1f}%)",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                        loss_before_own, n_batches = self.evaluate_model_on_batches(
                            model_own_data_eval, batches_own, sampled_indices_own
                        )

                    self.loss_before_per_batch_own = (
                        loss_before_own / n_batches if n_batches > 0 else 0
                    )
                    avg_loss_before_per_batch_own += self.loss_before_per_batch_own
                    tplr.log_with_context(
                        level="debug",
                        message=f"Loss before (own data): {self.loss_before_per_batch_own}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )

                    # 9. Apply gradient and compute loss after
                    try:
                        self.optimizer.zero_grad()
                        model_own_data_eval.zero_grad()

                        # First validate all gradients before applying any
                        for n, p in model_own_data_eval.named_parameters():
                            idxs_key = n + "idxs"
                            vals_key = n + "vals"
                            idxs = state_dict.get(idxs_key, None)
                            vals = state_dict.get(vals_key, None)

                            if idxs is not None and vals is not None:
                                # Move tensors to device
                                idxs = idxs.to(self.config.device)
                                vals = vals.to(self.config.device)

                                # Validate indices are within bounds
                                if self.totalks.get(n) is None:
                                    tplr.log_with_context(
                                        level="warning",
                                        message=f"Missing totalk for parameter {n}, skipping peer {eval_uid}",
                                        sync_window=self.sync_window,
                                        current_window=self.current_window,
                                        eval_uid=eval_uid,
                                    )
                                    raise ValueError(
                                        f"Invalid gradient data from peer {eval_uid}: Missing totalk for parameter {n}"
                                    )

                                # Check compressed indices are valid
                                self.comms.check_compressed_indices(
                                    idxs_key,
                                    idxs,
                                    self.totalks[n],
                                    allowed_topk=self.hparams.topk_compression,
                                )

                                # Check for NaN or Inf values
                                if torch.isnan(vals).any() or torch.isinf(vals).any():
                                    tplr.log_with_context(
                                        level="warning",
                                        message=f"Values contain NaN or Inf for parameter {vals_key}, skipping peer {eval_uid}",
                                        sync_window=self.sync_window,
                                        current_window=self.current_window,
                                        eval_uid=eval_uid,
                                    )
                                    raise ValueError(
                                        f"Invalid gradient data from peer {eval_uid}: NaN or Inf values in {vals_key}"
                                    )

                        # If all validations pass, apply the gradients
                        for n, p in model_own_data_eval.named_parameters():
                            idxs_key = n + "idxs"
                            vals_key = n + "vals"
                            idxs = state_dict.get(idxs_key, None)
                            vals = state_dict.get(vals_key, None)

                            if idxs is not None and vals is not None:
                                idxs = idxs.to(self.config.device)
                                vals = vals.to(self.config.device)

                                grad = self.transformer.decode(
                                    self.compressor.decompress(
                                        p.to(self.config.device),
                                        idxs,
                                        vals,
                                        self.xshapes[n],
                                        self.totalks[n],
                                    )
                                ).to(self.config.device)

                                # Final safety check on the gradient itself
                                if torch.isnan(grad).any() or torch.isinf(grad).any():
                                    tplr.log_with_context(
                                        level="warning",
                                        message=f"Decompressed gradient for {n} contains NaN/Inf, skipping peer {eval_uid}",
                                        sync_window=self.sync_window,
                                        current_window=self.current_window,
                                        eval_uid=eval_uid,
                                    )
                                    raise ValueError(
                                        f"Invalid gradient from peer {eval_uid}: NaN or Inf in decompressed gradient for {n}"
                                    )

                                p.data.sub_(
                                    grad.sign(),
                                    alpha=self.scheduler.get_last_lr()[0]
                                    * self.hparams.eval_lr_factor,
                                )
                    except Exception as e:
                        old_score = self.final_scores[eval_uid].item()

                        if old_score > 0:
                            # Reset positive scores to zero explicitly
                            self.final_scores[eval_uid] = 0.0
                            tplr.log_with_context(
                                level="warning",
                                message=f"Set positive score of UID {eval_uid} from {old_score:.4f} to 0.0 - invalid gradient data",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                                eval_uid=eval_uid,
                            )
                        else:
                            # Negative score is worse than zero; keep it as-is.
                            tplr.log_with_context(
                                level="warning",
                                message=f"UID {eval_uid} had negative score {old_score:.4f}; retaining due to invalid gradient data",
                                sync_window=self.sync_window,
                                current_window=self.current_window,
                                eval_uid=eval_uid,
                            )

                        # Include in evaluated UIDs so it gets logged in metrics
                        self.evaluated_uids.add(eval_uid)

                        # Log to WandB
                        self.wandb.log(
                            {
                                f"validator/slash/{eval_uid}/score_before": old_score,
                                f"validator/slash/{eval_uid}/score_after": self.final_scores[
                                    eval_uid
                                ].item(),
                                f"validator/slash/{eval_uid}/reason": str(e),
                            },
                            step=self.global_step,
                        )

                        # Log to InfluxDB metrics with primitive types
                        self.metrics_logger.log(
                            measurement="validator_slash",
                            tags={
                                "eval_uid": str(eval_uid),
                                "window": int(self.sync_window),
                                "global_step": int(self.global_step),
                                "reason_code": "invalid_gradient",
                            },
                            fields={
                                "score_before": float(old_score),
                                "score_after": float(
                                    self.final_scores[eval_uid].item()
                                ),
                                "reason": str(e)[:255],  # Truncate long error messages
                            },
                            with_system_metrics=True,
                            with_gpu_metrics=True,
                        )

                        # Skip the rest of processing for this peer
                        continue

                    # 10. Compute loss after gradient application
                    self.optimizer.zero_grad()
                    model_own_data_eval.zero_grad()
                    loss_after_own, n_batches = self.evaluate_model_on_batches(
                        model_own_data_eval, batches_own, sampled_indices_own
                    )

                    # Clean up stored batches
                    del (
                        batches_own,
                        local_pages,
                        loader_own,
                        model_own_data_eval,
                        loader_data,
                    )
                    torch.cuda.empty_cache()

                    self.loss_after_per_batch_own = (
                        loss_after_own / n_batches if n_batches > 0 else 0
                    )
                    avg_loss_after_per_batch_own += self.loss_after_per_batch_own
                    tplr.log_with_context(
                        level="debug",
                        message=f"Loss after (own data): {self.loss_after_per_batch_own}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )

                    # 11. Calculate improvements and update scores
                    # Compute and assign the loss improvement to self
                    self.loss_improvement_own = (
                        self.loss_before_per_batch_own - self.loss_after_per_batch_own
                    )
                    tplr.log_with_context(
                        level="debug",
                        message=f"Loss improvement (own data): {self.loss_improvement_own}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )

                    self.relative_improvement_own = (
                        self.loss_improvement_own / self.loss_before_per_batch_own
                        if self.loss_before_per_batch_own > 0
                        else 0.0
                    )
                    tplr.log_with_context(
                        level="debug",
                        message=f"Relative improvement (own data): {self.relative_improvement_own:.4f}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )

                    # 7. Use common random loader for evaluation
                    model_random_data_eval = copy.deepcopy(self.model)

                    loader_random = common_loader_random

                    # 8. Compute initial loss
                    self.optimizer.zero_grad()
                    model_random_data_eval.zero_grad()
                    with torch.no_grad():
                        model_random_data_eval.eval()
                        # Sample random batches from the loader
                        batches_random = []
                        for batch in loader_random:
                            batches_random.append(batch)

                        total_batches_random = len(batches_random)
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
                        sampled_indices_random = sorted(
                            sampled_indices_random
                        )  # Sort for sequential access

                        tplr.log_with_context(
                            level="info",
                            message=f"Evaluating {sample_size_random}/{total_batches_random} batches ({self.hparams.validator_sample_rate * 100:.1f}%)",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                        loss_before_random, n_batches = self.evaluate_model_on_batches(
                            model_random_data_eval,
                            batches_random,
                            sampled_indices_random,
                        )

                    self.loss_before_per_batch_random = (
                        loss_before_random / n_batches if n_batches > 0 else 0
                    )
                    avg_loss_before_per_batch_random += (
                        self.loss_before_per_batch_random
                    )
                    tplr.log_with_context(
                        level="debug",
                        message=f"Loss before (random data): {self.loss_before_per_batch_random}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )
                    # 9. Apply gradient and compute loss after
                    try:
                        self.optimizer.zero_grad()
                        model_random_data_eval.zero_grad()

                        for n, p in model_random_data_eval.named_parameters():
                            idxs_key = n + "idxs"
                            vals_key = n + "vals"
                            idxs = state_dict.get(idxs_key, None)
                            vals = state_dict.get(vals_key, None)

                            if idxs is not None and vals is not None:
                                idxs = idxs.to(self.config.device)
                                vals = vals.to(self.config.device)

                                grad = self.transformer.decode(
                                    self.compressor.decompress(
                                        p.to(self.config.device),
                                        idxs,
                                        vals,
                                        self.xshapes[n],
                                        self.totalks[n],
                                    )
                                ).to(self.config.device)

                                p.data.sub_(
                                    grad.sign(),
                                    alpha=self.scheduler.get_last_lr()[0]
                                    * self.hparams.eval_lr_factor,
                                )
                    except Exception as e:
                        tplr.log_with_context(
                            level="error",
                            message=f"Failed to apply gradient for UID {eval_uid}: {str(e)}",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )
                        continue

                    # 10. Compute loss after gradient application for random data
                    self.optimizer.zero_grad()
                    model_random_data_eval.zero_grad()
                    loss_after_random, n_batches = self.evaluate_model_on_batches(
                        model_random_data_eval,
                        batches_random,
                        sampled_indices_random,
                    )

                    # Clean up stored batches, loader & pages
                    del (
                        batches_random,
                        model_random_data_eval,
                    )
                    torch.cuda.empty_cache()

                    self.loss_after_per_batch_random = (
                        loss_after_random / n_batches if n_batches > 0 else 0
                    )
                    avg_loss_after_per_batch_random += self.loss_after_per_batch_random
                    tplr.log_with_context(
                        level="info",
                        message=f"Loss after (random data): {self.loss_after_per_batch_random}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )

                    # 11. Calculate improvements and update scores
                    # Compute and assign the loss improvement to self
                    self.loss_improvement_random = (
                        self.loss_before_per_batch_random
                        - self.loss_after_per_batch_random
                    )
                    tplr.log_with_context(
                        level="info",
                        message=f"Loss improvement (random data): {self.loss_improvement_random}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )

                    self.relative_improvement_random = (
                        self.loss_improvement_random / self.loss_before_per_batch_random
                        if self.loss_before_per_batch_random > 0
                        else 0.0
                    )
                    tplr.log_with_context(
                        level="debug",
                        message=f"Relative improvement (random data): {self.relative_improvement_random}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )

                    # Calculate original performance score (gradient quality)
                    self.gradient_scores[eval_uid] = (
                        loss_before_random - loss_after_random
                    ) / loss_before_random
                    tplr.log_with_context(
                        level="debug",
                        message=f"Gradient Score: {self.gradient_scores[eval_uid]}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )

                    # Initialize or update OpenSkill rating for this peer
                    if eval_uid not in self.openskill_ratings:
                        self.openskill_ratings[eval_uid] = self.openskill_model.rating(
                            name=str(eval_uid)
                        )

                    # Record the gradient score for later OpenSkill updates
                    if not hasattr(self, "current_window_scores"):
                        self.current_window_scores = {}
                    self.current_window_scores[eval_uid] = self.gradient_scores[
                        eval_uid
                    ].item()

                    # Calculate binary indicator for overfitting detection
                    improvement_own = (
                        (loss_before_own - loss_after_own) / loss_before_own
                        if loss_before_own > 0
                        else 0
                    )
                    improvement_random = (
                        (loss_before_random - loss_after_random) / loss_before_random
                        if loss_before_random > 0
                        else 0
                    )
                    self.binary_indicator_scores[eval_uid] = (
                        1 if improvement_own > improvement_random else -1
                    )
                    tplr.log_with_context(
                        level="info",
                        message=f"Binary Indicator Score : {self.binary_indicator_scores[eval_uid]}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )

                    # Update binary moving average using exponential moving average formula:
                    # new_avg = (1-alpha) * old_avg + alpha * new_value
                    # where alpha is binary_score_ma_alpha hyperparameter
                    self.binary_moving_averages[eval_uid] = (
                        (1 - self.hparams.binary_score_ma_alpha)
                        * self.binary_moving_averages[eval_uid]
                        + self.hparams.binary_score_ma_alpha
                        * self.binary_indicator_scores[eval_uid]
                    )
                    tplr.log_with_context(
                        level="debug",
                        message=f"Binary Moving Average Score : {self.binary_moving_averages[eval_uid]}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )

                    sync_result = await self.evaluate_miner_sync(eval_uid)
                    sync_score = cast(
                        float,
                        sync_result.get("sync_score", 0.0),
                    )
                    self.log_sync_score(eval_uid, sync_result)

                    # Store the sync score for this miner
                    self.sync_scores[eval_uid] = sync_score

                    self.evaluated_uids.add(eval_uid)

                    evaluated_peers += 1
                    tplr.log_with_context(
                        level="info",
                        message=f"{tplr.P(self.sync_window, tplr.T() - eval_start)} Completed evaluation",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )

                else:
                    tplr.log_with_context(
                        level="info",
                        message=f"No gradient received from UID {eval_uid}. Slashing moving average score by {1 - self.missing_gradient_slash_rate:.2%}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                        eval_uid=eval_uid,
                    )
                    old_score = self.final_scores[eval_uid].item()

                    if self.final_scores[eval_uid] > 0:
                        self.final_scores[eval_uid] *= self.missing_gradient_slash_rate
                        self.binary_moving_averages[eval_uid] *= (
                            self.missing_gradient_slash_rate
                        )

                        new_score = self.final_scores[eval_uid].item()
                        tplr.log_with_context(
                            level="info",
                            message=f"Reduced score of UID {eval_uid} from {old_score:.4f} to {new_score:.4f} due to missing gradient.",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )
                    else:
                        tplr.log_with_context(
                            level="info",
                            message=f"Skipped reducing score of UID {eval_uid} (current score: {old_score:.4f}) due to negative or zero value.",
                            sync_window=self.sync_window,
                            current_window=self.current_window,
                            eval_uid=eval_uid,
                        )

                    # Ensure the UID is included in evaluated_uids
                    self.evaluated_uids.add(eval_uid)

                    # Log updated scores
                    tplr.log_with_context(
                        level="info",
                        message="Updated scores for evaluated UIDs after slashing:",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                    # Log evaluated UID scores (fixed join call)
                    line = " | ".join(
                        f"UID {uid}: {self.final_scores[uid]:.4f}"
                        for uid in sorted(self.evaluated_uids)
                    )
                    tplr.logger.info(line)

                    # Optionally, log to WandB
                    self.wandb.log(
                        {
                            f"validator/final_scores/{eval_uid}": self.final_scores[
                                eval_uid
                            ].item(),
                            f"validator/weights/{eval_uid}": self.weights[
                                eval_uid
                            ].item(),
                        },
                        step=self.global_step,
                    )
                    tplr.log_with_context(
                        level="info",
                        message=f"{tplr.P(self.sync_window, tplr.T() - scoring_start)} Computed scores and weights",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )

                tplr.log_with_context(
                    level="info",
                    message=f"{tplr.P(self.sync_window, tplr.T() - eval_start)} Completed evaluation",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )

            # Cancel any remaining preload task if exiting the loop early
            if (
                next_uid_dataloader_task is not None
                and not next_uid_dataloader_task.done()
            ):
                next_uid_dataloader_task.cancel()
                try:
                    await next_uid_dataloader_task
                except asyncio.CancelledError:
                    pass

            # Clean up common random loader
            del common_loader_random
            torch.cuda.empty_cache()

            self.update_openskill_ratings()
            self.update_weights()
            # Log scores and metrics for evaluated UIDs as a table
            headers = [
                "UID",
                "Gradient Score",
                "Binary Indicator",
                "Binary Moving Avg",
                "Final Score",
                "Sync score",
                "Weight",
                "OpenSkill",
            ]
            table = [headers]
            for uid in sorted(self.evaluated_uids):
                openscore_info = "N/A"
                if uid in self.openskill_ratings:
                    rating = self.openskill_ratings[uid]
                    openscore_info = f"{rating.ordinal():.2f} (μ={rating.mu:.1f}, σ={rating.sigma:.1f})"
                row = [
                    str(uid),
                    f"{self.gradient_scores[uid]:.6f}",
                    f"{self.binary_indicator_scores[uid]:.4f}",
                    f"{self.binary_moving_averages[uid]:.4f}",
                    f"{self.final_scores[uid]:.4f}",
                    f"{self.sync_scores[uid]:.4f}",
                    f"{self.weights[uid]:.4f}",
                    openscore_info,
                ]
                table.append(row)

            try:
                try:
                    width = os.get_terminal_size().columns
                except Exception:
                    width = 0
                os.environ["COLUMNS"] = str(max(200, width))

                rich_table = Table(title="Updated scores for evaluated UIDs")
                for header in headers:
                    rich_table.add_column(header)
                for row in table[1:]:
                    rich_table.add_row(*row)
                sio = StringIO()
                console = Console(file=sio, width=int(os.environ["COLUMNS"]))
                console.print(rich_table)
                table_str = sio.getvalue()
            except ImportError:
                tplr.log_with_context(
                    level="warning",
                    message="rich module not found; falling back to basic formatting.",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                col_widths = [
                    max(len(row[i]) for row in table) for i in range(len(headers))
                ]
                lines = []
                for i, row in enumerate(table):
                    line = " | ".join(
                        row[j].ljust(col_widths[j]) for j in range(len(row))
                    )
                    lines.append(line)
                    if i == 0:
                        separator = "-+-".join(
                            "-" * col_widths[j] for j in range(len(headers))
                        )
                        lines.append(separator)
                table_str = "\n".join(lines)

            tplr.log_with_context(
                level="info",
                message="Updated scores for evaluated UIDs:\n" + table_str,
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            should_downsample = len(self.evaluated_uids) > 8
            for uid in sorted(self.evaluated_uids):
                # Extract primitive values from tensors for WandB
                gradient_score = float(self.gradient_scores[uid].item())
                binary_indicator = float(self.binary_indicator_scores[uid].item())
                binary_moving_avg = float(self.binary_moving_averages[uid].item())
                sync_score = float(self.sync_scores[uid].item())
                final_score = float(self.final_scores[uid].item())
                weight = float(self.weights[uid].item())

                self.wandb.log(
                    {
                        f"validator/gradient_scores/{uid}": gradient_score,
                        f"validator/binary_indicators/{uid}": binary_indicator,
                        f"validator/binary_moving_averages/{uid}": binary_moving_avg,
                        f"validator/final_scores/{uid}": final_score,
                        f"validator/sync_score/{uid}": sync_score,
                        f"validator/weights/{uid}": weight,
                    },
                    step=self.global_step,
                )

                self.metrics_logger.log(
                    measurement="validator_scores",
                    tags={
                        "eval_uid": str(uid),
                        "window": int(self.sync_window),
                        "global_step": int(self.global_step),
                    },
                    fields={
                        "gradient_score": gradient_score,
                        "binary_indicator": binary_indicator,
                        "binary_moving_avg": binary_moving_avg,
                        "sync_score": sync_score,
                        "final_score": final_score,
                        "weight": weight,
                    },
                    with_system_metrics=True,
                    with_gpu_metrics=True,
                    sample_rate=0.8 if should_downsample else 1.0,
                )

            # 17. Set weights periodically

            if self.sync_window % self.hparams.windows_per_weights == 0:
                # Only set weights for evaluated peers with non-negative (positive) weight values.
                positive_weighted_uids = sorted(
                    [uid for uid in self.evaluated_uids if self.weights[uid] > 0]
                )
                if positive_weighted_uids:
                    self.subtensor.set_weights(
                        wallet=self.wallet,
                        netuid=self.config.netuid,
                        uids=positive_weighted_uids,
                        weights=self.weights[positive_weighted_uids],
                        wait_for_inclusion=False,
                        wait_for_finalization=False,
                    )

            # 14. Now, merge the gathered gradients into the model AFTER finishing evaluation
            self.model.train()
            update_start = tplr.T()
            self.optimizer.zero_grad()
            self.model.zero_grad()
            lr = self.scheduler.get_last_lr()[0]
            # Apply weight decay just like in the miner
            for n, p in self.model.named_parameters():
                p.data.mul_(1.0 - lr * self.hparams.weight_decay)

            if aggregation_result is not None:
                self.apply_aggregated_gradients(aggregation_result=aggregation_result)
            elif gather_result is not None and gather_result.state_dict is not None:
                self.apply_gathered_gradients(gather_result=gather_result)
            else:
                tplr.log_with_context(
                    level="warning",
                    message="No gradients to apply.",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                self.scheduler.step()
                torch.cuda.empty_cache()

            tplr.log_with_context(
                level="info",
                message=f"{tplr.P(self.sync_window, tplr.T() - update_start)} Updated model",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            # Add debug data including successfully gathered peers
            debug_dict = {}

            # Add model parameters debug info
            for name, param in self.model.named_parameters():
                if (
                    param is not None and param.numel() >= 2
                ):  # Check if tensor has at least 2 elements
                    debug_dict[name + "_debug"] = (
                        param.flatten()[:2].detach().cpu().tolist()
                    )

            # Add successful peers information
            if len(skipped_uids) > 0:
                debug_dict["successful_peers"] = sorted(
                    list(set(self.comms.peers) - set(skipped_uids))
                )
                debug_dict["skipped_peers"] = sorted(list(skipped_uids))

            # 15. Store debug values and model metadata
            asyncio.create_task(
                self.comms.put(
                    state_dict=debug_dict,
                    uid=str(self.uid),
                    window=self.sync_window,
                    key="debug",
                    local=False,
                )
            )
            tplr.log_with_context(
                level="info",
                message=f"Stored debug values for window {self.current_window}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            # Log total window time and metrics
            tplr.log_with_context(
                level="info",
                message=f"{tplr.P(self.sync_window, tplr.T() - window_start)} Completed window iteration",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            if evaluated_peers == 0:
                # Use the values from the previous step
                avg_loss_before_per_batch_own = self.previous_avg_loss_before_own
                avg_loss_after_per_batch_own = self.previous_avg_loss_after_own
                avg_loss_before_per_batch_random = self.previous_avg_loss_before_random
                avg_loss_after_per_batch_random = self.previous_avg_loss_after_random
            else:
                # Calculate averages normally
                avg_loss_before_per_batch_own /= evaluated_peers
                avg_loss_after_per_batch_own /= evaluated_peers
                avg_loss_before_per_batch_random /= evaluated_peers
                avg_loss_after_per_batch_random /= evaluated_peers

                # Store current values for future use when evaluated_peers might be 0
                self.previous_avg_loss_before_own = avg_loss_before_per_batch_own
                self.previous_avg_loss_after_own = avg_loss_after_per_batch_own
                self.previous_avg_loss_before_random = avg_loss_before_per_batch_random
                self.previous_avg_loss_after_random = avg_loss_after_per_batch_random

            # 16. Log evaluation metrics once all evaluations are done
            evaluation_metrics = {
                "validator/loss/own/before": avg_loss_before_per_batch_own,
                "validator/loss/own/after": avg_loss_after_per_batch_own,
                "validator/loss/random/before": avg_loss_before_per_batch_random,
                "validator/loss/random/after": avg_loss_after_per_batch_random,
                "validator/loss/own/improvement": self.relative_improvement_own,
                "validator/loss/random/improvement": self.relative_improvement_random,
                "validator/network/block": self.current_block,
                "validator/network/window": self.sync_window,
                "validator/network/step": self.global_step,
                "validator/network/evaluated_uids": len(self.evaluated_uids),
                "validator/optimizer/learning_rate": self.scheduler.get_last_lr()[0],
                "validator/network/active_miners": len(self.valid_score_indices),
                "validator/gather/success_rate": success_rate * 100,
                "validator/timing/window_total": tplr.T() - window_start,
                "validator/timing/peer_update": tplr.T() - peer_start,
                "validator/timing/gather": gather_time,
                "validator/timing/evaluation": tplr.T() - eval_start,
                "validator/timing/model_update": tplr.T() - update_start,
            }
            self.wandb.log(evaluation_metrics, step=self.global_step)

            # Log metrics to InfluxDB in parallel using primitive types
            gather_success_rate = float(success_rate * 100)
            total_skipped = len(skipped_uids)

            self.metrics_logger.log(
                measurement="validator_window_v2",
                tags={
                    "window": int(self.sync_window),
                    "global_step": int(self.global_step),
                },
                fields={
                    "loss_own_before": float(avg_loss_before_per_batch_own),
                    "loss_own_after": float(avg_loss_after_per_batch_own),
                    "loss_random_before": float(avg_loss_before_per_batch_random),
                    "loss_random_after": float(avg_loss_after_per_batch_random),
                    "loss_own_improvement": float(self.relative_improvement_own),
                    "loss_random_improvement": float(self.relative_improvement_random),
                    "current_block": int(self.current_block),
                    "evaluated_uids_count": int(len(self.evaluated_uids)),
                    "learning_rate": float(self.scheduler.get_last_lr()[0]),
                    "active_miners_count": int(len(self.valid_score_indices)),
                    "gather_success_rate": gather_success_rate,
                    "window_total_time": float(tplr.T() - window_start),
                    "peer_update_time": float(tplr.T() - peer_start),
                    "gather_time": float(gather_time),
                    "evaluation_time": float(tplr.T() - eval_start),
                    "model_update_time": float(tplr.T() - update_start),
                    "total_peers": int(len(self.comms.peers)),
                    "total_skipped": int(total_skipped),
                },
                with_system_metrics=True,
                with_gpu_metrics=True,
            )
            tplr.log_with_context(
                level="info",
                message="Finished metrics logging call for validator",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            # 17. Create checkpoints periodically
            if (
                self.global_step % self.hparams.checkpoint_frequency == 0
                and self.global_step != 0
            ):
                tplr.log_with_context(
                    level="info",
                    message=f"Creating checkpoint at global_step {self.global_step}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                checkpoint_data = {
                    "model_state_dict": {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    },
                    "optimizer_state_dict": {
                        k: v.cpu().clone() if torch.is_tensor(v) else v
                        for k, v in self.optimizer.state_dict().items()
                    },
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "momentum": {
                        n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                    },
                    "start_window": self.start_window,
                    "current_window": self.current_window,
                    "sync_window": self.sync_window,
                }
                asyncio.create_task(
                    self.comms.put(
                        state_dict=checkpoint_data,
                        uid=str(self.uid),
                        window=self.sync_window,
                        key="checkpoint",
                        global_step=self.global_step,
                        local=False,
                    )
                )

            # 18. Increment global step
            self.global_step += 1

            torch.cuda.empty_cache()

    def select_initial_peers(self) -> tplr.comms.PeerArray | None:
        try:
            tplr.log_with_context(
                level="info",
                message="Starting selection of initial gather peers",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            # 1. Add top incentive peers
            uid_to_non_zero_incentive = {
                uid: incentive
                for uid, incentive in zip(self.metagraph.uids, self.metagraph.I)
                if incentive > 0 and uid in self.comms.active_peers
            }
            top_incentive_peers = sorted(
                uid_to_non_zero_incentive,
                key=uid_to_non_zero_incentive.get,
                reverse=True,
            )[: self.hparams.max_topk_peers]

            # Convert to list to ensure it's not a dict_keys object
            top_incentive_peers = np.array(top_incentive_peers, dtype=np.int64)

            assert len(top_incentive_peers) <= self.hparams.max_topk_peers
            if len(top_incentive_peers) == self.hparams.max_topk_peers:
                tplr.log_with_context(
                    level="info",
                    message=f"Selected {len(top_incentive_peers)} initial peers purely based on incentive: {top_incentive_peers}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                selected_peers = np.array(top_incentive_peers, dtype=np.int64)
                selected_peers = np.unique(selected_peers)
                return selected_peers

            # 2. If needed, fill up with active peers
            remaining_active_peers = np.array(
                list(set(self.comms.active_peers) - set(top_incentive_peers))
            )
            top_incentive_and_active_peers = np.concatenate(
                [top_incentive_peers, remaining_active_peers]
            )[: self.hparams.max_topk_peers]

            assert len(top_incentive_and_active_peers) <= self.hparams.max_topk_peers
            if len(top_incentive_and_active_peers) >= self.hparams.minimum_peers:
                tplr.log_with_context(
                    level="info",
                    message=f"Selected {len(top_incentive_and_active_peers)} initial peers. "
                    f"{len(top_incentive_peers)} with incentive: {top_incentive_peers} "
                    f"and {len(top_incentive_and_active_peers) - len(top_incentive_peers)} without: "
                    f"{remaining_active_peers[: len(top_incentive_and_active_peers) - len(top_incentive_peers)]}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                selected_peers = np.array(
                    top_incentive_and_active_peers, dtype=np.int64
                )
                selected_peers = np.unique(selected_peers)
                return selected_peers

            # 3. Give up
            tplr.log_with_context(
                level="info",
                message=f"Failed to select at least {self.hparams.minimum_peers} initial gather "
                f"peers. Found only {len(top_incentive_and_active_peers)} active "
                f"peers, of which {len(top_incentive_peers)} had incentive and "
                f"{len(top_incentive_and_active_peers) - len(top_incentive_peers)} "
                f"were incentiveless active peers.",
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

    @staticmethod
    def _replace_peers(
        old: tplr.comms.PeerArray,
        ingoing: tplr.comms.PeerArray,
        outgoing: tplr.comms.PeerArray,
    ) -> tplr.comms.PeerArray:
        """Removes `outgoing` and adds `ingoing` from `old`.

        `ingoing` and `outgoing` don't need to be the same length.
        """
        old_without_outgoing = np.setdiff1d(old, outgoing)
        return np.concatenate([old_without_outgoing, ingoing])

    def select_next_peers(self) -> tplr.comms.PeerArray | None:
        """
        1) Drop inactive gather peers and fill up to max_topk_peers with active
        ones. Use non-zero-weights if possible. Abort selection if there are
        less than `minimum_peers` in total. Continue if there are still
        candidates.
        2) Replace peers with zero weight with ones with non-zero weight. Both
        current gather peers and peers added in the previous step can be
        dropped. Continue if there are still candidates.
        3) Replace at least 1, at most peers_to_replace original gather peers.
        Note that both the the outgoing and the ingoing peer(s) are guaranteed
        to have non-zero weight due to the previous steps.
        """

        old_peers = self.comms.peers

        # ----------------------------------------------------------------------
        # 0. Identify candidate peers:
        #    - active,
        #    - non-zero weight,
        #    - not already a gather peer
        # ----------------------------------------------------------------------
        non_zero_weight_uids = torch.nonzero(self.weights.cpu()).flatten().numpy()
        active_non_zero_weight_uids = np.intersect1d(
            list(self.comms.active_peers),
            non_zero_weight_uids,
        )
        candidates = np.setdiff1d(active_non_zero_weight_uids, old_peers)
        num_initial_candidates = len(candidates)
        tplr.log_with_context(
            level="info",
            message=f"Starting off with {num_initial_candidates} initial candidates.",
            sync_window=self.sync_window,
            current_window=self.current_window,
        )

        # ----------------------------------------------------------------------
        # 1. Drop inactive gather peers, fill up with active peers
        # ----------------------------------------------------------------------
        active_gather_peers = np.intersect1d(old_peers, list(self.comms.active_peers))
        inactive_gather_peers = np.setdiff1d(old_peers, active_gather_peers)
        selected_peers = old_peers
        if (
            len(inactive_gather_peers) == 0
            and len(selected_peers) == self.hparams.max_topk_peers
        ):
            if len(candidates) == 0:
                tplr.log_with_context(
                    level="info",
                    message="Step 1: Peer list already full of active peers but there are no "
                    "candidates, aborting selection",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                return None
            tplr.log_with_context(
                level="info",
                message="Step 1: Peer list already full of active peers, continuing",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
        else:
            # use random subset of candidates
            if len(candidates) > self.hparams.max_topk_peers - len(active_gather_peers):
                ingoing = np.random.choice(
                    candidates,
                    size=self.hparams.max_topk_peers - len(active_gather_peers),
                    replace=False,
                )
            # use all candidates
            elif (
                len(active_gather_peers) + len(candidates)
                >= self.hparams.max_topk_peers
            ):
                ingoing = candidates
            # use all candidates plus some secondary candidates
            # edge case: we need to use zero-weight peers to get max_topk_peers active peers
            else:
                secondary_candidates = np.setdiff1d(
                    list(self.comms.active_peers), candidates
                )
                # even bigger edge case: we still can't reach minimum peers
                if (
                    len(active_gather_peers)
                    + len(candidates)
                    + len(secondary_candidates)
                    < self.hparams.minimum_peers
                ):
                    tplr.log_with_context(
                        level="info",
                        message=f"There are only {len(self.comms.active_peers)} active peers"
                        f"in total, which is less than the minimum amount "
                        f"{self.hparams.minimum_peers}, aborting selection",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                    return None
                num_needed_secondary_candidates = (
                    self.hparams.max_topk_peers
                    - len(active_gather_peers)
                    - len(candidates)
                )
                ingoing_secondary_candidates = (
                    secondary_candidates
                    if len(secondary_candidates) < num_needed_secondary_candidates
                    else np.random.choice(
                        secondary_candidates,
                        size=num_needed_secondary_candidates,
                        replace=False,
                    )
                )
                ingoing = np.concatenate([candidates, ingoing_secondary_candidates])
                tplr.log_with_context(
                    level="info",
                    message=f"Using {len(candidates)} candidates and "
                    f"{len(ingoing_secondary_candidates)} secondary candidates",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
            selected_peers = self._replace_peers(
                old_peers, ingoing, inactive_gather_peers
            )
            # if we've used all candidates, we're done
            if len(ingoing) >= len(candidates):
                tplr.log_with_context(
                    level="info",
                    message=f"Step 1: We've used all candidates, returning {len(selected_peers)} "
                    "selected peers.",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                selected_peers = np.array(selected_peers, dtype=np.int64)
                selected_peers = np.unique(selected_peers)
                return selected_peers
            # else, update the candidates
            else:
                tplr.log_with_context(
                    level="info",
                    message=f"Finished step 1: Dropped {len(inactive_gather_peers)} inactive "
                    f"peers and added {len(ingoing)} active ones. We now have "
                    f"{len(selected_peers)} selected peers",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                candidates = np.setdiff1d(candidates, ingoing)

        # ----------------------------------------------------------------------
        # 2. Drop zero-weight gather peers
        # ----------------------------------------------------------------------
        zero_weight_selected_peers = np.setdiff1d(selected_peers, non_zero_weight_uids)
        original_peers_left = np.intersect1d(old_peers, selected_peers)
        if len(zero_weight_selected_peers) == 0:
            tplr.log_with_context(
                level="info",
                message="Step 2: No zero-weight peers to drop, continuing",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
        else:
            # if we have enough candidates to replace all zero-weight peers, do it
            if len(candidates) >= len(zero_weight_selected_peers):
                outgoing = zero_weight_selected_peers
                ingoing = (
                    candidates
                    if len(candidates) == len(outgoing)
                    else np.random.choice(
                        candidates,
                        size=len(outgoing),
                        replace=False,
                    )
                )
            # else replace as many as we have candidates
            else:
                ingoing = candidates
                outgoing = np.random.choice(
                    zero_weight_selected_peers,
                    size=len(ingoing),
                    replace=False,
                )
            selected_peers = self._replace_peers(selected_peers, ingoing, outgoing)
            candidates = np.setdiff1d(candidates, ingoing)
            original_peers_left = np.intersect1d(old_peers, selected_peers)
            if (
                # no more candidates
                len(candidates) == 0
                # all selected peers are new
                or len(original_peers_left) == 0
            ):
                tplr.log_with_context(
                    level="info",
                    message=f"Step 2: No more candidates, returning {len(selected_peers)} "
                    "selected peers.",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                selected_peers = np.array(selected_peers, dtype=np.int64)
                selected_peers = np.unique(selected_peers)
                return selected_peers
            tplr.log_with_context(
                level="info",
                message="Finished step 2 (dropped zero-weight peers) and we now have "
                f"{len(selected_peers)} selected peers",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

        # ----------------------------------------------------------------------
        # 3. Replace at least 1, at most peers_to_replace original gather peers
        # ----------------------------------------------------------------------
        outgoing = (
            original_peers_left
            if len(original_peers_left) <= len(candidates)
            and len(original_peers_left) <= self.hparams.peers_to_replace
            else np.random.choice(
                original_peers_left,
                size=min(
                    [
                        len(candidates),
                        self.hparams.peers_to_replace,
                    ]
                ),
                replace=False,
            )
        )
        ingoing = (
            candidates
            if len(candidates) == len(outgoing)
            else np.random.choice(
                candidates,
                size=len(outgoing),
                replace=False,
            )
        )
        selected_peers = self._replace_peers(selected_peers, ingoing, outgoing)
        tplr.log_with_context(
            level="info",
            message=f"Finished step 3 (replaced {len(outgoing)} peers with non-zero weight) "
            f"and we now have {len(selected_peers)} selected peers",
            sync_window=self.sync_window,
            current_window=self.current_window,
        )
        tplr.log_with_context(
            level="info",
            message=f"Step 3: Done, returning {len(selected_peers)} selected peers.",
            sync_window=self.sync_window,
            current_window=self.current_window,
        )
        selected_peers = np.array(selected_peers, dtype=np.int64)
        selected_peers = np.unique(selected_peers)
        return selected_peers

    async def evaluate_miner_sync(
        self, eval_uid: int
    ) -> dict[str, bool | float | int | str]:
        """
        Evaluates the synchronization of a specific miner with the validator's model.

        Args:
            validator: The validator instance
            eval_uid: The UID of the miner to evaluate

        Returns:
            dict: Synchronization metrics and score
        """
        # Fetch the miner's debug dictionary
        debug_result = await self.comms.get(
            uid=str(eval_uid),
            window=self.sync_window - 1,
            key="debug",
            local=False,
            stale_retention=10,
        )

        # Check if we got a valid result
        if debug_result is None:
            return {
                "success": False,
                "error": "Failed to retrieve debug dictionary",
                "sync_score": 0.0,
            }

        miner_debug_dict = cast(dict, debug_result[0])

        # Validate debug dictionary format
        if miner_debug_dict is None or not isinstance(miner_debug_dict, dict):
            return {
                "success": False,
                "error": "Invalid debug dictionary format",
                "sync_score": 0.0,
            }

        # Get current learning rate
        current_lr = self.scheduler.get_last_lr()[0]

        # Compare miner's debug dict with validator's model
        comparison_metrics = await tplr.neurons.compare_model_with_debug_dict(
            model=self.model,
            debug_dict=miner_debug_dict,
            learning_rate=current_lr,
            index_range=(10, 12),
        )

        if not comparison_metrics["success"]:
            return {
                "success": False,
                "error": "Failed to compare model with debug dictionary",
                "sync_score": 0.0,
            }

        # Calculate sync score using the formula: score = (1-x/5)^2.5
        # where x is the average steps behind, capped at 5
        avg_steps_behind = comparison_metrics["avg_steps_behind"]
        x = min(avg_steps_behind, 5.0)
        sync_score = max(0.0, (1.0 - x / 5.0) ** 2.5)

        # Add the sync score to the metrics
        result = {**comparison_metrics, "sync_score": sync_score}

        return result

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
    def _state_dict(self) -> dict:
        """Return cpu tensors ready for torch.save."""
        return {
            "global_step": self.global_step,
            "gradient_scores": self.gradient_scores.cpu(),
            "sync_scores": self.sync_scores.cpu(),
            "binary_indicator_scores": self.binary_indicator_scores.cpu(),
            "final_scores": self.final_scores.cpu(),
            "binary_moving_averages": self.binary_moving_averages.cpu(),
            "weights": self.weights.cpu(),
            # Store OpenSkill statistics per‑uid so we can fully restore them.
            # ordinal is redundant (mu/σ ⇒ ordinal) but handy for debugging.
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
        """Saves the current validator state to disk.

        This method serializes the validator's state dictionary to the configured state path.
        The state includes global step, various score metrics, and weights.

        Exceptions during saving are caught and logged as warnings.
        """
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
        """Loads the validator state from disk.

        This method deserializes the validator's state from the configured state path
        and updates the validator's internal state variables. The state includes:
        - global_step: Training iteration counter
        - gradient_scores: Scores based on gradient quality
        - sync_scores: Scores based on synchronization performance
        - binary_indicator_scores: Binary classification scores
        - final_scores: Combined final evaluation scores
        - binary_moving_averages: Moving averages of binary indicators
        - weights: Peer weighting values
        - openskill_ratings: Dictionary mapping UIDs to OpenSkill rating objects
          that track each peer's skill level using a Bayesian rating system.
          Each rating contains:
          - mu: Mean skill estimate (higher is better)
          - sigma: Uncertainty in the skill estimate (lower means more certainty)
          - ordinal: Conservative skill estimate (mu - n*sigma) used for ranking

        All tensors are converted to float and moved to the configured device.
        Exceptions during loading are caught and logged as warnings.
        """
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
