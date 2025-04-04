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
from rich.console import Console
from rich.table import Table
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from transformers import LlamaForCausalLM

# Local
import tplr

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
        parser.add_argument(
            "--log-to-private-wandb",
            action="store_true",
            help="Logs to the entity you are signed in to if true, else to the public 'tplr'.",
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

        # Init comms with required chain management args,
        # including transformer and compressor for gradient decoding.
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location="/tmp",
            key_prefix="model",
            config=self.config,
            netuid=self.config.netuid,
            metagraph=self.metagraph,
            hparams=self.hparams,
            uid=self.uid,
            totalks=self.totalks,
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
        self.valid_score_indices = []

        # Caching
        self.state_path = f"validator-state-{tplr.__version__}.npz"
        if os.path.isfile(self.state_path):
            self.load_state()
        else:
            self.gradient_scores = torch.zeros(256, dtype=torch.float32)
            self.sync_scores = torch.zeros(256, dtype=torch.float32)
            self.binary_indicator_scores = torch.zeros(256, dtype=torch.float32)
            self.gradient_moving_avg_scores = torch.zeros(256, dtype=torch.float32)
            self.final_moving_avg_scores = torch.zeros(256, dtype=torch.float32)
            self.binary_moving_averages = torch.zeros(256, dtype=torch.float32)
            self.weights = torch.zeros(256, dtype=torch.float32)
            self.normalised_binary_moving_averages = torch.zeros(
                256, dtype=torch.float32
            )
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

        # Initialize final score history (for sliding-window averaging)
        self.final_score_history = defaultdict(list)

        # Initialize peer related attributes
        self.next_peers: tplr.comms.PeerArray | None = None
        self.peers_update_window = -1

    def reset_peer(self, inactive_since: int, uid: int) -> bool:
        if self.current_window - inactive_since > self.hparams.reset_inactivity_windows:
            self.final_score_history[uid] = []
            self.final_moving_avg_scores[uid] = 0.0
            self.weights[uid] = 0.0
            self.gradient_scores[uid] = 0.0
            self.gradient_moving_avg_scores[uid] = 0.0
            self.binary_moving_averages[uid] = 0.0
            self.binary_indicator_scores[uid] = 0.0
            self.normalised_binary_moving_averages[uid] = 0.0
            self.sync_scores[uid] = 0.0
            if uid in self.eval_peers:
                del self.eval_peers[uid]
            del self.inactive_scores[uid]
            tplr.logger.info(f"UID {uid} fully reset after extended inactivity")
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
        )

    async def run(self):
        # Start background block listener
        self.loop = asyncio.get_running_loop()
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
            # Post start_window to R2
            await self.comms.post_start_window(self.start_window)
            tplr.logger.info(
                f"This validator is the highest staked. Posted start_window: {self.start_window}"
            )
        else:
            tplr.logger.info(
                "This validator is not the highest staked. Waiting to fetch start_window."
            )
            # Fetch start_window from highest stake validator
            self.start_window = await self.comms.get_start_window()
            self.global_step = self.current_window - self.start_window
            tplr.logger.info(
                f"Using start_window: {self.start_window}, global_step: {self.global_step}"
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
        )
        if success:
            self.momentum = loaded_momentum
            self.global_step = loaded_checkpoint_window - self.start_window
            self.optimizer = loaded_optimizer
            self.scheduler = loaded_scheduler
            tplr.logger.info(
                f"Loaded checkpoint with global_step={self.global_step}, "
                f"optimizer_step={self.optimizer.state_dict()['state'].get(0, {}).get('step', 0)}, "
                f"scheduler_step={self.scheduler.last_epoch}"
            )
            # Only catch up if we're behind
            if loaded_checkpoint_window < self.current_window:
                tplr.logger.info(
                    f"Checkpoint is behind current window ({loaded_checkpoint_window} < {self.current_window}), starting catchup..."
                )
                await tplr.neurons.catchup_with_aggregation_server(
                    self, loaded_checkpoint_window
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
                tplr.logger.info(
                    f"Waiting for validator window offset, synced: {self.sync_window}, current:{self.current_window}, offset:{self.hparams.validator_offset}"
                )
                await asyncio.sleep(12)

            # 2. Increment sync window and update peer lists
            window_start = tplr.T()
            self.sync_window += 1
            tplr.logger.info(
                f"Sync Window: {self.sync_window}, Scheduler epoch: {self.scheduler.last_epoch}, Global step: {self.global_step}"
            )

            tplr.logger.info(
                f"Processing window: {self.sync_window} current: {self.current_window}"
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

                tplr.logger.info(
                    f"Time to create and post a new peer list because {reason}"
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
            tplr.logger.info(
                f"{tplr.P(self.sync_window, tplr.T() - peer_start)} Updated peers - eval:{len(self.eval_peers)}"
            )

            tplr.logger.info(f"Current gather peers: {self.comms.peers}")
            tplr.logger.info(f"Current evaluation peers: {self.eval_peers}")

            tplr.logger.info(f"Current gather peers: {self.comms.peers}")
            tplr.logger.info(
                f"Current evaluation peers: {list(self.eval_peers.keys())}"
            )

            newly_inactive = self.comms.inactive_peers
            current_window = self.sync_window

            # 3. Process inactive peers and apply penalties
            for uid in newly_inactive:
                if uid not in self.inactive_scores:
                    self.inactive_scores[uid] = (
                        current_window,
                        self.final_moving_avg_scores[uid].item(),
                    )
                    tplr.logger.info(
                        f"UID {uid} became inactive at window {current_window} with score {self.final_moving_avg_scores[uid].item():.4f}"
                    )

            # Apply penalties to all inactive peers
            for uid, (inactive_since, _) in list(self.inactive_scores.items()):
                # If peer became active again, remove from inactive tracking
                if uid in self.eval_peers.keys():
                    del self.inactive_scores[uid]
                    tplr.logger.info(f"UID {uid} became active again")
                    continue

                peer_reset = self.reset_peer(inactive_since, uid)
                if peer_reset:
                    continue

                # Apply flat 25% penalty instead of exponential decay
                old_score = self.final_moving_avg_scores[uid].item()
                new_score = old_score  # Initialize new_score with old_score value
                if self.final_moving_avg_scores[uid] > 0:
                    self.final_moving_avg_scores[uid] *= (
                        0.75  # Apply flat 25% reduction for positive scores only
                    )

                    self.final_score_history[uid] = [
                        final_score * 0.75 if final_score > 0 else final_score
                        for final_score in self.final_score_history[uid]
                    ]
                    new_score = self.final_moving_avg_scores[uid].item()

                    tplr.logger.info(
                        f"UID {uid} penalized for inactivity: "
                        f"{old_score:.4f} -> {new_score:.4f}"
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
                    tplr.logger.error(
                        f"Failed to query timestamp for block {sync_block}: {str(e)}. Retry {retries + 1}/{max_retries}"
                    )
                    retries += 1
                    if retries > max_retries:
                        tplr.logger.error(
                            "Exceeded maximum retries for timestamp query. Falling back to current system time."
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
            tplr.logger.info(f"Using time window for gather: {time_min} to {time_max}")
            tplr.logger.info(f"We are using peers {self.comms.peers}")

            gather_start = tplr.T()
            # Refresh peers explicitly before starting gather to avoid missing updated active peers.
            tplr.logger.info("Refreshing eval peers before gather task in validator...")

            if self.config.test:
                # In test mode, use all UIDs from metagraph except self
                tplr.logger.info("Test mode active: Using all peers from metagraph.")
                all_uids = list(range(len(self.metagraph.S)))
                self.comms.peers = [uid for uid in all_uids if uid != self.uid]

                # For evaluation, also use all peers but track separately with equal initial weight
                self.eval_peers = {uid: 1 for uid in self.comms.peers}
            else:
                # Normal operation - update and filter peers
                self.comms.update_peers_with_buckets()
                self.eval_peers = self.comms.eval_peers

            tplr.logger.info(f"Validator gather peers: {self.comms.peers}")

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
                    tplr.logger.error(
                        "Failed to gather gradients from peers. Waiting for next window."
                    )
                    self.global_step += 1
                    continue
                skipped_uids = gather_result.skipped_uids
                success_rate = gather_result.success_rate
            else:
                state_dict = cast(dict, aggregation_result.get("state_dict"))
                skipped_uids = cast(list[int], state_dict.get("skipped_uids", []))
                success_rate = aggregation_result.get("success_rate", 0.0)

            tplr.logger.info(f"Skipped UIDs: {skipped_uids}")

            gather_sync_scores = await asyncio.gather(
                *(self.evaluate_miner_sync(uid) for uid in self.comms.peers)
            )

            for score_info, uid in zip(gather_sync_scores, self.comms.peers):
                avg_steps_behind = score_info.get("avg_steps_behind", 99.0)
                success = score_info.get("success", False)
                if not success or avg_steps_behind > self.hparams.sync_max_steps_behind:
                    tplr.logger.info(
                        "Slashing %s: avg_steps_behind=%.2f > max=%d",
                        uid,
                        avg_steps_behind,
                        self.hparams.sync_max_steps_behind,
                    )
                    if self.final_moving_avg_scores[uid] > 0:
                        self.final_moving_avg_scores[uid] *= self.sync_score_slash_rate
                        self.final_score_history[uid] = [
                            final_score * self.sync_score_slash_rate
                            if final_score > 0
                            else final_score
                            for final_score in self.final_score_history[uid]
                        ]

            # Slash peers failing to submit gradients
            for uid in skipped_uids:
                tplr.logger.info(
                    f"No gradient gathered from UID {uid}. Slashing moving average score by {1 - self.missing_gradient_slash_rate:.2%}."
                )
                if 0 <= uid < self.final_moving_avg_scores.size(0):
                    old_score = self.final_moving_avg_scores[uid].item()

                    # Only reduce positive scores
                    if self.final_moving_avg_scores[uid] > 0:
                        self.final_moving_avg_scores[uid] *= (
                            self.missing_gradient_slash_rate
                        )
                        self.final_score_history[uid] = [
                            final_score * self.missing_gradient_slash_rate
                            if final_score > 0
                            else final_score
                            for final_score in self.final_score_history[uid]
                        ]

                        new_score = self.final_moving_avg_scores[uid].item()
                        tplr.logger.info(
                            f"Reduced moving average score of UID {uid} from {old_score:.4f} to {new_score:.4f} "
                            f"due to missing gradient in gather."
                        )
                    else:
                        tplr.logger.info(
                            f"Skipped reducing moving average score of UID {uid} (current score: {old_score:.4f}) "
                            f"due to negative or zero value."
                        )
                    self.evaluated_uids.add(uid)
                else:
                    tplr.logger.info(
                        f"UID {uid} not found in final_moving_avg_scores; skipping penalty."
                    )

            # Add check for empty peers (evaluating all peer uids)
            if len(self.comms.eval_peers) == 0:
                tplr.logger.warning(
                    f"No peers available for evaluation in window {self.sync_window}. Waiting for next window."
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

            tplr.logger.info(f"Evaluating random subset of peers: {evaluation_uids}")

            for eval_uid in evaluation_uids:
                tplr.logger.info(f"Evaluating uid: {eval_uid}")

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
                if (
                    eval_result is not None
                    and not (
                        isinstance(eval_result, dict)
                        and eval_result.get("__status") in ["TOO_LATE", "TOO_EARLY"]
                    )
                    and eval_result[0] is not None
                ):
                    state_dict, _ = eval_result

                    # Pull miner-sent pages info from metadata
                    miner_pages = None
                    if (
                        "metadata" in state_dict
                        and "pages_info" in state_dict["metadata"]
                    ):
                        miner_pages = state_dict["metadata"]["pages_info"]
                    else:
                        tplr.logger.warning(
                            f"Missing pages info metadata from miner UID {eval_uid}"
                        )

                    # Load local pages exactly once from the dataset loader with retry handling.
                    local_pages = await retry_call(
                        tplr.r2_dataset.R2DatasetLoader.next_pages,
                        offset=self.sync_window,
                        n_pages=self.hparams.pages_per_window,
                        seed=eval_uid,
                        attempts=3,
                        delay=1,
                        context=f"local pages for UID {eval_uid}",
                        **{},
                    )
                    if local_pages is None:
                        tplr.logger.error(
                            f"Failed to load local pages for UID {eval_uid}. Skipping evaluation for this peer."
                        )
                        continue
                    if miner_pages is not None:
                        if local_pages != miner_pages:
                            tplr.logger.warning(
                                f"Pages mismatch for UID {eval_uid}: miner sent {miner_pages} vs local pages {local_pages}"
                            )
                        else:
                            tplr.logger.info(
                                f"Pages verified for UID {eval_uid}: pages match."
                            )
                    else:
                        tplr.logger.info(
                            f"Using local pages for UID {eval_uid} as miner metadata is missing."
                        )
                    data_start = tplr.T()
                    # Create the evaluation loader using the locally loaded pages.
                    loader_own = await retry_call(
                        tplr.r2_dataset.R2DatasetLoader.create,
                        batch_size=self.hparams.batch_size,
                        sequence_length=self.hparams.sequence_length,
                        pages_info=local_pages,
                        tokenizer=self.tokenizer,
                        attempts=3,
                        delay=1,
                        context=f"own loader for UID {eval_uid}",
                        **{},
                    )
                    if loader_own is None:
                        tplr.logger.error(
                            f"Failed to create loader for own data for UID {eval_uid}. Skipping evaluation."
                        )
                        continue
                    tplr.logger.info(
                        f"{tplr.P(self.sync_window, tplr.T() - data_start)} Loaded evaluation data using pages: {[p[1] for p in local_pages]}"
                    )

                    state_dict, _ = eval_result
                    model_own_data_eval = copy.deepcopy(self.model)

                    # 9. Compute loss before applying gradient
                    self.optimizer.zero_grad()
                    model_own_data_eval.zero_grad()
                    loss_before_own = 0.0
                    n_batches = 0

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

                        tplr.logger.info(
                            f"Evaluating {sample_size_own}/{total_batches_own} batches ({self.hparams.validator_sample_rate * 100:.1f}%)"
                        )

                        for i, batch in enumerate(batches_own):
                            if i not in sampled_indices_own:
                                continue
                            input_ids = torch.tensor(batch, dtype=torch.long).to(
                                model_own_data_eval.device
                            )
                            labels = input_ids.clone()
                            labels = torch.where(
                                labels == self.tokenizer.pad_token_id, -100, labels
                            )
                            outputs = model_own_data_eval(
                                input_ids=input_ids, labels=labels
                            )
                            loss_before_own += outputs.loss.item()
                            n_batches += 1
                            del input_ids, labels, outputs
                            torch.cuda.empty_cache()

                    self.loss_before_per_batch_own = (
                        loss_before_own / n_batches if n_batches > 0 else 0
                    )
                    tplr.logger.debug(
                        f"Loss before (own data): {self.loss_before_per_batch_own}"
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
                                    tplr.logger.warning(
                                        f"Missing totalk for parameter {n}, skipping peer {eval_uid}"
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
                                    tplr.logger.warning(
                                        f"Values contain NaN or Inf for parameter {vals_key}, skipping peer {eval_uid}"
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
                                    tplr.logger.warning(
                                        f"Decompressed gradient for {n} contains NaN/Inf, skipping peer {eval_uid}"
                                    )
                                    raise ValueError(
                                        f"Invalid gradient from peer {eval_uid}: NaN or Inf in decompressed gradient for {n}"
                                    )

                                p.data.sub_(
                                    grad.sign(), alpha=self.scheduler.get_last_lr()[0]
                                )
                    except Exception as e:
                        old_score = self.final_moving_avg_scores[eval_uid].item()

                        if old_score > 0:
                            # Reset positive scores to zero explicitly
                            self.final_moving_avg_scores[eval_uid] = 0.0
                            self.final_score_history[eval_uid] = []
                            tplr.logger.warning(
                                f"Set positive score of UID {eval_uid} from {old_score:.4f} to 0.0 - invalid gradient data"
                            )
                        else:
                            # Negative score is worse than zero; keep it as-is.
                            tplr.logger.warning(
                                f"UID {eval_uid} had negative score {old_score:.4f}; retaining due to invalid gradient data"
                            )

                        # Include in evaluated UIDs so it gets logged in metrics
                        self.evaluated_uids.add(eval_uid)

                        # Log to WandB
                        self.wandb.log(
                            {
                                f"validator/slash/{eval_uid}/score_before": old_score,
                                f"validator/slash/{eval_uid}/score_after": self.final_moving_avg_scores[
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
                                    self.final_moving_avg_scores[eval_uid].item()
                                ),
                                "reason": str(e)[:255],  # Truncate long error messages
                            },
                        )

                        # Skip the rest of processing for this peer
                        continue

                    # 10. Compute loss after gradient application
                    self.optimizer.zero_grad()
                    model_own_data_eval.zero_grad()
                    loss_after_own = 0.0
                    n_batches = 0
                    with torch.no_grad():
                        model_own_data_eval.eval()
                        for i, batch in enumerate(batches_own):
                            if i not in sampled_indices_own:
                                continue
                            input_ids = torch.tensor(batch, dtype=torch.long).to(
                                model_own_data_eval.device
                            )
                            labels = input_ids.clone()
                            labels = torch.where(
                                labels == self.tokenizer.pad_token_id, -100, labels
                            )
                            outputs = model_own_data_eval(
                                input_ids=input_ids, labels=labels
                            )
                            loss_after_own += outputs.loss.item()
                            n_batches += 1
                            del input_ids, labels, outputs
                            torch.cuda.empty_cache()

                    # Clean up stored batches
                    del batches_own, local_pages, loader_own, model_own_data_eval
                    torch.cuda.empty_cache()

                    self.loss_after_per_batch_own = (
                        loss_after_own / n_batches if n_batches > 0 else 0
                    )
                    tplr.logger.debug(
                        f"Loss after (own data): {self.loss_after_per_batch_own}"
                    )

                    # 11. Calculate improvements and update scores
                    # Compute and assign the loss improvement to self
                    self.loss_improvement_own = (
                        self.loss_before_per_batch_own - self.loss_after_per_batch_own
                    )
                    tplr.logger.debug(
                        f"Loss improvement (own data): {self.loss_improvement_own}"
                    )

                    self.relative_improvement_own = (
                        self.loss_improvement_own / self.loss_before_per_batch_own
                        if self.loss_before_per_batch_own > 0
                        else 0.0
                    )
                    tplr.logger.debug(
                        f"Relative improvement (own data): {self.relative_improvement_own:.4f}"
                    )

                    # 7. Load evaluation data from random page
                    model_random_data_eval = copy.deepcopy(self.model)
                    data_start = tplr.T()
                    pages_random = await retry_call(
                        tplr.r2_dataset.R2DatasetLoader.next_pages,
                        offset=self.sync_window,
                        n_pages=self.hparams.pages_per_window,
                        seed=random.randint(0, 10000),
                        attempts=3,
                        delay=1,
                        context="random pages",
                        **{},
                    )
                    if pages_random is None:
                        tplr.logger.error(
                            "Failed to load random pages. Skipping evaluation."
                        )
                        continue

                    loader_random = await retry_call(
                        tplr.r2_dataset.R2DatasetLoader.create,
                        batch_size=self.hparams.batch_size,
                        sequence_length=self.hparams.sequence_length,
                        pages_info=pages_random,
                        tokenizer=self.tokenizer,
                        attempts=3,
                        delay=1,
                        context="random loader",
                        **{},
                    )
                    if loader_random is None:
                        tplr.logger.error(
                            "Failed to create random loader. Skipping evaluation."
                        )
                        continue
                    tplr.logger.info(
                        f"{tplr.P(self.sync_window, tplr.T() - data_start)} Loaded random evaluation data"
                    )
                    state_dict, _ = eval_result

                    # 8. Compute initial loss
                    self.optimizer.zero_grad()
                    model_random_data_eval.zero_grad()
                    loss_before_random = 0.0
                    n_batches = 0

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

                        tplr.logger.info(
                            f"Evaluating {sample_size_random}/{total_batches_random} batches ({self.hparams.validator_sample_rate * 100:.1f}%)"
                        )

                        for idx in sampled_indices_random:
                            batch = batches_random[idx]
                            input_ids = torch.tensor(batch, dtype=torch.long).to(
                                model_random_data_eval.device
                            )
                            labels = input_ids.clone()
                            labels = torch.where(
                                labels == self.tokenizer.pad_token_id, -100, labels
                            )
                            outputs = model_random_data_eval(
                                input_ids=input_ids, labels=labels
                            )
                            loss_before_random += outputs.loss.item()
                            n_batches += 1
                            del input_ids, labels, outputs
                            torch.cuda.empty_cache()

                    self.loss_before_per_batch_random = (
                        loss_before_random / n_batches if n_batches > 0 else 0
                    )
                    tplr.logger.debug(
                        f"Loss before (random data): {self.loss_before_per_batch_random}"
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
                                    grad.sign(), alpha=self.scheduler.get_last_lr()[0]
                                )
                    except Exception as e:
                        tplr.logger.error(
                            f"Failed to apply gradient for UID {eval_uid}: {str(e)}"
                        )
                        continue

                    # 10. Compute loss after gradient application for random data
                    self.optimizer.zero_grad()
                    model_random_data_eval.zero_grad()
                    loss_after_random = 0.0
                    n_batches = 0
                    with torch.no_grad():
                        model_random_data_eval.eval()
                        for i, batch in enumerate(batches_random):
                            if i not in sampled_indices_random:
                                continue
                            input_ids = torch.tensor(batch, dtype=torch.long).to(
                                model_random_data_eval.device
                            )
                            labels = input_ids.clone()
                            labels = torch.where(
                                labels == self.tokenizer.pad_token_id, -100, labels
                            )
                            outputs = model_random_data_eval(
                                input_ids=input_ids, labels=labels
                            )
                            loss_after_random += outputs.loss.item()
                            n_batches += 1
                            del input_ids, labels, outputs
                            torch.cuda.empty_cache()

                    # Clean up stored batches, loader & pages
                    del (
                        batches_random,
                        pages_random,
                        loader_random,
                        model_random_data_eval,
                    )
                    torch.cuda.empty_cache()

                    self.loss_after_per_batch_random = (
                        loss_after_random / n_batches if n_batches > 0 else 0
                    )
                    tplr.logger.info(
                        f"Loss after (random data): {self.loss_after_per_batch_random}"
                    )

                    # 11. Calculate improvements and update scores
                    # Compute and assign the loss improvement to self
                    self.loss_improvement_random = (
                        self.loss_before_per_batch_random
                        - self.loss_after_per_batch_random
                    )
                    tplr.logger.info(
                        f"Loss improvement (random data): {self.loss_improvement_random}"
                    )

                    self.relative_improvement_random = (
                        self.loss_improvement_random / self.loss_before_per_batch_random
                        if self.loss_before_per_batch_random > 0
                        else 0.0
                    )
                    tplr.logger.debug(
                        f"Relative improvement (random data): {self.relative_improvement_random}"
                    )

                    # Calculate original performance score (gradient quality)
                    self.gradient_scores[eval_uid] = min(
                        self.hparams.max_gradient_score,
                        (loss_before_random - loss_after_random) / loss_before_random,
                    )
                    tplr.logger.debug(
                        f"Gradient Score: {self.gradient_scores[eval_uid]}"
                    )

                    # Update exponential moving average of gradient scores with alpha=gradient_score_ma_alpha
                    # New score = (1-alpha)*old_score + alpha*new_score
                    self.gradient_moving_avg_scores[eval_uid] = (
                        1 - self.hparams.gradient_score_ma_alpha
                    ) * self.gradient_moving_avg_scores[
                        eval_uid
                    ] + self.hparams.gradient_score_ma_alpha * self.gradient_scores[
                        eval_uid
                    ]
                    tplr.logger.debug(
                        f"Gradient moving average : {self.gradient_moving_avg_scores[eval_uid]}"
                    )

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
                    tplr.logger.info(
                        f"Binary Indicator Score : {self.binary_indicator_scores[eval_uid]}"
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
                    tplr.logger.debug(
                        f"Binary Moving Average Score : {self.binary_moving_averages[eval_uid]}"
                    )

                    # Normalize binary moving average to [0,1] range
                    self.normalised_binary_moving_averages[eval_uid] = (
                        (self.binary_moving_averages[eval_uid]) / 2
                    )
                    tplr.logger.debug(
                        f"Normalised Binary Moving Average Score : {self.normalised_binary_moving_averages[eval_uid]}"
                    )

                    sync_result = await self.evaluate_miner_sync(eval_uid)
                    sync_score = cast(
                        float,
                        sync_result.get("sync_score", 0.0),
                    )
                    self.log_sync_score(eval_uid, sync_result)

                    # Store the sync score for this miner
                    self.sync_scores[eval_uid] = sync_score

                    # Your existing final_score calculation with sync_score added
                    final_score = (
                        sign_preserving_multiplication(
                            self.gradient_moving_avg_scores[eval_uid],
                            self.normalised_binary_moving_averages[eval_uid],
                        )
                        * sync_score
                    )
                    tplr.logger.debug(
                        f"Computed Final Score for UID {eval_uid}: {final_score}"
                    )

                    # Sliding window update for the final moving average score
                    self.final_score_history[eval_uid].append(final_score)
                    if (
                        len(self.final_score_history[eval_uid])
                        > self.hparams.moving_average_window
                    ):
                        self.final_score_history[eval_uid].pop(0)
                    self.final_moving_avg_scores[eval_uid] = sum(
                        self.final_score_history[eval_uid]
                    ) / len(self.final_score_history[eval_uid])
                    tplr.logger.debug(
                        f"Updated Final Moving Average Score for UID {eval_uid}: {self.final_moving_avg_scores[eval_uid]}"
                    )

                    self.evaluated_uids.add(eval_uid)

                    # 12. Calculate weights using min power norm
                    self.weights = torch.zeros_like(self.final_moving_avg_scores)
                    evaluated_mask = torch.zeros_like(
                        self.final_moving_avg_scores, dtype=torch.bool
                    )
                    evaluated_mask[list(self.evaluated_uids)] = True
                    positive_mask = (self.final_moving_avg_scores > 0) & evaluated_mask
                    if positive_mask.any():
                        self.weights[positive_mask] = min_power_normalization(
                            self.final_moving_avg_scores[positive_mask],
                            power=self.hparams.power_normalisation,
                        )
                        weight_sum = self.weights.sum().item()
                        tplr.logger.debug(f"Weight sum: {weight_sum}")
                        if abs(weight_sum - 1.0) > 1e-6:
                            tplr.logger.warning(
                                f"Weights sum to {weight_sum}, expected close to 1.0"
                            )
                    else:
                        tplr.logger.info(
                            "No positive scores found, all weights set to 0"
                        )

                    tplr.logger.info(
                        f"{tplr.P(self.sync_window, tplr.T() - eval_start)} Completed evaluation"
                    )

                else:
                    tplr.logger.info(
                        f"No gradient received from UID {eval_uid}. Slashing moving average score by {1 - self.missing_gradient_slash_rate:.2%}"
                    )
                    old_score = self.final_moving_avg_scores[eval_uid].item()

                    if self.final_moving_avg_scores[eval_uid] > 0:
                        self.final_moving_avg_scores[eval_uid] *= (
                            self.missing_gradient_slash_rate
                        )
                        self.final_score_history[eval_uid] = [
                            final_score * self.missing_gradient_slash_rate
                            if final_score > 0
                            else final_score
                            for final_score in self.final_score_history[eval_uid]
                        ]
                        new_score = self.final_moving_avg_scores[eval_uid].item()
                        tplr.logger.info(
                            f"Reduced moving average score of UID {eval_uid} from {old_score:.4f} to {new_score:.4f} due to missing gradient."
                        )
                    else:
                        tplr.logger.info(
                            f"Skipped reducing moving average score of UID {eval_uid} (current score: {old_score:.4f}) due to negative or zero value."
                        )

                    # Ensure the UID is included in evaluated_uids
                    self.evaluated_uids.add(eval_uid)

                    # Recalculate weights
                    self.weights = torch.zeros_like(self.final_moving_avg_scores)
                    evaluated_mask = torch.zeros_like(
                        self.final_moving_avg_scores, dtype=torch.bool
                    )
                    evaluated_mask[list(self.evaluated_uids)] = True

                    positive_mask = (self.final_moving_avg_scores > 0) & evaluated_mask

                    if positive_mask.any():
                        # Apply normalization to all positive scores at once
                        self.weights[positive_mask] = min_power_normalization(
                            self.final_moving_avg_scores[positive_mask],
                            power=self.hparams.power_normalisation,
                        )

                        # Log warning if weights don't sum to 1
                        weight_sum = self.weights.sum().item()
                        tplr.logger.debug(f"Weight sum: {weight_sum}")
                        if abs(weight_sum - 1.0) > 1e-6:
                            tplr.logger.warning(
                                f"Weights sum to {weight_sum}, expected close to 1.0"
                            )
                    else:
                        tplr.logger.info(
                            "No positive scores found, all weights set to 0"
                        )

                    # Log updated scores
                    tplr.logger.info(
                        "Updated scores for evaluated UIDs after slashing:"
                    )
                    # Log evaluated UID scores (fixed join call)
                    line = " | ".join(
                        f"UID {uid}: {self.final_moving_avg_scores[uid]:.4f}"
                        for uid in sorted(self.evaluated_uids)
                    )
                    tplr.logger.info(line)

                    # Optionally, log to WandB
                    self.wandb.log(
                        {
                            f"validator/final_moving_avg_scores/{eval_uid}": self.final_moving_avg_scores[
                                eval_uid
                            ].item(),
                            f"validator/weights/{eval_uid}": self.weights[
                                eval_uid
                            ].item(),
                        },
                        step=self.global_step,
                    )
                    tplr.logger.info(
                        f"{tplr.P(self.sync_window, tplr.T() - scoring_start)} Computed scores and weights"
                    )

                tplr.logger.info(
                    f"{tplr.P(self.sync_window, tplr.T() - eval_start)} Completed evaluation"
                )

            # Log scores and metrics for evaluated UIDs as a table
            headers = [
                "UID",
                "Last Score",
                "Binary Indicator",
                "Binary Moving Avg",
                "Norm Binary Score",
                "Final Moving Avg",
                "Sync score",
                "Weight",
            ]
            table = [headers]
            for uid in sorted(self.evaluated_uids):
                row = [
                    str(uid),
                    f"{self.gradient_scores[uid]:.4f}",
                    f"{self.binary_indicator_scores[uid]:.4f}",
                    f"{self.binary_moving_averages[uid]:.4f}",
                    f"{self.normalised_binary_moving_averages[uid]:.4f}",
                    f"{self.final_moving_avg_scores[uid]:.4f}",
                    f"{self.sync_scores[uid]:.4f}",
                    f"{self.weights[uid]:.4f}",
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
                tplr.logger.warning(
                    "rich module not found; falling back to basic formatting."
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

            tplr.logger.info("Updated scores for evaluated UIDs:\n" + table_str)

            # Log WandB metrics per UID
            for uid in sorted(self.evaluated_uids):
                # Extract primitive values from tensors for WandB
                gradient_score = float(self.gradient_scores[uid].item())
                binary_indicator = float(self.binary_indicator_scores[uid].item())
                binary_moving_avg = float(self.binary_moving_averages[uid].item())
                normalised_binary = float(
                    self.normalised_binary_moving_averages[uid].item()
                )
                sync_score = float(self.sync_scores[uid].item())
                final_moving_avg = float(self.final_moving_avg_scores[uid].item())
                weight = float(self.weights[uid].item())

                self.wandb.log(
                    {
                        f"validator/gradient_scores/{uid}": gradient_score,
                        f"validator/binary_indicators/{uid}": binary_indicator,
                        f"validator/binary_moving_averages/{uid}": binary_moving_avg,
                        f"validator/normalised_binary_scores/{uid}": normalised_binary,
                        f"validator/final_moving_avg_scores/{uid}": final_moving_avg,
                        f"validator/sync_score/{uid}": sync_score,
                        f"validator/weights/{uid}": weight,
                    },
                    step=self.global_step,
                )

                # Log to InfluxDB metrics per UID with primitive types
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
                        "normalised_binary": normalised_binary,
                        "sync_score": sync_score,
                        "final_moving_avg_score": final_moving_avg,
                        "weight": weight,
                    },
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
                tplr.logger.warning("No gradients to apply.")
                self.scheduler.step()
                torch.cuda.empty_cache()

            tplr.logger.info(
                f"{tplr.P(self.sync_window, tplr.T() - update_start)} Updated model"
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
            tplr.logger.info(f"Stored debug values for window {self.current_window}")
            # Log total window time and metrics
            tplr.logger.info(
                f"{tplr.P(self.sync_window, tplr.T() - window_start)} Completed window iteration"
            )

            # 16. Log evaluation metrics once all evaluations are done
            evaluation_metrics = {
                "validator/loss/own/before": self.loss_before_per_batch_own,
                "validator/loss/own/after": self.loss_after_per_batch_own,
                "validator/loss/random/before": self.loss_before_per_batch_random,
                "validator/loss/random/after": self.loss_after_per_batch_random,
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
                "validator/timing/gather": tplr.T() - gather_start,
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
                    "loss_own_before": float(self.loss_before_per_batch_own),
                    "loss_own_after": float(self.loss_after_per_batch_own),
                    "loss_random_before": float(self.loss_before_per_batch_random),
                    "loss_random_after": float(self.loss_after_per_batch_random),
                    "loss_own_improvement": float(self.relative_improvement_own),
                    "loss_random_improvement": float(self.relative_improvement_random),
                    "current_block": int(self.current_block),
                    "evaluated_uids_count": int(len(self.evaluated_uids)),
                    "learning_rate": float(self.scheduler.get_last_lr()[0]),
                    "active_miners_count": int(len(self.valid_score_indices)),
                    "gather_success_rate": gather_success_rate,
                    "window_total_time": float(tplr.T() - window_start),
                    "peer_update_time": float(tplr.T() - peer_start),
                    "gather_time": float(tplr.T() - gather_start),
                    "evaluation_time": float(tplr.T() - eval_start),
                    "model_update_time": float(tplr.T() - update_start),
                    "total_peers": int(len(self.comms.peers)),
                    "total_skipped": int(total_skipped),
                },
            )
            tplr.logger.info("Finished metrics logging call for validator")

            # 17. Create checkpoints periodically
            if (
                self.global_step % self.hparams.checkpoint_frequency == 0
                and self.global_step != 0
            ):
                tplr.logger.info(
                    f"Creating checkpoint at global_step {self.global_step}"
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

    def select_initial_peers(self) -> tplr.comms.PeerArray | None:
        try:
            tplr.logger.info("Starting selection of initial gather peers")
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
                tplr.logger.info(
                    f"Selected {len(top_incentive_peers)} initial peers purely based "
                    f"on incentive: {top_incentive_peers}"
                )
                return top_incentive_peers

            # 2. If needed, fill up with active peers
            remaining_active_peers = np.array(
                list(set(self.comms.active_peers) - set(top_incentive_peers))
            )
            top_incentive_and_active_peers = np.concatenate(
                [top_incentive_peers, remaining_active_peers]
            )[: self.hparams.max_topk_peers]

            assert len(top_incentive_and_active_peers) <= self.hparams.max_topk_peers
            if len(top_incentive_and_active_peers) >= self.hparams.minimum_peers:
                tplr.logger.info(
                    f"Selected {len(top_incentive_and_active_peers)} initial peers. "
                    f"{len(top_incentive_peers)} with incentive: {top_incentive_peers} "
                    f"and {len(top_incentive_and_active_peers) - len(top_incentive_peers)} without: "
                    f"{remaining_active_peers[: len(top_incentive_and_active_peers) - len(top_incentive_peers)]}"
                )
                return top_incentive_and_active_peers

            # 3. Give up
            tplr.logger.info(
                f"Failed to select at least {self.hparams.minimum_peers} initial gather "
                f"peers. Found only {len(top_incentive_and_active_peers)} active "
                f"peers, of which {len(top_incentive_peers)} had incentive and "
                f"{len(top_incentive_and_active_peers) - len(top_incentive_peers)} "
                f"were incentiveless active peers."
            )
            return None

        except Exception as e:
            tplr.logger.error(f"Failed to create new peer list: {e}")
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
        non_zero_weight_uids = torch.nonzero(self.weights).flatten().numpy()
        active_non_zero_weight_uids = np.intersect1d(
            list(self.comms.active_peers),
            non_zero_weight_uids,
        )
        candidates = np.setdiff1d(active_non_zero_weight_uids, old_peers)
        num_initial_candidates = len(candidates)
        tplr.logger.info(
            f"Starting off with {num_initial_candidates} initial candidates."
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
                tplr.logger.info(
                    "Step 1: Peer list already full of active peers but there are no "
                    "candidates, aborting selection"
                )
                return None
            tplr.logger.info(
                "Step 1: Peer list already full of active peers, continuing"
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
                    tplr.logger.info(
                        f"There are only {len(self.comms.active_peers)} active peers"
                        f"in total, which is less than the minimum amount "
                        f"{self.hparams.minimum_peers}, aborting selection"
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
                tplr.logger.info(
                    f"Using {len(candidates)} candidates and "
                    f"{len(ingoing_secondary_candidates)} secondary candidates"
                )
            selected_peers = self._replace_peers(
                old_peers, ingoing, inactive_gather_peers
            )
            # if we've used all candidates, we're done
            if len(ingoing) >= len(candidates):
                tplr.logger.info(
                    f"Step 1: We've used all candidates, returning {len(selected_peers)} "
                    "selected peers."
                )
                return selected_peers
            # else, update the candidates
            else:
                tplr.logger.info(
                    f"Finished step 1: Dropped {len(inactive_gather_peers)} inactive "
                    f"peers and added {len(ingoing)} active ones. We now have "
                    f"{len(selected_peers)} selected peers"
                )
                candidates = np.setdiff1d(candidates, ingoing)

        # ----------------------------------------------------------------------
        # 2. Drop zero-weight gather peers
        # ----------------------------------------------------------------------
        zero_weight_selected_peers = np.setdiff1d(selected_peers, non_zero_weight_uids)
        original_peers_left = np.intersect1d(old_peers, selected_peers)
        if len(zero_weight_selected_peers) == 0:
            tplr.logger.info("Step 2: No zero-weight peers to drop, continuing")
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
                tplr.logger.info(
                    f"Step 2: No more candidates, returning {len(selected_peers)} "
                    "selected peers."
                )
                return selected_peers
            tplr.logger.info(
                "Finished step 2 (dropped zero-weight peers) and we now have "
                f"{len(selected_peers)} selected peers"
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
        tplr.logger.info(
            f"Finished step 3 (replaced {len(outgoing)} peers with non-zero weight) "
            f"and we now have {len(selected_peers)} selected peers"
        )
        tplr.logger.info(
            f"Step 3: Done, returning {len(selected_peers)} selected peers."
        )
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
                tplr.logger.warning("No state_dict found in aggregation result")
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
                tplr.logger.info(
                    f"Set gradients for {tensors_applied} tensors in {time.time() - update_start:.2f}s"
                )

                # Update parameters with optimizer
                self.optimizer.step()
                self.scheduler.step()
                torch.cuda.empty_cache()

                tplr.logger.info("Successfully applied aggregation")
                return True
            else:
                tplr.logger.warning("No tensors were applied during aggregation")
                return False

        except Exception as e:
            tplr.logger.error(f"Error applying aggregated gradients: {e}")
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
                tplr.logger.info(f"Gradient data missing for parameter {n}, skipping.")
        self.optimizer.step()
        self.scheduler.step()
        torch.cuda.empty_cache()

    def save_state(self):
        """Saves the state of the validator to a file."""
        try:
            tplr.logger.info("Saving validator state.")

            # Save the state of the validator to file.
            np.savez(
                self.state_path,
                global_step=self.global_step,
                gradient_scores=self.gradient_scores,
                sync_scores=self.sync_scores,
                binary_indicator_scores=self.binary_indicator_scores,
                gradient_moving_avg_scores=self.gradient_moving_avg_scores,
                final_moving_avg_scores=self.final_moving_avg_scores,
                binary_moving_averages=self.binary_moving_averages,
                weights=self.weights,
            )
        except Exception as e:
            tplr.logger.warning(f"Failed to save validator state: {e}")

    def load_state(self):
        """Loads the state of the validator from a file."""
        try:
            tplr.logger.info("Loading validator state.")

            # Load the state of the validator from file.
            state = np.load(self.state_path)
            self.gradient_scores = state["gradient_scores"]
            self.sync_scores = state["sync_scores"]
            self.binary_indicator_scores = state["binary_indicator_scores"]
            self.gradient_moving_avg_scores = state["gradient_moving_avg_scores"]
            self.final_moving_avg_scores = state["final_moving_avg_scores"]
            self.binary_moving_averages = state["binary_moving_averages"]
            self.weights = state["weights"]
            tplr.logger.info(
                f"Loaded state from global state {state.global_state}: {state}"
            )
        except Exception as e:
            tplr.logger.warning(f"Failed to load validator state: {e}")

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
    asyncio.run(Validator().run())
