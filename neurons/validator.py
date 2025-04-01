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

# type: ignore

# Standard library
import os
import argparse
import asyncio
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
import bittensor as bt
import numpy as np
from rich.table import Table
from rich.console import Console
from collections import defaultdict
from datetime import datetime, timedelta, timezone


# Third party
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from transformers import LlamaForCausalLM

# Local
import tplr
from tplr import evaluation

# GPU optimizations.
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
        self.hparams = tplr.load_hparams()

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

        # Init comms
        self.comms = tplr.Comms(
            wallet=self.wallet,
            config=self.config,
            metagraph=self.metagraph,
            hparams=self.hparams,
            uid=self.uid,
        )
        self.bucket = self.comms.bucket  # Bucket info comes from blockchain commitments

        self.peer_manager = tplr.PeerManager(
            chain=self.comms, hparams=self.hparams, metagraph=self.metagraph
        )
        self.chain_sync = tplr.ChainSync(
            config=self.config,
            netuid=self.config.netuid,
            metagraph=self.metagraph,
            hparams=self.hparams,
            wallet=self.wallet,
        )

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
        self.gradient_scores = torch.zeros(self.metagraph.n, dtype=torch.float32)
        self.binary_indicator_scores = torch.zeros(
            self.metagraph.n, dtype=torch.float32
        )
        self.gradient_moving_avg_scores = torch.zeros(
            self.metagraph.n, dtype=torch.float32
        )
        self.final_moving_avg_scores = torch.zeros(
            self.metagraph.n, dtype=torch.float32
        )
        self.binary_moving_averages = torch.zeros(self.metagraph.n, dtype=torch.float32)
        self.weights = torch.zeros(self.metagraph.n, dtype=torch.float32)
        self.normalised_binary_moving_averages = torch.zeros(
            self.metagraph.n, dtype=torch.float32
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

        # Initialize peers
        self.peers = []
        # Weighted selection counters for fair picking of eval peers
        self.eval_peers = defaultdict(lambda: 1)

        # Track inactive peer scores
        self.inactive_scores = {}  # {uid: (last_active_window, last_score)}
        self.inactivity_slash_rate = 0.25  # 25% slash per window
        self.missing_gradient_slash_rate = 0.75

        # Initialize final score history (for sliding-window averaging)
        self.final_score_history = defaultdict(list)

        #  TODO: Move out
        def reset_peer(self, inactive_since: int, uid: int) -> bool:
            if (
                self.current_window - inactive_since
                > self.hparams.reset_inactivity_windows
            ):
                self.final_score_history[uid] = []
                self.final_moving_avg_scores[uid] = 0.0
                self.weights[uid] = 0.0
                self.gradient_scores[uid] = 0.0
                self.gradient_moving_avg_scores[uid] = 0.0
                self.binary_moving_averages[uid] = 0.0
                self.binary_indicator_scores[uid] = 0.0
                self.normalised_binary_moving_averages[uid] = 0.0
                if uid in self.eval_peers:
                    del self.eval_peers[uid]
                del self.inactive_scores[uid]
                tplr.logger.info(f"UID {uid} fully reset after extended inactivity")
                return True
            return False

        # Add lock for metrics and initialize evaluation metrics collection
        self.metrics_lock = asyncio.Lock()
        self.eval_metrics_collection = {
            "own_before": [],
            "own_after": [],
            "random_before": [],
            "random_after": [],
            "own_improvement": [],
            "random_improvement": [],
        }

    async def run(self):
        # Start background block listener
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(
            target=self.block_listener, args=(self.loop,), daemon=True
        ).start()
        # Start chain sync tasks
        # self.chain_sync.start_commitment_fetcher()
        self.comms.start_background_tasks()

        # Start active peer tracking as a background task
        asyncio.create_task(self.peer_manager.track_active_peers())

        # Check and commit your bucket
        await self.chain_sync.check_and_commit_bucket()

        # Load Peers
        self.peers = self.chain_sync.peers
        tplr.logger.info(f"Filtered gather peers with buckets: {self.peers}")

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

        time_min = None
        while True:
            # 1. Wait for the validator window offset
            while self.sync_window >= (
                self.current_window - self.hparams.validator_offset
            ):
                tplr.logger.info(
                    f"Waiting for validator window offset, synced: {self.sync_window}, current: {self.current_window}, offset: {self.hparams.validator_offset}"
                )
                await asyncio.sleep(12)
            # Reset timer for peer update right after waiting
            peer_start = tplr.T()
            # 2. Increment sync window and update peer lists
            window_start = tplr.T()
            self.sync_window += 1
            tplr.logger.info(
                f"Sync Window: {self.sync_window}, Scheduler epoch: {self.scheduler.last_epoch}, Global step: {self.global_step}"
            )

            tplr.logger.info(
                f"Processing window: {self.sync_window} current: {self.current_window}"
            )
            # self.comms.update_peers_with_buckets()
            self.peers = self.chain_sync.peers
            self.eval_peers = self.chain_sync.eval_peers
            tplr.logger.info(
                f"{tplr.P(self.sync_window, tplr.T() - peer_start)} Updated peers - gather: {len(self.peers)}, eval: {len(self.eval_peers)}"
            )

            tplr.logger.info(f"Current gather peers: {self.peers}")
            tplr.logger.info(f"Current evaluation peers: {self.eval_peers}")

            tplr.logger.info(f"Current gather peers: {self.peers}")
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
            tplr.logger.info(f"We are using peers {self.peers}")

            gather_start = tplr.T()
            # Refresh peers explicitly before starting gather to avoid missing updated active peers.
            tplr.logger.info("Refreshing peers before gather task in validator...")

            self.chain_sync.update_peers_with_buckets()
            self.peers = self.chain_sync.peers
            self.eval_peers = self.chain_sync.eval_peers

            tplr.logger.info(f"Validator gather peers: {self.peers}")

            gather_start = tplr.T()
            gather_result = await self.comms.gather(
                my_uid=str(self.uid),
                uids=self.peers,
                window=self.sync_window,
                key="gradient",
                device=self.config.device,
                timeout=35,
                local=False,
                time_min=time_min,
                time_max=time_max,
            )

            if gather_result is None:
                tplr.logger.error(
                    "Failed to gather gradients from peers. Waiting for next window."
                )
                self.global_step += 1
                continue

            tplr.logger.info(f"Skipped UIDs: {gather_result.skipped_uids}")

            # ---- Validate gathered compressed gradients ----
            valid_uids = []
            invalid_uids = []
            num_peers = len(gather_result.uids)
            # Iterate over each peer's data.
            for idx, uid in enumerate(gather_result.uids):
                # Build per-peer gradient dictionary from aggregated state_dict lists.
                peer_state = {}
                for key, tensor_list in gather_result.state_dict.__dict__.items():
                    if isinstance(tensor_list, list) and len(tensor_list) == num_peers:
                        peer_state[key] = tensor_list[idx]
                # Validate using the new logic.
                is_valid, err_msg = tplr.neurons.validate_compressed_gradients(
                    peer_state,
                    self.totalks,
                    allowed_topk=self.hparams.topk_compression,
                    device=self.config.device,
                )
                if is_valid:
                    valid_uids.append(uid)
                else:
                    tplr.logger.warning(
                        f"Gradient from UID {uid} failed validation: {err_msg}"
                    )
                    invalid_uids.append(uid)
            # Update gathered result to include only valid peers.
            gather_result.uids = valid_uids
            gather_result.skipped_uids.extend(invalid_uids)
            # ----------------------------------------------------------

            # Add check for empty peers (evaluating all peer uids)
            if not self.peers:
                tplr.logger.warning(
                    f"No peers available for evaluation in window {self.sync_window}. Waiting for next window."
                )
                self.global_step += 1
                continue

            # 5. Evaluate peers in parallel using modular evaluation logic.
            eval_start = tplr.T()
            # Use weighted candidate selection instead of random sampling.
            candidate_uids = list(self.eval_peers.keys())
            candidate_weights = [self.eval_peers[uid] for uid in candidate_uids]
            k = min(self.hparams.uids_per_window, len(candidate_uids))
            evaluation_uids = evaluation.weighted_random_sample_no_replacement(
                candidate_uids, candidate_weights, k
            )

            # Reset counters for chosen peers.
            for uid in evaluation_uids:
                self.eval_peers[uid] = 1

            # Increment counters for peers not chosen.
            for uid in candidate_uids:
                if uid not in evaluation_uids:
                    self.eval_peers[uid] += 1
            self.comms.eval_peers = self.eval_peers

            tplr.logger.info(f"Evaluating random subset of peers: {evaluation_uids}")

            peers_per_round = self.hparams.peers_per_eval_round
            eval_results = {}
            if not evaluation_uids:
                tplr.logger.warning("No eval peers available.")
            else:
                for i in range(0, len(evaluation_uids), peers_per_round):
                    current_batch = evaluation_uids[i : i + peers_per_round]
                    tplr.logger.info(
                        f"Evaluating batch {i // peers_per_round + 1}: {current_batch}"
                    )
                    batch_results = await evaluation.evaluate_peers_parallel(
                        current_batch,
                        self.comms,
                        self.sync_window,
                        self.hparams,
                        self.tokenizer,
                        self.config,
                        self.model,
                        self.transformer,
                        self.compressor,
                        self.xshapes,
                        self.totalks,
                        self.config.device,
                        self.scheduler.get_last_lr()[0],
                        self.optimizer,
                        self.scheduler,
                        time_min,
                        time_max,
                    )
                    eval_results.update(batch_results)
            # Process evaluation results.
            for eval_uid, result in eval_results.items():
                if result is not None:
                    self.gradient_scores[eval_uid] = result["gradient_score"]
                    self.loss_before_per_batch_own = result["loss_before_per_batch_own"]
                    self.loss_after_per_batch_own = result["loss_after_per_batch_own"]
                    self.relative_improvement_own = result["relative_improvement_own"]
                    self.loss_before_per_batch_random = result[
                        "loss_before_per_batch_random"
                    ]
                    self.loss_after_per_batch_random = result[
                        "loss_after_per_batch_random"
                    ]
                    self.relative_improvement_random = result[
                        "relative_improvement_random"
                    ]
                    self.binary_indicator_scores[eval_uid] = result["binary_indicator"]
                    self.binary_moving_averages[eval_uid] = (
                        1 - self.hparams.binary_score_ma_alpha
                    ) * self.binary_moving_averages[
                        eval_uid
                    ] + self.hparams.binary_score_ma_alpha * result["binary_indicator"]
                    self.normalised_binary_moving_averages[eval_uid] = (
                        self.binary_moving_averages[eval_uid] / 2
                    )
                    final_score = sign_preserving_multiplication(
                        self.gradient_moving_avg_scores[eval_uid],
                        self.normalised_binary_moving_averages[eval_uid],
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
                    tplr.logger.debug(
                        f"Random metrics for peer {eval_uid}: before={self.loss_before_per_batch_random:.4f}, after={self.loss_after_per_batch_random:.4f}"
                    )
                    async with self.metrics_lock:
                        self.eval_metrics_collection["own_before"].append(
                            float(self.loss_before_per_batch_own)
                        )
                        self.eval_metrics_collection["own_after"].append(
                            float(self.loss_after_per_batch_own)
                        )
                        self.eval_metrics_collection["random_before"].append(
                            float(self.loss_before_per_batch_random)
                        )
                        self.eval_metrics_collection["random_after"].append(
                            float(self.loss_after_per_batch_random)
                        )
                        self.eval_metrics_collection["own_improvement"].append(
                            float(self.relative_improvement_own)
                        )
                        self.eval_metrics_collection["random_improvement"].append(
                            float(self.relative_improvement_random)
                        )
                else:
                    tplr.logger.info(f"No evaluation result for UID {eval_uid}.")

            tplr.logger.info(f"Evaluation phase took {tplr.T() - eval_start:.2f}s")

            evaluation_metrics = {
                "validator/loss/own/before": evaluation.safe_last(
                    self.eval_metrics_collection["own_before"]
                ),
                "validator/loss/own/after": evaluation.safe_last(
                    self.eval_metrics_collection["own_after"]
                ),
                "validator/loss/random/before": evaluation.safe_last(
                    self.eval_metrics_collection["random_before"]
                ),
                "validator/loss/random/after": evaluation.safe_last(
                    self.eval_metrics_collection["random_after"]
                ),
                "validator/loss/own/improvement": evaluation.safe_last(
                    self.eval_metrics_collection["own_improvement"]
                ),
                "validator/loss/random/improvement": evaluation.safe_last(
                    self.eval_metrics_collection["random_improvement"]
                ),
                "validator/network/block": self.current_block,
                "validator/network/window": self.sync_window,
                "validator/network/step": self.global_step,
                "validator/network/evaluated_uids": len(self.evaluated_uids),
                "validator/optimizer/learning_rate": self.scheduler.get_last_lr()[0],
                "validator/network/active_miners": len(self.valid_score_indices),
                "validator/gather/success_rate": gather_result.success_rate * 100
                if gather_result
                else 0,
                "validator/timing/window_total": tplr.T() - window_start,
                "validator/timing/peer_update": tplr.T() - peer_start,
                "validator/timing/gather": tplr.T() - gather_start,
                "validator/timing/evaluation": tplr.T() - eval_start,
                # "validator/timing/model_update": tplr.T() - update_start,
            }
            self.wandb.log(evaluation_metrics, step=self.global_step)
            tplr.logger.info(f"Skipped UIDs: {gather_result.skipped_uids}")

            # Calculate weights using min power normalization over evaluated peers with positive final scores
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
                tplr.logger.info("No positive scores found, all weights set to 0")

            # Log scores and metrics for evaluated UIDs
            # Build a table with headers and one row per evaluated UID
            headers = [
                "UID",
                "Last Score",
                "Binary Indicator",
                "Binary Moving Avg",
                "Norm Binary Score",
                "Final Moving Avg",
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
                    f"{self.weights[uid]:.4f}",
                ]
                table.append(row)

            # Format the table using Rich for better visual appearance in PM2 logs.

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
                final_moving_avg = float(self.final_moving_avg_scores[uid].item())
                weight = float(self.weights[uid].item())

                self.wandb.log(
                    {
                        f"validator/gradient_scores/{uid}": gradient_score,
                        f"validator/binary_indicators/{uid}": binary_indicator,
                        f"validator/binary_moving_averages/{uid}": binary_moving_avg,
                        f"validator/normalised_binary_scores/{uid}": normalised_binary,
                        f"validator/final_moving_avg_scores/{uid}": final_moving_avg,
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

            if gather_result is not None and gather_result.state_dict is not None:
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
                        tplr.logger.info(
                            f"Gradient data missing for parameter {n}, skipping."
                        )
            tplr.logger.info(
                f"{tplr.P(self.sync_window, tplr.T() - update_start)} Updated model"
            )

            self.optimizer.step()
            self.scheduler.step()
            torch.cuda.empty_cache()

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
            if gather_result is not None:
                debug_dict["successful_peers"] = sorted(
                    list(set(self.peers) - set(gather_result.skipped_uids))
                )
                debug_dict["skipped_peers"] = sorted(list(gather_result.skipped_uids))

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
                "validator/gather/success_rate": gather_result.success_rate * 100
                if gather_result
                else 0,  # Success percentage
                "validator/timing/window_total": tplr.T() - window_start,
                "validator/timing/peer_update": tplr.T() - peer_start,
                "validator/timing/gather": tplr.T() - gather_start,
                "validator/timing/evaluation": tplr.T() - eval_start,
                "validator/timing/model_update": tplr.T() - update_start,
            }
            self.wandb.log(evaluation_metrics, step=self.global_step)

            # Log metrics to InfluxDB in parallel using primitive types
            gather_success_rate = (
                float(gather_result.success_rate * 100) if gather_result else 0.0
            )
            total_skipped = len(gather_result.skipped_uids) if gather_result else 0

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
                    "total_peers": int(len(self.peers)),
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
                    "momentum": {k: v.cpu().clone() for k, v in self.momentum.items()},
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


def sign_preserving_multiplication(a, b):
    return -abs(a) * abs(b) if a < 0 or b < 0 else a * b


if __name__ == "__main__":
    asyncio.run(Validator().run())
