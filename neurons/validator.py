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
        # Initialize final score history (for sliding-window averaging)
        self.final_score_history = defaultdict(list)

        # Initialize peer related attributes
        self.next_peers: tplr.comms.PeerArray | None = None
        self.peers_update_window = -1

    async def run(self):
        # Start background block listener
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(
            target=self.block_listener, args=(self.loop,), daemon=True
        ).start()

        all_uids = list(range(len(self.metagraph.S)))
        # Use config peers if provided
        self.comms.peers = np.array([uid for uid in all_uids if uid not in [0, 1]])

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

            peer_start = tplr.T()
            tplr.logger.info(f"Current gather peers: {self.comms.peers}")

            # Log the time window we're using
            tplr.logger.info(f"Using time window for gather: {time_min} to {time_max}")
            tplr.logger.info(f"We are using peers {self.comms.peers}")

            # Refresh peers explicitly before starting gather to avoid missing updated active peers.
            tplr.logger.info("Refreshing eval peers before gather task in validator...")

            tplr.logger.info(f"Validator gather peers: {self.comms.peers}")

            gather_start = tplr.T()
            skipped_uids: list[int] = []
            success_rate = 0.0
            gather_result = None
            # TODO: make load from disc
            gather_result = await self.comms.gather(
                my_uid=self.uid,
                uids=self.comms.peers,
                window=self.sync_window,
                key="gradient",
                timeout=35,
                device=self.config.device,
                local=False,
                totalks=self.totalks,
            )

            if gather_result is None:
                tplr.logger.error(
                    "Failed to gather gradients from peers. Waiting for next window."
                )
                self.global_step += 1
                continue
            skipped_uids = gather_result.skipped_uids
            success_rate = gather_result.success_rate
            gather_time = tplr.T() - gather_start

            tplr.logger.info(f"Skipped UIDs: {skipped_uids}")

            # 5. Save original model state for evaluation
            eval_start = tplr.T()

            # 6. Select peers to evaluate
            evaluation_uids = self.comms.peers
            tplr.logger.info(f"Evaluating random subset of peers: {evaluation_uids}")

            avg_loss_before_per_batch_own = 0.0
            avg_loss_after_per_batch_own = 0.0
            avg_loss_before_per_batch_random = 0.0
            avg_loss_after_per_batch_random = 0.0
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
                    avg_loss_before_per_batch_own += self.loss_before_per_batch_own
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
                    except Exception:
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
                    avg_loss_after_per_batch_own += self.loss_after_per_batch_own
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
                    avg_loss_before_per_batch_random += (
                        self.loss_before_per_batch_random
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
                    avg_loss_after_per_batch_random += self.loss_after_per_batch_random
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

                    # Your existing final_score calculation with sync_score added
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
                self.apply_gathered_gradients(gather_result=gather_result)
            else:
                tplr.logger.warning("No gradients to apply.")
                self.scheduler.step()
                torch.cuda.empty_cache()

            tplr.logger.info(
                f"{tplr.P(self.sync_window, tplr.T() - update_start)} Updated model"
            )

            # Log total window time and metrics
            tplr.logger.info(
                f"{tplr.P(self.sync_window, tplr.T() - window_start)} Completed window iteration"
            )

            avg_loss_before_per_batch_own /= len(evaluation_uids)
            avg_loss_after_per_batch_own /= len(evaluation_uids)
            avg_loss_before_per_batch_random /= len(evaluation_uids)
            avg_loss_after_per_batch_random /= len(evaluation_uids)

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
            )
            tplr.logger.info("Finished metrics logging call for validator")

            # 18. Increment global step
            self.global_step += 1

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
