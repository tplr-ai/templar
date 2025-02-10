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
import sys
import copy
import time
import random
import asyncio
import argparse
import threading
from contextlib import contextmanager
from time import perf_counter

# Third party
import torch
import numpy as np
import bittensor as bt
from torch.optim import SGD
from transformers import LlamaForCausalLM
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)

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
def timer(name: str, wandb_obj=None, step=None):
    start = perf_counter()
    yield
    duration = perf_counter() - start
    tplr.logger.debug(f"{name} took {duration:.2f}s")
    if wandb_obj and step is not None:
        wandb_obj.log({f"validator/{name}": duration}, step=step)


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
            T_0=10000,
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
        self.comms.fetch_commitments()

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
        self.binary_indicator_scores = torch.full(
            (self.metagraph.n,), 0.5, dtype=torch.float32
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

        # Initialize peers
        self.peers = []
        self.eval_peers = []

        # Track inactive peer scores
        self.inactive_scores = {}  # {uid: (last_active_window, last_score)}
        self.inactivity_slash_rate = 0.25  # 25% slash per window

    async def run(self):
        # Start background block listener
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(
            target=self.block_listener, args=(self.loop,), daemon=True
        ).start()
        # Load Peers
        if not self.config.peers:
            self.peers = self.comms.peers
            tplr.logger.info(f"Filtered gather peers with buckets: {self.peers}")
        else:
            self.peers = self.config.peers
        if self.uid not in self.peers:
            self.peers.append(self.uid)

        self.comms.commitments = self.comms.get_commitments_sync()
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
            loaded_global_step,
            loaded_optimizer,
            loaded_scheduler,
        ) = await self.comms.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            transformer=self.transformer,
            compressor=self.compressor,
            current_window=self.current_window,
            device=self.config.device,
            peers=self.peers,
            uid=self.uid,
            totalks=self.totalks,
        )
        if success:
            self.momentum = loaded_momentum
            self.global_step = loaded_global_step
            self.optimizer = loaded_optimizer
            self.scheduler = loaded_scheduler
            tplr.logger.info(
                f"Loaded checkpoint with global_step={self.global_step}, "
                f"optimizer_step={self.optimizer.state_dict()['state'].get(0, {}).get('step', 0)}, "
                f"scheduler_step={self.scheduler.last_epoch}"
            )
        else:
            tplr.logger.info("Starting from scratch")
            self.momentum = {
                n: torch.zeros_like(p) for n, p in self.model.named_parameters()
            }
            self.model.to(self.config.device)

        self.comms.start_commitment_fetcher()
        self.comms.start_background_tasks()

        while True:
            while self.sync_window >= (
                self.current_window - self.hparams.validator_offset
            ):
                tplr.logger.info(
                    f"Waiting for validator window offset, synced: {self.sync_window}, current:{self.current_window}, offset:{self.hparams.validator_offset}"
                )
                await asyncio.sleep(12)

            tplr.logger.info(
                f"Sync Window: {self.sync_window}, Scheduler epoch: {self.scheduler.last_epoch}, Global step: {self.global_step}"
            )

            # 2. Increment sync window and update peer lists
            window_start = tplr.T()
            self.sync_window += 1
            tplr.logger.info(
                f"Processing window: {self.sync_window} current: {self.current_window}"
            )

            peer_start = tplr.T()
            self.comms.update_peers_with_buckets()
            self.peers = self.comms.peers
            self.eval_peers = self.comms.eval_peers
            tplr.logger.info(
                f"{tplr.P(self.sync_window, tplr.T() - peer_start)} Updated peers - gather:{len(self.peers)}, eval:{len(self.eval_peers)}"
            )

            tplr.logger.info(f"Current gather peers: {self.peers}")
            tplr.logger.info(f"Current evaluation peers: {self.eval_peers}")

            newly_inactive = self.comms.inactive_peers
            current_window = self.sync_window

            # Process newly inactive peers
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
                if uid in self.eval_peers:
                    del self.inactive_scores[uid]
                    tplr.logger.info(f"UID {uid} became active again")
                    continue

                # Apply flat 25% penalty instead of exponential decay
                old_score = self.final_moving_avg_scores[uid].item()
                self.final_moving_avg_scores[uid] *= 0.75  # Apply flat 25% reduction
                new_score = self.final_moving_avg_scores[uid].item()

                tplr.logger.info(
                    f"UID {uid} penalized for inactivity: "
                    f"{old_score:.4f} -> {new_score:.4f}"
                )

                # Log slash metrics
                self.wandb.log(
                    {
                        f"validator/inactivity/{uid}/score_before": old_score,
                        f"validator/inactivity/{uid}/score_after": new_score,
                    },
                    step=self.global_step,
                )

            # 3. Gather gradients from peers
            gather_start = tplr.T()
            gather_result = await self.comms.gather(
                my_uid=self.uid,
                uids=self.peers,
                window=self.sync_window,
                key="gradient",
                timeout=60,
                device=self.config.device,
                local=False,
                totalks=self.totalks,
            )
            if gather_result is None:
                tplr.logger.warning(
                    f"No gradients gathered for window {self.sync_window}. Skipping this window."
                )
                self.global_step += 1
                continue
            tplr.logger.info(f"Skipped UIDs: {gather_result.skipped_uids}")
            tplr.logger.info(
                f"{tplr.P(self.sync_window, tplr.T() - gather_start)} Gathered gradients from peers"
            )

            # Add check for empty peers (evaluating all peer uids)
            if not self.peers:
                tplr.logger.warning(
                    f"No peers available for evaluation in window {self.sync_window}. Waiting for next window."
                )
                self.global_step += 1
                continue

            # 5. Save original model state for evaluation
            eval_start = tplr.T()
            # Sample a random subset of evaluation peers based on hparam uids_per_window
            evaluation_uids = random.sample(
                self.eval_peers, min(self.hparams.uids_per_window, len(self.eval_peers))
            )
            tplr.logger.info(f"Evaluating random subset of peers: {evaluation_uids}")
            for eval_uid in evaluation_uids:
                tplr.logger.info(f"Evaluating uid: {eval_uid}")

                eval_result = await self.comms.get(
                    uid=str(eval_uid),
                    window=self.sync_window,
                    key="gradient",
                    timeout=30,
                    local=False,
                    stale_retention=10,
                )

                scoring_start = tplr.T()
                if eval_result is not None and eval_result[0] is not None:
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

                    # Load pages_own exactly once from the dataset loader
                    local_pages = await tplr.r2_dataset.R2DatasetLoader.next_pages(
                        offset=self.sync_window,
                        n_pages=self.hparams.pages_per_window,
                        seed=eval_uid,
                    )

                    # Verify the pages_info from the miner matches our locally loaded pages.
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
                    loader_own = await tplr.r2_dataset.R2DatasetLoader.create(
                        batch_size=self.hparams.batch_size,
                        sequence_length=self.hparams.sequence_length,
                        pages_info=local_pages,
                        tokenizer=self.tokenizer,
                    )
                    tplr.logger.info(
                        f"{tplr.P(self.sync_window, tplr.T() - data_start)} Loaded evaluation data using pages: {[p[1] for p in local_pages]}"
                    )

                    state_dict, _ = eval_result
                    model_own_data_eval = copy.deepcopy(self.model)
                    # 8. Compute initial loss
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

                                p.data.sub_(
                                    grad.sign(), alpha=self.scheduler.get_last_lr()[0]
                                )
                    except Exception as e:
                        tplr.logger.error(
                            f"Failed to apply gradient for uid {uid}: {str(e)}"
                        )
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
                    pages_random = await tplr.r2_dataset.R2DatasetLoader.next_pages(
                        offset=self.sync_window,
                        n_pages=self.hparams.pages_per_window,
                        seed=random.randint(0, 10000),
                    )
                    loader_random = await tplr.r2_dataset.R2DatasetLoader.create(
                        batch_size=self.hparams.batch_size,
                        sequence_length=self.hparams.sequence_length,
                        pages_info=pages_random,
                        tokenizer=self.tokenizer,
                    )
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
                    self.gradient_scores[eval_uid] = (
                        (loss_before_random - loss_after_random) / loss_before_random
                        if loss_before_random > 0
                        else 0
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
                    # Calculate final score incorporating both metrics
                    final_score = (
                        self.gradient_scores[eval_uid]
                        * self.normalised_binary_moving_averages[eval_uid]
                    )
                    tplr.logger.debug(
                        f"Final Score : {self.final_moving_avg_scores[eval_uid]}"
                    )

                    # Ensure moving average score is non-negative
                    self.final_moving_avg_scores[eval_uid] = max(
                        self.hparams.final_score_ma_alpha
                        * self.final_moving_avg_scores[eval_uid]
                        + (1 - self.hparams.final_score_ma_alpha) * final_score,
                        0.0,
                    )
                    tplr.logger.debug(
                        f"Final Moving Average Score : {self.final_moving_avg_scores[eval_uid]}"
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
                    # TODO: move out
                    # 13. Log evaluation metrics once all evaluations are done
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
                        "validator/optimizer/learning_rate": self.scheduler.get_last_lr()[
                            0
                        ],
                        "validator/network/active_miners": len(
                            self.valid_score_indices
                        ),
                        "validator/gather/success_rate": gather_result.success_rate
                        * 100,  # Success percentage
                    }
                    self.wandb.log(evaluation_metrics, step=self.global_step)
                    tplr.logger.info(
                        f"{tplr.P(self.sync_window, tplr.T() - eval_start)} Completed evaluation"
                    )

                else:
                    tplr.logger.info(
                        f"No gradient received from UID {eval_uid}. Slashing moving average score by 50%."
                    )
                    # Reduce the moving average score by 50%
                    old_score = self.final_moving_avg_scores[
                        eval_uid
                    ].item()  # Get the actual value
                    self.final_moving_avg_scores[eval_uid] *= 0.5  # Apply 50% reduction
                    new_score = self.final_moving_avg_scores[
                        eval_uid
                    ].item()  # Get new value for logging
                    tplr.logger.info(
                        f"Reduced moving average score of UID {eval_uid} from {old_score:.4f} to {new_score:.4f} due to missing gradient."
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
                    for uid in self.evaluated_uids:
                        tplr.logger.info(f"UID {uid}:")
                        tplr.logger.info(
                            f"  - Moving avg score: {self.final_moving_avg_scores[uid]:.4f}"
                        )

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

            # Log scores and metrics for evaluated UIDs
            tplr.logger.info("Updated scores for evaluated UIDs:")
            for uid in self.evaluated_uids:
                tplr.logger.info(f"UID {uid}:")
                tplr.logger.info(f"  - Last score: {self.gradient_scores[uid]}")
                tplr.logger.info(
                    f"  - Binary indicator: {self.binary_indicator_scores[uid]:.4f}"
                )
                tplr.logger.info(
                    f"  - Binary moving avg: {self.binary_moving_averages[uid]:.4f}"
                )
                tplr.logger.info(
                    f"  - Normalised binary score: {self.normalised_binary_moving_averages[uid]:.4f}"
                )
                tplr.logger.info(
                    f"  - Final Moving avg score: {self.final_moving_avg_scores[uid]}"
                )
                tplr.logger.info(f"  - Weight: {self.weights[uid]:.4f}")

            # Log WandB metrics per UID
            for uid in sorted(self.evaluated_uids):
                self.wandb.log(
                    {
                        f"validator/gradient_scores/{uid}": self.gradient_scores[
                            uid
                        ].item(),
                        f"validator/binary_indicators/{uid}": self.binary_indicator_scores[
                            uid
                        ].item(),
                        f"validator/binary_moving_averages/{uid}": self.binary_moving_averages[
                            uid
                        ].item(),
                        f"validator/normalised_binary_scores/{uid}": self.normalised_binary_moving_averages[
                            uid
                        ].item(),
                        f"validator/final_moving_avg_scores/{uid}": self.final_moving_avg_scores[
                            uid
                        ].item(),
                        f"validator/weights/{uid}": self.weights[uid].item(),
                    },
                    step=self.global_step,
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

            # 15. Create checkpoints periodically
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
                }
                asyncio.create_task(
                    self.comms.put(
                        state_dict=checkpoint_data,
                        uid=str(self.uid),
                        window=self.current_window,
                        key="checkpoint",
                        global_step=self.global_step,
                        local=False,
                    )
                )

            # 16. Now, merge the gathered gradients into the model AFTER finishing evaluation
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
            # Log total window time and metrics
            tplr.logger.info(
                f"{tplr.P(self.sync_window, tplr.T() - window_start)} Completed window iteration"
            )

            self.wandb.log(
                {
                    "validator/timing/window_total": tplr.T() - window_start,
                    "validator/timing/peer_update": tplr.T() - peer_start,
                    "validator/timing/gather": tplr.T() - gather_start,
                    "validator/timing/evaluation": tplr.T() - eval_start,
                    "validator/timing/model_update": tplr.T() - update_start,
                },
                step=self.global_step,
            )

            # 18. Increment global step
            self.global_step += 1

    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event["header"]["number"])  # type : ignore
            new_window = int(self.current_block / self.hparams.blocks_per_window)
            if new_window != self.current_window:
                self.current_window = new_window
                self.comms.current_window = (
                    self.current_window
                )  # Synchronize comms current_window

        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(
                    handler
                )
                break
            except Exception:
                time.sleep(1)


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


if __name__ == "__main__":
    asyncio.run(Validator().run())
