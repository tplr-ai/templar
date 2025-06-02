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
import json
import logging  # ← NEW
import os
import random
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, cast

import bittensor as bt
import bittensor.core.subtensor as bt_subtensor
import numpy as np
import torch
import torch.optim as optim
import uvloop
import websockets
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from transformers import LlamaForCausalLM

# Local
import tplr
import tplr.distrib as distrib  # Using the provided distrib.py content
from tplr.distrib import (
    ddp_init,
    # rank_world is already available as distrib.rank_world
)

CPU_COUNT = os.cpu_count() or 4
CPU_MAX_CONNECTIONS = min(100, max(30, CPU_COUNT * 4))


class Miner:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description="Miner script")
        parser.add_argument(
            "--netuid", type=int, default=268, help="Bittensor network UID."
        )
        parser.add_argument(
            "--project", type=str, default="templar", help="Wandb project."
        )
        parser.add_argument(
            "--actual-batch-size",
            type=int,
            default=None,
            help="Override the batch size defined in hparams.",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            help="Device to use for training (will be rank-specific).",
        )
        parser.add_argument(
            "--debug", action="store_true", help="Enable debug logging."
        )
        parser.add_argument(
            "--trace", action="store_true", help="Enable trace logging."
        )
        parser.add_argument(
            "--store-gathers",
            action="store_true",
            help="Store gathered gradients in R2.",
        )
        parser.add_argument(
            "--test",
            action="store_true",
            help="Test mode - use all peers without filtering.",
        )
        parser.add_argument(
            "--local",
            action="store_true",
            help="Local run - use toy model, small enough for a laptop.",
        )
        parser.add_argument(
            "--gpus",
            type=int,
            default=None,
            help="Sanity check against WORLD_SIZE (use torchrun to spawn).",
        )
        # Add any other peer-related configs if necessary
        parser.add_argument(
            "--peers", nargs="*", type=int, help="Fixed list of peer UIDs to use."
        )

        bt.subtensor.add_args(parser)
        bt.wallet.add_args(parser)

        config = bt.config(parser)  # All ranks parse config initially
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
        self.config = Miner.config()  # All ranks parse CLI

        if self.config.actual_batch_size is not None:
            tplr.logger.info(
                f"Overriding hparams batch size: {self.hparams.batch_size} -> {self.config.actual_batch_size}"
            )
            self.hparams.batch_size = self.config.actual_batch_size

        # Init bittensor objects
        # WORLD_SIZE and LOCAL_RANK should be set by torchrun
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        # LOCAL_RANK is the rank of the process on the current node.
        # RANK is the global rank across all nodes. For single-node, RANK == LOCAL_RANK.
        # ddp_init will use RANK.
        local_rank = int(os.getenv("LOCAL_RANK", "0"))  # This is the device index

        if self.config.gpus is not None:
            assert self.config.gpus == world_size, (
                f"--gpus={self.config.gpus} but WORLD_SIZE={world_size}"
            )

        # Pass local_rank for torch.cuda.set_device, global_rank for dist.init_process_group
        ddp_init(local_rank, world_size)  # Initializes DDP, sets device to local_rank

        self.rank = distrib.get_rank()  # Global rank
        self.world_size = distrib.get_world_size()
        self.device = (
            f"cuda:{local_rank}"
            if torch.cuda.is_available() and self.config.device == "cuda"
            else "cpu"
        )

        # Configure logging: Rank 0 full, others warning (after DDP init)
        if not distrib.is_rank0():
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
        if distrib.is_rank0():
            self.hparams = tplr.load_hparams(use_local_run_hparams=self.config.local)
        else:
            self.hparams = None  # Placeholder for non-rank0
        self.hparams = distrib.broadcast_object(self.hparams, src=0)
        tplr.logger.info(f"[Rank {self.rank}] HParams synchronized.")

        # Bittensor objects
        # Wallet and Subtensor can be initialized by all ranks
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)

        # Metagraph: fetch on every rank ( cheap ) and only broadcast the UID
        self.metagraph = self.subtensor.metagraph(cast(int, self.config.netuid))

        # Rank-0 validates registration
        if (
            distrib.is_rank0()
            and self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys
        ):
            tplr.logger.error(
                f"[Rank 0] Wallet {self.wallet} not registered on subnet {self.metagraph.netuid}"
            )
            distrib.broadcast_object({"error": "Wallet not registered"}, src=0)
            sys.exit(1)

        # Compute / broadcast UID so that all ranks are consistent
        uid_local = (
            self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            if self.wallet.hotkey.ss58_address in self.metagraph.hotkeys
            else -1
        )
        self.uid = distrib.broadcast_object(uid_local, src=0)

        # React to broadcasted error from rank-0 (if any)
        if self.uid == -1:
            tplr.logger.error(f"[Rank {self.rank}] Wallet not registered – exiting.")
            sys.exit(1)

        tplr.logger.info(
            f"[Rank {self.rank}] Metagraph and UID {self.uid} synchronized."
        )

        # --------------------------------------------------------
        # 2.  Model + DDP wrapper
        # --------------------------------------------------------
        # All ranks create the model instance. DDP will synchronize rank 0's weights.
        self.model = LlamaForCausalLM(self.hparams.model_config).to(self.device)
        tplr.logger.info(f"[Rank {self.rank}] Model created on {self.device}.")

        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[local_rank] if self.device.startswith("cuda") else None,
                output_device=local_rank if self.device.startswith("cuda") else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            tplr.logger.info(f"[Rank {self.rank}] Model wrapped with DDP.")

        # Tokenizer should be loaded consistently. Assuming hparams provide necessary info.
        self.tokenizer = self.hparams.tokenizer  # Assuming this is lightweight

        # Init compression (all ranks, based on common hparams and model structure)
        # Ensure transformer uses the correct model (self.model.module if DDP)
        model_to_transform = self.model.module if self.world_size > 1 else self.model
        self.transformer = tplr.compress.TransformDCT(
            model_to_transform, target_chunk=self.hparams.target_chunk
        )
        self.compressor = tplr.compress.CompressDCT(
            use_quantization=True,
            quantization_bins=self.hparams.quantization_bins,
            quantization_range=self.hparams.quantization_range,
        )

        # Optimizer (all ranks)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.hparams.learning_rate
        )

        # Momentum, xshapes, totalks (all ranks, derived from model parameters)
        self.momentum: Dict[str, torch.Tensor] = {}
        self.xshapes: Dict[str, Any] = {}
        self.totalks: Dict[str, Any] = {}
        self.quant_params: Dict[str, Any] = {}

        # Use self.model.module.named_parameters() if DDP wrapped, else self.model.named_parameters()
        named_params_iter = (
            self.model.module if self.world_size > 1 else self.model
        ).named_parameters()  # type: ignore
        for n, p in named_params_iter:
            self.momentum[n] = torch.zeros_like(
                p, device=self.device
            )  # Ensure momentum is on correct device
            # Encode with model.module if DDP
            encoded_param = self.transformer.encode(self.momentum[n])
            _, _, xshape, totalk, quant_params = self.compressor.compress(
                encoded_param, self.hparams.topk_compression
            )
            self.xshapes[n] = xshape
            self.totalks[n] = totalk
            self.quant_params[n] = quant_params

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
            f"[Rank {self.rank}] Miner code_version={tplr.__version__} "
            f"checkpoint_init_flag={self.bootstrap_version or '<none>'}"
        )

        # Init comms (all ranks, config might be used by rank 0 for setup)
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

        if distrib.is_rank0():  # Bucket operations by rank 0
            self.bucket = self.comms.get_own_bucket("gradients", "read")
            self.comms.try_commit(self.wallet, self.bucket)
        distrib.barrier()  # Ensure rank 0 completes before others might proceed if dependent

        # Init state params (all ranks)
        self.stop_event = (
            asyncio.Event()
        )  # For async tasks, typically managed by rank 0 actions
        self.current_block = 0  # Will be updated by listener or broadcast
        self.current_window = 0  # Will be updated
        # start_window will be determined and synced in run()
        self.global_step = 0
        # self.comms.current_window needs to be updated based on synced current_window
        self.step_counter = 0
        self.window_step = 0
        self.total_tokens_processed = 0
        self.batch_times: List[float] = []
        self.gradient_iteration_counter = 0

        # --------------------------------------------------------
        # 3. Logging / WANDB only on rank-0
        # --------------------------------------------------------
        self.wandb = None
        self.metrics_logger = None
        if distrib.is_rank0():
            self.wandb = tplr.initialize_wandb(
                run_prefix="M",
                uid=self.uid,
                config=self.config,
                group="miner",
                job_type="mining",
            )

            if self.wandb:
                tplr.logger.info(
                    f"[Rank 0 W&B] Run initialized: {self.wandb.url or 'local run'}"
                )
            else:
                tplr.logger.warning("[Rank 0 W&B] WandB initialization failed.")

            self.metrics_logger = tplr.metrics.MetricsLogger(
                prefix="M",
                uid=self.uid,
                config=self.config,
                role="miner",
                group="miner",
                job_type="mining",
            )

        distrib.barrier()  # Ensure rank 0 logging is set up before proceeding

        self.next_peers: Optional[tplr.comms.PeerArray] = (
            None  # Will be determined by rank 0 and broadcast
        )
        self.peers_update_window = -1
        tplr.logger.info(f"[Rank {self.rank}] Initialization complete.")

        # let EVERY rank chat at INFO for the dataset loader
        from tplr.logging import enable_all_rank_logging

        enable_all_rank_logging("tplr.r2_dataset", level=logging.INFO)

    async def _load_checkpoint_and_sync(self, start_window: int):
        """Rank 0 loads checkpoint, broadcasts states to all ranks."""
        checkpoint_window_buffer = 5
        has_new_checkpoint_opportunity = (
            self.global_step
            >= self.hparams.checkpoint_frequency + checkpoint_window_buffer
        )

        loaded_model_state_dict = None
        loaded_optimizer_state_dict = None
        loaded_scheduler_state_dict = None
        loaded_checkpoint_window_val = 0
        load_success = False

        if distrib.is_rank0():
            tplr.logger.info(
                f"[Rank 0] Attempting to load checkpoint for current_window: {self.current_window}"
            )
            model_to_load = self.model.module if self.world_size > 1 else self.model
            (
                success,
                _loaded_checkpoint_window,
                _loaded_optimizer,
                _loaded_scheduler,
            ) = await self.comms.load_checkpoint(
                model=model_to_load,  # Pass the actual model module for rank 0 to load into
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                current_window=self.current_window,
                device="cpu",  # Load to CPU first for broadcasting
                init_version=tplr.__version__
                if has_new_checkpoint_opportunity
                else self.bootstrap_version,
            )
            if success:
                load_success = True
                loaded_optimizer_state_dict = (
                    _loaded_optimizer.state_dict() if _loaded_optimizer else None
                )
                loaded_scheduler_state_dict = (
                    _loaded_scheduler.state_dict() if _loaded_scheduler else None
                )
                loaded_checkpoint_window_val = _loaded_checkpoint_window
                tplr.logger.info(
                    f"[Rank 0] Checkpoint loaded successfully. Window: {loaded_checkpoint_window_val}"
                )
            else:
                tplr.logger.info("[Rank 0] No checkpoint found or load failed.")

        # Broadcast results
        # Use broadcast_object for simplicity, for very large states, consider dist.broadcast for tensors
        load_results_list = [
            load_success,
            loaded_optimizer_state_dict,
            loaded_scheduler_state_dict,
            loaded_checkpoint_window_val,
        ]

        # All ranks participate in broadcast_object_list
        if distrib.is_rank0():
            distrib.broadcast_object(load_results_list, src=0)
        else:
            # Create placeholders for receiving objects
            placeholders = [None] * len(load_results_list)
            distrib.broadcast_object(placeholders, src=0)
            load_results_list = placeholders

        (
            load_success,
            loaded_optimizer_state_dict,
            loaded_scheduler_state_dict,
            loaded_checkpoint_window_val,
        ) = load_results_list

        if load_success:
            tplr.logger.info(
                f"[Rank {self.rank}] Received checkpoint data. Applying..."
            )
            # All ranks load the broadcasted states
            model_to_load_into = (
                self.model.module if self.world_size > 1 else self.model
            )
            if loaded_model_state_dict:
                # Ensure keys match, especially if DDP wrapper adds 'module.' prefix
                if self.world_size > 1 and not all(
                    k.startswith("module.") for k in loaded_model_state_dict.keys()
                ):
                    # If model was saved as model.module.state_dict(), it's fine.
                    # If saved as DDP(model).state_dict(), it might have 'module.' prefix. Adjust if necessary.
                    # Here, we assume load_checkpoint returns a state_dict compatible with model_to_load_into
                    pass  # Add key adjustment logic if needed based on how checkpoint is saved
                model_to_load_into.load_state_dict(
                    {k: v.to(self.device) for k, v in loaded_model_state_dict.items()}
                )

            if loaded_optimizer_state_dict:
                self.optimizer.load_state_dict(
                    {
                        k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                        for k, v in loaded_optimizer_state_dict.items()
                    }
                )  # Optimizer states might need device transfer

            if loaded_scheduler_state_dict:
                self.scheduler.load_state_dict(loaded_scheduler_state_dict)

            tplr.logger.info(
                f"[Rank {self.rank}] Loaded checkpoint. Global Step: {self.global_step}, "
                f"Optimizer Step: {self.optimizer.state_dict()['state'].get(0, {}).get('step', 0)}, "
                f"Scheduler Step: {self.scheduler.last_epoch}"
            )

            # Catchup logic (coordinated or rank 0 leads)
            if (
                loaded_checkpoint_window_val < self.current_window
                and self.global_step > checkpoint_window_buffer
            ):
                tplr.logger.info(
                    f"[Rank {self.rank}] Checkpoint window {loaded_checkpoint_window_val} is behind current {self.current_window}. Catching up..."
                )
                # Ensure catchup_with_aggregation_server is DDP aware or its effects are broadcast
                await tplr.neurons.catchup_with_aggregation_server(
                    self, max(loaded_checkpoint_window_val, start_window)
                )  # self here refers to the Miner instance
            else:
                tplr.logger.info(
                    f"[Rank {self.rank}] Checkpoint up-to-date or no catchup needed."
                )
        else:
            tplr.logger.info(
                f"[Rank {self.rank}] No checkpoint found or load failed. Initializing model from scratch or proceeding with current state."
            )
            # Ensure model is on the correct device (already done in __init__)
            # self.model.to(self.device) # Redundant if already done and DDP handles sync

            # Re-initialize momentum if not loaded
            self.momentum = {}
            named_params_iter = (
                self.model.module if self.world_size > 1 else self.model
            ).named_parameters()
            for n, p in named_params_iter:
                self.momentum[n] = torch.zeros_like(p, device=self.device)

            tplr.logger.info(
                f"[Rank {self.rank}] Starting catchup from start window {start_window} to current {self.current_window}..."
            )
            await tplr.neurons.catchup_with_aggregation_server(self, start_window)

        distrib.barrier()  # Ensure all ranks are done with checkpoint loading/catchup

    async def run(self):
        # Start background block listener (all ranks run their own listener for now)
        # For strict window synchronization, rank 0 could listen and broadcast window updates.
        self.loop = asyncio.get_running_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
        self.loop.set_default_executor(self.executor)

        self.listener = threading.Thread(
            target=self.block_listener, args=(self.loop,), daemon=True
        )
        self.listener.start()

        # Commitments: Rank 0 gets, broadcasts
        if distrib.is_rank0():
            initial_commitments = await self.comms.get_commitments()
            tplr.logger.info("[Rank 0] Loaded initial commitments.")
        else:
            initial_commitments = None
        self.comms.commitments = distrib.broadcast_object(initial_commitments, src=0)
        tplr.logger.info(f"[Rank {self.rank}] Initial commitments synchronized.")

        # --------------------------------------------------------
        # Determine start window – ask highest-stake validator once
        # --------------------------------------------------------
        if distrib.is_rank0():
            start_window = await self.comms.get_start_window()
            if start_window is None:
                raise RuntimeError(
                    "Could not find a valid start window. This should not be possible."
                )
            tplr.logger.info(f"[Rank 0] Using start_window: {start_window}")
        else:
            start_window = None  # placeholder
        start_window = distrib.broadcast_object(start_window, src=0)
        self.start_window = start_window  # Store it

        # Sync current_block and current_window initially after start_window is known
        # Block listener might take time to get first block. Rank 0 can get current block & broadcast.
        if distrib.is_rank0():
            self.current_block = self.subtensor.block  # Get latest block on rank 0
            self.current_window = int(
                self.current_block / self.hparams.blocks_per_window
            )
            block_window_payload = {
                "block": self.current_block,
                "window": self.current_window,
            }
        else:
            block_window_payload = None

        synced_bw_payload = distrib.broadcast_object(block_window_payload, src=0)
        self.current_block = synced_bw_payload["block"]
        self.current_window = synced_bw_payload["window"]
        self.comms.current_window = self.current_window  # Update comms attribute

        self.global_step = self.current_window - self.start_window
        tplr.logger.info(
            f"[Rank {self.rank}] Starting at Global Step: {self.global_step}, Current Window: {self.current_window}"
        )

        # # Load checkpoint and synchronize across ranks
        await self._load_checkpoint_and_sync(self.start_window)

        if tplr.distrib.is_rank0():  # Commitment fetcher only on rank 0
            self.comms.start_commitment_fetcher()

        distrib.barrier()

        # Main loop
        while True:
            window_start_time = tplr.T()  # Timer T

            # Sync current window at the beginning of each loop iteration from Rank 0 via listener
            # This ensures all ranks operate on the same window deterministically
            if distrib.is_rank0():
                # Rank 0's listener updates self.current_window. Broadcast it.
                window_to_process_payload = {"window": self.current_window}
            else:
                window_to_process_payload = None

            synced_window_payload = distrib.broadcast_object(
                window_to_process_payload, src=0
            )
            step_window = synced_window_payload["window"]

            self.global_step = (
                step_window - self.start_window
            )  # Update global_step based on synced window
            tplr.logger.info(
                f"\n{'-' * 40} [Rank {self.rank}] Window: {step_window} (Global Step: {self.global_step}) {'-' * 40}"
            )

            peer_update_start_time = tplr.T()
            # Peer update: Rank 0 decides, broadcasts
            if distrib.is_rank0():
                await tplr.neurons.update_peers(
                    instance=self, window=step_window, peer_start=peer_update_start_time
                )
                # self.comms.peers should be updated by update_peers
                peers_to_broadcast = self.comms.peers
            else:
                peers_to_broadcast = None
            self.comms.peers = distrib.broadcast_object(peers_to_broadcast, src=0)
            if self.comms.peers is None:
                self.comms.peers = []  # Ensure it's a list
            tplr.logger.info(
                f"[Rank {self.rank}] {tplr.P(step_window, tplr.T() - peer_update_start_time)} Peers for window {step_window} synchronized: {self.comms.peers}"
            )

            data_load_start_time = tplr.T()
            # Data loading: Each rank loads its own shard (DDP standard)
            # rank_world() from distrib is fine
            current_rank, world_size_for_data = distrib.rank_world()

            base = (
                self.hparams.pages_per_window // world_size_for_data
            )  # minimum pages / rank
            remainder = (
                self.hparams.pages_per_window % world_size_for_data
            )  # first "remainder" ranks get +1

            pages_per_rank = base + (current_rank < remainder)

            # guarantee ≥1 page if world_size > pages_per_window (will oversample in that edge-case)
            if pages_per_rank == 0:
                pages_per_rank = 1

            pages = await tplr.r2_dataset.R2DatasetLoader.next_pages(
                offset=step_window * self.hparams.pages_per_window,
                n_pages=pages_per_rank,
                seed=self.uid,
                rank=current_rank,
                world_size=world_size_for_data,
            )

            debug_pages = distrib.all_gather_object(
                pages
            )  # list[ list[tuple] ] from every rank
            debug_pages_flat = [tuple(p) for sub in debug_pages for p in sub]
            if distrib.is_rank0():
                dups = set(x for x in debug_pages_flat if debug_pages_flat.count(x) > 1)
                if dups:
                    tplr.logger.warning(
                        f"Page overlap across ranks: {sorted(dups)[:10]} …"
                    )
                else:
                    tplr.logger.info(
                        f"Page split OK • total={len(debug_pages_flat)} • per-rank={[len(x) for x in debug_pages]}"
                    )

            loader = await tplr.r2_dataset.R2DatasetLoader.create(
                batch_size=self.hparams.batch_size,
                sequence_length=self.hparams.sequence_length,
                pages_info=pages,
                tokenizer=self.tokenizer,
            )
            tplr.logger.info(
                f"{tplr.P(step_window, tplr.T() - data_load_start_time)} [Rank {self.rank}] Loaded training data. "
                f"Pages: {[p[1] for p in pages] if pages else 'No pages'}"
            )

            training_start_time = tplr.T()
            tplr.logger.info(
                f"[Rank {self.rank}] Start accumulating gradients for window {step_window}..."
            )

            # Zero gradients before training loop for the window
            self.optimizer.zero_grad()

            total_loss_this_window = 0.0
            num_batches_this_window = 0
            tokens_this_window = 0

            self.model.train()  # Set model to train mode

            for i, batch in enumerate(loader):
                input_ids = torch.tensor(batch, dtype=torch.long).to(self.device)
                tokens_this_batch = input_ids.numel()
                tokens_this_window += tokens_this_batch
                labels = input_ids.clone()
                labels = torch.where(
                    labels == self.tokenizer.pad_token_id, -100, labels
                )

                with autocast(
                    device_type=self.device.split(":")[0],
                    dtype=torch.bfloat16,
                    enabled=self.device.startswith("cuda"),
                ):
                    outputs = self.model(input_ids=input_ids, labels=labels)

                # full loss, no gradient accumulation
                loss = outputs.loss
                total_loss_this_window += loss.item()

                if num_batches_this_window > 0:
                    tplr.logger.info(
                        f"Normalizing gradients by {num_batches_this_window} accumulation steps"
                    )
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.div_(num_batches_this_window)

                loss.backward()  # ← DDP syncs grads right here
                num_batches_this_window += 1

                tplr.logger.info(
                    f"[Rank {self.rank}] Loss: {outputs.loss.item()} [Window {step_window}, Batch {i + 1}]"
                )

                tplr.logger.info(
                    f"{tplr.P(step_window, tplr.T() - training_start_time)} "
                    f"[Rank {self.rank}] Completed local gradient accumulation. "
                    f"Batches: {num_batches_this_window}"
                )

            if distrib.is_rank0():  # only rank-0 polls chain
                while self.current_window == step_window:
                    await asyncio.sleep(0.1)
                rolled_window = self.current_window  # new window number
            else:
                rolled_window = None  # placeholder

            # rank-0 tells everybody the new window; others just block here
            rolled_window = distrib.broadcast_object(rolled_window, src=0)
            self.current_window = rolled_window  # keep local view in-sync
            distrib.barrier()  # all ranks aligned
            # ----------------------------------------------------------------------

            compression_start_time = tplr.T()
            put_completion_duration = 0.0
            upload_data_size_bytes = 0

            if distrib.is_rank0():
                # tplr.prepare_gradient_dict should take this model_ref_for_grads and extract .grad
                gradient_to_put, _, _, _ = tplr.prepare_gradient_dict(
                    self, debug_pages_flat, step_window
                )  # type: ignore
                tplr.logger.info(
                    f"{tplr.P(step_window, tplr.T() - compression_start_time)} [Rank 0] Compressed local DDP-synced gradients for upload."
                )

                processed_state_dict_to_put = {}
                for (
                    k,
                    v_tensor,
                ) in (
                    gradient_to_put.items()
                ):  # Assuming gradient_to_put is a dict of tensors
                    if isinstance(v_tensor, torch.Tensor):
                        processed_state_dict_to_put[k] = (
                            v_tensor.cpu()
                        )  # Move to CPU for upload
                        upload_data_size_bytes += (
                            v_tensor.element_size() * v_tensor.nelement()
                        )
                    else:  # Handle other types if any
                        processed_state_dict_to_put[k] = v_tensor

                tplr.logger.info(
                    f"[Rank 0] Uploading {upload_data_size_bytes / (1024 * 1024):.2f} MB of own state for UID: {self.uid}, Window: {step_window}"
                )
                _put_start_time = tplr.T()
                await self.comms.put(
                    state_dict=processed_state_dict_to_put,
                    uid=str(self.uid),
                    window=step_window,
                    key="gradient",
                    global_step=self.global_step,
                    local=self.config.local,  # Use config.local
                    stale_retention=100,
                )
                put_completion_duration = tplr.T() - _put_start_time
                tplr.logger.info(
                    f"[Rank 0] Put task completed in {put_completion_duration:.2f}s."
                )

            distrib.barrier()  # Ensure rank 0 put completes if others depend or for timing consistency

            # Now, self.total_tokens_processed is consistent across all ranks and holds the global sum.

            # Update global counters (all ranks, consistently)
            # self.global_step is already updated based on step_window

            # # Checkpoint Saving (Rank 0 Only)
            # if distrib.is_rank0() and (self.global_step % self.hparams.checkpoint_frequency == 0):
            #     tplr.logger.info(f"[Rank 0] Creating checkpoint at global_step {self.global_step}")
            #     # Ensure momentum is current if it's part of SGD and not explicitly self.momentum
            #     # If self.momentum is custom, it should be passed.
            #     # Optimizer state will contain its own momentum if applicable.
            #     asyncio.create_task(
            #         self.comms.save_checkpoint(
            #             model=(self.model.module if self.world_size > 1 else self.model), # Save the underlying model
            #             optimizer=self.optimizer,
            #             scheduler=self.scheduler,
            #             momentum=self.momentum, # Pass the custom momentum dict
            #             global_step=self.global_step,
            #             sync_window=step_window, # Use the window that was just processed
            #             start_window=self.start_window
            #         )
            #     )

            # Timestamp for Gather: Rank 0 queries, broadcasts
            time_min_gather, time_max_gather = None, None
            if distrib.is_rank0():
                sync_block = self.current_window * self.hparams.blocks_per_window
                retries = 0
                delay = 1
                max_retries = 5
                max_delay = 60
                while True:
                    try:
                        response = self.subtensor.query_module(
                            "Timestamp", "Now", block=sync_block
                        )
                        if response is None or not isinstance(
                            response, bt_subtensor.ScaleObj
                        ):
                            raise ValueError(
                                f"Could not query timestamp for {sync_block}"
                            )
                        ts_value = (
                            cast(int, response.value) / 1000
                        )  # convert milliseconds to seconds
                        break
                    except Exception as e:
                        tplr.logger.error(
                            f"[Rank 0] Failed to query timestamp for block {sync_block}: {str(e)}. Retry {retries + 1}/{max_retries}"
                        )
                        retries += 1
                        if retries > max_retries:
                            tplr.logger.error(
                                "[Rank 0] Exceeded maximum retries for timestamp query."
                            )
                            raise e
                        time.sleep(delay)
                        delay = min(delay * 2, max_delay)

                time_min_gather = datetime.fromtimestamp(ts_value, tz=timezone.utc)
                time_max_gather = time_min_gather + timedelta(
                    seconds=self.hparams.time_window_delta_seconds
                )
                tplr.logger.info(
                    f"[Rank 0] Using time window for gather: {time_min_gather} to {time_max_gather}"
                )

            time_window_payload = distrib.broadcast_object(
                [time_min_gather, time_max_gather], src=0
            )
            time_min_gather, time_max_gather = (
                time_window_payload[0],
                time_window_payload[1],
            )
            # All ranks now have the same time_min_gather, time_max_gather (or None if rank 0 failed)

            # Gather Gradients: Rank 0 gathers, broadcasts the state_dict for application
            gather_start_time = tplr.T()
            gathered_state_dict_for_apply = None
            gather_success_rate = 0.0
            skipped_uids_gather: List[int] = []

            if distrib.is_rank0():
                tplr.logger.info(
                    f"[Rank 0] Starting gather task with peers: {self.comms.peers}"
                )
                # Ensure self.totalks and self.xshapes are based on the non-DDP model structure if needed by comms.gather
                # They were initialized based on self.model before DDP wrapping, which is usually fine.
                gather_result = await self.comms.gather(
                    my_uid=self.uid,
                    uids=self.comms.peers,
                    window=step_window,
                    key="gradient",
                    timeout=35,
                    device="cpu",
                    local=self.config.local,
                    stale_retention=100,
                    totalks=self.totalks,  # Pass xshapes here if gather needs it
                    time_min=time_min_gather,
                    time_max=time_max_gather,
                )
                if gather_result and gather_result.state_dict:
                    gathered_state_dict_for_apply = (
                        gather_result.state_dict
                    )  # This is the object with .idxs_key, .vals_key attrs
                    gather_success_rate = gather_result.success_rate * 100
                    skipped_uids_gather = gather_result.skipped_uids
                    tplr.logger.info(
                        f"[Rank 0] Gather task completed. Success rate: {gather_success_rate:.2f}%"
                    )
                else:
                    tplr.logger.warning(
                        "[Rank 0] Gather task failed or returned no state_dict."
                    )

            # Broadcast the gathered state_dict and related info
            # This object might be complex, ensure it's picklable
            gathered_payload = distrib.broadcast_object(
                [
                    gathered_state_dict_for_apply,
                    gather_success_rate,
                    skipped_uids_gather,
                ],
                src=0,
            )
            gathered_state_dict_for_apply, gather_success_rate, skipped_uids_gather = (
                gathered_payload
            )

            gather_duration = tplr.T() - gather_start_time
            tplr.logger.info(
                f"[Rank {self.rank}] Gather synchronization complete. Duration: {gather_duration:.2f}s"
            )

            # Apply Gathered Gradients and Optimizer Step
            model_update_start_time = tplr.T()
            model_ref_for_update = (
                self.model.module if self.world_size > 1 else self.model
            )

            # CRITICAL: The original code overwrites p.grad with gathered gradients.
            # This means DDP-synced local gradients are discarded unless they were part of the "put"
            # and then re-incorporated via the "gather" mechanism (e.g. from an aggregation server).
            # We need to zero_grad() before setting new grads from gathered results.
            self.optimizer.zero_grad()  # Clear any existing grads (e.g. DDP-synced local grads)

            if gathered_state_dict_for_apply:
                num_params_updated = 0
                for n, p in model_ref_for_update.named_parameters():
                    if not p.requires_grad:
                        continue

                    idxs_key = n + "idxs"
                    vals_key = n + "vals"
                    quant_key = n + "quant_params"

                    # gathered_state_dict_for_apply is the custom object, not a PyTorch state_dict
                    idxs = getattr(gathered_state_dict_for_apply, idxs_key, None)
                    vals = getattr(gathered_state_dict_for_apply, vals_key, None)
                    quant_params = getattr(
                        gathered_state_dict_for_apply, quant_key, None
                    )

                    if idxs is not None and vals is not None:
                        if not isinstance(idxs, (list, tuple)):
                            idxs = [idxs]
                        if not isinstance(vals, (list, tuple)):
                            vals = [vals]

                        # Decompress on the correct device
                        # Ensure p is on self.device. self.transformer/compressor should handle device placement or expect inputs on device.
                        # Ensure xshapes and totalks are correctly sourced (from __init__)
                        new_grad_val = self.transformer.decode(
                            self.compressor.batch_decompress(
                                p.to(
                                    self.device
                                ),  # Ensure p is on device for decompression shape reference
                                [
                                    i.to(self.device) for i in idxs
                                ],  # Ensure idxs/vals tensors are on device
                                [v.to(self.device) for v in vals],
                                self.xshapes[n],
                                self.totalks[n],
                                quant_params,
                            )
                        )

                        # Assign to p.grad
                        p.grad = (
                            new_grad_val.sign_()
                        )  # Apply sign operation as in original
                        num_params_updated += 1
                    else:
                        # If a param has no gathered gradient, its .grad will be None (due to optimizer.zero_grad())
                        # This means it won't be updated by optimizer.step(). This might be intended.
                        tplr.logger.debug(
                            f"[Rank {self.rank}] Param '{n}' missing idx/val in gathered_state_dict – grad will be None."
                        )
                if num_params_updated > 0:
                    self.optimizer.step()  # Apply updates from gathered (and signed) gradients
                    self.scheduler.step()  # Step scheduler after optimizer
                    tplr.logger.info(
                        f"{tplr.P(step_window, tplr.T() - model_update_start_time)} [Rank {self.rank}] Updated model using gathered gradients for {num_params_updated} params."
                    )
                else:
                    tplr.logger.warning(
                        f"{tplr.P(step_window, tplr.T() - model_update_start_time)} [Rank {self.rank}] No parameters updated from gathered gradients (all grads were None). Optimizer step skipped."
                    )

            else:
                tplr.logger.info(
                    f"{tplr.P(step_window, tplr.T() - model_update_start_time)} [Rank {self.rank}] No gathered gradients to apply. Optimizer step skipped."
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model_update_duration = tplr.T() - model_update_start_time
            distrib.barrier()  # Ensure all ranks finish window processing before starting next

            # ──────────────────────────────────────────────────────────────
            # Metrics (all ranks → gather → aggregate on rank-0)
            # ──────────────────────────────────────────────────────────────

            # --- local stats ------------------------------------------------
            local_stats = {
                "loss_sum": total_loss_this_window,
                "batches": num_batches_this_window,
                "tokens": tokens_this_window,
                "train_time": tplr.T() - training_start_time,
                "data_time": tplr.T() - data_load_start_time,
                "comp_time": tplr.T() - compression_start_time,
                "put_time": put_completion_duration,
                "gather_time": gather_duration,
                "model_update_time": model_update_duration,
                "grad_norms": [
                    p.grad.norm().item()
                    for p in model_ref_for_update.parameters()
                    if p.grad is not None
                ],
                "weight_norms": [
                    p.norm().item() for p in model_ref_for_update.parameters()
                ],
                "momentum_norms": [
                    m.norm().item() for m in self.momentum.values() if m is not None
                ],
            }

            gathered_stats = distrib.all_gather_object(
                local_stats
            )  # list[dict] (len = world_size)

            if distrib.is_rank0():
                # --- aggregate ---------------------------------------------
                agg = {
                    "loss_sum": sum(s["loss_sum"] for s in gathered_stats),
                    "batches": sum(s["batches"] for s in gathered_stats),
                    "tokens": sum(s["tokens"] for s in gathered_stats),
                    "train_time": max(
                        s["train_time"] for s in gathered_stats
                    ),  # wall-clock
                    "data_time": max(s["data_time"] for s in gathered_stats),
                    "comp_time": max(s["comp_time"] for s in gathered_stats),
                    "put_time": max(s["put_time"] for s in gathered_stats),
                    "gather_time": max(s["gather_time"] for s in gathered_stats),
                    "model_update_time": max(
                        s["model_update_time"] for s in gathered_stats
                    ),
                }
                # concatenate & compute means/stds
                all_grad_norms = [x for s in gathered_stats for x in s["grad_norms"]]
                all_weight_norms = [
                    x for s in gathered_stats for x in s["weight_norms"]
                ]
                all_momentum_norms = [
                    x for s in gathered_stats for x in s["momentum_norms"]
                ]

                avg_loss_this_window = float(agg["loss_sum"] / max(1, agg["batches"]))
                tokens_per_sec_cluster = float(
                    agg["tokens"] / max(1e-6, agg["train_time"])
                )
                mean_grad_norm = (
                    sum(all_grad_norms) / len(all_grad_norms) if all_grad_norms else 0.0
                )

                # Calculate grad_norm_std and ensure it's explicitly a Python float.
                # The .item() from std() on a float tensor should be float, and 0.0 is float.
                grad_norm_std_value = 0.0
                if len(all_grad_norms) > 1:
                    grad_norm_std_value = float(
                        torch.tensor(all_grad_norms).std().item()
                    )
                # else, it remains 0.0, which is already a float.

                mean_weight_norm = (
                    sum(all_weight_norms) / len(all_weight_norms)
                    if all_weight_norms
                    else 0.0
                )
                mean_momentum_norm = (
                    sum(all_momentum_norms) / len(all_momentum_norms)
                    if all_momentum_norms
                    else 0.0
                )

                window_total_duration = tplr.T() - window_start_time

                current_window_global_tokens = agg["tokens"]
                new_global_total_tokens = (
                    self.total_tokens_processed + current_window_global_tokens
                )

                log_payload = {
                    "miner/loss": avg_loss_this_window,
                    "miner/tokens_per_sec": tokens_per_sec_cluster,
                    "miner/total_tokens": new_global_total_tokens,
                    "miner/batch_tokens": current_window_global_tokens,
                    "miner/global_step": self.global_step,
                    "miner/learning_rate": self.scheduler.get_last_lr()[0],
                    "miner/mean_grad_norm": mean_grad_norm,
                    "miner/grad_norm_std_float": grad_norm_std_value,
                    "miner/mean_weight_norm": mean_weight_norm,
                    "miner/mean_momentum_norm": mean_momentum_norm,
                    "miner/gather/success_rate": gather_success_rate,
                    "miner/gather/num_peers": len(self.comms.peers),
                    "miner/gather/skipped_peers_count": len(skipped_uids_gather),
                    "timing/window_total_duration": window_total_duration,
                    "timing/data_loading_duration": agg["data_time"],
                    "timing/training_duration": agg["train_time"],
                    "timing/compression_duration": agg["comp_time"],
                    "timing/put_duration": agg["put_time"],
                    "timing/gather_duration": agg["gather_time"],
                    "timing/model_update_duration": agg["model_update_time"],
                }
                if torch.cuda.is_available():
                    log_payload["miner/gpu_memory_allocated_mb"] = (
                        torch.cuda.memory_allocated(self.device) / 1024**2
                    )
                    log_payload["miner/gpu_memory_reserved_mb"] = (
                        torch.cuda.memory_reserved(self.device) / 1024**2
                    )

                if self.wandb:
                    try:
                        self.wandb.log(log_payload, step=self.global_step)
                    except Exception as e:
                        tplr.logger.warning(f"[Rank 0 W&B] Log failed: {e}")

                if self.metrics_logger:
                    # Adapt fields for InfluxDB
                    # The key "miner/grad_norm_std_float" will become "grad_norm_std_float" in InfluxDB
                    influx_fields = {
                        k.replace("miner/", "").replace("timing/", "time_"): _safe_json(
                            v
                        )
                        for k, v in log_payload.items()
                    }

                    # Log profiling summary every 10 windows
                    if self.current_window % 10 == 0:
                        tplr.logger.info("Logging performance profiling summary...")
                        tplr.r2_dataset.R2DatasetLoader.log_profiling_summary()

                    # Ensure all numeric values are explicitly converted to the same type (float)
                    # to avoid type conflicts with InfluxDB
                    for key in influx_fields:
                        if isinstance(
                            influx_fields[key], (int, float, np.integer, np.floating)
                        ):
                            influx_fields[key] = float(influx_fields[key])

                    influx_fields["gather_peers_list"] = json.dumps(
                        [int(u) for u in self.comms.peers]
                    )
                    influx_fields["skipped_peers_list"] = json.dumps(
                        skipped_uids_gather
                    )

                    self.metrics_logger.log(
                        measurement="training_step_v2",
                        tags={"window": step_window, "global_step": self.global_step},
                        fields=influx_fields,
                    )
                tplr.logger.info(
                    f"[Rank 0] Metrics (cluster-wide) logged for global_step {self.global_step}."
                )

                # Update rank 0's self.total_tokens_processed to the new global total
                self.total_tokens_processed = new_global_total_tokens

            # Broadcast the updated global total_tokens_processed from rank 0 to all ranks
            # All ranks (including rank 0) will receive this value.
            if distrib.is_rank0():
                # Rank 0 has the authoritative value to broadcast
                value_to_broadcast = self.total_tokens_processed
            else:
                # Other ranks provide a placeholder for the broadcast operation
                value_to_broadcast = None

            self.total_tokens_processed = distrib.broadcast_object(
                value_to_broadcast, src=0
            )

            # Wait for next window signal from Rank 0 (managed at the start of the loop)
            # The old "while self.current_window == step_window: await asyncio.sleep(0.1)"
            # is now implicitly handled by the broadcast of `step_window` at the loop start.
            # If all ranks finish quickly, they will wait at the broadcast_object for the next window.
            tplr.logger.info(
                f"[Rank {self.rank}] Finished window {step_window}. Waiting for next window signal."
            )
            self.window_step += (
                1  # Local counter for windows processed by this miner instance
            )

            # Log total window time at the end
            window_total_duration = tplr.T() - window_start_time
            tplr.logger.info(
                f"{tplr.P(step_window, window_total_duration)} [Rank {self.rank}] Completed window iteration"
            )

    def block_listener(self, loop: asyncio.AbstractEventLoop):
        # This listener runs on all ranks. Rank 0's updates to self.current_window will be broadcast.
        # Other ranks' listeners are mostly for their own logging or potential fallback.
        asyncio.set_event_loop(loop)  # Set event loop for this thread

        def handler(
            event_details,
        ):  # Renamed 'event' to 'event_details' to avoid conflict
            try:
                new_block = int(event_details["header"]["number"])
                new_window = int(new_block / self.hparams.blocks_per_window)

                # Only Rank 0's updates to self.current_block/window are authoritative for the main loop
                # via broadcasting. Other ranks can log or use for local checks.
                if distrib.is_rank0():
                    if new_window != self.current_window:
                        tplr.logger.info(
                            f"[Rank 0 BlockListener] New window detected: {new_window} (from block {new_block}). Old: {self.current_window}"
                        )
                        self.current_block = new_block
                        self.current_window = new_window
                        self.comms.current_window = new_window  # Update comms as well
                else:  # Non-rank0 listeners can log if their view diverges significantly (for debugging)
                    if (
                        abs(new_block - self.current_block)
                        > self.hparams.blocks_per_window
                    ):  # Example divergence check
                        tplr.logger.debug(
                            f"[Rank {self.rank} BlockListener] Local block {new_block}, window {new_window}. Synced block is {self.current_block}"
                        )

            except Exception as e:
                tplr.logger.error(
                    f"[Rank {self.rank} BlockListener] Error processing block event: {e}"
                )

        backoff = 1
        max_backoff = 60
        while (
            not self.stop_event.is_set()
        ):  # Assuming self.stop_event is set on shutdown
            try:
                # Each rank subscribes. Redundant but simple.
                # Could be Rank 0 only + internal broadcast if subtensor connections are an issue.
                sub_instance = bt.subtensor(
                    config=self.config
                )  # Fresh instance for thread safety potentially
                sub_instance.substrate.subscribe_block_headers(
                    handler
                )  # This is a blocking call
                # If subscribe_block_headers returns, it means connection was lost or exited cleanly.
                tplr.logger.warning(
                    f"[Rank {self.rank} BlockListener] Subscription ended. Reconnecting..."
                )
                backoff = (
                    1  # Reset backoff on successful exit (though usually it's an error)
                )
            except websockets.exceptions.ConnectionClosedError as e:  # type: ignore
                tplr.logger.warning(
                    f"[Rank {self.rank} BlockListener] Websocket ConnectionClosedError: {e}. Retrying in {backoff}s."
                )
            except Exception as e:
                tplr.logger.error(
                    f"[Rank {self.rank} BlockListener] Subscription error: {e}. Retrying in {backoff}s."
                )

            if self.stop_event.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
        tplr.logger.info(f"[Rank {self.rank} BlockListener] Listener thread stopped.")

    async def _wait_for_start_window(
        self, poll_interval: float = 5.0, timeout: int = 300
    ) -> int:
        """Rank 0 polls chain until a non-zero start window appears or determined by current block."""
        assert distrib.is_rank0(), "Only Rank 0 should wait for start window this way."

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Get current block and calculate window
            current_block_val = self.subtensor.block
            current_window_val = int(current_block_val / self.hparams.blocks_per_window)

            # A simple heuristic: if current window is positive, consider it the start.
            # Or, if there's a specific minimum window defined in hparams.
            min_start_window = getattr(self.hparams, "min_start_window", 1)
            if current_window_val >= min_start_window:
                tplr.logger.info(
                    f"[Rank 0] Determined start window {current_window_val} from current block {current_block_val}."
                )
                # Update self for consistency before broadcasting
                self.current_block = current_block_val
                self.current_window = current_window_val
                return current_window_val

            tplr.logger.info(
                f"[Rank 0] Waiting for a valid start window. Current block: {current_block_val}, window: {current_window_val}. Polling in {poll_interval}s."
            )
            await asyncio.sleep(poll_interval)

        raise RuntimeError(f"[Rank 0] Timed out ({timeout}s) waiting for start window.")

    def _safe_json(val):
        """Convert tensors / numpy / containers → plain python scalars/lists."""
        if isinstance(val, torch.Tensor):
            return val.item() if val.ndim == 0 else val.tolist()
        if isinstance(val, (np.ndarray,)):
            return val.tolist()
        if isinstance(val, (np.generic,)):  # numpy scalar
            return val.item()
        if isinstance(val, (list, tuple)):
            return [_safe_json(x) for x in val]
        if isinstance(val, dict):
            return {k: _safe_json(v) for k, v in val.items()}
        return val


def _safe_json(val):
    """Convert tensors / numpy / containers → plain python scalars/lists."""
    if isinstance(val, torch.Tensor):
        return val.item() if val.ndim == 0 else val.tolist()
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if isinstance(val, (np.generic,)):  # numpy scalar
        return val.item()
    if isinstance(val, (list, tuple)):
        return [_safe_json(x) for x in val]
    if isinstance(val, dict):
        return {k: _safe_json(v) for k, v in val.items()}
    return val


def main():
    uvloop.install()
    miner_instance = Miner()
    try:
        asyncio.run(miner_instance.run())
    except KeyboardInterrupt:
        tplr.logger.info("Miner shutting down...")
    except Exception as e:
        tplr.logger.error(f"Unhandled exception in miner: {e}", exc_info=True)
    finally:
        if distrib.is_initialized():
            distrib.cleanup()
        tplr.logger.info("Miner run loop finished.")


if __name__ == "__main__":
    main()
