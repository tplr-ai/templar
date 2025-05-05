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
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Optional

# Third party
import bittensor as bt
import numpy as np
import torch
import uvloop
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from transformers import LlamaForCausalLM
import atexit

# Local
import tplr
import tplr.distrib as distrib 

CPU_COUNT = os.cpu_count() or 4
CPU_MAX_CONNECTIONS = min(100, max(30, CPU_COUNT * 4))

# GPU optimizations (seed setting should ideally happen per-worker)
# torch.manual_seed(42) # Move seeding inside worker
# torch.cuda.manual_seed_all(42) # Move seeding inside worker
# np.random.seed(42) # Move seeding inside worker
# random.seed(42) # Move seeding inside worker
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Miner:
    # Command line config items.
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description="Miner script")
        parser.add_argument(
            "--netuid", type=int, default=268, help="Bittensor network UID."
        )
        parser.add_argument(
            "--project", type=str, default="templar", help="Wandb project."
        )
        # Removed --device, will be determined by local_rank
        # Add DDP specific args
        parser.add_argument(
            "--use_ddp", action="store_true", help="Enable Distributed Data Parallel (DDP) mode."
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

        # Logging level setup
        if config.trace:
            tplr.trace()
        elif config.debug:
            tplr.debug()
        return config

    @classmethod
    def launch(cls):
        """Main entry point modified for DDP launch or single process run."""
        config = cls.config()
        wallet = bt.wallet(config)

        if config.use_ddp:
            num_gpus = torch.cuda.device_count()
            if num_gpus <= 1:
                tplr.logger.warning("DDP requires more than 1 GPU. Running in single GPU mode.")
                config.use_ddp = False # Force single GPU mode
                miner_instance = cls(config, wallet, local_rank=0, world_size=1)
                uvloop.install()
                asyncio.run(miner_instance.run())
            else:
                tplr.logger.info(f"Launching DDP Miner with {num_gpus} GPUs.")
                # Find a free port for DDP
                port = distrib.find_free_port()
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = str(port)
                
                # Register cleanup before spawning
                atexit.register(distrib.cleanup)
                
                try:
                    mp.spawn(
                        cls._run_worker,
                        args=(num_gpus, config, wallet),
                        nprocs=num_gpus,
                        join=True
                    )
                except Exception as e:
                    tplr.logger.error(f"DDP Spawn failed: {e}", exc_info=True)
                finally:
                    # Ensure cleanup happens even if spawn fails or is interrupted
                    atexit.unregister(distrib.cleanup)
                    distrib.cleanup()
        else:
            tplr.logger.info("Running Miner in single GPU/CPU mode (DDP disabled).")
            miner_instance = cls(config, wallet, local_rank=0, world_size=1)
            uvloop.install()
            asyncio.run(miner_instance.run())

    @classmethod
    def _run_worker(cls, local_rank, world_size, config, wallet):
        """Worker process function called by mp.spawn."""
        try:
            # Set seeds per worker for reproducibility if desired, but might affect DDP performance slightly
            seed = 42 + local_rank
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # Initialize the distributed environment
            distrib.ddp_init(local_rank, world_size)
            
            # Create and run the miner instance
            # Note: Pass local_rank and world_size to __init__
            miner = cls(config, wallet, local_rank=local_rank, world_size=world_size)
            
            # Install uvloop here for the async process
            uvloop.install()
            asyncio.run(miner.run())

        except Exception as e:
            tplr.logger.error(f"[Rank {local_rank}] Worker failed: {e}", exc_info=True)
            # Optionally re-raise or perform other error handling
        finally:
            # Ensure cleanup is called by each worker (DDP might require this)
            # Although the atexit handler should cover the main process exit
            distrib.cleanup()

    def __init__(self, config, wallet, local_rank: int, world_size: int):
        """Initialize the miner with DDP support."""
        self.config = config
        self.wallet = wallet
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f'cuda:{self.local_rank}' if torch.cuda.is_available() and self.config.use_ddp else "cuda" if torch.cuda.is_available() else "cpu"
        self.is_ddp = self.config.use_ddp and self.world_size > 1

        # Configure logging (potentially rank-specific)
        # Consider having only rank 0 log INFO level to console
        # tplr.logging setup might need adjustment for multi-process if writing to shared files
        # if self.is_ddp and not distrib.is_rank0():

        tplr.logger.info(f"[Rank {self.local_rank}] Initializing Miner on device: {self.device} (DDP: {self.is_ddp})")

        # Init config and load hparams
        self.hparams = tplr.load_hparams(use_local_run_hparams=self.config.local)

        # --- Bittensor objects ---
        # Each process might need its own subtensor connection
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = bt.metagraph(netuid=self.config.netuid, network=self.subtensor.network)
        # # Initial sync only on rank 0? Or let all sync? Let all sync initially.
        # try:
        #     self.metagraph.sync(subtensor=self.subtensor)
        # except Exception as e:
        #      tplr.logger.error(f"[Rank {self.local_rank}] Metagraph sync failed: {e}. Exiting.", exc_info=True)
        #      # Handle failure appropriately - maybe sys.exit or raise
        #      # If rank 0 fails, others might hang waiting for it.
        #      # Need robust failure handling across ranks.
        #      sys.exit(1) # Simple exit for now

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            # Log error only on rank 0
            if distrib.is_rank0():
                tplr.logger.error(
                    f"\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}[/bold]"
                )
            # All processes should exit if wallet not registered
            sys.exit()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        tplr.logger.info(f"[Rank {self.local_rank}] Miner UID: {self.uid}")

        # --- Model ---
        self.model = LlamaForCausalLM(self.hparams.model_config)
        self.model.to(self.device)
        self.tokenizer = self.hparams.tokenizer

        # --- DDP Wrapping (if enabled) ---
        if self.is_ddp:
            # Apply torch.compile if configured (before DDP wrapping)
            # if getattr(self.hparams, "use_compile", False):
            #     tplr.logger.info(f"[Rank {self.local_rank}] Compiling model...")
            #     self.model = torch.compile(self.model, dynamic=False) # Requires PT 2.0+

            tplr.logger.info(f"[Rank {self.local_rank}] Wrapping model with DDP...")
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if self.device != 'cpu' else None,
                output_device=self.local_rank if self.device != 'cpu' else None,
                gradient_as_bucket_view=True, # Optimization
                find_unused_parameters=False   # Assuming no unused params
            )
            tplr.logger.info(f"[Rank {self.local_rank}] Model wrapped with DDP.")

        # --- Compression ---
        # Transformer might depend on the DDP-wrapped model structure if accessing modules directly
        # Access underlying model for TransformDCT if needed: model_to_transform = self.model.module if self.is_ddp else self.model
        model_to_transform = self.model.module if self.is_ddp else self.model
        self.transformer = tplr.compress.TransformDCT(
            model_to_transform, target_chunk=self.hparams.target_chunk
        )
        self.compressor = tplr.compress.CompressDCT()

        # --- Optimizer and Momentum ---
        self.optimizer = SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        # Momentum only needed on rank 0 for preparing gradients for peers
        self.momentum = {}
        self.xshapes = {}
        self.totalks = {}
        # Only rank 0 needs to calculate shapes/totalks for compression info
        # Use the underlying model parameters for shape calculation
        model_params_for_shape = (self.model.module if self.is_ddp else self.model).named_parameters()
        for n, p in model_params_for_shape:
            if distrib.is_rank0():
                self.momentum[n] = torch.zeros_like(p, device='cpu') # Keep momentum on CPU for Rank 0
            # Calculate shapes/totalks on all ranks for consistency? Or broadcast from rank 0?
            # Let's calculate everywhere for now, it's cheap. Ensure consistent results.
            # Use a dummy tensor on the correct device/dtype to calculate shapes
            dummy_tensor_for_shape = torch.zeros_like(p, device=self.device, dtype=p.dtype)
            _, _, xshape, totalk = self.compressor.compress(
                 self.transformer.encode(dummy_tensor_for_shape), self.hparams.topk_compression
            )
            self.xshapes[n] = xshape
            self.totalks[n] = totalk
        # Ensure all ranks have the same xshapes/totalks (e.g., broadcast from rank 0 if calculation isn't deterministic)
        # Let's assume calculation is deterministic for now.

        # --- Scheduler ---
        # Schedulers need to be stepped on all ranks
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.hparams.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.hparams.t_max, T_mult=2, eta_min=self.hparams.learning_rate * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.hparams.warmup_steps],
        )

        # --- Comms ---
        # Comms object used primarily by Rank 0, but other ranks might need basic info?
        # Initialize on all ranks, but R2 ops guarded by is_rank0()
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location=os.path.join("/tmp", f"miner_{self.uid}_rank{self.local_rank}"), # Rank specific temp dir
            key_prefix="model",
            config=self.config,
            netuid=self.config.netuid,
            metagraph=self.metagraph, # Share the metagraph synced by rank 0? Or sync everywhere? Sync everywhere safer.
            hparams=self.hparams,
            uid=self.uid,
            local_rank=self.local_rank, # Pass rank info
            world_size=self.world_size # Pass world size
        )

        # Rank 0 tries to commit
        if distrib.is_rank0():
            self.bucket = self.comms.get_own_bucket("gradients", "write") # Need write bucket for commit
            self.comms.try_commit(self.wallet, self.bucket)

        # --- State ---
        self.stop_event = asyncio.Event() # May need cross-process signaling if used for termination
        self.current_block = self.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        self.start_window = 0 # Needs to be fetched/set later
        self.global_step = 0
        self.step_counter = 0
        self.window_step = 0

        # --- Metrics Logging (Rank 0 Only) ---
        if distrib.is_rank0():
            self.wandb = tplr.initialize_wandb(
                run_prefix="M", uid=self.uid, config=self.config, group="miner", job_type="mining",
            )
            self.metrics_logger = tplr.metrics.MetricsLogger(
                prefix="M", uid=self.uid, config=self.config, role="miner", group="miner", job_type="mining",
            )
        else:
            self.wandb = None
            self.metrics_logger = None

        # --- Peers (Mainly used by Rank 0) ---
        self.next_peers: Optional[tplr.comms.PeerArray] = None
        self.peers_update_window = -1

        # --- Dataset & DataLoader ---
        # Create placeholder - actual data loading happens in run based on window
        # We need a dummy dataset object to create the sampler initially?
        # Let's create the sampler and loader within the run loop instead.
        self.dataset = None # Placeholder
        self.sampler = None
        self.dataloader = None

        self.bootstrap_version = getattr(self.hparams, "checkpoint_init_version", None)
        tplr.logger.info(
            f"[Miner] code_version={tplr.__version__} "
            f"checkpoint_init_flag={self.bootstrap_version or '<none>'}"
        )

        tplr.logger.info(f"[Rank {self.local_rank}] Miner initialization complete.")

    async def run(self):
        """Main async run loop, adapted for DDP."""
        tplr.logger.info(f"[Rank {self.local_rank}] Starting run loop...")

        # --- Initial Setup ---
        # Start block listener thread (only rank 0?) - careful with shared state modification
        # Let's have only Rank 0 manage the block listener and broadcast window changes if needed
        # For simplicity now, assume all ranks poll self.subtensor.block frequently enough
        # TODO: Revisit block listening strategy for DDP
        # if distrib.is_rank0():
        #     self.loop = asyncio.get_running_loop()
        #     self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
        #     self.loop.set_default_executor(self.executor)
        #     self.listener = threading.Thread(target=self.block_listener, args=(self.loop,), daemon=True)
        #     self.listener.start()

        # --- Commitments and Start Window (Rank 0 fetches, broadcasts) ---
        commitments_list = [None]
        if distrib.is_rank0():
            self.comms.commitments = await self.comms.get_commitments()
            tplr.logger.info("[Rank 0] Loaded commitments")
            commitments_list = [self.comms.commitments]
        if self.is_ddp:
            dist.broadcast_object_list(commitments_list, src=0)
            distrib.barrier() # Ensure all ranks have commitments before proceeding
        self.comms.commitments = commitments_list[0]
        if not distrib.is_rank0():
             tplr.logger.info(f"[Rank {self.local_rank}] Received commitments from Rank 0.")


        start_window_list = [None]
        if distrib.is_rank0():
            fetched_start_window = await self.comms.get_start_window()
            if fetched_start_window is None:
                 tplr.logger.error("[Rank 0] Failed to fetch start window. Exiting.")
                 # Need a way to signal other ranks to exit gracefully
                 sys.exit(1) # Abrupt exit for now
            start_window_list = [fetched_start_window]
        if self.is_ddp:
            dist.broadcast_object_list(start_window_list, src=0)
            distrib.barrier()
        self.start_window = start_window_list[0]
        if self.start_window is None:
            # This should only happen if Rank 0 failed and exited.
            tplr.logger.error(f"[Rank {self.local_rank}] Did not receive start window. Exiting.")
            sys.exit(1)
        tplr.logger.info(f"[Rank {self.local_rank}] Using start_window: {self.start_window}")

        # Calculate initial global step based on potentially updated current_window
        self.current_block = self.subtensor.block # Ensure current block is fresh
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        self.global_step = max(0, self.current_window - self.start_window)
        tplr.logger.info(f"[Rank {self.local_rank}] Starting at Global Step: {self.global_step}")

        # --- Checkpoint Loading (Rank 0 loads, broadcasts state) ---
        checkpoint_state = [None]
        loaded_checkpoint_window = -1
        load_success = False

        if distrib.is_rank0():
            checkpoint_window_buffer = 5
            has_new_checkpoint = (
                self.global_step
                >= self.hparams.checkpoint_frequency + checkpoint_window_buffer
            )
            _success, _loaded_momentum, _loaded_checkpoint_window, _loaded_optimizer, _loaded_scheduler = await self.comms.load_checkpoint(
                model=self.model.module, # Load into the underlying model
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                current_window=self.current_window,
                device='cpu', # Load to CPU first before broadcasting/moving
                init_version=tplr.__version__ if has_new_checkpoint else self.bootstrap_version
            )
            if _success:
                tplr.logger.info(f"[Rank 0] Successfully loaded checkpoint data for window {_loaded_checkpoint_window}.")
                load_success = True
                loaded_checkpoint_window = _loaded_checkpoint_window
                # Prepare state to broadcast: model, optim, scheduler, momentum
                checkpoint_state = [{
                    'model_state_dict': self.model.module.state_dict(), # Get state from underlying model
                    'optimizer_state_dict': _loaded_optimizer.state_dict(),
                    'scheduler_state_dict': _loaded_scheduler.state_dict(),
                    'momentum': _loaded_momentum,
                    'loaded_checkpoint_window': _loaded_checkpoint_window,
                    'load_success': True
                }]
                # Update Rank 0's momentum, optimizer, scheduler immediately
                self.momentum = _loaded_momentum
                self.optimizer = _loaded_optimizer
                self.scheduler = _loaded_scheduler
            else:
                 tplr.logger.info("[Rank 0] No checkpoint found or load failed, initializing from scratch.")
                 checkpoint_state = [{'load_success': False}] # Signal failure

        # Broadcast checkpoint state
        if self.is_ddp:
            dist.broadcast_object_list(checkpoint_state, src=0)
            distrib.barrier()

        # Apply loaded state on all ranks
        loaded_state_data = checkpoint_state[0]
        load_success = loaded_state_data['load_success']

        if load_success:
            tplr.logger.info(f"[Rank {self.local_rank}] Applying loaded checkpoint state.")
            # Load model state dict (apply to DDP model, it will handle underlying module)
            # Move state dict tensors to the correct device before loading
            model_state = {k: v.to(self.device) for k, v in loaded_state_data['model_state_dict'].items()}
            self.model.load_state_dict(model_state)

            # Load optimizer state dict
            # Need to handle moving optimizer state tensors to the correct device
            optim_state = loaded_state_data['optimizer_state_dict']
            for state in optim_state['state'].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.optimizer.load_state_dict(optim_state)

            # Load scheduler state dict
            self.scheduler.load_state_dict(loaded_state_data['scheduler_state_dict'])

            loaded_checkpoint_window = loaded_state_data['loaded_checkpoint_window']
            # Only Rank 0 needs the momentum dict for `prepare_gradient_dict`
            if distrib.is_rank0():
                self.momentum = loaded_state_data['momentum']
                # Move momentum tensors to CPU for Rank 0
                self.momentum = {k: v.to('cpu') for k, v in self.momentum.items()}

            tplr.logger.info(f"[Rank {self.local_rank}] Loaded checkpoint state for window {loaded_checkpoint_window}.")
            # Sync global step based on loaded checkpoint window
            self.global_step = max(0, loaded_checkpoint_window - self.start_window)
            tplr.logger.info(f"[Rank {self.local_rank}] Global step synced to {self.global_step}")

        else:
            # Initialize momentum (only Rank 0 needs it)
            if distrib.is_rank0():
                self.momentum = {
                    n: torch.zeros_like(p, device='cpu') # Keep on CPU
                    for n, p in (self.model.module if self.is_ddp else self.model).named_parameters()
                }
            # Apply scratch model to ensure consistency if DDP wrapping happened before potential load failure broadcast
            # DDP should handle initial parameter broadcast, but an explicit sync might be safer
            if self.is_ddp:
                distrib.sync_model_params(self.model) # Needs implementation in distrib.py

            tplr.logger.info(f"[Rank {self.local_rank}] Initialized model from scratch.")


        # --- Catchup Logic ---
        # This needs careful synchronization. Let Rank 0 decide if catchup is needed and maybe broadcast the target catchup window.
        # For now, simplify: Assume catchup is primarily state update, handled *before* the main loop starts if possible.
        # The existing catchup logic uses `comms.load_aggregation` (needs rank 0 guard) and applies gradients.
        # This conflicts with DDP's expected synchronization.
        # Simplification: If checkpoint is old, maybe just log a warning for now and rely on future windows to converge.
        # TODO: Design a DDP-compatible catchup mechanism (potentially rank 0 loads agg, processes, broadcasts final state delta).
        if load_success and loaded_checkpoint_window < self.current_window - 1 : # Check if significantly behind
             tplr.logger.warning(f"[Rank {self.local_rank}] Checkpoint ({loaded_checkpoint_window}) is behind current window ({self.current_window}). Catchup logic needs DDP adaptation and is currently skipped.")
        # Ensure all ranks are synchronized before starting the main loop
        if self.is_ddp: 
            distrib.barrier()


        # --- Main Loop ---
        while True:
            try:
                # --- Window Setup ---
                window_start_time = tplr.T()
                # Get current block/window (maybe Rank 0 polls and broadcasts?)
                # Simple approach: all ranks poll, slight potential for mismatch but barriers should handle.
                self.current_block = self.subtensor.block
                step_window = int(self.current_block / self.hparams.blocks_per_window)

                # Simple loop condition check (all ranks check independently)
                if step_window <= self.current_window and self.current_window > 0: # Wait for next window
                    await asyncio.sleep(1) # Yield control
                    continue
                self.current_window = step_window # Update window for this iteration
                self.global_step = max(0, self.current_window - self.start_window)

                tplr.logger.info(f"\n[Rank {self.local_rank}] {'-' * 40} Window: {self.current_window} (Global Step: {self.global_step}) {'-' * 40}")

                # --- Peer Update (Rank 0 Only) ---
                # TODO: Implement `update_peers` correctly in `tplr.neurons` for DDP
                if distrib.is_rank0():
                    peer_update_start = tplr.T()
                    await tplr.neurons.update_peers(
                        instance=self, window=self.current_window, peer_start=peer_update_start
                    )
                # Barrier? Needed if other ranks rely on updated self.comms.peers (they shouldn't directly)
                if self.is_ddp: 
                    distrib.barrier() # Ensure rank 0 peer update finishes

                # --- Data Loading for Window ---
                # All ranks load page *info* deterministically
                # Note: R2DatasetLoader methods might need rank guarding if they modify shared state
                pages = await tplr.r2_dataset.R2DatasetLoader.next_pages(
                    offset=self.current_window * self.hparams.pages_per_window,
                    n_pages=self.hparams.pages_per_window,
                    seed=self.uid, # Use UID for deterministic page selection per miner
                )

                # Create Dataset, Sampler, DataLoader for this window
                # Assuming R2DatasetLoader can be initialized without immediate loading
                self.dataset = await tplr.r2_dataset.R2DatasetLoader.create(
                     batch_size=self.hparams.batch_size,
                     sequence_length=self.hparams.sequence_length,
                     pages_info=pages,
                     tokenizer=self.tokenizer,
                )
                self.sampler = DistributedSampler(
                     self.dataset,
                     num_replicas=self.world_size,
                     rank=self.local_rank,
                     shuffle=True, # Shuffle batches within the window's data
                     drop_last=True # Important for DDP
                )
                self.dataloader = DataLoader(
                     self.dataset,
                     batch_size=self.hparams.batch_size, # Batch size *per GPU*
                     sampler=self.sampler,
                     # num_workers=self.hparams.num_workers, # Can cause issues with multiprocessing context
                     pin_memory=True,
                     drop_last=True
                )
                self.sampler.set_epoch(self.current_window) # CRITICAL: Set epoch for sampler

                if distrib.is_rank0():
                     tplr.logger.info(f"Data loaded for window {self.current_window}. Pages: {[p[1] for p in pages]}. Batches per GPU: {len(self.dataloader)}")

                # --- Local Training & Grad Sync ---
                self.model.train()
                self.optimizer.zero_grad() # Zero gradients once before the window loop
                total_loss_this_window = torch.zeros(1, device=self.device)
                window_tokens_this_gpu = 0
                num_batches_this_gpu = 0

                # The actual DDP training part - use no_sync and manual reduce
                # This loop needs to be effectively synchronous across ranks
                with self.model.no_sync():
                    for i, batch in enumerate(self.dataloader):
                        batch_start_time = tplr.T()
                        # Input IDs are expected directly based on R2DatasetLoader
                        input_ids = batch.to(self.device) # Move batch to device
                        tokens_this_batch = input_ids.numel()
                        window_tokens_this_gpu += tokens_this_batch

                        labels = input_ids.clone()
                        # Assuming pad_token_id is consistent across ranks
                        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)

                        # Autocast for mixed precision
                        with autocast(device_type=self.device.type if self.device != 'cpu' else 'cpu', dtype=torch.bfloat16):
                            outputs = self.model(input_ids=input_ids, labels=labels)
                            loss = outputs.loss

                        # Loss scaling for averaging later
                        scaled_loss = loss / self.world_size
                        total_loss_this_window += scaled_loss.detach() # Accumulate detached loss on device
                        num_batches_this_gpu += 1

                        scaled_loss.backward() # Accumulate gradients locally

                        if distrib.is_rank0() and i % 10 == 0: # Log progress infrequently on rank 0
                             tplr.logger.debug(f"Rank 0, Window {self.current_window}, Batch {i}, Loss: {loss.item():.4f}, Time: {tplr.T() - batch_start_time:.2f}s")

                        # Check if window changed mid-loop (less critical with barriers, but good practice)
                        # current_block_check = self.subtensor.block # Frequent polling
                        # if int(current_block_check / self.hparams.blocks_per_window) != self.current_window:
                        #     tplr.logger.warning(f"[Rank {self.local_rank}] Window changed during batch processing. Breaking loop.")
                        #     break # Exit batch loop if window changes


                # --- Manual Gradient Aggregation ---
                # (Blocking operation across all ranks)
                tplr.logger.debug(f"[Rank {self.local_rank}] Starting gradient all_reduce...")
                all_reduce_start_time = tplr.T()
                grads_to_reduce = [p.grad for p in self.model.parameters() if p.grad is not None]
                distrib.all_reduce_params(grads_to_reduce, op=dist.ReduceOp.AVG)
                distrib.barrier() # Wait for all ranks to finish reduction
                tplr.logger.info(f"[Rank {self.local_rank}] Gradient all_reduce finished in {tplr.T() - all_reduce_start_time:.2f}s")

                # --- Templar Processing (Rank 0) & Gradient Broadcast ---
                flat_grad = torch.empty(0, device=self.device) # Placeholder
                shapes_dtypes_list = [None, None] # Placeholder

                if distrib.is_rank0():
                    # 1. Prepare Gradient Dict (using averaged grads now available on rank 0)
                    prepare_start = tplr.T()
                    # NOTE: `tplr.prepare_gradient_dict` needs access to `self.model.module.parameters()`
                    # if accessing model directly, and uses `self.momentum` (CPU tensor)
                    # It should read p.grad (on device) and update self.momentum (on CPU)
                    gradient_dict, _, _, _ = tplr.prepare_gradient_dict(
                        self, pages, self.current_window
                    )
                    tplr.logger.info(f"[Rank 0] Prepared gradient dict in {tplr.T() - prepare_start:.2f}s")

                    # 2. PUT compressed gradient to R2
                    put_start = tplr.T()
                    # Move dict tensors to CPU before putting
                    processed_state_dict_cpu = {}
                    for k, v in gradient_dict.items():
                        if isinstance(v, torch.Tensor):
                            processed_state_dict_cpu[k] = v.to("cpu")
                        else:
                            processed_state_dict_cpu[k] = v
                    # R2 PUT must be awaited
                    put_completion_time = await self.comms.put(
                        state_dict=processed_state_dict_cpu,
                        uid=str(self.uid), window=self.current_window, key="gradient",
                        global_step=self.global_step, local=False
                    )
                    tplr.logger.info(f"[Rank 0] R2 PUT finished in {tplr.T() - put_start:.2f}s (Reported: {put_completion_time:.2f}s)")


                    # 3. GATHER gradients from peers/aggregator
                    gather_start = tplr.T()
                    # Need timestamp logic for gather window
                    # Reuse existing timestamp logic, ensuring it runs only on Rank 0
                    sync_block = (self.current_window + 1) * self.hparams.blocks_per_window # Predict next window start block
                    ts_value = await self.comms.get_block_timestamp(sync_block) # Needs implementation in Comms, rank 0 only
                    time_min = datetime.fromtimestamp(ts_value, tz=timezone.utc)
                    time_max = time_min + timedelta(seconds=self.hparams.time_window_delta_seconds)
                    tplr.logger.info(f"[Rank 0] Using time window for gather: {time_min} to {time_max}")


                    gather_result = await self.comms.gather(
                         my_uid=self.uid, uids=self.comms.peers, window=self.current_window, key="gradient",
                         timeout=35, device='cpu', # Gather to CPU on rank 0
                         local=False, totalks=self.totalks,
                         time_min=time_min, time_max=time_max
                    )
                    gather_time = tplr.T() - gather_start
                    tplr.logger.info(f"[Rank 0] R2 GATHER finished in {gather_time:.2f}s")

                    # 4. Process Gathered Gradients
                    process_start = tplr.T()
                    # Define self.process_gathered_gradients - takes gather_result, returns list of tensors (one per param)
                    final_grad_list = self.process_gathered_gradients(gather_result) # Implement this method!
                    tplr.logger.info(f"[Rank 0] Processed gathered gradients in {tplr.T() - process_start:.2f}s")


                    # 5. Flatten for Broadcast
                    flatten_start = tplr.T()
                    # Ensure final_grad_list tensors are on the correct device for broadcast
                    final_grad_list_device = [g.to(self.device) for g in final_grad_list]
                    flat_grad, shapes, dtypes = distrib.flatten_grads(final_grad_list_device)
                    shapes_dtypes_list = [shapes, dtypes] # Store for broadcast
                    tplr.logger.info(f"[Rank 0] Flattened gradients in {tplr.T() - flatten_start:.2f}s. Size: {flat_grad.numel()}")


                # --- Broadcast Shapes/Dtypes & Tensor Size ---
                # (Blocking operation across all ranks)
                broadcast_meta_start = tplr.T()
                dist.broadcast_object_list(shapes_dtypes_list, src=0)
                shapes, dtypes = shapes_dtypes_list
                
                size_tensor = torch.tensor([flat_grad.numel() if distrib.is_rank0() else 0], dtype=torch.long, device=self.device)
                dist.broadcast(size_tensor, src=0)
                flat_grad_size = size_tensor.item()
                tplr.logger.debug(f"[Rank {self.local_rank}] Broadcast meta finished in {tplr.T() - broadcast_meta_start:.2f}s. Flat size: {flat_grad_size}")


                # --- Broadcast Flat Gradient Tensor ---
                # (Blocking operation across all ranks)
                broadcast_tensor_start = tplr.T()
                if not distrib.is_rank0():
                    # Allocate buffer on non-rank-0 processes
                    # Use the first dtype received for allocation, assuming homogeneity or handle mixed dtypes if needed
                    alloc_dtype = dtypes[0] if dtypes else torch.float32
                    flat_grad = torch.zeros(flat_grad_size, dtype=alloc_dtype, device=self.device)

                dist.broadcast(flat_grad, src=0)
                distrib.barrier() # Ensure broadcast completes
                tplr.logger.info(f"[Rank {self.local_rank}] Broadcast tensor finished in {tplr.T() - broadcast_tensor_start:.2f}s")


                # --- Unflatten and Apply Gradient ---
                # (Local operation on all ranks)
                apply_start = tplr.T()
                if flat_grad_size > 0: # Check if we actually received gradients
                    final_grad_list = distrib.unflatten_grads(flat_grad, shapes, dtypes)
                    
                    # Get underlying model parameters if using DDP
                    model_params = list(self.model.module.parameters() if self.is_ddp else self.model.parameters())
                    
                    param_idx = 0
                    for p in model_params:
                        if p.requires_grad: # Only update params that require grads
                           if param_idx < len(final_grad_list):
                               # Assign received gradient to p.grad
                               if p.grad is None: # Handle cases where grad might be None initially (though unlikely after backward)
                                    p.grad = final_grad_list[param_idx].to(p.device)
                               else:
                                    p.grad.copy_(final_grad_list[param_idx]) # Overwrite local avg grad with broadcasted final grad
                               param_idx += 1
                           else:
                                tplr.logger.warning(f"[Rank {self.local_rank}] Mismatch between parameters and unflattened gradients at index {param_idx}")
                                break
                else:
                    tplr.logger.warning(f"[Rank {self.local_rank}] No valid gradients received or processed by Rank 0. Skipping gradient application.")
                    # Ensure gradients are zeroed if none applied?
                    self.optimizer.zero_grad() # Re-zero gradients if we skipped application


                distrib.barrier() # Ensure all ranks have applied gradients
                tplr.logger.info(f"[Rank {self.local_rank}] Applied final gradients in {tplr.T() - apply_start:.2f}s")


                # --- Optimizer and Scheduler Step ---
                # (Blocking operation across all ranks)
                step_start = tplr.T()
                self.optimizer.step()
                self.scheduler.step()
                distrib.barrier() # Ensure steps complete before next window
                tplr.logger.info(f"[Rank {self.local_rank}] Optimizer/Scheduler step finished in {tplr.T() - step_start:.2f}s")

                # --- Checkpointing (Rank 0 Only) ---
                if distrib.is_rank0() and self.global_step > 0 and self.global_step % self.hparams.checkpoint_frequency == 0:
                     tplr.logger.info(f"[Rank 0] Creating checkpoint at global_step {self.global_step}")
                     await self.comms.save_checkpoint(
                          model=self.model.module, # Save underlying model state
                          optimizer=self.optimizer,
                          scheduler=self.scheduler,
                          momentum=self.momentum, # Rank 0's CPU momentum
                          global_step=self.global_step,
                          current_window=self.current_window,
                          start_window=self.start_window,
                     )


                # --- Logging (Aggregate & Rank 0 Logs) ---
                # Aggregate loss
                dist.all_reduce(total_loss_this_window, op=dist.ReduceOp.SUM)
                avg_loss_this_window = total_loss_this_window.item() / num_batches_this_gpu if num_batches_this_gpu > 0 else 0
                # Aggregate tokens
                total_window_tokens = torch.tensor([window_tokens_this_gpu], device=self.device)
                dist.all_reduce(total_window_tokens, op=dist.ReduceOp.SUM)
                total_window_tokens = total_window_tokens.item()

                if distrib.is_rank0():
                    window_duration = tplr.T() - window_start_time
                    tokens_per_sec = total_window_tokens / window_duration if window_duration > 0 else 0
                    self.total_tokens_processed += total_window_tokens # Track total tokens on Rank 0

                    # Log to WandB and InfluxDB (Rank 0 only)
                    if self.wandb:
                         # Simplified logging example
                         log_data = {
                              "miner/loss": avg_loss_this_window,
                              "miner/tokens_per_sec": tokens_per_sec,
                              "miner/total_tokens": self.total_tokens_processed,
                              "miner/batch_tokens": total_window_tokens, # Total across GPUs
                              "miner/global_step": self.global_step,
                              "miner/learning_rate": self.scheduler.get_last_lr()[0],
                              "miner/timing/window_total": window_duration,
                              # Add other relevant timings if measured
                         }
                         self.wandb.log(log_data, step=self.global_step)
                    if self.metrics_logger:
                         self.metrics_logger.log(
                              measurement="training_step_v2",
                              tags={"window": self.current_window, "global_step": self.global_step},
                              fields={
                                   "loss": avg_loss_this_window,
                                   "tokens_per_sec": tokens_per_sec,
                                   # Add other fields
                              }
                         )
                    tplr.logger.info(f"[Rank 0] Window {self.current_window} finished. Loss: {avg_loss_this_window:.4f}, Tokens/sec: {tokens_per_sec:.2f}")


                self.window_step += 1

                # --- Wait for Next Window Start ---
                # Already handled by the loop condition and sleep at the start

            except KeyboardInterrupt:
                 tplr.logger.info(f"[Rank {self.local_rank}] Keyboard interrupt detected. Shutting down.")
                 self.stop_event.set() # Signal other parts if needed
                 break # Exit the while loop
            except Exception as e:
                 tplr.logger.error(f"[Rank {self.local_rank}] Error in main loop: {e}", exc_info=True)
                 # Consider more robust error handling, maybe signal other ranks?
                 await asyncio.sleep(30) # Prevent tight loop on errors


    # --------------------------------------------------------------------------
    # Helper Methods (Potentially need implementation or refinement)
    # --------------------------------------------------------------------------

    def process_gathered_gradients(self, gather_result) -> List[torch.Tensor]:
        """
        Processes the result from comms.gather (decompresses, aggregates)
        and returns a list of final gradient tensors (one per model parameter).
        This MUST be implemented. Runs on Rank 0 only.
        """
        tplr.logger.info("[Rank 0] Processing gathered gradients...")
        if gather_result is None or gather_result.state_dict is None:
             tplr.logger.warning("[Rank 0] No gathered gradients to process. Returning zero gradients.")
             # Return list of zero tensors matching model param shapes/devices
             model_params = list(self.model.module.parameters() if self.is_ddp else self.model.parameters())
             return [torch.zeros_like(p.grad) if p.grad is not None else torch.zeros_like(p) for p in model_params]

        final_grads = {}
        # Use underlying model parameters for reference shapes/devices on Rank 0
        model_to_iterate = self.model.module if self.is_ddp else self.model

        for n, p in model_to_iterate.named_parameters():
            if not p.requires_grad:
                continue

            idxs_key = n + "idxs"
            vals_key = n + "vals"
            idxs_list = getattr(gather_result.state_dict, idxs_key, None)
            vals_list = getattr(gather_result.state_dict, vals_key, None)

            if idxs_list is not None and vals_list is not None:
                if not isinstance(idxs_list, (list, tuple)):
                    idxs_list = [idxs_list]
                if not isinstance(vals_list, (list, tuple)): 
                    vals_list = [vals_list]

                if not idxs_list: # Skip if list is empty
                    final_grads[n] = torch.zeros_like(p)
                    continue

                try:
                    # Decompress and aggregate on CPU (as gather received on CPU)
                    # Ensure the reference parameter 'p' for shape is also on CPU temporarily
                    p_cpu = p.to('cpu')
                    xshape = self.xshapes[n] # From __init__
                    totalk = self.totalks[n] # From __init__

                    # Move indices and values to CPU for decompression if they aren't already
                    idxs_cpu = [i.to('cpu') for i in idxs_list]
                    vals_cpu = [v.to('cpu') for v in vals_list]

                    # Batch decompress requires reference tensor `p`, indices, values, shape, totalk
                    aggregated_grad_cpu = self.transformer.decode(
                         self.compressor.batch_decompress(
                              p_cpu, # Use CPU version for shape ref
                              idxs_cpu,
                              vals_cpu,
                              xshape,
                              totalk
                         )
                    )
                    # Apply SignSGD
                    final_grads[n] = aggregated_grad_cpu.sign()
                    tplr.logger.debug(f"[Rank 0] Processed gradient for {n}")

                except Exception as e:
                    tplr.logger.error(f"[Rank 0] Error processing gradient for {n}: {e}", exc_info=True)
                    final_grads[n] = torch.zeros_like(p) # Zero grad on error
            else:
                # If no gradient was gathered for this param, assume zero
                tplr.logger.debug(f"[Rank 0] No gathered gradient for {n}, using zero.")
                final_grads[n] = torch.zeros_like(p)

        # Ensure the order matches model parameters
        ordered_grads = []
        for n, p in model_to_iterate.named_parameters():
            if p.requires_grad:
                ordered_grads.append(final_grads.get(n, torch.zeros_like(p))) # Default to zeros if missing

        tplr.logger.info(f"[Rank 0] Finished processing {len(ordered_grads)} gathered gradients.")
        return ordered_grads


    # block_listener remains the same as provided before, but needs careful
    # consideration about which rank runs it and how window changes are communicated/handled.
    # For now, removing it from the main class body assuming polling is sufficient.
    # async def block_listener(self, loop): ...

# --- Entry Point ---
if __name__ == "__main__":
    Miner.launch()