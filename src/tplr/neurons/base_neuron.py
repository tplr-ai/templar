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

import asyncio
import concurrent.futures
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import cast

import bittensor as bt
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from transformers import LlamaForCausalLM
import websockets.exceptions

import tplr
from ..common import config as common_config
from ..common.torch_utils import setup_gpu_optimizations, CPU_COUNT


class BaseNeuron(ABC):
    """
    Abstract base class containing all shared logic and initialization steps for Miner and Validator.
    """

    def __init__(self, neuron_type: str):
        """
        Initialize the base neuron with common setup.
        
        Args:
            neuron_type: Either "miner" or "validator"
        """
        tplr.logger.debug("Starting initialization...")
        
        self.neuron_type = neuron_type
        
        # Get config using common config module
        self.config = self._get_config()
        
        # Setup GPU optimizations early
        setup_gpu_optimizations()
        
        # Load hparams
        self.hparams = tplr.load_hparams(use_local_run_hparams=self.config.local)
        
        # Override hparams from config if needed
        self._override_hparams_from_config()
        
        # Initialize components in order
        self._init_bittensor_objects()
        self._init_model_and_tokenizer()
        self._init_compression_artefacts()
        self._init_optimizer_and_scheduler()
        self._init_monitoring()
        self._init_comms()
        self._init_internal_state()

    def _get_config(self) -> bt.Config:
        """Get configuration using common config module."""
        return common_config.get_config(self.neuron_type)

    def _override_hparams_from_config(self) -> None:
        """Override hparams from config if needed."""
        if self.neuron_type == "miner" and self.config.actual_batch_size is not None:
            tplr.logger.info(
                f"Overriding hparams batch size: {self.hparams.batch_size} -> {self.config.actual_batch_size}"
            )
            self.hparams.batch_size = self.config.actual_batch_size

    def _init_bittensor_objects(self) -> None:
        """Initialize Bittensor wallet, subtensor, and metagraph."""
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(cast(int, self.config.netuid))
        
        # Check wallet registration
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            tplr.logger.error(
                f"\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}[/bold]"
            )
            sys.exit()
            
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        
        # Setup enhanced logging with UID if available
        try:
            version = tplr.__version__
            tplr.logger = tplr.setup_loki_logger(
                service=self.neuron_type, uid=str(self.uid), version=version
            )
            tplr.logger.info(f"Loki logging enabled for {self.neuron_type} UID: {self.uid}")
        except Exception as e:
            tplr.logger.warning(f"Failed to initialize Loki logging: {e}")

    def _init_model_and_tokenizer(self) -> None:
        """Initialize model and tokenizer."""
        self.model = LlamaForCausalLM(self.hparams.model_config)
        self.model.to(self.config.device)
        self.tokenizer = self.hparams.tokenizer
        
        # Enable gradient checkpointing for miners
        if self.neuron_type == "miner":
            self.model.gradient_checkpointing_enable()

    def _init_compression_artefacts(self) -> None:
        """Initialize compression transformer and compressor."""
        self.transformer = tplr.compress.TransformDCT(
            self.model, target_chunk=self.hparams.target_chunk
        )
        self.compressor = tplr.compress.CompressDCT(
            use_quantization=True,
            quantization_bins=self.hparams.quantization_bins,
            quantization_range=self.hparams.quantization_range,
        )
        
        # Initialize compression metadata
        self.xshapes = {}
        self.totalks = {}
        
        # Initialize momentum for miners
        if self.neuron_type == "miner":
            self.momentum = {}
        
        for n, p in self.model.named_parameters():
            if self.neuron_type == "miner":
                self.momentum[n] = torch.zeros_like(p)
                
            _, _, xshape, totalk, _ = self.compressor.compress(
                self.transformer.encode(
                    self.momentum[n] if self.neuron_type == "miner" else torch.zeros_like(p)
                ),
                self.hparams.topk_compression,
            )
            self.xshapes[n] = xshape
            self.totalks[n] = totalk

    def _init_optimizer_and_scheduler(self) -> None:
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        
        # Set up scheduler
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

    def _init_monitoring(self) -> None:
        """Initialize WandB and metrics logging."""
        self.bootstrap_version = getattr(self.hparams, "checkpoint_init_version", None)
        tplr.logger.info(
            f"[{self.neuron_type.capitalize()}] code_version={tplr.__version__} "
            f"checkpoint_init_flag={self.bootstrap_version or '<none>'}"
        )
        
        # Initialize WandB
        run_prefix = "M" if self.neuron_type == "miner" else "V"
        self.wandb = tplr.initialize_wandb(
            run_prefix=run_prefix,
            uid=self.uid,
            config=self.config,
            group=self.neuron_type,
            job_type="mining" if self.neuron_type == "miner" else "validation",
        )

        # Initialize metrics logger for InfluxDB
        self.metrics_logger = tplr.metrics.MetricsLogger(
            prefix=run_prefix,
            uid=self.uid,
            config=self.config,
            role=self.neuron_type,
            group=self.neuron_type,
            job_type="mining" if self.neuron_type == "miner" else "validation",
        )

    def _init_comms(self) -> None:
        """Initialize communications."""
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

    def _init_internal_state(self) -> None:
        """Initialize internal state variables."""
        self.stop_event = asyncio.Event()
        self.current_block = self.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        self.start_window = self.current_window
        self.global_step = 0
        
        # Set current window on comms
        self.comms.current_window = self.current_window
        
        # Initialize peer state
        self.next_peers: list[int] | None = None
        self.peers_update_window = -1

    def _block_listener_loop(self, loop_for_thread: asyncio.AbstractEventLoop) -> None:
        """Core logic for the block listener thread."""
        def handler(event):
            try:
                self.current_block = int(event["header"]["number"])
                new_window = int(self.current_block / self.hparams.blocks_per_window)
                if new_window != self.current_window:
                    old_window = self.current_window
                    self.current_window = new_window
                    self.comms.current_window = self.current_window
                    tplr.logger.info(
                        f"New block received. Current window updated: {old_window} -> {self.current_window}"
                    )
            except Exception as e:
                tplr.logger.error(f"Error processing block event: {e}")

        backoff = 1
        max_backoff = 60

        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler)
                backoff = 1
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

    def _start_background_tasks(self) -> None:
        """Initialize and start the executor and listener thread."""
        self.loop = asyncio.get_running_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
        self.loop.set_default_executor(self.executor)

        self.listener_thread = threading.Thread(
            target=self._block_listener_loop, args=(self.loop,), daemon=True
        )
        self.listener_thread.start()

        # Use config peers if provided
        if self.config.peers:
            self.comms.peers = self.config.peers

        # Start comms background tasks
        self.comms.start_background_tasks()

    async def _determine_start_window(self) -> None:
        """Logic to fetch or post the start_window."""
        # Fetch commitments first
        self.comms.commitments = await self.comms.get_commitments()
        tplr.logger.info("Loaded commitments")

        # Handle start window logic
        if self.neuron_type == "validator" and self.uid == self.metagraph.S.argmax().item():
            # Validator: only post start window if highest stake validator
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
                await self.comms.post_start_window(self.start_window)
                tplr.logger.info(
                    f"This validator is the highest staked. Posted start_window: {self.start_window}"
                )
        else:
            # Miner or non-highest stake validator: fetch start window
            if self.neuron_type == "validator":
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

    async def _load_checkpoint_and_catchup(self) -> None:
        """Common checkpoint loading and catchup logic."""
        checkpoint_window_buffer = 5
        has_new_checkpoint = (
            self.global_step >= self.hparams.checkpoint_frequency + checkpoint_window_buffer
        )

        # Load checkpoint
        (
            success,
            loaded_checkpoint_window,
            loaded_optimizer,
            loaded_scheduler,
        ) = await self.comms.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            current_window=self.current_window,
            device=cast(str, self.config.device),
            init_version=tplr.__version__ if has_new_checkpoint else self.bootstrap_version,
        )

        if success:
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
            if self.neuron_type == "miner":
                tplr.logger.info("No checkpoint found, initializing model from scratch")
                self.momentum = {
                    n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                }
                self.model.to(self.config.device)
                
                # Catch up with aggregation server from start window
                tplr.logger.info(
                    f"Starting catchup from start window {self.start_window} to current window {self.current_window})..."
                )
                await tplr.neurons.catchup_with_aggregation_server(self, self.start_window)
            else:
                tplr.logger.info("Starting from scratch")
                self.model.to(self.config.device)

        # Start commitment fetcher
        self.comms.start_commitment_fetcher()

    async def run(self) -> None:
        """Main orchestrator method."""
        # Start background tasks
        self._start_background_tasks()
        
        # Determine start window
        await self._determine_start_window()
        
        # Load checkpoint and catchup
        await self._load_checkpoint_and_catchup()
        
        try:
            # Run the main loop (implemented by subclasses)
            await self.main_loop()
        finally:
            # Cleanup resources
            await self.cleanup_resources()

    @abstractmethod
    async def main_loop(self) -> None:
        """Abstract method to be implemented by MinerCore and ValidatorCore."""
        pass

    async def cleanup_resources(self) -> None:
        """Close comms resources."""
        try:
            await self.comms.close_all_resources()
            tplr.logger.info("Successfully cleaned up communications resources")
        except Exception as e:
            tplr.logger.error(f"Error cleaning up communications: {e}") 