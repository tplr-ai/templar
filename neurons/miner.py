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
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast

import bittensor as bt
import numpy as np
import torch
import torch.distributed as dist
import uvloop
from torch.amp.grad_scaler import GradScaler
from torch.distributed.tensor import DTensor as DT

import tplr
from neurons import BaseNode, Trainer
from neurons.base_node import CPU_COUNT

# GPU optimizations
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Miner(BaseNode, Trainer):
    def log_gpu_memory(self, stage: str):
        """Log current GPU memory allocation and reservation"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            tplr.logger.info(
                f"[GPU Memory - {stage}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
            )

    def check_memory_threshold(self, threshold_gb: float = 0.5):
        """Check if available memory is below threshold and cleanup if needed"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            max_memory = (
                torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            )
            available = max_memory - allocated

            if available < threshold_gb:
                tplr.logger.warning(f"Low GPU memory: {available:.2f} GB available")
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)
                self.log_gpu_memory("After emergency cleanup")

    # Command line config items.
    @staticmethod
    def miner_config():
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
            "--device", type=str, default="cuda", help="Device to use for training"
        )
        parser.add_argument(
            "--amp-dtype",
            choices=["bf16", "fp16"],
            default="bf16",
            help="Mixed-precision data type. Use «fp16» to enable GradScaler.",
        )
        parser.add_argument(
            "--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0))
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
            "--profile-iters",
            type=int,
            default=0,
            help="Active iterations per Torch‑Profiler trace (0 = disable)",
        )
        parser.add_argument(
            "--profile-dir",
            type=str,
            default="./log/profiler",
            help="Directory to save profiler traces",
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

    @staticmethod
    def should_continue(local_has_batch: bool, device) -> bool:
        """
        Synchronize across all ranks. If *any* rank runs out of batches, all must stop.
        """
        flag_tensor = torch.tensor([int(local_has_batch)], device=device)
        dist.all_reduce(flag_tensor, op=dist.ReduceOp.MIN)
        return bool(flag_tensor.item())

    def __init__(self):
        tplr.logger.debug("Starting initialization...")

        # Init config and load hparams
        self.config = Miner.miner_config()
        # ---------------------------------------------------------------------
        # Distributed initialisation
        # ---------------------------------------------------------------------
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        tplr.logger.info(
            f"[Init] rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank}"
        )

        if self.world_size >= 1:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=45),
                rank=self.rank,
                world_size=self.world_size,
            )
            torch.cuda.set_device(self.local_rank)
            tplr.logger.info("[Init] NCCL process-group ready and GPU selected")
            self.config.device = f"cuda:{self.local_rank}"
        else:
            self.config.device = self.config.device or "cuda"
        self.device = torch.device(self.config.device)
        tplr.logger.info(f"[Init] device set → {self.device}")

        # Mixed precision setup
        self.amp_dtype = (
            torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        )
        self.scaler = GradScaler(
            enabled=(self.amp_dtype is torch.float16 and self.device.type == "cuda")
        )
        tplr.logger.info(
            f"[Init] Using {self.config.amp_dtype}. GradScaler enabled: {self.scaler.is_enabled()}"
        )

        # Convenience flags
        self.is_master = self.rank == 0
        self.config.local = cast(bool, self.config.local)
        self.hparams = tplr.load_hparams(use_local_run_hparams=self.config.local)

        if self.config.actual_batch_size is not None:
            tplr.logger.info(
                f"Overriding hparams batch size: {self.hparams.batch_size} -> {self.config.actual_batch_size}"
            )
            self.hparams.batch_size = self.config.actual_batch_size

        # Init bittensor objects
        self.wallet = bt.wallet(config=self.config)
        tplr.logger.info("[Init] Bittensor wallet loaded")
        super().__init__()

        self.init_model()
        self.bare_model = getattr(self.model, "module", self.model)

        # Store parallelization parameters for later use
        tt = getattr(self.hparams, "torchtitan", SimpleNamespace())
        self.tp_degree = int(getattr(tt, "tp_degree", 1))
        self.pp_degree = int(getattr(tt, "pp_degree", 1))
        self.cp_degree = int(getattr(tt, "cp_degree", 1))
        self.dp_replicate = int(getattr(tt, "dp_replicate", 1))
        self.dp_shard = int(getattr(tt, "dp_shard", 1))

        # Init compression
        self.transformer = tplr.compress.ChunkingTransformer(
            self.model, target_chunk=self.hparams.target_chunk
        )
        self.compressor = tplr.compress.TopKCompressor(
            use_quantization=True,
            quantization_bins=self.hparams.quantization_bins,
            quantization_range=self.hparams.quantization_range,
        )
        tplr.logger.info("[Init] compression pipeline ready")

        # Init optimizer and momentum
        self.init_optimizers_schedulers()

        self.error_feedback = {}
        self.error_feedback_cpu_buffers = {}
        self.owned_params = set()

        self.xshapes = {}
        self.totalks = {}
        model_iterator = self.bare_model.named_parameters()

        for idx, (n, p) in enumerate(model_iterator):
            if idx % self.world_size == self.rank:
                # this rank "owns" the parameter
                self.owned_params.add(n)
                # For DTensors, create error feedback based on full tensor since TP is not supported
                self.error_feedback[n] = None
                self.error_feedback_cpu_buffers[n] = torch.empty(
                    p.shape, device="cpu", pin_memory=True
                )

            enc = self.transformer.encode(
                torch.empty(p.shape, dtype=torch.float16, device=self.device),
                use_dct=self.hparams.use_dct,
            )
            _, _, xshape, totalk, _ = self.compressor.compress(
                enc,
                self.hparams.topk_compression,
            )
            self.xshapes[n] = xshape
            self.totalks[n] = totalk

        tplr.logger.info(
            f"[Init] Compression initialized for {len(self.xshapes)} parameters"
        )

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
            hparams=self.hparams,
            uid=None,  # UID will be set after comms is initialized
        )

        if self.wallet.hotkey.ss58_address not in self.comms.metagraph.hotkeys:
            tplr.logger.error(
                f"\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.comms.metagraph.netuid}[/bold]"
            )
            sys.exit()
        self.uid = self.comms.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.comms.uid = self.uid

        self.bucket = self.comms.get_own_bucket("gradients", "read")
        if self.is_master:
            self.comms.try_commit(self.wallet, self.bucket)
        if self.world_size > 1:
            dist.barrier(device_ids=[self.local_rank])

        # Init state params
        self.current_block = self.comms.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        tplr.logger.info(
            f"[Init] chain at block {self.current_block}, window {self.current_window}"
        )

        self.start_window = self.current_window  # Record the start window
        self.global_step = 0  # Initialize global_step to zero
        self.comms.current_window = self.current_window
        self.step_counter = 0

        # Track additional metrics
        self.total_tokens_processed = 0

        if self.is_master:
            # Initialize WandB
            self.wandb = tplr.initialize_wandb(
                run_prefix="M",
                uid=self.uid,
                config=self.config,
                group="miner",
                job_type="mining",
            )
            tplr.logger.info("[Init] WandB session started")

            # Initialize metrics logger for InfluxDB
            self.metrics_logger = tplr.metrics.MetricsLogger(
                prefix="M",
                uid=self.uid,
                config=self.config,
                role="miner",
                group="miner",
                job_type="mining",
            )

        # Initialize peer related attributes
        self.next_peers: list[int] | None = None
        self.next_reserve_peers: list[int] | None = None
        self.peers_update_window = -1

        self.dataset_manager = tplr.sharded_dataset.ShardedDatasetManager(
            sequence_length=self.hparams.sequence_length,
            rank=self.local_rank,
            world_size=self.world_size,
            comms=self.comms,
        )
        self.windows_per_shard = getattr(self.hparams, "windows_per_shard")

        tplr.logger.info("[Init] ✔ fully done – entering run()")

    # Main training loop.
    async def run(self):
        # Start background block listener
        self.loop = asyncio.get_running_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
        self.loop.set_default_executor(self.executor)

        # Use config peers if provided
        if self.config.peers:
            self.comms.peers = self.config.peers

        self.comms.commitments = await self.comms.get_commitments()
        tplr.logger.info("Loaded commitments")

        peer_start = tplr.T()
        # Fetch peers and get start_window from highest stake validator
        if self.is_master:
            await tplr.neurons.update_peers(
                instance=self, window=self.current_window, peer_start=peer_start
            )

            self.start_window = await self.comms.get_start_window()
            tplr.logger.info(f"Using start_window: {self.start_window}")

            val = -1 if self.start_window is None else self.start_window
            tensor = torch.tensor([val], dtype=torch.long, device=self.device)
            dist.broadcast(tensor, src=0)
        else:
            tensor = torch.zeros(1, dtype=torch.long, device=self.device)
            dist.broadcast(tensor, src=0)
            val = tensor.item()
            self.start_window = None if val == -1 else int(val)

        self.global_step = self.current_window - self.start_window
        current_shard = self.global_step // self.windows_per_shard
        tplr.logger.info(f"starting at Global Step : {self.global_step}")

        if self.is_master:
            _ = await self.dataset_manager.initialize_datasets(current_shard)
            dist.barrier(device_ids=[self.local_rank])

        else:
            # barrier to start so that master finalized the dataset download
            dist.barrier(device_ids=[self.local_rank])
            await self.dataset_manager.initialize_datasets(current_shard)

        # All workers need to instantiate dataloader
        self.set_dataloader()

        checkpoint_window_buffer = 5
        has_new_checkpoint = (
            self.global_step
            >= self.hparams.checkpoint_frequency + checkpoint_window_buffer
        )
        # ------------------------------------------------------------------
        # Proceed to load checkpoint
        #   • rank-0 (or single-GPU run) downloads & catches-up
        #   • remaining ranks receive state via NCCL broadcast
        # ------------------------------------------------------------------

        ckpt_ok, ckpt_sync_win = await self.comms.load_checkpoint(
            model=self.bare_model,
            current_window=self.current_window,
            init_version=tplr.__version__
            if has_new_checkpoint
            else self.bootstrap_version,
            is_master=self.is_master,
        )
        if ckpt_ok:
            tplr.logger.info(f"Checkpoint loaded (sync_window={ckpt_sync_win})")
        else:
            tplr.logger.info("No checkpoint found – starting from scratch")

        # Decide catch-up windows and run catch-up on ALL ranks (avoids DTensor collective hangs)
        need_catchup = (not ckpt_ok) or (
            ckpt_ok
            and ckpt_sync_win < self.current_window
            and self.global_step > checkpoint_window_buffer
        )
        if need_catchup:
            start_from = (
                self.start_window
                if not ckpt_ok
                else max(ckpt_sync_win, self.start_window)
            )
            await tplr.neurons.catchup_with_aggregation_server(self, start_from)

        if ckpt_ok:
            steps_to_replay = (
                ckpt_sync_win - self.start_window + 1
            ) * self.hparams.inner_steps
            for _ in range(steps_to_replay):
                self.inner_scheduler.step()

        self.comms.start_commitment_fetcher()

        # Put a dummy gradient to mark this miner as active for validators
        if self.is_master:
            tplr.logger.info("Putting dummy gradient to mark miner as active...")
            dummy_gradient = {
                "metadata": {"window": self.current_window, "dummy": True}
            }
            await self.comms.put(
                state_dict=dummy_gradient,
                uid=str(self.uid),
                window=self.current_window,
                key="gradient",
                local=False,
            )
            tplr.logger.info("Dummy gradient posted successfully")

        while not self.stop_event.is_set():
            await asyncio.sleep(0)
            # 1. Initialize window and update peers
            window_start = tplr.T()
            # Start the gather in the background:
            step_window = self.current_window
            self.global_step = (
                self.current_window - self.start_window
            )  # Update global_step
            tplr.logger.info(
                f"\n{'-' * 40} Window: {step_window} (Global Step: {self.global_step}) {'-' * 40}"
            )

            peer_start = tplr.T()
            if self.is_master:
                await tplr.neurons.update_peers(
                    instance=self, window=step_window, peer_start=peer_start
                )
            peer_update_time = tplr.T() - peer_start

            # 2. Load data
            data_start = tplr.T()

            # Update sampler for current window
            self.sampler.set_window_uid(self.uid, step_window)

            if self.global_step > 0 and self.global_step % self.windows_per_shard == 0:
                tplr.logger.info(f"Swapping dataset at window {step_window}")
                await self.dataset_manager.swap_datasets()
                self.set_dataloader()
                if self.world_size > 1:
                    dist.barrier(device_ids=[self.local_rank])

            data_loading_time = tplr.T() - data_start
            tplr.logger.info(
                f"{tplr.P(step_window, data_loading_time)} Loaded training data"
            )

            # 3. Accumulate gradients over batches
            train_start = tplr.T()
            tplr.logger.info("Start accumulating...")
            res = await self.inner_steps(loader=self.loader, step_window=step_window)
            training_time = tplr.T() - train_start
            window_entry_loss = res["window_entry_loss"]
            n_batches = res["batch_count"]
            window_tokens = res["batch_tokens"]

            # If training finishes early, wait until the *next* chain-window starts.
            if self.current_window == step_window:
                tplr.logger.info(
                    "Training complete; waiting for window to be exhausted..."
                )
                await self.wait_until_window(step_window + 1)
            tplr.logger.info(
                f"{tplr.P(step_window, tplr.T() - train_start)} Completed training"
            )

            # Synchronise all ranks
            if self.world_size > 1:
                dist.barrier(device_ids=[self.local_rank])

            # 1️⃣ every rank builds its momentum shard
            compress_start = tplr.T()
            shard_gradient, _, _ = tplr.prepare_gradient_dict(self, step_window)
            compression_time = tplr.T() - compress_start
            tplr.logger.info(
                f"{tplr.P(step_window, compression_time)} "
                f"Compressed local shard with {len(shard_gradient) - 1} tensors"
            )

            # gather the shards → rank-0
            if self.world_size > 1:
                gathered = [None] * self.world_size
                dist.gather_object(  # NCCL / Gloo friendly
                    shard_gradient,
                    gathered if self.is_master else None,
                    dst=0,
                )
            else:  # single-GPU run
                gathered = [shard_gradient]

            # ------------------------------------------------------------
            #  rank-0 merges & uploads the full gradient
            # ------------------------------------------------------------
            gradient = {}
            processed_state_dict = {}
            if self.is_master:
                for i, shard in enumerate(gathered):
                    if shard is not None:
                        gradient.update(shard)
                        gathered[i] = None  # Free memory immediately after using shard

                # dataset metadata
                gidx = self.sampler._global_indices()
                ids = self.sampler.ids_for_indices(gidx.tolist())
                h = hashlib.blake2b(digest_size=16)
                h.update(np.asarray(sorted(ids), dtype=np.uint64).tobytes())
                sample_digest = h.hexdigest()
                sample_count = len(ids)

                # ── attach window + sample receipt ─────────────────────
                gradient["metadata"] = {
                    "window": step_window,
                    "sample_digest": sample_digest,
                    "sample_count": sample_count,
                }
                tplr.logger.info(
                    f"Attached metadata to gradient: {gradient['metadata']}"
                )

                tplr.logger.info(
                    f"Merged {len(gathered)} shards → {len(gradient) - 1} tensors"
                )
                del gathered  # Free gathered list after merging

                # move to CPU before R2 upload
                processed_state_dict = {
                    k: (v.to("cpu") if isinstance(v, torch.Tensor) else v)
                    for k, v in gradient.items()
                }

                put_start = tplr.T()
                await self.comms.put(
                    state_dict=processed_state_dict,
                    uid=str(self.uid),
                    window=step_window,
                    key="gradient",
                    global_step=self.global_step,
                    local=False,
                    stale_retention=100,
                )

                upload_size = sum(
                    t.element_size() * t.nelement()
                    for t in processed_state_dict.values()
                    if isinstance(t, torch.Tensor)
                )
                put_time = tplr.T() - put_start  # ⏱ done
                tplr.logger.info(
                    f"Uploaded {upload_size / 1e6:.1f} MB shard-merged gradient"
                )

                # Free memory immediately after upload
                del processed_state_dict
                del gradient
                torch.cuda.empty_cache()

            else:
                # non-master ranks simply wait; they don't upload
                put_time = 0.0
                if self.world_size > 1:
                    del gathered  # Free gathered list on non-master ranks too

            tplr.logger.info(f"Stopped accumulating: {n_batches} batches")
            if self.world_size > 1:
                dist.barrier(device_ids=[self.local_rank])

            sync_block = self.current_window * self.hparams.blocks_per_window
            ts_value = await self.loop.run_in_executor(
                None, self.query_block_timestamp, sync_block
            )
            if ts_value is None:
                tplr.logger.warning(
                    f"Could not get timestamp for sync block {sync_block}. Using current time as fall back.",
                )
                ts_value = time.time()
            time_min = datetime.fromtimestamp(ts_value, tz=timezone.utc)
            time_max = time_min + timedelta(
                seconds=self.hparams.time_window_delta_seconds
            )

            # Log the time window we're using
            tplr.logger.info(f"Using time window for gather: {time_min} to {time_max}")

            if self.config.test:
                # In test mode, use all UIDs from metagraph except self
                tplr.logger.info("Test mode active: Using all peers from metagraph.")
                all_uids = list(range(1, len(self.comms.metagraph.S)))
                self.comms.peers = [uid for uid in all_uids if uid != self.uid]

            tplr.logger.info(f"Final peers for gather: {self.comms.peers}")

            gather_result = None
            gather_time = 0.0
            if self.is_master:
                gather_start = tplr.T()
                tplr.logger.info("Waiting on gather task...")
                gather_result = await self.comms.gather_with_reserve(
                    my_uid=self.uid,
                    gather_uids=self.comms.peers,
                    reserve_uids=self.comms.reserve_peers,
                    window=step_window,
                    key="gradient",
                    timeout=75,
                    device=str(self.device),
                    local=False,
                    stale_retention=100,
                    totalks=self.totalks,
                    compressor=self.compressor,
                    time_min=time_min,
                    time_max=time_max,
                    expected_compressed_params=self.expected_compressed_params,
                )
                tplr.logger.info("Gather task completed!")
                gather_time = tplr.T() - gather_start

            # 5. Calculate and log metrics
            self.total_tokens_processed += window_tokens
            tokens_per_sec = window_tokens / training_time if training_time else 0.0

            # ─────────────── gradient & weight norms (local) ────────────────
            grad_norms = [
                p.grad.norm().item()
                for p in self.model.parameters()
                if p.grad is not None
            ]
            weight_norms = [p.norm().item() for p in self.model.parameters()]

            # ---------------------------------------------------------------------
            # 6. Await both gather
            # ---------------------------------------------------------------------

            # 8. Apply gathered gradients
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            update_start = tplr.T()
            _ = self.outer_step(gather_result)

            model_update_time = tplr.T() - update_start
            tplr.logger.info(f"{tplr.P(step_window, model_update_time)} Updated model")

            if self.is_master:
                # Add debug data including successfully gathered peers
                debug_dict = {}

                # Add model parameters debug info
                for name, param in self.bare_model.named_parameters():
                    if param is not None:
                        # Handle DTensor vs regular tensor
                        if isinstance(param, DT):
                            local_param = param.to_local()
                            if local_param.numel() >= 2:
                                debug_dict[name + "_debug"] = (
                                    local_param.flatten()[10:12].detach().cpu().tolist()
                                )
                        else:
                            if param.numel() >= 2:
                                debug_dict[name + "_debug"] = (
                                    param.flatten()[10:12].detach().cpu().tolist()
                                )

                # Add successful peers information
                if gather_result is not None:
                    debug_dict["successful_peers"] = sorted(
                        list(set(self.comms.peers) - set(gather_result.skipped_uids))
                    )
                    debug_dict["skipped_peers"] = sorted(
                        list(gather_result.skipped_uids)
                    )

                # Store the debug dictionary
                await self.comms.put(
                    state_dict=debug_dict,
                    uid=str(self.uid),
                    window=step_window,
                    key="debug",
                    local=False,
                )

                tplr.logger.info(
                    f"Stored debug values for window {self.current_window}"
                )

            # Log total window time and metrics
            tplr.logger.info(
                f"{tplr.P(self.current_window, tplr.T() - window_start)} Completed window iteration"
            )

            # ─────────────── momentum norms (gathered across ranks) ─────────
            local_mom_norms: list[float] = [
                m.norm().item() for m in self.error_feedback.values()
            ]
            if self.world_size > 1:
                gathered_mom: list[list[float]] = [None] * self.world_size  # type: ignore[var-annotated]
                dist.all_gather_object(gathered_mom, local_mom_norms)
            else:
                gathered_mom = [local_mom_norms]

            momentum_norms = []
            # Log metrics to WandB
            if self.is_master:
                # Calculate common metrics values
                momentum_norms: list[float] = sum(gathered_mom, [])
                mean_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
                grad_norm_std = (
                    torch.tensor(grad_norms).std().item() if grad_norms else 0
                )
                mean_weight_norm = (
                    sum(weight_norms) / len(weight_norms) if weight_norms else 0
                )
                mean_momentum_norm = (
                    sum(momentum_norms) / len(momentum_norms) if momentum_norms else 0
                )
                window_total_time = tplr.T() - window_start
                gather_success_rate = (
                    gather_result.success_rate * 100 if gather_result else 0.0
                )
                inner_lr = self.inner_scheduler.get_last_lr()[0]

                self.wandb.log(
                    {
                        # Add timing metrics
                        "miner/timing/window_total": window_total_time,
                        "miner/timing/peer_update": peer_update_time,
                        "miner/timing/data_loading": data_loading_time,
                        "miner/timing/training": training_time,
                        "miner/timing/compression": compression_time,
                        "miner/timing/gather": gather_time,
                        "miner/timing/put": put_time,
                        "miner/timing/model_update": model_update_time,
                        # Existing metrics
                        "miner/window_entry_loss": window_entry_loss,
                        "miner/tokens_per_sec": tokens_per_sec,
                        "miner/total_tokens": self.total_tokens_processed,
                        "miner/batch_tokens": window_tokens,
                        "miner/global_step": self.global_step,
                        "miner/gpu_memory_allocated": torch.cuda.memory_allocated()
                        / 1024**2,
                        "miner/gpu_memory_cached": torch.cuda.memory_reserved()
                        / 1024**2,
                        "miner/gather_peers": len(self.comms.peers),
                        "miner/effective_batch_size": len(self.comms.peers)
                        * self.hparams.batch_size,
                        "miner/inner_lr": inner_lr,
                        "miner/mean_grad_norm": mean_grad_norm,
                        "miner/max_grad_norm": max(grad_norms) if grad_norms else 0,
                        "miner/min_grad_norm": min(grad_norms) if grad_norms else 0,
                        "miner/grad_norm_std": grad_norm_std,
                        "miner/mean_weight_norm": mean_weight_norm,
                        "miner/mean_momentum_norm": mean_momentum_norm,
                        # Added gather success rate in %
                        "miner/gather/success_rate": gather_success_rate,
                    },
                    step=self.global_step,
                )

                self.metrics_logger.log(
                    measurement="training_step_v2",
                    tags={
                        "window": self.current_window,
                        "global_step": self.global_step,
                    },
                    fields={
                        "loss": window_entry_loss,
                        "n_gather_peers": int(len(self.comms.peers)),
                        "gather_success_rate": gather_success_rate,
                        "gather_peers": json.dumps(self.comms.peers),
                        "skipped_peers": json.dumps(
                            gather_result.skipped_uids if gather_result else []
                        ),
                        "window_total_time": window_total_time,
                        "peer_update_time": peer_update_time,
                        "compression_time": compression_time,
                        "gather_time": gather_time,
                        "put_time": put_time,
                        "model_update_time": model_update_time,
                        "tokens_per_sec": tokens_per_sec,
                    },
                )
                tplr.logger.info("Finished metrics logging call for miner")

            self.global_step += 1
            tplr.logger.info(f"Total optimization steps: {self.global_step}")

            if self.world_size > 1:
                dist.barrier(device_ids=[self.local_rank])

            # Delete any remaining local variables to clear up memory
            del shard_gradient
            if gather_result is not None:
                del gather_result
            torch.cuda.empty_cache()
            # Check memory threshold periodically
            self.check_memory_threshold(threshold_gb=0.5)

            await self.cleanup_window()

            # 4. Wait for next window
            tplr.logger.info("Wait for next window...")
            await self.wait_until_window(step_window + 1)

    def _get_offloaded_param(self):
        """Get a copy of current parameters and offload them to CPU"""
        params_offloaded = []
        param_info = []  # Store DTensor info for restoration

        for param in self.bare_model.parameters():
            if isinstance(param, DT):
                # Get the local TP shard and store the spec
                local_param = param.to_local()
                params_offloaded.append(local_param.detach().clone().to("cpu"))
                param_info.append(
                    {  # Store the DTensor placement info
                        "is_dtensor": True,
                        "device_mesh": param.device_mesh,
                        "placements": param.placements,
                        "local_shape": local_param.shape,
                    }
                )
            else:
                # For regular tensors
                params_offloaded.append(param.data.detach().clone().to("cpu"))
                param_info.append({"is_dtensor": False})

        return params_offloaded, param_info

    async def cleanup_window(self):
        """Aggressive memory cleanup between windows"""
        import gc

        # Clear gradients more thoroughly
        self.model.zero_grad(set_to_none=True)
        self.inner_optimizer.zero_grad(set_to_none=True)

        # Clear error feedback for non-owned params to save memory
        for name in list(self.error_feedback.keys()):
            if name not in self.owned_params and self.error_feedback[name] is not None:
                self.error_feedback[name] = None

        # Clear any cached autocast states
        torch.clear_autocast_cache()

        # Empty CUDA cache multiple times for thorough cleanup
        for _ in range(3):
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

        # Force garbage collection
        gc.collect()

        # Check memory threshold after cleanup
        self.check_memory_threshold(threshold_gb=1.0)

        # Log memory status
        tplr.logger.info(
            f"After cleanup - GPU allocated: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB"
        )
        tplr.logger.info(
            f"After cleanup - GPU reserved: {torch.cuda.memory_reserved(self.device) / 1024**3:.2f} GB"
        )


# Start miner.
if __name__ == "__main__":
    uvloop.install()
    try:
        asyncio.run(Miner().main())
    except KeyboardInterrupt:
        pass
