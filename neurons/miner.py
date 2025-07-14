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
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from typing import cast

import bittensor as bt
import numpy as np
import torch
import torch.distributed as dist

# Third party
import torch.nn.parallel
import uvloop
from torch import autocast
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from transformers.models.llama import LlamaForCausalLM

import tplr

# Local
from neurons import BaseNode

# Local

CPU_COUNT = os.cpu_count() or 4
CPU_MAX_CONNECTIONS = min(100, max(30, CPU_COUNT * 4))

# GPU optimizations
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Miner(BaseNode):
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
            "--save-location",
            type=str,
            default="/tmp",
            help="Directory to save temporary files, gradients, checkpoints, and debug data.",
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

    def _is_distributed(self) -> bool:
        """True iff torch.distributed is initialised and world_size > 1."""
        return dist.is_available() and dist.is_initialized() and self.world_size > 1

    def _ddp_reduce(
        self,
        value: int | float | torch.Tensor,
        op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
    ) -> float:
        """
        Reduce ``value`` across all ranks and return a **python float**.
        Use ``op=dist.ReduceOp.AVG`` for mean; default is SUM.
        """
        # single-GPU fast path
        if not self._is_distributed():
            return float(value.item() if isinstance(value, torch.Tensor) else value)

        # convert to tensor on the right device
        if not isinstance(value, torch.Tensor):
            tensor = torch.tensor(float(value), device=self.device)
        else:
            tensor = value.to(self.device)

        dist.all_reduce(tensor, op=op)
        return float(tensor.item())

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
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(cast(int, self.config.netuid))
        tplr.logger.info("[Init] Bittensor wallet/metagraph loaded")
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            tplr.logger.error(
                f"\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}[/bold]"
            )
            sys.exit()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        super().__init__()

        # Init model with hparams config
        self.model = LlamaForCausalLM(self.hparams.model_config)
        self.model.to(self.device)  # type: ignore[reportArgumentType]
        if self.world_size < 4:
            self.model.gradient_checkpointing_enable()
        tplr.logger.info("[Init] Llama model instantiated & on device")

        compile_mode = "default"
        self.model = cast(
            LlamaForCausalLM, torch.compile(self.model, mode=compile_mode)
        )

        if self.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                gradient_as_bucket_view=True,
            )
            tplr.logger.info("[Init] wrapped model with DistributedDataParallel")
        self.tokenizer = self.hparams.tokenizer

        # Init compression
        self.transformer = tplr.compress.TransformDCT(
            self.model, target_chunk=self.hparams.target_chunk
        )
        self.compressor = tplr.compress.CompressDCT(
            use_quantization=True,
            quantization_bins=self.hparams.quantization_bins,
            quantization_range=self.hparams.quantization_range,
        )
        tplr.logger.info("[Init] compression pipeline ready")

        # Init optimizer and momentum
        self.error_feedback = {}
        self.owned_params = set()

        self.xshapes = {}
        self.totalks = {}
        model_iterator = (
            self.model.module.named_parameters()
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model.named_parameters()
        )

        self.outer_optimizer = SGD(
            self.model.parameters(), lr=self.hparams.outer_learning_rate
        )
        self.inner_optimizer = ZeroRedundancyOptimizer(
            self.model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=self.hparams.inner_learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
            parameters_as_bucket_view=True,
            overlap_with_ddp=False,
        )
        inner_steps_before_outer_step = self.hparams.inner_steps * (
            self.hparams.validator_offset + self.hparams.peer_list_window_margin + 1
        )
        init_scheduler = LinearLR(
            self.inner_optimizer,
            start_factor=0.1,
            end_factor=0.1,
            total_iters=inner_steps_before_outer_step,
        )
        warmup_scheduler = LinearLR(
            self.inner_optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.inner_optimizer,
            T_max=self.hparams.t_max,
            eta_min=self.hparams.inner_learning_rate * 0.1,
        )
        self.inner_scheduler = SequentialLR(
            self.inner_optimizer,
            schedulers=[init_scheduler, warmup_scheduler, cosine_scheduler],
            milestones=[inner_steps_before_outer_step, self.hparams.warmup_steps],
        )
        tplr.logger.info("[Init] optimizers & schedulers constructed")

        for idx, (n, p) in enumerate(model_iterator):
            if idx % self.world_size == self.rank:
                # this rank “owns” the parameter
                self.owned_params.add(n)
                self.error_feedback[n] = torch.zeros_like(p, device=self.device)
            _, _, xshape, totalk, _ = self.compressor.compress(
                self.transformer.encode(
                    torch.zeros_like(p), use_dct=self.hparams.use_dct
                ),
                self.hparams.topk_compression,
            )
            self.xshapes[n] = xshape
            self.totalks[n] = totalk

        self.bootstrap_version = getattr(self.hparams, "checkpoint_init_version", None)
        tplr.logger.info(
            f"[Miner] code_version={tplr.__version__} "
            f"checkpoint_init_flag={self.bootstrap_version or '<none>'}"
        )

        # Init comms
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location=self.config.save_location,
            key_prefix="model",
            config=self.config,
            netuid=self.config.netuid,
            metagraph=self.metagraph,
            hparams=self.hparams,
            uid=self.uid,
        )

        self.bucket = self.comms.get_own_bucket("gradients", "read")
        if self.is_master:
            self.comms.try_commit(self.wallet, self.bucket)
        if self.world_size > 1:
            dist.barrier(device_ids=[self.local_rank])

        # Init state params
        self.current_block = self.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        tplr.logger.info(
            f"[Init] chain at block {self.current_block}, window {self.current_window}"
        )

        self.start_window = self.current_window  # Record the start window
        self.global_step = 0  # Initialize global_step to zero
        self.comms.current_window = self.current_window
        self.step_counter = 0

        # Add step tracking
        self.window_step = 0

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
        self.peers_update_window = -1
        self.dataset = tplr.SharedShardedDataset(
            sequence_length=self.hparams.sequence_length,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.sampler = tplr.MinerSampler(
            dataset=self.dataset,
            uid=self.uid,
            window=self.current_window,
            steps_per_window=self.hparams.inner_steps,
            micro_bs=self.hparams.micro_batch_size,
            batch_size=self.hparams.batch_size,
            target_batch_size=self.hparams.target_batch_size,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            sampler=self.sampler,
            batch_size=self.hparams.micro_batch_size,
            num_workers=2,
            pin_memory=True,
        )

        tplr.logger.info("[Init] dataset + sampler ready")
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
        if self.is_master:
            await tplr.neurons.update_peers(
                instance=self, window=self.current_window, peer_start=peer_start
            )

        # Fetch start_window from highest stake validator
        if self.is_master:
            self.start_window = await self.comms.get_start_window()
            val = -1 if self.start_window is None else self.start_window
            tensor = torch.tensor([val], dtype=torch.long, device=self.device)
            dist.broadcast(tensor, src=0)
        else:
            tensor = torch.zeros(1, dtype=torch.long, device=self.device)
            dist.broadcast(tensor, src=0)
            val = tensor.item()
            self.start_window = None if val == -1 else int(val)
        if self.start_window is None:
            raise RuntimeError(
                "Could not find a valid start window. This should not be possible."
            )

        tplr.logger.info(f"Using start_window: {self.start_window}")

        self.global_step = self.current_window - self.start_window
        tplr.logger.info(f"starting at Global Step : {self.global_step}")

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

        bare_model = (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )

        if self.world_size == 1 or self.is_master:
            (
                ckpt_ok,
                ckpt_sync_win,
            ) = await self.comms.load_checkpoint(
                model=bare_model,
                current_window=self.current_window,
                device=str(self.device),
                init_version=tplr.__version__
                if has_new_checkpoint
                else self.bootstrap_version,
            )

            if ckpt_ok:
                tplr.logger.info(f"Checkpoint loaded (sync_window={ckpt_sync_win})")

                # catch-up only if the checkpoint lags behind
                if (
                    ckpt_sync_win < self.current_window
                    and self.global_step > checkpoint_window_buffer
                ):
                    await tplr.neurons.catchup_with_aggregation_server(
                        self, max(ckpt_sync_win, self.start_window)
                    )

            else:
                tplr.logger.info("No checkpoint found – starting from scratch")

                # still perform the full catch-up from the very first window
                await tplr.neurons.catchup_with_aggregation_server(
                    self, self.start_window
                )

        # ---- broadcast to other ranks (if any) --------------------------------
        if self.world_size > 1:
            bcast_start = tplr.T()

            # 1) parameters & buffers
            for tensor in bare_model.state_dict().values():
                if torch.is_tensor(tensor):
                    dist.broadcast(tensor.data, src=0)

            bcast_time = tplr.T() - bcast_start
            tplr.logger.info(
                f"{tplr.P(self.current_window, bcast_time)} "
                f"Broadcast checkpoint to {self.world_size - 1} ranks"
            )
            dist.barrier(device_ids=[self.local_rank])

        self.comms.start_commitment_fetcher()

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
                for shard in gathered:
                    if shard is not None:
                        gradient.update(shard)

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

            else:
                # non-master ranks simply wait; they don't upload
                put_time = 0.0

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
                all_uids = list(range(len(self.metagraph.S)))
                self.comms.peers = [uid for uid in all_uids if uid != self.uid]

            tplr.logger.info(f"Final peers for gather: {self.comms.peers}")

            gather_result = None
            gather_time = 0.0
            if self.is_master:
                gather_start = tplr.T()
                tplr.logger.info("Waiting on gather task...")
                gather_result = await self.comms.gather(
                    my_uid=self.uid,
                    uids=self.comms.peers,
                    window=step_window,
                    key="gradient",
                    timeout=60,
                    device=str(self.device),
                    local=False,
                    stale_retention=100,
                    totalks=self.totalks,
                    compressor=self.compressor,
                    time_min=time_min,
                    time_max=time_max,
                )
                tplr.logger.info("Gather task completed!")
                gather_time = tplr.T() - gather_start
            if self.world_size > 1:
                dist.barrier(device_ids=[self.local_rank])

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
            update_start = tplr.T()
            tplr.neurons.outer_step(
                self.model,
                self.outer_optimizer,
                gather_result=gather_result,
                transformer=self.transformer,
                compressor=self.compressor,
                xshapes=self.xshapes,
                totalks=self.totalks,
                device=str(self.device),
                is_master=self.is_master,
                world_size=self.world_size,
                use_dct=self.hparams.use_dct,
            )
            model_update_time = tplr.T() - update_start
            tplr.logger.info(f"{tplr.P(step_window, model_update_time)} Updated model")

            if self.is_master:
                # Add debug data including successfully gathered peers
                debug_dict = {}

                # Add model parameters debug info
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    model_iterator = self.model.module.named_parameters()
                else:
                    model_iterator = self.model.named_parameters()
                for name, param in model_iterator:
                    if (
                        param is not None and param.numel() >= 2
                    ):  # Check if tensor has at least 2 elements
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
                t = asyncio.create_task(
                    self.comms.put(
                        state_dict=debug_dict,
                        uid=str(self.uid),
                        window=step_window,
                        key="debug",
                        local=False,
                    )
                )
                self._bg_tasks.add(t)
                t.add_done_callback(self._bg_tasks.discard)

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
                momentum_norms: list[float] = [
                    v for sublist in gathered_mom for v in sublist
                ]
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
            self.window_step += 1
            tplr.logger.info(f"Total optimization steps: {self.global_step}")

            if self.world_size > 1:
                dist.barrier(device_ids=[self.local_rank])

            # Delete local variables to clear up memory
            del gather_result, shard_gradient
            if self.is_master:
                del processed_state_dict, gradient

            await self.cleanup_window()
            # 4. Wait for next window
            tplr.logger.info("Wait for next window...")
            await self.wait_until_window(step_window + 1)

    async def inner_steps(
        self,
        loader: DataLoader,
        step_window: int,
    ) -> dict:
        """
        One inner-loop optimisation pass that is gradient-accumulation aware and
        synchronised across distributed ranks.

        Returns
        -------
        dict
            Keys: total_loss (float), batch_count (int), batch_tokens (int)
        """
        self.inner_optimizer.zero_grad()

        total_loss: float = 0.0
        batch_count: int = 0
        batch_tokens: int = 0  # local counter
        accum_batch_size: int = 0
        window_entry_loss: float = 0.0
        global_tokens: int = 0  # after cross-rank reduction
        global_loss_sum: float = 0.0
        local_tokens_sum: int = 0  # local running totals
        local_loss_sum: float = 0.0

        params_offloaded = self._get_offloaded_param()

        inner_step_count: int = 0
        loader_iter = iter(loader)

        while not self.stop_event.is_set():
            # ------------------------------------------------------------------ #
            # 1. Fetch a batch (or detect EOS) on *each* rank
            # ------------------------------------------------------------------ #
            try:
                batch = await self.loop.run_in_executor(None, next, loader_iter)
                local_has_batch = True
            except StopIteration:
                local_has_batch = False
                batch = None

            # Decide collectively whether we should continue
            if self.world_size > 1:
                cont = self.should_continue(local_has_batch, self.device)
                if not cont:
                    if self.is_master:
                        tplr.logger.info(
                            "Stopping batch loop: at least one rank exhausted."
                        )
                    break
                if not local_has_batch:  # exhausted only on this rank
                    continue

            # ------------------------------------------------------------------ #
            # 2. Prepare inputs
            # ------------------------------------------------------------------ #
            if isinstance(batch, torch.Tensor):
                input_ids = batch.to(self.device, dtype=torch.long, non_blocking=True)
            else:
                input_ids = torch.tensor(batch, dtype=torch.long, device=self.device)

            local_bs = len(batch)  # type: ignore
            accum_batch_size += local_bs

            tokens_this_batch = input_ids.numel()
            batch_tokens += tokens_this_batch
            local_tokens_sum += tokens_this_batch

            labels = input_ids.clone()
            labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)

            # ------------------------------------------------------------------ #
            # 3. Forward + backward
            # ------------------------------------------------------------------ #
            with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                outputs = self.model(input_ids=input_ids, labels=labels)

            loss = outputs.loss / self.sampler.grad_accum_steps
            loss_item = outputs.loss.detach().item()

            # -------------------------------------------------------------- #
            # 3-a.  Back-prop with no_sync() on non-final micro-batches
            # -------------------------------------------------------------- #
            final_micro_batch = (batch_count + 1) % self.sampler.grad_accum_steps == 0
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) and (
                self.world_size > 1 and not final_micro_batch
            ):
                sync_ctx = self.model.no_sync()
            else:
                sync_ctx = nullcontext()
            with sync_ctx:
                loss.backward()
            total_loss += loss_item
            local_loss_sum += loss_item  # defer collective

            batch_count += 1
            window_changed = self.current_window != step_window

            # ------------------------------------------------------------------ #
            # 4. Decide *together* whether to take an optimiser step
            # ------------------------------------------------------------------ #
            step_now = final_micro_batch or window_changed
            if self.world_size > 1:
                flag = torch.tensor([int(step_now)], device=self.device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)  # 1-byte sync
                step_now = bool(flag.item())  # identical on all ranks

            if step_now:
                # ── one collective for scalar stats per inner step ───────────
                global_tokens_step = int(self._ddp_reduce(local_tokens_sum))
                global_loss_step = self._ddp_reduce(local_loss_sum)
                global_tokens += global_tokens_step
                global_loss_sum += global_loss_step

                # mean loss of this accumulation step for logging
                log_loss = self._ddp_reduce(loss_item, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.inner_optimizer.step()
                self.inner_scheduler.step()
                self.inner_optimizer.zero_grad(set_to_none=True)

                inner_step_count += 1

                # reset local accumulators BEFORE the next micro-batch
                local_tokens_sum = 0
                local_loss_sum = 0

                accum_batch_size = int(self._ddp_reduce(accum_batch_size))
                if self.is_master:
                    tplr.logger.info(
                        f"Inner Step {inner_step_count}, "
                        f"Batch {batch_count}, loss: {log_loss:.4f}, "
                        f"accum: {accum_batch_size}/{self.hparams.batch_size}"
                    )
                if window_entry_loss == 0.0:
                    total_batches_first_step = int(self._ddp_reduce(batch_count))
                    window_entry_loss = global_loss_sum / total_batches_first_step
                accum_batch_size = 0  # reset on *all* ranks

            # ------------------------------------------------------------------ #
            # 5. Outer-loop window control
            # ------------------------------------------------------------------ #
            need_sync = window_changed or inner_step_count == self.hparams.inner_steps
            local_done = torch.tensor(
                [need_sync],
                dtype=torch.uint8,
                device=self.device,
            )

            if self.world_size > 1:
                dist.all_reduce(local_done, op=dist.ReduceOp.MAX)
            global_done = bool(local_done.item())

            if global_done:
                if self.is_master:
                    tplr.logger.info("<Exhausted window: exiting synchronously>")
                for _ in range(inner_step_count, self.hparams.inner_steps):
                    self.inner_scheduler.step()
                break

        await asyncio.sleep(0)

        # ------------------------------------------------------------------ #
        # 6. parameter offloading logic
        # ------------------------------------------------------------------ #
        bare_model = (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )
        with torch.no_grad():
            for saved_param, p in zip(params_offloaded, bare_model.parameters()):
                saved_param = saved_param.to(p.device, non_blocking=True)

                # (a) pseudo-gradient for outer step
                p.grad = saved_param - p.data

                # (b) ***in-place*** restore original weights
                p.data.copy_(saved_param)

        # ---------------------------------------------------------------------- #
        # 7. Return aggregated metrics
        # ---------------------------------------------------------------------- #
        batch_count = int(self._ddp_reduce(batch_count))
        return {
            "total_loss": global_loss_sum,  # cross-rank sum
            "window_entry_loss": window_entry_loss,
            "batch_count": batch_count,  # cross-rank sum
            "batch_tokens": global_tokens,  # cross-rank sum
        }

    def _get_offloaded_param(self):
        """Get a copy of current parameters and offload them to CPU"""
        bare_model = (
            self.model
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )
        return [
            param.data.detach().clone().to("cpu") for param in bare_model.parameters()
        ]

    async def cleanup_window(self):
        """Aggressive memory cleanup between windows"""
        # Clear gradients more thoroughly
        self.model.zero_grad(set_to_none=True)
        self.inner_optimizer.zero_grad(set_to_none=True)

        # Empty CUDA cache
        torch.cuda.empty_cache()
        torch.clear_autocast_cache()

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
