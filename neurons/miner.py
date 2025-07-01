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
import os
import random
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import cast

import bittensor as bt
import numpy as np
import torch
import torch.distributed as dist
import uvloop

# Third party
from bittensor.core.subtensor import ScaleObj
from torch import autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from transformers.models.llama import LlamaForCausalLM

# Local
import tplr

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


class Miner:
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

        if self.world_size > 1:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=45),
            )
            torch.cuda.set_device(self.local_rank)
            self.config.device = f"cuda:{self.local_rank}"
        else:
            self.config.device = self.config.device or "cuda"
        self.device = torch.device(self.config.device)

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
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            tplr.logger.error(
                f"\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}[/bold]"
            )
            sys.exit()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Init model with hparams config
        self.model = LlamaForCausalLM(self.hparams.model_config)
        self.model.to(self.device)  # type: ignore[reportArgumentType]
        self.model.gradient_checkpointing_enable()
        if self.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
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

        # Init optimizer and momentum
        self.optimizer = SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        self.momentum = {}
        self.owned_params = set()

        self.xshapes = {}
        self.totalks = {}
        model_iterator = (
            self.model.module.named_parameters()
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model.named_parameters()
        )

        for idx, (n, p) in enumerate(model_iterator):
            if idx % self.world_size == self.rank:
                # this rank “owns” the parameter
                self.owned_params.add(n)
                self.momentum[n] = torch.zeros_like(p, device=self.device)
            _, _, xshape, totalk, _ = self.compressor.compress(
                self.transformer.encode(torch.zeros_like(p)),
                self.hparams.topk_compression,
            )
            self.xshapes[n] = xshape
            self.totalks[n] = totalk
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
        if self.is_master:
            self.comms.try_commit(self.wallet, self.bucket)
        if self.world_size > 1:
            dist.barrier()

        # Init state params
        self.stop_event = asyncio.Event()
        self.current_block = self.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        self.start_window = self.current_window  # Record the start window
        self.global_step = 0  # Initialize global_step to zero
        self.comms.current_window = self.current_window
        self.step_counter = 0

        # Add step tracking
        self.window_step = 0

        # Track additional metrics
        self.total_tokens_processed = 0
        self.batch_times = []  # For tracking processing speed

        if self.is_master:
            # Initialize WandB
            self.wandb = tplr.initialize_wandb(
                run_prefix="M",
                uid=self.uid,
                config=self.config,
                group="miner",
                job_type="mining",
            )

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

    # Main training loop.
    async def run(self):
        # Start background block listener
        self.loop = asyncio.get_running_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
        self.loop.set_default_executor(self.executor)

        self.listener = threading.Thread(
            target=self.block_listener,
            args=(self.loop,),
            daemon=True,
        )
        self.listener.start()  #

        # Use config peers if provided
        if self.config.peers:
            self.comms.peers = self.config.peers

        self.comms.commitments = await self.comms.get_commitments()
        tplr.logger.info("Loaded commitments")

        # Fetch start_window from highest stake validator
        self.start_window = await self.comms.get_start_window()
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
                self.optimizer,
                self.scheduler,
            ) = await self.comms.load_checkpoint(
                model=bare_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
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

            # 2) optimizer state  (broadcast as one object ➜ load on every rank)
            opt_pkt = [self.optimizer.state_dict()]
            dist.broadcast_object_list(opt_pkt, src=0)
            self.optimizer.load_state_dict(opt_pkt[0])

            # 3) scheduler state (same idea)
            sched_pkt = [self.scheduler.state_dict()]
            dist.broadcast_object_list(sched_pkt, src=0)
            self.scheduler.load_state_dict(sched_pkt[0])

            bcast_time = tplr.T() - bcast_start
            tplr.logger.info(
                f"{tplr.P(self.current_window, bcast_time)} "
                f"Broadcast checkpoint to {self.world_size - 1} ranks"
            )
            dist.barrier()

        self.comms.start_commitment_fetcher()

        while True:
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

            # 2. Load ONLY the pages that belong to *this* rank -------------------
            data_start = tplr.T()
            total_pages = self.hparams.pages_per_window
            start_idx, n_my_pages = self.pages_for_rank(
                total_pages, self.rank, self.world_size
            )

            pages = await tplr.r2_dataset.R2DatasetLoader.next_pages(
                offset=step_window * total_pages,
                n_pages=total_pages,
                seed=self.uid,
            )
            own_pages = pages[start_idx : start_idx + n_my_pages]
            tplr.logger.info(
                f"[Rank {self.rank}/{self.world_size}] pages "
                f"{list(range(start_idx, start_idx + n_my_pages))}"
            )
            loader = await tplr.r2_dataset.R2DatasetLoader.create(
                batch_size=self.hparams.batch_size,
                sequence_length=self.hparams.sequence_length,
                pages_info=own_pages,
                tokenizer=self.tokenizer,
            )
            tplr.logger.info(
                f"{tplr.P(step_window, tplr.T() - data_start)} Loaded training data"
            )
            tplr.logger.info(
                f"Pages: {[p[1] for p in pages]} for  Window: {step_window}"
            )  # type: ignore

            # 3. Accumulate gradients over batches
            train_start = tplr.T()
            tplr.logger.info("Start accumulating...")
            self.optimizer.zero_grad()
            self.model.zero_grad()
            total_loss = 0.0
            n_batches = 0
            window_tokens = 0  # Initialize token count for this window

            loader_iter = iter(loader)
            while True:
                try:
                    batch = next(loader_iter)
                    local_has_batch = True
                except StopIteration:
                    local_has_batch = False
                    batch = None

                if self.world_size > 1:
                    cont = self.should_continue(local_has_batch, self.device)
                    if not cont:
                        if self.is_master:
                            tplr.logger.info(
                                "Stopping batch loop: at least one rank exhausted."
                            )
                        break
                    if not local_has_batch:
                        continue

                input_ids = torch.tensor(batch, dtype=torch.long).to(self.device)
                tokens_this_batch = input_ids.numel()
                window_tokens += tokens_this_batch
                labels = input_ids.clone()
                labels = torch.where(
                    labels == self.tokenizer.pad_token_id, -100, labels
                )

                with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(input_ids=input_ids, labels=labels)

                total_loss += outputs.loss.item()
                outputs.loss.backward()
                n_batches += 1
                tplr.logger.info(f"loss: {outputs.loss.item()} [Batch {n_batches}]")

                # Clear intermediate activations immediately
                del outputs, batch
                if "input_ids" in locals():
                    del input_ids
                if "labels" in locals():
                    del labels
                torch.cuda.empty_cache()

                if self.current_window != step_window:
                    tplr.logger.info("<Exhausted window>")
                    break

            if n_batches > 0:
                tplr.logger.info(
                    f"Normalizing gradients by {n_batches} accumulation steps"
                )
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.div_(n_batches)

            # If training completes before the window is exhausted, wait until the window ends.
            if self.current_window == step_window:
                tplr.logger.info(
                    "Training complete; waiting for window to be exhausted..."
                )
                while self.current_window == step_window:
                    await asyncio.sleep(0.1)
            tplr.logger.info(
                f"{tplr.P(step_window, tplr.T() - train_start)} Completed training"
            )

            # Synchronise all ranks
            if self.world_size > 1:
                dist.barrier()

            # 1️⃣ every rank builds its momentum shard
            compress_start = tplr.T()
            shard_gradient, _, _ = tplr.prepare_gradient_dict(
                self, own_pages, step_window
            )
            tplr.logger.info(
                f"{tplr.P(step_window, tplr.T() - compress_start)} "
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

            # rank-0 merges & uploads the full gradient
            gradient = {}
            processed_state_dict = {}
            if self.is_master:
                merged_pages: list[tuple[int, int]] = []

                for shard in gathered:
                    if shard is not None:
                        m = shard.pop("metadata", None)
                        if m and "pages_info" in m:
                            merged_pages.extend(m["pages_info"])

                        gradient.update(shard)

                gradient["metadata"] = {
                    "pages_info": merged_pages,
                    "window": step_window,
                }
                tplr.logger.info(
                    f"Attached metadata to gradient: {gradient['metadata']}"
                )

                tplr.logger.info(
                    f"Merged {len(gathered)} shards → "
                    f"{len(gradient) - 1} tensors  |  "
                    f"pages_info len = {len(merged_pages)}"
                )

                # move to CPU before R2 upload
                processed_state_dict = {
                    k: (v.to("cpu") if isinstance(v, torch.Tensor) else v)
                    for k, v in gradient.items()
                }

                put_completion_time = await self.comms.put(
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
                tplr.logger.info(
                    f"Uploaded {upload_size / 1e6:.1f} MB shard-merged gradient"
                )

            else:
                # non-master ranks simply wait; they don't upload
                put_completion_time = 0.0

            tplr.logger.info(f"Stopped accumulating: {n_batches} batches")
            if self.world_size > 1:
                dist.barrier()

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
                    if response is None or not isinstance(response, ScaleObj):
                        raise ValueError(f"Could not query timestamp for {sync_block}")
                    ts_value = (
                        cast(int, response.value) / 1000
                    )  # convert milliseconds to seconds
                    break
                except Exception as e:
                    tplr.logger.error(
                        f"Failed to query timestamp for block {sync_block}: {str(e)}. Retry {retries + 1}/{max_retries}"
                    )
                    retries += 1
                    if retries > max_retries:
                        tplr.logger.error(
                            "Exceeded maximum retries for timestamp query."
                        )
                        raise e
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)

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
                    timeout=45,
                    device=str(self.device),
                    local=False,
                    stale_retention=100,
                    totalks=self.totalks,
                    time_min=time_min,
                    time_max=time_max,
                )
                tplr.logger.info("Gather task completed!")
                gather_time = tplr.T() - gather_start
            if self.world_size > 1:
                dist.barrier()

            # 5. Calculate and log metrics
            duration = time.time() - train_start
            self.batch_times.append(duration)
            self.total_tokens_processed += window_tokens
            tokens_per_sec = window_tokens / duration

            # ─────────────── gradient & weight norms (local) ────────────────
            grad_norms = [
                p.grad.norm().item()
                for p in self.model.parameters()
                if p.grad is not None
            ]
            weight_norms = [p.norm().item() for p in self.model.parameters()]

            # ─────────────── momentum norms (gathered across ranks) ─────────
            local_mom_norms: list[float] = [
                m.norm().item() for m in self.momentum.values()
            ]
            if self.world_size > 1:
                gathered_mom: list[list[float]] = [None] * self.world_size  # type: ignore[var-annotated]
                dist.all_gather_object(gathered_mom, local_mom_norms)
            else:
                gathered_mom = [local_mom_norms]

            momentum_norms = []
            if self.is_master:
                momentum_norms: list[float] = [
                    v for sublist in gathered_mom for v in sublist
                ]
                self.wandb.log(
                    {
                        # Training metrics
                        "miner/loss": total_loss / n_batches if n_batches > 0 else 0,
                        "miner/tokens_per_sec": tokens_per_sec,
                        "miner/batch_duration": duration,
                        "miner/total_tokens": self.total_tokens_processed,
                        "miner/batch_tokens": window_tokens,
                        "miner/global_step": self.global_step,
                        # Resource metrics
                        "miner/gpu_memory_allocated": torch.cuda.memory_allocated()
                        / 1024**2,  # MB
                        "miner/gpu_memory_cached": torch.cuda.memory_reserved()
                        / 1024**2,  # MB
                        # Network metrics
                        "miner/gather_peers": len(self.comms.peers),
                        "miner/effective_batch_size": len(self.comms.peers)
                        * self.hparams.batch_size,
                        # Optimization metrics
                        "miner/learning_rate": self.scheduler.get_last_lr()[0],
                        # Gradient statistics as points
                        "miner/mean_grad_norm": sum(grad_norms) / len(grad_norms)
                        if grad_norms
                        else 0,
                        "miner/max_grad_norm": max(grad_norms) if grad_norms else 0,
                        "miner/min_grad_norm": min(grad_norms) if grad_norms else 0,
                        "miner/grad_norm_std": torch.tensor(grad_norms).std().item()
                        if grad_norms
                        else 0,
                        "miner/mean_weight_norm": sum(weight_norms) / len(weight_norms),
                        "miner/mean_momentum_norm": sum(momentum_norms)
                        / len(momentum_norms)
                        if momentum_norms
                        else 0,
                    },
                    step=self.global_step,
                )

            # ---------------------------------------------------------------------
            # 6. Await both gather
            # ---------------------------------------------------------------------

            # 8. Apply gathered gradients
            update_start = tplr.T()
            new_grad = None
            if self.is_master:
                self.model.train()
                self.optimizer.zero_grad()

                new_grad = None
                if gather_result is not None and gather_result.state_dict is not None:
                    if isinstance(
                        self.model, torch.nn.parallel.DistributedDataParallel
                    ):
                        model_iterator = self.model.module.named_parameters()
                    else:
                        model_iterator = self.model.named_parameters()
                    for n, p in model_iterator:
                        idxs_key = n + "idxs"
                        vals_key = n + "vals"
                        quant_key = n + "quant_params"

                        idxs = getattr(gather_result.state_dict, idxs_key, None)
                        vals = getattr(gather_result.state_dict, vals_key, None)
                        quant_params = getattr(
                            gather_result.state_dict, quant_key, None
                        )
                        if idxs is not None and vals is not None:
                            if not isinstance(idxs, (list, tuple)):
                                idxs = [idxs]
                            if not isinstance(vals, (list, tuple)):
                                vals = [vals]
                            new_grad = self.transformer.decode(
                                self.compressor.batch_decompress(
                                    p.to(self.device),
                                    cast(list[torch.Tensor], idxs),
                                    cast(list[torch.Tensor], vals),
                                    self.xshapes[n],
                                    self.totalks[n],
                                    quant_params,
                                )
                            )

                            if p.grad is None:
                                p.grad = new_grad
                            else:
                                p.grad.copy_(new_grad)
                            p.grad.sign_()
                        else:
                            tplr.logger.info(
                                f"Gradient data missing for parameter {n}, skipping."
                            )

                self.optimizer.step()
                self.scheduler.step()
                torch.cuda.empty_cache()
                if self.world_size > 1:
                    for t in bare_model.state_dict().values():
                        if torch.is_tensor(t):
                            dist.broadcast(t.data, src=0)
            else:
                if self.world_size > 1:
                    for t in bare_model.state_dict().values():
                        if torch.is_tensor(t):
                            dist.broadcast(t.data, src=0)

            tplr.logger.info(
                f"{tplr.P(step_window, tplr.T() - update_start)} Updated model"
            )

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
                asyncio.create_task(
                    self.comms.put(
                        state_dict=debug_dict,
                        uid=str(self.uid),
                        window=step_window,
                        key="debug",
                        local=False,
                    )
                )
                tplr.logger.info(
                    f"Stored debug values for window {self.current_window}"
                )
            # Log total window time and metrics
            tplr.logger.info(
                f"{tplr.P(self.current_window, tplr.T() - window_start)} Completed window iteration"
            )

            # Log metrics to WandB
            if self.is_master:
                # Calculate common metrics values
                loss_value = total_loss / n_batches if n_batches > 0 else 0
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
                peer_update_time = tplr.T() - peer_start
                data_loading_time = tplr.T() - data_start
                training_time = tplr.T() - train_start
                compression_time = tplr.T() - compress_start
                model_update_time = tplr.T() - update_start
                gather_success_rate = (
                    gather_result.success_rate * 100 if gather_result else 0.0
                )

                self.wandb.log(
                    {
                        # Add timing metrics
                        "miner/timing/window_total": window_total_time,
                        "miner/timing/peer_update": peer_update_time,
                        "miner/timing/data_loading": data_loading_time,
                        "miner/timing/training": training_time,
                        "miner/timing/compression": compression_time,
                        "miner/timing/gather": gather_time,
                        "miner/timing/put": put_completion_time,
                        "miner/timing/model_update": model_update_time,
                        # Existing metrics
                        "miner/loss": loss_value,
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
                        "miner/learning_rate": self.scheduler.get_last_lr()[0],
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
                        "loss": loss_value,
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
                        "put_time": put_completion_time,
                        "model_update_time": model_update_time,
                        "tokens_per_sec": tokens_per_sec,
                    },
                )
                tplr.logger.info("Finished metrics logging call for miner")

            self.global_step += 1
            self.window_step += 1
            tplr.logger.info(f"Total optimization steps: {self.global_step}")

            # Log profiling summary every 10 windows
            if self.current_window % 10 == 0:
                tplr.logger.info("Logging performance profiling summary...")
                tplr.r2_dataset.R2DatasetLoader.log_profiling_summary()

            if self.world_size > 1:
                dist.barrier()

            # Delete local variables to clear up memory
            del loader, pages, gather_result, shard_gradient
            if self.is_master:
                del processed_state_dict, gradient, new_grad

            await self.cleanup_window()
            # 4. Wait for next window
            tplr.logger.info("Wait for next window...")
            while self.current_window == step_window:
                await asyncio.sleep(0.1)

    def pages_for_rank(
        self, total_pages: int, rank: int, world: int
    ) -> tuple[int, int]:
        """
        Returns (start_index, n_pages) for the slice assigned to `rank`.
        First `rem` ranks get one extra page when total_pages % world != 0.
        """
        pages_per_rank, rem = divmod(total_pages, world)
        my_pages = pages_per_rank + (1 if rank < rem else 0)
        global_start = pages_per_rank * rank + min(rank, rem)
        return global_start, my_pages

    async def cleanup_window(self):
        """Aggressive memory cleanup between windows"""
        # Clear gradients more thoroughly
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

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

    # Listens for new blocks and sets self.current_block and self.current_window
    def block_listener(self, _):
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


# Start miner.
if __name__ == "__main__":
    uvloop.install()
    asyncio.run(Miner().run())
