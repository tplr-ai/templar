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
import json
import time
from datetime import datetime, timedelta, timezone
from typing import cast

import torch
from bittensor.core.subtensor import ScaleObj
from torch import autocast

import tplr
from tplr.neurons import neuron_utils
from ..neurons.base_neuron import BaseNeuron


class MinerCore(BaseNeuron):
    """
    Miner implementation that inherits from BaseNeuron.
    Implements the miner-specific operational logic.
    """

    def __init__(self):
        """Initialize MinerCore."""
        super().__init__(neuron_type="miner")
        self._miner_specific_init()

    def _miner_specific_init(self) -> None:
        """Initialize miner-only attributes."""
        # Tracking metrics
        self.total_tokens_processed = 0
        self.batch_times = []
        
        # Step tracking
        self.step_counter = 0
        self.window_step = 0
    async def main_loop(self) -> None:
        """Implements the main operational loop for miners."""
        while True:
            # 1. Determine step_window and update global_step
            window_start = tplr.T()
            step_window = self.current_window
            self.global_step = self.current_window - self.start_window
            
            tplr.logger.info(
                f"\n{'-' * 40} Window: {step_window} (Global Step: {self.global_step}) {'-' * 40}"
            )

            # 2. Update peers
            peer_start = tplr.T()
            await neuron_utils.update_peers(
                instance=self, window=step_window, peer_start=peer_start
            )

            # 3. Load training data
            pages, loader = await self._load_training_data(step_window)

            # 4. Train window batches and accumulate local gradients
            train_metrics = await self._train_window_batches(step_window, loader)

            # 5. Prepare and put gradients
            await self._prepare_and_put_gradients(pages, step_window)

            # 6. Get gather time window
            time_min, time_max = await self._get_gather_time_window()

            # 7. Gather peer gradients
            gather_result, gather_time = await self._gather_peer_gradients(
                step_window, time_min, time_max
            )

            # 8. Apply gathered gradients
            await self._apply_gathered_gradients(gather_result)

            # 9. Log window metrics
            await self._log_window_metrics(
                step_window, train_metrics, gather_time, gather_result, window_start
            )

            # 10. Store debug data
            await self._store_debug_data(step_window, gather_result)

            # 11. Save miner checkpoint
            await self._save_miner_checkpoint()

            # 12. Cleanup window GPU memory
            await self._cleanup_window_gpu_memory()

            # 13. Wait for next window
            await self._wait_for_next_window(step_window)

    async def _load_training_data(self, step_window: int) -> tuple:
        """Load training data for the current window."""
        data_start = tplr.T()
        pages = await tplr.r2_dataset.R2DatasetLoader.next_pages(
            offset=step_window * self.hparams.pages_per_window,
            n_pages=self.hparams.pages_per_window,
            seed=self.uid,
        )
        loader = await tplr.r2_dataset.R2DatasetLoader.create(
            batch_size=self.hparams.batch_size,
            sequence_length=self.hparams.sequence_length,
            pages_info=pages,
            tokenizer=self.tokenizer,
        )
        
        tplr.logger.info(
            f"{tplr.P(step_window, tplr.T() - data_start)} Loaded training data"
        )
        tplr.logger.info(f"Pages: {[p[1] for p in pages]} for Window: {step_window}")
        
        return pages, loader

    async def _train_window_batches(self, step_window: int, loader) -> dict:
        """Handle batch-wise training, gradient accumulation, and waiting for window exhaustion."""
        train_start = tplr.T()
        tplr.logger.info("Start accumulating...")
        
        self.optimizer.zero_grad()
        self.model.zero_grad()
        total_loss = 0.0
        n_batches = 0
        window_tokens = 0

        for i, batch in enumerate(loader):
            input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
            tokens_this_batch = input_ids.numel()
            window_tokens += tokens_this_batch
            labels = input_ids.clone()
            labels = torch.where(
                labels == self.tokenizer.pad_token_id, -100, labels
            )

            with autocast(device_type=self.model.device.type, dtype=torch.bfloat16):
                outputs = self.model(input_ids=input_ids, labels=labels)

            total_loss += outputs.loss.item()
            outputs.loss.backward()
            n_batches += 1
            tplr.logger.info(f"loss: {outputs.loss.item()} [Batch {i + 1}]")
            
            if self.current_window != step_window:
                tplr.logger.info("<Exhausted window>")
                break

        if n_batches > 0:
            tplr.logger.info(f"Normalizing gradients by {n_batches} accumulation steps")
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(n_batches)

        # Wait for window exhaustion if training completes early
        if self.current_window == step_window:
            tplr.logger.info("Training complete; waiting for window to be exhausted...")
            while self.current_window == step_window:
                await asyncio.sleep(0.1)
                
        train_time = tplr.T() - train_start
        tplr.logger.info(f"{tplr.P(step_window, train_time)} Completed training")

        return {
            "total_loss": total_loss,
            "n_batches": n_batches,
            "window_tokens": window_tokens,
            "train_time": train_time,
        }

    async def _prepare_and_put_gradients(self, pages, step_window: int) -> None:
        """Use neuron_utils.prepare_gradient_dict, process for CPU, and call self.comms.put."""
        compress_start = tplr.T()
        gradient, xshapes, totalks = neuron_utils.prepare_gradient_dict(self, pages, step_window)
        
        tplr.logger.info(
            f"{tplr.P(step_window, tplr.T() - compress_start)} Compressed local gradients"
        )
        tplr.logger.debug(f"Putting own state dict for UID {self.uid}")

        # Move everything to CPU before upload
        processed_state_dict = {}
        for k, v in gradient.items():
            if isinstance(v, torch.Tensor):
                processed_state_dict[k] = v.to("cpu")
            else:
                processed_state_dict[k] = v

        # Launch the put operation
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
            tensor.element_size() * tensor.nelement()
            for tensor in processed_state_dict.values()
            if isinstance(tensor, torch.Tensor)
        )
        tplr.logger.info(f"Uploading {upload_size} bytes of own state for UID: {self.uid}")
        tplr.logger.info("Put task completed!")

    async def _get_gather_time_window(self) -> tuple[datetime, datetime]:
        """Query blockchain for timestamp to define time_min, time_max."""
        sync_block = self.current_window * self.hparams.blocks_per_window
        retries = 0
        delay = 1
        max_retries = 5
        max_delay = 60
        
        while True:
            try:
                response = self.subtensor.query_module("Timestamp", "Now", block=sync_block)
                if response is None or not isinstance(response, ScaleObj):
                    raise ValueError(f"Could not query timestamp for {sync_block}")
                ts_value = cast(int, response.value) / 1000  # convert ms to seconds
                break
            except Exception as e:
                tplr.logger.error(
                    f"Failed to query timestamp for block {sync_block}: {str(e)}. "
                    f"Retry {retries + 1}/{max_retries}"
                )
                retries += 1
                if retries > max_retries:
                    tplr.logger.error("Exceeded maximum retries for timestamp query.")
                    raise e
                time.sleep(delay)
                delay = min(delay * 2, max_delay)

        time_min = datetime.fromtimestamp(ts_value, tz=timezone.utc)
        time_max = time_min + timedelta(seconds=self.hparams.time_window_delta_seconds)
        
        tplr.logger.info(f"Using time window for gather: {time_min} to {time_max}")
        return time_min, time_max

    async def _gather_peer_gradients(
        self, step_window: int, time_min: datetime, time_max: datetime
    ) -> tuple:
        """Call self.comms.gather. Handle test mode peer selection."""
        tplr.logger.info("Refreshing peers before gather task...")

        if self.config.test:
            tplr.logger.info("Test mode active: Using all peers from metagraph.")
            all_uids = list(range(len(self.metagraph.S)))
            self.comms.peers = [uid for uid in all_uids if uid != self.uid]

        tplr.logger.info(f"Final peers for gather: {self.comms.peers}")

        gather_start = tplr.T()
        tplr.logger.info("Waiting on gather task...")
        
        gather_result = await self.comms.gather(
            my_uid=self.uid,
            uids=self.comms.peers,
            window=step_window,
            key="gradient",
            timeout=45,
            device="cpu",
            local=False,
            stale_retention=100,
            totalks=self.totalks,
            time_min=time_min,
            time_max=time_max,
        )
        
        gather_time = tplr.T() - gather_start
        tplr.logger.info("Gather task completed!")
        
        return gather_result, gather_time

    async def _apply_gathered_gradients(self, gather_result) -> None:
        """Decompress and apply gradients using SignSGD."""
        update_start = tplr.T()
        self.model.train()
        self.optimizer.zero_grad()

        if gather_result is not None and gather_result.state_dict is not None:
            for n, p in self.model.named_parameters():
                idxs_key = n + "idxs"
                vals_key = n + "vals"
                quant_key = n + "quant_params"

                idxs = getattr(gather_result.state_dict, idxs_key, None)
                vals = getattr(gather_result.state_dict, vals_key, None)
                quant_params = getattr(gather_result.state_dict, quant_key, None)
                
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
                            quant_params,
                        )
                    )

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
            
            tplr.logger.info(
                f"{tplr.P(self.start_window, tplr.T() - update_start)} Updated model"
            )

    async def _log_window_metrics(
        self, step_window: int, train_metrics: dict, gather_time: float, 
        gather_result, window_start: float
    ) -> None:
        """Log to WandB and InfluxDB."""
        # Calculate metrics
        duration = train_metrics["train_time"]
        window_tokens = train_metrics["window_tokens"]
        n_batches = train_metrics["n_batches"]
        total_loss = train_metrics["total_loss"]
        
        self.batch_times.append(duration)
        self.total_tokens_processed += window_tokens
        tokens_per_sec = window_tokens / duration if duration > 0 else 0

        # Calculate gradient and weight norms
        grad_norms = [
            p.grad.norm().item() for p in self.model.parameters() if p.grad is not None
        ]
        weight_norms = [p.norm().item() for p in self.model.parameters()]
        momentum_norms = [m.norm().item() for m in self.momentum.values()]

        # Common metrics
        loss_value = total_loss / n_batches if n_batches > 0 else 0
        mean_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        grad_norm_std = torch.tensor(grad_norms).std().item() if grad_norms else 0
        mean_weight_norm = sum(weight_norms) / len(weight_norms) if weight_norms else 0
        mean_momentum_norm = sum(momentum_norms) / len(momentum_norms) if momentum_norms else 0
        gather_success_rate = gather_result.success_rate * 100 if gather_result else 0.0

        # Log to WandB
        self.wandb.log(
            {
                "miner/loss": loss_value,
                "miner/tokens_per_sec": tokens_per_sec,
                "miner/total_tokens": self.total_tokens_processed,
                "miner/batch_tokens": window_tokens,
                "miner/global_step": self.global_step,
                "miner/gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2,
                "miner/gpu_memory_cached": torch.cuda.memory_reserved() / 1024**2,
                "miner/gather_peers": len(self.comms.peers),
                "miner/effective_batch_size": len(self.comms.peers) * self.hparams.batch_size,
                "miner/learning_rate": self.scheduler.get_last_lr()[0],
                "miner/mean_grad_norm": mean_grad_norm,
                "miner/max_grad_norm": max(grad_norms) if grad_norms else 0,
                "miner/min_grad_norm": min(grad_norms) if grad_norms else 0,
                "miner/grad_norm_std": grad_norm_std,
                "miner/mean_weight_norm": mean_weight_norm,
                "miner/mean_momentum_norm": mean_momentum_norm,
                "miner/gather/success_rate": gather_success_rate,
                "miner/timing/window_total": tplr.T() - window_start,
                "miner/timing/gather": gather_time,
            },
            step=self.global_step,
        )

        # Log to InfluxDB
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
                "window_total_time": tplr.T() - window_start,
                "gather_time": gather_time,
                "tokens_per_sec": tokens_per_sec,
            },
        )
        
        tplr.logger.info("Finished metrics logging call for miner")

    async def _store_debug_data(self, step_window: int, gather_result) -> None:
        """Put debug info (model params, peer success) to R2."""
        debug_dict = {}

        # Add model parameters debug info
        for name, param in self.model.named_parameters():
            if param is not None and param.numel() >= 2:
                debug_dict[name + "_debug"] = (
                    param.flatten()[10:12].detach().cpu().tolist()
                )

        # Add successful peers information
        if gather_result is not None:
            debug_dict["successful_peers"] = sorted(
                list(set(self.comms.peers) - set(gather_result.skipped_uids))
            )
            debug_dict["skipped_peers"] = sorted(list(gather_result.skipped_uids))

        asyncio.create_task(
            self.comms.put(
                state_dict=debug_dict,
                uid=str(self.uid),
                window=step_window,
                key="debug",
                local=False,
            )
        )
        tplr.logger.info(f"Stored debug values for window {self.current_window}")

    async def _save_miner_checkpoint(self) -> None:
        """Call self.comms.save_checkpoint with momentum."""
        if self.global_step % self.hparams.checkpoint_frequency == 0:
            tplr.logger.info(f"Creating checkpoint at global_step {self.global_step}")
            
            asyncio.create_task(
                self.comms.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    momentum=self.momentum,
                    global_step=self.global_step,
                    current_window=self.current_window,
                    start_window=self.start_window,
                )
            )
        else:
            tplr.logger.info("Skipping checkpoint save this round")

    async def _cleanup_window_gpu_memory(self) -> None:
        """Clear grads, empty CUDA cache."""
        # Clear gradients more thoroughly
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

        # Empty CUDA cache
        torch.cuda.empty_cache()
        torch.clear_autocast_cache()

        # Log memory status
        tplr.logger.info(
            f"After cleanup - GPU allocated: {torch.cuda.memory_allocated(self.config.device) / 1024**3:.2f} GB"
        )
        tplr.logger.info(
            f"After cleanup - GPU reserved: {torch.cuda.memory_reserved(self.config.device) / 1024**3:.2f} GB"
        )

    async def _wait_for_next_window(self, step_window: int) -> None:
        """Wait for next window and update tracking."""
        # Log profiling summary every 10 windows
        if self.current_window % 10 == 0:
            tplr.logger.info("Logging performance profiling summary...")
            tplr.r2_dataset.R2DatasetLoader.log_profiling_summary()

        self.global_step += 1
        self.window_step += 1
        tplr.logger.info(f"Total optimization steps: {self.global_step}")

        # Wait for next window
        tplr.logger.info("Wait for next window...")
        while self.current_window == step_window:
            await asyncio.sleep(0.1) 