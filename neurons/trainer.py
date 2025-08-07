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
import asyncio
import concurrent.futures
from contextlib import nullcontext
from typing import Iterable

import torch
import torch.distributed as dist
from torch import autocast
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.tensor import DTensor as DT
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from torchtitan.components.loss import cross_entropy_loss

import tplr
from neurons.base_node import CPU_COUNT
from tplr import model_factory


class Trainer:
    """This will be an ongoing project to separate
    torch operations from comms/chain/miner/validator
    behavior
    """

    def __init__(self):
        pass

    def set_dataloader(self, validator: bool = False) -> None:
        self.dataset = self.dataset_manager.active_dataset

        shared_args = dict(
            dataset=self.dataset,
            uid=self.uid,
            window=self.current_window,
            steps_per_window=self.hparams.inner_steps,
            micro_bs=self.hparams.micro_batch_size,
            rank=self.rank,
            world_size=self.world_size,
        )

        if validator:
            SamplerClass = tplr.EvalSampler
            kwargs = shared_args | dict(
                batch_size=self.hparams.target_batch_size,
                validation_bs=self.hparams.validator_sample_micro_bs
                * self.hparams.micro_batch_size,
            )
        else:
            SamplerClass = tplr.MinerSampler
            kwargs = shared_args | dict(
                micro_bs=self.hparams.micro_batch_size,
                batch_size=self.hparams.batch_size,
                target_batch_size=self.hparams.target_batch_size,
            )

        self.sampler = SamplerClass(**kwargs)

        self.loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            sampler=self.sampler,
            batch_size=self.hparams.micro_batch_size,
            num_workers=10,
            pin_memory=True,
            prefetch_factor=2,
        )
        tplr.logger.info("[Run] dataset + sampler ready")
        return

    def get_expected_params(self) -> set[str]:
        """
        Creates a set of expected names for validation

            Returns: The names of all expected keys from a miner
        """
        expected_compressed_params = set()
        for param_name, _ in self.model.named_parameters():
            expected_compressed_params.add(param_name + "idxs")
            expected_compressed_params.add(param_name + "vals")
            expected_compressed_params.add(param_name + "quant_params")

        return expected_compressed_params

    def init_model(self, validator=False):
        role = "miner"
        if validator:
            role = "validator"

        # Init model with hparams config
        # Initialize TorchTitan model using model factory
        self.model = model_factory.initialize_torchtitan_model(
            hparams=self.hparams,
            role=role,
            device=str(self.device),
            world_size=self.world_size,
        )
        self.expected_compressed_params = self.get_expected_params()
        self.tokenizer = self.hparams.tokenizer

        return

    def init_optimizers_schedulers(self, validator=False):
        self.lr = float(self.hparams.outer_learning_rate)
        self.outer_optimizer = SGD(self.model.parameters(), lr=self.lr)

        self.inner_optimizer = self.get_inner_optimizer(validator)

        inner_steps_before_outer_step = self.hparams.inner_steps * (
            self.hparams.validator_offset + self.hparams.peer_list_window_margin + 1
        )

        init_scheduler = lr_scheduler.LinearLR(
            self.inner_optimizer,
            start_factor=0.1,
            end_factor=0.1,
            total_iters=inner_steps_before_outer_step,
        )
        warmup_scheduler = lr_scheduler.LinearLR(
            self.inner_optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            self.inner_optimizer,
            T_max=self.hparams.t_max,
            eta_min=self.hparams.inner_learning_rate * 0.1,
        )
        self.inner_scheduler = lr_scheduler.SequentialLR(
            self.inner_optimizer,
            schedulers=[init_scheduler, warmup_scheduler, cosine_scheduler],
            milestones=[inner_steps_before_outer_step, self.hparams.warmup_steps],
        )
        tplr.logger.info("[Init] optimizers & schedulers constructed")
        return

    def get_inner_optimizer(self, validator: bool):
        if validator:
            # inner scheduler and dummy optimizer for logging purposes
            _dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            # Any optimiser will do; SGD is the simplest and has no extra state.
            inner_optimizer = torch.optim.SGD(
                [_dummy_param],
                lr=self.hparams.inner_learning_rate,
            )
        else:
            inner_optimizer = ZeroRedundancyOptimizer(
                self.model.parameters(),
                optimizer_class=torch.optim.AdamW,
                lr=self.hparams.inner_learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.95),
                parameters_as_bucket_view=True,
                overlap_with_ddp=False,
            )
        return inner_optimizer

    async def evaluate_model(
        self,
        model: torch.nn.Module,
        loader: Iterable[torch.Tensor],
    ) -> tuple[float, int]:
        device: torch.device = next(model.parameters()).device
        total_loss = 0.0
        n_batches = 0

        with torch.inference_mode():
            model.eval()
            for i, batch in enumerate(loader):
                if batch is None or len(batch) == 0:
                    tplr.log_with_context(
                        level="warning",
                        message=f"Empty batch at index {i}, skipping",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                    continue

                if isinstance(batch, torch.Tensor):
                    input_ids = batch.to(device, dtype=torch.long, non_blocking=True)
                else:
                    input_ids = torch.tensor(batch, dtype=torch.long, device=device)
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]  # shift left by one
                labels[:, -1] = self.tokenizer.pad_token_id
                labels = torch.where(
                    labels == self.tokenizer.pad_token_id, -100, labels
                )
                with autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(input_ids)
                loss = cross_entropy_loss(logits, labels)
                total_loss += loss.item()
                n_batches += 1
                del input_ids, labels, logits
                torch.cuda.empty_cache()

                await asyncio.sleep(0)

        return total_loss, n_batches

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

        # Ensure the event loop and executor are initialized, which is normally
        # done in the `run()` method. This makes the method safe for isolated testing.
        if not hasattr(self, "loop"):
            self.loop = asyncio.get_running_loop()
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
            self.loop.set_default_executor(self.executor)

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

        params_offloaded, param_specs = self._get_offloaded_param()

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
            labels[:, :-1] = input_ids[:, 1:]  # ✓ shift by +1
            labels[:, -1] = self.tokenizer.pad_token_id
            labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)

            # ------------------------------------------------------------------ #
            # 3. Forward + backward
            # ------------------------------------------------------------------ #
            with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                outputs = self.model(input_ids, labels)

            calculated_loss = cross_entropy_loss(outputs, labels)

            loss = calculated_loss / self.sampler.grad_accum_steps
            loss_item = calculated_loss.detach().item()

            # -------------------------------------------------------------- #
            # 3-a.  Back-prop with no_sync() on non-final micro-batches
            # -------------------------------------------------------------- #
            final_micro_batch = (batch_count + 1) % self.sampler.grad_accum_steps == 0
            if (
                hasattr(self.model, "no_sync")
                and self.world_size > 1
                and not final_micro_batch
            ):
                sync_ctx = self.model.no_sync()
            else:
                sync_ctx = nullcontext()
            with sync_ctx:
                self.scaler.scale(loss).backward()

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

                # Unscale, clip, then step via GradScaler if using fp16
                self.scaler.unscale_(self.inner_optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.inner_optimizer)
                self.scaler.update()
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
        with torch.no_grad():
            for (saved_param, param_meta), p in zip(
                zip(params_offloaded, param_specs), self.bare_model.parameters()
            ):
                if param_meta["is_dtensor"] and isinstance(p, DT):
                    # Handle TP DTensors
                    saved_param = saved_param.to(p.device, non_blocking=True)

                    # Create a DTensor from the local shard directly
                    saved_param_dtensor = DT.from_local(
                        saved_param,
                        device_mesh=param_meta["device_mesh"],
                        placements=param_meta["placements"],
                        run_check=False,
                    )
                    p.grad = saved_param_dtensor - p
                    p.data.copy_(saved_param_dtensor.data)
                else:
                    saved_param = saved_param.to(p.device, non_blocking=True)
                    p.grad = saved_param - p.data
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

    def outer_step(self, gather_result):
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
        return

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
    def set_dataloader(self, validator: bool = False) -> None:
        # put here for now...
        self.dataset = self.dataset_manager.active_dataset

        shared_args = dict(
            dataset=self.dataset,
            uid=self.uid,
            window=self.current_window,
            steps_per_window=self.hparams.inner_steps,
            micro_bs=self.hparams.micro_batch_size,
            rank=self.rank,
            world_size=self.world_size,
        )

        if validator:
            SamplerClass = tplr.EvalSampler
            kwargs = shared_args | dict(
                batch_size=self.hparams.target_batch_size,
                validation_bs=self.hparams.validator_sample_micro_bs
                * self.hparams.micro_batch_size,
            )
        else:
            SamplerClass = tplr.MinerSampler
            kwargs = shared_args | dict(
                micro_bs=self.hparams.micro_batch_size,
                batch_size=self.hparams.batch_size,
                target_batch_size=self.hparams.target_batch_size,
            )

        self.sampler = SamplerClass(**kwargs)

        self.loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            sampler=self.sampler,
            batch_size=self.hparams.micro_batch_size,
            num_workers=10,
            pin_memory=True,
            prefetch_factor=2,
        )
        tplr.logger.info("[Run] dataset + sampler ready")
        return

    def get_expected_params(self) -> set[str]:
        """
        Creates a set of expected names for validation

            Returns: The names of all expected keys from a miner
        """
        expected_compressed_params = set()
        for param_name, _ in self.model.named_parameters():
            expected_compressed_params.add(param_name + "idxs")
            expected_compressed_params.add(param_name + "vals")
            expected_compressed_params.add(param_name + "quant_params")

        return expected_compressed_params
    

    async def load_checkpoint(self) -> None:
        """Load the latest checkpoint"""
        checkpoint_window_buffer = 5
        has_new_checkpoint = (
            self.global_step
            >= self.hparams.checkpoint_frequency + checkpoint_window_buffer
        )
        # Proceed to load checkpoint
        (
            success,
            loaded_checkpoint_window,
        ) = await self.comms.load_checkpoint(
            model=self.model,
            current_window=self.current_window,
            device=cast(str, self.config.device),
            init_version=tplr.__version__
            if has_new_checkpoint
            else self.bootstrap_version,
        )
        if success:
            tplr.logger.info(f"Loaded checkpoint with global_step={self.global_step}")
            # Only catch up if we're behind
            for _ in range(
                self.start_window,
                (loaded_checkpoint_window + 1) * self.hparams.inner_steps,
            ):
                self.inner_scheduler.step()
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
            tplr.logger.info("Starting from scratch")
            self.model.to(self.config.device)  # type: ignore
        return
