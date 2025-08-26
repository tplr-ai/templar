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
import time
from contextlib import nullcontext
from typing import Iterable

import torch
import torch.distributed as dist
import torch.profiler as tp
from torch import autocast
from torch.distributed.tensor import DTensor as DT
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from torchtitan.components.loss import cross_entropy_loss

import tplr
from neurons.base_node import CPU_COUNT
from tplr import model_factory
from tplr.muon import Muon, SingleDeviceMuonWithAuxAdam


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
        self.inner_scheduler = self.get_inner_scheduler()

        tplr.logger.info("[Init] optimizers & schedulers constructed")
        return

    def get_inner_scheduler(self):
        # Get optimizer configuration
        optimizer_config = getattr(self.hparams, "optimizer", {})
        optimizer_type = optimizer_config.get("type", "muon").lower()

        # Get optimizer-specific config (includes learning_rate and scheduler)
        opt_specific_config = optimizer_config.get(optimizer_type, {})
        scheduler_config = opt_specific_config.get("scheduler", {})

        # Get the effective learning rate used by the optimizer
        # Default based on optimizer type
        default_lr = 2e-4 if optimizer_type == "adamw" else 0.02
        effective_lr = opt_specific_config.get("learning_rate", default_lr)

        inner_steps_before_outer_step = self.hparams.inner_steps * (
            self.hparams.validator_offset + self.hparams.peer_list_window_margin + 1
        )

        # Get scheduler parameters with optimizer-specific overrides
        warmup_steps = scheduler_config.get("warmup_steps", 750)
        t_max = scheduler_config.get("t_max", 20000)
        eta_min_factor = scheduler_config.get("eta_min_factor", 0.1)

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
            total_iters=warmup_steps,
        )
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            self.inner_optimizer,
            T_max=t_max,
            eta_min=effective_lr * eta_min_factor,
        )
        inner_scheduler = lr_scheduler.SequentialLR(
            self.inner_optimizer,
            schedulers=[init_scheduler, warmup_scheduler, cosine_scheduler],
            milestones=[
                inner_steps_before_outer_step,
                inner_steps_before_outer_step + warmup_steps,
            ],
        )

        tplr.logger.info(
            f"[Init] Constructed {optimizer_type} scheduler with lr={effective_lr}, "
            f"warmup_steps={warmup_steps}, t_max={t_max}, eta_min_factor={eta_min_factor}"
        )

        return inner_scheduler

    def get_inner_optimizer(self, validator: bool):
        # Get optimizer config from hparams
        optimizer_config = getattr(self.hparams, "optimizer", {})
        optimizer_type = optimizer_config.get("type", "muon").lower()

        if validator:
            # inner scheduler and dummy optimizer for logging purposes
            _dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            # Any optimiser will do; SGD is the simplest and has no extra state.
            # Get the learning rate from the configured optimizer
            opt_specific_config = optimizer_config.get(optimizer_type, {})
            # Default based on optimizer type
            default_lr = 2e-4 if optimizer_type == "adamw" else 0.02
            lr = opt_specific_config.get("learning_rate", default_lr)
            inner_optimizer = torch.optim.SGD(
                [_dummy_param],
                lr=lr,
            )
        else:
            if optimizer_type == "adamw":
                adamw_config = optimizer_config.get("adamw", {})
                # Use optimizer-specific learning rate if provided
                adamw_lr = adamw_config.get("learning_rate", 2e-4)
                adamw_weight_decay = adamw_config.get("weight_decay", 0.1)
                inner_optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=adamw_lr,
                    weight_decay=adamw_weight_decay,
                    betas=tuple(adamw_config.get("betas", [0.9, 0.95])),
                    eps=adamw_config.get("eps", 1e-8),
                )
                tplr.logger.info(
                    f"[Init] Using AdamW inner optimizer with lr={adamw_lr}, "
                    f"weight_decay={adamw_weight_decay}, "
                    f"betas={adamw_config.get('betas', [0.9, 0.95])}"
                )
            elif optimizer_type == "muon":
                # Get bare model for parameter grouping
                bare_model = getattr(self.model, "module", self.model)

                # Get Muon-specific config
                muon_config = optimizer_config.get("muon", {})
                # Use optimizer-specific learning rate if provided
                muon_lr = muon_config.get("learning_rate", 0.02)

                # Check if we're using FSDP (DTensor parameters)
                is_fsdp = False
                for param in self.model.parameters():
                    if isinstance(param, DT):
                        is_fsdp = True
                        break

                # Separate parameters for Muon (2D matrices) and Adam (embeddings, scalars, head)
                hidden_2d_params = []
                embed_params = []
                scalar_params = []
                head_params = []

                for name, param in bare_model.named_parameters():
                    if not param.requires_grad:
                        continue

                    if (
                        param.ndim >= 2
                        and "embed" not in name
                        and "lm_head" not in name
                    ):
                        hidden_2d_params.append(param)
                    elif "embed" in name:
                        embed_params.append(param)
                    elif "lm_head" in name:
                        head_params.append(param)
                    else:
                        scalar_params.append(param)

                # Create parameter groups
                adam_groups = []
                if head_params:
                    adam_groups.append(
                        dict(
                            params=head_params,
                            lr=muon_lr * muon_config.get("head_lr_scale", 0.5),
                            weight_decay=self.hparams.weight_decay,
                        )
                    )
                if embed_params:
                    adam_groups.append(
                        dict(
                            params=embed_params,
                            lr=muon_lr * muon_config.get("embed_lr_scale", 0.5),
                            weight_decay=self.hparams.weight_decay,
                        )
                    )
                if scalar_params:
                    adam_groups.append(
                        dict(
                            params=scalar_params,
                            lr=muon_lr * muon_config.get("scalar_lr_scale", 0.2),
                            weight_decay=self.hparams.weight_decay,
                        )
                    )

                adam_groups = [
                    dict(**g, betas=(0.9, 0.95), eps=1e-8, use_muon=False)
                    for g in adam_groups
                ]

                if not hidden_2d_params:
                    tplr.logger.error(
                        "No hidden 2D parameters found for Muon optimizer"
                    )
                    raise ValueError("Model must have 2D weight matrices for Muon")

                # Use new Muon optimizer if FSDP is enabled, otherwise use single device version
                if is_fsdp:
                    # FSDP version with additional options
                    muon_group = dict(
                        params=hidden_2d_params,
                        lr=muon_lr,
                        momentum=muon_config.get("momentum", 0.95),
                        weight_decay=muon_config.get("weight_decay", 0.01),
                        use_muon=True,
                        rms_scale=muon_config.get(
                            "rms_scale", True
                        ),  # Add RMS scaling option
                        nesterov=muon_config.get(
                            "nesterov", True
                        ),  # Add Nesterov option
                    )
                    param_groups = adam_groups + [muon_group]
                    inner_optimizer = Muon(param_groups)
                    optimizer_name = "Muon (FSDP2)"
                else:
                    # Single device version with original options
                    muon_group = dict(
                        params=hidden_2d_params,
                        lr=muon_lr,
                        momentum=muon_config.get("momentum", 0.95),
                        weight_decay=muon_config.get("weight_decay", 0.01),
                        use_muon=True,
                    )
                    param_groups = adam_groups + [muon_group]
                    inner_optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
                    optimizer_name = "Muon (Single Device)"

                tplr.logger.info(
                    f"[Init] Using {optimizer_name} inner optimizer with lr={muon_lr}, "
                    f"momentum={muon_config.get('momentum', 0.95)}, weight_decay={muon_config.get('weight_decay', 0.01)}"
                )
                tplr.logger.info(
                    f"  - Hidden matrix params: {len(hidden_2d_params)} (Muon)"
                )
                tplr.logger.info(f"  - Embedding params: {len(embed_params)} (Adam)")
                tplr.logger.info(f"  - Scalar params: {len(scalar_params)} (Adam)")
                tplr.logger.info(f"  - Head params: {len(head_params)} (Adam)")
            else:
                raise ValueError(
                    f"Unknown optimizer type: {optimizer_type}. Supported: 'adamw', 'muon'"
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

        world_size = dist.get_world_size() if dist.is_initialized() else 1

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

        # Average loss across all ranks, sum batches for distributed training
        if world_size > 1 and dist.is_initialized():
            loss_tensor = torch.tensor([total_loss], dtype=torch.float32, device=device)
            batch_tensor = torch.tensor([n_batches], dtype=torch.int32, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item()  # Total loss across ranks
            n_batches = int(batch_tensor.item())  # Total batches across all ranks

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

        # Initialize profiler if config is available (from BaseNode)
        prof_config = getattr(self, "_prof_config", None)
        prof = None
        prof_ctx = nullcontext()

        if prof_config is not None:
            # Create the profiler context for this inner_steps call
            prof = tp.profile(**prof_config)
            prof_ctx = prof
            self._prof = prof  # Store for access in other methods if needed

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

        offload_start = time.time()
        params_offloaded, param_specs = self._get_offloaded_param()
        offload_time = time.time() - offload_start
        tplr.logger.info(f"Parameter offload to CPU took {offload_time:.4f}s")

        inner_step_count: int = 0
        loader_iter = iter(loader)

        # Wrap the training loop with profiler context (does nothing if prof_ctx is nullcontext)
        with prof_ctx:
            while not self.stop_event.is_set():
                # ------------------------------------------------------------------ #
                # 1. Fetch a batch (or detect EOS) on *each* rank
                # ------------------------------------------------------------------ #
                with tp.record_function("Data Loading") if prof else nullcontext():
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
                with tp.record_function("Prepare Inputs") if prof else nullcontext():
                    if isinstance(batch, torch.Tensor):
                        input_ids = batch.to(
                            self.device, dtype=torch.long, non_blocking=True
                        )
                    else:
                        input_ids = torch.tensor(
                            batch, dtype=torch.long, device=self.device
                        )

                    local_bs = len(batch)  # type: ignore
                    accum_batch_size += local_bs

                    tokens_this_batch = input_ids.numel()
                    batch_tokens += tokens_this_batch
                    local_tokens_sum += tokens_this_batch

                    labels = input_ids.clone()
                    labels[:, :-1] = input_ids[:, 1:]  # ✓ shift by +1
                    labels[:, -1] = self.tokenizer.pad_token_id
                    labels = torch.where(
                        labels == self.tokenizer.pad_token_id, -100, labels
                    )

                # ------------------------------------------------------------------ #
                # 3. Forward + backward
                # ------------------------------------------------------------------ #
                with tp.record_function("Forward Pass") if prof else nullcontext():
                    with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        outputs = self.model(input_ids, labels)

                    calculated_loss = cross_entropy_loss(outputs, labels)

                    loss = calculated_loss / self.sampler.grad_accum_steps
                    loss_item = calculated_loss.detach().item()

                # -------------------------------------------------------------- #
                # 3-a.  Back-prop with no_sync() on non-final micro-batches
                # -------------------------------------------------------------- #
                with tp.record_function("Backward Pass") if prof else nullcontext():
                    corrected_accum_steps = max(self.sampler.grad_accum_steps, 1)
                    final_micro_batch = (batch_count + 1) % corrected_accum_steps == 0
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
                    with (
                        tp.record_function("Optimizer Step") if prof else nullcontext()
                    ):
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
                            total_batches_first_step = int(
                                self._ddp_reduce(batch_count)
                            )
                            window_entry_loss = (
                                global_loss_sum / total_batches_first_step
                            )
                        accum_batch_size = 0  # reset on *all* ranks

                    # Step the profiler after optimizer step
                    if prof is not None and hasattr(prof, "step"):
                        prof.step()

                # ------------------------------------------------------------------ #
                # 5. Outer-loop window control
                # ------------------------------------------------------------------ #
                need_sync = (
                    window_changed or inner_step_count == self.hparams.inner_steps
                )
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
            with tp.record_function("Parameter Offloading") if prof else nullcontext():
                restore_start = time.time()
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

                restore_time = time.time() - restore_start
                tplr.logger.info(f"Parameter restore to GPU took {restore_time:.4f}s")

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


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_model_state_dict(
            self.model,
            self.state_dict()["model"],
            # optim_state_dict=state_dict["optim"],
            options=StateDictOptions(
                full_state_dict=True, 
                broadcast_from_rank0=True,
                strict=True
            ),
        )