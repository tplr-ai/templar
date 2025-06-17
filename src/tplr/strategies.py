from abc import ABC, abstractmethod
import typing
from typing import NotRequired
import torch
import torch.distributed as dist
from torch import autocast, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.models.auto.tokenization_auto import AutoTokenizer
import tplr
from tplr.r2_dataset import R2DatasetLoader


class stepReturnValues(typing.TypedDict):
    total_loss: float
    first_loss: float
    batch_count: int
    batch_tokens: int
    accum_batch_size: NotRequired[int]


class InnerOuterStrategy(ABC):
    @abstractmethod
    def inner_step(
        self,
        model: nn.Module,
        loader,
        inner_optimizer: Optimizer | None,
        inner_scheduler: LRScheduler | None,
    ) -> stepReturnValues:
        """Execute inner optimization step"""
        pass

    @abstractmethod
    def outer_step(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        **kwargs,
    ) -> None:
        """Execute outer optimization step"""
        pass


class SimpleAccum(InnerOuterStrategy):
    def __init__(
        self,
        device: torch.device,
        world_size: int,
        is_master: bool,
        tokenizer: AutoTokenizer,
        xshapes: dict,
        totalks: dict,
        should_continue: typing.Callable[[bool, torch.device], bool],
        current_window: int,
        step_window: int,
    ):
        self.device = device
        self.world_size = world_size
        self.is_master = is_master
        self.tokenizer = tokenizer
        self.xshapes = xshapes
        self.totalks = totalks
        self.should_continue = should_continue
        self.current_window = current_window
        self.step_window = step_window

    def inner_step(
        self,
        model: nn.Module,
        loader: R2DatasetLoader,
        inner_optimizer: Optimizer | None = None,
        inner_scheduler: LRScheduler | None = None,
    ) -> stepReturnValues:
        """
        Process batches from the loader until the accumulation batch size target is reached.
        Returns metrics dictionary with loss and token counts.
        """
        total_loss = 0.0
        batch_count = 0
        batch_tokens = 0
        accum_batch_size = 0

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
            batch_tokens += tokens_this_batch
            labels = input_ids.clone()
            labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)

            # Update accumulated batch size
            assert batch is not None
            current_batch_size = len(batch)
            accum_batch_size += current_batch_size

            with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)

            total_loss += outputs.loss.item()
            outputs.loss.backward()
            batch_count += 1
            tplr.logger.info(f"loss: {outputs.loss.item()} [Batch {batch_count}]")

            # Clear intermediate activations immediately
            del outputs, batch
            if "input_ids" in locals():
                del input_ids
            if "labels" in locals():
                del labels
            torch.cuda.empty_cache()

            if self.current_window != self.step_window:
                tplr.logger.info("<Exhausted window>")
                break

        # Return metrics
        return {
            "total_loss": total_loss,
            "batch_count": batch_count,
            "batch_tokens": typing.cast(int, batch_tokens),
            "accum_batch_size": accum_batch_size,
        }

    def outer_step(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        *,
        gather_result,
        transformer,
        compressor,
        **kwargs,
    ) -> None:
        """
        Synchronize gradients (if DDP) and apply optimizer step
        """
        bare_model = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )
        new_grad = None
        assert scheduler is not None
        if self.is_master:
            model.train()
            optimizer.zero_grad()

            new_grad = None
            if gather_result is not None and gather_result.state_dict is not None:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model_iterator = model.module.named_parameters()
                else:
                    model_iterator = model.named_parameters()
                for n, p in model_iterator:
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
                        new_grad = transformer.decode(
                            compressor.batch_decompress(
                                p.to(self.device),
                                typing.cast(list[torch.Tensor], idxs),
                                typing.cast(list[torch.Tensor], vals),
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

            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
            for t in bare_model.state_dict().values():
                if torch.is_tensor(t):
                    dist.broadcast(t.data, src=0)
        else:
            for t in bare_model.state_dict().values():
                if torch.is_tensor(t):
                    dist.broadcast(t.data, src=0)


class Diloco(InnerOuterStrategy):
    def __init__(
        self,
        device: torch.device,
        world_size: int,
        is_master: bool,
        tokenizer: AutoTokenizer,
        xshapes: dict,
        totalks: dict,
        should_continue: typing.Callable[[bool, torch.device], bool],
        get_current_window: typing.Callable[[], int],
        step_window: int,
    ):
        self.device = device
        self.world_size = world_size
        self.is_master = is_master
        self.tokenizer = tokenizer
        self.xshapes = xshapes
        self.totalks = totalks
        self.should_continue = should_continue
        self.get_current_window = get_current_window
        self.step_window = step_window

        # Store offloaded parameters
        self.params_offloaded = None

    def inner_step(
        self,
        model: torch.nn.Module,
        loader,
        inner_optimizer,
        inner_scheduler,
    ) -> "stepReturnValues":
        """
        One inner-loop optimisation pass that is gradient-accumulation aware and
        synchronised across distributed ranks.

        Returns
        -------
        dict
            Keys: total_loss (float), batch_count (int), batch_tokens (int)
        """

        SOME_TODO_CONFIG = 32  # ← global samples per inner-step
        assert inner_optimizer is not None
        assert inner_scheduler is not None

        inner_optimizer.zero_grad()
        model.zero_grad()

        total_loss: float = 0.0
        batch_count: int = 0
        batch_tokens: int = 0
        accum_batch_size: int = 0  # ← global counter
        first_loss = 0.0

        self.params_offloaded = self._get_offloaded_param(model)

        inner_step_count: int = 0
        loader_iter = iter(loader)

        while True:
            # ------------------------------------------------------------------ #
            # 1. Fetch a batch (or detect EOS) on *each* rank
            # ------------------------------------------------------------------ #
            try:
                batch = next(loader_iter)
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
            input_ids = torch.tensor(batch, dtype=torch.long, device=self.device)
            tokens_this_batch = input_ids.numel()
            batch_tokens += int(tokens_this_batch)

            labels = input_ids.clone()
            labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)

            # ------------------------------------------------------------------ #
            # 3. Global (cross-rank) accumulation of batch size
            # ------------------------------------------------------------------ #
            current_batch_size_local = len(batch)
            total_batch_tensor = torch.tensor(
                [current_batch_size_local],
                device=self.device,
                dtype=torch.long,
            )
            if self.world_size > 1:
                dist.all_reduce(total_batch_tensor, op=dist.ReduceOp.SUM)

            global_batch_size_this_iter = int(total_batch_tensor.item())
            accum_batch_size += global_batch_size_this_iter

            # ------------------------------------------------------------------ #
            # 4. Forward + backward
            # ------------------------------------------------------------------ #
            ddp_should_sync = (
                accum_batch_size >= SOME_TODO_CONFIG
            ) or self.world_size == 1
            ddp_context_mgr = (
                model.no_sync
                if (self.world_size > 1 and not ddp_should_sync)
                else torch.enable_grad
            )
            with ddp_context_mgr():
                with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, labels=labels)

                loss = outputs.loss
                if first_loss == 0.0:
                    first_loss = loss
                total_loss += float(loss.item())
                loss.backward()
                batch_count += 1
                tplr.logger.info(f"loss: {loss.item():.4f} [Batch {batch_count}]")

            # ------------------------------------------------------------------ #
            # 5. Step only when *global* accumulation threshold is hit
            # ------------------------------------------------------------------ #
            if accum_batch_size >= SOME_TODO_CONFIG:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                inner_optimizer.step()
                inner_scheduler.step()
                inner_optimizer.zero_grad(set_to_none=True)

                inner_step_count += 1
                tplr.logger.info(
                    f"Inner Step {inner_step_count}, "
                    f"Batch {batch_count}, loss: {loss.item():.4f}, "
                    f"accum: {accum_batch_size}/{SOME_TODO_CONFIG}"
                )
                accum_batch_size = 0  # reset on *all* ranks

            # ------------------------------------------------------------------ #
            # 6. Memory hygiene
            # ------------------------------------------------------------------ #
            del outputs, batch, input_ids, labels
            torch.cuda.empty_cache()

            # ------------------------------------------------------------------ #
            # 7. Outer-loop window control (unchanged from your code)
            # ------------------------------------------------------------------ #
            if self.get_current_window() != self.step_window or inner_step_count == 15:
                tplr.logger.info("<Exhausted window>")
                break

        # ------------------------------------------------------------------ #
        # 8. Your “parameter offloading” logic (unchanged)
        # ------------------------------------------------------------------ #
        bare_model = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )
        for param_offloaded, param in zip(
            self.params_offloaded, bare_model.parameters()
        ):
            param_offloaded_on_device = param_offloaded.to(param.device)
            param.grad = param_offloaded_on_device - param.data
            param.data = param_offloaded_on_device  # reset for next inner step

        # ---------------------------------------------------------------------- #
        # 9. Return aggregated metrics
        # ---------------------------------------------------------------------- #
        return {
            "total_loss": total_loss,
            "first_loss": first_loss,
            "batch_count": batch_count,
            "batch_tokens": batch_tokens,
        }

    def outer_step(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        *,
        gather_result,
        transformer,
        compressor,
        **kwargs,
    ) -> None:
        """
        Synchronize gradients (if DDP) and apply optimizer step
        """
        bare_model = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )
        new_grad = None
        if self.is_master:
            model.train()
            optimizer.zero_grad()

            new_grad = None
            if gather_result is not None and gather_result.state_dict is not None:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model_iterator = model.module.named_parameters()
                else:
                    model_iterator = model.named_parameters()
                for n, p in model_iterator:
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
                        new_grad = transformer.decode(
                            compressor.batch_decompress(
                                p.to(self.device),
                                typing.cast(list[torch.Tensor], idxs),
                                typing.cast(list[torch.Tensor], vals),
                                self.xshapes[n],
                                self.totalks[n],
                                quant_params,
                                normalise=False,
                            )
                        )

                        if p.grad is None:
                            p.grad = new_grad
                        else:
                            p.grad.copy_(new_grad)
                    else:
                        tplr.logger.info(
                            f"Gradient data missing for parameter {n}, skipping."
                        )

            optimizer.step()
            torch.cuda.empty_cache()
            for t in bare_model.state_dict().values():
                if torch.is_tensor(t):
                    dist.broadcast(t.data, src=0)
        else:
            for t in bare_model.state_dict().values():
                if torch.is_tensor(t):
                    dist.broadcast(t.data, src=0)

    def _get_offloaded_param(self, model):
        """Get a copy of current parameters and offload them to CPU"""
        actual_model = model.module if hasattr(model, "module") else model
        return [
            param.data.detach().clone().to("cpu") for param in actual_model.parameters()
        ]
