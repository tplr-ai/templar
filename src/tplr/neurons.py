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


import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypeVar
import typing

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from wandb.sdk.wandb_run import Run

import tplr

if TYPE_CHECKING:
    from neurons.miner import Miner
    from neurons.validator import Validator

NeuronT = TypeVar("NeuronT", "Miner", "Validator")


def prepare_gradient_dict(miner: "Miner", step_window: int):
    """
    Prepares the gradient dictionary for sharing by compressing the
    momentum for each parameter and attaching metadata.

    Args:
        miner (Miner): Instance of Miner containing model, scheduler, momentum, compressor, transformer and hparams.
        step_window (int): The current window number.

    Returns:
        tuple: (gradient, xshapes, totalks, transmitted) where:
            gradient (dict): Contains keys for each parameter's compressed gradients and metadata.
            xshapes (dict): The computed shapes for each parameter.
            totalks (dict): Total length information for each parameter.
    """
    gradient = {}
    xshapes = {}
    totalks = {}
    lr = float(miner.hparams.outer_learning_rate)

    if isinstance(miner.model, torch.nn.parallel.DistributedDataParallel):
        model_iterator = miner.model.module.named_parameters()
    else:
        model_iterator = miner.model.named_parameters()
    for n, p in model_iterator:
        # Skip parameters not owned by this rank
        if n not in miner.owned_params:
            p.grad = None
            continue

        # Apply momentum decay.
        miner.error_feedback[n].mul_(miner.hparams.momentum_decay)

        # Ensure the gradient is on the same device as the parameter.
        assert p.grad is not None
        grad = p.grad.to(p.device)
        if miner.error_feedback[n].device != p.device:
            miner.error_feedback[n] = miner.error_feedback[n].to(p.device)

        # Normal behavior for later iterations
        miner.error_feedback[n].add_(grad, alpha=lr)

        # Compress momentum
        encoded = miner.transformer.encode(
            miner.error_feedback[n], use_dct=miner.hparams.use_dct
        )
        idxs, vals, xshape, totalk, quant_params = miner.compressor.compress(
            encoded, miner.hparams.topk_compression
        )
        if totalk is None:
            tplr.logger.info("totalk is None")
        del encoded  # Free the encoded tensor immediately

        # Estimate transmitted gradient
        decompressed = miner.compressor.decompress(
            p, idxs, vals, xshape, totalk, quant_params
        )
        transmit_grad = miner.transformer.decode(
            decompressed, use_dct=miner.hparams.use_dct
        )
        del decompressed  # Free intermediate tensor

        miner.error_feedback[n].sub_(transmit_grad)

        # Move compressed values to CPU to save GPU memory
        gradient[n + "idxs"] = idxs.cpu() if isinstance(idxs, torch.Tensor) else idxs
        gradient[n + "vals"] = vals.cpu() if isinstance(vals, torch.Tensor) else vals
        gradient[n + "quant_params"] = quant_params
        xshapes[n] = xshape
        totalks[n] = totalk

        del transmit_grad

        # Clear gradient to free memory
        p.grad = None

    torch.cuda.empty_cache()

    gradient["metadata"] = {"window": step_window}

    return gradient, xshapes, totalks


def outer_step(
    model: nn.Module,
    optimizer: Optimizer,
    *,
    gather_result: SimpleNamespace | None,
    transformer: tplr.compress.TransformDCT,
    compressor: tplr.compress.CompressDCT,
    xshapes: dict,
    totalks: dict,
    device: str,
    is_master: bool,
    world_size: int,
    use_dct: bool = False,
    wandb_run: Run | None = None,
    global_step: int | None = None,
) -> None:
    """
    Synchronize gradients (if DDP) and apply optimizer step
    """
    bare_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    if is_master:
        min_median_norm = float("inf")
        max_median_norm = float("-inf")

        model.train()
        optimizer.zero_grad()

        if gather_result is not None and gather_result.state_dict is not None:
            for n, p in bare_model.named_parameters():
                idxs = getattr(gather_result.state_dict, n + "idxs", None)
                vals = getattr(gather_result.state_dict, n + "vals", None)
                qps = getattr(gather_result.state_dict, n + "quant_params", None)

                if idxs is None or vals is None:
                    tplr.logger.info(f"Gradient data missing for {n}, skipping.")
                    continue

                # normalise container types
                if not isinstance(idxs, (list, tuple)):
                    idxs = [idxs]
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]

                # ------------------------------------------------------------------
                # 1️⃣  Ensure every vals tensor is fp32/fp16 (de-quant if needed)
                # ------------------------------------------------------------------
                vals_f32: list[torch.Tensor] = []
                for i, v in enumerate(vals):
                    v = v.to(device)
                    if v.dtype == torch.uint8:  # still quantised → decode
                        if qps is None:
                            tplr.logger.warning(f"Missing quant_params for {n}; skip.")
                            break
                        qp = qps[i] if isinstance(qps, (list, tuple)) else qps
                        v = compressor._dequantize_values(v, qp).to(device)
                    vals_f32.append(v)

                if len(vals_f32) != len(vals):  # some decode failed
                    continue

                block_norms = torch.stack([torch.norm(v, p=2) for v in vals_f32])

                new_grad = transformer.decode(
                    compressor.batch_decompress(
                        p.to(device),
                        typing.cast(list[torch.Tensor], idxs),
                        typing.cast(list[torch.Tensor], vals_f32),
                        xshapes[n],
                        totalks[n],
                        quantize_params=None,  # already de-quantised
                        block_norms=block_norms,
                        normalise=False,
                        clip_norm=True,
                    ),
                    use_dct=use_dct,
                )

                # 2️⃣  last-chance validation
                if (not torch.isfinite(new_grad).all()) or new_grad.abs().max() > 1e3:
                    tplr.logger.warning(f"Non-finite gradient for {n}; dropping.")
                    continue

                # track stats
                med = torch.median(block_norms).item()
                min_median_norm = min(min_median_norm, med)
                max_median_norm = max(max_median_norm, med)

                p.grad = new_grad if p.grad is None else p.grad.copy_(new_grad)

        # ------------------------------------------------------------------
        # log med-norm range
        # ------------------------------------------------------------------
        if (
            wandb_run is not None
            and global_step is not None
            and max_median_norm > float("-inf")
        ):
            wandb_run.log(
                {
                    "compress/min_median_block_norm": min_median_norm,
                    "compress/max_median_block_norm": max_median_norm,
                },
                step=global_step,
            )

        # ------------------------------------------------------------------
        # 4️⃣  global grad-norm clip then optimiser step
        # ------------------------------------------------------------------
        optimizer.step()
        torch.cuda.empty_cache()

        # broadcast updated weights to other ranks
        if world_size > 1:
            for t in bare_model.state_dict().values():
                if torch.is_tensor(t):
                    dist.broadcast(t.data, src=0)

    else:  # non-master ranks just receive the broadcast
        if world_size > 1:
            for t in bare_model.state_dict().values():
                if torch.is_tensor(t):
                    dist.broadcast(t.data, src=0)


async def update_peers(instance: NeuronT, window: int, peer_start: float) -> None:
    # Check if peers list is empty and fetch previous list if needed
    if len(instance.comms.peers) == 0:
        tplr.logger.info(
            "Current peers list is empty, attempting to fetch previous peer list"
        )
        result = await instance.comms.get_peer_list(fetch_previous=True)
        if result is not None:
            prev_peers, prev_update_window = result
            tplr.logger.info(
                f"Got previous peer list with {len(prev_peers)} peers "
                f"and update window {prev_update_window}"
            )
            instance.comms.peers = prev_peers
            # Don't set next_peers here, as we want the normal update process to continue
        else:
            tplr.logger.warning(
                "Failed to fetch previous peer list, continuing with empty peers"
            )

    # Get next peers
    if (
        instance.next_peers is None  # next peers are not fetched yet
        and instance.peers_update_window  # they should be on bucket by now
        + instance.hparams.peer_replacement_frequency
        - window
        < instance.hparams.peer_list_window_margin
    ):
        result = await instance.comms.get_peer_list()
        if result is None:
            tplr.logger.info("Unable to get peer list from bucket")
        else:
            next_peers, peers_update_window = result
            tplr.logger.info(
                f"Got peer list {next_peers} and update window "
                f"{peers_update_window} from bucket"
            )
            if (
                instance.peers_update_window is None
                or peers_update_window > instance.peers_update_window
            ):
                instance.next_peers = next_peers
                instance.peers_update_window = peers_update_window
                tplr.logger.info("This list is new, updating next_peers")

    # Update peers, if it's time
    if instance.next_peers is not None and window >= instance.peers_update_window:
        instance.comms.peers = instance.next_peers
        late_text = (
            f"{window - instance.peers_update_window} windows late"
            if window - instance.peers_update_window > 0
            else "on time"
        )
        tplr.logger.info(
            f"{tplr.P(window, tplr.T() - peer_start)} Updated peers "
            f"{late_text} - gather:{len(instance.comms.peers)}. Next update "
            f"expected on step window "
            f"{instance.peers_update_window + instance.hparams.peer_list_window_margin}"
        )
        instance.next_peers = None
    else:
        reason = (
            "next peers are not defined yet"
            if instance.next_peers is None
            else f"sync window is {window} and peers update window "
            f"is {instance.peers_update_window}"
        )
        tplr.logger.info(f"Not time to replace peers: {reason}")


async def catchup_with_aggregation_server(
    instance: NeuronT, checkpoint_current_window: int
) -> None:
    """
    Synchronise the local model with the chain.

    For every window between the checkpoint and the current chain head:

    1. **Primary path** – download the pre-computed `aggregated_gradients`
       object uploaded by the *leader* validator and apply it via
       `tplr.neurons.outer_step`.

    2. **Fallback for the final window only** – if the leader has not yet
       published an aggregator object for `target_window - 1`, perform a live
       `instance.comms.gather( ..., key="gradient", ... )` against the current
       peer-set and apply those gradients instead.

    After each application we advance the inner LR scheduler, clear CUDA
    cache, and (optionally) log a debug-dict comparison so we can estimate how
    many optimisation steps we were behind the leader.

    The loop exits when `start_w` has caught up with `instance.current_window`
    (taking into account that the chain head may advance while we are replaying).
    """
    tplr.logger.info("Starting catch‑up using aggregated_gradients...")
    assert instance.start_window is not None

    leader_uid: int = instance.metagraph.S.argmax().item()

    start_w = checkpoint_current_window + 1
    target_w = instance.current_window
    tplr.logger.info(f"Replaying windows {start_w} ... {target_w - 1}")

    prev_param_state: dict[str, torch.Tensor] = {}
    param_avg_change: dict[str, torch.Tensor] = {}
    alpha: float = 0.20
    slice_idx = slice(0, 2)

    while start_w < target_w:
        tplr.logger.info(f"  • window {start_w}")

        # ------------------------------------------------------------------
        # 1) Fetch the aggregated object dumped by the leader validator.
        # ------------------------------------------------------------------
        fetch = await instance.comms.get(
            uid=str(leader_uid),
            window=start_w,
            key="aggregator",
            timeout=60,
            local=False,
            stale_retention=10,
        )

        # ── A. aggregated object exists → normal path ────────────────────
        if fetch is not None and fetch[0] is not None and "state_dict" in fetch[0]:
            payload, _ = fetch

            # ------------------------------------------------------------------
            # Re‑create the SimpleNamespace expected by `outer_step`.
            # ------------------------------------------------------------------
            gather_ns = SimpleNamespace(
                state_dict=SimpleNamespace(**payload["state_dict"]),
                uids=payload.get("uids", []),
                skipped_uids=payload.get("skipped_uids", []),
                success_rate=payload.get("success_rate", 0.0),
            )

        # ── B. aggregated object *missing* or *malformed* ────────────────
        else:
            is_last_window = start_w == target_w - 1
            tplr.logger.warning(
                "    ↳ %s – %s",
                "not available" if fetch is None else "malformed payload",
                "attempting gather‑fallback" if is_last_window else "skipping",
            )

            if not is_last_window:
                start_w += 1
                continue

            sync_block = (start_w + 1) * instance.hparams.blocks_per_window
            ts_value = await instance.loop.run_in_executor(
                None, instance.query_block_timestamp, sync_block
            )
            if ts_value is None:
                tplr.logger.warning(
                    f"Could not get timestamp for sync block {sync_block}.",
                )
                time_min = time_max = None
            else:
                time_min = datetime.fromtimestamp(ts_value, tz=timezone.utc)
                time_max = time_min + timedelta(
                    seconds=instance.hparams.time_window_delta_seconds
                )

            # ---- Gather fallback ----------------------------------------
            gather_ns = await instance.comms.gather(
                my_uid=instance.uid,
                uids=instance.comms.peers,
                window=start_w,
                key="gradient",
                timeout=45,
                device=str(instance.config.device),
                local=False,
                stale_retention=10,
                totalks=instance.totalks,
                compressor=instance.compressor,
                time_min=time_min,
                time_max=time_max,
            )

            if gather_ns is None:
                tplr.logger.warning("    ↳ gather‑fallback failed – skipping")
                start_w += 1
                continue

            tplr.logger.info("    ↳ gather‑fallback succeeded – applying")

        # ------------------------------------------------------------------
        # 2) Apply those gradients through the shared helper.
        # ------------------------------------------------------------------
        tplr.neurons.outer_step(
            instance.model,
            instance.outer_optimizer,
            gather_result=gather_ns,
            transformer=instance.transformer,
            compressor=instance.compressor,
            xshapes=instance.xshapes,
            totalks=instance.totalks,
            device=instance.config.device,
            is_master=True,
            world_size=1,
            use_dct=instance.hparams.use_dct,
        )

        # advance LR scheduler if one exists.
        inner_sched: LRScheduler | None = getattr(instance, "inner_scheduler", None)
        if inner_sched is not None:
            for _ in range(instance.hparams.inner_steps):
                inner_sched.step()

        torch.cuda.empty_cache()
        # ──────────────────────────────────────────────────────────────────────
        # 3) Debug‑dict comparison to estimate “how many steps behind” we are
        # ──────────────────────────────────────────────────────────────────────
        try:
            debug_fetch = await instance.comms.get(
                uid=str(leader_uid),
                window=start_w,
                key="debug",
                local=False,
                stale_retention=10,
            )

            if debug_fetch is not None and isinstance(debug_fetch[0], dict):
                debug_dict = debug_fetch[0]  # validator’s payload

                # --- update EMA of parameter‑slice changes ------------------
                bare_model = (
                    instance.model.module
                    if isinstance(
                        instance.model, torch.nn.parallel.DistributedDataParallel
                    )
                    else instance.model
                )
                for name, p in bare_model.named_parameters():
                    if p.numel() < 2:
                        continue
                    curr_slice = p.detach().cpu().flatten()[slice_idx]
                    if name in prev_param_state:
                        delta = (curr_slice - prev_param_state[name]).abs()
                        if name not in param_avg_change:
                            param_avg_change[name] = delta.clone()
                        else:
                            param_avg_change[name].mul_(1 - alpha).add_(delta * alpha)
                    prev_param_state[name] = curr_slice.clone()

                # --- call shared comparison helper --------------------------
                lr = instance.outer_optimizer.param_groups[0]["lr"]
                cmp = await tplr.neurons.compare_model_with_debug_dict(
                    model=instance.model,
                    debug_dict=debug_dict,
                    learning_rate=lr,
                    index_range=(0, 2),
                    param_avg_change=param_avg_change,
                )

                if cmp["success"]:
                    tplr.logger.info(
                        f"[catch‑up] window {start_w} "
                        f"avg_steps_behind={cmp['avg_steps_behind']:.3f}, "
                        f"l2_norm={cmp['l2_norm']:.4f}"
                    )
                else:
                    tplr.logger.warning(
                        f"[catch‑up] debug‑dict comparison failed for window {start_w}"
                    )
            else:
                tplr.logger.warning(
                    f"[catch‑up] no debug‑dict found for window {start_w}"
                )
        except Exception as exc:
            tplr.logger.warning(f"[catch‑up] debug‑dict processing error: {exc}")

        instance.global_step = start_w - instance.start_window
        start_w += 1

        # If the chain progressed while we were busy, extend the target.
        if instance.current_window > target_w:
            target_w = instance.current_window

    instance.global_step = target_w - instance.start_window
    tplr.logger.info("Catch‑up finished – model now in sync.")


async def compare_model_with_debug_dict(
    model: nn.Module,
    debug_dict: dict[str, list[float]],
    learning_rate: float,
    param_avg_change: dict[str, torch.Tensor] | None = None,
    *,
    min_step_size: float = 1e-9,
    index_range: tuple[int, int] = (0, 2),
) -> dict[str, bool | float | int]:
    """
    Compare weights with published debug snippets and return sync metrics.
    """
    # Initialize metrics
    total_squared_diff = 0.0
    total_abs_diff = 0.0
    param_count = 0
    max_diff = 0.0  # largest raw parameter diff
    max_steps = 0.0

    steps_sum = 0.0
    tensors = 0

    named_params = (
        model.module.named_parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.named_parameters()
    )

    for name, p in named_params:
        key = name + "_debug"
        if key not in debug_dict or not isinstance(debug_dict[key], list):
            continue

        # --- grab the slice we care about --------------------------------
        curr_slice = p.data.flatten()[index_range[0] : index_range[1]]
        debug_slice = torch.tensor(debug_dict[key], dtype=p.dtype, device=p.device)

        diff_vec = curr_slice - debug_slice
        abs_vec = torch.abs(diff_vec)

        total_squared_diff += torch.sum(diff_vec**2).item()
        total_abs_diff += abs_vec.sum().item()
        raw_max = abs_vec.max().item()
        max_diff = max(max_diff, raw_max)
        param_count += abs_vec.numel()

        # --- element-wise steps-behind -----------------------------------
        if param_avg_change and name in param_avg_change:
            step_vec = torch.clamp(
                param_avg_change[name].to(p.device), min=min_step_size
            )
            if step_vec.numel() != abs_vec.numel():
                # fallback if stored slice has wrong length
                step_vec = abs_vec.new_full(abs_vec.size(), learning_rate)
        else:
            step_vec = abs_vec.new_full(abs_vec.size(), learning_rate)

        step_ratio = abs_vec / step_vec
        steps_sum += step_ratio.mean().item()
        max_steps = max(max_steps, step_ratio.max().item())
        tensors += 1

    l2_norm = math.sqrt(total_squared_diff)
    avg_l2_norm = math.inf if tensors == 0 else l2_norm / param_count
    avg_abs_diff = math.inf if tensors == 0 else total_abs_diff / param_count
    avg_steps = math.inf if tensors == 0 else steps_sum / tensors
    if tensors == 0:
        max_steps = math.inf  # nothing compared → undefined

    return {
        "success": True,
        "l2_norm": l2_norm,
        "avg_l2_norm": avg_l2_norm,
        "avg_abs_diff": avg_abs_diff,
        "max_diff": max_diff,
        "avg_steps_behind": avg_steps,
        "max_steps_behind": max_steps,
        "param_count": param_count,
        "learning_rate": learning_rate,
    }
