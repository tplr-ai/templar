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
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor as DT
from torch.distributed.tensor import distribute_tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from wandb.sdk.wandb_run import Run

import tplr
from tplr.compress import unpack_12bit_indices

if TYPE_CHECKING:
    from neurons.miner import Miner
    from neurons.validator import Validator

NeuronT = TypeVar("NeuronT", "Miner", "Validator")


def prepare_gradient_dict(miner: "Miner", step_window: int, null_round: bool = False):
    """
    DTensor-deadlock-safe:
    - All ranks: rendezvous on DTensor grads (GFULL) and DTensor params (PFULL).
    - Only owning ranks: momentum update, encode, compress, estimate/decode, EF update.

    Args:
        miner: Miner instance containing model, compressor, transformer, etc.
        step_window: Current window number
        null_round: If True, this is a null/warmup round and error feedback should be cleared
    """

    # ------------ helpers ------------
    def ddp_initialized():
        return dist.is_available() and dist.is_initialized()

    def is_dtensor(x):
        try:
            from torch.distributed._tensor import DTensor  # type: ignore[attr-defined]

            return isinstance(x, DTensor)
        except Exception:
            return type(x).__name__ in {"DTensor", "DistributedTensor", "DT"}

    def get_mesh_group(x):
        if not is_dtensor(x):
            return None
        mesh = getattr(x, "device_mesh", None)
        if mesh is None:
            spec = getattr(x, "_spec", None)
            mesh = getattr(spec, "mesh", None)
        if mesh is not None:
            try:
                return mesh.get_group()
            except Exception:
                pass
        return dist.group.WORLD if ddp_initialized() else None

    def barrier(group=None):
        if ddp_initialized() and group is not None:
            dist.barrier(group=group)

    # ------------ start ------------
    gradient, xshapes, totalks = {}, {}, {}
    lr = float(miner.hparams.outer_learning_rate)
    use_dct = getattr(miner.hparams, "use_dct", False)
    topk = getattr(miner.hparams, "topk_compression", 32)

    if isinstance(miner.model, torch.nn.parallel.DistributedDataParallel):
        model_iterator = miner.model.module.named_parameters()
    else:
        model_iterator = miner.model.named_parameters()

    # Batch load all error feedback tensors to GPU
    for n in miner.owned_params:
        if miner.error_feedback.get(n, None) is not None:
            if miner.error_feedback[n].is_cuda:
                continue
            # Get the device from the corresponding parameter
            param = dict(miner.model.named_parameters()).get(n)
            if param is not None:
                miner.error_feedback[n] = miner.error_feedback[n].to(
                    param.device, non_blocking=True
                )
    # Synchronize to ensure all transfers complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _, (n, p) in enumerate(model_iterator, 1):
        owned = n in miner.owned_params
        p_is_dt = is_dtensor(p)
        g = getattr(p, "grad", None)
        g_is_dt = is_dtensor(g)

        # --- 1) Grad full_tensor rendezvous (GFULL) ---
        if g_is_dt:
            grp_g = get_mesh_group(g)
            barrier(grp_g)
            assert g is not None
            grad_full = g.full_tensor().to(p.device)
            barrier(grp_g)
        else:
            if g is None and not p_is_dt:
                continue
            assert g is not None, f"p.grad is None for {n}"
            grad_full = g.to(p.device)

        # --- 2) Param full_tensor rendezvous (PFULL) for DT params ---
        full_p = None
        if p_is_dt:
            grp_p = get_mesh_group(p)
            barrier(grp_p)
            assert isinstance(p, DT)
            full_p = p.full_tensor().to(p.device)
            barrier(grp_p)

        # Non-owners: after participating in collectives, drop grad and continue.
        if not owned:
            p.grad = None
            full_p = None
            continue

        # --- 3) Momentum buffer update (owner only) ---
        # Handle DTensor error feedback by creating new regular tensor if needed
        error_feedback = miner.error_feedback[n]
        if error_feedback is None:
            error_feedback = torch.zeros_like(grad_full, device=p.device)
        elif error_feedback.device != p.device:
            # Should already be on GPU from batch load, but handle edge cases
            error_feedback = error_feedback.to(p.device)

        # Clear error feedback during null rounds to prevent accumulation of invalid gradients
        if null_round:
            error_feedback.zero_()
        else:
            error_feedback.mul_(miner.hparams.momentum_decay)
            error_feedback.add_(grad_full, alpha=lr)

        # --- 4) Encode & compress (owner only) ---
        encoded = miner.transformer.encode(error_feedback, use_dct=use_dct)

        idxs, vals, xshape, totalk, quant_params = miner.compressor.compress(
            encoded, topk
        )
        del encoded

        # --- 5) Decompress reference (owner only) ---
        if p_is_dt:
            assert full_p is not None
            decompressed = miner.compressor.decompress(
                full_p, idxs, vals, xshape, totalk, quant_params
            )
        else:
            decompressed = miner.compressor.decompress(
                p, idxs, vals, xshape, totalk, quant_params
            )

        # --- 6) Decode & error-feedback update (owner only) ---
        transmit_grad = miner.transformer.decode(decompressed, use_dct=use_dct)
        del decompressed
        error_feedback.sub_(transmit_grad)
        # Keep error feedback on GPU for now, batch offload later
        miner.error_feedback[n] = error_feedback
        del transmit_grad, error_feedback
        full_p = None

        # --- 7) Pack outputs (move compressed artifacts to CPU) ---
        gradient[n + "idxs"] = idxs.cpu() if isinstance(idxs, torch.Tensor) else idxs
        gradient[n + "vals"] = vals.cpu() if isinstance(vals, torch.Tensor) else vals
        gradient[n + "quant_params"] = quant_params
        xshapes[n] = xshape
        totalks[n] = totalk

        # Clear per-param grad
        p.grad = None

    # Batch offload all error feedback tensors to CPU with pinned memory
    for name in miner.error_feedback:
        if (
            miner.error_feedback[name] is not None
            and miner.error_feedback[name].is_cuda
        ):
            # Copy to the pre-allocated pinned buffer
            miner.error_feedback_cpu_buffers[name].copy_(
                miner.error_feedback[name], non_blocking=True
            )
            miner.error_feedback[name] = miner.error_feedback_cpu_buffers[name]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gradient["metadata"] = {"window": step_window}
    return gradient, xshapes, totalks


@torch.no_grad()
def outer_step(
    model: nn.Module,
    optimizer: Optimizer,
    *,
    gather_result: SimpleNamespace | None,
    transformer: tplr.compress.ChunkingTransformer,
    compressor: tplr.compress.TopKCompressor,
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
    Memory-minimizing variant:
      - Builds and applies ONE param's grad at a time.
      - Calls optimizer.step() per param (others have grad=None, so they're skipped).
      - Frees all temporaries and grad immediately after each step.
    """
    bare_model = getattr(model, "module", model)
    bare_model.train()

    # Free any existing grads entirely (do not allocate zeros)
    optimizer.zero_grad(set_to_none=True)

    ddp = world_size > 1 and dist.is_available() and dist.is_initialized()
    src_rank = 0
    on_src = is_master or not ddp

    # All ranks have the full aggregated payload from the distributed gather
    src_sd: dict | None = None
    if (
        gather_result is not None
        and getattr(gather_result, "state_dict", None) is not None
    ):
        src_sd = gather_result.state_dict
        if isinstance(src_sd, SimpleNamespace):
            src_sd = vars(src_sd).copy()

    # optional stats
    min_median_norm = float("inf")
    max_median_norm = float("-inf")

    for name, p in bare_model.named_parameters():
        has_update = 0
        payload = None

        if src_sd is not None:
            idxs = src_sd.get(name + "idxs")
            vals = src_sd.get(name + "vals")
            qps = src_sd.get(name + "quant_params")

            if idxs is not None and vals is not None:
                if not isinstance(idxs, (list, tuple)):
                    idxs = [idxs]
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]

                # Ensure all tensors in the list are on the correct device
                idxs = [
                    t.to(device) if isinstance(t, torch.Tensor) else t for t in idxs
                ]
                vals = [
                    t.to(device) if isinstance(t, torch.Tensor) else t for t in vals
                ]

                # Ensure qps is a list, even if empty. Cast to list[Any] to satisfy type checker.
                qps_list = qps if isinstance(qps, (list, tuple)) else [qps]
                vals_f32 = compressor.maybe_dequantize_values(
                    vals, qps_list, torch.device(device)
                )  # Cast device to torch.device
                if vals_f32:
                    payload = (idxs, vals_f32)
                    has_update = 1

        # No need for _bcast_flag, as all ranks have the full aggregated_state_dict
        if has_update == 0:
            continue

        full_grad_local_shard = None
        decompressed = None
        block_norms = None

        # Each rank processes its own shard of the gradient
        idxs, vals_f32 = payload  # type: ignore[misc]
        block_norms = torch.stack([torch.norm(v, p=2) for v in vals_f32])

        # stats
        med = float(torch.median(block_norms).item())
        min_median_norm = min(min_median_norm, med)
        max_median_norm = max(max_median_norm, med)

        # Decompress on the local shard
        # The `ref` parameter should be the local shard of the parameter `p`
        if isinstance(p, DT):
            ref = p.to_local()
        else:
            ref = p

        decompressed = compressor.batch_decompress(
            ref,
            idxs,
            vals_f32,
            xshapes[name],
            totalks[name],
            quantize_params=None,
            block_norms=block_norms,
            normalise=False,
            clip_norm=True,
        )

        full_grad_local_shard = transformer.decode(decompressed, use_dct=use_dct)
        full_grad_local_shard = full_grad_local_shard.to(
            dtype=p.dtype, device=p.device, non_blocking=True
        )

        # Free intermediate pieces ASAP
        del vals_f32, idxs, vals, qps, ref, decompressed
        decompressed = None

        # Apply the local shard of the gradient
        if isinstance(p, DT):
            # For DTensor, assign the local shard directly
            # Ensure full_grad_local_shard is a local tensor
            if not isinstance(full_grad_local_shard, torch.Tensor):
                raise TypeError("Expected full_grad_local_shard to be a torch.Tensor")

            # Create a DTensor from the local shard
            new_grad_dtensor = distribute_tensor(
                full_grad_local_shard,
                device_mesh=p.device_mesh,
                placements=p.placements,
                # No src_data_rank needed as each rank has its own shard
            )
            p.grad = new_grad_dtensor
            del new_grad_dtensor
        else:
            # For replicated parameters, the full_grad_local_shard is the full gradient
            p.grad = full_grad_local_shard

        if p.grad is not None and not torch.isfinite(p.grad).all():
            p.grad = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # ---- apply update immediately for THIS param and free its grad ----
        optimizer.step()
        p.grad = None  # free grad storage right away
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # optional W&B (master only)
    if (
        on_src
        and wandb_run is not None
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

    # Extra safety: ensure no grads are left allocated
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


async def update_peers(instance: NeuronT, window: int, peer_start: float) -> None:
    # Check if peers list is empty and fetch previous list if needed
    if len(instance.comms.peers) == 0:
        tplr.logger.info(
            "Current peers list is empty, attempting to fetch previous peer list"
        )
        result = await instance.comms.get_peer_list(fetch_previous=True)
        if result is not None:
            prev_peers, prev_reserve, prev_update_window = result
            tplr.logger.info(
                f"Got previous peer list with {len(prev_peers)} peers "
                f"and update window {prev_update_window}"
            )
            instance.comms.peers = prev_peers
            instance.comms.reserve_peers = prev_reserve

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
            next_peers, reserve_peers, peers_update_window = result
            tplr.logger.info(
                f"Got peer list {next_peers} and update window "
                f"{peers_update_window} from bucket"
            )
            if (
                instance.peers_update_window is None
                or peers_update_window > instance.peers_update_window
            ):
                instance.next_peers = next_peers
                instance.next_reserve_peers = reserve_peers
                instance.peers_update_window = peers_update_window
                tplr.logger.info("This list is new, updating next_peers")

    # Update peers, if it's time
    if instance.next_peers is not None and window >= instance.peers_update_window:
        # ── atomic switch ─────────────────────────────────────────────
        instance.comms.peers = instance.next_peers
        instance.comms.reserve_peers = (
            instance.next_reserve_peers
            if instance.next_reserve_peers is not None
            else []
        )
        late_text = (
            f"{window - instance.peers_update_window} windows late"
            if window - instance.peers_update_window > 0
            else "on time"
        )
        tplr.logger.info(
            f"{tplr.P(window, tplr.T() - peer_start)} Updated peers "
            f"{late_text} - gather:{len(instance.comms.peers)}, "
            f"reserve:{len(instance.comms.reserve_peers)}. Next update "
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

    leader_uid: int = instance.comms.metagraph.S.argmax().item()

    start_w = checkpoint_current_window + 1
    target_w = instance.current_window
    tplr.logger.info(f"Replaying windows {start_w} ... {target_w - 1}")

    # Verify checkpoint loaded correctly before applying any gradients
    if checkpoint_current_window > 0 and instance.is_master:
        tplr.logger.info(
            f"Verifying checkpoint state at window {checkpoint_current_window}"
        )
        debug_fetch = await instance.comms.get(
            uid=str(leader_uid),
            window=checkpoint_current_window,
            key="debug",
            local=False,
            stale_retention=10,
        )

        if debug_fetch.success and isinstance(debug_fetch.data, dict):
            debug_dict = debug_fetch.data  # validator's payload

            cmp = await compare_model_with_debug_dict(
                instance.model,
                debug_dict,
                param_avg_change={},  # Empty since we haven't started tracking yet
                learning_rate=instance.hparams.learning_rate,
            )
            if cmp["success"]:
                tplr.logger.info(
                    f"✓ Checkpoint verification: model matches window {checkpoint_current_window} "
                    f"(l2_norm={cmp['l2_norm']:.4f}, avg_steps_behind={cmp['avg_steps_behind']:.3f})"
                )
                if cmp["l2_norm"] > 0.1:  # Threshold for acceptable difference
                    tplr.logger.warning(
                        f"⚠️ Large L2 norm difference detected: {cmp['l2_norm']:.4f}. "
                        f"Checkpoint may not have loaded correctly."
                    )
            else:
                tplr.logger.warning(
                    f"⚠️ Could not verify checkpoint state for window {checkpoint_current_window}"
                )
        else:
            tplr.logger.info(
                f"No debug dict available for window {checkpoint_current_window}, skipping verification"
            )

    prev_param_state: dict[str, torch.Tensor] = {}
    param_avg_change: dict[str, torch.Tensor] = {}
    alpha: float = 0.20
    slice_idx = slice(0, 2)

    while start_w < target_w:
        tplr.logger.info(f"  • window {start_w}")

        # ------------------------------------------------------------------
        # 1) Fetch the aggregated object dumped by the leader validator.
        # ------------------------------------------------------------------
        if instance.is_master:
            fetch = await instance.comms.get(
                uid=str(leader_uid),
                window=start_w,
                key="aggregator",
                timeout=60,
                local=False,
                stale_retention=10,
            )

            # ── A. aggregated object exists → normal path ────────────────────
            if fetch.success and fetch.data is not None and "state_dict" in fetch.data:
                payload = fetch.data

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
                gather_ns = None
                is_last_window = start_w == target_w - 1
                tplr.logger.warning(
                    "    ↳ %s – %s",
                    "not available" if fetch is None else "malformed payload",
                    "attempting gather‑fallback" if is_last_window else "skipping",
                )

                if is_last_window:
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
                        time_min=time_min,
                        time_max=time_max,
                    )

                if gather_ns is None:
                    tplr.logger.warning("    ↳ gather‑fallback failed – skipping")
                else:
                    tplr.logger.info("    ↳ gather‑fallback succeeded – applying")
        else:
            gather_ns = None

        # Broadcast whether we should skip this window (master decides)
        skip_window = False
        if dist.is_available() and dist.is_initialized():
            if instance.is_master:
                skip_flag = 1 if gather_ns is None else 0
                skip_tensor = torch.tensor(
                    [skip_flag], dtype=torch.int32, device=instance.config.device
                )
            else:
                skip_tensor = torch.tensor(
                    [0], dtype=torch.int32, device=instance.config.device
                )
            dist.broadcast(skip_tensor, src=0)
            skip_window = bool(skip_tensor.item())
        elif instance.is_master and gather_ns is None:
            skip_window = True

        # If skipping, continue to next window without updating scheduler
        if skip_window:
            instance.global_step = start_w - instance.start_window
            start_w += 1
            continue

        # ------------------------------------------------------------------
        # 2) All ranks apply the update.
        # ------------------------------------------------------------------
        # Synchronize all ranks before applying the outer step to ensure
        # they're processing the same window together
        if dist.is_available() and dist.is_initialized():
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                dist.barrier(device_ids=[device_id])
            else:
                dist.barrier()

        outer_step(
            instance.model,
            instance.outer_optimizer,
            gather_result=gather_ns,
            transformer=instance.transformer,
            compressor=instance.compressor,
            xshapes=instance.xshapes,
            totalks=instance.totalks,
            device=instance.config.device,
            is_master=instance.is_master,  # rank-0 handles logging
            world_size=instance.world_size,
            use_dct=instance.hparams.use_dct,
            wandb_run=instance.wandb if instance.is_master else None,
            global_step=instance.global_step,
        )

        # advance LR scheduler if one exists.
        inner_sched: LRScheduler | None = getattr(instance, "inner_scheduler", None)
        if inner_sched is not None:
            for _ in range(instance.hparams.inner_steps):
                inner_sched.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ──────────────────────────────────────────────────────────────────────
        # 3) Debug‑dict comparison to estimate “how many steps behind” we are
        # ──────────────────────────────────────────────────────────────────────
        try:
            if instance.is_master:
                debug_fetch = await instance.comms.get(
                    uid=str(leader_uid),
                    window=start_w,
                    key="debug",
                    local=False,
                    stale_retention=10,
                )

                if debug_fetch.success and isinstance(debug_fetch.data, dict):
                    debug_dict = debug_fetch.data  # validator's payload

                    # --- update EMA of parameter‑slice changes ------------------
                    bare_model = getattr(instance.model, "module", instance.model)
                    for name, p in bare_model.named_parameters():
                        if p.numel() < 2:
                            continue

                        # Handle DTensor parameters
                        if isinstance(p, DT):
                            curr_slice = (
                                p.to_local().detach().cpu().flatten()[slice_idx]
                            )
                        else:
                            curr_slice = p.detach().cpu().flatten()[slice_idx]

                        if name in prev_param_state:
                            delta = (curr_slice - prev_param_state[name]).abs()
                            if name not in param_avg_change:
                                param_avg_change[name] = delta.clone()
                            else:
                                param_avg_change[name].mul_(1 - alpha).add_(
                                    delta * alpha
                                )
                        prev_param_state[name] = curr_slice.clone()

                    # --- call shared comparison helper --------------------------
                    lr = instance.outer_optimizer.param_groups[0]["lr"]
                    cmp = await compare_model_with_debug_dict(
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

        if dist.is_available() and dist.is_initialized():
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                dist.barrier(device_ids=[device_id])
            else:
                dist.barrier()

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

    # Collect per‑tensor step‑ratio vectors so we can take
    # a single global median later
    tensors = 0
    step_ratio_list: list[torch.Tensor] = []

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
        if isinstance(p, DT):
            curr_slice = p.to_local().data.flatten()[index_range[0] : index_range[1]]
        else:
            curr_slice = p.data.flatten()[index_range[0] : index_range[1]]

        debug_slice = torch.tensor(
            debug_dict[key], dtype=p.dtype, device=curr_slice.device
        )

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
                param_avg_change[name].to(curr_slice.device), min=min_step_size
            )
            if step_vec.numel() != abs_vec.numel():
                # fallback if stored slice has wrong length
                step_vec = abs_vec.new_full(abs_vec.size(), learning_rate)
        else:
            step_vec = abs_vec.new_full(abs_vec.size(), learning_rate)

        step_ratio = abs_vec / step_vec
        # Accumulate for global median
        step_ratio_list.append(step_ratio)
        max_steps = max(max_steps, step_ratio.max().item())
        tensors += 1

    l2_norm = math.sqrt(total_squared_diff)
    avg_l2_norm = math.inf if tensors == 0 else l2_norm / param_count
    avg_abs_diff = math.inf if tensors == 0 else total_abs_diff / param_count
    if not step_ratio_list:  # nothing compared
        median_steps = math.inf
        max_steps = math.inf
    else:
        all_steps = torch.cat([t.flatten() for t in step_ratio_list])
        median_steps = all_steps.median().item()

    return {
        "success": True,
        "l2_norm": l2_norm,
        "avg_l2_norm": avg_l2_norm,
        "avg_abs_diff": avg_abs_diff,
        "max_diff": max_diff,
        "avg_steps_behind": median_steps,
        "max_steps_behind": max_steps,
        "param_count": param_count,
        "learning_rate": learning_rate,
    }


@torch.no_grad()
async def check_uid_index_overlap(
    neuron: NeuronT,
    gather_result: SimpleNamespace,
    window: int,
    *,
    overlap_threshold: float = 0.90,
) -> dict:
    """
    For every peer-pair compute the per-chunk *set* overlap of their top-k index
    lists on each parameter.  A pair is flagged **only if the size-weighted
    average across *all* checked parameters** is ≥ `overlap_threshold`.
    """

    # ── 0. basic sanity ───────────────────────────────────────────────────
    uids: list[int] = list(getattr(gather_result, "uids", []))
    Ptot = len(uids)
    if Ptot < 2:
        tplr.logger.info("[overlap] <2 peers – skip")
        return dict(
            pairs_checked=0,
            pairs_high_ovlap=0,
            ratio_high_ovlap=0.0,
            mean_overlap=0.0,
            min_overlap=0.0,
            max_overlap=0.0,
            pairs_over_thresh=[],
            uids_over_thresh={},
        )

    ts_map = dict(
        zip(
            uids,
            await asyncio.gather(
                *[neuron.comms.gradient_timestamp(uid, window - 1) for uid in uids]
            ),
        )
    )

    # ── 1. bookkeeping ────────────────────────────────────────────────────
    pair_acc: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    total_weighted_sum = 0.0
    total_weight = 0.0

    # ── 2. iterate over parameters that have compressed indices ───────────
    bare_model = getattr(neuron.model, "module", neuron.model)
    for pname, _ in bare_model.named_parameters():
        idx_key = pname + "idxs"
        idxs_all = getattr(gather_result.state_dict, idx_key, None)
        if idxs_all is None:
            continue

        # Get values for unpacking shape
        vals_key = pname + "vals"
        vals_all = getattr(gather_result.state_dict, vals_key, None)
        if vals_all is None:
            continue

        # Unpack all 12-bit packed indices using values shape
        unpacked_indices = []
        for i in range(Ptot):
            idx_data = idxs_all[i] if isinstance(idxs_all, list) else idxs_all
            val_data = vals_all[i] if isinstance(vals_all, list) else vals_all

            # 12-bit packed format - use values shape for unpacking
            unpacked = unpack_12bit_indices(
                idx_data.to(neuron.config.device), val_data.shape
            )
            unpacked_indices.append(unpacked)

        idxs_tensor = torch.stack(unpacked_indices, dim=0)
        P, *chunk_dims, k = idxs_tensor.shape
        C = int(torch.prod(torch.tensor(chunk_dims)))  # num chunks
        idxs_flat = idxs_tensor.reshape(P, C, k)

        param_weight = C * k  # size weight

        for i in range(P):
            for j in range(i + 1, P):
                a = idxs_flat[i].unsqueeze(-1)  # (C,k,1)
                b = idxs_flat[j].unsqueeze(-2)  # (C,1,k)
                inter = (a == b).any(-1).sum(-1)  # (C,)
                mean_frac = (inter.float() / k).mean().item()

                total_weighted_sum += mean_frac * param_weight
                total_weight += param_weight

                acc = pair_acc[(i, j)]
                acc[0] += mean_frac * param_weight
                acc[1] += param_weight

    # ── 3. second pass – decide offenders & track min/max ─────────────────
    pairs_high, pairs_over, uids_with_slashing = 0, [], {}
    min_pair, min_val = None, 1.0
    max_pair, max_val = None, 0.0

    for (i, j), (w_sum, w_tot) in pair_acc.items():
        avg_overlap = w_sum / w_tot if w_tot > 0 else 0.0

        # --- track global min / max --------------------------------------
        if avg_overlap < min_val:
            min_val, min_pair = avg_overlap, (uids[i], uids[j])
        if avg_overlap > max_val:
            max_val, max_pair = avg_overlap, (uids[i], uids[j])
        # ------------------------------------------------------------------

        if avg_overlap >= overlap_threshold:
            pairs_high += 1
            uid_i, uid_j = uids[i], uids[j]
            offender = uid_i if ts_map[uid_i] >= ts_map[uid_j] else uid_j
            uids_with_slashing[offender] = determine_slash_egregiousness(avg_overlap)

            pairs_over.append((uid_i, uid_j, avg_overlap))
            tplr.logger.debug(
                f"[overlap] peers {uid_i}/{uid_j} share "
                f"{avg_overlap * 100:.1f}% of indices (size-weighted avg)"
            )

    mean_overlap = total_weighted_sum / total_weight if total_weight else 0.0
    ratio_high = pairs_high / len(pair_acc) if pair_acc else 0.0

    # ── 4. summary log with min / max -------------------------------------
    tplr.logger.info(
        f"[overlap] {len(pair_acc)} pairs, {pairs_high} ≥{overlap_threshold * 100:.0f}% "
        f"({ratio_high * 100:.2f}%), size-weighted mean {mean_overlap * 100:.1f}%"
    )
    if min_pair is not None and max_pair is not None:
        tplr.logger.info(
            f"[overlap]   min {min_val * 100:.1f}%  (peers {min_pair[0]}/{min_pair[1]}) ; "
            f"max {max_val * 100:.1f}%  (peers {max_pair[0]}/{max_pair[1]})"
        )
    if uids_with_slashing:
        tplr.logger.warning(
            f"[overlap] offenders: {sorted(list(uids_with_slashing.keys()))}"
        )

    return dict(
        pairs_checked=len(pair_acc),
        pairs_high_ovlap=pairs_high,
        ratio_high_ovlap=ratio_high,
        mean_overlap=mean_overlap,
        min_overlap=min_val if min_pair is not None else 0.0,
        max_overlap=max_val if max_pair is not None else 0.0,
        pairs_over_thresh=pairs_over,
        uids_over_thresh=uids_with_slashing,
    )


def determine_slash_egregiousness(overlap_pct: float) -> str:
    """
    Based on the overlap_pct, return a level corresponding
    to an action which will be taken

    Args:
        overlap_pct: The percentage of overlap in the grads with
             other miners

    Returns:
        Category of overlap pct
    """

    invalid_number = overlap_pct < 0.0 or overlap_pct > 1.0
    if invalid_number:
        raise ValueError(f"overlap_pct must be between 0.0 and 1.0, got {overlap_pct}")

    egregiousness = "high"
    if overlap_pct >= 0.5:
        egregiousness = "max"
    if overlap_pct >= 0.6:
        egregiousness = "mega"

    return egregiousness


def instantiate_slashing_multiplier():
    """Centralize slashing config

    We multiply these percentages against the base final_score
    """
    return {
        "high": 0.5,  # case when similarity high
        "max": 0.0,  # case when similarity >= 95%
        "mega": 0.0,  # case when similarity = 100%
    }
