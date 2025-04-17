"""
Evaluation utilities for the validator
"""

from __future__ import annotations

import asyncio
import copy
import random
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import torch
from transformers import PreTrainedTokenizerBase  # type: ignore

import tplr
from tplr.logging import logger
from .r2_dataset import R2DatasetLoader
from .compress import TransformDCT, CompressDCT
from .comms import Comms
from .hparams import HParams


# ---------------------------------------------------------------------------#
# Utility aliases – keeps signatures readable
# ---------------------------------------------------------------------------#
TensorDict = Dict[str, torch.Tensor]
ShapeDict = Dict[str, Tuple[int, ...]]
KsDict = Dict[str, int]
UID = int


# ---------------------------------------------------------------------------#
# Core helpers
# ---------------------------------------------------------------------------#
def apply_compressed_gradient(
    model: torch.nn.Module,
    state_dict: TensorDict,
    transformer: TransformDCT,
    compressor: CompressDCT,
    xshapes: ShapeDict,
    totalks: KsDict,
    device: torch.device,
    lr: float,
) -> torch.nn.Module:
    """
    Apply the compressed gradient stored in ``state_dict`` to ``model``
    using a sign‑based update.

    Returns
    -------
    torch.nn.Module
        Reference to ``model`` (updated in‑place, returned for convenience).
    """
    for n, p in model.named_parameters():
        idxs_key: str = n + "idxs"
        vals_key: str = n + "vals"
        idxs = state_dict.get(idxs_key, None)
        vals = state_dict.get(vals_key, None)
        if idxs is not None and vals is not None:
            idxs = idxs.to(device)
            vals = vals.to(device)
            decompressed = compressor.decompress(
                p.to(device), idxs, vals, xshapes[n], totalks[n]
            )
            del idxs, vals
            grad_tensor = transformer.decode(decompressed).to(device)
            del decompressed
            # Sign‑based SGD
            p.data.sub_(grad_tensor.sign(), alpha=lr)
            del grad_tensor
            torch.cuda.empty_cache()
        else:
            logger.info(f"Gradient data missing for parameter {n}, skipping.")
    return model


async def load_and_compare_pages(
    uid: UID,
    sync_window: int,
    hparams: HParams,
    state_dict: Mapping[str, Any],
) -> Tuple[Optional[List[int]], List[int]]:
    """
    Load the *local* pages and compare them with the miner‑provided pages.

    Returns
    -------
    Tuple[Optional[List[int]], List[int]]
        (miner_pages, local_pages)
    """
    miner_pages: Optional[List[int]] = state_dict.get("metadata", {}).get(
        "pages_info", None
    )
    local_pages: List[int] = await R2DatasetLoader.next_pages(
        offset=sync_window * hparams.pages_per_window,
        n_pages=hparams.pages_per_window,
        seed=uid,
    )
    if miner_pages is not None:
        if local_pages != miner_pages:
            logger.warning(
                f"Pages mismatch for UID {uid}: miner pages {miner_pages} vs local pages {local_pages}"
            )
        else:
            logger.info(f"Pages verified for UID {uid}: pages match.")
    else:
        logger.info(f"Using local pages for UID {uid} as miner metadata is missing.")
    return miner_pages, local_pages


def compute_average_loss(
    model: torch.nn.Module,
    batches: Sequence[Sequence[int]],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    sample_rate: float,
) -> Tuple[float, int, List[int], int]:
    """
    Sample a fraction of ``batches`` and compute the mean LM loss.

    Returns
    -------
    Tuple[float, int, List[int], int]
        (avg_loss, n_sampled, sampled_indices, total_batches)
    """
    total_batches: int = len(batches)
    sample_size: int = max(1, int(total_batches * sample_rate))
    sampled_indices: List[int] = sorted(
        random.sample(range(total_batches), sample_size)
    )
    total_loss: float = 0.0
    count: int = 0

    model.eval()
    with torch.no_grad():
        for i in sampled_indices:
            batch = batches[i]
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            pad_raw = tokenizer.pad_token_id
            pad_val: int = pad_raw if isinstance(pad_raw, int) else 0
            pad_t = torch.tensor(pad_val, dtype=input_ids.dtype, device=device)
            mask = input_ids.eq(pad_t)  # Tensor[bool]
            input_ids = input_ids.masked_fill(mask, 0)
            with torch.autocast("cuda"):
                outputs = model(input_ids=input_ids, labels=input_ids)
            total_loss += float(outputs.loss)
            count += 1
            del input_ids, outputs
            torch.cuda.empty_cache()

    avg_loss: float = total_loss / count if count else 0.0
    return avg_loss, count, sampled_indices, total_batches


def evaluate_loss_change(
    model: torch.nn.Module,
    batches: Sequence[Sequence[int]],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    sample_rate: float,
    state_dict: TensorDict,
    transformer: TransformDCT,
    compressor: CompressDCT,
    xshapes: ShapeDict,
    totalks: KsDict,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> Tuple[float, float, int, int, List[int], int]:
    """
    Compute loss *before* and *after* the peer gradient is applied.
    """
    loss_before, count_before, sampled_indices, total_batches = compute_average_loss(
        model, batches, tokenizer, device, sample_rate
    )
    logger.info(
        f"Loss before gradient: {loss_before} on {count_before}/{total_batches} batches"
    )

    current_lr: float = scheduler.get_last_lr()[0]
    model_after = apply_compressed_gradient(
        model, state_dict, transformer, compressor, xshapes, totalks, device, current_lr
    )

    loss_after, count_after, _, _ = compute_average_loss(
        model_after, batches, tokenizer, device, sample_rate
    )
    logger.info(f"Loss after gradient: {loss_after}")

    return (
        loss_before,
        loss_after,
        count_before,
        count_after,
        sampled_indices,
        total_batches,
    )


def compute_improvement_metrics(
    loss_before_own: float,
    loss_after_own: float,
    loss_before_random: float,
    loss_after_random: float,
) -> Tuple[float, float, float, int]:
    """
    Produce relative improvement metrics + a simple binary indicator.
    """
    loss_improvement_own: float = loss_before_own - loss_after_own
    relative_improvement_own: float = (
        loss_improvement_own / loss_before_own if loss_before_own > 0 else 0.0
    )

    loss_improvement_random: float = loss_before_random - loss_after_random
    relative_improvement_random: float = (
        loss_improvement_random / loss_before_random if loss_before_random > 0 else 0.0
    )

    gradient_score: float = relative_improvement_random
    binary_indicator: int = (
        1 if relative_improvement_own > relative_improvement_random else -1
    )
    return (
        relative_improvement_own,
        relative_improvement_random,
        gradient_score,
        binary_indicator,
    )


async def evaluate_peer(
    uid: UID,
    state_dict: TensorDict,
    sync_window: int,
    hparams: Any,
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    transformer: TransformDCT,
    compressor: CompressDCT,
    xshapes: ShapeDict,
    totalks: KsDict,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    random_batches: Sequence[Sequence[int]],
    random_pages: List[int],
    comms: Comms,
) -> Dict[str, Any]:
    """
    Evaluate a peer's gradient on *own* data and shared *random* data.
    Returns a rich metrics dict.
    """
    start_time: float = tplr.T()

    # --------------- own data ----------------#
    loader_own, _ = await R2DatasetLoader.get_loader(
        offset=sync_window * hparams.pages_per_window,
        hparams=hparams,
        tokenizer=tokenizer,
        data_type="own",
        seed=int(uid),
    )
    batches_own: List[Sequence[int]] = [
        cast(Sequence[int], batch.tolist() if hasattr(batch, "tolist") else list(batch))
        for batch in loader_own
    ]
    del loader_own
    torch.cuda.empty_cache()

    model_own_eval = await asyncio.to_thread(copy.deepcopy, model)
    own_task = asyncio.to_thread(
        evaluate_loss_change,
        model_own_eval,
        batches_own,
        tokenizer,
        device,
        hparams.validator_sample_rate,
        state_dict,
        transformer,
        compressor,
        xshapes,
        totalks,
        scheduler,
    )

    # --------------- random data -------------#
    model_random_eval = await asyncio.to_thread(copy.deepcopy, model)
    random_task = asyncio.to_thread(
        evaluate_loss_change,
        model_random_eval,
        random_batches,
        tokenizer,
        device,
        hparams.validator_sample_rate,
        state_dict,
        transformer,
        compressor,
        xshapes,
        totalks,
        scheduler,
    )

    (
        loss_before_own,
        loss_after_own,
        _,
        _,
        sampled_indices_own,
        _,
    ) = await own_task
    logger.info(
        f"UID {uid}: Own data evaluation completed. Loss before: {loss_before_own}, after: {loss_after_own}"
    )

    del model_own_eval, own_task, batches_own
    torch.cuda.empty_cache()

    (
        loss_before_random,
        loss_after_random,
        _,
        _,
        sampled_indices_random,
        _,
    ) = await random_task
    logger.info(
        f"UID {uid}: Random data evaluation completed. Loss before: {loss_before_random}, after: {loss_after_random}"
    )

    del model_random_eval, random_task
    torch.cuda.empty_cache()

    (
        relative_improvement_own,
        relative_improvement_random,
        gradient_score,
        binary_indicator,
    ) = compute_improvement_metrics(
        loss_before_own, loss_after_own, loss_before_random, loss_after_random
    )

    miner_pages, local_pages = await load_and_compare_pages(
        uid, sync_window, hparams, state_dict
    )

    total_time: float = tplr.T() - start_time
    logger.info(f"UID {uid}: Completed evaluation in {total_time} seconds")

    # --- Sync score ----------------------------------------------------------------
    debug_result = await comms.get(
        uid=str(uid),
        window=sync_window - 1,
        key="debug",
        local=False,
        stale_retention=10,
    )

    if debug_result is not None and debug_result[0] is not None:
        miner_debug_dict: Dict[str, Any] = debug_result[0]
        sync_score = await compute_sync_score(
            model, miner_debug_dict, scheduler, index_range=(10, 12)
        )
        logger.info(f"UID {uid}: Sync score: {sync_score}")
    else:
        sync_score = 0.0

    result: Dict[str, Any] = {
        "uid": uid,
        "loss_before_per_batch_own": loss_before_own,
        "loss_after_per_batch_own": loss_after_own,
        "relative_improvement_own": relative_improvement_own,
        "loss_before_per_batch_random": loss_before_random,
        "loss_after_per_batch_random": loss_after_random,
        "relative_improvement_random": relative_improvement_random,
        "gradient_score": gradient_score,
        "binary_indicator": binary_indicator,
        "miner_pages": miner_pages,
        "local_pages": local_pages,
        "pages_random": random_pages,
        "sync_score": sync_score,
    }

    del (
        loss_before_own,
        loss_after_own,
        loss_before_random,
        loss_after_random,
        sampled_indices_own,
        sampled_indices_random,
    )
    torch.cuda.empty_cache()

    return result


async def evaluate_peers_parallel(
    evaluation_uids: Sequence[UID],
    comms: Any,
    sync_window: int,
    hparams: HParams,
    tokenizer: PreTrainedTokenizerBase,
    config: Any,
    model: torch.nn.Module,
    transformer: TransformDCT,
    compressor: CompressDCT,
    xshapes: ShapeDict,
    totalks: KsDict,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    time_min: Optional[float],
    time_max: Optional[float],
) -> Tuple[Dict[UID, Optional[Dict[str, Any]]], Dict[str, Any]]:
    """
    Evaluate multiple peers concurrently, limited by ``hparams.parallel_eval_uids``.
    """
    offset: int = sync_window * hparams.pages_per_window
    random_loader, random_pages = await R2DatasetLoader.get_loader(
        offset=offset, hparams=hparams, tokenizer=tokenizer, data_type="random"
    )
    random_batches: List[Sequence[int]] = [
        cast(Sequence[int], batch.tolist() if hasattr(batch, "tolist") else list(batch))
        for batch in random_loader
    ]
    del random_loader
    torch.cuda.empty_cache()

    sem = asyncio.Semaphore(hparams.parallel_eval_uids)

    async def evaluate_uid(uid: UID) -> Tuple[UID, Optional[Dict[str, Any]]]:
        async with sem:
            logger.info(f"Evaluating uid: {uid}")
            eval_result = await comms.get(
                uid=str(uid),
                window=sync_window,
                key="gradient",
                local=False,
                stale_retention=10,
                time_min=time_min,
                time_max=time_max,
            )
            if eval_result is not None and eval_result[0] is not None:
                state_dict, _ = eval_result
                eval_payload = await evaluate_peer(
                    uid,
                    state_dict,
                    sync_window,
                    hparams,
                    tokenizer,
                    model,
                    transformer,
                    compressor,
                    xshapes,
                    totalks,
                    device,
                    scheduler,
                    random_batches,  # noqa
                    random_pages,
                    comms,
                )
                return uid, eval_payload
            else:
                logger.info(f"No gradient received from UID {uid}. Penalizing score.")
                return uid, None

    tasks: List["asyncio.Task[Tuple[UID, Optional[Dict[str, Any]]]]"] = [
        asyncio.create_task(evaluate_uid(uid)) for uid in evaluation_uids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    del random_batches
    torch.cuda.empty_cache()

    eval_dict: Dict[UID, Optional[Dict[str, Any]]] = {}
    for res in results:
        if isinstance(res, Exception):
            # TODO: attach uid context & traceback
            continue
        uid, result = res  # type: ignore[misc]
        eval_dict[uid] = result

    aggregated_metrics: Dict[str, Any] = aggregate_evaluation_metrics(eval_dict)
    return eval_dict, aggregated_metrics


def compute_avg_loss(
    model: torch.nn.Module,
    batches: Sequence[Sequence[int]],
    sampled_indices: Sequence[int],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> Tuple[float, int]:
    """
    Compute the average loss over ``sampled_indices`` only.
    """
    total_loss: float = 0.0
    n_batches_count: int = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(batches):
            if i in sampled_indices:
                input_ids = torch.tensor(batch, dtype=torch.long, device=device)
                pad_raw = tokenizer.pad_token_id
                pad_val: int = pad_raw if isinstance(pad_raw, int) else 0
                pad_t = torch.tensor(pad_val, dtype=input_ids.dtype, device=device)
                mask = input_ids.eq(pad_t)
                input_ids = input_ids.masked_fill(mask, 0)
                outputs = model(input_ids=input_ids, labels=input_ids)
                total_loss += float(outputs.loss)
                n_batches_count += 1
    return (
        total_loss / n_batches_count if n_batches_count > 0 else 0.0,
        n_batches_count,
    )


def apply_gradient_update(
    model: torch.nn.Module,
    state_dict: TensorDict,
    transformer: TransformDCT,
    compressor: CompressDCT,
    xshapes: ShapeDict,
    totalks: KsDict,
    device: torch.device,
    lr: float,
) -> torch.nn.Module:
    """
    In‑place application of the *peer* gradient.
    """
    for n, p in model.named_parameters():
        idxs_key: str = n + "idxs"
        vals_key: str = n + "vals"
        if idxs_key in state_dict and vals_key in state_dict:
            idxs = state_dict[idxs_key].to(device)
            vals = state_dict[vals_key].to(device)
            grad = transformer.decode(
                compressor.decompress(
                    p.data,
                    idxs,
                    vals,
                    xshapes[n],
                    totalks[n],
                )
            ).to(device)
            p.data.sub_(grad.sign(), alpha=lr)
    return model


def aggregate_evaluation_metrics(
    eval_results: Mapping[UID, Optional[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """
    Aggregate peer metrics for logging/dashboard.

    Only non‑None peer results are considered.
    """
    keys: Tuple[str, ...] = (
        "loss_before_per_batch_own",
        "loss_after_per_batch_own",
        "loss_before_per_batch_random",
        "loss_after_per_batch_random",
        "binary_indicator",
    )
    totals: Dict[str, float] = {key: 0.0 for key in keys}
    valid_results: List[Mapping[str, Any]] = [
        res for res in eval_results.values() if res is not None
    ]
    count: int = len(valid_results) or 1  # avoid div/0

    for res in valid_results:
        for key in keys:
            totals[key] += float(res.get(key, 0.0))

    aggregated: Dict[str, Any] = {key: totals[key] / count for key in keys}
    aggregated["evaluated_count"] = len(valid_results)
    return aggregated


async def compute_sync_score(
    model: torch.nn.Module,
    debug_dict: Mapping[str, Any],
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    index_range: Tuple[int, int] = (10, 12),
) -> float:
    """
    Compare local model params with the miner's debug dict.
    """
    current_lr: float = scheduler.get_last_lr()[0]
    neuron_call = await tplr.neurons.compare_model_with_debug_dict(
        model=model,
        debug_dict=cast("dict[str, list[float]]", debug_dict),  # satisfy stub
        learning_rate=current_lr,
        index_range=index_range,
    )
    if not neuron_call.get("success", False):
        return 0.0
    avg_steps_behind: float = neuron_call.get("avg_steps_behind", 5.0)
    x = min(avg_steps_behind, 5.0)
    sync_score: float = max(0.0, (1.0 - x / 5.0) ** 2.5)
    return sync_score
