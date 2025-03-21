import torch
import random
from tplr.logging import logger
import tplr
from .r2_dataset import R2DatasetLoader
import asyncio
import copy


def evaluate_model_loss(model, loader, tokenizer, device):
    """
    Evaluates a model over a provided data loader and returns the average loss.
    """
    total_loss = 0.0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = torch.tensor(batch, dtype=torch.long).to(device)
            labels = input_ids.clone()
            labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1
            del input_ids, labels, outputs
            torch.cuda.empty_cache()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, num_batches

def apply_compressed_gradient(model, state_dict, transformer, compressor, xshapes, totalks, device, lr):
    """
    Applies the compressed gradient (stored in state_dict) to the model parameters.
    """
    for n, p in model.named_parameters():
        idxs_key = n + 'idxs'
        vals_key = n + 'vals'
        idxs = state_dict.get(idxs_key, None)
        vals = state_dict.get(vals_key, None)
        if idxs is not None and vals is not None:
            idxs = idxs.to(device)
            vals = vals.to(device)
            decompressed = compressor.decompress(p.to(device), idxs, vals, xshapes[n], totalks[n])
            # Remove temporary tensors for idxs/vals
            del idxs, vals
            grad_tensor = transformer.decode(decompressed).to(device)
            del decompressed  # free decompressed tensor
            # Apply sign-based update
            p.data.sub_(grad_tensor.sign(), alpha=lr)
            del grad_tensor
            torch.cuda.empty_cache()  # allow fragmentation to be cleaned up
        else:
            logger.info(f"Gradient data missing for parameter {n}, skipping.")
    return model


async def load_and_compare_pages(uid, sync_window, hparams, tokenizer, state_dict):
    """
    Loads local pages using uid as seed and compares them with miner-provided pages.
    Returns (miner_pages, local_pages).
    """
    miner_pages = state_dict.get("metadata", {}).get("pages_info", None)
    local_pages = await R2DatasetLoader.next_pages(
        offset=sync_window,
        n_pages=hparams.pages_per_window,
        seed=uid
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

async def create_loader_from_pages(pages, hparams, tokenizer, sync_window):
    data_start = tplr.T()
    loader = await tplr.r2_dataset.R2DatasetLoader.create(
        batch_size=hparams.batch_size,
        sequence_length=hparams.sequence_length,
        pages_info=pages,
        tokenizer=tokenizer
    )
    logger.info(f'{tplr.P(sync_window, tplr.T() - data_start)} Loaded evaluation data using pages: {[p[1] for p in pages]}')
    return loader

def collect_batches(loader):
    batches = []
    for batch in loader:
        batches.append(batch)
    return batches

def compute_average_loss(model, batches, tokenizer, device, sample_rate):
    total_batches = len(batches)
    sample_size = max(1, int(total_batches * sample_rate))
    sampled_indices = sorted(random.sample(range(total_batches), sample_size))
    total_loss = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for i in sampled_indices:
            batch = batches[i]
            input_ids = torch.tensor(batch, dtype=torch.long).to(device)
            labels = input_ids.clone()
            labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
            # Use autocast for mixed precision during the forward pass:
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
            count += 1
            del input_ids, labels, outputs
            torch.cuda.empty_cache()
    avg_loss = total_loss / count if count else 0.0
    return avg_loss, count, sampled_indices, total_batches

def evaluate_loss_change(model, batches, tokenizer, device, sample_rate, 
                         state_dict, transformer, compressor, xshapes, totalks, scheduler):
    model.eval()
    loss_before, count_before, sampled_indices, total_batches = compute_average_loss(
        model, batches, tokenizer, device, sample_rate
    )
    logger.info(f"Loss before gradient: {loss_before} on {count_before}/{total_batches} batches")
    
    # Use the current learning rate from scheduler 
    current_lr = scheduler.get_last_lr()[0]
    
    # Apply the compressed gradient update to the model copy
    model_after = apply_compressed_gradient(model, state_dict, transformer, compressor, xshapes, totalks, device, current_lr)
    loss_after, count_after, _, _ = compute_average_loss(
        model_after, batches, tokenizer, device, sample_rate
    )
    logger.info(f"Loss after gradient: {loss_after}")
    
    return loss_before, loss_after, count_before, count_after, sampled_indices, total_batches

def compute_improvement_metrics(loss_before_own, loss_after_own, loss_before_random, loss_after_random):
    """
    Computes loss improvements and computes the relative improvements and gradient score.
    Returns:
      (relative_improvement_own, relative_improvement_random, gradient_score, binary_indicator)
    """
    loss_improvement_own = loss_before_own - loss_after_own
    relative_improvement_own = (loss_improvement_own / loss_before_own) if loss_before_own > 0 else 0.0

    loss_improvement_random = loss_before_random - loss_after_random
    relative_improvement_random = (loss_improvement_random / loss_before_random) if loss_before_random > 0 else 0.0

    gradient_score = relative_improvement_random
    binary_indicator = 1 if relative_improvement_own > relative_improvement_random else -1
    return relative_improvement_own, relative_improvement_random, gradient_score, binary_indicator

async def evaluate_peer(uid, state_dict, sync_window, hparams, tokenizer,
                        config, model, transformer, compressor, xshapes, totalks,
                        device, lr, optimizer, scheduler, random_batches, random_pages):
    """
    Evaluates a peer's gradient and returns a dictionary containing evaluation metrics.
    """
    start_time = tplr.T()
    
    # load and prepare own evaluation batches.
    loader_own, _ = await R2DatasetLoader.get_loader(
        window=sync_window, hparams=hparams, tokenizer=tokenizer,
        data_type="own", seed=int(uid)
    )
    batches_own = [batch for batch in loader_own]
    del loader_own
    torch.cuda.empty_cache()
    
    model_own_eval = await asyncio.to_thread(copy.deepcopy, model)
    own_task = asyncio.to_thread(
        evaluate_loss_change,
        model_own_eval, batches_own, tokenizer, device,
        hparams.validator_sample_rate, state_dict, transformer, compressor,
        xshapes, totalks, scheduler
    )
    
    # RANDOM EVALUATION: use shared random batches.
    model_random_eval = await asyncio.to_thread(copy.deepcopy, model)
    random_task = asyncio.to_thread(
        evaluate_loss_change,
        model_random_eval, random_batches, tokenizer, device,
        hparams.validator_sample_rate, state_dict, transformer, compressor,
        xshapes, totalks, scheduler
    )
    
    (loss_before_own, loss_after_own,
     count_before_own, count_after_own,
     sampled_indices_own, total_batches_own) = await own_task
    logger.info(f"UID {uid}: Own data evaluation completed. Loss before: {loss_before_own}, after: {loss_after_own}")
    
    # Free own-eval copy and batches.
    del model_own_eval, own_task, batches_own
    torch.cuda.empty_cache()
    
    (loss_before_random, loss_after_random,
     count_before_random, count_after_random,
     sampled_indices_random, total_batches_random) = await random_task
    logger.info(f"UID {uid}: Random data evaluation completed. Loss before: {loss_before_random}, after: {loss_after_random}")
    
    # Free random-eval copy.
    del model_random_eval, random_task
    torch.cuda.empty_cache()
    
    (relative_improvement_own, relative_improvement_random,
     gradient_score, binary_indicator) = compute_improvement_metrics(
        loss_before_own, loss_after_own, loss_before_random, loss_after_random
    )
    
    # Load and verify pages.
    miner_pages, local_pages = await load_and_compare_pages(uid, sync_window, hparams, tokenizer, state_dict)
    
    total_time = tplr.T() - start_time
    logger.info(f"UID {uid}: Completed evaluation in {total_time} seconds")
    
    result = {
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
    }
    
    # Further cleanup.
    del loss_before_own, loss_after_own, loss_before_random, loss_after_random
    torch.cuda.empty_cache()
    
    return result

async def evaluate_peers_parallel(
    evaluation_uids, 
    comms, 
    sync_window, 
    hparams, 
    tokenizer, 
    config, 
    model, 
    transformer, 
    compressor, 
    xshapes, 
    totalks, 
    device, 
    lr, 
    optimizer, 
    scheduler,
    time_min,
    time_max
):
    """
    Evaluates multiple peers concurrently.
    Loads the "random" evaluation data only once for the current sync window.
    """
    # Load random evaluation data once.
    random_loader, random_pages = await R2DatasetLoader.get_loader(
        window=sync_window, hparams=hparams, tokenizer=tokenizer,
        data_type="random"
    )
    common_random_batches = [batch for batch in random_loader]
    del random_loader
    torch.cuda.empty_cache()
    
    async def evaluate_uid(uid):
        tplr.logger.info(f"Evaluating uid: {uid}")
        eval_result = await comms.get(
            uid=str(uid),
            window=sync_window,
            key='gradient',
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
                config,
                model,
                transformer,
                compressor,
                xshapes,
                totalks,
                device,
                lr,
                optimizer,
                scheduler,
                common_random_batches,
                random_pages
            )
            return uid, eval_payload
        else:
            tplr.logger.info(f"No gradient received from UID {uid}. Penalizing score.")
            return uid, None

    tasks = [asyncio.create_task(evaluate_uid(uid)) for uid in evaluation_uids]
    results = await asyncio.gather(*tasks)
    
    # Clean up common random data.
    del common_random_batches
    torch.cuda.empty_cache()
    
    return {uid: result for uid, result in results}

def compute_avg_loss(model, batches, sampled_indices, tokenizer, device):
    """
    Computes the average loss over selected batches.
    Uses torch.no_grad() to save GPU memory.
    """
    total_loss = 0.0
    n_batches_count = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(batches):
            if i in sampled_indices:
                input_ids = torch.tensor(batch, dtype=torch.long, device=device)
                labels = input_ids.clone()
                labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
                outputs = model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
                n_batches_count += 1
    return total_loss / n_batches_count if n_batches_count > 0 else 0.0, n_batches_count

def apply_gradient_update(model, state_dict, transformer, compressor, xshapes, totalks, device, lr):
    """
    Applies the peer gradient to the model.
    For each parameter, decode the compressed update and applies a sign-based update.
    """
    for n, p in model.named_parameters():
        idxs_key = n + "idxs"
        vals_key = n + "vals"
        if idxs_key in state_dict and vals_key in state_dict:
            idxs = state_dict[idxs_key].to(device)
            vals = state_dict[vals_key].to(device)
            # Note that p.data is used as the baseline tensor.
            grad = transformer.decode(
                compressor.decompress(
                    p.data,
                    idxs,
                    vals,
                    xshapes[n],
                    totalks[n],
                )
            ).to(device)
            # Apply the update using a sign-based step
            p.data.sub_(grad.sign(), alpha=lr)

async def parallel_evaluate_peer(
    eval_uid: int, 
    state_dict: dict,
    data_own_batches: list,
    data_random_batches: list,
    base_model,
    tokenizer,
    device: str,
    transformer,
    compressor,
    xshapes: dict,
    totalks: dict,
    hparams,
    lr: float
) -> tuple:
    """
    Isolated evaluation for one peer.
    
    Implements:
      1. Isolation: deep copy of the base model for both "own" and "random" evaluations.
      2. Strict ordering: compute loss-before, apply gradient update, then compute loss-after.
      3. Deterministic sampling: use fixed RNG seeds per UID to sample batches.
      
    Returns:
      (eval_uid, loss_before_own, loss_after_own, loss_before_random, loss_after_random, binary_indicator)
    """
    # Set up deterministic sampling using per-peer seeds.
    # rng_own = random.Random(eval_uid)
    # rng_random = random.Random(eval_uid + 1000)
    
    sample_size_own = max(1, int(len(data_own_batches) * hparams.validator_sample_rate))
    sampled_indices_own = sorted(random.sample(range(len(data_own_batches)), sample_size_own))

    sample_size_random = max(1, int(len(data_random_batches) * hparams.validator_sample_rate))
    sampled_indices_random = sorted(random.sample(range(len(data_random_batches)), sample_size_random))
    
    # --- Own Data Evaluation ---
    # Isolate model copy for own evaluation.
    model_own = copy.deepcopy(base_model).to(device)
    loss_before_own, _ = compute_avg_loss(model_own, data_own_batches, sampled_indices_own, tokenizer, device)
    
    # Apply the gradient update.
    apply_gradient_update(model_own, state_dict, transformer, compressor, xshapes, totalks, device, lr)
    
    loss_after_own, _ = compute_avg_loss(model_own, data_own_batches, sampled_indices_own, tokenizer, device)
    
    # --- Random Data Evaluation ---
    # Isolate model copy for random evaluation.
    model_random = copy.deepcopy(base_model).to(device)
    loss_before_random, _ = compute_avg_loss(model_random, data_random_batches, sampled_indices_random, tokenizer, device)
    
    apply_gradient_update(model_random, state_dict, transformer, compressor, xshapes, totalks, device, lr)
    
    loss_after_random, _ = compute_avg_loss(model_random, data_random_batches, sampled_indices_random, tokenizer, device)
    
    # Compute improvements.
    improvement_own = ((loss_before_own - loss_after_own) / loss_before_own) if loss_before_own > 0 else 0.0
    improvement_random = ((loss_before_random - loss_after_random) / loss_before_random) if loss_before_random > 0 else 0.0
    
    binary_indicator = 1 if improvement_own > improvement_random else -1
    
    # Clean up memory.
    del model_own, model_random
    torch.cuda.empty_cache()
    
    return (eval_uid, loss_before_own, loss_after_own, loss_before_random, loss_after_random, binary_indicator)

async def evaluate_all_peers(
    peer_gradients: dict,
    data_own_batches: list,
    data_random_batches: list,
    base_model,
    tokenizer,
    device: str,
    transformer,
    compressor,
    xshapes: dict,
    totalks: dict,
    hparams,
    lr: float
) -> dict:
    """
    Schedules parallel evaluations for all peers.
    
    Args:
      peer_gradients: Mapping {peer_uid: state_dict} of peer gradient states.
      data_own_batches: List of batched own evaluation data.
      data_random_batches: List of batched random evaluation data.
      (Other parameters passed to the evaluator functions.)
      
    Returns:
      Dictionary mapping uid -> {
          loss_before_own, loss_after_own, loss_before_random,
          loss_after_random, binary_indicator
      }
    """
    tasks = []
    for eval_uid, state_dict in peer_gradients.items():
        tasks.append(
            parallel_evaluate_peer(
                eval_uid,
                state_dict,
                data_own_batches,
                data_random_batches,
                base_model,
                tokenizer,
                device,
                transformer,
                compressor,
                xshapes,
                totalks,
                hparams,
                lr,
            )
        )
    results = await asyncio.gather(*tasks, return_exceptions=True)
    eval_results = {}
    for res in results:
        if isinstance(res, Exception):
            # Depending on your logging, you may want to record errors here.
            continue
        uid, loss_before_own, loss_after_own, loss_before_random, loss_after_random, binary_indicator = res
        eval_results[uid] = {
            "loss_before_own": loss_before_own,
            "loss_after_own": loss_after_own,
            "loss_before_random": loss_before_random,
            "loss_after_random": loss_after_random,
            "binary_indicator": binary_indicator,
        }
    return eval_results


def weighted_random_sample_no_replacement(candidates: list[str], weights: list[int], k: int) -> list[str]:
    tplr.logger.debug("Starting weighted random sampling.")
    tplr.logger.debug(f"Candidates: {candidates}")
    tplr.logger.debug(f"Weights: {weights}")
    tplr.logger.debug(f"Sample size (k): {k}")

    # Safety checks.
    if not candidates or not weights or k <= 0:
        tplr.logger.warning("Invalid input detected. Returning empty list.")
        return []

    # If the number of candidates is less than or equal to k, return all.
    if len(candidates) <= k:
        tplr.logger.info("Candidate count is within limit. Returning all candidates.")
        return candidates

    pool = list(zip(candidates, weights))
    total_w = float(sum(weights))
    # Fall back to unweighted random sampling if total weight is zero.
    if total_w <= 0:
        tplr.logger.warning("Total weight is zero, selecting random sample instead.")
        return random.sample(candidates, k)
    
    tplr.logger.debug(f"Initial total weight: {total_w}")
    selected = []

    for _ in range(k):
        if total_w <= 0 or len(pool) == 0:
            tplr.logger.info("No more items to sample. Stopping early.")
            break

        r = random.uniform(0.0, total_w)
        tplr.logger.debug(f"Random threshold: {r}")
        cumulative = 0.0
        for idx, (uid, w) in enumerate(pool):
            cumulative += w
            if cumulative >= r:
                selected.append(uid)
                tplr.logger.info(f"Selected candidate: {uid} with weight: {w}")
                total_w -= w
                pool.pop(idx)
                tplr.logger.debug(f"Updated total weight: {total_w}")
                break

    tplr.logger.debug(f"Final selected candidates: {selected}")
    return selected

# Returns the last value instead of averaging.
def safe_last(metric_list):
    """Return the last metric value in the list or 0.0 if empty.
    
    This replaces the averaging logic so we report a single miner's loss—as in the original behavior.
    """
    if not metric_list:
        tplr.logger.warning("Empty metric list!")
        return 0.0
    return metric_list[-1]