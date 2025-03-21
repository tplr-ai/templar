import torch
import random
import copy
import asyncio
from tplr.logging import logger
import tplr
from .r2_dataset import R2DatasetLoader

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
            torch.cuda.empty_cache()  # TODO: Consider reducing frequency of empty_cache calls.
        else:
            logger.info(f"Gradient data missing for parameter {n}, skipping.")
    return model

def compute_average_loss(model, batches, tokenizer, device, sample_rate):
    """
    Computes the average loss over selected batches based on the sample rate.
    """
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
            # Use autocast for mixed precision during the forward pass.
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
    """
    Evaluates loss before and after applying a compressed gradient update.
    """
    model.eval()
    loss_before, count_before, sampled_indices, total_batches = compute_average_loss(
        model, batches, tokenizer, device, sample_rate
    )
    logger.info(f"Loss before gradient: {loss_before} on {count_before}/{total_batches} batches")
    
    # Get current learning rate from scheduler.
    current_lr = scheduler.get_last_lr()[0]
    
    # Apply gradient update.
    model_after = apply_compressed_gradient(model, state_dict, transformer, compressor, xshapes, totalks, device, current_lr)
    loss_after, count_after, _, _ = compute_average_loss(
        model_after, batches, tokenizer, device, sample_rate
    )
    logger.info(f"Loss after gradient: {loss_after}")
    
    return loss_before, loss_after, count_before, count_after, sampled_indices, total_batches

def compute_improvement_metrics(loss_before_own, loss_after_own, loss_before_random, loss_after_random):
    """
    Computes loss improvements and relative improvements.
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

async def load_and_compare_pages(uid, sync_window, hparams, tokenizer, state_dict):
    """
    Loads local evaluation pages and compares them with miner-provided metadata.
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

async def evaluate_peer(uid, state_dict, sync_window, hparams, tokenizer,
                        config, model, transformer, compressor, xshapes, totalks,
                        device, lr, optimizer, scheduler, random_batches, random_pages):
    """
    Evaluates a peer's gradient and returns a dictionary containing evaluation metrics.
    """
    start_time = tplr.T()
    
    # Load and prepare own evaluation data.
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
    
    del model_own_eval, own_task, batches_own
    torch.cuda.empty_cache()
    
    (loss_before_random, loss_after_random,
     count_before_random, count_after_random,
     sampled_indices_random, total_batches_random) = await random_task
    logger.info(f"UID {uid}: Random data evaluation completed. Loss before: {loss_before_random}, after: {loss_after_random}")
    
    del model_random_eval, random_task
    torch.cuda.empty_cache()
    
    (relative_improvement_own, relative_improvement_random,
     gradient_score, binary_indicator) = compute_improvement_metrics(
        loss_before_own, loss_after_own, loss_before_random, loss_after_random
    )
    
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
    random_loader, random_pages = await R2DatasetLoader.get_loader(
        window=sync_window, hparams=hparams, tokenizer=tokenizer,
        data_type="random"
    )
    random_batches = [batch for batch in random_loader]
    del random_loader
    torch.cuda.empty_cache()
    
    async def evaluate_uid(uid, random_batches=random_batches, random_pages=random_pages):
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
                random_batches,
                random_pages
            )
            return uid, eval_payload
        else:
            tplr.logger.info(f"No gradient received from UID {uid}. Penalizing score.")
            return uid, None

    tasks = [asyncio.create_task(evaluate_uid(uid)) for uid in evaluation_uids]
    results = await asyncio.gather(*tasks)
    
    del random_batches
    torch.cuda.empty_cache()
    
    return {uid: result for uid, result in results}

def compute_avg_loss(model, batches, sampled_indices, tokenizer, device):
    """
    Computes the average loss over selected batches.
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
    For each parameter, decode the compressed update and apply a sign-based step.
    """
    for n, p in model.named_parameters():
        idxs_key = n + "idxs"
        vals_key = n + "vals"
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
        else:
            logger.info(f"Gradient data missing for parameter {n}, skipping.")
    return

def weighted_random_sample_no_replacement(candidates: list[str], weights: list[int], k: int) -> list[str]:
    tplr.logger.debug("Starting weighted random sampling.")
    tplr.logger.debug(f"Candidates: {candidates}")
    tplr.logger.debug(f"Weights: {weights}")
    tplr.logger.debug(f"Sample size (k): {k}")

    if not candidates or not weights or k <= 0:
        tplr.logger.warning("Invalid input detected. Returning empty list.")
        return []

    if len(candidates) <= k:
        tplr.logger.info("Candidate count is within limit. Returning all candidates.")
        return candidates

    pool = list(zip(candidates, weights))
    total_w = float(sum(weights))
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

def safe_last(metric_list):
    if not metric_list:
        tplr.logger.warning("Empty metric list!")
        return 0.0
    return metric_list[-1]