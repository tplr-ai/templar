import torch
import random
from tplr.logging import logger
import tplr
from .r2_dataset import R2DatasetLoader
import asyncio
import random
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
            grad = transformer.decode(
                compressor.decompress(p.to(device), idxs, vals, xshapes[n], totalks[n])
            ).to(device)
            p.data.sub_(grad.sign(), alpha=lr)
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
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
            count += 1
            del input_ids, labels, outputs
            torch.cuda.empty_cache()
    avg_loss = total_loss / count if count else 0.0
    return avg_loss, count, sampled_indices, total_batches

def evaluate_loss_change(model, batches, tokenizer, device, sample_rate, 
                         state_dict, transformer, compressor, xshapes, totalks, lr, optimizer):
    """
    Evaluates the model loss before and after applying the gradient from state_dict.
    Returns a tuple:
      (loss_before, loss_after, count_before, count_after, sampled_indices, total_batches)
    """
    optimizer.zero_grad()
    model.eval()
    loss_before, count_before, sampled_indices, total_batches = compute_average_loss(
        model, batches, tokenizer, device, sample_rate
    )
    logger.info(f"Loss before gradient: {loss_before} on {count_before}/{total_batches} batches")
    
    optimizer.zero_grad()
    # Apply the compressed gradient
    model_after = apply_compressed_gradient(model, state_dict, transformer, compressor, xshapes, totalks, device, lr)
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

    gradient_score = (loss_improvement_own / loss_before_own) if loss_before_own > 0 else 0.0
    binary_indicator = 1 if relative_improvement_own > relative_improvement_random else -1
    return relative_improvement_own, relative_improvement_random, gradient_score, binary_indicator

async def evaluate_peer(uid, state_dict, sync_window, hparams, tokenizer,
                        config, model, transformer, compressor, xshapes, totalks,
                        device, lr, optimizer, scheduler, random_batches, random_pages):
    """
    Evaluates a peer's gradient by comparing loss improvements on "own" and shared "random" evaluation data.
    Uses:
      - evaluate_loss_change: computes loss before/after gradient application.
      - compute_improvement_metrics: computes relative improvements and gradient score.
      - load_and_compare_pages: verifies data consistency.
    Returns:
      A dictionary with evaluation metrics.
    """
    start_time = tplr.T()

    ## OWN EVALUATION (using uid-specific seed)
    loader_own, _ = await R2DatasetLoader.get_loader(
        window=sync_window, hparams=hparams, tokenizer=tokenizer,
        data_type="own", seed=uid
    )
    batches_own = [batch for batch in loader_own]
    model_own_eval = await asyncio.to_thread(copy.deepcopy, model)
    own_task = asyncio.to_thread(
        evaluate_loss_change,
        model_own_eval, batches_own, tokenizer, device,
        hparams.validator_sample_rate, state_dict, transformer, compressor,
        xshapes, totalks, lr, optimizer
    )

    ## RANDOM EVALUATION (shared among all UIDs)
    model_random_eval = await asyncio.to_thread(copy.deepcopy, model)
    random_task = asyncio.to_thread(
        evaluate_loss_change,
        model_random_eval, random_batches, tokenizer, device,
        hparams.validator_sample_rate, state_dict, transformer, compressor,
        xshapes, totalks, lr, optimizer
    )

    (loss_before_own, loss_after_own,
     count_before_own, count_after_own,
     sampled_indices_own, total_batches_own) = await own_task
    logger.info(f"UID {uid}: Own data evaluation completed. Loss before: {loss_before_own}, after: {loss_after_own}")

    (loss_before_random, loss_after_random,
     count_before_random, count_after_random,
     sampled_indices_random, total_batches_random) = await random_task
    logger.info(f"UID {uid}: Random data evaluation completed. Loss before: {loss_before_random}, after: {loss_after_random}")

    (relative_improvement_own, relative_improvement_random,
     gradient_score, binary_indicator) = compute_improvement_metrics(
         loss_before_own, loss_after_own, loss_before_random, loss_after_random
    )
    logger.info(f"UID {uid}: Gradient score: {gradient_score}, Binary indicator: {binary_indicator}")

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
    scheduler
):
    """
    Evaluates multiple peers concurrently.
    Loads the "random" evaluation data only once for the current sync window,
    and then passes it to each evaluation of the given UIDs.
    
    Returns:
        dict: Mapping from uid -> evaluation result (or None if no gradient received).
    """
    # Load random evaluation data once.
    random_loader, random_pages = await R2DatasetLoader.get_loader(
        window=sync_window, hparams=hparams, tokenizer=tokenizer,
        data_type="random"
    )
    common_random_batches = [batch for batch in random_loader]

    async def evaluate_uid(uid):
        tplr.logger.info(f"Evaluating uid: {uid}")
        eval_result = await comms.get(
            uid=str(uid),
            window=sync_window,
            key='gradient',
            timeout=30,
            local=False,
            stale_retention=10
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
    return {uid: result for uid, result in results}