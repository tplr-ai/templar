from tplr.logging import logger
import copy
import torch
import random
import tplr


def prepare_gradient_dict(miner, pages, step_window):
    """
    Prepares the gradient dictionary for sharing by compressing the
    momentum for each parameter and attaching metadata.

    Args:
        miner (Miner): Instance of Miner containing model, scheduler, momentum, compressor, transformer and hparams.
        pages (list): The pages information used for training data.
        step_window (int): The current window number.

    Returns:
        tuple: (gradient, xshapes, totalks, transmitted) where:
            gradient (dict): Contains keys for each parameter's compressed gradients and metadata.
            xshapes (dict): The computed shapes for each parameter.
            totalks (dict): Total length information for each parameter.
            transmitted (dict): The estimated transmitted gradients per parameter.
    """
    gradient = {}
    xshapes = {}
    totalks = {}
    transmitted = {}
    lr = miner.scheduler.get_last_lr()[0]

    for n, p in miner.model.named_parameters():
        # Apply weight decay.
        p.data.mul_(1.0 - lr * miner.hparams.weight_decay)
        # Apply momentum decay.
        miner.momentum[n].mul_(miner.hparams.momentum_decay)
        # Update momentum with the current gradient scaled by lr.
        miner.momentum[n].add_(p.grad, alpha=lr)
        # Compress momentum via DCT-based compression.
        idxs, vals, xshape, totalk = miner.compressor.compress(
            miner.transformer.encode(miner.momentum[n]), miner.hparams.topk_compression
        )
        # Estimate the transmitted gradient via decompression.
        transmit_grad = miner.transformer.decode(
            miner.compressor.decompress(p, idxs, vals, xshape, totalk)
        )
        # Subtract the transmitted gradient from momentum.
        miner.momentum[n].sub_(transmit_grad)
        # Save compressed gradient information.
        gradient[n + "idxs"] = idxs
        gradient[n + "vals"] = vals
        xshapes[n] = xshape
        totalks[n] = totalk
        transmitted[n] = transmit_grad

    # Attach metadata for pages info and window.
    gradient["metadata"] = {"pages_info": pages, "window": step_window}
    logger.info(f"Attached metadata to gradient: {gradient['metadata']}")

    return gradient, xshapes, totalks, transmitted

async def load_evaluation_loader(data_type: str, sync_window: int, hparams, tokenizer):
    """
    Loads evaluation data using the R2DatasetLoader.
    
    Args:
        data_type (str): 'own' or 'random'; use a fixed seed for own data.
        sync_window (int): the current sync/evaluation window.
        hparams: hyperparameters containing pages_per_window, batch_size and sequence_length.
        tokenizer: the tokenizer to be used.
    
    Returns:
        tuple: (loader, pages_info)
    """
    seed_val = 42 if data_type == "own" else random.randint(0, 10000)
    pages = await tplr.r2_dataset.R2DatasetLoader.next_pages(
        offset=sync_window,
        n_pages=hparams.pages_per_window,
        seed=seed_val
    )
    loader = await tplr.r2_dataset.R2DatasetLoader.create(
        batch_size=hparams.batch_size,
        sequence_length=hparams.sequence_length,
        pages_info=pages,
        tokenizer=tokenizer
    )
    return loader, pages

def evaluate_model_loss(model, loader, tokenizer, device):
    """
    Evaluates a model on a given data loader and returns the average loss.
    
    Args:
        model (torch.nn.Module): Model to evaluate (should be in eval mode).
        loader (iterable): Evaluation data loader.
        tokenizer: Tokenizer to determine the pad token.
        device (str): Device for computation.
    
    Returns:
        tuple: (average_loss, num_batches)
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
    Applies the compressed gradient extracted from state_dict to the model parameters.
    
    Args:
        model (torch.nn.Module): The model to update.
        state_dict (dict): Contains compressed gradient data with keys {param_name + 'idxs', param_name + 'vals'}.
        transformer: The DCT-based transformer.
        compressor: The compressor instance to decompress.
        xshapes (dict): Precomputed shapes for each parameter.
        totalks (dict): Total length info for each parameter.
        device (str): Device for computation.
        lr (float): Learning rate to use as alpha.
        
    Returns:
        torch.nn.Module: Updated model with gradient applied.
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

async def evaluate_peer(uid, state_dict, sync_window, hparams, tokenizer, config, model, transformer, compressor, xshapes, totalks, device, lr):
    """
    Evaluates one peer's gradient on both its "own" data and on random data.
    This async method encapsulates:
      - Loading evaluation data (own/random)
      - Computing loss before gradient application
      - Applying the gradient (using DCT decode/decompress)
      - Computing loss after application
      - Returning computed metrics, e.g. gradient score and binary indicator.
    
    Args:
        uid (int): Evaluation UID.
        state_dict (dict): Compressed gradient dictionary from the peer.
        sync_window (int): Current sync/evaluation window.
        hparams: Hyperparameters (contains learning rate, pages_per_window, etc.).
        tokenizer: The tokenizer for converting data to tensors.
        config: (Optional) Additional config if needed.
        model (torch.nn.Module): Baseline model.
        transformer, compressor: Compression utilities.
        xshapes (dict): xshapes computed during initialization.
        totalks (dict): Total length info for each parameter.
        device (str): Device to use.
        lr (float): The learning rate to use when applying gradients.
    
    Returns:
        dict: A dictionary containing evaluation results.
    """
    # Evaluate on own data
    model_own = copy.deepcopy(model)
    loader_own, pages_own = await load_evaluation_loader("own", sync_window, hparams, tokenizer)
    loss_before_own, _ = evaluate_model_loss(model_own, loader_own, tokenizer, device)
    model_own = apply_compressed_gradient(model_own, state_dict, transformer, compressor, xshapes, totalks, device, lr)
    loss_after_own, _ = evaluate_model_loss(model_own, loader_own, tokenizer, device)

    # Evaluate on random data
    model_rand = copy.deepcopy(model)
    loader_rand, pages_rand = await load_evaluation_loader("random", sync_window, hparams, tokenizer)
    loss_before_rand, _ = evaluate_model_loss(model_rand, loader_rand, tokenizer, device)
    model_rand = apply_compressed_gradient(model_rand, state_dict, transformer, compressor, xshapes, totalks, device, lr)
    loss_after_rand, _ = evaluate_model_loss(model_rand, loader_rand, tokenizer, device)

    # Compute improvements and scores
    improvement_own = loss_before_own - loss_after_own
    improvement_rand = loss_before_rand - loss_after_rand
    gradient_score = (improvement_own / loss_before_own) if loss_before_own > 0 else 0.0
    binary_indicator = 1 if (improvement_own / loss_before_own) > (improvement_rand / loss_before_rand) else -1

    return {
        "uid": uid,
        "loss_before_own": loss_before_own,
        "loss_after_own": loss_after_own,
        "loss_before_rand": loss_before_rand,
        "loss_after_rand": loss_after_rand,
        "gradient_score": gradient_score,
        "binary_indicator": binary_indicator,
        "pages_own": pages_own,
        "pages_rand": pages_rand,
    }
