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

import time
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Tuple, cast

import torch
from torch import autocast

import tplr
from tplr.neurons import neuron_utils
from ..utils.logging import log_with_context

if TYPE_CHECKING:
    from .validator_core import ValidatorCore


async def get_gradients_for_window(
    validator_instance: "ValidatorCore", time_min: datetime, time_max: datetime
) -> Tuple[dict | None, dict]:
    """
    Try to load from aggregation server, if fails perform live gather.
    
    Args:
        validator_instance: ValidatorCore instance
        time_min: Minimum time for gather window
        time_max: Maximum time for gather window
        
    Returns:
        Tuple of (gradient_data, source_details_dict)
    """
    gather_start = tplr.T()
    skipped_uids: List[int] = []
    success_rate = 0.0
    
    # Try aggregation server first
    load_aggregation_start = tplr.T()
    aggregation_result = await validator_instance.comms.load_aggregation(validator_instance.sync_window)
    load_aggregation_end = tplr.T()
    
    if aggregation_result is not None:
        log_with_context(
            "info",
            f"{tplr.P(validator_instance.sync_window, load_aggregation_end - load_aggregation_start)} Loaded aggregation data.",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
        
        state_dict = cast(dict, aggregation_result.get("state_dict"))
        skipped_uids = cast(List[int], state_dict.get("skipped_uids", []))
        success_rate = cast(float, state_dict.get("success_rate", 0.0))
        
        source_details = {
            "from_aggregator": True,
            "skipped_uids": skipped_uids,
            "success_rate": success_rate,
            "gather_time": load_aggregation_end - load_aggregation_start,
        }
        
        return aggregation_result, source_details
    
    # Fall back to live gather
    gather_result = await validator_instance.comms.gather(
        my_uid=validator_instance.uid,
        uids=validator_instance.comms.peers,
        window=validator_instance.sync_window,
        key="gradient",
        timeout=45,
        device=validator_instance.config.device,
        local=False,
        totalks=validator_instance.totalks,
        time_min=time_min,
        time_max=time_max,
    )

    gather_time = tplr.T() - gather_start
    
    if gather_result is None:
        log_with_context(
            "error",
            "Failed to gather gradients from peers. Waiting for next window.",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
        
        source_details = {
            "from_aggregator": False,
            "skipped_uids": [],
            "success_rate": 0.0,
            "gather_time": gather_time,
        }
        
        return None, source_details
    
    skipped_uids = gather_result.skipped_uids
    success_rate = gather_result.success_rate
    
    source_details = {
        "from_aggregator": False,
        "skipped_uids": skipped_uids,
        "success_rate": success_rate,
        "gather_time": gather_time,
    }
    
    return gather_result, source_details


async def evaluate_miner_sync_status(
    validator_instance: "ValidatorCore", eval_uid: int, debug_dict_window: int
) -> Dict[str, bool | float | int | str]:
    """
    Fetch miner's debug dict, call tplr.neurons.compare_model_with_debug_dict, calculate sync score.
    
    Args:
        validator_instance: ValidatorCore instance
        eval_uid: UID of miner to evaluate
        debug_dict_window: Window to fetch debug dict from
        
    Returns:
        Dictionary of sync metrics including sync_score
    """
    # Fetch the miner's debug dictionary
    debug_result = await validator_instance.comms.get(
        uid=str(eval_uid),
        window=debug_dict_window,
        key="debug",
        local=False,
        stale_retention=10,
    )

    # Check if we got a valid result
    if debug_result is None:
        return {
            "success": False,
            "error": "Failed to retrieve debug dictionary",
            "sync_score": 0.0,
        }

    miner_debug_dict = cast(dict, debug_result[0])

    # Validate debug dictionary format
    if miner_debug_dict is None or not isinstance(miner_debug_dict, dict):
        return {
            "success": False,
            "error": "Invalid debug dictionary format",
            "sync_score": 0.0,
        }

    # Get current learning rate
    current_lr = validator_instance.scheduler.get_last_lr()[0]

    # Compare miner's debug dict with validator's model
    comparison_metrics = await neuron_utils.compare_model_with_debug_dict(
        model=validator_instance.model,
        debug_dict=miner_debug_dict,
        learning_rate=current_lr,
        index_range=(10, 12),
    )

    if not comparison_metrics["success"]:
        return {
            "success": False,
            "error": "Failed to compare model with debug dictionary",
            "sync_score": 0.0,
        }

    # Calculate sync score using the formula: score = (1-x/5)^2.5
    # where x is the average steps behind, capped at 5
    avg_steps_behind = comparison_metrics["avg_steps_behind"]
    x = min(avg_steps_behind, 5.0)
    sync_score = max(0.0, (1.0 - x / 5.0) ** 2.5)

    # Add the sync score to the metrics
    result = {**comparison_metrics, "sync_score": sync_score}

    return result


def evaluate_gradient_impact(
    validator_instance: "ValidatorCore",
    model_to_eval_on: torch.nn.Module,
    miner_gradient_state_dict: dict,
    data_loader,
    sample_rate: float = 1.0
) -> Tuple[float, float] | None:
    """
    Evaluate model with data_loader before applying miner_gradient_state_dict.
    Apply the (decompressed, signed) gradient.
    Evaluate after.
    
    Args:
        validator_instance: ValidatorCore instance
        model_to_eval_on: Model to evaluate on
        miner_gradient_state_dict: Gradient state dict from miner
        data_loader: Data loader for evaluation
        sample_rate: Sampling rate for batches
        
    Returns:
        Tuple of (loss_before, loss_after) or None on failure
    """
    try:
        # Sample indices for evaluation
        total_batches = len(data_loader)
        num_sampled = max(1, int(total_batches * sample_rate))
        sampled_indices = torch.randperm(total_batches)[:num_sampled].tolist()
        
        # Evaluate before applying gradient
        loss_before, n_batches_before = _evaluate_model_on_batches(
            validator_instance, model_to_eval_on, data_loader, sampled_indices
        )
        
        if n_batches_before == 0:
            return None
            
        # Apply the miner's gradient
        _apply_miner_gradient_to_model(
            validator_instance, model_to_eval_on, miner_gradient_state_dict
        )
        
        # Evaluate after applying gradient
        loss_after, n_batches_after = _evaluate_model_on_batches(
            validator_instance, model_to_eval_on, data_loader, sampled_indices
        )
        
        if n_batches_after == 0:
            return None
            
        avg_loss_before = loss_before / n_batches_before
        avg_loss_after = loss_after / n_batches_after
        
        return avg_loss_before, avg_loss_after
        
    except Exception as e:
        log_with_context(
            "error",
            f"Error evaluating gradient impact: {e}",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
        return None


def _evaluate_model_on_batches(
    validator_instance: "ValidatorCore",
    model: torch.nn.Module,
    data_loader,
    sampled_indices: List[int]
) -> Tuple[float, int]:
    """Helper function to evaluate model on sampled batches."""
    total_loss = 0.0
    n_batches = 0

    # TODO: Add validation for empty inputs
    if not data_loader or not sampled_indices:
        log_with_context(
            "warning",
            "Empty dataloader or sampled_indices provided to evaluate_model_on_batches",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
        return 0.0, 0

    with torch.no_grad():
        model.eval()
        with autocast(device_type=model.device.type, dtype=torch.bfloat16):
            for i, batch in enumerate(data_loader):
                if i not in sampled_indices:
                    continue
                    
                # TODO: Add validation for empty batches - handle arrays/tensors properly
                if batch is None or len(batch) == 0:
                    log_with_context(
                        "warning",
                        f"Empty batch at index {i}, skipping",
                        sync_window=validator_instance.sync_window,
                        current_window=validator_instance.current_window,
                    )
                    continue

                input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                labels = input_ids.clone()
                labels = torch.where(
                    labels == validator_instance.tokenizer.pad_token_id, -100, labels
                )
                outputs = model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
                n_batches += 1
                del input_ids, labels, outputs
                torch.cuda.empty_cache()

    return total_loss, n_batches


def _apply_miner_gradient_to_model(
    validator_instance: "ValidatorCore",
    model: torch.nn.Module,
    miner_gradient_state_dict: dict
) -> None:
    """Helper function to apply miner gradient to model."""
    for n, p in model.named_parameters():
        idxs_key = n + "idxs"
        vals_key = n + "vals"
        quant_key = n + "quant_params"

        idxs = miner_gradient_state_dict.get(idxs_key)
        vals = miner_gradient_state_dict.get(vals_key)
        quant_params = miner_gradient_state_dict.get(quant_key)
        
        if idxs is not None and vals is not None:
            if not isinstance(idxs, (list, tuple)):
                idxs = [idxs]
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
                
            new_grad = validator_instance.transformer.decode(
                validator_instance.compressor.batch_decompress(
                    p.to(validator_instance.config.device),
                    idxs,
                    vals,
                    validator_instance.xshapes[n],
                    validator_instance.totalks[n],
                    quant_params,
                )
            )

            if p.grad is None:
                p.grad = new_grad
            else:
                p.grad.copy_(new_grad)
            p.grad.sign_()


async def evaluate_individual_miner(
    validator_instance: "ValidatorCore",
    eval_uid: int,
    uid_specific_loader,
    common_random_loader,
    time_min: datetime,
    time_max: datetime
) -> Dict | None:
    """
    Get miner's gradient and evaluate it if valid.
    Call scoring functions to update scores for eval_uid.
    
    Args:
        validator_instance: ValidatorCore instance
        eval_uid: UID to evaluate
        uid_specific_loader: Data loader specific to this UID
        common_random_loader: Common random data loader
        time_min: Minimum time for gather window
        time_max: Maximum time for gather window
        
    Returns:
        Dictionary of loss metrics or None on major failure
    """
    # Fetch gradient data for evaluation
    eval_result = await validator_instance.comms.get(
        uid=str(eval_uid),
        window=validator_instance.sync_window,
        key="gradient",
        local=False,
        stale_retention=10,
        time_max=time_max,
        time_min=time_min,
    )

    if eval_result is not None:
        miner_gradient_state_dict = cast(dict, eval_result[0])
        
        # Validate gradient data
        if miner_gradient_state_dict is None or not isinstance(miner_gradient_state_dict, dict):
            # Import here to avoid circular import
            from . import scoring
            scoring.slash_peer_for_eval_failure(validator_instance, eval_uid, "invalid_gradient")
            return None

        # Create a copy of the model for evaluation
        model_copy = torch.nn.utils.deepcopy(validator_instance.model)
        
        try:
            # Evaluate gradient impact with UID-specific data
            uid_result = evaluate_gradient_impact(
                validator_instance, model_copy, miner_gradient_state_dict, uid_specific_loader
            )
            
            if uid_result is None:
                # Import here to avoid circular import
                from . import scoring
                scoring.slash_peer_for_eval_failure(validator_instance, eval_uid, "uid_eval_failed")
                return None
                
            loss_own_before, loss_own_after = uid_result
            
            # Reset model for second evaluation
            model_copy.load_state_dict(validator_instance.model.state_dict())
            
            # Evaluate gradient impact with common random data
            random_result = evaluate_gradient_impact(
                validator_instance, model_copy, miner_gradient_state_dict, common_random_loader
            )
            
            if random_result is None:
                # Import here to avoid circular import  
                from . import scoring
                scoring.slash_peer_for_eval_failure(validator_instance, eval_uid, "random_eval_failed")
                return None
                
            loss_random_before, loss_random_after = random_result
            
            # Update scores after successful evaluation
            # Import here to avoid circular import
            from . import scoring
            scoring.update_scores_after_evaluation(
                validator_instance, eval_uid, loss_own_before, loss_own_after, 
                loss_random_before, loss_random_after
            )
            
            # Evaluate sync status
            sync_result = await evaluate_miner_sync_status(validator_instance, eval_uid, validator_instance.sync_window - 1)
            sync_score = cast(float, sync_result.get("sync_score", 0.0))
            
            # Log sync score
            _log_sync_score(validator_instance, eval_uid, sync_result)
            
            # Store the sync score for this miner
            validator_instance.sync_scores[eval_uid] = sync_score
            
            # Add to evaluated UIDs
            validator_instance.evaluated_uids.add(eval_uid)
            
            return {
                "loss_own_before": loss_own_before,
                "loss_own_after": loss_own_after,
                "loss_random_before": loss_random_before,
                "loss_random_after": loss_random_after,
                "sync_score": sync_score,
            }
            
        finally:
            # Clean up model copy
            del model_copy
            torch.cuda.empty_cache()
    else:
        # No gradient received - slash the peer
        # Import here to avoid circular import
        from . import scoring
        scoring.slash_peer_for_eval_failure(validator_instance, eval_uid, "no_gradient")
        validator_instance.evaluated_uids.add(eval_uid)
        return None


def _log_sync_score(
    validator_instance: "ValidatorCore", eval_uid: int, sync_result: Dict[str, bool | float | int | str]
) -> None:
    """Helper function to log sync score metrics."""
    l2_norm = float(sync_result.get("l2_norm", 99.0))
    avg_l2_norm = float(sync_result.get("avg_l2_norm", 99.0))
    avg_abs_diff = float(sync_result.get("avg_abs_diff", 99.0))
    max_diff = float(sync_result.get("max_diff", 99.0))
    avg_steps_behind = float(sync_result.get("avg_steps_behind", 99.0))
    max_steps_behind = float(sync_result.get("max_steps_behind", 99.0))
    
    validator_instance.wandb.log(
        {
            f"validator/sync/l2_norm/{eval_uid}": l2_norm,
            f"validator/sync/avg_l2_norm/{eval_uid}": avg_l2_norm,
            f"validator/sync/avg_abs_diff/{eval_uid}": avg_abs_diff,
            f"validator/sync/sync_max_diff/{eval_uid}": max_diff,
            f"validator/sync/avg_steps_behind/{eval_uid}": avg_steps_behind,
            f"validator/sync/max_steps_behind/{eval_uid}": max_steps_behind,
        },
        step=validator_instance.global_step,
    )
    
    validator_instance.metrics_logger.log(
        measurement="validator_sync_score",
        tags={
            "uid": str(eval_uid),
            "window": int(validator_instance.sync_window),
            "global_step": int(validator_instance.global_step),
        },
        fields={
            "l2_norm": l2_norm,
            "avg_l2_norm": avg_l2_norm,
            "avg_abs_diff": avg_abs_diff,
            "max_diff": max_diff,
            "avg_steps_behind": avg_steps_behind,
            "max_steps_behind": max_steps_behind,
        },
        with_system_metrics=True,
        with_gpu_metrics=True,
    )


def apply_gradients_to_validator_model(validator_instance: "ValidatorCore", processed_gradient_data) -> None:
    """
    Apply the processed_gradient_data (from gather or aggregator) to the validator's own model 
    using its optimizer and scheduler.
    
    Args:
        validator_instance: ValidatorCore instance
        processed_gradient_data: Gradient data from gather or aggregation
    """
    # Apply weight decay just like in the miner
    lr = validator_instance.scheduler.get_last_lr()[0]
    for n, p in validator_instance.model.named_parameters():
        p.data.mul_(1.0 - lr * validator_instance.hparams.weight_decay)

    if hasattr(processed_gradient_data, 'get') and processed_gradient_data.get('state_dict'):
        # From aggregation server - apply aggregated gradients
        _apply_aggregated_gradients(validator_instance, processed_gradient_data)
    elif hasattr(processed_gradient_data, 'state_dict') and processed_gradient_data.state_dict is not None:
        # From live gather - apply gathered gradients
        _apply_gathered_gradients(validator_instance, processed_gradient_data)
    else:
        log_with_context(
            "warning",
            "No gradients to apply.",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
        validator_instance.scheduler.step()
        torch.cuda.empty_cache()


def _apply_aggregated_gradients(validator_instance: "ValidatorCore", aggregation_result: dict) -> bool:
    """Apply aggregated gradients from the aggregation server."""
    try:
        update_start = time.time()

        state_dict = aggregation_result.get("state_dict")
        if state_dict is None:
            log_with_context(
                "warning",
                "No state_dict found in aggregation result",
                sync_window=validator_instance.sync_window,
                current_window=validator_instance.current_window,
            )
            return False

        tensors_applied = 0

        for name, param in validator_instance.model.named_parameters():
            if name in state_dict:
                packed_tensor = state_dict[name]
                if packed_tensor is None:
                    continue

                # Unpack binary tensor
                unpacked_tensor = neuron_utils.unpack_binary_tensor(
                    packed_tensor, param.shape
                )

                # Move to appropriate device
                unpacked_tensor = unpacked_tensor.to(validator_instance.config.device)

                # Set as gradient for optimizer
                if param.grad is None:
                    param.grad = unpacked_tensor
                else:
                    param.grad.copy_(unpacked_tensor)

                tensors_applied += 1

        if tensors_applied > 0:
            log_with_context(
                "info",
                f"Set gradients for {tensors_applied} tensors in {time.time() - update_start:.2f}s",
                sync_window=validator_instance.sync_window,
                current_window=validator_instance.current_window,
            )

            # Update parameters with optimizer
            validator_instance.optimizer.step()
            validator_instance.scheduler.step()
            torch.cuda.empty_cache()

            log_with_context(
                "info",
                "Successfully applied aggregation",
                sync_window=validator_instance.sync_window,
                current_window=validator_instance.current_window,
            )
            return True
        else:
            log_with_context(
                "warning",
                "No tensors were applied during aggregation",
                sync_window=validator_instance.sync_window,
                current_window=validator_instance.current_window,
            )
            return False

    except Exception as e:
        log_with_context(
            "error",
            f"Error applying aggregated gradients: {e}",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
        return False


def _apply_gathered_gradients(validator_instance: "ValidatorCore", gather_result) -> None:
    """Apply gathered gradients from peers to the model."""
    for n, p in validator_instance.model.named_parameters():
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
                
            new_grad = validator_instance.transformer.decode(
                validator_instance.compressor.batch_decompress(
                    p.to(validator_instance.config.device),
                    idxs,
                    vals,
                    validator_instance.xshapes[n],
                    validator_instance.totalks[n],
                    quant_params,
                )
            )
            
            if p.grad is None:
                p.grad = new_grad
            else:
                p.grad.copy_(new_grad)
            p.grad.sign_()
        else:
            log_with_context(
                "info",
                f"Gradient data missing for parameter {n}, skipping.",
                sync_window=validator_instance.sync_window,
                current_window=validator_instance.current_window,
            )
            
    validator_instance.optimizer.step()
    validator_instance.scheduler.step()
    torch.cuda.empty_cache()


async def store_validator_debug_info(validator_instance: "ValidatorCore", source_details: dict) -> None:
    """
    Put validator-specific debug info (e.g., successful gather peers from source_details) to R2.
    
    Args:
        validator_instance: ValidatorCore instance
        source_details: Details from gather/aggregation operation
    """
    debug_dict = {}

    # Add model parameters debug info
    for name, param in validator_instance.model.named_parameters():
        if param is not None and param.numel() >= 2:
            debug_dict[name + "_debug"] = (
                param.flatten()[:2].detach().cpu().tolist()
            )

    # Add successful peers information from source details
    if len(source_details.get("skipped_uids", [])) > 0:
        debug_dict["successful_peers"] = sorted(
            list(set(validator_instance.comms.peers) - set(source_details["skipped_uids"]))
        )
        debug_dict["skipped_peers"] = sorted(list(source_details["skipped_uids"]))

    # Store debug dictionary
    import asyncio
    asyncio.create_task(
        validator_instance.comms.put(
            state_dict=debug_dict,
            uid=str(validator_instance.uid),
            window=validator_instance.sync_window,
            key="debug",
            local=False,
        )
    )
    
    log_with_context(
        "info",
        f"Stored debug values for window {validator_instance.current_window}",
        sync_window=validator_instance.sync_window,
        current_window=validator_instance.current_window,
    ) 