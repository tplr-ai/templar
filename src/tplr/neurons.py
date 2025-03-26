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


from typing import TYPE_CHECKING, TypeVar, cast

import torch
from torch._prims_common import DeviceLikeType
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)

from tplr.logging import logger

if TYPE_CHECKING:
    from neurons.miner import Miner
    from neurons.validator import Validator

NeuronT = TypeVar("NeuronT", "Miner", "Validator")


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
        # Ensure the gradient is on the same device as the parameter.
        grad = p.grad.to(p.device)
        # Ensure the momentum tensor is on the same device.
        if miner.momentum[n].device != p.device:
            miner.momentum[n] = miner.momentum[n].to(p.device)
        miner.momentum[n].add_(grad, alpha=lr)
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


async def catchup_with_aggregation_server(
    instance: NeuronT, checkpoint_current_window: int
):
    """
    Catch up the model by applying aggregated gradients from the aggregation server
    and verifying against the validator's debug dict.
    """
    logger.info("Starting catchup with aggregation server...")

    # Start from the checkpoint window and continue until we reach the current window
    checkpoint_window = checkpoint_current_window + 1
    target_window = instance.current_window

    logger.info(
        f"Catching up from window {checkpoint_window} to current window {target_window}"
    )

    weight_decay = instance.hparams.weight_decay

    # Apply aggregation for each step, checking for current window changes
    current_step = checkpoint_window
    while current_step <= target_window:
        # Check if current_window has changed during processing
        if instance.current_window > target_window:
            target_window = instance.current_window
            logger.info(
                f"Current window advanced during catchup, new target: {target_window}"
            )

        logger.info(
            f"\nProcessing catchup for window {current_step} (Target: {target_window})"
        )

        # Load aggregation for current window - pass version explicitly
        agg_data = await instance.comms.load_aggregation(window=current_step)
        if not agg_data:
            logger.warning(
                f"No aggregation data found for window {current_step}, skipping"
            )
            current_step += 1
            continue

        processed_agg_data = process_loaded_data(instance.model, agg_data)

        # Get learning rate for this step
        lr = get_lr_at_step(
            instance.hparams.learning_rate, current_step - instance.start_window
        )
        logger.info(f"Using learning rate: {lr:.6f} for catchup step")

        # Get debug dictionary for verification
        debug_dict = await instance.comms.get_debug_dict(current_step - 1)

        # Apply aggregation to model
        logger.info(f"Applying aggregation to model for window {current_step}...")
        with torch.no_grad():
            for name, param in instance.model.named_parameters():
                if name in processed_agg_data["tensors"]:
                    # Move aggregation tensor to device
                    agg_tensor = processed_agg_data["tensors"][name].to(
                        instance.config.device  # type: ignore
                    )

                    # Apply weight decay to parameter
                    param.data.mul_(1.0 - lr * weight_decay)

                    # Apply gradient update with learning rate
                    param.data.add_(agg_tensor, alpha=-lr)

        logger.info(f"Successfully applied aggregation for window {current_step}")

        # Calculate L2 norm of difference between model and debug values after this step
        if isinstance(debug_dict, dict) and "state_dict" in debug_dict:
            debug_state_dict = debug_dict["state_dict"]
            total_squared_diff = 0.0
            param_count = 0
            abs_diff = 0

            for name, param in instance.model.named_parameters():
                # Check if there's a corresponding debug entry
                debug_key = name + "_debug"
                if debug_key in debug_state_dict:
                    # Calculate L2 norm for this parameter
                    param_data = param.data.cpu().flatten()[
                        :2
                    ]  # Take only first two values
                    debug_data = torch.tensor(
                        cast(float, debug_state_dict[debug_key])
                    ).cpu()
                    squared_diff = torch.sum((param_data - debug_data) ** 2).item()
                    total_squared_diff += squared_diff
                    abs_diff += (
                        torch.abs(torch.tensor(param_data - debug_data)).mean().item()
                    )
                    param_count += param_data.numel()

            # Final L2 norm across all parameters
            final_l2_norm = torch.sqrt(torch.tensor(total_squared_diff)).item()
            logger.info(
                f"Window {current_step} - L2 norm difference between model and debug values: {final_l2_norm}"
            )
            logger.info(
                f"Window {current_step} - Average L2 norm per parameter: {final_l2_norm / param_count if param_count > 0 else 0}"
            )
            logger.info(
                f"Window {current_step} - Average absolute difference per parameter: {abs_diff / param_count / lr if param_count > 0 else 0}"
            )
        else:
            logger.info(
                f"No valid debug dictionary available for window {current_step - 1} - cannot compute L2 norm"
            )

        # Update global step and move to next window
        instance.global_step = current_step - instance.start_window
        current_step += 1

        # Step optimizer and scheduler to match the current window
        instance.optimizer.step()
        instance.scheduler.step()

    # Update global step after catchup
    instance.global_step = target_window - instance.start_window
    logger.info(f"Catchup complete. Global step updated to {instance.global_step}")


def process_loaded_data(model: torch.nn.Module, compressed_data: dict):
    """
    Unpack the compressed tensor data from the aggregation server.

    Args:
        compressed_data: The compressed tensor data

    Returns:
        Dictionary with unpacked tensors
    """
    state_dict = compressed_data.get("state_dict")
    result = {
        "timestamp": state_dict.get("timestamp", None),
        "window": state_dict.get("window", None),
        "version": state_dict.get("version", None),
        "tensors": {},
    }

    for name, param in model.named_parameters():
        if name in state_dict:
            original_shape = param.shape
            # Use unpack_binary_tensor from the sample, but in our context
            unpacked = unpack_binary_tensor(state_dict[name], original_shape)
            result["tensors"][name] = unpacked
            logger.debug(f"Unpacked tensor {name} with shape {original_shape}")

    logger.info(f"Successfully unpacked {len(result['tensors'])} tensors")
    return result


def unpack_binary_tensor(packed_tensor: torch.Tensor, original_shape: torch.Size):
    """
    Unpack a 1-bit representation tensor back to ±1 values.

    Args:
        packed_tensor: The packed binary tensor
        original_shape: The original shape of the tensor

    Returns:
        Unpacked tensor with original shape
    """
    total_elements = int(torch.prod(torch.tensor(original_shape)).item())

    # Create a flat tensor to hold the unpacked values
    unpacked = torch.zeros(total_elements, dtype=torch.float32)

    for i in range(8):
        mask = 1 << i
        bits = (packed_tensor & mask) >> i
        # Convert 0/1 to -1/+1
        unpacked[i::8] = (bits.float() * 2) - 1

    return unpacked.reshape(original_shape)


# Function to pack signed weights into 1-bit representation
def pack_binary_tensor(tensor: torch.Tensor, device: DeviceLikeType):
    """Pack a tensor of +1/-1 values into a compact binary representation."""
    tensor = (tensor > 0).to(torch.uint8)  # Convert +1 to 1, -1 to 0
    tensor = tensor.view(-1)  # Flatten
    packed_tensor = torch.zeros(
        (tensor.shape[0] + 7) // 8, dtype=torch.uint8, device=device
    )

    for i in range(8):
        packed_tensor |= tensor[i::8] << i  # Pack 8 values per byte

    return packed_tensor


def get_lr_at_step(lr, global_step):
    """
    Get the learning rate at a specific global step.

    Args:
        global_step: The global step to calculate LR for

    Returns:
        Learning rate value
    """
    # Create temporary objects to avoid modifying originals
    temp_optimizer = SGD([torch.tensor([1.0])], lr=lr)

    # Recreate the schedulers to match our existing scheduler configuration
    temp_warmup_scheduler = LinearLR(
        temp_optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=250,
    )
    temp_cosine_scheduler = CosineAnnealingWarmRestarts(
        temp_optimizer,
        T_0=10000,
        T_mult=2,
        eta_min=lr * 0.1,
    )
    temp_scheduler = SequentialLR(
        temp_optimizer,
        schedulers=[temp_warmup_scheduler, temp_cosine_scheduler],
        milestones=[250],
    )

    # Step to the target global step
    for _ in range(global_step):
        temp_scheduler.step()

    # Return the current learning rate
    return temp_optimizer.param_groups[0]["lr"]
