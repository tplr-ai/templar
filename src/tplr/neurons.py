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


from torch._prims_common import DeviceLikeType
from tplr.logging import logger
import torch


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


def process_loaded_data(model: torch.nn.Module, compressed_data: dict):
    """
    Unpack the compressed tensor data from the aggregation server.

    Args:
        compressed_data: The compressed tensor data

    Returns:
        Dictionary with unpacked tensors
    """
    result = {
        "timestamp": compressed_data.get("timestamp", None),
        "window": compressed_data.get("window", None),
        "version": compressed_data.get("version", None),
        "tensors": {},
    }

    for name, param in model.named_parameters():
        if name in compressed_data:
            original_shape = param.shape
            # Use unpack_binary_tensor from the sample, but in our context
            unpacked = unpack_binary_tensor(compressed_data[name], original_shape)
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
