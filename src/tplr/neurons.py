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


def validate_compressed_gradients(state_dict, totalks, allowed_topk=None, device="cpu"):
    """
    Validates compressed gradients received from peers.
    
    Args:
        state_dict (dict): Dictionary containing compressed gradient data
        totalks (dict): Dictionary of parameter total sizes
        allowed_topk (float, optional): Maximum allowed compression percentage
        device (str): Device to move tensors to for validation
        
    Returns:
        bool: Whether the compressed gradients are valid
        str: Error message if invalid, None otherwise
    """
    # Validate tensor indices and values
    for param_name, tensor in state_dict.items():
        # Check compressed indices
        if param_name.endswith("idxs"):
            base_name = param_name[:-4]
            totalk = totalks.get(base_name)
            if totalk is None:
                return False, f"Missing totalk for parameter {base_name}"
                
            if tensor is None or not isinstance(tensor, torch.Tensor):
                return False, f"Invalid indices for {param_name}: expected tensor, got {type(tensor)}"
                
            if tensor.numel() == 0:
                continue  # Empty tensor is valid
                
            # Check that indices are within bounds
            tensor_to_check = tensor.to(device)
            if torch.any(tensor_to_check < 0) or torch.any(tensor_to_check >= totalk):
                return False, (f"Indices out of bounds for {param_name}: min={tensor_to_check.min().item()}, "
                              f"max={tensor_to_check.max().item()}, totalk={totalk}")
                
            # Check if topk is reasonable
            if allowed_topk is not None:
                max_allowed = int(totalk * (allowed_topk / 100.0))
                if tensor_to_check.numel() > max_allowed:
                    return False, (f"Too many indices for {param_name}: got {tensor_to_check.numel()}, "
                                  f"max allowed is {max_allowed} ({allowed_topk}% of {totalk})")
                    
        # Check tensor values
        elif param_name.endswith("vals"):
            tensor_to_check = tensor.to(device)
            try:
                if torch.isnan(tensor_to_check).any() or torch.isinf(tensor_to_check).any():
                    return False, f"Values contain NaN or Inf for parameter {param_name}"
            except Exception as e:
                return False, f"Values check failed for parameter {param_name}: {e}"
                
    return True, None
