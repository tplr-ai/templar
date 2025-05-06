"""
Distributed training utilities for Templar.
Handles PyTorch DDP initialization, synchronization, and gradient operations.
"""

import os
import socket
import torch
import torch.distributed as dist
import datetime

import tplr

logger = tplr.logger

def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def ddp_init(local_rank: int, world_size: int, timeout_min: int = 5):
    """
    Initialize the distributed process group for DDP.
    
    Args:
        local_rank: Local rank of this process
        world_size: Total number of processes
        timeout_min: Timeout in minutes for operations
    """
    if dist.is_initialized():
        logger.warning("Process group already initialized, skipping initialization")
        return
    
    # make sure the vars required by `init_method="env://"` are present
    os.environ.setdefault("RANK", str(local_rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    
    # Set the device for this process
    torch.cuda.set_device(local_rank)
    
    # Initialize the process group
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=local_rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=timeout_min),
        )
        logger.info(f"Initialized process group: rank={dist.get_rank()}, world_size={dist.get_world_size()}")
    except Exception as e:
        logger.error(f"Failed to initialize process group: {e}")
        raise

def is_rank0():
    """Check if this process is rank 0 in the distributed group."""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_rank():
    """Get the rank of this process, or 0 if not initialized."""
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size():
    """Get the world size, or 1 if not initialized."""
    return dist.get_world_size() if dist.is_initialized() else 1

def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()

def all_reduce_params(params, op=dist.ReduceOp.AVG):
    """
    Perform a coalesced all-reduce operation on a list of parameters.
    
    Args:
        params: List of parameter tensors (typically gradients)
        op: Reduction operation (default: AVG)
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return  # No-op for single process
    
    if not params:
        return  # Empty list
    
    # Filter out None gradients
    params = [p for p in params if p is not None]
    
    # Group parameters into buckets (max 25MB per bucket as a heuristic)
    max_bucket_size = 25 * 1024 * 1024  # 25MB in bytes
    buckets = []
    current_bucket = []
    current_size = 0
    
    for param in params:
        param_size = param.numel() * param.element_size()
        if current_size + param_size > max_bucket_size and current_bucket:
            buckets.append(current_bucket)
            current_bucket = [param]
            current_size = param_size
        else:
            current_bucket.append(param)
            current_size += param_size
    
    if current_bucket:
        buckets.append(current_bucket)
    
    # All-reduce each bucket
    for bucket in buckets:
        dist.all_reduce_coalesced(bucket, op)

def flatten_grads(grads):
    """
    Flatten a list of gradient tensors into a single 1D tensor.
    
    Args:
        grads: List of gradient tensors
    
    Returns:
        flat_grads: Flattened 1D tensor
        shapes: List of original shapes
        dtypes: List of original dtypes
    """
    shapes = [g.shape for g in grads]
    dtypes = [g.dtype for g in grads]
    flat_grads = torch.cat([g.reshape(-1) for g in grads])
    return flat_grads, shapes, dtypes

def unflatten_grads(flat_grads, shapes, dtypes):
    """
    Unflatten a 1D tensor back into a list of gradient tensors.
    
    Args:
        flat_grads: Flattened 1D tensor
        shapes: List of original shapes
        dtypes: List of original dtypes
    
    Returns:
        grads: List of gradient tensors with original shapes
    """
    grads = []
    offset = 0
    for shape, dtype in zip(shapes, dtypes):
        numel = torch.prod(torch.tensor(shape)).item()
        grad = flat_grads[offset:offset+numel].reshape(shape).to(dtype)
        grads.append(grad)
        offset += numel
    return grads

def broadcast_object(obj, src=0):
    """
    Broadcast a Python object from src rank to all other ranks.
    
    Args:
        obj: Python object to broadcast (must be picklable)
        src: Source rank
    
    Returns:
        The broadcast object
    """
    if not dist.is_initialized():
        return obj
    
    container = [obj if get_rank() == src else None]
    dist.broadcast_object_list(container, src=src)
    return container[0]

def cleanup():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        logger.info(f"Destroying process group for rank {dist.get_rank()}")
        dist.destroy_process_group()

def rank_world() -> tuple[int, int]:
    """Return (rank, world_size)."""
    return get_rank(), get_world_size() 