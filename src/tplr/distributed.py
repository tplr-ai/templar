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

"""
Distributed training utilities for multi-GPU training.
"""

import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor as DT

import tplr


class DistributedHelper:
    """Helper class for distributed training operations."""

    def __init__(self):
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.is_master = self.rank == 0
        self.device = None

    def init_process_group(
        self, backend: str = "nccl", timeout_minutes: int = 45
    ) -> None:
        """Initialize the distributed process group."""
        if not dist.is_initialized() and self.world_size > 1:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(minutes=timeout_minutes),
                rank=self.rank,
                world_size=self.world_size,
            )

            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = torch.device("cpu")

            tplr.logger.info(
                f"[Distributed] Initialized process group - rank={self.rank}, "
                f"world_size={self.world_size}, local_rank={self.local_rank}, "
                f"master={self.is_master}, device={self.device}"
            )

    def destroy_process_group(self) -> None:
        """Cleanup distributed process group."""
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def is_distributed(self) -> bool:
        """Check if distributed training is enabled."""
        return dist.is_available() and dist.is_initialized() and self.world_size > 1

    def should_continue(self, local_has_batch: bool, device: torch.device) -> bool:
        """
        Synchronize across all ranks. If *any* rank runs out of batches, all must stop.

        Args:
            local_has_batch: Whether this rank has more batches
            device: Device to use for synchronization

        Returns:
            True if all ranks should continue, False otherwise
        """
        if not self.is_distributed():
            return local_has_batch

        flag_tensor = torch.tensor([int(local_has_batch)], device=device)
        dist.all_reduce(flag_tensor, op=dist.ReduceOp.MIN)
        return bool(flag_tensor.item())

    def ddp_reduce(
        self,
        value: int | float | torch.Tensor,
        op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
        device: torch.device | None = None,
    ) -> float:
        """
        Reduce value across all ranks and return a python float.

        Args:
            value: Value to reduce
            op: Reduction operation (SUM, AVG, MIN, MAX)
            device: Device to use for reduction

        Returns:
            Reduced value as a float
        """
        # Single-GPU fast path
        if not self.is_distributed():
            return float(value.item() if isinstance(value, torch.Tensor) else value)

        # Use provided device or default
        if device is None:
            device = self.device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Convert to tensor on the right device
        if not isinstance(value, torch.Tensor):
            tensor = torch.tensor(float(value), device=device)
        else:
            tensor = value.to(device)

        dist.all_reduce(tensor, op=op)
        return float(tensor.item())

    def barrier(self, device_ids: list | None = None) -> None:
        """
        Synchronization barrier across all ranks.

        Args:
            device_ids: Optional device IDs for NCCL backend
        """
        if not self.is_distributed():
            return

        if device_ids is not None and dist.get_backend() == "nccl":
            dist.barrier(device_ids=device_ids)
        else:
            dist.barrier()

    def safe_barrier(self, tag: str = "", local_rank: int | None = None) -> bool:
        """
        Safe barrier with fallback handling for older PyTorch versions.

        Args:
            tag: Tag for logging
            local_rank: Local rank for device-specific barrier

        Returns:
            True if barrier succeeded, False otherwise
        """
        if not self.is_distributed():
            return True

        try:
            # If we're on NCCL, ensure the barrier runs on this rank's CUDA device.
            # Fall back cleanly on older PyTorch or non-NCCL backends.
            if torch.cuda.is_available() and local_rank is not None:
                try:
                    dist.barrier(device_ids=[local_rank])
                    return True
                except TypeError:
                    # Older PyTorch or backend that doesn't accept device_ids
                    pass
                except Exception as e:
                    # Unexpected error with device_ids; try a plain barrier
                    if tag:
                        tplr.logger.warning(
                            f"[barrier:{tag}] device_ids barrier warn: {e}"
                        )
            dist.barrier()
            return True
        except Exception as e:
            if tag:
                tplr.logger.error(f"[barrier:{tag}] failed: {e}")
            return False

    def all_ok(self, ok: bool, device: torch.device, tag: str = "") -> bool:
        """
        Group-wide MIN-reduce on a 1/0 flag: returns True only if all ranks reported ok=True.
        Never raises; logs and returns False if collectives fail.

        Args:
            ok: Local flag value
            device: Device for synchronization
            tag: Optional tag for logging

        Returns:
            True if all ranks reported ok=True
        """
        if not self.is_distributed():
            return ok

        try:
            t = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            out = bool(t.item())
            if not out and self.is_master and tag:
                tplr.logger.warning(f"[sync:{tag}] some rank failed; skipping step")
            return out
        except Exception as e:
            if tag:
                tplr.logger.error(f"[sync:{tag}] all_reduce failed: {e}")
            return False

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> None:
        """
        Broadcast tensor from source rank to all other ranks.

        Args:
            tensor: Tensor to broadcast (modified in-place)
            src: Source rank for broadcast
        """
        if self.is_distributed():
            dist.broadcast(tensor, src=src)

    def gather_object(
        self, obj: Any, object_list: list | None = None, dst: int = 0
    ) -> list | None:
        """
        Gather arbitrary Python objects from all ranks to destination rank.

        Args:
            obj: Object to send from this rank
            object_list: List to receive gathered objects (only needed on dst rank)
            dst: Destination rank

        Returns:
            List of gathered objects on dst rank, None on other ranks
        """
        if not self.is_distributed():
            return [obj] if self.rank == dst else None

        if self.rank == dst and object_list is None:
            object_list = [None] * self.world_size

        dist.gather_object(obj, object_list, dst=dst)
        return object_list if self.rank == dst else None

    def all_gather_object(self, obj: Any, object_list: list | None = None) -> list:
        """
        Gather arbitrary Python objects from all ranks to all ranks.

        Args:
            obj: Object to send from this rank
            object_list: List to receive gathered objects (will be created if None)

        Returns:
            List of gathered objects from all ranks
        """
        if not self.is_distributed():
            return [obj]

        if object_list is None:
            object_list = [None] * self.world_size

        dist.all_gather_object(object_list, obj)
        return object_list

    def all_reduce_flag(self, ok: bool, device: torch.device, tag: str = "") -> bool:
        """
        Group-wide MIN-reduce on a 1/0 flag: returns True only if all ranks reported ok=True.

        Args:
            ok: Local flag value
            device: Device for synchronization
            tag: Optional tag for logging

        Returns:
            True if all ranks reported ok=True
        """
        if not self.is_distributed():
            return ok

        t = torch.tensor([int(ok)], dtype=torch.uint8, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        result = bool(t.item())

        if not result and self.is_master and tag:
            tplr.logger.warning(f"[sync:{tag}] some rank failed; skipping step")

        return result

    def get_offloaded_params(self, model: torch.nn.Module) -> tuple[list, list]:
        """
        Get a copy of current parameters and offload them to CPU.

        Args:
            model: Model to offload parameters from

        Returns:
            Tuple of (offloaded_params, param_specs) for restoration
        """
        params_offloaded = []
        param_info = []

        for param in model.parameters():
            if isinstance(param, DT):
                # Get the local TP shard and store the spec
                local_param = param.to_local()
                params_offloaded.append(local_param.detach().clone().to("cpu"))
                param_info.append(
                    {
                        "is_dtensor": True,
                        "device_mesh": param.device_mesh,
                        "placements": param.placements,
                        "local_shape": local_param.shape,
                    }
                )
            else:
                # For regular tensors
                params_offloaded.append(param.data.detach().clone().to("cpu"))
                param_info.append({"is_dtensor": False})

        return params_offloaded, param_info

    def restore_offloaded_params(
        self, model: torch.nn.Module, params_offloaded: list, param_specs: list
    ) -> None:
        """
        Restore offloaded parameters back to the model.

        Args:
            model: Model to restore parameters to
            params_offloaded: List of offloaded parameters
            param_specs: List of parameter specifications
        """
        with torch.no_grad():
            for (saved_param, param_meta), p in zip(
                zip(params_offloaded, param_specs), model.parameters()
            ):
                if param_meta["is_dtensor"] and isinstance(p, DT):
                    # Handle TP DTensors
                    saved_param = saved_param.to(p.device, non_blocking=True)

                    # Create a DTensor from the local shard directly
                    saved_param_dtensor = DT.from_local(
                        saved_param,
                        device_mesh=param_meta["device_mesh"],
                        placements=param_meta["placements"],
                        run_check=False,
                    )
                    p.grad = saved_param_dtensor - p
                    p.data.copy_(saved_param_dtensor.data)
                else:
                    saved_param = saved_param.to(p.device, non_blocking=True)
                    p.grad = saved_param - p.data
                    p.data.copy_(saved_param)


# Global instance for convenience
dist_helper = DistributedHelper()
