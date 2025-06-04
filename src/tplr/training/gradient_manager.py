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


from typing import Any
import torch

import tplr
from ..storage.client import StorageClient
from ..storage.file_manager import FileManager
from ..compress import CompressDCT, TransformDCT


class GradientManager:
    """Handles gradient-specific operations"""

    def __init__(
        self,
        storage_client: StorageClient,
        file_manager: FileManager,
        device: str,
        hparams,
    ):
        """Initialize with storage and file management dependencies"""
        self.storage_client = storage_client
        self.file_manager = file_manager
        self.device = device
        self.hparams = hparams

    async def serialize_gradient(self, state_dict: dict, global_step: int) -> str:
        """Serialize gradient state dict to temporary file and return path"""
        temp_file_path = self.file_manager.create_temp_file("gradient_serialize")

        save_data = {
            "state_dict": state_dict,
            "global_step": global_step,
        }

        torch.save(save_data, temp_file_path)
        return temp_file_path

    async def deserialize_gradient(self, file_path: str) -> tuple[dict, int]:
        """Deserialize gradient from file path"""
        loaded_data = torch.load(
            file_path, map_location=self.device, weights_only=False
        )

        state_dict = loaded_data.get("state_dict", {})
        global_step = loaded_data.get("global_step", 0)

        return state_dict, global_step

    def validate_gradient(self, state_dict: dict, totalks: dict) -> bool:
        """Validate gradient state dict against totalks"""
        try:
            for param_name, tensor in state_dict.items():
                if param_name.endswith("idxs"):
                    base_name = param_name[:-4]
                    totalk = totalks.get(base_name)
                    if totalk is None:
                        tplr.logger.warning(f"Missing totalk for parameter {base_name}")
                        return False

                    try:
                        self.check_compressed_indices(
                            param_name,
                            tensor.to(self.device),
                            totalk,
                            allowed_topk=self.hparams.topk_compression,
                        )
                    except Exception as e:
                        tplr.logger.warning(
                            f"Compressed indices check failed for {param_name}: {e}"
                        )
                        return False

                elif param_name.endswith("vals"):
                    tensor_to_check = tensor.to(self.device)
                    if (
                        torch.isnan(tensor_to_check).any()
                        or torch.isinf(tensor_to_check).any()
                    ):
                        tplr.logger.warning(f"NaN/Inf in {param_name}")
                        return False

            return True

        except Exception as e:
            tplr.logger.error(f"Error validating gradient: {e}")
            return False

    def check_compressed_indices(
        self,
        param_name: str,
        idxs: Any,
        totalk: int,
        allowed_topk: int | None = None,
    ) -> None:
        allowed_topk = (
            min(self.hparams.topk_compression, totalk)
            if allowed_topk is None
            else min(allowed_topk, totalk)
        )

        def _bounds_check(t: torch.Tensor):
            """fast min/max bounds check"""
            if t.numel() == 0:
                raise ValueError(f"[{param_name}] empty index list")
            if t.min().item() < 0 or t.max().item() >= totalk:
                bad = t[(t < 0) | (t >= totalk)][0].item()
                raise ValueError(
                    f"[{param_name}] Index {bad} out of bounds (totalk = {totalk})"
                )

        if isinstance(idxs, (int, float)) or (torch.is_tensor(idxs) and idxs.ndim == 0):
            idx_int = int(idxs)
            if not (0 <= idx_int < totalk):
                raise ValueError(
                    f"[{param_name}] Index {idx_int} out of bounds (totalk = {totalk})"
                )
            return  # single scalar is always length-independent

        if (
            isinstance(idxs, (list, tuple))
            and idxs
            and isinstance(idxs[0], (list, tuple))
        ):
            for sub in idxs:
                if len(sub) != allowed_topk:
                    raise ValueError(
                        f"[{param_name}] Invalid number of indices: "
                        f"got {len(sub)} but expected {allowed_topk}"
                    )
                # vectorised bounds check on each sub-tensor
                t = torch.as_tensor(sub, dtype=torch.long)
                _bounds_check(t)
            return

        try:
            t = (
                idxs
                if torch.is_tensor(idxs)
                else torch.as_tensor(idxs, dtype=torch.long)
            )
        except Exception as e:
            raise ValueError(f"[{param_name}] Failed to convert indices to tensor: {e}")

        if t.ndim == 1:  # flat
            if t.numel() != allowed_topk:
                raise ValueError(
                    f"[{param_name}] Invalid number of indices: "
                    f"{t.numel()} but expected {allowed_topk}"
                )
            _bounds_check(t)
            return

        # n-D compressed: last dim must be allowed_topk
        if t.size(-1) != allowed_topk:
            raise ValueError(
                f"[{param_name}] Last dimension size invalid: "
                f"{t.size(-1)} but expected {allowed_topk}"
            )
        _bounds_check(t)

    async def apply_gradients_to_model(
        self,
        gather_result,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        transformer: TransformDCT,
        compressor: CompressDCT,
        device: str,
        window: int,
        global_step: int,
        shapes: dict | None = None,
        totalks: dict | None = None,
    ) -> tuple[bool, int]:
        """Apply gathered gradients to model parameters.

        Args:
            gather_result: Gathered gradient data
            model: The model to update
            optimizer: SGD optimizer
            scheduler: Learning rate scheduler
            transformer: DCT transformer
            compressor: Gradient compressor
            device: Computing device
            window: Current window number
            global_step: Global step counter
            shapes: Parameter shapes dict (required for reconstruction)
            totalks: Total parameters dict (required for reconstruction)

        Returns:
            Tuple[bool, int]: (success, new_global_step)
        """
        try:
            if not gather_result or not gather_result.state_dict:
                return False, global_step

            if shapes is None or totalks is None:
                tplr.logger.error("shapes and totalks parameters are required")
                return False, global_step

            model.train()
            optimizer.zero_grad()
            model.zero_grad()

            # Apply gradients
            for n, p in model.named_parameters():
                idxs = getattr(gather_result.state_dict, f"{n}idxs", None)
                vals = getattr(gather_result.state_dict, f"{n}vals", None)

                if idxs is not None and vals is not None:
                    if not isinstance(idxs, (list, tuple)):
                        idxs = [idxs]
                    if not isinstance(vals, (list, tuple)):
                        vals = [vals]

                    if n not in shapes or n not in totalks:
                        tplr.logger.warning(
                            f"Missing shape or totalk for parameter {n}"
                        )
                        continue

                    new_grad = transformer.decode(
                        compressor.batch_decompress(
                            p.to(device),
                            idxs,
                            vals,
                            shapes[n],
                            totalks[n],
                        )
                    )
                    if p.grad is None:
                        p.grad = new_grad
                    else:
                        p.grad.copy_(new_grad)
                    p.grad.sign_()

            optimizer.step()
            scheduler.step()
            global_step += 1

            tplr.logger.info(
                f"Applied gradients for window {window}, global_step => {global_step}"
            )
            return True, global_step

        except Exception as e:
            tplr.logger.error(
                f"Failed to apply gradients for window {window}: {str(e)}"
            )
            return False, global_step

    def validate_gradient_tensor(self, tensor: torch.Tensor, param_name: str) -> bool:
        """Validate individual gradient tensor"""
        try:
            if torch.isnan(tensor).any():
                tplr.logger.warning(f"NaN detected in {param_name}")
                return False
            if torch.isinf(tensor).any():
                tplr.logger.warning(f"Inf detected in {param_name}")
                return False
            return True
        except Exception as e:
            tplr.logger.error(f"Error validating tensor {param_name}: {e}")
            return False
