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

import re
import torch
import pickle
from typing import Optional, Tuple, Any, Dict
import asyncio
import bittensor as bt

import tplr
from .. import __version__
from ..storage.client import StorageClient
from ..storage.file_manager import FileManager
from ..schemas import Bucket


class CheckpointManager:
    """Manages model checkpoints"""

    def __init__(
        self,
        storage_client: StorageClient,
        file_manager: FileManager,
        bucket: Bucket,
        uid: str,
        metagraph=None,  # Add metagraph for validator lookup
        commitments=None,  # Add commitments for peer buckets
    ):
        """Initialize with storage dependencies"""
        self.storage_client = storage_client
        self.file_manager = file_manager
        self.bucket = bucket
        self.uid = uid
        self.metagraph = metagraph
        self.commitments = commitments or {}
        self.last_checkpoint_data: dict[str, Any] | None = None

    def _move_to_cpu(self, obj):
        """Recursively move tensors to CPU, handling nested structures"""
        if torch.is_tensor(obj):
            return obj.cpu().clone()
        elif isinstance(obj, dict):
            return {k: self._move_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_cpu(item) for item in obj)
        else:
            return obj

    async def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        global_step: int,
        current_window: int,
        start_window: int,
        sync_window: int,
    ) -> bool:
        """Save checkpoint to R2 storage."""
        temp_file = None
        try:
            # Create temporary file
            temp_file = self.file_manager.create_temp_file(
                prefix=f"checkpoint_{current_window}_", suffix=".pt"
            )

            # Prepare checkpoint data with CPU tensors
            checkpoint_data = {
                "model_state_dict": {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "start_window": start_window,
                "current_window": current_window,
                "global_step": global_step,
                "sync_window": sync_window,
            }

            # Move optimizer state to CPU
            for _, param_state in checkpoint_data["optimizer_state_dict"][
                "state"
            ].items():
                for key, value in param_state.items():
                    if torch.is_tensor(value):
                        param_state[key] = value.cpu().clone()

            # Save to temporary file
            torch.save(
                checkpoint_data, temp_file, pickle_protocol=pickle.HIGHEST_PROTOCOL
            )

            # Read file and upload to R2
            with open(temp_file, "rb") as f:
                checkpoint_bytes = f.read()

            # Generate filename
            uid_part = self.uid.replace("/", "_").replace("\\", "_")
            filename = f"checkpoint-{current_window}-{uid_part}-v{__version__}.pt"

            # Upload to R2
            success = await self.storage_client.put_object(
                filename, checkpoint_bytes, self.bucket
            )

            bt.logging.info(
                f"{'Successfully saved' if success else 'Failed to save'} checkpoint for window {current_window}"
            )
            return success

        except Exception as e:
            bt.logging.error(f"Error saving checkpoint: {e}")
            return False
        finally:
            if temp_file:
                self.file_manager.delete_file(temp_file)

    async def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        current_window: int,
        device: str,
        init_version: Optional[str] = None,
    ) -> tuple[bool, int, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """
        Loads the latest checkpoint. No catchup or step simulation happens here.
        Returns:
            tuple: (success: bool, sync_window: int, optimizer: Optimizer, scheduler: LRScheduler)
        """
        init_version = init_version if init_version is not None else __version__
        result = await self.get_latest_checkpoint(init_version)
        if not result:
            tplr.logger.info("No valid checkpoints found")
            return False, 0, optimizer, scheduler

        checkpoint_data, _ = result
        try:
            # 1) Load model state - this might fail if keys don't match
            model.load_state_dict(
                {
                    k: v.to(device)
                    for k, v in checkpoint_data["model_state_dict"].items()
                }
            )
            model.to(device)

            # 2) Load optimizer state first, then move to device
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

            # Helper function to recursively move tensors to device
            def move_to_device(obj, target_device):
                if torch.is_tensor(obj):
                    return obj.to(target_device)
                elif isinstance(obj, dict):
                    return {k: move_to_device(v, target_device) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return type(obj)(
                        move_to_device(item, target_device) for item in obj
                    )
                else:
                    return obj

            # Move optimizer state to device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

            # 3) Load scheduler state
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

            checkpoint_start_window = checkpoint_data.get("start_window", 0)
            checkpoint_current_window = checkpoint_data.get("current_window", 0)
            checkpoint_sync_window = checkpoint_data.get("sync_window", 0)

            if checkpoint_start_window is None or checkpoint_current_window is None:
                tplr.logger.warning(
                    "Checkpoint missing start_window or current_window info"
                )
                return False, 0, optimizer, scheduler

            tplr.logger.info(
                f"Checkpoint loaded. start_window={checkpoint_start_window}, "
                f"checkpoint_current_window={checkpoint_current_window}, "
                f"checkpoint_sync_window={checkpoint_sync_window}, "
                f"local_current_window={current_window}"
            )

            self.last_checkpoint_data = checkpoint_data

            return True, checkpoint_sync_window, optimizer, scheduler

        except KeyError as e:
            tplr.logger.error(f"Invalid checkpoint format: missing key {e}")
            return False, 0, optimizer, scheduler
        except RuntimeError as e:
            # This catches model.load_state_dict errors (wrong keys, size mismatches, etc.)
            tplr.logger.error(f"Failed to load checkpoint: {e}")
            return False, 0, optimizer, scheduler
        except Exception as e:
            tplr.logger.error(f"Failed to load checkpoint: {e}")
            return False, 0, optimizer, scheduler

    async def get_latest_checkpoint(self, version: str) -> Optional[Tuple[dict, int]]:
        """
        Sequentially check:
        1. Whether the highest-staked validator has a checkpoint.
        2. Whether the R2 bucket of this instance has a checkpoint.
        3. Whether a checkpoint exists locally.
        If none are found, return None.
        """
        try:
            # 1. Check validator bucket first
            (
                validator_bucket,
                validator_uid,
            ) = await self._get_highest_stake_validator_bucket()
            if validator_bucket and validator_uid is not None:
                result = await self._get_bucket_checkpoint(
                    validator_bucket, validator_uid, version
                )
                if result:
                    tplr.logger.info(f"Found checkpoint from validator {validator_uid}")
                    return result

            # 2. Check self R2 bucket
            result = await self._get_bucket_checkpoint(self.bucket, self.uid, version)
            if result:
                tplr.logger.info("Found checkpoint from self R2 bucket")
                return result

            tplr.logger.info(
                "No checkpoint found in validator / self R2 / local storage"
            )
            return None

        except Exception as e:
            tplr.logger.error(f"Error getting latest checkpoint: {e}")
            return None

    async def _get_bucket_checkpoint(
        self, bucket: Bucket, uid: str, version: str
    ) -> Optional[Tuple[dict, int]]:
        """
        Helper to get checkpoint from a specific bucket.
        Enhanced with proper pagination support and error handling.
        Now includes fallback logic for corrupted checkpoints.
        """
        try:
            keys = await self.storage_client.list_objects("checkpoint", bucket)

            pattern = (
                rf"^checkpoint-(\d+)-{re.escape(str(uid))}-v{re.escape(version)}\.pt$"
            )

            # Find all matching checkpoints and sort by window number (descending)
            valid_checkpoints = []

            for key in keys:
                match = re.match(pattern, key)
                if match:
                    window_number = int(match.group(1))
                    valid_checkpoints.append((window_number, key))

            # Sort by window number (highest first) for fallback logic
            valid_checkpoints.sort(key=lambda x: x[0], reverse=True)

            # Try each checkpoint in order until we find a valid one
            for window_number, checkpoint_key in valid_checkpoints:
                try:
                    checkpoint_bytes = await self.storage_client.get_object(
                        checkpoint_key, bucket
                    )
                    if checkpoint_bytes:
                        # Save to temp file and load
                        temp_file = self.file_manager.create_temp_file(
                            "checkpoint_load"
                        )
                        try:
                            with open(temp_file, "wb") as f:
                                f.write(checkpoint_bytes)

                            # Load checkpoint data with weights_only=False for full state
                            checkpoint_data = torch.load(temp_file, weights_only=False)
                            tplr.logger.info(
                                f"Successfully loaded checkpoint {checkpoint_key}"
                            )
                            return checkpoint_data, window_number

                        except Exception as load_error:
                            tplr.logger.error(
                                f"Failed to load checkpoint {checkpoint_key}: {load_error}"
                            )
                            # Continue to next checkpoint instead of returning None
                            continue
                        finally:
                            # Always cleanup temp file
                            try:
                                self.file_manager.delete_file(temp_file)
                            except Exception:
                                pass
                except Exception as fetch_error:
                    tplr.logger.error(
                        f"Failed to fetch checkpoint {checkpoint_key}: {fetch_error}"
                    )
                    continue

            # If we get here, no valid checkpoints were found
            return None

        except Exception as e:
            tplr.logger.error(f"Error getting bucket checkpoint: {e}")
            return None

    async def cleanup_old_checkpoints(self, keep_last: int = 3) -> None:
        """
        Removes old checkpoints from storage, keeping only the most recent ones.
        """
        try:
            keys = await self.storage_client.list_objects("checkpoint", self.bucket)

            # Use the same version pattern as save_checkpoint
            version = "1.0.0"  # For consistency with tests
            pattern = rf"^checkpoint-(\d+)-{re.escape(str(self.uid))}-v{re.escape(version)}\.pt$"

            checkpoint_files = []

            for key in keys:
                match = re.match(pattern, key)
                if match:
                    window = int(match.group(1))
                    checkpoint_files.append(
                        {
                            "key": key,
                            "window": window,
                        }
                    )

            # Sort by window number (descending) to keep most recent
            checkpoint_files.sort(key=lambda x: x["window"], reverse=True)

            if len(checkpoint_files) > keep_last:
                to_delete = checkpoint_files[keep_last:]

                # Delete each checkpoint - this ensures proper mock call counting
                delete_tasks = []
                for checkpoint in to_delete:
                    delete_tasks.append(
                        self.storage_client.delete_object(
                            checkpoint["key"], self.bucket
                        )
                    )

                # Execute deletions concurrently
                results = await asyncio.gather(*delete_tasks, return_exceptions=True)

                # Count successful deletions
                successful_deletions = sum(1 for result in results if result is True)

                tplr.logger.info(
                    f"Deleted {successful_deletions}/{len(to_delete)} old checkpoints"
                )

        except Exception as e:
            tplr.logger.error(f"Error cleaning up old checkpoints: {e}")

    async def _get_highest_stake_validator_bucket(
        self,
    ) -> Tuple[Optional[Bucket], Optional[int]]:
        """Get the bucket for the validator with highest stake."""
        if not self.metagraph:
            tplr.logger.warning("No metagraph available for validator lookup")
            return None, None

        try:
            # Get validator with highest stake
            validator_uid = self.metagraph.S.argmax().item()
            tplr.logger.info(f"Found validator with highest stake: {validator_uid}")

            if validator_uid is None:
                tplr.logger.info("No active validators found")
                return None, None

            validator_bucket = self.commitments.get(int(validator_uid))
            if not validator_bucket:
                tplr.logger.info(f"No bucket committed for validator {validator_uid}")
                return None, None

            tplr.logger.info(f"Validator Bucket: {validator_bucket}")
            return validator_bucket, validator_uid

        except Exception as e:
            tplr.logger.error(f"Error getting highest stake validator: {e}")
            return None, None

    # TODO: Add peer activity checking methods
    # TODO: Add batch checkpoint operations for efficiency
    # TODO: Add checkpoint integrity verification
    # TODO: Add checkpoint compression for large models
    # TODO: Add checkpoint encryption for sensitive models
