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

import os
import re
import torch
from typing import Optional, Tuple, Any, Dict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import tplr
from .. import __version__
from ..storage.client import StorageClient
from ..storage.file_manager import FileManager, LOCAL_TMP_DIR
from ..schemas import Bucket


class CheckpointManager:
    """Manages model checkpoints"""

    def __init__(
        self,
        storage_client: StorageClient,
        file_manager: FileManager,
        bucket: Bucket,
        uid: str,
    ):
        """Initialize with storage dependencies"""
        self.storage_client = storage_client
        self.file_manager = file_manager
        self.bucket = bucket
        self.uid = uid
        self.last_checkpoint_data: Optional[Dict[str, Any]] = None

    async def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        momentum: dict,
        global_step: int,
        current_window: int,
        start_window: int,
    ) -> bool:
        """Save checkpoint to R2 storage only."""
        try:
            checkpoint_data = {
                "model_state_dict": {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                },
                "optimizer_state_dict": {
                    k: v.cpu().clone() if torch.is_tensor(v) else v
                    for k, v in optimizer.state_dict().items()
                },
                "scheduler_state_dict": scheduler.state_dict(),
                "momentum": {k: v.cpu().clone() for k, v in momentum.items()},
                "start_window": start_window,
                "current_window": current_window,
                "global_step": global_step,
            }

            filename = f"checkpoint-{current_window}-{self.uid}-v{__version__}.pt"

            # Serialize to temp file
            temp_file_path = self.file_manager.create_temp_file("checkpoint")
            torch.save(checkpoint_data, temp_file_path)

            # Upload to R2
            with open(temp_file_path, "rb") as f:
                data = f.read()

            success = await self.storage_client.put_object(filename, data, self.bucket)

            # Cleanup temp file
            self.file_manager.delete_file(temp_file_path)

            if success:
                tplr.logger.info(
                    f"Successfully saved checkpoint for window {current_window}"
                )
                return True
            else:
                tplr.logger.error(
                    f"Failed to upload checkpoint for window {current_window}"
                )
                return False

        except Exception as e:
            tplr.logger.error(f"Error saving checkpoint: {e}")
            return False

    async def load_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        current_window: int,
        device: str,
        init_version: Optional[str] = None,
    ) -> Tuple[bool, int, Optimizer, _LRScheduler]:
        """
        Loads the latest checkpoint. No catchup or step simulation happens here.
        Returns:
            tuple: (success: bool, checkpoint_sync_window: int, optimizer: Optimizer, scheduler: LRScheduler)
        """
        init_version = init_version if init_version is not None else __version__
        result = await self.get_latest_checkpoint(init_version)
        if not result:
            tplr.logger.info("No valid checkpoints found")
            return False, 0, optimizer, scheduler

        checkpoint_data, checkpoint_window = result
        try:
            # 1) Load model and optimizer state
            model.load_state_dict(
                {
                    k: v.to(device)
                    for k, v in checkpoint_data["model_state_dict"].items()
                }
            )
            model.to(device)

            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

            checkpoint_start_window = checkpoint_data.get("start_window")
            checkpoint_current_window = checkpoint_data.get("current_window")
            checkpoint_sync_window = checkpoint_data.get(
                "sync_window", checkpoint_current_window
            )

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
            # For now, we'll check self R2 bucket and local storage
            # TODO: Add validator bucket checking when ChainManager is available

            # 1. Check self R2 bucket
            result = await self._get_bucket_checkpoint(self.bucket, self.uid, version)
            if result:
                return result

            # 2. Check local storage
            local_result = self._load_latest_local_checkpoint(version)
            if local_result:
                return local_result

            tplr.logger.info("No checkpoint found in self R2 / local storage")
            return None

        except Exception as e:
            tplr.logger.error(f"Error getting latest checkpoint: {e}")
            return None

    async def _get_bucket_checkpoint(
        self, bucket: Bucket, uid: str, version: str
    ) -> Optional[Tuple[dict, int]]:
        """Helper to get checkpoint from a specific bucket."""
        try:
            keys = await self.storage_client.list_objects("checkpoint", bucket)

            pat = re.compile(rf"^checkpoint-(\d+)-{uid}-v{re.escape(version)}\.pt$")

            # Find the latest checkpoint window
            latest_checkpoint = None
            max_window = -1

            for key in keys:
                match = pat.match(key)
                if match:
                    window_number = int(match.group(1))
                    if window_number > max_window:
                        max_window = window_number
                        latest_checkpoint = key

            # If we found a valid checkpoint, fetch it
            if latest_checkpoint:
                data = await self.storage_client.get_object(latest_checkpoint, bucket)
                if data:
                    # Save to temp file and load
                    temp_file = self.file_manager.create_temp_file("checkpoint_load")
                    with open(temp_file, "wb") as f:
                        f.write(data)

                    loaded_data = torch.load(temp_file, weights_only=False)
                    self.file_manager.delete_file(temp_file)

                    return loaded_data, max_window

            return None

        except Exception as e:
            tplr.logger.error(f"Error getting bucket checkpoint: {e}")
            return None

    def _load_latest_local_checkpoint(self, version: str) -> Optional[Tuple[dict, int]]:
        """Load latest local checkpoint"""
        try:
            local_dir = os.path.join(LOCAL_TMP_DIR, str(self.uid))
            pattern = rf"checkpoint-(\d+)-{self.uid}-v{re.escape(version)}\.pt$"

            if not os.path.exists(local_dir):
                return None

            checkpoints = []
            for window_dir in os.listdir(local_dir):
                path = os.path.join(local_dir, window_dir)
                if not os.path.isdir(path):
                    continue

                for file in os.listdir(path):
                    match = re.match(pattern, file)
                    if match:
                        # window number comes from match.group(1)
                        w = int(match.group(1))
                        file_path = os.path.join(path, file)
                        checkpoints.append(
                            {
                                "path": file_path,
                                "window": w,
                                "modified": os.path.getmtime(file_path),
                            }
                        )

            if checkpoints:
                # choose the last modified checkpoint
                latest = max(checkpoints, key=lambda x: x["modified"])
                checkpoint_data = torch.load(latest["path"], weights_only=False)
                return checkpoint_data, latest["window"]
            else:
                return None
        except Exception as e:
            tplr.logger.error(f"Error in local checkpoint loading: {e}")
            return None

    async def cleanup_old_checkpoints(self, keep_last: int = 3) -> None:
        """
        Removes old checkpoints from storage, keeping only the most recent ones.
        """
        try:
            keys = await self.storage_client.list_objects("checkpoint", self.bucket)

            # Filter for this uid's checkpoints
            pattern = rf"^checkpoint-(\d+)-{self.uid}-v{re.escape(__version__)}\.pt$"
            checkpoint_files = []

            for key in keys:
                match = re.match(pattern, key)
                if match:
                    window = int(match.group(1))
                    checkpoint_files.append({"key": key, "window": window})

            # Sort by window number (descending)
            checkpoint_files.sort(key=lambda x: x["window"], reverse=True)

            if len(checkpoint_files) > keep_last:
                to_delete = checkpoint_files[keep_last:]
                for checkpoint in to_delete:
                    await self.storage_client.delete_object(
                        checkpoint["key"], self.bucket
                    )

                tplr.logger.info(f"Deleted {len(to_delete)} old checkpoints")

        except Exception as e:
            tplr.logger.error(f"Error cleaning up old checkpoints: {e}")
