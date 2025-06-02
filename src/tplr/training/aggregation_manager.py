# ruff: noqa

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

import asyncio
import torch
from datetime import datetime
from types import SimpleNamespace
from typing import Optional, List, Dict
import os
import time

import tplr
from .. import __version__
from ..config import BUCKET_SECRETS
from ..schemas import Bucket
from ..storage.client import StorageClient
from ..storage.file_manager import FileManager
from .gradient_manager import GradientManager
from ..network.peer_manager import PeerManager


class AggregationManager:
    """Handles gradient gathering and aggregation"""

    def __init__(
        self,
        gradient_manager: GradientManager,
        peer_manager: PeerManager,
        storage_client: StorageClient,
        file_manager: FileManager,
        hparams,
        device: str,
    ):
        """Initialize with dependencies"""
        self.gradient_manager = gradient_manager
        self.peer_manager = peer_manager
        self.storage_client = storage_client
        self.file_manager = file_manager
        self.hparams = hparams
        self.device = device

        # Semaphore for controlling concurrent gather operations
        self.gather_semaphore = asyncio.Semaphore(15)

    async def gather_gradients(
        self,
        my_uid: int,
        uids: List[int],
        window: int,
        timeout: int,
        device: str,
        totalks: dict,
        local: bool = True,
        time_min: datetime = None,
        time_max: datetime = None,
        stale_retention: int = 10,
    ) -> Optional[SimpleNamespace]:
        """
        Gather gradients from multiple UIDs for a specific window.
        Equivalent to the original gather method with proper aggregation logic.
        """
        start_time = time.time()
        metrics = {"upload_bytes": 0, "download_bytes": 0, "successes": []}

        tplr.logger.debug(
            f"Starting gather for window {window} with time window: {time_min} to {time_max}"
        )
        tplr.logger.debug(
            f"Gather operation - my_uid: {my_uid}, window: {window}, timeout: {timeout}"
        )
        tplr.logger.debug(f"Target UIDs for gathering: {uids}")

        # Collect raw responses from all UIDs
        raw_responses = []
        async with self.gather_semaphore:
            batch_tasks = [
                self._get_with_retry(
                    str(uid),
                    window,
                    "gradient",
                    timeout,
                    local=local,
                    time_min=time_min,
                    time_max=time_max,
                )
                for uid in uids
            ]

            try:
                batch_responses = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )

                for uid, response in zip(uids, batch_responses):
                    raw_responses.append(
                        {
                            "uid": uid,
                            "response": response,
                            "is_exception": isinstance(response, Exception),
                        }
                    )

            except Exception as e:
                tplr.logger.error(f"Error processing uid batch: {str(e)}")

        # Aggregate the responses
        aggregation_result = self._aggregate_gradients(
            raw_responses, device, totalks, metrics
        )

        if aggregation_result is None:
            return None

        total_time = time.time() - start_time
        tplr.logger.info(
            f"Gather done in {total_time:.2f}s. Success rate: {len(aggregation_result['valid_uids'])}/{len(uids)}, "
            f"Upload: {metrics['upload_bytes']} bytes, Download: {metrics['download_bytes']} bytes"
        )

        # Add debug logging before creating result
        if aggregation_result is not None:
            tplr.logger.debug(
                f"Aggregation result keys: {list(aggregation_result['aggregated_state_dict'].keys())}"
            )
            for key, value in aggregation_result["aggregated_state_dict"].items():
                if isinstance(value, list):
                    tplr.logger.debug(
                        f"Key {key}: list of {len(value)} items, first item type: {type(value[0]) if value else 'empty'}"
                    )
                else:
                    tplr.logger.debug(f"Key {key}: {type(value)}")

        result = SimpleNamespace(
            time=total_time,
            upload_bytes=metrics["upload_bytes"],
            download_bytes=metrics["download_bytes"],
            success_rate=len(aggregation_result["valid_uids"]) / len(uids),
            state_dict=SimpleNamespace(**aggregation_result["aggregated_state_dict"]),
            uids=aggregation_result["valid_uids"],
            global_steps=aggregation_result["global_steps"],
            skipped_uids=aggregation_result["skipped_uids"],
        )

        # Add debug logging after creating result
        tplr.logger.debug(f"Result state_dict attributes: {dir(result.state_dict)}")
        for attr in dir(result.state_dict):
            if not attr.startswith("_"):
                value = getattr(result.state_dict, attr)
                tplr.logger.debug(
                    f"result.state_dict.{attr}: {type(value)} with length {len(value) if hasattr(value, '__len__') else 'N/A'}"
                )

        return result

    def _aggregate_gradients(
        self, raw_responses: List[dict], device: str, totalks: dict, metrics: dict
    ) -> Optional[dict]:
        """
        Aggregate gradients from raw responses with validation.
        """
        aggregated_state_dict = {}
        valid_uids = []
        skipped_uids = []
        global_steps = []

        for response_data in raw_responses:
            uid = response_data["uid"]
            response = response_data["response"]

            if response_data["is_exception"]:
                tplr.logger.debug(f"Error from UID {uid}: {str(response)}")
                skipped_uids.append(uid)
                continue

            if response is None:
                tplr.logger.info(f"Skipped UID {uid} - gradient not found.")
                skipped_uids.append(uid)
                continue

            try:
                # FIXED: Handle tuple format (state_dict, global_step)
                if isinstance(response, tuple) and len(response) == 2:
                    state_dict_resp, global_step_resp = response
                    tplr.logger.debug(
                        f"UID {uid} state_dict_resp keys: {list(state_dict_resp.keys()) if isinstance(state_dict_resp, dict) else type(state_dict_resp)}"
                    )
                    tplr.logger.debug(f"UID {uid} global_step_resp: {global_step_resp}")

                    # Extract the actual state_dict from the nested structure
                    if (
                        isinstance(state_dict_resp, dict)
                        and "state_dict" in state_dict_resp
                    ):
                        actual_state_dict = state_dict_resp["state_dict"]
                        actual_global_step = state_dict_resp["global_step"]
                        tplr.logger.debug(
                            f"UID {uid} actual state_dict keys: {list(actual_state_dict.keys()) if isinstance(actual_state_dict, dict) else type(actual_state_dict)}"
                        )
                    else:
                        tplr.logger.warning(
                            f"Unexpected state_dict structure from UID {uid}"
                        )
                        skipped_uids.append(uid)
                        continue
                else:
                    # Fallback for unexpected format
                    tplr.logger.warning(
                        f"Unexpected response format from UID {uid}: {type(response)}"
                    )
                    skipped_uids.append(uid)
                    continue

                tplr.logger.debug(
                    f"Received state dict and global step {actual_global_step} from UID {uid}"
                )
            except (TypeError, ValueError, AttributeError) as e:
                tplr.logger.debug(f"Invalid response from UID {uid}: {e}")
                skipped_uids.append(uid)
                continue

            if actual_state_dict is None:
                tplr.logger.debug(f"Empty state dict from UID {uid}")
                skipped_uids.append(uid)
                continue

            # Validate the gradient
            if not self._validate_gradient_response(
                actual_state_dict, uid, device, totalks
            ):
                skipped_uids.append(uid)
                continue

            # Process valid tensors
            self._process_gradient_tensors(
                actual_state_dict, uid, device, aggregated_state_dict, metrics
            )

            valid_uids.append(uid)
            global_steps.append(actual_global_step)

        if not valid_uids:
            tplr.logger.info("No valid gradients received from any UID")
            return None

        # Return the lists directly like the original gather method
        return {
            "aggregated_state_dict": aggregated_state_dict,  # Keep as lists
            "valid_uids": valid_uids,
            "global_steps": global_steps,
            "skipped_uids": skipped_uids,
        }

    def _validate_gradient_response(
        self, state_dict_resp: dict, uid: int, device: str, totalks: dict
    ) -> bool:
        """Validate gradient response from a UID."""
        for param_name, tensor in state_dict_resp.items():
            if param_name.endswith("idxs"):
                base_name = param_name[:-4]
                totalk = totalks.get(base_name)
                if totalk is None:
                    tplr.logger.warning(
                        f"Missing totalk for parameter {base_name} from UID {uid}, skipping UID."
                    )
                    return False
                try:
                    self.gradient_manager.check_compressed_indices(
                        param_name,
                        tensor.to(device),
                        totalk,
                        allowed_topk=self.hparams.topk_compression,
                    )
                except Exception as e:
                    tplr.logger.warning(
                        f"Compressed indices check failed for parameter {param_name} from UID {uid}: {e}"
                    )
                    return False
            elif param_name.endswith("vals"):
                tensor_to_check = tensor.to(device)
                if (
                    torch.isnan(tensor_to_check).any()
                    or torch.isinf(tensor_to_check).any()
                ):
                    tplr.logger.warning(
                        f"NaN/Inf in {param_name} from UID {uid}, skipping"
                    )
                    return False
        return True

    def _process_gradient_tensors(
        self,
        state_dict_resp: dict,
        uid: int,
        device: str,
        aggregated_state_dict: dict,
        metrics: dict,
    ) -> None:
        """
        Process and aggregate tensors from a validated gradient response.
        """
        tplr.logger.debug(
            f"Processing gradient tensors for UID {uid}, keys: {list(state_dict_resp.keys())}"
        )

        # Process tensors (with normalization on 'vals' keys).
        for param_name, tensor in state_dict_resp.items():
            if isinstance(tensor, torch.Tensor):
                moved_tensor = tensor.to(device)
                aggregated_state_dict.setdefault(param_name, []).append(moved_tensor)
                metrics["download_bytes"] += tensor.element_size() * tensor.nelement()
                tplr.logger.debug(
                    f"Added tensor {param_name} to aggregated_state_dict on device {moved_tensor.device}"
                )
            elif param_name.endswith("quant_params"):
                # Recursively move all tensors in quant_params to target device
                moved_quant_params = self._move_to_device_recursive(tensor, device)
                aggregated_state_dict.setdefault(param_name, []).append(
                    moved_quant_params
                )
                tplr.logger.debug(
                    f"Added quant_params {param_name} to aggregated_state_dict"
                )

                # Debug: Check what's actually in quant_params
                if isinstance(moved_quant_params, dict):
                    for k, v in moved_quant_params.items():
                        if isinstance(v, torch.Tensor):
                            tplr.logger.debug(f"  quant_params[{k}]: {v.device}")

        tplr.logger.debug(
            f"Processed gradient tensors for UID {uid}, final aggregated keys: {list(aggregated_state_dict.keys())}"
        )

    def _move_to_device_recursive(self, obj, device: str):
        """Recursively move all tensors in a nested structure to the target device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {
                k: self._move_to_device_recursive(v, device) for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return type(obj)(
                self._move_to_device_recursive(item, device) for item in obj
            )
        else:
            return obj

    async def _get_with_retry(
        self,
        uid: str,
        window: int,
        key: str,
        timeout: int,
        local: bool = True,
        time_min: datetime = None,
        time_max: datetime = None,
        **kwargs,
    ) -> Optional[tuple]:  # Change return type to tuple
        """Get gradient with retry logic - returns tuple (state_dict, global_step)"""
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Use the chain manager to get the gradient
                result = await self._get_gradient_from_uid(
                    uid,
                    window,
                    timeout,
                    local=local,
                    time_min=time_min,
                    time_max=time_max,
                    **kwargs,
                )
                if result is not None:
                    # result is now a tuple (state_dict, global_step)
                    state_dict, global_step = result

                    return result

            except Exception as e:
                tplr.logger.debug(
                    f"Attempt {attempt + 1} failed for UID {uid}, window {window}: {e}"
                )

                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    await asyncio.sleep(delay)

        return None

    async def _get_gradient_from_uid(
        self,
        uid: str,
        window: int,
        timeout: int,
        local: bool = True,
        time_min: datetime = None,
        time_max: datetime = None,
        **kwargs,
    ) -> Optional[tuple]:
        """Get gradient from a specific UID - returns tuple (state_dict, global_step)"""
        try:
            # Get the bucket for this UID through the peer manager's chain manager
            bucket = self.peer_manager.chain_manager.get_bucket(int(uid))
            if bucket is None:
                tplr.logger.debug(f"No bucket found for UID {uid}")
                return None

            # Construct the gradient key
            gradient_key = f"gradient-{window}-{uid}-v{tplr.__version__}.pt"

            # Check local storage first if requested
            if local:
                try:
                    local_result = await self._check_local_gradient(
                        uid, window, time_min=time_min, time_max=time_max, **kwargs
                    )
                    if local_result is not None:
                        # Return tuple format for consistency
                        state_dict = local_result.get("state_dict", local_result)
                        global_step = local_result.get("global_step", 0)
                        return (state_dict, global_step)
                except Exception as e:
                    tplr.logger.debug(f"Local gradient check failed for UID {uid}: {e}")
                    # Continue to remote download fallback

            # Download the gradient from remote storage with time constraints
            data = await self.storage_client.get_object(
                gradient_key,
                bucket,
                timeout=timeout,
                time_min=time_min,
                time_max=time_max,
            )

            if data is None:
                return None

            # Save to temp file and load
            temp_file = self.file_manager.create_temp_file("gradient")
            try:
                with open(temp_file, "wb") as f:
                    f.write(data)

                # Load the gradient data
                loaded_data = torch.load(
                    temp_file, map_location=self.device, weights_only=False
                )

                # Handle malformed data gracefully
                if loaded_data is None:
                    return (None, 0)

                # Return tuple format (state_dict, global_step) like the old code
                if isinstance(loaded_data, dict):
                    state_dict = loaded_data.get("state_dict", loaded_data)
                    global_step = loaded_data.get("global_step", 0)
                else:
                    # For non-dict data, use the data itself as state_dict
                    state_dict = loaded_data
                    global_step = 0

                return (state_dict, global_step)
            finally:
                self.file_manager.delete_file(temp_file)

        except Exception as e:
            tplr.logger.error(f"Error getting gradient from UID {uid}: {e}")
            return None

    async def _check_local_gradient(
        self,
        uid: str,
        window: int,
        time_min: datetime = None,
        time_max: datetime = None,
        **kwargs,
    ) -> Optional[dict]:
        """Check for gradient in local storage"""
        try:
            local_path = self.file_manager.get_local_storage_path(
                uid, window, f"gradient-{window}-{uid}-v{tplr.__version__}.pt"
            )

            if not os.path.exists(local_path):
                return None

            # Check file timestamp against time constraints
            if time_min is not None or time_max is not None:
                from datetime import datetime, timezone

                file_mtime = datetime.fromtimestamp(
                    os.path.getmtime(local_path), tz=timezone.utc
                )

                if time_min is not None and file_mtime < time_min:
                    tplr.logger.debug(
                        f"Local gradient from UID {uid} too old: {file_mtime} < {time_min}"
                    )
                    return None
                if time_max is not None and file_mtime > time_max:
                    tplr.logger.debug(
                        f"Local gradient from UID {uid} too new: {file_mtime} > {time_max}"
                    )
                    return None

            # Load the local gradient
            loaded_data = torch.load(local_path, weights_only=False)

            # Add timestamp if not present
            if "timestamp" not in loaded_data:
                loaded_data["timestamp"] = datetime.fromtimestamp(
                    os.path.getmtime(local_path), tz=timezone.utc
                )

            tplr.logger.debug(
                f"Found valid local gradient for UID {uid}, window {window}"
            )
            return loaded_data

        except Exception as e:
            tplr.logger.debug(f"Error checking local gradient for UID {uid}: {e}")
            return None

    async def load_aggregation(
        self, window: int, chunk_size: int = 50_000_000
    ) -> Optional[dict]:
        """
        Load aggregated gradients for a specified window from the aggregation server.

        Args:
            window: Window number to load
            chunk_size: Size of each chunk for multipart download (default 50MB)

        Returns:
            Processed aggregation data or None if failed
        """
        try:
            bucket_config = BUCKET_SECRETS["aggregator"]
            credentials = bucket_config["credentials"]["read"]

            # Create a Bucket object using specified credentials
            bucket = Bucket(
                name=bucket_config["name"],
                account_id=bucket_config["account_id"],
                access_key_id=credentials["access_key_id"],
                secret_access_key=credentials["secret_access_key"],
            )

            filename = f"aggregator-{window}-v{__version__}.pt"

            tplr.logger.info(f"Attempting to download aggregation file: {filename}")

            # Get file size
            file_size = await self.storage_client.get_object_size(filename, bucket)
            if file_size is None:
                tplr.logger.warning(f"No aggregation file found for window {window}")
                return None

            # For small files, use the existing get_object method
            if file_size <= 100 * 1024 * 1024:  # 100MB threshold
                tplr.logger.info(
                    f"File size {file_size} bytes, using standard download"
                )
                data = await self.storage_client.get_object(
                    filename, bucket, timeout=45
                )
                if data is None:
                    tplr.logger.warning(f"Failed to download {filename}")
                    return None

                # Save to temp file and load
                temp_file = self.file_manager.create_temp_file("aggregation_load")
                with open(temp_file, "wb") as f:
                    f.write(data)

                result = torch.load(
                    temp_file, map_location=self.device, weights_only=False
                )
                self.file_manager.delete_file(temp_file)

                tplr.logger.info(
                    f"Successfully loaded aggregation data for window {window}"
                )
                return result

            # For large files, use multipart download
            tplr.logger.info(f"File size {file_size} bytes, using multipart download")

            temp_file = self.file_manager.create_temp_file("aggregation_large")
            success = await self.storage_client.multipart_download(
                filename, temp_file, bucket
            )

            if not success:
                tplr.logger.error(
                    f"Failed to download large aggregation file {filename}"
                )
                return None

            # Load the data
            result = torch.load(temp_file, map_location=self.device, weights_only=False)
            self.file_manager.delete_file(temp_file)

            tplr.logger.info(
                f"Successfully loaded large aggregation data for window {window}"
            )
            return result

        except Exception as e:
            tplr.logger.error(
                f"Error loading aggregation file for window {window}: {e}"
            )
            return None

    async def gather_window_batch(
        self,
        batch_windows: List[int],
        uid: str,
        peers: List[int],
        device: str,
        totalks: dict,
        global_step: int,
    ) -> Dict[int, SimpleNamespace]:
        """Gather gradients for multiple windows in parallel."""
        try:
            gather_tasks = [
                self.gather_gradients(
                    my_uid=int(uid),
                    uids=peers,
                    window=w,
                    timeout=30,
                    device=device,
                    totalks=totalks,
                    local=False,
                )
                for w in batch_windows
            ]

            # Wait for all gather tasks to complete
            batch_results = await asyncio.gather(*gather_tasks, return_exceptions=True)

            # Filter out exceptions and create window->result mapping
            result_dict = {w: None for w in batch_windows}  # Initialize with None
            for window, result in zip(batch_windows, batch_results):
                if not isinstance(result, Exception) and result is not None:
                    result_dict[window] = result

            return result_dict

        except Exception as e:
            tplr.logger.error(
                f"Failed to gather window batch {batch_windows}: {str(e)}"
            )
            return {
                w: None for w in batch_windows
            }  # Return dict with None values on failure
