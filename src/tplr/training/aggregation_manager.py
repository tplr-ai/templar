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
from datetime import datetime, timezone
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
        uids: list[int],
        window: int,
        timeout: int,
        device: str,
        totalks: dict,
        local: bool = False,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        stale_retention: int = 10,
        show_progress: bool = False,
    ) -> SimpleNamespace | None:
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
        gather_phase_start = time.time()
        async with self.gather_semaphore:
            batch_tasks = [
                self._get_with_retry(
                    uid,
                    window,
                    "gradient",
                    timeout,
                    local=local,
                    time_min=time_min,
                    time_max=time_max,
                    show_progress=show_progress,
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

        gather_phase_time = time.time() - gather_phase_start

        # Aggregate the responses
        aggregation_phase_start = time.time()
        aggregation_result = self._aggregate_gradients(
            raw_responses, device, totalks, metrics
        )
        aggregation_phase_time = time.time() - aggregation_phase_start

        if aggregation_result is None:
            return None

        total_time = time.time() - start_time
        
        # Convert bytes to MB for more readable logging
        upload_mb = metrics['upload_bytes'] / (1024 * 1024)
        download_mb = metrics['download_bytes'] / (1024 * 1024)
        
        tplr.logger.info(
            f"Gather done in {total_time:.2f}s (gather: {gather_phase_time:.2f}s, aggregation: {aggregation_phase_time:.2f}s). "
            f"Success rate: {len(aggregation_result['valid_uids'])}/{len(uids)}, "
            f"Upload: {upload_mb:.2f} MB, Download: {download_mb:.2f} MB"
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
        self, raw_responses: list[dict], device: str, totalks: dict, metrics: dict
    ) -> dict | None:
        """
        Aggregate gradients from raw responses with validation.
        Matches original gather method exactly with timing instrumentation.
        """
        aggregation_start = time.time()
        aggregated_state_dict = {}
        valid_uids = []
        skipped_uids = []
        global_steps = []

        for response_data in raw_responses:
            uid = response_data["uid"]
            response = response_data["response"]
            uid_start = time.time()

            if response_data["is_exception"]:
                tplr.logger.debug(f"Error from UID {uid}: {str(response)}")
                skipped_uids.append(uid)
                continue

            if response is None:
                tplr.logger.info(f"Skipped UID {uid} - gradient not found.")
                skipped_uids.append(uid)
                continue

            try:
                # Handle tuple format (state_dict, global_step)
                if isinstance(response, tuple) and len(response) == 2:
                    state_dict_resp, global_step_resp = response
                    tplr.logger.debug(f"Received state dict and global step {global_step_resp} from UID {uid}")
                else:
                    tplr.logger.warning(f"Unexpected response format from UID {uid}: {type(response)}")
                    skipped_uids.append(uid)
                    continue
            except (TypeError, ValueError, AttributeError) as e:
                tplr.logger.debug(f"Invalid response from UID {uid}: {e}")
                skipped_uids.append(uid)
                continue

            if state_dict_resp is None:
                tplr.logger.debug(f"Empty state dict from UID {uid}")
                skipped_uids.append(uid)
                continue

            # ---------- Begin Compressed Indices and Values Check (CPU OPTIMIZED) ----------
            validation_start = time.time()
            valid_response = True
            for param_name, tensor in state_dict_resp.items():
                if param_name.endswith("idxs"):
                    base_name = param_name[:-4]
                    totalk = totalks.get(base_name)
                    if totalk is None:
                        tplr.logger.warning(f"Missing totalk for parameter {base_name} from UID {uid}, skipping UID.")
                        valid_response = False
                        break
                    try:
                        # Time the expensive check_compressed_indices call - KEEP ON CPU
                        check_start = time.time()
                        self.gradient_manager.check_compressed_indices(
                            param_name,
                            tensor,  # Keep on CPU for validation - much faster
                            totalk,
                            allowed_topk=self.hparams.topk_compression,
                        )
                        check_time = time.time() - check_start
                        if check_time > 0.1:  # Log if > 100ms
                            tplr.logger.info(f"UID {uid} {param_name} check_compressed_indices took {check_time:.2f}s")
                    except Exception as e:
                        tplr.logger.warning(f"Compressed indices check failed for parameter {param_name} from UID {uid}: {e}")
                        valid_response = False
                        break
                elif param_name.endswith("vals"):
                    # Time the NaN/Inf check - KEEP ON CPU
                    nan_check_start = time.time()
                    # CPU NaN/Inf check - much faster than GPU
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        tplr.logger.warning(f"NaN/Inf in {param_name} from UID {uid}, skipping")
                        valid_response = False
                        break
                    nan_check_time = time.time() - nan_check_start
                    if nan_check_time > 0.1:  # Log if > 100ms
                        tplr.logger.info(f"UID {uid} {param_name} NaN/Inf check took {nan_check_time:.2f}s")

            validation_time = time.time() - validation_start
            # ---------- End Compressed Indices and Values Check ----------

            # If any check failed, skip this UID entirely
            if not valid_response:
                tplr.logger.info(f"Skipping UID {uid} due to validation failures")
                skipped_uids.append(uid)
                continue

            # Process tensors (EXACTLY like old code)
            processing_start = time.time()
            for param_name, tensor in state_dict_resp.items():
                if isinstance(tensor, torch.Tensor):
                    # Time the tensor.to(device) call
                    device_start = time.time()
                    moved_tensor = tensor.to(device)
                    device_time = time.time() - device_start
                    if device_time > 0.1:  # Log if > 100ms
                        tplr.logger.info(f"UID {uid} {param_name} tensor.to(device) took {device_time:.2f}s")
                    
                    aggregated_state_dict.setdefault(param_name, []).append(moved_tensor)
                    metrics["download_bytes"] += tensor.element_size() * tensor.nelement()
                elif param_name.endswith("quant_params"):
                    # Exactly like old code - just append tensor, no recursive device moving
                    aggregated_state_dict.setdefault(param_name, []).append(tensor)

            processing_time = time.time() - processing_start
            
            valid_uids.append(uid)
            global_steps.append(global_step_resp)
            
            uid_total = time.time() - uid_start
            tplr.logger.info(f"UID {uid} aggregation: validation={validation_time:.2f}s, processing={processing_time:.2f}s, total={uid_total:.2f}s")

        if not valid_uids:
            tplr.logger.info("No valid gradients received from any UID")
            return None

        total_aggregation_time = time.time() - aggregation_start
        tplr.logger.info(f"Total aggregation completed in {total_aggregation_time:.2f}s")

        return {
            "aggregated_state_dict": aggregated_state_dict,
            "valid_uids": valid_uids,
            "global_steps": global_steps,
            "skipped_uids": skipped_uids,
        }

    def _move_to_device_recursive(
        self, obj: torch.Tensor | dict | list | tuple, device: str
    ) -> torch.Tensor | dict | list | tuple:
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
        uid: int,
        window: int,
        key: str,
        timeout: int,
        local: bool = False,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        show_progress: bool = False,
        **kwargs,
    ) -> tuple | None:
        """Get gradient with retry logic - returns tuple (state_dict, global_step)"""
        max_retries = 3
        base_delay = 1.0
        uid_start_time = time.time()

        for attempt in range(max_retries):
            attempt_start = time.time()
            try:
                # Use the chain manager to get the gradient
                result = await self._get_gradient_from_uid(
                    str(uid),
                    window,
                    timeout,
                    local=local,
                    time_min=time_min,
                    time_max=time_max,
                    show_progress=show_progress,
                    **kwargs,
                )
                if result is not None:
                    total_time = time.time() - uid_start_time
                    attempt_time = time.time() - attempt_start
                    tplr.logger.info(
                        f"UID {uid} successful on attempt {attempt + 1} in {attempt_time:.2f}s (total: {total_time:.2f}s)"
                    )
                    return result

            except Exception as e:
                attempt_time = time.time() - attempt_start
                tplr.logger.info(
                    f"Attempt {attempt + 1} failed for UID {uid}, window {window} in {attempt_time:.2f}s: {e}"
                )

                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    tplr.logger.info(f"UID {uid} sleeping {delay}s before retry")
                    await asyncio.sleep(delay)

        total_time = time.time() - uid_start_time
        tplr.logger.info(f"UID {uid} failed all {max_retries} attempts in {total_time:.2f}s")
        return None

    async def _get_gradient_from_uid(
        self,
        uid: str,
        window: int,
        timeout: int,
        local: bool = False,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        show_progress: bool = False,
        **kwargs,
    ) -> tuple | None:
        """Get gradient from a specific UID - returns tuple (state_dict, global_step)"""
        try:
            # Get the bucket for this UID through the peer manager's chain manager
            bucket = self.peer_manager.chain_manager.get_bucket(int(uid))
            if bucket is None:
                tplr.logger.info(f"No bucket found for UID {uid}")
                return None

            # Construct the gradient key
            gradient_key = f"gradient-{window}-{uid}-v{tplr.__version__}.pt"

            # Check local storage first ONLY if explicitly requested
            if local:
                local_start = time.time()
                try:
                    local_result = await self._check_local_gradient(
                        uid, window, time_min=time_min, time_max=time_max, **kwargs
                    )
                    local_time = time.time() - local_start
                    if local_result is not None:
                        tplr.logger.info(f"UID {uid} found local gradient in {local_time:.2f}s")
                        # Return tuple format for consistency
                        state_dict = local_result.get("state_dict", local_result)
                        global_step = local_result.get("global_step", 0)
                        return (state_dict, global_step)
                    else:
                        tplr.logger.info(f"UID {uid} no local gradient found in {local_time:.2f}s")
                except Exception as e:
                    local_time = time.time() - local_start
                    tplr.logger.info(f"Local gradient check failed for UID {uid} in {local_time:.2f}s: {e}")
                    # Continue to remote download fallback

            # Download the gradient from remote storage with time constraints
            remote_start = time.time()
            data = await self.storage_client.get_object(
                gradient_key,
                bucket,
                timeout=timeout,
                time_min=time_min,
                time_max=time_max,
                show_progress=show_progress,
            )
            remote_download_time = time.time() - remote_start

            if data is None:
                tplr.logger.info(f"UID {uid} remote download failed in {remote_download_time:.2f}s")
                return None

            tplr.logger.info(f"UID {uid} remote download completed in {remote_download_time:.2f}s")

            # Save to temp file and load
            processing_start = time.time()
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

                processing_time = time.time() - processing_start
                tplr.logger.info(f"UID {uid} gradient processing completed in {processing_time:.2f}s")
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
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        **kwargs,
    ) -> dict | None:
        """Check for gradient in local storage"""
        try:
            local_path = self.file_manager.get_local_storage_path(
                uid, window, f"gradient-{window}-{uid}-v{tplr.__version__}.pt"
            )

            if not os.path.exists(local_path):
                return None

            # Check file timestamp against time constraints
            if time_min is not None or time_max is not None:
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

    async def apply_catchup_aggregations(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler.LRScheduler, 
        start_catchup_window: int, 
        target_window: int, 
        metadata_manager, 
        chain_manager
    ) -> int:
        """
        Apply catchup aggregations from start_catchup_window to target_window.
        Implements logic from neuron_utils.catchup_with_aggregation_server.
        
        Args:
            model: The model to apply gradients to
            optimizer: The optimizer for stepping
            scheduler: The scheduler for stepping
            start_catchup_window: Starting window for catchup
            target_window: Target window to reach
            metadata_manager: MetadataManager for debug info
            chain_manager: ChainManager for current window updates
            
        Returns:
            Last successfully processed window number
        """
        tplr.logger.info("Starting catchup with aggregation server...")
        
        current_step = start_catchup_window + 1
        
        tplr.logger.info(
            f"Catching up from window {current_step} to current window {target_window}"
        )
        
        # If no catchup is needed, return the target window
        if current_step >= target_window:
            tplr.logger.info(f"No catchup needed - already at target window {target_window}")
            return target_window
        
        while current_step < target_window:
            tplr.logger.info(
                f"\nProcessing catchup for window {current_step} (Target: {target_window})"
            )
            
            # Load aggregation for current window
            agg_data = await self.load_aggregation(window=current_step)
            
            # For the last window in catchup, we might need to retry a few times
            if agg_data is None and current_step == target_window - 1:
                max_retries = 7
                retry_count = 0
                retry_delay = 10
                
                tplr.logger.info(
                    f"No aggregation for latest window {current_step}, will retry up to {max_retries} times"
                )
                
                while retry_count < max_retries and agg_data is None:
                    retry_count += 1
                    tplr.logger.info(
                        f"Retry {retry_count}/{max_retries} for window {current_step}"
                    )
                    await asyncio.sleep(retry_delay)
                    
                    # Try to load aggregation again
                    agg_data = await self.load_aggregation(window=current_step)
                    
                    if agg_data is not None:
                        tplr.logger.info(
                            f"Successfully loaded aggregation on retry {retry_count}"
                        )
                
                if agg_data is None:
                    tplr.logger.warning(
                        f"Failed to load aggregation after {max_retries} retries"
                    )
            
            # Process the aggregation data if available
            if agg_data:
                update_start = time.time()
                
                # Process the loaded data using helper from neuron_utils
                from ..neurons.neuron_utils import process_loaded_data
                processed_agg_data = process_loaded_data(model, agg_data)
                
                if processed_agg_data is not None:
                    # Get learning rate for this step
                    lr = scheduler.get_last_lr()[0]
                    weight_decay = self.hparams.weight_decay
                    
                    # Apply the gradients to the model parameters
                    for name, param in model.named_parameters():
                        if name in processed_agg_data["tensors"]:
                            # Apply weight decay to the parameter manually if needed
                            if weight_decay > 0:
                                with torch.no_grad():
                                    param.data.mul_(1.0 - lr * weight_decay)
                            
                            # Move aggregation tensor to device
                            agg_tensor = processed_agg_data["tensors"][name].to(self.device)
                            
                            # Set the gradient instead of directly updating the parameter
                            if param.grad is None:
                                param.grad = agg_tensor
                            else:
                                param.grad.copy_(agg_tensor)
                            
                            del agg_tensor
                            torch.cuda.empty_cache()
                    
                    tplr.logger.info(
                        f"Window {current_step} - Set gradients in {time.time() - update_start:.2f}s"
                    )
                    
                    # Let the optimizer handle the parameter updates
                    optimizer.step()
                    scheduler.step()
                    torch.cuda.empty_cache()
                    
                    tplr.logger.info(
                        f"Successfully applied aggregation for window {current_step}"
                    )
                    
                    # Get debug dict and compare with current model parameters
                    debug_dict_result = await metadata_manager.get_debug_dict(current_step)
                    if (
                        isinstance(debug_dict_result, dict)
                        and "state_dict" in debug_dict_result
                    ):
                        debug_state_dict = debug_dict_result["state_dict"]
                        
                        # Use comparison function from neuron_utils
                        from ..neurons.neuron_utils import compare_model_with_debug_dict
                        comparison_metrics = await compare_model_with_debug_dict(
                            model=model,
                            debug_dict=debug_state_dict,
                            learning_rate=lr,
                        )
                        
                        if comparison_metrics["success"]:
                            # Log the comparison metrics
                            tplr.logger.info(
                                f"Window {current_step} - L2 norm difference between model and debug values: "
                                f"{comparison_metrics['l2_norm']}"
                            )
                            tplr.logger.info(
                                f"Window {current_step} - Average L2 norm per parameter: "
                                f"{comparison_metrics['avg_l2_norm']}"
                            )
                            tplr.logger.info(
                                f"Window {current_step} - Average absolute difference per parameter: "
                                f"{comparison_metrics['avg_abs_diff']}"
                            )
                            tplr.logger.info(
                                f"Window {current_step} - Average steps behind: "
                                f"{comparison_metrics['avg_steps_behind']}"
                            )
                        else:
                            tplr.logger.warning(
                                f"Failed to compare model with debug dict for window {current_step}"
                            )
                    else:
                        tplr.logger.warning(
                            f"Invalid debug dict format for window {current_step}"
                        )
                else:
                    tplr.logger.warning(
                        f"Failed to process aggregation data for window {current_step}"
                    )
                    # Still advance the optimizer and scheduler
                    optimizer.step()
                    scheduler.step()
                
                del processed_agg_data
                torch.cuda.empty_cache()
            else:
                tplr.logger.warning(f"No aggregation data found for window {current_step}")
                # Don't advance the optimizer and scheduler
            
            # Move to next window
            current_step += 1
            
            # Check if current_window has changed during processing
            if hasattr(chain_manager, 'current_window') and chain_manager.current_window > target_window:
                target_window = chain_manager.current_window
                tplr.logger.info(
                    f"Current window advanced during catchup, new target: {target_window}"
                )
        
        # Return the last window we actually processed, not current_step - 1
        last_processed = current_step - 1
        tplr.logger.info(f"Catchup complete. Last processed window: {last_processed}")
        return last_processed

    async def load_aggregation(
        self, window: int, chunk_size: int = 50_000_000
    ) -> dict | None:
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
                filename, temp_file, bucket, show_progress=False
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
