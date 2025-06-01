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
        device: str
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
    ) -> Optional[SimpleNamespace]:
        """
        Gather gradients from multiple UIDs for a specific window.
        
        Args:
            my_uid: Current node's UID
            uids: List of UIDs to gather from
            window: Window number
            timeout: Timeout for individual operations
            device: Device to load tensors on
            totalks: Dictionary of total k values for validation
            local: Whether to check local storage first
            time_min: Minimum timestamp for gradient validity
            time_max: Maximum timestamp for gradient validity
            
        Returns:
            SimpleNamespace containing aggregated gradients or None if failed
        """
        async with self.gather_semaphore:
            try:
                tplr.logger.info(f"Gathering gradients for window {window} from {len(uids)} UIDs")
                
                # Collect gradients from all UIDs
                gathered_gradients = []
                successful_uids = []
                
                # Create tasks for concurrent gathering
                gather_tasks = []
                for uid in uids:
                    task = asyncio.create_task(
                        self._get_with_retry(
                            str(uid), window, "gradient", timeout,
                            local=local, time_min=time_min, time_max=time_max
                        )
                    )
                    gather_tasks.append((uid, task))
                
                # Wait for all tasks to complete
                for uid, task in gather_tasks:
                    try:
                        result = await task
                        if result is not None:
                            # Validate the gradient
                            state_dict = result.get("state_dict", {})
                            if self.gradient_manager.validate_gradient(state_dict, totalks):
                                gathered_gradients.append(result)
                                successful_uids.append(uid)
                            else:
                                tplr.logger.warning(f"Invalid gradient from UID {uid}")
                        else:
                            tplr.logger.debug(f"No gradient received from UID {uid}")
                    except Exception as e:
                        tplr.logger.error(f"Error gathering from UID {uid}: {e}")
                
                if not gathered_gradients:
                    tplr.logger.warning(f"No valid gradients gathered for window {window}")
                    return None
                
                tplr.logger.info(f"Successfully gathered {len(gathered_gradients)} gradients from UIDs: {successful_uids}")
                
                # Aggregate the gradients
                aggregated_result = self._aggregate_gradients(gathered_gradients, device)
                
                return aggregated_result
                
            except Exception as e:
                tplr.logger.error(f"Error in gather_gradients for window {window}: {e}")
                return None

    async def _get_with_retry(
        self, uid: str, window: int, key: str, timeout: int, **kwargs
    ) -> Optional[dict]:
        """Get gradient with retry logic"""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # TODO: This should use the actual get method from the main Comms class
                # For now, we'll implement a basic version
                result = await self._get_gradient_from_uid(uid, window, timeout, **kwargs)
                if result is not None:
                    return result
                    
            except Exception as e:
                tplr.logger.debug(f"Attempt {attempt + 1} failed for UID {uid}, window {window}: {e}")
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        return None

    async def _get_gradient_from_uid(
        self, uid: str, window: int, timeout: int, local: bool = True, **kwargs
    ) -> Optional[dict]:
        """Get gradient from a specific UID"""
        try:
            # TODO: This needs to be implemented with proper bucket resolution
            # For now, return None as placeholder
            tplr.logger.debug(f"Getting gradient from UID {uid} for window {window}")
            return None
            
        except Exception as e:
            tplr.logger.error(f"Error getting gradient from UID {uid}: {e}")
            return None

    def _aggregate_gradients(self, gradients: List[dict], device: str) -> SimpleNamespace:
        """Aggregate multiple gradients into a single result"""
        try:
            if not gradients:
                return SimpleNamespace(state_dict={})
            
            # Initialize aggregated state dict
            aggregated_state = {}
            
            # Get parameter names from first gradient
            first_gradient = gradients[0]
            state_dict = first_gradient.get("state_dict", {})
            
            # Aggregate each parameter
            for param_name in state_dict.keys():
                if param_name.endswith("idxs") or param_name.endswith("vals"):
                    # Collect all values for this parameter
                    param_values = []
                    for gradient in gradients:
                        grad_state = gradient.get("state_dict", {})
                        if param_name in grad_state:
                            param_values.append(grad_state[param_name])
                    
                    if param_values:
                        # For now, just take the first value
                        # TODO: Implement proper aggregation logic
                        aggregated_state[param_name] = param_values[0]
            
            # Create result namespace
            result = SimpleNamespace()
            result.state_dict = SimpleNamespace(**aggregated_state)
            
            return result
            
        except Exception as e:
            tplr.logger.error(f"Error aggregating gradients: {e}")
            return SimpleNamespace(state_dict={})

    async def load_aggregation(self, window: int, chunk_size: int = 50_000_000) -> Optional[dict]:
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
                tplr.logger.info(f"File size {file_size} bytes, using standard download")
                data = await self.storage_client.get_object(filename, bucket, timeout=45)
                if data is None:
                    tplr.logger.warning(f"Failed to download {filename}")
                    return None
                
                # Save to temp file and load
                temp_file = self.file_manager.create_temp_file("aggregation_load")
                with open(temp_file, "wb") as f:
                    f.write(data)
                
                result = torch.load(temp_file, map_location=self.device, weights_only=False)
                self.file_manager.delete_file(temp_file)
                
                tplr.logger.info(f"Successfully loaded aggregation data for window {window}")
                return result

            # For large files, use multipart download
            tplr.logger.info(f"File size {file_size} bytes, using multipart download")
            
            temp_file = self.file_manager.create_temp_file("aggregation_large")
            success = await self.storage_client.multipart_download(filename, temp_file, bucket)
            
            if not success:
                tplr.logger.error(f"Failed to download large aggregation file {filename}")
                return None
            
            # Load the data
            result = torch.load(temp_file, map_location=self.device, weights_only=False)
            self.file_manager.delete_file(temp_file)
            
            tplr.logger.info(f"Successfully loaded large aggregation data for window {window}")
            return result

        except Exception as e:
            tplr.logger.error(f"Error loading aggregation file for window {window}: {e}")
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
            tplr.logger.error(f"Failed to gather window batch {batch_windows}: {str(e)}")
            return {w: None for w in batch_windows}  # Return dict with None values on failure

    # TODO: Add gradient aggregation algorithms (averaging, weighted averaging, etc.)
    # TODO: Add gradient compression/decompression during aggregation
    # TODO: Add aggregation quality metrics
    # TODO: Add Byzantine fault tolerance for aggregation
    # TODO: Add aggregation caching mechanisms 