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
import concurrent.futures
import os
import re
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Tuple
import bittensor as bt
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import SequentialLR
from transformers.models.llama import LlamaForCausalLM

import tplr as tplr
from . import __version__
from .chain import ChainManager
from .compress import CompressDCT, TransformDCT
from .config import BUCKET_SECRETS
from .schemas import Bucket

# Import all the new managers
from .storage.client import StorageClient
from .storage.file_manager import FileManager
from .training.gradient_manager import GradientManager
from .training.checkpoint_manager import CheckpointManager
from .training.aggregation_manager import AggregationManager
from .network.peer_manager import PeerManager
from .protocol.metadata_manager import MetadataManager

# Constants
CPU_COUNT = os.cpu_count() or 4


class Comms:
    """Main orchestration layer for distributed training communications"""

    def __init__(
        self,
        chain_manager: ChainManager,
        wallet: "bt.wallet | None",
        hparams=None,
        key_prefix: str = "model",
        uid=None,
    ):
        """
        Initialize Comms with composition over inheritance.
        
        Args:
            chain_manager: ChainManager instance for blockchain interactions
            wallet: Wallet for signing transactions
            hparams: Hyperparameters for configuration
            key_prefix: Prefix for model keys
            uid: Unique identifier for this neuron
        """
        self.chain_manager = chain_manager
        self.uid = uid
        self.wallet = wallet
        self.hparams = hparams or chain_manager.hparams

        # Create temp directory for this instance
        self.temp_dir = os.path.join("/tmp", f"templar_{self.uid}")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Get the bucket directly
        self.bucket = self.get_own_bucket("gradients", "write")

        # Use the hotkey directly in the save_location
        if self.wallet is not None:
            hotkey = self.wallet.hotkey.ss58_address
            self.save_location = os.path.join("/tmp", f"hotkey_{hotkey}")
            os.makedirs(self.save_location, exist_ok=True)
        self.key_prefix = key_prefix

        # Initialize all managers
        self._initialize_managers()

        # Background task management
        self.loop = None
        self.executor = None

    def _initialize_managers(self):
        """Initialize all manager components"""
        # Foundation components
        self.storage_client = StorageClient(self.temp_dir)
        self.file_manager = FileManager(self.temp_dir, self.uid)

        # Domain-specific managers
        self.gradient_manager = GradientManager(
            self.storage_client,
            self.file_manager,
            self.chain_manager.config.device if hasattr(self.chain_manager.config, "device") else "cpu",
            self.hparams,
        )

        self.checkpoint_manager = CheckpointManager(
            self.storage_client,
            self.file_manager,
            self.bucket,
            self.uid if self.uid is not None else 0,
            self.chain_manager.metagraph,
            self.chain_manager.commitments,
        )

        self.peer_manager = PeerManager(
            self.chain_manager,
            self.storage_client,
            self.hparams,
        )

        # Aggregation and metadata managers
        self.aggregation_manager = AggregationManager(
            self.gradient_manager,
            self.peer_manager,
            self.storage_client,
            self.file_manager,
            self.hparams,
            self.chain_manager.config.device if hasattr(self.chain_manager.config, "device") else "cpu",
        )

        self.metadata_manager = MetadataManager(
            self.storage_client,
            self.file_manager,
            self.chain_manager,
            __version__,
            self.bucket,
        )

    def get_own_bucket(
        self,
        bucket_type: Literal["gradients", "dataset", "aggregator"],
        access_type=None,
    ) -> Bucket:
        """Gets bucket configuration from environment variables via config.BUCKET_SECRETS."""
        try:
            if bucket_type not in ["gradients", "dataset", "aggregator"]:
                raise ValueError("bucket_type must be either 'gradients' or 'dataset'")

            if bucket_type in ["gradients", "aggregator"]:
                if access_type not in ["read", "write"]:
                    raise ValueError(
                        f"For {bucket_type} bucket, access_type must be either 'read' or 'write'"
                    )

                bucket_config = BUCKET_SECRETS[bucket_type]
                credentials = bucket_config["credentials"][access_type]
            else:  # dataset bucket
                bucket_config = BUCKET_SECRETS["dataset"]
                credentials = bucket_config["credentials"]["read"]

            # Create a Bucket object using specified credentials
            bucket = Bucket(
                name=bucket_config["name"],
                account_id=bucket_config["account_id"],
                access_key_id=credentials["access_key_id"],
                secret_access_key=credentials["secret_access_key"],
            )

            tplr.logger.debug(
                f"Created {bucket_type} bucket with {'read/write' if bucket_type == 'dataset' else access_type} access: {bucket}"
            )
            return bucket

        except KeyError as e:
            tplr.logger.error(f"Missing required R2 configuration: {e}")
            raise
        except Exception as e:
            tplr.logger.error(f"Error creating bucket: {e}")
            raise

    def start_background_tasks(self):
        """Start background tasks and thread pool"""
        self.loop = asyncio.get_running_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
        self.loop.set_default_executor(self.executor)

        # Start peer tracking
        asyncio.create_task(self.peer_manager.start_peer_tracking())

    async def close_all_resources(self) -> None:
        """Close all resources and cleanup"""
        try:
            # Stop peer tracking
            await self.peer_manager.stop_peer_tracking()

            # Close storage clients
            await self.storage_client.close_all_clients()

            # Cleanup temp files
            await self.file_manager.cleanup_temp_files()

            # Close executor
            if self.executor:
                self.executor.shutdown(wait=True)

            tplr.logger.info("All resources closed successfully")

        except Exception as e:
            tplr.logger.error(f"Error closing resources: {e}")

    # High-level interface methods (delegate to appropriate managers)
    async def put(
        self,
        state_dict: dict,
        uid: int | None,
        window: int,
        key: Literal["checkpoint", "debug", "gradient", "aggregator"],
        global_step: int = 0,
        local: bool = True,
        stale_retention: int = 10,
    ) -> float:
        """
        Saves the data locally or uploads to S3, then cleans up stale files.
        """
        if key == "aggregator":
            filename = f"{key}-{window}-v{__version__}.pt"
        else:
            filename = f"{key}-{window}-{uid}-v{__version__}.pt"

        tplr.logger.debug(f"PUT {filename} -->")
        put_start: float = tplr.T()

        try:
            # Create temp file and serialize
            temp_file_path = await self.gradient_manager.serialize_gradient(
                state_dict, global_step
            )

            if local:
                # Local storage
                await self.file_manager.cleanup_local_data(
                    uid=str(uid) if uid is not None else "0",
                    current_window=window,
                    stale_retention=stale_retention,
                )
                local_dir = self.file_manager.get_local_storage_path(
                    str(uid) if uid is not None else "0", window, ""
                )
                self.file_manager.ensure_directory_exists(local_dir)
                final_path = os.path.join(local_dir, filename)

                import shutil

                shutil.move(temp_file_path, final_path)
            else:
                # Remote storage
                with open(temp_file_path, "rb") as f:
                    data = f.read()

                success = await self.storage_client.put_object(
                    filename, data, self.bucket
                )
                self.file_manager.delete_file(temp_file_path)

                if not success:
                    tplr.logger.error(f"Failed to upload {filename}")
                    return 0.0

                # Cleanup old data in background
                asyncio.create_task(self._cleanup_s3_data(uid, window, stale_retention))

        except Exception as e:
            tplr.logger.error(f"Error in put operation: {e}")
            return 0.0

        put_end: float = tplr.T()
        duration: float = put_end - put_start
        tplr.logger.info(f"{tplr.P(window, duration)} PUT {filename} <--")
        return duration

    async def get(
        self,
        uid: str,
        window: int,
        key: str,
        local: bool = True,
        stale_retention: int = 10,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
    ) -> tuple[dict, int | None] | None:
        """GET operation."""
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        tplr.logger.debug(f"GET {filename} -->")

        try:
            if local:
                # Local storage
                await self.file_manager.cleanup_local_data(
                    uid=uid, current_window=window, stale_retention=stale_retention
                )
                local_path = self.file_manager.get_local_storage_path(
                    uid, window, filename
                )

                if not os.path.exists(local_path):
                    tplr.logger.debug(f"Local file not found: {local_path}")
                    return None

                (
                    state_dict,
                    global_step,
                ) = await self.gradient_manager.deserialize_gradient(local_path)

                if key == "checkpoint":
                    return state_dict, None
                return state_dict, global_step
            else:
                # Remote storage
                peer_bucket = self.chain_manager.commitments.get(int(uid))
                if not peer_bucket:
                    return None

                # Use storage client to get object with time constraints
                data = await self.storage_client.get_object(
                    filename,
                    peer_bucket,
                    timeout=15,
                    time_min=time_min,
                    time_max=time_max,
                )
                if data is None:
                    return None

                # Check for time markers
                if isinstance(data, dict) and data.get("__status") in [
                    "TOO_LATE",
                    "TOO_EARLY",
                ]:
                    return data  # type: ignore

                # Save to temp file and deserialize
                temp_file = self.file_manager.create_temp_file("get_temp")
                with open(temp_file, "wb") as f:
                    f.write(data)

                (
                    state_dict,
                    global_step,
                ) = await self.gradient_manager.deserialize_gradient(temp_file)
                self.file_manager.delete_file(temp_file)

                if key == "checkpoint":
                    return state_dict, None
                return state_dict, global_step

        except Exception as e:
            tplr.logger.debug(f"GET error {filename}: {e}")
            return None
        finally:
            tplr.logger.debug(f"GET {filename} <--")

    async def gather(
        self,
        my_uid: int | None,
        uids: List[int] | None = None,
        window: int = None,
        timeout: int = None,
        device: str = None,
        totalks: dict = None,
        local: bool = True,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        show_progress: bool = False,
    ) -> SimpleNamespace | None:
        """
        Gather operation - delegate to aggregation manager.
        Can use internal gather target peers if uids not provided.
        """
        if my_uid is None:
            return None
            
        # Use internal gather target peers if uids not provided
        if uids is None:
            uids = self.peer_manager.gather_target_peers
            if not uids:
                tplr.logger.warning("No gather target peers available and no uids provided")
                return None
                
        # Use current window if not provided
        if window is None:
            window = self.chain_manager.current_window
            
        # Default values for other parameters
        if timeout is None:
            timeout = getattr(self.hparams, 'gather_timeout', 30)
        if device is None:
            device = getattr(self.chain_manager.config, 'device', 'cpu')
        if totalks is None:
            totalks = {}
            
        return await self.aggregation_manager.gather_gradients(
            my_uid=my_uid,
            uids=uids,
            window=window,
            timeout=timeout,
            device=device,
            totalks=totalks,
            local=local,
            time_min=time_min,
            time_max=time_max,
            show_progress=show_progress,
        )

    # Backward compatibility methods (delegate to managers)
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
        """Save checkpoint - delegate to checkpoint manager"""
        return await self.checkpoint_manager.save_checkpoint(
            model,
            optimizer,
            scheduler,
            global_step,
            current_window,
            start_window,
            sync_window,
        )

    async def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        current_window: int,
        device: str,
        init_version: Optional[str] = None,
    ) -> tuple[bool, int, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Load checkpoint - delegate to checkpoint manager"""
        return await self.checkpoint_manager.load_checkpoint(
            model, optimizer, scheduler, current_window, device, init_version
        )

    # Peer management methods
    async def is_miner_active(self, uid: int, recent_windows: int = 3) -> bool:
        """Check if miner is active - delegate to peer manager"""
        return await self.peer_manager.is_peer_active(uid, recent_windows)

    def weighted_random_sample_no_replacement(
        self, candidates: List[int], weights: List[int], k: int
    ) -> List[int]:
        """Weighted random sampling - delegate to peer manager"""
        return self.peer_manager.weighted_random_sample_no_replacement(
            candidates, weights, k
        )

    # Metadata methods
    async def post_start_window(self, start_window: int) -> None:
        """Post start window - delegate to metadata manager"""
        await self.metadata_manager.post_start_window(start_window)

    async def get_start_window(self, retries: int = -1) -> int | None:
        """Get start window - delegate to metadata manager"""
        return await self.metadata_manager.get_start_window(retries)

    async def post_peer_list(
        self,
        peers: list[int],
        first_effective_window: int,
        sync_window: int,
        weights: torch.Tensor,
        initial_selection: bool,
    ) -> None:
        """Post peer list - delegate to metadata manager"""
        await self.metadata_manager.post_peer_list(
            peers, first_effective_window, sync_window, weights, initial_selection
        )

    async def get_peer_list(
        self, fetch_previous: bool = False
    ) -> tuple[list[int], int] | None:
        """Get peer list - delegate to metadata manager"""
        return await self.metadata_manager.get_peer_list(fetch_previous)

    async def get_debug_dict(self, window: int) -> dict | None:
        """Get debug dict - delegate to metadata manager"""
        return await self.metadata_manager.get_debug_dict(window)

    # Aggregation methods
    async def load_aggregation(
        self, window: int, chunk_size: int = 50_000_000
    ) -> dict | None:
        """Load aggregation - delegate to aggregation manager"""
        return await self.aggregation_manager.load_aggregation(window, chunk_size)

    async def _apply_gathered_gradients(
        self,
        gather_result,
        model: LlamaForCausalLM,
        optimizer: SGD,
        scheduler: SequentialLR,
        transformer: TransformDCT,
        compressor: CompressDCT,
        device: str,
        window: int,
        global_step: int,
    ) -> tuple[bool, int]:
        """Apply gathered gradients - delegate to gradient manager"""
        return await self.gradient_manager.apply_gradients_to_model(
            gather_result,
            model,
            optimizer,
            scheduler,
            transformer,
            compressor,
            device,
            window,
            global_step,
        )

    def check_compressed_indices(
        self,
        param_name: str,
        idxs: Any,
        totalk: int,
        allowed_topk: int | None = None,
    ) -> None:
        """Check compressed indices - delegate to gradient manager"""
        self.gradient_manager.check_compressed_indices(
            param_name,
            idxs,
            totalk,
            allowed_topk if allowed_topk is not None else totalk,
        )

    # Helper methods for cleanup
    async def _cleanup_s3_data(
        self, uid: int | None, current_window: int, stale_retention: int
    ):
        """Clean up stale S3 data for a given uid."""
        if uid is None:
            return

        uid_str = str(uid)
        try:
            min_allowed_window = current_window - stale_retention
            pattern = re.compile(rf"^gradient-(\d+)-{uid_str}-v{tplr.__version__}.pt$")

            keys = await self.storage_client.list_objects("gradient", self.bucket)

            # Identify stale objects to delete
            stale_objects = []
            for key in keys:
                match = pattern.match(key)
                if match is None:
                    continue

                try:
                    file_window = int(match.group(1))
                except ValueError:
                    continue

                if file_window < min_allowed_window:
                    stale_objects.append(key)

            # Delete stale objects
            for key in stale_objects:
                await self.storage_client.delete_object(key, self.bucket)

            if stale_objects:
                tplr.logger.debug(
                    f"Removed {len(stale_objects)} stale S3 objects for {uid_str}"
                )

        except Exception as e:
            tplr.logger.error(f"Error cleaning up S3 data: {e}")

    # Properties for backward compatibility
    @property
    def active_peers(self):
        """Get active peers from peer manager"""
        return self.peer_manager.get_active_peers()

    @property
    def last_checkpoint_data(self):
        """Get last checkpoint data from checkpoint manager"""
        return self.checkpoint_manager.last_checkpoint_data

    @property
    def current_window(self):
        """Get current window"""
        return getattr(self.chain_manager, "_current_window", 0)

    @current_window.setter
    def current_window(self, value):
        """Set current window and propagate to managers"""
        setattr(self.chain_manager, "_current_window", value)

    def get_current_window(self) -> int | None:
        """Get current window for peer manager"""
        return self.current_window

    def start_commitment_fetcher(self):
        """Start commitment fetcher - delegate to parent"""
        if hasattr(self.chain_manager, "start_commitment_fetcher"):
            self.chain_manager.start_commitment_fetcher()

    async def get_commitments(self, block: int | None = None):
        """Get commitments - delegate to parent"""
        if hasattr(self.chain_manager, "get_commitments"):
            commitments = await self.chain_manager.get_commitments(block)
            # Ensure commitments are available in both places
            self.chain_manager.commitments = commitments
            self.checkpoint_manager.commitments = commitments
            return commitments
        return {}

    def try_commit(self, wallet, bucket):
        """Try commit - delegate to parent"""
        if hasattr(self.chain_manager, "try_commit"):
            self.chain_manager.try_commit(wallet, bucket)

    @property
    def peers(self):
        """Get peers list for miner compatibility"""
        return getattr(self.chain_manager, "_peers", [])

    @peers.setter
    def peers(self, value):
        """Set peers list"""
        setattr(self.chain_manager, "_peers", value)

    async def get_activity_time_window(self, reference_block_window: int) -> Optional[Tuple[datetime, datetime]]:
        """
        Get activity time window for protocol-specific operations.
        Centralizes time window calculation logic from neurons.
        
        Args:
            reference_block_window: Window number to base time calculation on
            
        Returns:
            Tuple of (time_min, time_max) or None if calculation fails
        """
        try:
            # Calculate the block number for the reference window
            reference_block = reference_block_window * self.hparams.blocks_per_window
            
            # Get blockchain timestamp for the reference block
            time_min = await self.chain_manager.get_block_timestamp(reference_block)
            
            if time_min is None:
                tplr.logger.error(f"Could not get timestamp for block {reference_block}")
                return None
            
            # Calculate time_max using configured delta
            time_max = time_min + timedelta(seconds=self.hparams.time_window_delta_seconds)
            
            tplr.logger.debug(
                f"Activity time window for window {reference_block_window}: "
                f"{time_min} to {time_max}"
            )
            
            return (time_min, time_max)
            
        except Exception as e:
            tplr.logger.error(f"Failed to calculate activity time window: {e}")
            return None

    async def refresh_gather_targets(self) -> None:
        """Refresh gather target peers using PeerManager."""
        await self.peer_manager.refresh_gather_targets(
            self.chain_manager.current_window, 
            self.metadata_manager
        )

    async def perform_model_catchup(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler.LRScheduler, 
        last_known_sync_window: int, 
        current_chain_window: int, 
        start_neuron_window: int
    ) -> int:
        """
        Orchestrate model catchup with aggregation server.
        
        Args:
            model: Model to catchup
            optimizer: Optimizer for parameter updates
            scheduler: Learning rate scheduler
            last_known_sync_window: Last window the model was synced to
            current_chain_window: Current blockchain window
            start_neuron_window: Starting window for this neuron
            
        Returns:
            New global step after catchup
        """
        # Check if catchup is needed
        if last_known_sync_window >= current_chain_window:
            tplr.logger.info("Model is already up-to-date, no catchup needed")
            return current_chain_window - start_neuron_window
        
        tplr.logger.info(
            f"Performing model catchup from window {last_known_sync_window} to {current_chain_window}"
        )
        
        # Use AggregationManager to apply catchup aggregations
        last_processed_window = await self.aggregation_manager.apply_catchup_aggregations(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            start_catchup_window=last_known_sync_window,
            target_window=current_chain_window,
            metadata_manager=self.metadata_manager,
            chain_manager=self.chain_manager
        )
        
        # Calculate new global step
        new_global_step = last_processed_window - start_neuron_window
        
        tplr.logger.info(
            f"Model catchup complete. Last processed window: {last_processed_window}, "
            f"New global step: {new_global_step}"
        )
        
        return new_global_step
