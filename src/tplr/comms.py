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
from datetime import datetime
from types import SimpleNamespace
from typing import Any, List, Literal, Optional

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
from .network.peer_manager import PeerManager
from .protocol.coordinator_manager import CoordinatorManager
from .schemas import Bucket

# Import all the new managers
from .storage.client import StorageClient
from .storage.file_manager import FileManager
from .training.aggregation_manager import AggregationManager
from .training.checkpoint_manager import CheckpointManager
from .training.gradient_manager import GradientManager

# Constants
CPU_COUNT = os.cpu_count() or 4


class Comms(ChainManager):
    """Main orchestration layer for distributed training communications"""

    def __init__(
        self,
        wallet: "bt.wallet | None",
        key_prefix: str = "model",
        config=None,
        netuid=None,
        metagraph=None,
        hparams=None,
        uid=None,
    ):
        self.uid = uid
        self.wallet = wallet

        # Create temp directory for this instance
        self.temp_dir = os.path.join("/tmp", f"templar_{self.uid}")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Get the bucket directly
        self.bucket = self.get_own_bucket("gradients", "write")

        # Initialize ChainManager with all required parameters
        super().__init__(
            config=config,
            netuid=netuid,
            metagraph=metagraph,
            hparams=hparams,
            wallet=self.wallet,
            bucket=self.bucket,
        )

        # Use the hotkey directly in the save_location
        if self.wallet is not None:
            hotkey = self.wallet.hotkey.ss58_address
            self.save_location = os.path.join("/tmp", f"hotkey_{hotkey}")
            os.makedirs(self.save_location, exist_ok=True)
        self.key_prefix = key_prefix

        # Current window is managed by ChainManager parent class

        # Initialize all managers
        self._initialize_managers()

        # Background task management
        self.loop = None
        self.executor = None

    def _initialize_managers(self):
        """Initialize all manager components"""
        # Foundation components
        self.file_manager = FileManager(self.temp_dir, self.uid)
        self.storage_client = StorageClient(self.file_manager)

        # Domain-specific managers
        self.gradient_manager = GradientManager(
            self.storage_client,
            self.file_manager,
            self.config.device if hasattr(self.config, "device") else "cpu",
            self.hparams,
        )

        self.checkpoint_manager = CheckpointManager(
            self.storage_client,
            self.file_manager,
            self.bucket,
            self.uid if self.uid is not None else 0,
            self,  # Pass ChainManager (self) for validator lookup
        )

        self.peer_manager = PeerManager(
            self,  # ChainManager
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
            self.config.device if hasattr(self.config, "device") else "cpu",
        )

        self.coordinator_manager = CoordinatorManager(
            self.storage_client,
            self.file_manager,
            self,  # ChainManager
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

    async def gradient_timestamp(
        self, uid: int, window: int, version: str = tplr.__version__
    ) -> float:
        """
        Return POSIX seconds of the gradient file’s Last-Modified header,
        or 0.0 if it does not exist / fails.
        """
        bucket = self.get_bucket(int(uid))
        if not bucket:
            return 0.0
        try:
            key = f"gradient-{window}-{uid}-v{version}.pt"
            metadata = await self.storage_client.get_object_metadata(key, bucket)
            if metadata and "LastModified" in metadata:
                return metadata["LastModified"].timestamp()
            return 0.0
        except Exception:
            return 0.0

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
                peer_bucket = self.get_bucket(int(uid))
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
        uids: List[int],
        window: int,
        timeout: int,
        device: str,
        totalks: dict,
        local: bool = True,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        show_progress: bool = False,
    ) -> SimpleNamespace | None:
        """Gather operation - delegate to aggregation manager"""
        if my_uid is None:
            return None
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

    # Coordination methods
    async def post_start_window(self, start_window: int) -> None:
        """Post start window - delegate to coordinator manager"""
        await self.coordinator_manager.post_start_window(start_window)

    async def get_start_window(self, retries: int = -1) -> int | None:
        """Get start window - delegate to coordinator manager"""
        return await self.coordinator_manager.get_start_window(retries)

    async def post_peer_list(
        self,
        peers: list[int],
        first_effective_window: int,
        sync_window: int,
        weights: torch.Tensor,
        initial_selection: bool,
    ) -> None:
        """Post peer list - delegate to coordinator manager"""
        await self.coordinator_manager.post_peer_list(
            peers, first_effective_window, sync_window, weights, initial_selection
        )

    async def get_peer_list(
        self, fetch_previous: bool = False
    ) -> tuple[list[int], int] | None:
        """Get peer list - delegate to coordinator manager"""
        return await self.coordinator_manager.get_peer_list(fetch_previous)

    async def get_debug_dict(self, window: int) -> dict | None:
        """Get debug dict - delegate to coordinator manager"""
        return await self.coordinator_manager.get_debug_dict(window)

    # Aggregation methods
    async def load_aggregation(self, window: int) -> dict | None:
        """Load aggregation - delegate to aggregation manager"""
        return await self.aggregation_manager.load_aggregation(window)

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

    def get_current_window(self) -> int | None:
        """Get current window for peer manager"""
        return self.current_window

    @property
    def peers(self):
        """Get peers list for miner compatibility"""
        return getattr(self, "_peers", [])

    @peers.setter
    def peers(self, value):
        """Set peers list"""
        self._peers = value
