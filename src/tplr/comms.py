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
# type: ignore
import os
import json
import torch
import asyncio
from datetime import datetime, timezone
import bittensor as bt

from typing import List, Dict, Optional, TypeVar, Any
from aiobotocore.session import get_session

from . import __version__
from .config import BUCKET_SECRETS
from .schemas import Bucket

import tplr as tplr
# from .hparams import HParams

from types import SimpleNamespace
from typing import Tuple

from tplr.storage import StorageManager
from tplr.peer_manager import PeerManager
from tplr.chain_sync import ChainSync


# Constants
CF_REGION_NAME: str = "enam"
LOCAL_TMP_DIR = "/tmp/local_store"

T = TypeVar("T", bound=Any)
FixtureFunction = TypeVar("FixtureFunction", bound=Any)


class Comms:
    """
    Communication interface for the tlpr protocol.
    Provides three core primitives: put, get, and gather.
    """

    def __init__(
        self,
        wallet: "bt.wallet",
        config=None,
        metagraph=None,
        hparams=None,
        uid=None,
        **kwargs,
    ):
        """
        Initialize the communication system.

        Args:
            wallet: Bittensor wallet for authentication
            config: Network configuration
            metagraph: Network metagraph
            hparams: Hyperparameters
            uid: Neuron unique identifier
            **kwargs: Additional arguments
        """
        self.wallet = wallet
        self.uid = uid
        self.config = config
        self.metagraph = metagraph
        self.hparams = hparams

        # Create working directories
        self.temp_dir = os.path.join("/tmp", f"templar_{self.uid}")
        os.makedirs(self.temp_dir, exist_ok=True)

        hotkey = self.wallet.hotkey.ss58_address
        self.save_location = os.path.join("/tmp", f"hotkey_{hotkey}")
        os.makedirs(self.save_location, exist_ok=True)

        # Initialize component services
        self.storage = StorageManager(
            temp_dir=self.temp_dir, save_location=self.save_location, wallet=self.wallet
        )

        self.chain = ChainSync(
            config=config,
            netuid=config.netuid if config else None,
            metagraph=metagraph,
            hparams=hparams,
            wallet=wallet,
        )

        self.peer_manager = PeerManager(
            chain=self.chain, hparams=hparams, metagraph=metagraph
        )

        # Setup semaphores for connection limiting
        self.session = get_session()
        self.client_semaphore = asyncio.Semaphore(30)

        # Set up active peer tracking
        self.active_peers = set()
        self.inactive_peers = set()

        # Get our own bucket for writing
        self.bucket = self._get_own_bucket("gradients", "write")

        # Initialize loop reference for background tasks
        self.loop = None

        tplr.logger.info("Comms initialized with bucket: %s", self.bucket)

    def start_background_tasks(self):
        """Start all background monitoring tasks"""
        self.loop = asyncio.get_running_loop()
        # Start task for tracking active peers
        self.loop.create_task(self.peer_manager.track_active_peers())
        # Start task for periodically fetching commitments
        self.chain.start_commitment_fetcher()

    async def put(
        self,
        state_dict: Dict,
        uid: str,
        window: int,
        key: str,
        global_step: int = 0,
        local: bool = True,
    ) -> bool:
        """
        Store data either locally or in remote storage.

        Args:
            state_dict: Data to store
            uid: User ID for storage location
            window: Window number for storage location
            key: Key for stored data
            global_step: Current global step (for checkpoints)
            local: Whether to store locally

        Returns:
            Success status
        """
        # Add metadata to state_dict
        full_state_dict = {
            "state_dict": state_dict,
            "global_step": global_step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if local:
            # Store locally
            return await self.storage.store_local(
                state_dict=full_state_dict, uid=uid, window=window, key=key
            )
        else:
            # Store in remote bucket
            return await self.storage.store_remote(
                state_dict=full_state_dict,
                uid=uid,
                window=window,
                key=key,
                bucket=self.bucket,
            )

    async def get(
        self,
        uid: str,
        window: int,
        key: str,
        local: bool = True,
        device: str = "cpu",
        timeout: int = 30,
    ) -> Tuple[Dict, int]:
        """
        Retrieve data either locally or from remote storage.

        Args:
            uid: User ID for storage location
            window: Window number for storage location
            key: Key for stored data
            local: Whether to retrieve locally
            device: Device to load tensors on
            timeout: Timeout for remote requests

        Returns:
            Tuple of (state_dict, global_step)
        """
        if local:
            # Get from local storage
            data = await self.storage.get_local(uid=uid, window=window, key=key)
        else:
            # Get from remote bucket for the specified UID
            peer_bucket = self.chain.get_bucket(int(uid))
            if peer_bucket:
                data = await self.storage.get_remote(
                    uid=uid, window=window, key=key, bucket=peer_bucket, timeout=timeout
                )
            else:
                tplr.logger.warning(f"No bucket found for UID {uid}")
                return None, 0

        if data:
            # Extract and return state_dict and global_step
            state_dict = data.get("state_dict", {})
            global_step = data.get("global_step", 0)

            # Move tensors to the specified device
            state_dict = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in state_dict.items()
            }

            return state_dict, global_step

        return None, 0

    async def gather(
        self,
        my_uid: str,
        uids: List[str],
        window: int,
        key: str,
        timeout: int,
        device: str,
        totalks: dict,
        local: bool = True,
        stale_retention: int = 10,
        time_min: datetime = None,
        time_max: datetime = None,
    ) -> Optional[SimpleNamespace]:
        """
        Gather data from multiple peers.

        Args:
            my_uid: Own user ID
            uids: List of peer UIDs to gather from
            window: Window number to gather
            key: Data key to gather
            timeout: Request timeout
            device: Device to load tensors on
            totalks: Dictionary of parameter total sizes
            local: Whether to gather locally
            stale_retention: Number of windows to retain stale data
            time_min: Minimum timestamp for data
            time_max: Maximum timestamp for data

        Returns:
            SimpleNamespace with aggregated data
        """
        skipped_uids = []
        valid_uids = []
        valid_state_dicts = []
        valid_global_steps = []

        # Extract optional time parameters from kwargs without removing them
        time_min = time_min or None
        time_max = time_max or None
        if time_min or time_max:
            tplr.logger.debug(f"Gathering with time window: {time_min} to {time_max}")

        # Create tasks for all peers to fetch in parallel
        tasks = []
        for uid in uids:
            if uid == my_uid:
                continue

            # Create task to get data from this peer
            task = self.get_with_retry(
                uid=uid,
                window=window,
                key=key,
                local=local,
                device=device,
                timeout=timeout,
                time_min=time_min,
                time_max=time_max,
            )
            tasks.append((uid, task))

        # Process all responses
        for uid, task in tasks:
            try:
                response = await task

                # Validate response format
                try:
                    state_dict_resp, global_step_resp = response
                    tplr.logger.debug(
                        f"Received state dict and global step {global_step_resp} from UID {uid}"
                    )
                except (TypeError, ValueError) as e:
                    tplr.logger.debug(f"Invalid response format from UID {uid}: {e}")
                    skipped_uids.append(uid)
                    continue

                if state_dict_resp is None:
                    tplr.logger.debug(f"Empty state dict from UID {uid}")
                    skipped_uids.append(uid)
                    continue

                # This peer's data passed basic format validation
                valid_uids.append(uid)
                valid_state_dicts.append(state_dict_resp)
                valid_global_steps.append(global_step_resp)

            except Exception as e:
                tplr.logger.warning(f"Error processing response from UID {uid}: {e}")
                skipped_uids.append(uid)

        # If no valid responses, return None
        if not valid_uids:
            tplr.logger.warning(
                f"No valid responses gathered from peers. Skipped UIDs: {skipped_uids}"
            )
            return None

        # Aggregate all valid state dicts
        aggregated_dict = SimpleNamespace()
        for param_suffix in ["idxs", "vals"]:
            for uid_idx, state_dict in enumerate(valid_state_dicts):
                for key, tensor in state_dict.items():
                    if key.endswith(param_suffix):
                        if not hasattr(aggregated_dict, key):
                            setattr(aggregated_dict, key, [])
                        getattr(aggregated_dict, key).append(tensor)

        # Return the aggregated results with UIDs included
        return SimpleNamespace(
            state_dict=aggregated_dict,
            uids=valid_uids,
            global_steps=valid_global_steps,
            skipped_uids=skipped_uids,
        )

    # Helper methods
    async def get_with_retry(
        self, uid, window, key, local, device, timeout, time_min, time_max
    ):
        """Get data with retries on failure"""
        max_attempts = 3
        backoff_base = 1.5

        for attempt in range(max_attempts):
            try:
                return await self.get(
                    uid=uid,
                    window=window,
                    key=key,
                    local=local,
                    device=device,
                    timeout=timeout,
                    time_min=time_min,
                    time_max=time_max,
                )
            except Exception as e:
                if attempt < max_attempts - 1:
                    wait_time = backoff_base**attempt
                    tplr.logger.debug(
                        f"Get failed for UID {uid}, retrying in {wait_time:.1f}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    tplr.logger.warning(
                        f"Get failed for UID {uid} after {max_attempts} attempts: {e}"
                    )
                    raise

    def _get_own_bucket(self, bucket_type: str, rw: str) -> Bucket:
        bucket_conf = tplr.config.BUCKET_SECRETS.get(bucket_type)
        tplr.logger.debug("Bucket config for %s: %s", bucket_type, bucket_conf)
        if not bucket_conf:
            raise ValueError(f"No bucket configuration found for '{bucket_type}'.")

        name = bucket_conf.get("name", "").strip()
        if not name:
            raise ValueError(f"Bucket name for '{bucket_type}' must not be empty.")

        account_id = bucket_conf.get("account_id", "").strip()
        if not account_id:
            raise ValueError(f"Bucket account_id for '{bucket_type}' must not be empty.")

        creds = bucket_conf.get("credentials", {}).get(rw, {})
        access_key = creds.get("access_key_id", "").strip()
        secret_key = creds.get("secret_access_key", "").strip()
        if not access_key or not secret_key:
            raise ValueError(
                f"Bucket credentials for '{bucket_type}' in '{rw}' mode must not be empty."
            )

        bucket = Bucket(
            name=name,
            account_id=account_id,
            access_key_id=access_key,
            secret_access_key=secret_key,
        )
        tplr.logger.debug("Created Bucket: %s", bucket)
        return bucket

    # Chain sync and other methods needed for compatibility
    async def post_start_window(self, start_window):
        """Post start window information to the bucket"""
        data = json.dumps({"start_window": start_window}).encode("utf-8")
        key = f"start_window_v{__version__}.json"
        return await self.storage.store_bytes(data=data, key=key, bucket=self.bucket)

    async def get_start_window(self) -> int:
        """Get start window from the highest staked validator"""
        while True:
            try:
                # Get bucket of highest staked validator
                (
                    validator_bucket,
                    validator_uid,
                ) = await self.chain._get_highest_stake_validator_bucket()
                if validator_bucket is None:
                    tplr.logger.warning(
                        "No highest staked validator bucket found. Retrying in 10 seconds."
                    )
                    await asyncio.sleep(10)
                    continue

                tplr.logger.info(f"Fetching start_window from UID {validator_uid}")

                # Fetch start_window.json file
                key = f"start_window_v{__version__}.json"
                start_window_data = await self.storage.get_bytes(
                    key=key, bucket=validator_bucket
                )

                if start_window_data:
                    # Parse JSON data
                    if isinstance(start_window_data, dict):
                        start_window_json = start_window_data
                    else:
                        start_window_json = json.loads(
                            start_window_data.decode("utf-8")
                        )

                    start_window = start_window_json["start_window"]
                    tplr.logger.info(f"Fetched start_window: {start_window}")
                    return start_window

                tplr.logger.warning(
                    "start_window.json not found or empty. Retrying in 10 seconds."
                )
                await asyncio.sleep(10)
            except Exception as e:
                tplr.logger.error(f"Error fetching start_window: {e}")
                await asyncio.sleep(10)

    async def load_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        transformer,
        compressor,
        current_window: int,
        device: str,
        peers: list,
        uid: str,
        totalks: dict,
    ):
        """Load the latest checkpoint and handle catch-up if needed"""
        try:
            # First try to load from local storage
            checkpoint = await self.storage.load_latest_checkpoint(uid)

            if checkpoint is None:
                # Try to load from remote storage
                checkpoint = await self.storage.load_remote_checkpoint(
                    uid=uid, device=device, bucket=self.bucket
                )

            if checkpoint is None:
                tplr.logger.info("No checkpoint found. Starting fresh.")
                return False, {}, 0, optimizer, scheduler

            # Extract checkpoint data
            model_state_dict = checkpoint.get("model_state_dict", {})
            optimizer_state_dict = checkpoint.get("optimizer_state_dict", {})
            scheduler_state_dict = checkpoint.get("scheduler_state_dict", {})
            momentum = checkpoint.get("momentum", {})
            ckpt_start_window = checkpoint.get("start_window", 0)
            ckpt_window = checkpoint.get("current_window", 0)
            global_step = ckpt_window - ckpt_start_window

            # Load model state
            model.load_state_dict(model_state_dict)

            # Load optimizer state
            optimizer.load_state_dict(optimizer_state_dict)

            # Load scheduler state
            scheduler.load_state_dict(scheduler_state_dict)

            # Determine if catch-up is needed
            window_difference = current_window - ckpt_window
            tplr.logger.info(
                f"Checkpoint window: {ckpt_window}, Current window: {current_window}"
            )

            # Synchronize with missed windows if needed
            if window_difference > 0:
                tplr.logger.info(
                    f"Syncing optimizer/scheduler by stepping {window_difference} times…"
                )
                for _ in range(window_difference):
                    optimizer.step()
                    scheduler.step()

            return True, momentum, global_step, optimizer, scheduler

        except KeyError as e:
            tplr.logger.error(f"Invalid checkpoint format: missing key {e}")
            return False, {}, 0, optimizer, scheduler
        except Exception as e:
            tplr.logger.error(f"Failed to load checkpoint: {e}")
            return False, {}, 0, optimizer, scheduler
