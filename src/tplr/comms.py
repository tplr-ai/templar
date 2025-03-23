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
import json
import torch
import asyncio
from datetime import datetime, timezone

from typing import List, Dict, Optional
from aiobotocore.session import get_session

from . import __version__
from .schemas import Bucket

import tplr as tplr
from . import config

from types import SimpleNamespace
from typing import Tuple

from tplr.storage import StorageManager
from tplr.peer_manager import PeerManager
from tplr.chain_sync import ChainSync


# Constants
CF_REGION_NAME: str = "enam"
LOCAL_TMP_DIR = "/tmp/local_store"


class Comms:
    """
    Communication interface for the tlpr protocol.
    Provides three core primitives: put, get, and gather.
    """

    def __init__(
        self,
        wallet,
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
        stale_retention: int = 10,
    ) -> bool:
        """
        Store data either locally or in remote storage, and cleanup old data beyond stale_retention.

        Args:
            state_dict: Data to store.
            uid: User ID for storage location.
            window: Window number for storage location.
            key: Key for stored data.
            global_step: Current global step (for checkpoints).
            local: Whether to store locally.
            stale_retention: Number of recent files to retain.
        """
        from datetime import datetime

        full_state_dict = {
            "state_dict": state_dict,
            "global_step": global_step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if local:
            success = await self.storage.store_local(
                state_dict=full_state_dict, uid=uid, window=window, key=key
            )
            if success:
                await self.storage.cleanup_local_gradients(
                    uid, key, retention=stale_retention
                )
            return success
        else:
            success = await self.storage.store_remote(
                state_dict=full_state_dict,
                uid=uid,
                window=window,
                key=key,
                bucket=self.bucket,
            )
            if success:
                await self.storage.cleanup_remote_gradients(
                    uid, key, retention=stale_retention, bucket=self.bucket
                )
            return success

    async def get(
        self,
        uid: str,
        window: int,
        key: str,
        local: bool = True,
        device: str = "cpu",
        timeout: int = 30,
        stale_retention: int = 10,
        time_min=None,
        time_max=None,
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
            stale_retention: Number of recent files to retain (passed to storage)
            time_min: Minimum timestamp for filtering gradients
            time_max: Maximum timestamp for filtering gradients

        Returns:
            Tuple of (state_dict, global_step)
        """
        if local:
            # Get from local storage with stale retention and time boundaries.
            data = await self.storage.get_local(
                uid=uid,
                window=window,
                key=key,
                stale_retention=stale_retention,
                time_min=time_min,
                time_max=time_max,
            )
        else:
            # Get from remote bucket for the specified UID with proper parameters.
            peer_bucket = self.chain.get_bucket(int(uid))
            if peer_bucket:
                data = await self.storage.get_remote(
                    uid=uid,
                    window=window,
                    key=key,
                    bucket=peer_bucket,
                    timeout=timeout,
                    stale_retention=stale_retention,
                    time_min=time_min,
                    time_max=time_max,
                )
            else:
                tplr.logger.warning(f"No bucket found for UID {uid}")
                return {}, 0

        if data:
            # Extract and return state_dict and global_step.
            if isinstance(data, dict):
                state_dict = data.get("state_dict", {})
                global_step = data.get("global_step", 0)
            else:
                state_dict = {}
                global_step = 0

            # Move tensors to the specified device.
            state_dict = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in state_dict.items()
            }
            return state_dict, global_step

        return {}, 0

    async def gather(
        self,
        my_uid: str,
        uids: List[int],
        window: int,
        key: str,
        timeout: int,
        device: str,
        local: bool = True,
        stale_retention: int = 10,
        time_min: datetime = None,
        time_max: datetime = None,
    ) -> Optional[SimpleNamespace]:
        import time

        start_time = time.time()
        metrics = {"upload_bytes": 0, "download_bytes": 0}

        tplr.logger.debug(
            f"Starting gather for window {window} with time window: {time_min} to {time_max}"
        )
        tplr.logger.debug(
            f"Starting gather operation - my_uid: {my_uid}, window: {window}, key: {key}, timeout: {timeout}"
        )
        tplr.logger.debug(f"Target UIDs for gathering: {uids}")

        uid_state_dicts = {}  # Map each valid UID to its processed state_dict
        valid_uids = []
        skipped_uids = []
        global_steps = {}  # Map UID to its global step

        async with self.client_semaphore:
            batch_tasks = [
                self.get_with_retry(
                    uid=uid,
                    window=window,
                    key=key,
                    local=local,
                    device=device,
                    timeout=timeout,
                    stale_retention=stale_retention,
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
                    if isinstance(response, Exception):
                        tplr.logger.debug(f"Error from UID {uid}: {str(response)}")
                        skipped_uids.append(uid)
                        continue
                    if response is None:
                        tplr.logger.info(
                            f"Skipped UID {uid} - gradient might not exist or was uploaded too late"
                        )
                        skipped_uids.append(uid)
                        continue

                    try:
                        state_dict_resp, global_step_resp = response
                        tplr.logger.debug(
                            f"Received state dict and global step {global_step_resp} from UID {uid}"
                        )
                    except (TypeError, ValueError) as e:
                        tplr.logger.debug(
                            f"Invalid response format from UID {uid}: {e}"
                        )
                        skipped_uids.append(uid)
                        continue

                    if state_dict_resp is None:
                        tplr.logger.debug(f"Empty state dict from UID {uid}")
                        skipped_uids.append(uid)
                        continue

                    # Process tensors and normalize those with "vals" keys.
                    processed_state = {}
                    for param_name, tensor in state_dict_resp.items():
                        if isinstance(tensor, torch.Tensor):
                            tensor = tensor.to(device)
                            if param_name.endswith("vals"):
                                norm = torch.norm(tensor)
                                normalized = tensor / (norm + 1e-8)
                                processed_state[param_name] = normalized
                            else:
                                processed_state[param_name] = tensor
                            metrics["download_bytes"] += (
                                tensor.element_size() * tensor.nelement()
                            )
                        else:
                            processed_state[param_name] = tensor

                    uid_state_dicts[uid] = processed_state
                    valid_uids.append(uid)
                    global_steps[uid] = global_step_resp
            except Exception as e:
                tplr.logger.error(f"Error processing uid batch: {str(e)}")

        if not valid_uids:
            tplr.logger.info("No valid gradients received from any UID")
            return None

        total_time = time.time() - start_time
        tplr.logger.info(
            f"Gather completed in {total_time:.2f}s. Success rate: {len(valid_uids)}/{len(uids)}, "
            f"Upload: {metrics['upload_bytes']} bytes, Download: {metrics['download_bytes']} bytes"
        )

        result = SimpleNamespace(
            time=total_time,
            upload_bytes=metrics["upload_bytes"],
            download_bytes=metrics["download_bytes"],
            success_rate=len(valid_uids) / len(uids),
            state_dicts=uid_state_dicts,
            uids=valid_uids,
            global_steps=global_steps,
            skipped_uids=skipped_uids,
        )
        return result

    # Helper methods
    async def get_with_retry(
        self,
        uid,
        window,
        key,
        local,
        device,
        timeout,
        stale_retention,
        time_min,
        time_max,
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
                    stale_retention=stale_retention,
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
                    raise Exception(
                        f"Get failed for UID {uid} after {max_attempts} attempts: {e}"
                    )

    def _get_own_bucket(self, bucket_type: str, rw: str) -> Bucket:
        bucket_conf = config.BUCKET_SECRETS.get(bucket_type)
        tplr.logger.debug("Bucket config for %s: %s", bucket_type, bucket_conf)
        if not bucket_conf:
            raise ValueError(f"No bucket configuration found for '{bucket_type}'.")

        name = bucket_conf.get("name", "").strip()
        if not name:
            raise ValueError(f"Bucket name for '{bucket_type}' must not be empty.")

        account_id = bucket_conf.get("account_id", "").strip()
        if not account_id:
            raise ValueError(
                f"Bucket account_id for '{bucket_type}' must not be empty."
            )

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
                    if isinstance(start_window_data, (bytes, bytearray)):
                        start_window_json = json.loads(
                            start_window_data.decode("utf-8")
                        )
                    elif isinstance(start_window_data, dict):
                        start_window_json = start_window_data
                    else:
                        # Handle unexpected type
                        start_window_json = {"start_window": 0}

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
        device: str,
    ):
        """Load the latest available checkpoint with matching version and from highest staked validator"""
        bucket = self.chain.get_own_bucket("gradients", "read")
        if bucket is None:
            tplr.logger.warning("No bucket available for loading checkpoint")
            return False, None, None, None, None

        latest_window = None
        latest_checkpoint = None

        try:
            import boto3
            from botocore.client import Config
            import re

            s3_config = Config(
                region_name="auto", signature_version="s3v4", max_pool_connections=256
            )

            s3_client = boto3.client(
                "s3",
                endpoint_url=f"https://{bucket.account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=bucket.access_key_id,
                aws_secret_access_key=bucket.secret_access_key,
                config=s3_config,
            )

            # List objects with checkpoint prefix
            response = s3_client.list_objects_v2(
                Bucket=bucket.name, Prefix="checkpoint-"
            )

            current_version = __version__
            tplr.logger.info(
                f"Looking for checkpoints with current version {current_version}"
            )

            # Regex to parse: checkpoint-{window}-{uid}-v{version}.pt
            pattern = re.compile(r"^checkpoint-(\d+)-(\d+)-v(.+)\.pt$")

            if "Contents" in response:
                for obj in response["Contents"]:
                    key = obj["Key"]
                    m = pattern.match(key)
                    if not m:
                        tplr.logger.debug(f"Skipping invalid checkpoint format: {key}")
                        continue

                    try:
                        checkpoint_window = int(m.group(1))
                        checkpoint_uid = int(m.group(2))
                        checkpoint_version = m.group(3)
                    except Exception as e:
                        tplr.logger.debug(f"Error parsing checkpoint key {key}: {e}")
                        continue

                    # Version must match current version.
                    if checkpoint_version != current_version:
                        tplr.logger.debug(
                            f"Skipping checkpoint with non-matching version: {key} (version {checkpoint_version})"
                        )
                        continue

                    # Get the highest staked validator UID.
                    (
                        _,
                        highest_validator_uid,
                    ) = await self.chain._get_highest_stake_validator_bucket(
                        refresh_commitments=True
                    )
                    if highest_validator_uid is None:
                        tplr.logger.warning("Highest staked validator UID not found.")
                        continue

                    # Filter checkpoints by highest staked validator UID
                    if checkpoint_uid != highest_validator_uid:
                        tplr.logger.debug(
                            f"Skipping checkpoint not from highest staked validator (expected uid {highest_validator_uid}): {key}"
                        )
                        continue

                    if latest_window is None or checkpoint_window > latest_window:
                        latest_window = checkpoint_window
                        latest_checkpoint = key

            if latest_checkpoint:
                tplr.logger.info(
                    f"Found checkpoint: {latest_checkpoint} (window {latest_window})"
                )
            else:
                tplr.logger.info(
                    f"No checkpoints found matching version {current_version} from the highest staked validator. Starting from scratch."
                )
                return False, None, None, None, None

        except Exception as e:
            tplr.logger.warning(f"Error listing checkpoint files: {e}")
            return False, None, None, None, None

        try:
            # Download the checkpoint
            temp_path = os.path.join(self.storage.temp_dir, latest_checkpoint)
            tplr.logger.info(
                f"Downloading checkpoint {latest_checkpoint} to {temp_path}"
            )
            result = await self.storage.s3_get_object(
                key=latest_checkpoint, bucket=bucket, file_path=temp_path, timeout=60
            )

            if not result:
                tplr.logger.warning(
                    f"Failed to download checkpoint {latest_checkpoint}"
                )
                return False, None, None, None, None

            tplr.logger.info(f"Loading checkpoint from {temp_path}")
            try:
                checkpoint = await asyncio.to_thread(
                    torch.load, temp_path, map_location=device
                )

                if "model_state_dict" not in checkpoint:
                    tplr.logger.error("Checkpoint missing model_state_dict key")
                    return False, None, None, None, None

                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                momentum = checkpoint["momentum"]
                global_step = checkpoint["global_step"]

                tplr.logger.info(f"Loaded checkpoint with global_step={global_step}")

                os.remove(temp_path)

                return True, momentum, global_step, optimizer, scheduler

            except Exception as e:
                tplr.logger.error(f"Error loading checkpoint data: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False, None, None, None, None

        except Exception as e:
            tplr.logger.error(f"Error in checkpoint loading process: {e}")
            return False, None, None, None, None

    async def save_remote_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        momentum,
        current_window: int,
        start_window: int,
    ):
        """Save checkpoint to R2 and local storage"""
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
        }

        # upload to R2
        await self.storage.store_remote(
            state_dict=checkpoint_data,
            uid=self.uid,
            window=current_window,
            key="checkpoint",
            bucket=self.bucket,
        )

        return True
