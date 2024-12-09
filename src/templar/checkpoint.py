import aiofiles
import asyncio
import bittensor as bt
import botocore
import concurrent
import glob
import os
import re
import shutil
import torch
from aiobotocore.session import get_session
from typing import List, Optional, Union

from . import __version__
from .config import BUCKET_SECRETS, client_config
from .constants import CF_REGION_NAME
from .commitment import get_all_commitments
from .logging import logger
from .schemas import Bucket


def get_base_url(account_id: str) -> str:
    """Get base URL for R2 storage"""
    url = f"https://{account_id}.r2.cloudflarestorage.com"
    logger.debug(f"Base URL constructed: {url}")
    return url


def get_all_buckets(
    subtensor: bt.Subtensor,
    netuid: int,
    metagraph,
) -> List[Optional[Union[str, Bucket]]]:
    """
    Retrieves and parses all bucket commitments from the network.

    Args:
        subtensor: The subtensor instance
        netuid: Network UID
        metagraph: Network metagraph

    Returns:
        List[Optional[Union[str, Bucket]]]: List of bucket strings or Bucket objects,
        with None for invalid buckets. Index corresponds to UID.
    """
    buckets = []
    commitments = get_all_commitments(
        substrate=subtensor.substrate,
        netuid=netuid,
        metagraph=metagraph,
    )

    for uid in metagraph.uids:
        bucket = commitments.get(uid)
        logger.debug(f"UID {uid} bucket: {bucket}")

        if bucket is not None:
            logger.debug(f"Retrieved valid bucket for UID {uid}: {bucket}")
            buckets.append(bucket)
        else:
            logger.debug(f"No valid bucket found for UID {uid}")
            buckets.append(None)

    logger.debug(f"Final list of buckets: {buckets}")
    return buckets


async def save_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    global_step: int = 0,
    **kwargs,
):
    """
    Saves the checkpoint to the specified filename asynchronously.
    """
    checkpoint = {
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
    }
    if optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    # Include additional state variables
    for key, value in kwargs.items():
        checkpoint[key] = value

    # Save the checkpoint asynchronously to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, torch.save, checkpoint, filename)
    logger.info(f"Checkpoint saved at {filename}")


async def load_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    device: str = "cpu",
):
    """
    Loads the checkpoint from the specified filename.
    """
    try:
        checkpoint = torch.load(filename, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        global_step = checkpoint.get("global_step", 0)
        additional_state = {
            k: checkpoint[k]
            for k in checkpoint
            if k
            not in [
                "global_step",
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
            ]
        }
        return global_step, additional_state
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {filename}: {e}")
        return 0, {}


async def upload_checkpoint(
    checkpoint_path: str,
    wallet: bt.wallet,
):
    """
    Uploads the checkpoint file to S3 storage using configured bucket.
    """
    filename = f"neuron_checkpoints_{wallet.hotkey.ss58_address}_v{__version__}.pth"
    logger.debug(f"Uploading checkpoint to S3: {filename}")

    # Extract just the bucket name without the full path
    bucket_name = BUCKET_SECRETS["bucket_name"].split("/")[-1]

    temp_dir = os.path.dirname(checkpoint_path)
    temp_file = os.path.join(temp_dir, f"temp_{filename}")

    try:
        shutil.copy2(checkpoint_path, temp_file)

        session = get_session()
        async with session.create_client(
            "s3",
            endpoint_url=get_base_url(BUCKET_SECRETS["account_id"]),
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
            aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
        ) as s3_client:
            try:
                with open(temp_file, "rb") as f:
                    await s3_client.put_object(
                        Bucket=bucket_name,  # Use just the bucket name
                        Key=filename,
                        Body=f,
                    )
                logger.debug(f"Successfully uploaded checkpoint to S3: {filename}")
            except Exception as e:
                logger.exception(
                    f"Failed to upload checkpoint to S3: {filename}. Error: {e}"
                )
                raise
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logger.debug(f"Temporary file {temp_file} removed")


async def download_checkpoint_from_neuron(
    bucket_info: Bucket,
    neuron_hotkey: str,
    checkpoint_dir: str,
) -> Optional[str]:
    """
    Downloads the latest checkpoint file (by block number) from the neuron's S3 storage.

    Args:
        bucket_info (Bucket): The neuron's bucket credentials.
        neuron_hotkey (str): Hotkey of the neuron with the highest stake.
        checkpoint_dir (str): Directory to save the checkpoint.

    Returns:
        Optional[str]: Path to the downloaded checkpoint file, or None if failed.
    """
    import re

    # Prepare the checkpoint filename pattern
    regex_pattern = rf"neuron_checkpoint_{neuron_hotkey}_b(\d+)_v[\d\.]+\.pth"

    local_checkpoint_path = None

    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=get_base_url(bucket_info.account_id),
        region_name=CF_REGION_NAME,
        config=client_config,
        aws_access_key_id=bucket_info.access_key_id,
        aws_secret_access_key=bucket_info.secret_access_key,
    ) as s3_client:
        try:
            # List all checkpoint files for this neuron in their bucket
            paginator = s3_client.get_paginator("list_objects_v2")
            latest_block_number = -1
            latest_filename = None

            async for page in paginator.paginate(
                Bucket=bucket_info.name, Prefix=f"neuron_checkpoint_{neuron_hotkey}_"
            ):
                contents = page.get("Contents", [])
                if not contents:
                    continue  # Move to the next page

                for obj in contents:
                    key = obj["Key"]
                    match = re.match(regex_pattern, key)
                    if match:
                        block_number = int(match.group(1))
                        if block_number > latest_block_number:
                            latest_block_number = block_number
                            latest_filename = key

            if latest_filename:
                # Download the latest checkpoint
                local_checkpoint_path = os.path.join(checkpoint_dir, latest_filename)
                os.makedirs(os.path.dirname(local_checkpoint_path), exist_ok=True)

                response = await s3_client.get_object(
                    Bucket=bucket_info.name, Key=latest_filename
                )
                async with aiofiles.open(local_checkpoint_path, "wb") as f:
                    while True:
                        chunk = await response["Body"].read(1024 * 1024)
                        if not chunk:
                            break
                        await f.write(chunk)

                logger.debug(
                    f"Successfully downloaded checkpoint: {local_checkpoint_path}"
                )
                return local_checkpoint_path
            else:
                logger.info(f"No valid checkpoints found for neuron {neuron_hotkey}")
                return None

        except botocore.exceptions.ClientError as e:
            logger.error(f"Error downloading checkpoint: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading checkpoint: {str(e)}")
            return None


def get_neuron_with_highest_stake(
    metagraph, buckets: List[Optional[Union[str, Bucket]]]
) -> Optional[str]:
    """
    Get the hotkey of the neuron with highest stake that has a valid bucket.

    Args:
        metagraph: Network metagraph
        buckets: List of buckets corresponding to UIDs

    Returns:
        Optional[str]: Hotkey of highest stake neuron, or None if no valid neuron found
    """
    try:
        # Get highest stake UID using metagraph.S
        highest_stake_uid = int(metagraph.S.argmax())

        # Check if this UID has a valid bucket
        if highest_stake_uid < len(buckets) and buckets[highest_stake_uid] is not None:
            return metagraph.hotkeys[highest_stake_uid]

        logger.warning("No valid bucket found for highest stake neuron")
        return None

    except Exception as e:
        logger.error(f"Error finding highest stake neuron: {e}")
        return None


async def load_highest_stake_checkpoint(
    metagraph,
    buckets: List[Optional[Union[str, Bucket]]],
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> int:
    """
    Attempts to load checkpoint from the highest stake neuron.

    Args:
        metagraph: Network metagraph
        buckets: List of buckets corresponding to UIDs
        model: Model to load checkpoint into
        checkpoint_path: Base path for checkpoint storage
        device: Device to load checkpoint to
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state

    Returns:
        int: Global step from loaded checkpoint, or 0 if starting fresh
    """
    try:
        highest_stake_hotkey = get_neuron_with_highest_stake(
            metagraph=metagraph, buckets=buckets
        )

        if highest_stake_hotkey:
            uid = metagraph.hotkeys.index(highest_stake_hotkey)
            bucket_info = buckets[uid]

            if bucket_info:
                checkpoint_dir = os.path.dirname(checkpoint_path)
                os.makedirs(checkpoint_dir, exist_ok=True)

                checkpoint_file = await download_checkpoint_from_neuron(
                    bucket_info=bucket_info,
                    neuron_hotkey=highest_stake_hotkey,
                    checkpoint_dir=checkpoint_dir,
                )

                if checkpoint_file:
                    # Load the checkpoint with optional optimizer/scheduler
                    global_step, _ = await load_checkpoint(
                        filename=checkpoint_file,
                        model=model,
                        device=device,
                        optimizer=optimizer if optimizer is not None else None,
                        scheduler=scheduler if scheduler is not None else None,
                    )
                    logger.info(f"Resumed from global step {global_step}")
                    return global_step if global_step is not None else 0

                logger.warning(
                    "Failed to download neuron checkpoint. Starting from scratch."
                )
                return 0

            logger.warning(
                f"No bucket info for neuron {highest_stake_hotkey}. Starting from scratch."
            )
            return 0

        logger.warning("No neurons found. Starting from scratch.")
        return 0

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return 0


async def get_latest_checkpoint_from_neuron(bucket_info, neuron_hotkey):
    """
    Lists available checkpoints in the neuron's bucket and downloads the latest one.

    Args:
        bucket_info: Bucket information for the neuron
        neuron_hotkey: Hotkey of the neuron

    Returns:
        str: Path to the downloaded checkpoint file
    """
    import re  # Ensure 're' is imported at the top of your module

    try:
        bucket = bucket_info["bucket_name"].split("/")[-1]

        # Set up client
        session = get_session()
        async with session.create_client(
            "s3",
            endpoint_url=get_base_url(bucket_info["account_id"]),
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=bucket_info["read"]["access_key_id"],
            aws_secret_access_key=bucket_info["read"]["secret_access_key"],
        ) as s3_client:
            # List objects in the bucket with prefix matching the neuron's checkpoints
            prefix = f"neuron_checkpoint_{neuron_hotkey}"
            paginator = s3_client.get_paginator("list_objects_v2")
            async for result in paginator.paginate(Bucket=bucket, Prefix=prefix):
                objects = result.get("Contents", [])
                break  # For simplicity, only considering the first page

            if not objects:
                logger.warning("No checkpoints found in neuron's bucket")
                return None

            # Sort checkpoints by block number
            checkpoints = []
            for obj in objects:
                key = obj["Key"]
                match = re.match(
                    rf"neuron_checkpoint_{neuron_hotkey}_b(\d+)_v.+\.pth", key
                )
                if match:
                    block_number = int(match.group(1))
                    checkpoints.append((block_number, key))

            if not checkpoints:
                logger.warning("No valid checkpoints found")
                return None

            # Get the checkpoint with the highest block number
            checkpoints.sort(reverse=True)
            latest_block, latest_key = checkpoints[0]
            logger.info(f"Latest checkpoint found: {latest_key} (block {latest_block})")

            # Download the checkpoint
            local_checkpoint_path = os.path.join(
                "checkpoints",
                latest_key,  # Adjust the path as needed
            )
            os.makedirs(os.path.dirname(local_checkpoint_path), exist_ok=True)

            async with aiofiles.open(local_checkpoint_path, "wb") as f:
                response = await s3_client.get_object(Bucket=bucket, Key=latest_key)
                async for chunk in response["Body"].iter_chunked(1024 * 1024):
                    await f.write(chunk)

            return local_checkpoint_path

    except Exception as e:
        logger.error(f"Failed to get latest checkpoint from neuron: {e}")
        return None


class CheckpointManager:
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        wallet: bt.wallet,
        device: str = "cpu",
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.wallet = wallet
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.getcwd()  # Default to current working directory

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._shutdown = False
        self._active_futures = set()
        self._initialize_thread_pool()

    def _initialize_thread_pool(self):
        """Initialize a new thread pool if needed"""
        if not hasattr(self, "thread_pool") or self.thread_pool._shutdown:
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=5,  # Adjust max_workers as needed
                thread_name_prefix="checkpoint_worker",
            )
            self._shutdown = False

    def _save_checkpoint_sync(self, global_step: int, block_number: int, **kwargs):
        """Synchronous checkpoint save."""
        checkpoint = {
            "global_step": global_step,
            "block_number": block_number,
            "model_state_dict": self.model.state_dict(),
        }
        if self.optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        # Include any additional info
        checkpoint.update(kwargs)

        # Construct the filename with block number
        filename = f"neuron_checkpoint_{self.wallet.hotkey.ss58_address}_b{block_number}_v{__version__}.pth"

        # Full path
        self.checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        # Save the checkpoint
        torch.save(checkpoint, self.checkpoint_path)
        logger.info(f"Checkpoint saved at {self.checkpoint_path}")

    async def _upload_and_cleanup(self):
        """Async method to upload the checkpoint and clean up old checkpoints."""
        try:
            # Upload the checkpoint
            await self._upload_checkpoint_async()

            # Cleanup old checkpoints
            await self._cleanup_old_checkpoints_async()

        except Exception as e:
            logger.exception(f"Exception in _upload_and_cleanup: {e}")

    async def _upload_checkpoint_async(self):
        """Async checkpoint upload."""
        try:
            filename = os.path.basename(self.checkpoint_path)
            logger.info(f"Uploading checkpoint to S3: {filename}")

            # Get clean bucket name
            bucket = BUCKET_SECRETS["bucket_name"].split("/")[-1]

            # Upload using S3 client
            session = get_session()
            async with session.create_client(
                "s3",
                endpoint_url=get_base_url(BUCKET_SECRETS["account_id"]),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
                aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
            ) as s3_client:
                async with aiofiles.open(self.checkpoint_path, "rb") as f:
                    file_content = await f.read()
                    response = await s3_client.put_object(
                        Bucket=bucket,
                        Key=filename,
                        Body=file_content,
                        CacheControl="no-cache, no-store, must-revalidate",
                    )
                logger.info(f"Successfully uploaded checkpoint to S3: {filename}")
                logger.debug(f"Upload response: {response}")

        except Exception as e:
            logger.exception(f"Failed to upload checkpoint: {e}")
            raise

    async def _cleanup_old_checkpoints_async(self, max_checkpoints=3):
        """
        Asynchronously deletes old checkpoints, keeping only the latest 'max_checkpoints'
        checkpoints both locally and in S3.
        """
        # Pattern to match checkpoint filenames
        pattern = os.path.join(
            self.checkpoint_dir,
            f"neuron_checkpoint_{self.wallet.hotkey.ss58_address}_b*_v{__version__}.pth",
        )

        # Get a list of checkpoint files
        checkpoint_files = glob.glob(pattern)
        if len(checkpoint_files) <= max_checkpoints:
            return  # No need to delete any checkpoints

        # Extract block numbers and sort the checkpoints
        checkpoints = []
        for filepath in checkpoint_files:
            filename = os.path.basename(filepath)
            match = re.match(
                rf"neuron_checkpoint_{self.wallet.hotkey.ss58_address}_b(\d+)_v{__version__}\.pth",
                filename,
            )
            if match:
                block_number = int(match.group(1))
                checkpoints.append((block_number, filepath))

        # Sort the checkpoints by block number in descending order
        checkpoints.sort(reverse=True)

        # Keep the latest 'max_checkpoints' and delete the rest
        old_checkpoints = checkpoints[max_checkpoints:]
        # Delete local files synchronously (file operations are I/O bound but quick)
        for _, filepath in old_checkpoints:
            try:
                os.remove(filepath)
                logger.debug(f"Deleted local old checkpoint: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to delete local old checkpoint {filepath}: {e}")

        # Delete old checkpoints from S3
        await self._delete_old_checkpoints_from_s3(old_checkpoints)

    async def _delete_old_checkpoints_from_s3(self, old_checkpoints):
        """
        Deletes the specified old checkpoints from S3.

        Args:
            old_checkpoints (list): A list of tuples containing block numbers and file paths.
        """
        try:
            # Extract just the bucket name
            bucket = BUCKET_SECRETS["bucket_name"].split("/")[-1]

            # Set up S3 client
            session = get_session()
            async with session.create_client(
                "s3",
                endpoint_url=get_base_url(BUCKET_SECRETS["account_id"]),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
                aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
            ) as s3_client:
                # Build list of keys to delete
                delete_objects = {
                    "Objects": [
                        {"Key": os.path.basename(filepath)}
                        for _, filepath in old_checkpoints
                    ],
                    "Quiet": True,
                }
                if delete_objects["Objects"]:
                    response = await s3_client.delete_objects(
                        Bucket=bucket, Delete=delete_objects
                    )
                    logger.debug(
                        f"Deleted old checkpoints from S3: {delete_objects['Objects']}"
                    )
                    logger.debug(f"S3 deletion response: {response}")
                else:
                    logger.debug("No old checkpoints to delete from S3.")
        except Exception as e:
            logger.warning(f"Failed to delete old checkpoints from S3: {e}")

    async def load_from_highest_stake(
        self,
        metagraph,
        buckets,
    ) -> int:
        """
        Attempts to load checkpoint from the highest stake neuron.

        Args:
            metagraph: Network metagraph
            buckets: List of buckets corresponding to UIDs

        Returns:
            int: Global step from loaded checkpoint, or 0 if starting fresh
        """
        try:
            highest_stake_hotkey = get_neuron_with_highest_stake(
                metagraph=metagraph, buckets=buckets
            )

            if highest_stake_hotkey:
                uid = metagraph.hotkeys.index(highest_stake_hotkey)
                bucket_info = buckets[uid]

                if bucket_info:
                    checkpoint_dir = os.path.dirname(self.checkpoint_path)
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    checkpoint_file = await download_checkpoint_from_neuron(
                        bucket_info=bucket_info,
                        neuron_hotkey=highest_stake_hotkey,
                        checkpoint_dir=checkpoint_dir,
                    )

                    if checkpoint_file:
                        # Load the checkpoint with optional optimizer/scheduler
                        global_step, _ = await load_checkpoint(
                            filename=checkpoint_file,
                            model=self.model,
                            device=self.device,
                            optimizer=self.optimizer
                            if self.optimizer is not None
                            else None,
                            scheduler=self.scheduler
                            if self.scheduler is not None
                            else None,
                        )
                        logger.info(f"Resumed from global step {global_step}")
                        return global_step if global_step is not None else 0

                    logger.warning(
                        "Failed to download neuron checkpoint. Starting from scratch."
                    )
                    return 0

                logger.warning(
                    f"No bucket info for neuron {highest_stake_hotkey}. Starting from scratch."
                )
                return 0

            logger.warning("No neurons found. Starting from scratch.")
            return 0

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return 0

    def save_and_upload(self, global_step: int, block_number: int, **kwargs):
        """Non-blocking save and upload."""
        try:
            # Save checkpoint synchronously
            self._save_checkpoint_sync(global_step, block_number, **kwargs)

            # Submit the upload and cleanup task to the thread pool
            if not self._shutdown:
                future = self.thread_pool.submit(self._upload_and_cleanup_sync)
                future.add_done_callback(self._cleanup_future)
                self._active_futures.add(future)
                logger.debug("Submitted checkpoint upload to thread pool")

        except Exception as e:
            logger.error(f"Error in save_and_upload: {e}")

    def _upload_and_cleanup_sync(self):
        """Sync wrapper to run the async upload and cleanup."""
        try:
            logger.info("Starting upload and cleanup task")
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._upload_and_cleanup())
            loop.close()
            logger.info("Upload and cleanup task completed")
        except Exception as e:
            logger.exception(f"Exception in _upload_and_cleanup_sync: {e}")

    def _cleanup_future(self, future):
        """Callback to clean up completed futures."""
        self._active_futures.discard(future)

    def cleanup(self):
        """Cleanup background workers"""
        if not self._shutdown:
            self._shutdown = True
            # Cancel any remaining futures
            for future in self._active_futures:
                future.cancel()
            # Shutdown thread pool
            if hasattr(self, "thread_pool"):
                self.thread_pool.shutdown(wait=True)
            logger.info("CheckpointManager shutdown complete")

    def __del__(self):
        """Ensure thread pool is cleaned up"""
        self.cleanup()
