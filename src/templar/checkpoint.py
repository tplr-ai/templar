import aiofiles
import asyncio
import bittensor as bt
import botocore
import concurrent
import os
import queue
import shutil
import torch
from aiobotocore.session import get_session
import threading
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
    Downloads the checkpoint file from the neuron's S3 storage.

    Args:
        bucket_info (Bucket): The neuron's bucket credentials.
        neuron_hotkey (str): Hotkey of the neuron.
        checkpoint_dir (str): Directory to save the checkpoint.
        key_prefix (str, optional): S3 key prefix. Defaults to 'neuron_checkpoints'.

    Returns:
        Optional[str]: Path to the downloaded checkpoint file, or None if failed.
    """
    filename = f"neuron_checkpoints_{neuron_hotkey}_v{__version__}.pth"
    local_checkpoint_path = os.path.join(
        checkpoint_dir, f"checkpoint-{neuron_hotkey}.pth"
    )

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
            response = await s3_client.get_object(Bucket=bucket_info.name, Key=filename)
            async with aiofiles.open(local_checkpoint_path, "wb") as f:
                await f.write(await response["Body"].read())
            logger.debug(f"Successfully downloaded checkpoint: {local_checkpoint_path}")
            return local_checkpoint_path
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.info(f"No checkpoint found for neuron {neuron_hotkey}")
            else:
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
        self._shutdown = False
        self._active_futures = set()
        self.upload_queue = queue.Queue()
        self._initialize_thread_pool()
        self._start_upload_worker()

    def _initialize_thread_pool(self):
        """Initialize a new thread pool if needed"""
        if not hasattr(self, "thread_pool") or self.thread_pool._shutdown:
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="checkpoint_worker"
            )
            self._shutdown = False

    def _start_upload_worker(self):
        """Start a dedicated thread for handling uploads"""

        def upload_worker():
            while not self._shutdown:
                try:
                    # Wait for upload tasks with timeout
                    task = self.upload_queue.get(timeout=1.0)
                    if task is None:  # Shutdown signal
                        break

                    # Run the upload
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._upload_checkpoint_async())
                    except Exception as e:
                        logger.error(f"Background upload failed: {e}")
                    finally:
                        loop.close()
                        self.upload_queue.task_done()
                except queue.Empty:
                    continue  # Keep checking for new tasks

        self.upload_thread = threading.Thread(
            target=upload_worker, name="checkpoint_uploader", daemon=True
        )
        self.upload_thread.start()

    def _cleanup_future(self, future):
        """Remove completed future from active set"""
        self._active_futures.discard(future)

    def _save_checkpoint_sync(self, global_step: int, **kwargs):
        """Synchronous checkpoint save"""
        try:
            checkpoint = {
                "global_step": global_step,
                "model_state_dict": self.model.state_dict(),
            }

            if self.optimizer:
                checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

            # Include additional state variables
            for key, value in kwargs.items():
                checkpoint[key] = value

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

            # Save checkpoint
            torch.save(checkpoint, self.checkpoint_path)
            logger.info(f"Checkpoint saved locally: {self.checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    async def _upload_checkpoint_async(self):
        """Async checkpoint upload"""
        try:
            # Simple filename without paths
            filename = f"neuron_checkpoints_{self.wallet.hotkey.ss58_address}_v{__version__}.pth"
            logger.debug(f"Uploading checkpoint to S3: {filename}")

            # Get clean bucket name
            bucket = BUCKET_SECRETS["bucket_name"]
            if "/" in bucket:
                bucket = bucket.split("/")[-1]

            # Create temp file
            temp_dir = os.path.dirname(self.checkpoint_path)
            temp_file = os.path.join(temp_dir, f"temp_{filename}")

            try:
                # Copy to temp file
                shutil.copy2(self.checkpoint_path, temp_file)

                # Upload using same client setup as slice upload
                session = get_session()
                async with session.create_client(
                    "s3",
                    endpoint_url=get_base_url(BUCKET_SECRETS["account_id"]),
                    region_name=CF_REGION_NAME,
                    config=client_config,
                    aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
                    aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
                ) as s3_client:
                    with open(temp_file, "rb") as f:
                        await s3_client.put_object(Bucket=bucket, Key=filename, Body=f)
                    logger.info(f"Successfully uploaded checkpoint to S3: {filename}")
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Temporary file {temp_file} removed")

        except Exception as e:
            logger.error(f"Failed to upload checkpoint: {e}")
            raise

    def save_and_upload(self, global_step: int, **kwargs):
        """Non-blocking save and upload"""
        try:
            # Save checkpoint synchronously (usually fast)
            self._save_checkpoint_sync(global_step, **kwargs)

            # Queue the upload to happen in background
            if not self._shutdown:
                self.upload_queue.put_nowait(True)
                logger.debug("Queued checkpoint for background upload")

        except Exception as e:
            logger.error(f"Error in save_and_upload: {e}")

    async def load_from_highest_stake(
        self, metagraph, buckets: List[Optional[Union[str, Bucket]]]
    ) -> int:
        """
        Loads checkpoint from highest stake neuron.

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
                        checkpoint = torch.load(
                            checkpoint_file, map_location=self.device
                        )
                        self.model.load_state_dict(checkpoint["model_state_dict"])

                        if self.optimizer and "optimizer_state_dict" in checkpoint:
                            self.optimizer.load_state_dict(
                                checkpoint["optimizer_state_dict"]
                            )
                        if self.scheduler and "scheduler_state_dict" in checkpoint:
                            self.scheduler.load_state_dict(
                                checkpoint["scheduler_state_dict"]
                            )

                        global_step = checkpoint.get("global_step", 0)
                        logger.info(f"Resumed from global step {global_step}")
                        return global_step

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

    def cleanup(self):
        """Cleanup background workers"""
        if not self._shutdown:
            self._shutdown = True

            # Signal upload worker to stop
            try:
                self.upload_queue.put_nowait(None)
            except queue.Full:
                pass

            # Wait for upload thread to finish with timeout
            if hasattr(self, "upload_thread"):
                self.upload_thread.join(timeout=5.0)

            # Cleanup thread pool
            if hasattr(self, "thread_pool"):
                self.thread_pool.shutdown(wait=True)

            logger.info("CheckpointManager shutdown complete")

    def __del__(self):
        """Ensure thread pool is cleaned up"""
        self.cleanup()
