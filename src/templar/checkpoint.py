import asyncio
import aiofiles
import torch
import os
import glob
import re
import shutil
from typing import List, Optional, Union, Dict
from aiobotocore.session import get_session
from . import __version__
from .config import BUCKET_SECRETS, client_config
from .constants import CF_REGION_NAME
from .logging import logger
from .schemas import Bucket
from .commitment import get_all_commitments
import botocore.exceptions


def get_base_url(account_id: str) -> str:
    """Get base URL for R2 storage"""
    return f"https://{account_id}.r2.cloudflarestorage.com"


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
    Uses asyncio.to_thread to avoid blocking the main event loop.
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
    checkpoint.update(kwargs)

    try:
        await asyncio.to_thread(torch.save, checkpoint, filename)
        logger.info(f"Checkpoint saved at {filename}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint at {filename}: {e}")
        raise


async def load_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    device: str = "cpu",
):
    """
    Loads the checkpoint from the specified filename asynchronously.
    Uses asyncio.to_thread to avoid blocking the main event loop.
    """
    try:
        checkpoint = await asyncio.to_thread(torch.load, filename, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        global_step = checkpoint.get("global_step", 0)
        additional_state = {
            k: checkpoint[k]
            for k in checkpoint
            if k not in ["global_step", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict"]
        }
        return global_step, additional_state
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {filename}: {e}")
        return 0, {}


async def upload_checkpoint(
    checkpoint_path: str,
    wallet,
):
    """
    Uploads the checkpoint file to S3 storage using configured bucket.
    """
    filename = f"neuron_checkpoints_{wallet.hotkey.ss58_address}_v{__version__}.pth"
    logger.debug(f"Uploading checkpoint to S3: {filename}")

    bucket_name = BUCKET_SECRETS["bucket_name"].split("/")[-1]
    temp_dir = os.path.dirname(checkpoint_path)
    temp_file = os.path.join(temp_dir, f"temp_{filename}")

    # Copy file to a temporary file before uploading to ensure atomicity
    try:
        await asyncio.to_thread(shutil.copy2, checkpoint_path, temp_file)

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
                async with aiofiles.open(temp_file, "rb") as f:
                    body = await f.read()
                    await s3_client.put_object(
                        Bucket=bucket_name,
                        Key=filename,
                        Body=body,
                    )
                logger.debug(f"Successfully uploaded checkpoint to S3: {filename}")
            except Exception as e:
                logger.exception(f"Failed to upload checkpoint to S3: {filename}. Error: {e}")
                raise
    finally:
        if os.path.exists(temp_file):
            await asyncio.to_thread(os.remove, temp_file)
            logger.debug(f"Temporary file {temp_file} removed")


async def download_checkpoint_from_neuron(
    bucket_info: Bucket,
    neuron_hotkey: str,
    checkpoint_dir: str,
) -> Optional[str]:
    """
    Downloads the latest checkpoint file (by block number) from the neuron's S3 storage.
    """
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
            paginator = s3_client.get_paginator("list_objects_v2")
            latest_block_number = -1
            latest_filename = None

            async for page in paginator.paginate(
                Bucket=bucket_info.name, Prefix=f"neuron_checkpoint_{neuron_hotkey}_"
            ):
                contents = page.get("Contents", [])
                if not contents:
                    continue

                for obj in contents:
                    key = obj["Key"]
                    match = re.match(regex_pattern, key)
                    if match:
                        block_number = int(match.group(1))
                        if block_number > latest_block_number:
                            latest_block_number = block_number
                            latest_filename = key

            if latest_filename:
                local_checkpoint_path = os.path.join(checkpoint_dir, latest_filename)
                await asyncio.to_thread(os.makedirs, os.path.dirname(local_checkpoint_path), exist_ok=True)

                response = await s3_client.get_object(
                    Bucket=bucket_info.name, Key=latest_filename
                )

                async with aiofiles.open(local_checkpoint_path, "wb") as f:
                    while True:
                        chunk = await response["Body"].read(1024 * 1024)
                        if not chunk:
                            break
                        await f.write(chunk)

                logger.debug(f"Successfully downloaded checkpoint: {local_checkpoint_path}")
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


def get_all_buckets(
    subtensor,
    netuid: int,
    metagraph,
) -> List[Optional[Union[str, Bucket]]]:
    """
    Retrieves and parses all bucket commitments from the network.
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


def get_neuron_with_highest_stake(
    metagraph,
    buckets: List[Optional[Union[str, Bucket]]]
) -> Optional[str]:
    """
    Get the hotkey of the neuron with highest stake that has a valid bucket.
    """
    try:
        highest_stake_uid = int(metagraph.S.argmax())
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
    """
    try:
        highest_stake_hotkey = get_neuron_with_highest_stake(metagraph=metagraph, buckets=buckets)

        if highest_stake_hotkey:
            uid = metagraph.hotkeys.index(highest_stake_hotkey)
            bucket_info = buckets[uid]

            if bucket_info:
                checkpoint_dir = os.path.dirname(checkpoint_path)
                await asyncio.to_thread(os.makedirs, checkpoint_dir, exist_ok=True)

                checkpoint_file = await download_checkpoint_from_neuron(
                    bucket_info=bucket_info,
                    neuron_hotkey=highest_stake_hotkey,
                    checkpoint_dir=checkpoint_dir,
                )

                if checkpoint_file:
                    global_step, _ = await load_checkpoint(
                        filename=checkpoint_file,
                        model=model,
                        device=device,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
                    logger.info(f"Resumed from global step {global_step}")
                    return global_step if global_step is not None else 0

                logger.warning("Failed to download neuron checkpoint. Starting from scratch.")
                return 0

            logger.warning(f"No bucket info for neuron {highest_stake_hotkey}. Starting from scratch.")
            return 0

        logger.warning("No neurons found. Starting from scratch.")
        return 0

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return 0


class CheckpointManager:
    """
    Improved CheckpointManager that saves and uploads checkpoints asynchronously,
    and can clean up old checkpoints, all without blocking the main thread.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        wallet,
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
        self.upload_task = None  # Track the upload task

        self.checkpoint_dir = os.path.dirname(self.checkpoint_path) or os.getcwd()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._shutdown = False

    async def _save_checkpoint_async(self, global_step: int, block_number: int, **kwargs):
        """Asynchronously save a checkpoint."""
        checkpoint = {
            "global_step": global_step,
            "block_number": block_number,
            "model_state_dict": self.model.state_dict(),
        }

        if self.optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        checkpoint.update(kwargs)

        filename = f"neuron_checkpoint_{self.wallet.hotkey.ss58_address}_b{block_number}_v{__version__}.pth"
        full_path = os.path.join(self.checkpoint_dir, filename)

        await asyncio.to_thread(torch.save, checkpoint, full_path)
        self.checkpoint_path = full_path
        logger.info(f"Checkpoint saved at {self.checkpoint_path}")

    async def _upload_checkpoint_async(self):
        """Async checkpoint upload to S3."""
        filename = os.path.basename(self.checkpoint_path)
        logger.info(f"Uploading checkpoint to S3: {filename}")

        bucket = BUCKET_SECRETS["bucket_name"].split("/")[-1]
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
                async with aiofiles.open(self.checkpoint_path, "rb") as f:
                    data = await f.read()
                    await s3_client.put_object(
                        Bucket=bucket,
                        Key=filename,
                        Body=data,
                        CacheControl="no-cache, no-store, must-revalidate",
                    )
                logger.info(f"Successfully uploaded checkpoint to S3: {filename}")
            except Exception as e:
                logger.exception(f"Failed to upload checkpoint: {e}")
                raise

    async def _cleanup_old_checkpoints_async(self, max_checkpoints=3):
        """
        Asynchronously deletes old checkpoints locally and in S3.
        Keeps only the latest 'max_checkpoints'.
        """
        pattern = os.path.join(
            self.checkpoint_dir,
            f"neuron_checkpoint_{self.wallet.hotkey.ss58_address}_b*_v{__version__}.pth",
        )

        checkpoint_files = await asyncio.to_thread(glob.glob, pattern)
        if len(checkpoint_files) <= max_checkpoints:
            return  # no cleanup needed

        # Parse block numbers
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

        # Sort by block number descending
        checkpoints.sort(reverse=True)
        old_checkpoints = checkpoints[max_checkpoints:]

        # Delete local files
        for _, filepath in old_checkpoints:
            try:
                await asyncio.to_thread(os.remove, filepath)
                logger.debug(f"Deleted local old checkpoint: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to delete local old checkpoint {filepath}: {e}")

        # Delete old checkpoints from S3
        await self._delete_old_checkpoints_from_s3(old_checkpoints)

    async def _delete_old_checkpoints_from_s3(self, old_checkpoints):
        bucket = BUCKET_SECRETS["bucket_name"].split("/")[-1]

        session = get_session()
        async with session.create_client(
            "s3",
            endpoint_url=get_base_url(BUCKET_SECRETS["account_id"]),
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
            aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
        ) as s3_client:
            delete_objects = {
                "Objects": [
                    {"Key": os.path.basename(filepath)} for _, filepath in old_checkpoints
                ],
                "Quiet": True,
            }
            if delete_objects["Objects"]:
                try:
                    response = await s3_client.delete_objects(
                        Bucket=bucket, Delete=delete_objects
                    )
                    logger.debug(f"Deleted old checkpoints from S3: {delete_objects['Objects']}")
                    logger.debug(f"S3 deletion response: {response}")
                except Exception as e:
                    logger.warning(f"Failed to delete old checkpoints from S3: {e}")

    async def load_from_highest_stake(
        self,
        metagraph,
        buckets,
    ) -> int:
        """
        Attempts to load checkpoint from the highest stake neuron.
        """
        try:
            highest_stake_hotkey = get_neuron_with_highest_stake(metagraph=metagraph, buckets=buckets)

            if highest_stake_hotkey:
                uid = metagraph.hotkeys.index(highest_stake_hotkey)
                bucket_info = buckets[uid]

                if bucket_info:
                    checkpoint_dir = os.path.dirname(self.checkpoint_path)
                    await asyncio.to_thread(os.makedirs, checkpoint_dir, exist_ok=True)

                    checkpoint_file = await download_checkpoint_from_neuron(
                        bucket_info=bucket_info,
                        neuron_hotkey=highest_stake_hotkey,
                        checkpoint_dir=checkpoint_dir,
                    )

                    if checkpoint_file:
                        global_step, _ = await load_checkpoint(
                            filename=checkpoint_file,
                            model=self.model,
                            device=self.device,
                            optimizer=self.optimizer if self.optimizer else None,
                            scheduler=self.scheduler if self.scheduler else None,
                        )
                        logger.info(f"Resumed from global step {global_step}")
                        return global_step if global_step is not None else 0

                    logger.warning("Failed to download neuron checkpoint. Starting from scratch.")
                    return 0

                logger.warning(f"No bucket info for neuron {highest_stake_hotkey}. Starting from scratch.")
                return 0

            logger.warning("No neurons found. Starting from scratch.")
            return 0

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return 0

    async def save_and_upload(self, global_step: int, block_number: int, **kwargs):
        """Save and upload checkpoint asynchronously."""
        try:
            start_time = asyncio.get_event_loop().time()
            # Save checkpoint
            await self._save_checkpoint_async(global_step, block_number, **kwargs)
            save_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Checkpoint save took {save_time:.2f} seconds")

            # Schedule new upload and cleanup without canceling existing ones
            self.upload_task = asyncio.create_task(self._upload_and_cleanup())
        except Exception as e:
            logger.error(f"Error in save_and_upload: {e}")

    async def _upload_and_cleanup(self):
        """Uploads the checkpoint and cleans up old ones."""
        try:
            start_time = asyncio.get_event_loop().time()
            await self._upload_checkpoint_async()
            upload_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Checkpoint upload took {upload_time:.2f} seconds")

            cleanup_start = asyncio.get_event_loop().time()
            await self._cleanup_old_checkpoints_async()
            cleanup_time = asyncio.get_event_loop().time() - cleanup_start
            logger.info(f"Checkpoint cleanup took {cleanup_time:.2f} seconds")
        except Exception as e:
            logger.exception(f"Exception in _upload_and_cleanup: {e}")

    def cleanup(self):
        """Cleanup resources if needed."""
        self._shutdown = True
        # Let any pending upload tasks complete
        logger.info("CheckpointManager shutdown complete")
