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
    endpoint = get_base_url(BUCKET_SECRETS["account_id"])
    print(f'endpoint is {endpoint}')

    # Copy file to a temporary file before uploading to ensure atomicity
    try:
        await asyncio.to_thread(shutil.copy2, checkpoint_path, temp_file)

        session = get_session()
        async with session.create_client(
            "s3",
            endpoint_url=endpoint,
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
                logger.info(f"Successfully uploaded checkpoint to S3: {filename}")
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
        """Async checkpoint upload to S3 with verified parallel uploads."""
        try:
            filename = os.path.basename(self.checkpoint_path)
            logger.info(f"Starting checkpoint upload to S3: {filename}")

            bucket = BUCKET_SECRETS["bucket_name"].split("/")[-1]
            chunk_size = 16 * 1024 * 1024  # 16MB chunks
            max_concurrent_uploads = 50
            max_retries = 3
            retry_delay = 5

            session = get_session()
            async with session.create_client(
                "s3",
                endpoint_url=f"https://{BUCKET_SECRETS['account_id']}.r2.cloudflarestorage.com",
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
                aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
            ) as s3_client:
                # Initialize multipart upload
                response = await s3_client.create_multipart_upload(
                    Bucket=bucket,
                    Key=filename,
                    CacheControl="no-cache, no-store, must-revalidate"
                )
                upload_id = response['UploadId']
                logger.info(f"Initiated multipart upload with ID: {upload_id}")
                
                try:
                    total_size = os.path.getsize(self.checkpoint_path)
                    total_parts = (total_size + chunk_size - 1) // chunk_size
                    parts = {}  # Use dict to track parts by number
                    uploaded_size = 0
                    semaphore = asyncio.Semaphore(max_concurrent_uploads)
                    upload_tasks = []
                    failed_parts = set()

                    async def upload_part_with_retry(part_number: int, offset: int) -> dict:
                        """Upload a single part with retries and verification."""
                        for attempt in range(max_retries):
                            try:
                                async with semaphore:
                                    async with aiofiles.open(self.checkpoint_path, 'rb') as f:
                                        await f.seek(offset)
                                        chunk = await f.read(min(chunk_size, total_size - offset))
                                        
                                        response = await s3_client.upload_part(
                                            Bucket=bucket,
                                            Key=filename,
                                            PartNumber=part_number,
                                            UploadId=upload_id,
                                            Body=chunk
                                        )
                                        
                                        # Verify part upload
                                        part_size = len(chunk)
                                        if part_size == 0:
                                            raise ValueError(f"Zero-size chunk for part {part_number}")
                                        
                                        nonlocal uploaded_size
                                        uploaded_size += part_size
                                        progress = (uploaded_size / total_size) * 100
                                        
                                        if part_number % 5 == 0 or progress >= 100:
                                            logger.info(f"Upload progress: {progress:.1f}% - Part {part_number}/{total_parts}")
                                            logger.info(f"Active uploads: {len([t for t in upload_tasks if not t.done()])}")
                                        
                                        return {
                                            'PartNumber': part_number,
                                            'ETag': response['ETag'],
                                            'Size': part_size
                                        }
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    logger.warning(f"Retry {attempt + 1}/{max_retries} for part {part_number}: {str(e)}")
                                    await asyncio.sleep(retry_delay)
                                else:
                                    failed_parts.add(part_number)
                                    raise

                    # Create upload tasks for all parts
                    for part_number in range(1, total_parts + 1):
                        offset = (part_number - 1) * chunk_size
                        task = asyncio.create_task(upload_part_with_retry(part_number, offset))
                        upload_tasks.append(task)
                        
                    # Wait for all uploads and collect results
                    completed_parts = await asyncio.gather(*upload_tasks, return_exceptions=True)
                    
                    # Process results and check for failures
                    for part in completed_parts:
                        if isinstance(part, Exception):
                            raise Exception(f"Part upload failed: {str(part)}")
                        parts[part['PartNumber']] = part
                    
                    # Verify all parts are present and ordered
                    if len(parts) != total_parts:
                        missing_parts = set(range(1, total_parts + 1)) - set(parts.keys())
                        raise Exception(f"Missing parts: {missing_parts}")
                    
                    # Sort parts for completion
                    ordered_parts = [parts[i] for i in range(1, total_parts + 1)]
                    
                    # Complete multipart upload
                    completion_response = await s3_client.complete_multipart_upload(
                        Bucket=bucket,
                        Key=filename,
                        UploadId=upload_id,
                        MultipartUpload={'Parts': [{
                            'PartNumber': p['PartNumber'],
                            'ETag': p['ETag']
                        } for p in ordered_parts]}
                    )
                    
                    # Verify upload completion
                    try:
                        head_response = await s3_client.head_object(
                            Bucket=bucket,
                            Key=filename
                        )
                        if head_response['ContentLength'] != total_size:
                            raise Exception(f"Size mismatch: uploaded={head_response['ContentLength']}, expected={total_size}")
                        
                        logger.info(f"Successfully verified upload of {filename} ({total_size} bytes)")
                    except Exception as e:
                        raise Exception(f"Upload verification failed: {str(e)}")
                    
                except Exception as e:
                    logger.error(f"Error during upload: {str(e)}")
                    try:
                        await s3_client.abort_multipart_upload(
                            Bucket=bucket,
                            Key=filename,
                            UploadId=upload_id
                        )
                        logger.info(f"Aborted multipart upload {upload_id}")
                    except Exception as abort_e:
                        logger.error(f"Failed to abort multipart upload: {str(abort_e)}")
                    raise

        except Exception as e:
            logger.exception(f"Failed to upload checkpoint: {e}")
            raise

        finally:
            # Clean up any remaining tasks
            if 'upload_tasks' in locals():
                for task in upload_tasks:
                    if not task.done():
                        task.cancel()

    async def _cleanup_old_checkpoints_async(self, max_checkpoints=3):
        """
        Asynchronously deletes old checkpoints locally and in S3.
        Keeps only the latest 'max_checkpoints'.
        """
        logger.info(f"Starting checkpoint cleanup, keeping latest {max_checkpoints} checkpoints")
        pattern = os.path.join(
            self.checkpoint_dir,
            f"neuron_checkpoint_{self.wallet.hotkey.ss58_address}_b*_v{__version__}.pth",
        )
        logger.info(f"Looking for checkpoints matching pattern: {pattern}")

        checkpoint_files = await asyncio.to_thread(glob.glob, pattern)
        logger.info(f"Found {len(checkpoint_files)} total checkpoint files")
        if len(checkpoint_files) <= max_checkpoints:
            logger.info("No cleanup needed - number of checkpoints below threshold")
            return

        # Parse block numbers
        logger.info("Parsing block numbers from checkpoint filenames")
        checkpoints = []
        for filepath in checkpoint_files:
            filename = os.path.basename(filepath)
            logger.info(f"Processing checkpoint file: {filename}")
            match = re.match(
                rf"neuron_checkpoint_{self.wallet.hotkey.ss58_address}_b(\d+)_v{__version__}\.pth",
                filename,
            )
            if match:
                block_number = int(match.group(1))
                logger.info(f"Found checkpoint for block {block_number}")
                checkpoints.append((block_number, filepath))

        # Sort by block number descending
        checkpoints.sort(reverse=True)
        old_checkpoints = checkpoints[max_checkpoints:]
        logger.info(f"Identified {len(old_checkpoints)} checkpoints to delete")

        # Delete local files
        logger.info("Starting deletion of local checkpoint files")
        for block_num, filepath in old_checkpoints:
            try:
                logger.info(f"Attempting to delete checkpoint from block {block_num} at {filepath}")
                await asyncio.to_thread(os.remove, filepath)
                logger.info(f"Successfully deleted local checkpoint: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to delete local checkpoint {filepath}: {e}")
                logger.error(f"Error details: {str(e)}")

        # Delete old checkpoints from S3
        logger.info("Starting deletion of S3 checkpoint files")
        await self._delete_old_checkpoints_from_s3(old_checkpoints)

    async def _delete_old_checkpoints_from_s3(self, old_checkpoints):
        logger.info(f"Starting S3 checkpoint deletion for {len(old_checkpoints)} files")
        bucket = BUCKET_SECRETS["bucket_name"].split("/")[-1]
        logger.info(f"Using bucket: {bucket}")

        session = get_session()
        logger.info("Created aiobotocore session")
        
        logger.info(f"Connecting to S3 endpoint: {get_base_url(BUCKET_SECRETS['account_id'])}")
        async with session.create_client(
            "s3",
            endpoint_url=get_base_url(BUCKET_SECRETS["account_id"]),
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
            aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
        ) as s3_client:
            logger.info("Successfully connected to S3")

            delete_objects = {
                "Objects": [
                    {"Key": os.path.basename(filepath)} for _, filepath in old_checkpoints
                ],
                "Quiet": True,
            }
            logger.info(f"Prepared delete request for {len(delete_objects['Objects'])} objects")
            
            if delete_objects["Objects"]:
                try:
                    logger.info(f"Attempting to delete objects: {[obj['Key'] for obj in delete_objects['Objects']]}")
                    response = await s3_client.delete_objects(
                        Bucket=bucket, Delete=delete_objects
                    )
                    logger.info(f"Successfully initiated deletion request")
                    logger.info(f"Deleted old checkpoints from S3: {delete_objects['Objects']}")
                    logger.info(f"S3 deletion response: {response}")
                    
                    if 'Deleted' in response:
                        logger.info(f"Successfully deleted {len(response['Deleted'])} objects")
                    if 'Errors' in response:
                        logger.warning(f"Failed to delete {len(response['Errors'])} objects: {response['Errors']}")
                        
                except Exception as e:
                    logger.error(f"Failed to delete old checkpoints from S3: {str(e)}")
                    logger.error(f"Full error details: {e.__class__.__name__}: {str(e)}")

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
