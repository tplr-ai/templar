import asyncio
import time
import aiofiles
import torch
import os
import glob
import re
import shutil
from typing import List, Optional, Union
from aiobotocore.session import get_session
from tqdm import tqdm
from . import __version__
from .config import BUCKET_SECRETS, client_config
from .constants import CF_REGION_NAME
from .logging import logger
from .schemas import Bucket
from .commitment import get_all_commitments


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
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str = "cpu",
    is_validator: bool = False,
    hparams=None,
) -> int:
    """
    Loads the checkpoint from the specified filename asynchronously.
    Adjusts optimizer and scheduler for miners.
    """
    try:
        logger.info(f"Loading checkpoint from {filename}")
        checkpoint = await asyncio.to_thread(
            torch.load, filename, map_location=device, weights_only=True
        )

        # Load the model state
        model.load_state_dict(checkpoint["model_state_dict"])
        global_step = checkpoint.get("global_step", 0)
        logger.info(f"Loaded model state at global step {global_step}")

        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Adjust optimizer state if miner
        if not is_validator:
            # Retrieve validator's learning rate from optimizer state
            validator_lr = optimizer.param_groups[0]["lr"]
            miner_lr = hparams.learning_rate  # Miner's learning rate

            # Compute scaling factor
            scaling_factor = validator_lr / miner_lr

            # Scale optimizer's internal states
            for state in optimizer.state.values():
                if "exp_avg" in state:
                    state["exp_avg"].mul_(scaling_factor)
                if "exp_avg_sq" in state:
                    # Optionally adjust exp_avg_sq if needed
                    pass

            # Update optimizer's learning rate to miner's learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = miner_lr

            logger.info("Adjusted optimizer states for miner.")

        else:
            logger.info("Loaded optimizer states for validator.")

        return global_step

    except Exception as e:
        logger.error(f"Failed to load checkpoint from {filename}: {e}")
        return 0


async def download_checkpoint_from_neuron(
    bucket_info: Bucket,
    neuron_hotkey: str,
    checkpoint_dir: str,
) -> Optional[str]:
    """
    Downloads the latest checkpoint file with parallel processing and progress tracking.
    Handles multiple processes and provides detailed progress information.
    """
    start_time = time.time()
    regex_pattern = (
        rf"neuron_checkpoint_{neuron_hotkey}_b(\d+)_v({re.escape(__version__)})\.pth"
    )
    local_checkpoint_path = None
    chunk_size = 8 * 1024 * 1024  # 8MB chunks
    max_concurrent_downloads = 4
    max_retries = 3
    retry_delay = 5

    # Ensure checkpoint directory exists with absolute path
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    def format_size(size_bytes):
        """Convert bytes to human readable format"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0

    def create_progress_bar(progress, total_size):
        """Create a progress bar with size information"""
        width = 50
        filled = int(width * progress / 100)
        bar = "█" * filled + "-" * (width - filled)
        size_info = (
            f"{format_size(total_size * progress / 100)}/{format_size(total_size)}"
        )
        return f"[{bar}] {progress:.1f}% {size_info}"

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
            # Find latest checkpoint
            paginator = s3_client.get_paginator("list_objects_v2")
            latest_block_number = -1
            latest_filename = None
            file_size = None

            async for page in paginator.paginate(
                Bucket=bucket_info.name, Prefix=f"neuron_checkpoint_{neuron_hotkey}_"
            ):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    match = re.match(regex_pattern, key)
                    if match:
                        block_number = int(match.group(1))
                        if block_number > latest_block_number:
                            latest_block_number = block_number
                            latest_filename = key
                            file_size = obj["Size"]

            if not latest_filename:
                logger.info(f"No valid checkpoints found for neuron {neuron_hotkey}")
                return None

            logger.info(
                f"Found latest checkpoint: {latest_filename} ({format_size(file_size)})"
            )
            local_checkpoint_path = os.path.join(checkpoint_dir, latest_filename)
            temp_path = f"{local_checkpoint_path}.temp"
            lock_path = f"{local_checkpoint_path}.lock"

            # Check if file already exists and is complete
            if os.path.exists(local_checkpoint_path):
                if os.path.getsize(local_checkpoint_path) == file_size:
                    logger.info(
                        f"Checkpoint already exists and is complete: {local_checkpoint_path}"
                    )
                    return local_checkpoint_path

            # Try to acquire lock
            try:
                with open(lock_path, "x") as _:  # Atomic file creation
                    logger.info(f"Acquired lock for downloading: {lock_path}")
            except FileExistsError:
                # Another process is downloading, wait for it
                logger.info("Another process is downloading, waiting...")
                for _ in range(30):  # Wait up to 30 seconds
                    await asyncio.sleep(1)
                    if os.path.exists(local_checkpoint_path):
                        if os.path.getsize(local_checkpoint_path) == file_size:
                            logger.info("File downloaded by another process")
                            try:
                                os.remove(lock_path)  # Try to clean up lock
                            except OSError as e:
                                logger.warning(f"Failed to remove lock file: {e}")
                            return local_checkpoint_path
                logger.warning(
                    "Timeout waiting for other process, proceeding with download"
                )

            try:
                # Download chunks
                chunks_data = {}
                downloaded_size = 0
                semaphore = asyncio.Semaphore(max_concurrent_downloads)
                total_chunks = (file_size + chunk_size - 1) // chunk_size

                async def download_chunk(chunk_number: int):
                    start = chunk_number * chunk_size
                    end = min(start + chunk_size, file_size)

                    for attempt in range(max_retries):
                        try:
                            async with semaphore:
                                response = await s3_client.get_object(
                                    Bucket=bucket_info.name,
                                    Key=latest_filename,
                                    Range=f"bytes={start}-{end-1}",
                                )
                                chunk_data = await response["Body"].read()

                                nonlocal downloaded_size
                                downloaded_size += len(chunk_data)
                                progress = (downloaded_size / file_size) * 100

                                if chunk_number % 5 == 0 or progress >= 100:
                                    elapsed_time = time.time() - start_time
                                    speed = downloaded_size / (
                                        1024 * 1024 * elapsed_time
                                    )  # MB/s
                                    progress_bar = create_progress_bar(
                                        progress, file_size
                                    )
                                    logger.info(
                                        f"\nDownload Progress: {progress_bar} [{speed:.2f} MB/s]"
                                    )

                                chunks_data[chunk_number] = chunk_data
                                return True

                        except Exception as e:
                            if attempt == max_retries - 1:
                                logger.error(
                                    f"Failed to download chunk {chunk_number}: {str(e)}"
                                )
                                return False
                            await asyncio.sleep(retry_delay * (attempt + 1))

                # Download all chunks
                tasks = [download_chunk(i) for i in range(total_chunks)]
                results = await asyncio.gather(*tasks)

                if not all(results):
                    raise Exception("Some chunks failed to download")

                # Write chunks to temp file
                logger.info("Writing chunks to temp file...")
                async with aiofiles.open(temp_path, "wb") as f:
                    for chunk_num in range(total_chunks):
                        if chunk_num in chunks_data:
                            await f.write(chunks_data[chunk_num])
                        else:
                            raise Exception(f"Missing chunk {chunk_num}")

                await asyncio.sleep(0.5)  # Short delay for file system

                # Verify the temp file
                if not os.path.exists(temp_path):
                    raise Exception(f"Temp file not found at: {temp_path}")

                actual_size = os.path.getsize(temp_path)
                if actual_size != file_size:
                    raise Exception(
                        f"Size mismatch in temp file: expected {file_size}, got {actual_size}"
                    )

                # Move to final location with extra verification
                logger.info(
                    f"Moving temp file to final location: {local_checkpoint_path}"
                )

                # Remove destination file if it exists
                if os.path.exists(local_checkpoint_path):
                    logger.info(
                        f"Removing existing checkpoint file: {local_checkpoint_path}"
                    )
                    os.remove(local_checkpoint_path)

                try:
                    # Use shutil.move for more reliable cross-device moves
                    shutil.move(temp_path, local_checkpoint_path)

                    # Verify the move
                    if not os.path.exists(local_checkpoint_path):
                        raise Exception(
                            "Move operation failed - destination file doesn't exist"
                        )

                    # Double check the source file is gone
                    if os.path.exists(temp_path):
                        logger.warning(
                            "Temp file still exists after move, attempting cleanup"
                        )
                        try:
                            os.remove(temp_path)
                        except Exception as e:
                            logger.warning(f"Failed to cleanup temp file: {e}")

                    # Final size verification
                    final_size = os.path.getsize(local_checkpoint_path)
                    if final_size != file_size:
                        raise Exception(
                            f"Size mismatch in final file: expected {file_size}, got {final_size}"
                        )

                    # Extra verification - try to open the file
                    with open(local_checkpoint_path, "rb") as f:
                        # Read first few bytes to verify file is accessible
                        f.read(1024)

                    logger.info("Move operation successful and verified")

                    total_time = time.time() - start_time
                    avg_speed = (file_size / (1024 * 1024)) / total_time  # MB/s
                    logger.info(
                        f"Successfully downloaded checkpoint to: {local_checkpoint_path}"
                    )
                    logger.info(
                        f"Download completed in {total_time:.2f} seconds ({avg_speed:.2f} MB/s average)"
                    )

                    return local_checkpoint_path

                except Exception as move_e:
                    logger.error(f"Error during move operation: {str(move_e)}")
                    # Try to recover the temp file if move failed
                    if os.path.exists(temp_path) and not os.path.exists(
                        local_checkpoint_path
                    ):
                        try:
                            shutil.copy2(temp_path, local_checkpoint_path)
                            logger.info("Recovered file using copy operation")
                        except Exception as recover_e:
                            logger.error(f"Failed to recover file: {str(recover_e)}")
                    raise

            except Exception as e:
                logger.error(f"Error during file operations: {str(e)}")
                # Cleanup both temp and final files if they exist
                for filepath in [temp_path, local_checkpoint_path]:
                    if filepath and os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                            logger.info(f"Cleaned up file: {filepath}")
                        except Exception as rm_e:
                            logger.error(
                                f"Failed to cleanup file {filepath}: {str(rm_e)}"
                            )
                return None

            finally:
                # Clean up lock file
                try:
                    os.remove(lock_path)
                except Exception as e:
                    logger.warning(f"Failed to remove lock file: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None


def get_all_buckets(
    netuid: int,
    metagraph,
    config,
) -> List[Optional[Union[str, Bucket]]]:
    """
    Retrieves and parses all bucket commitments from the network.
    """
    buckets = []
    commitments = get_all_commitments(
        netuid=netuid,
        metagraph=metagraph,
        config=config,
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
    metagraph, buckets: List[Optional[Union[str, Bucket]]]
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
        highest_stake_hotkey = get_neuron_with_highest_stake(
            metagraph=metagraph, buckets=buckets
        )

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

    async def _save_checkpoint_async(
        self, global_step: int, block_number: int, **kwargs
    ):
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
                    CacheControl="no-cache, no-store, must-revalidate",
                )
                upload_id = response["UploadId"]
                logger.info(f"Initiated multipart upload with ID: {upload_id}")

                try:
                    total_size = os.path.getsize(self.checkpoint_path)
                    total_parts = (total_size + chunk_size - 1) // chunk_size
                    parts = {}  # Use dict to track parts by number
                    uploaded_size = 0
                    semaphore = asyncio.Semaphore(max_concurrent_uploads)
                    upload_tasks = []
                    failed_parts = set()

                    # Initialize progress bar
                    pbar = tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc="Uploading checkpoint",
                    )

                    async def upload_part_with_retry(
                        part_number: int, offset: int
                    ) -> dict:
                        """Upload a single part with retries and verification."""
                        for attempt in range(max_retries):
                            try:
                                async with semaphore:
                                    async with aiofiles.open(
                                        self.checkpoint_path, "rb"
                                    ) as f:
                                        await f.seek(offset)
                                        chunk = await f.read(
                                            min(chunk_size, total_size - offset)
                                        )

                                        response = await s3_client.upload_part(
                                            Bucket=bucket,
                                            Key=filename,
                                            PartNumber=part_number,
                                            UploadId=upload_id,
                                            Body=chunk,
                                        )

                                        # Verify part upload
                                        part_size = len(chunk)
                                        if part_size == 0:
                                            raise ValueError(
                                                f"Zero-size chunk for part {part_number}"
                                            )

                                        pbar.update(part_size)

                                        return {
                                            "PartNumber": part_number,
                                            "ETag": response["ETag"],
                                            "Size": part_size,
                                        }
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    logger.warning(
                                        f"Retry {attempt + 1}/{max_retries} for part {part_number}: {str(e)}"
                                    )
                                    await asyncio.sleep(retry_delay)
                                else:
                                    failed_parts.add(part_number)
                                    raise

                    # Create upload tasks for all parts
                    for part_number in range(1, total_parts + 1):
                        offset = (part_number - 1) * chunk_size
                        task = asyncio.create_task(
                            upload_part_with_retry(part_number, offset)
                        )
                        upload_tasks.append(task)

                    # Wait for all uploads and collect results
                    completed_parts = await asyncio.gather(
                        *upload_tasks, return_exceptions=True
                    )

                    # Close progress bar
                    pbar.close()

                    # Process results and check for failures
                    for part in completed_parts:
                        if isinstance(part, Exception):
                            raise Exception(f"Part upload failed: {str(part)}")
                        parts[part["PartNumber"]] = part

                    # Verify all parts are present and ordered
                    if len(parts) != total_parts:
                        missing_parts = set(range(1, total_parts + 1)) - set(
                            parts.keys()
                        )
                        raise Exception(f"Missing parts: {missing_parts}")

                    # Sort parts for completion
                    ordered_parts = [parts[i] for i in range(1, total_parts + 1)]

                    # Complete multipart upload
                    completion_response = await s3_client.complete_multipart_upload(
                        Bucket=bucket,
                        Key=filename,
                        UploadId=upload_id,
                        MultipartUpload={
                            "Parts": [
                                {"PartNumber": p["PartNumber"], "ETag": p["ETag"]}
                                for p in ordered_parts
                            ]
                        },
                    )

                    # Verify upload completion
                    try:
                        head_response = await s3_client.head_object(
                            Bucket=bucket, Key=filename
                        )
                        if head_response["ContentLength"] != total_size:
                            raise Exception(
                                f"Size mismatch: uploaded={head_response['ContentLength']}, expected={total_size}"
                            )

                        logger.info(
                            f"Successfully verified upload of {filename} ({total_size} bytes)"
                        )
                    except Exception as e:
                        raise Exception(f"Upload verification failed: {str(e)}")

                except Exception as e:
                    logger.error(f"Error during upload: {str(e)}")
                    try:
                        await s3_client.abort_multipart_upload(
                            Bucket=bucket, Key=filename, UploadId=upload_id
                        )
                        logger.info(f"Aborted multipart upload {upload_id}")
                    except Exception as abort_e:
                        logger.error(
                            f"Failed to abort multipart upload: {str(abort_e)}"
                        )
                    raise

        except Exception as e:
            logger.exception(f"Failed to upload checkpoint: {e}")
            raise

        finally:
            # Clean up any remaining tasks
            if "upload_tasks" in locals():
                for task in upload_tasks:
                    if not task.done():
                        task.cancel()

    async def _cleanup_old_checkpoints_async(self, max_checkpoints=3):
        """
        Asynchronously deletes old checkpoints locally and in S3.
        Keeps only the latest 'max_checkpoints'.
        """
        logger.info(
            f"Starting checkpoint cleanup, keeping latest {max_checkpoints} checkpoints"
        )
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
                logger.info(
                    f"Attempting to delete checkpoint from block {block_num} at {filepath}"
                )
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

        logger.info(
            f"Connecting to S3 endpoint: {get_base_url(BUCKET_SECRETS['account_id'])}"
        )
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
                    {"Key": os.path.basename(filepath)}
                    for _, filepath in old_checkpoints
                ],
                "Quiet": True,
            }
            logger.info(
                f"Prepared delete request for {len(delete_objects['Objects'])} objects"
            )

            if delete_objects["Objects"]:
                try:
                    logger.info(
                        f"Attempting to delete objects: {[obj['Key'] for obj in delete_objects['Objects']]}"
                    )
                    response = await s3_client.delete_objects(
                        Bucket=bucket, Delete=delete_objects
                    )
                    logger.info("Successfully initiated deletion request")
                    logger.info(
                        f"Deleted old checkpoints from S3: {delete_objects['Objects']}"
                    )
                    logger.info(f"S3 deletion response: {response}")

                    if "Deleted" in response:
                        logger.info(
                            f"Successfully deleted {len(response['Deleted'])} objects"
                        )
                    if "Errors" in response:
                        logger.warning(
                            f"Failed to delete {len(response['Errors'])} objects: {response['Errors']}"
                        )

                except Exception as e:
                    logger.error(f"Failed to delete old checkpoints from S3: {str(e)}")
                    logger.error(
                        f"Full error details: {e.__class__.__name__}: {str(e)}"
                    )

    async def load_from_highest_stake(
        self,
        metagraph,
        buckets,
        optimizer,
        scheduler,
        is_validator: bool = False,
        hparams=None,
    ) -> int:
        """
        Attempts to load checkpoint from the highest stake neuron.
        """
        try:
            await self.cleanup_old_version_checkpoints()
            highest_stake_hotkey = get_neuron_with_highest_stake(
                metagraph=metagraph, buckets=buckets
            )

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
                            optimizer=optimizer,
                            scheduler=scheduler,
                            is_validator=is_validator,
                            hparams=hparams,
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

    async def cleanup_old_version_checkpoints(self, keep_latest: bool = True) -> None:
        """
        Cleans up checkpoint files that don't match the current version number.
        Handles non-existent directories and empty paths gracefully.

        Args:
            keep_latest (bool): If True, keeps the latest checkpoint from old versions
                            as a backup. Defaults to True.
        """
        try:
            checkpoint_dir = os.path.dirname(self.checkpoint_path)

            # Check if directory exists
            if not os.path.exists(checkpoint_dir):
                logger.debug(f"Checkpoint directory does not exist: {checkpoint_dir}")
                return

            pattern = os.path.join(
                checkpoint_dir,
                f"neuron_checkpoint_{self.wallet.hotkey.ss58_address}_b*_v*.pth",
            )

            # Get list of checkpoint files
            checkpoint_files = await asyncio.to_thread(glob.glob, pattern)
            if not checkpoint_files:
                logger.debug(f"No checkpoint files found in {checkpoint_dir}")
                return

            # Group checkpoints by version
            version_groups = {}
            for filepath in checkpoint_files:
                if not os.path.exists(filepath):  # Check if file still exists
                    continue

                filename = os.path.basename(filepath)
                match = re.match(
                    rf"neuron_checkpoint_{self.wallet.hotkey.ss58_address}_b(\d+)_v(.+)\.pth",
                    filename,
                )
                if match:
                    block_number = int(match.group(1))
                    version = match.group(2)
                    if version not in version_groups:
                        version_groups[version] = []
                    version_groups[version].append((block_number, filepath))

            if not version_groups:
                logger.debug("No valid checkpoint files found")
                return

            # Identify files to delete
            to_delete = []
            for version, checkpoints in version_groups.items():
                if version != __version__:  # If not current version
                    if keep_latest:
                        # Sort by block number and keep only the latest
                        checkpoints.sort(key=lambda x: x[0], reverse=True)
                        to_delete.extend(filepath for _, filepath in checkpoints[1:])
                    else:
                        # Delete all checkpoints of old versions
                        to_delete.extend(filepath for _, filepath in checkpoints)

            if not to_delete:
                logger.debug("No old version checkpoints to clean up")
                return

            # Delete files
            deleted_count = 0
            for filepath in to_delete:
                try:
                    if os.path.exists(
                        filepath
                    ):  # Double check file exists before deletion
                        await asyncio.to_thread(os.remove, filepath)
                        deleted_count += 1
                        logger.info(f"Deleted old version checkpoint: {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint {filepath}: {e}")

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old version checkpoint(s)")

        except Exception as e:
            logger.error(f"Error during checkpoint cleanup: {e}")

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


async def load_model_for_eval(
    metagraph,
    buckets: List[Optional[Union[str, Bucket]]],
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = "cuda",
) -> tuple[int, int]:  # Return (global_step, block_number)
    """
    Simplified checkpoint loader that only loads model state for evaluation.
    Returns tuple of (global_step, block_number).
    """
    try:
        # Get highest stake neuron
        highest_stake_hotkey = get_neuron_with_highest_stake(metagraph, buckets)
        if not highest_stake_hotkey:
            logger.warning("No neurons found. Starting from scratch.")
            return 0, 0

        uid = metagraph.hotkeys.index(highest_stake_hotkey)
        bucket_info = buckets[uid]

        if bucket_info:
            # Download checkpoint
            checkpoint_dir = os.path.dirname(checkpoint_path)
            await asyncio.to_thread(os.makedirs, checkpoint_dir, exist_ok=True)

            checkpoint_file = await download_checkpoint_from_neuron(
                bucket_info=bucket_info,
                neuron_hotkey=highest_stake_hotkey,
                checkpoint_dir=checkpoint_dir,
            )

            if checkpoint_file:
                # Parse block number from filename
                regex_pattern = rf"neuron_checkpoint_{highest_stake_hotkey}_b(\d+)_v({re.escape(__version__)})\.pth"
                match = re.match(regex_pattern, os.path.basename(checkpoint_file))
                if not match:
                    logger.warning(
                        f"Could not parse block number from checkpoint filename: {checkpoint_file}"
                    )
                    return 0, 0

                block_number = int(match.group(1))

                # Load only model state
                checkpoint = torch.load(
                    checkpoint_file, map_location=device, weights_only=True
                )
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    global_step = checkpoint.get("global_step", 0)
                    logger.info(
                        f"Loaded model state at global step {global_step} from block {block_number}"
                    )
                    return global_step, block_number

            logger.warning("Failed to download or load checkpoint")
            return 0, 0

        logger.warning(f"No bucket info for neuron {highest_stake_hotkey}")
        return 0, 0

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return 0, 0
