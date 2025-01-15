import os
import re
import time
import torch
import asyncio
import aiofiles
import tempfile
import bittensor as bt
from typing import List, Dict, Optional
from types import SimpleNamespace
from aiobotocore.session import get_session
from . import __version__
from .config import client_config, BUCKET_SECRETS
from .chain import ChainManager
from .schemas import Bucket

import tplr as tplr
import botocore
import tqdm
import tqdm.asyncio
import numpy as np
import psutil
import mmap
import zstandard as zstd

# Constants
CF_REGION_NAME: str = "enam"
LOCAL_TMP_DIR = "/tmp/local_store"


class Comms(ChainManager):
    def __init__(
        self,
        wallet: "bt.wallet",
        save_location: str = "/tmp",
        key_prefix: str = "model",
        config=None,
        netuid=None,
        metagraph=None,
        hparams=None,
        uid=None,
        **kwargs,
    ):
        self.wallet = wallet
        self.uid = uid
        # Create temp directory for this instance
        self.temp_dir = os.path.join("/tmp", f"templar_{self.uid}")
        os.makedirs(self.temp_dir, exist_ok=True)
        # Get the bucket directly
        self.bucket = self.get_own_bucket()
        # Now initialize ChainManager with the bucket
        super().__init__(
            config=config,
            netuid=netuid,
            metagraph=metagraph,
            hparams=hparams,
            wallet=self.wallet,
            bucket=self.bucket,
        )

        # Use the hotkey directly in the save_location
        hotkey = self.wallet.hotkey.ss58_address
        self.save_location = os.path.join("/tmp", f"hotkey_{hotkey}")
        os.makedirs(self.save_location, exist_ok=True)
        self.key_prefix = key_prefix
        self.session = get_session()
        self.lock = asyncio.Lock()
        self.active_peers = set()  # Set to store active peers
        self.active_check_interval = (
            self.hparams.active_check_interval
        )  # Interval in seconds
        self.recent_windows = (
            self.hparams.recent_windows
        )  # Number of recent windows to check

    def start_background_tasks(self):
        self.loop = asyncio.get_running_loop()
        # Start background tasks
        self.loop.create_task(self.track_active_peers())

    def get_own_bucket(self) -> Bucket:
        """Gets bucket configuration from environment variables via config.BUCKET_SECRETS."""
        try:
            # Create a Bucket object using write credentials from BUCKET_SECRETS
            bucket = Bucket(
                name=BUCKET_SECRETS["account_id"],
                account_id=BUCKET_SECRETS["account_id"],
                access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
                secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
            )
            tplr.logger.debug(f"Created bucket from environment: {bucket}")
            return bucket

        except KeyError as e:
            tplr.logger.error(f"Missing required R2 configuration: {e}")
            raise
        except Exception as e:
            tplr.logger.error(f"Error creating bucket: {e}")
            raise

    def get_base_url(self, account_id):
        """Constructs the base URL for the R2 storage endpoint."""
        return f"https://{account_id}.r2.cloudflarestorage.com"

    def delete_local_directory(self, path: str):
        """Safely remove a local directory and all its contents."""
        if not os.path.exists(path):
            return
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(path)

    # Convert all the existing functions to methods
    async def cleanup_local_data(
        self, uid: str, current_window: int, stale_retention: int
    ):
        """Clean up stale local data for a given uid."""
        user_dir = os.path.join(LOCAL_TMP_DIR, str(uid))
        if not os.path.exists(user_dir):
            return

        min_allowed_window = current_window - stale_retention
        for wdir in os.listdir(user_dir):
            if wdir.isdigit():
                w = int(wdir)
                if w < min_allowed_window:
                    old_path = os.path.join(user_dir, wdir)
                    tplr.logger.debug(f"Removing stale local directory: {old_path}")
                    try:
                        self.delete_local_directory(old_path)
                    except Exception as e:
                        tplr.logger.debug(
                            f"Error removing stale directory {old_path}: {e}"
                        )

    async def cleanup_s3_data(
        self, uid: str, current_window: int, stale_retention: int
    ):
        """Clean up stale S3 data for a given uid."""
        min_allowed_window = current_window - stale_retention
        prefix = f"{uid}/"

        session = get_session()
        async with session.create_client(
            "s3",
            endpoint_url=self.get_base_url(BUCKET_SECRETS["account_id"]),
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
            aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
        ) as s3_client:
            continuation_token = None

            while True:
                list_args = {
                    "Bucket": BUCKET_SECRETS["bucket_name"],
                    "Prefix": prefix,
                    "MaxKeys": 1000,
                }
                if continuation_token:
                    list_args["ContinuationToken"] = continuation_token

                response = await s3_client.list_objects_v2(**list_args)
                contents = response.get("Contents", [])

                # Identify stale objects to delete
                stale_objects = []
                for obj in contents:
                    key = obj["Key"]
                    # Key format: uid/window/key
                    parts = key.split("/")
                    if len(parts) < 2:
                        continue
                    try:
                        w = int(parts[1])
                    except ValueError:
                        continue

                    if w < min_allowed_window:
                        stale_objects.append({"Key": key})

                # Batch delete stale objects
                if stale_objects:
                    tplr.logger.debug(
                        f"Removing stale S3 objects for {uid}: {stale_objects}"
                    )
                    await s3_client.delete_objects(
                        Bucket=BUCKET_SECRETS["bucket_name"],
                        Delete={"Objects": stale_objects},
                    )

                if response.get("IsTruncated"):
                    continuation_token = response.get("NextContinuationToken")
                else:
                    break

    async def s3_put_object(self, key: str, data: bytes | str | os.PathLike):
        """Upload object to S3 with automatic large file handling."""
        try:
            # Convert file path to bytes if needed
            if isinstance(data, (str, os.PathLike)):
                with open(data, "rb") as f:
                    data = f.read()

            file_size = len(data) if isinstance(data, bytes) else os.path.getsize(data)

            # Use regular upload for files under 5GB
            if file_size <= 5 * 1024 * 1024 * 1024:
                async with self.session.create_client(
                    "s3",
                    endpoint_url=self.get_base_url(BUCKET_SECRETS["account_id"]),
                    region_name=CF_REGION_NAME,
                    config=client_config,
                    aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
                    aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
                ) as s3_client:
                    await s3_client.put_object(
                        Bucket=BUCKET_SECRETS["bucket_name"], Key=key, Body=data
                    )
                return

            # For large files, use CPU-only parallel multipart upload with compression
            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(BUCKET_SECRETS["account_id"]),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
                aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
            ) as s3_client:
                # Calculate optimal chunk size
                cpu_count = os.cpu_count()
                memory_available = psutil.virtual_memory().available
                optimal_chunk_size = min(
                    max(
                        64 * 1024 * 1024,  # Min 64MB
                        file_size // (cpu_count * 2),
                    ),  # Split across cores
                    memory_available // (cpu_count * 4),  # Keep memory in check
                )

                # Initialize multipart upload
                response = await s3_client.create_multipart_upload(
                    Bucket=BUCKET_SECRETS["bucket_name"], Key=key
                )
                upload_id = response["UploadId"]

                try:
                    total_chunks = (
                        file_size + optimal_chunk_size - 1
                    ) // optimal_chunk_size

                    async def upload_chunk(part_number: int, retries: int = 3) -> dict:
                        start = part_number * optimal_chunk_size
                        end = min(start + optimal_chunk_size, file_size)
                        chunk = data[start:end]

                        compressor = zstd.ZstdCompressor(
                            level=3
                        )  # Balance between speed and compression
                        compressed_chunk = compressor.compress(chunk)

                        for attempt in range(retries):
                            try:
                                response = await s3_client.upload_part(
                                    Bucket=BUCKET_SECRETS["bucket_name"],
                                    Key=key,
                                    PartNumber=part_number + 1,
                                    UploadId=upload_id,
                                    Body=compressed_chunk,
                                )
                                return {
                                    "PartNumber": part_number + 1,
                                    "ETag": response["ETag"],
                                }
                            except Exception as e:
                                if attempt == retries - 1:
                                    raise
                                await asyncio.sleep(2**attempt)  # Exponential backoff

                    # Upload chunks in parallel with progress bar
                    tasks = []
                    pbar = tqdm.tqdm(
                        total=total_chunks, desc=f"Uploading {key}", unit="chunks"
                    )

                    for i in range(total_chunks):
                        task = asyncio.create_task(upload_chunk(i))
                        task.add_done_callback(lambda _: pbar.update(1))
                        tasks.append(task)

                    parts = await asyncio.gather(*tasks)
                    pbar.close()

                    # Complete upload
                    parts.sort(key=lambda x: x["PartNumber"])
                    await s3_client.complete_multipart_upload(
                        Bucket=BUCKET_SECRETS["bucket_name"],
                        Key=key,
                        UploadId=upload_id,
                        MultipartUpload={"Parts": parts},
                    )

                except Exception as e:
                    # Abort multipart upload on failure
                    await s3_client.abort_multipart_upload(
                        Bucket=BUCKET_SECRETS["bucket_name"],
                        Key=key,
                        UploadId=upload_id,
                    )
                    raise e

        except Exception as e:
            tplr.logger.error(f"Error in s3_put_object for {key}: {e}")
            raise

    async def s3_get_object(
        self, key: str, timeout: int = 30, bucket_secrets: Optional[dict] = None
    ) -> Optional[dict]:
        """Download object from S3 with automatic large file handling."""
        try:
            if bucket_secrets is None:
                raise ValueError("bucket_secrets is required")

            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(bucket_secrets["account_id"]),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=bucket_secrets["write"]["access_key_id"],
                aws_secret_access_key=bucket_secrets["write"]["secret_access_key"],
            ) as s3_client:
                # Check if file exists and get size
                try:
                    head = await s3_client.head_object(
                        Bucket=BUCKET_SECRETS["bucket_name"], Key=key
                    )
                    file_size = head["ContentLength"]
                except botocore.exceptions.ClientError:
                    return None

                # For small files (< 5GB), use direct download
                if file_size <= 5 * 1024 * 1024 * 1024:
                    response = await asyncio.wait_for(
                        s3_client.get_object(
                            Bucket=BUCKET_SECRETS["bucket_name"], Key=key
                        ),
                        timeout=timeout,
                    )
                    data = await response["Body"].read()

                    # Load state dict
                    with tempfile.NamedTemporaryFile(
                        delete=True, suffix=".pt"
                    ) as temp_file:
                        temp_file.write(data)
                        temp_file.flush()
                        return torch.load(temp_file.name, weights_only=True)

                # For large files, use parallel multipart download with compression
                cpu_count = os.cpu_count()
                memory_available = psutil.virtual_memory().available
                optimal_chunk_size = min(
                    max(
                        16 * 1024 * 1024,  # Min 16MB chunks
                        file_size // (cpu_count * 2),
                    ),
                    memory_available // (cpu_count * 4),
                )

                total_parts = (file_size + optimal_chunk_size - 1) // optimal_chunk_size
                shared_buffer = bytearray(file_size)

                async def download_chunk(part: int, retries: int = 3) -> None:
                    start = part * optimal_chunk_size
                    end = min(start + optimal_chunk_size, file_size)
                    for attempt in range(retries):
                        try:
                            response = await s3_client.get_object(
                                Bucket=BUCKET_SECRETS["bucket_name"],
                                Key=key,
                                Range=f"bytes={start}-{end - 1}",
                            )
                            chunk = await response["Body"].read()

                            # Decompress chunk if using GPU
                            if torch.cuda.is_available():
                                decompressor = zstd.ZstdDecompressor()
                                decompressed_chunk = decompressor.decompress(chunk)
                                chunk_tensor = torch.frombuffer(
                                    decompressed_chunk, dtype=torch.uint8
                                ).cuda()
                                shared_buffer[start : start + len(chunk_tensor)].copy_(
                                    chunk_tensor
                                )
                            else:
                                # For CPU buffer
                                decompressor = zstd.ZstdDecompressor()
                                decompressed_chunk = decompressor.decompress(chunk)
                                shared_buffer[
                                    start : start + len(decompressed_chunk)
                                ] = decompressed_chunk
                            return
                        except Exception as e:
                            if attempt == retries - 1:
                                raise
                            await asyncio.sleep(2**attempt)  # Exponential backoff

                # Download chunks in parallel with semaphore and progress bar
                semaphore = asyncio.Semaphore(min(32, cpu_count * 4))

                async def bounded_download(part: int):
                    async with semaphore:
                        await download_chunk(part)

                try:
                    tasks = []
                    pbar = tqdm.tqdm(
                        total=total_parts, desc=f"Downloading {key}", unit="chunks"
                    )

                    for i in range(total_parts):
                        task = asyncio.create_task(bounded_download(i))
                        task.add_done_callback(lambda _: pbar.update(1))
                        tasks.append(task)

                    await asyncio.gather(*tasks)
                    pbar.close()

                    # Convert buffer to bytes and load state dict
                    with tempfile.NamedTemporaryFile(
                        delete=True, suffix=".pt"
                    ) as temp_file:
                        temp_file.write(bytes(shared_buffer))
                        temp_file.flush()
                        return torch.load(temp_file.name, weights_only=True)

                except Exception as e:
                    raise e

        except asyncio.TimeoutError:
            tplr.logger.debug(f"Timeout occurred while downloading {key}")
            return None
        except Exception as e:
            tplr.logger.debug(f"Error in s3_get_object for {key}: {e}")
            return None

    async def put(
        self,
        state_dict: dict,
        uid: str,
        window: int,
        key: str,
        global_step: int = 0,
        local: bool = True,
        stale_retention: int = 10,
    ):
        """PUT operation: Store the state_dict and global_step."""
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        tplr.logger.debug(f"PUT {filename} -->")

        # Create per-uid temp directory
        temp_dir = os.path.join("/tmp", str(self.uid))
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"temp_{filename}")

        try:
            # Prepare the data to be saved
            if key == "checkpoint":
                save_data = state_dict
            else:
                save_data = {
                    "state_dict": state_dict,
                    "global_step": global_step,
                }

            # Save to temp file
            torch.save(save_data, temp_file_path)

            if local:
                # Local storage with per-uid directories
                await self.cleanup_local_data(
                    uid=uid, current_window=window, stale_retention=stale_retention
                )
                local_dir = os.path.join(LOCAL_TMP_DIR, str(uid), str(window))
                os.makedirs(local_dir, exist_ok=True)
                final_path = os.path.join(local_dir, filename)
                os.replace(temp_file_path, final_path)
            else:
                # Remote storage with automatic handling of large files
                await self.cleanup_s3_data(
                    uid=uid, current_window=window, stale_retention=stale_retention
                )
                await self.s3_put_object(filename, temp_file_path)

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        tplr.logger.debug(f"PUT {filename} <--")

    async def get(
        self,
        uid: str,
        window: int,
        key: str,
        timeout: int = 5,
        local: bool = True,
        stale_retention: int = 10,
    ) -> Optional[dict]:
        """GET operation with automatic large file handling."""
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        tplr.logger.debug(f"GET {filename} -->")

        try:
            # Use self.uid for temp directory
            temp_dir = os.path.join("/tmp", str(self.uid))
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, f"temp_{filename}")

            if local:
                # Local storage with per-uid directories
                await self.cleanup_local_data(
                    uid=uid, current_window=window, stale_retention=stale_retention
                )
                local_path = os.path.join(
                    LOCAL_TMP_DIR, str(uid), str(window), filename
                )
                if not os.path.exists(local_path):
                    return None
                loaded_data = torch.load(local_path, weights_only=True)
            else:
                # Get peer bucket info for remote storage
                peer_bucket = self.commitments.get(int(uid))
                if not peer_bucket:
                    tplr.logger.debug(f"No bucket commitment found for UID {uid}")
                    return None

                # Remote storage with automatic handling of large files
                loaded_data = await self.s3_get_object(
                    filename,
                    timeout=timeout,
                    bucket_secrets={
                        "account_id": peer_bucket.account_id,
                        "bucket_name": peer_bucket.name,
                        "write": {
                            "access_key_id": peer_bucket.access_key_id,
                            "secret_access_key": peer_bucket.secret_access_key,
                        },
                    },
                )
                if loaded_data is None:
                    return None

                # For checkpoints, save to temp file first
                if key == "checkpoint":
                    torch.save(loaded_data, temp_file_path)
                    loaded_data = torch.load(temp_file_path, weights_only=True)

            # Handle checkpoint vs regular data
            if key == "checkpoint":
                return loaded_data, None
            return loaded_data.get("state_dict"), loaded_data.get("global_step", 0)

        except Exception as e:
            tplr.logger.debug(f"GET error {filename}: {e}")
            return None
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    async def get_with_retry(
        self,
        uid: str,
        window: int,
        key: str,
        timeout: int,
        local: bool = True,
        stale_retention: int = 10,
    ) -> Optional[dict]:
        """GET with retry operation."""
        start_time = time.time()
        end_time = start_time + timeout

        while True:
            if time.time() >= end_time:
                tplr.logger.debug(f"GET {uid}/{window}/{key} timed out.")
                return None

            state_dict = await self.get(
                uid=uid,
                window=window,
                key=key,
                local=local,
                stale_retention=stale_retention,
            )
            if state_dict is not None:
                return state_dict

            # Retry after a short delay
            await asyncio.sleep(0.1)

    async def gather(
        self,
        state_dict: Optional[Dict[str, torch.Tensor]],
        my_uid: str,
        uids: List[str],
        window: int,
        key: str,
        timeout: int,
        device: str,
        global_step: int,
        local: bool = True,
        stale_retention: int = 10,
    ) -> Optional[SimpleNamespace]:
        """Gather operation."""
        start_time = time.time()
        metrics = {"upload_bytes": 0, "download_bytes": 0, "successes": []}

        # Put own state if provided
        if my_uid is not None and state_dict is not None:
            await self.put(
                state_dict=state_dict,
                uid=str(my_uid),
                window=window,
                key=key,
                global_step=global_step,
                local=local,
                stale_retention=stale_retention,
            )
            metrics["upload_bytes"] += sum(
                tensor.element_size() * tensor.nelement()
                for tensor in state_dict.values()
            )

        # Small delay to ensure data propagation
        await asyncio.sleep(0.1)

        # Prepare gather tasks
        gather_tasks = [
            self.get_with_retry(
                uid=uid,
                window=window,
                key=key,
                timeout=timeout,
                local=local,
                stale_retention=stale_retention,
            )
            for uid in uids
        ]

        # Initialize variables
        aggregated_state_dict = {}
        valid_uids = []
        global_steps = []

        # Process responses
        responses = await asyncio.gather(*gather_tasks)
        for idx, response in enumerate(responses):
            uid = uids[idx]

            # Skip if no response
            if response is None:
                tplr.logger.debug(f"No data received from UID {uid}")
                continue

            try:
                state_dict_resp, global_step_resp = response
            except (TypeError, ValueError) as e:
                tplr.logger.debug(f"Invalid response format from UID {uid}: {e}")
                continue

            # Skip if no state dict
            if state_dict_resp is None:
                tplr.logger.debug(f"Empty state dict from UID {uid}")
                continue

            valid_uids.append(uid)
            global_steps.append(global_step_resp)

            # Add tensors to aggregated_state_dict
            for param_name, tensor in state_dict_resp.items():
                if param_name not in aggregated_state_dict:
                    aggregated_state_dict[param_name] = []
                aggregated_state_dict[param_name].append(tensor.to(device))
                metrics["download_bytes"] += tensor.element_size() * tensor.nelement()

        # If no valid responses, return None
        if not valid_uids:
            tplr.logger.info("No valid gradients received from any UID")
            return None

        # Create result namespace
        result = SimpleNamespace(
            time=time.time() - start_time,
            upload_bytes=metrics["upload_bytes"],
            download_bytes=metrics["download_bytes"],
            success_rate=len(valid_uids) / len(uids),
            state_dict=SimpleNamespace(**aggregated_state_dict),
            uids=valid_uids,
            global_steps=global_steps,
        )

        tplr.logger.debug(f"Successfully gathered from UIDs: {valid_uids}")
        return result

    async def upload_large_file(self, file_path: str, filename: str) -> bool:
        """
        Uploads a large file to R2 using parallel multipart upload with shared memory.

        Optimizations:
        - Uses shared memory for inter-process communication
        - Parallel chunk processing with optimal chunk sizes
        - Memory-mapped file reading for large files
        - O(1) memory usage regardless of file size
        """
        try:
            # Calculate optimal chunk size based on system memory and CPU cores
            file_size = os.path.getsize(file_path)
            cpu_count = os.cpu_count()
            memory_available = psutil.virtual_memory().available
            optimal_chunk_size = min(
                max(
                    64 * 1024 * 1024,  # Minimum 64MB
                    file_size // (cpu_count * 2),
                ),  # Split across cores
                memory_available // (cpu_count * 4),  # Keep memory usage in check
            )

            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(self.bucket.account_id),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=self.bucket.access_key_id,
                aws_secret_access_key=self.bucket.secret_access_key,
            ) as s3_client:
                # Initialize multipart upload - O(1)
                response = await s3_client.create_multipart_upload(
                    Bucket=self.bucket.name, Key=filename
                )
                upload_id = response["UploadId"]

                # Use memory mapping for large files - O(1) memory
                with mmap.mmap(
                    os.open(file_path, os.O_RDONLY), 0, access=mmap.ACCESS_READ
                ) as mm:
                    total_chunks = (
                        file_size + optimal_chunk_size - 1
                    ) // optimal_chunk_size

                    # Progress bar
                    pbar = tqdm(
                        total=file_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Uploading {filename}",
                    )

                    async def upload_chunk(part_number: int) -> dict:
                        start = part_number * optimal_chunk_size
                        end = min(start + optimal_chunk_size, file_size)

                        # Read chunk from memory map - O(1) memory
                        chunk = mm[start:end]

                        # Upload chunk directly
                        response = await s3_client.upload_part(
                            Bucket=self.bucket.name,
                            Key=filename,
                            PartNumber=part_number + 1,  # S3 parts start at 1
                            UploadId=upload_id,
                            Body=chunk,
                        )

                        pbar.update(len(chunk))
                        return {"PartNumber": part_number + 1, "ETag": response["ETag"]}

                    # Upload chunks in parallel - O(n) where n is number of chunks
                    tasks = [upload_chunk(i) for i in range(total_chunks)]
                    parts = await asyncio.gather(*tasks)

                    # Sort parts by part number - O(n log n)
                    parts.sort(key=lambda x: x["PartNumber"])

                    # Complete upload - O(1)
                    await s3_client.complete_multipart_upload(
                        Bucket=self.bucket.name,
                        Key=filename,
                        UploadId=upload_id,
                        MultipartUpload={"Parts": parts},
                    )

                    pbar.close()

            return True

        except Exception as e:
            tplr.logger.error(f"Error uploading large file {filename}: {e}")
            if "pbar" in locals():
                pbar.close()
            return False

    async def download_large_file(
        self, filename: str, destination_path: str, use_gpu: bool = True
    ) -> bool:
        """
        Downloads a large file from R2 using parallel multipart download.

        Args:
            filename (str): File to download from R2
            destination_path (str): Local path to save the file
            use_gpu (bool): Whether to use GPU for processing (if available)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(self.bucket.account_id),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=self.bucket.access_key_id,
                aws_secret_access_key=self.bucket.secret_access_key,
            ) as s3_client:
                # Get file size
                response = await s3_client.head_object(
                    Bucket=self.bucket.name, Key=filename
                )
                file_size = response["ContentLength"]

                # Calculate optimal chunk size and parts
                cpu_count = os.cpu_count()
                chunk_size = max(
                    16 * 1024 * 1024, file_size // (cpu_count * 2)
                )  # At least 16MB chunks
                total_parts = (file_size + chunk_size - 1) // chunk_size

                # Initialize progress bar
                pbar = tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {filename}",
                    ncols=80,
                )

                async def download_chunk(part: int) -> tuple[int, bytes]:
                    start = part * chunk_size
                    end = min(start + chunk_size, file_size)

                    try:
                        response = await s3_client.get_object(
                            Bucket=self.bucket.name,
                            Key=filename,
                            Range=f"bytes={start}-{end - 1}",
                        )
                        chunk = await response["Body"].read()
                        pbar.update(len(chunk))

                        # If GPU is available and enabled, use it for decompression
                        if (
                            use_gpu
                            and torch.cuda.is_available()
                            and len(chunk) > 1024 * 1024
                        ):  # >1MB
                            # Convert bytes to writable numpy array first
                            chunk_np = np.frombuffer(
                                chunk, dtype=np.uint8
                            ).copy()  # .copy() makes it writable
                            chunk_tensor = torch.from_numpy(chunk_np).cuda()
                            # Any GPU processing here if needed
                            chunk = chunk_tensor.cpu().numpy().tobytes()

                        return part, chunk
                    except Exception as e:
                        tplr.logger.error(f"Error downloading part {part}: {e}")
                        return part, None

                # Download chunks in parallel
                chunks = {}
                retry_count = 3

                while retry_count > 0 and len(chunks) < total_parts:
                    # Create tasks for missing chunks
                    missing_parts = [
                        i
                        for i in range(total_parts)
                        if i not in chunks or chunks[i] is None
                    ]
                    if not missing_parts:
                        break

                    tasks = [download_chunk(part) for part in missing_parts]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Process results
                    for part, data in results:
                        if data is not None:
                            chunks[part] = data

                    retry_count -= 1
                    if len(chunks) < total_parts:
                        tplr.logger.warning(
                            f"Retrying {total_parts - len(chunks)} failed chunks..."
                        )

                # Check if we have all chunks
                if len(chunks) < total_parts:
                    raise Exception(
                        f"Failed to download all chunks after {3 - retry_count} retries"
                    )

                # Write chunks to file
                async with aiofiles.open(destination_path, "wb") as f:
                    for i in range(total_parts):
                        await f.write(chunks[i])

                pbar.close()
                tplr.logger.info(
                    f"Successfully downloaded {filename} to {destination_path}"
                )
                return True

        except Exception as e:
            pbar.close()
            tplr.logger.error(f"Error downloading large file {filename}: {e}")
            return False

        finally:
            # Cleanup GPU memory if used
            if use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def cleanup_old_checkpoints(self, keep_last: int = 3):
        """
        Removes old checkpoints from storage, keeping only the most recent ones.

        Args:
            keep_last (int): Number of most recent checkpoints to keep
        """
        try:
            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(self.bucket.account_id),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=self.bucket.access_key_id,
                aws_secret_access_key=self.bucket.secret_access_key,
            ) as s3_client:
                # List all checkpoint files
                paginator = s3_client.get_paginator("list_objects_v2")
                checkpoint_files = []

                async for page in paginator.paginate(
                    Bucket=self.bucket.name, Prefix="checkpoint"
                ):
                    for obj in page.get("Contents", []):
                        if obj["Key"].startswith("checkpoint"):
                            checkpoint_files.append(obj)

                # Sort by last modified time
                checkpoint_files.sort(key=lambda x: x["LastModified"], reverse=True)

                # Delete older checkpoints
                if len(checkpoint_files) > keep_last:
                    to_delete = checkpoint_files[keep_last:]
                    await s3_client.delete_objects(
                        Bucket=self.bucket.name,
                        Delete={"Objects": [{"Key": obj["Key"]} for obj in to_delete]},
                    )
                    tplr.logger.info(f"Deleted {len(to_delete)} old checkpoints")

        except Exception as e:
            tplr.logger.error(f"Error cleaning up old checkpoints: {e}")

    async def is_miner_active(self, uid: int, recent_windows: int = 3) -> bool:
        """Check if the miner has uploaded gradients in the last few windows."""
        tplr.logger.debug(f"Checking if UID {uid} is active")
        current_window = self.current_window

        peer_bucket = self.commitments.get(uid)
        if not peer_bucket:
            tplr.logger.debug(f"No bucket committed for UID {uid}")
            return False

        try:
            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(peer_bucket.account_id),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=peer_bucket.access_key_id,
                aws_secret_access_key=peer_bucket.secret_access_key,
            ) as s3_client:
                # Ensure that self.current_window is set
                if not hasattr(self, "current_window") or self.current_window is None:
                    tplr.logger.error(
                        "current_window is not set in comms. Please set comms.current_window from the main thread."
                    )
                    return False

                current_window = self.current_window

                for window in range(
                    current_window - recent_windows, current_window + 1
                ):
                    filename = f"gradient-{window}-{uid}-v{__version__}.pt"
                    tplr.logger.debug(
                        f"Checking for {filename} in bucket {peer_bucket.name}"
                    )
                    try:
                        await s3_client.head_object(
                            Bucket=peer_bucket.name, Key=filename
                        )
                        tplr.logger.debug(f"Found {filename} for UID {uid}")
                        return True
                    except botocore.exceptions.ClientError as e:
                        if e.response["Error"]["Code"] != "404":
                            tplr.logger.error(
                                f"Error checking activity for UID {uid}: {e}"
                            )
                            return False
                        tplr.logger.debug(f"{filename} not found for UID {uid}")
        except Exception as e:
            tplr.logger.error(f"Error accessing bucket for UID {uid}: {e}")
            return False

        return False

    async def track_active_peers(self):
        """Background task to keep track of active peers."""
        while True:
            active_peers = set()
            tasks = []
            semaphore = asyncio.Semaphore(10)  # Limit concurrent S3 requests

            tplr.logger.debug(f"Commitments: {self.commitments}")

            async def check_peer(uid):
                async with semaphore:
                    is_active = await self.is_miner_active(
                        uid, recent_windows=self.recent_windows
                    )
                    if is_active:
                        active_peers.add(uid)

            for uid in self.commitments.keys():
                tasks.append(check_peer(uid))

            await asyncio.gather(*tasks)
            self.active_peers = active_peers

            tplr.logger.info(
                f"Updated active peers: {[int(uid) for uid in self.active_peers]}"
            )

            await asyncio.sleep(self.active_check_interval)

    async def get_latest_checkpoint(self):
        """Get the latest checkpoint from R2 storage"""
        try:
            # Get validator with highest stake
            validator_uid = self.metagraph.S.argmax().item()
            tplr.logger.info(f"Found validator with highest stake: {validator_uid}")

            if validator_uid is None:
                tplr.logger.info("No active validators found")
                return None

            # List checkpoint files from validator's bucket
            checkpoint_files = []
            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(self.bucket.account_id),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=self.bucket.access_key_id,
                aws_secret_access_key=self.bucket.secret_access_key,
            ) as s3_client:
                # Use regex pattern to match checkpoint files
                pattern = re.compile(f"^checkpoint-(\\d+)-0-v{__version__}\\.pt$")

                paginator = s3_client.get_paginator("list_objects_v2")
                async for page in paginator.paginate(
                    Bucket=self.bucket.name, Prefix="checkpoint"
                ):
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        match = pattern.match(key)
                        if match:
                            window = int(match.group(1))
                            checkpoint_files.append(
                                {
                                    "key": key,
                                    "window": window,
                                    "size": obj["Size"],
                                    "last_modified": obj["LastModified"],
                                }
                            )

            if not checkpoint_files:
                tplr.logger.info("No checkpoint files found")
                return None

            # Sort by last_modified timestamp (descending) and get latest
            latest = max(checkpoint_files, key=lambda x: x["last_modified"])
            tplr.logger.info(
                f"Found latest checkpoint: {latest['key']} from window {latest['window']}, modified at {latest['last_modified']}"
            )

            # Get the checkpoint data using the window from the latest checkpoint
            result = await self.get(
                uid=str(validator_uid),
                window=latest["window"],
                key="checkpoint",
                timeout=240,
                local=False,
                stale_retention=10,
            )

            if result is None:
                tplr.logger.error(f"Failed to download checkpoint {latest['key']}")
                return None

            checkpoint_data, _ = result
            return checkpoint_data, latest["window"]

        except Exception as e:
            tplr.logger.error(f"Error getting latest checkpoint: {e}")
            return None
