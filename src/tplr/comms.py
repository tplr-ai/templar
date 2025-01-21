import os
import re
import time
import torch
import asyncio
import aiofiles
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
from tqdm import tqdm as std_tqdm

import math

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
        self._bad_peers = {}  # Initialize directly using private attribute

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

    async def s3_put_object(self, key: str, file_path: str):
        """Upload object to S3 using asynchronous streaming to prevent blocking.

        Args:
            key (str): The S3 object key.
            file_path (str): The local file path to upload.
        """
        try:
            file_size = os.path.getsize(file_path)
            multipart_threshold = 64 * 1024 * 1024  # 64MB

            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(self.bucket.account_id),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=self.bucket.access_key_id,
                aws_secret_access_key=self.bucket.secret_access_key,
            ) as s3_client:
                if file_size <= multipart_threshold:
                    # Simple upload for small files
                    async with aiofiles.open(file_path, "rb") as f:
                        data = await f.read()
                        await s3_client.put_object(
                            Bucket=self.bucket.name, Key=key, Body=data
                        )
                else:
                    # Multipart upload for large files
                    await self.upload_large_file(file_path, key, s3_client)

        except Exception as e:
            tplr.logger.error(f"Error uploading {key} to S3: {e}")
            raise

    async def upload_large_file(self, file_path: str, key: str, s3_client):
        """Uploads a large file to S3 using asynchronous multipart upload."""
        try:
            # Initiate multipart upload
            response = await s3_client.create_multipart_upload(
                Bucket=self.bucket.name, Key=key
            )
            upload_id = response["UploadId"]

            # Define part size (e.g., 64MB)
            part_size = 64 * 1024 * 1024  # 64MB
            file_size = os.path.getsize(file_path)
            total_parts = math.ceil(file_size / part_size)

            semaphore = asyncio.Semaphore(8)  # Limit concurrency

            # Queue to hold part numbers
            part_queue = asyncio.Queue()

            for part_number in range(1, total_parts + 1):
                part_queue.put_nowait(part_number)

            parts = []

            async def upload_part():
                part_results = []
                while not part_queue.empty():
                    part_number = await part_queue.get()
                    byte_range_start = (part_number - 1) * part_size
                    byte_range_end = min(byte_range_start + part_size, file_size)
                    async with semaphore:
                        async with aiofiles.open(file_path, "rb") as f:
                            await f.seek(byte_range_start)
                            data = await f.read(byte_range_end - byte_range_start)

                            response = await s3_client.upload_part(
                                Bucket=self.bucket.name,
                                Key=key,
                                PartNumber=part_number,
                                UploadId=upload_id,
                                Body=data,
                            )
                            part_results.append(
                                {"ETag": response["ETag"], "PartNumber": part_number}
                            )
                    part_queue.task_done()
                return part_results

            # Start worker tasks
            workers = [upload_part() for _ in range(min(8, total_parts))]
            results_nested = await asyncio.gather(*workers)

            # Flatten the results
            parts = [item for sublist in results_nested for item in sublist]
            parts.sort(key=lambda x: x["PartNumber"])

            # Complete multipart upload
            await s3_client.complete_multipart_upload(
                Bucket=self.bucket.name,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )

            tplr.logger.info(f"Successfully uploaded {key}")

        except Exception as e:
            tplr.logger.error(f"Error during multipart upload of {key}: {e}")
            # Abort the multipart upload in case of error
            await s3_client.abort_multipart_upload(
                Bucket=self.bucket.name, Key=key, UploadId=upload_id
            )
            raise

    async def s3_get_object(
        self,
        key: str,
        bucket: Bucket = None,
        bucket_secrets: dict = None,
        timeout: int = 5,
    ):
        """Download object from S3 using asynchronous streaming."""
        try:
            # Setup bucket credentials
            if bucket_secrets:
                access_key = bucket_secrets["write"]["access_key_id"]
                secret_key = bucket_secrets["write"]["secret_access_key"]
                account_id = bucket_secrets["account_id"]
                bucket_name = bucket_secrets["bucket_name"]
            elif bucket:
                access_key = bucket.access_key_id
                secret_key = bucket.secret_access_key
                account_id = bucket.account_id
                bucket_name = bucket.name
            else:
                raise ValueError("Either bucket or bucket_secrets must be provided")

            # Create temp directory if it doesn't exist
            os.makedirs(self.temp_dir, exist_ok=True)
            temp_file_path = os.path.join(self.temp_dir, f"temp_{key}")

            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(account_id),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            ) as s3_client:
                try:
                    response = await asyncio.wait_for(
                        s3_client.head_object(Bucket=bucket_name, Key=key),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    tplr.logger.debug(f"Timeout checking for {key}")
                    return None
                except Exception as e:
                    if "404" in str(e):
                        tplr.logger.debug(
                            f"Object {key} not found in bucket {bucket_name}"
                        )
                        return None
                    raise

                file_size = response["ContentLength"]

                try:
                    if file_size <= 5 * 1024 * 1024 * 1024:  # 5GB threshold
                        response = await asyncio.wait_for(
                            s3_client.get_object(Bucket=bucket_name, Key=key),
                            timeout=timeout,
                        )
                        async with aiofiles.open(temp_file_path, "wb") as f:
                            async with response["Body"] as stream:
                                data = await asyncio.wait_for(
                                    stream.read(), timeout=timeout
                                )
                                await f.write(data)
                    else:
                        success = await self.download_large_file(
                            s3_client=s3_client,
                            bucket_name=bucket_name,
                            key=key,
                            file_size=file_size,
                            temp_file_path=temp_file_path,
                        )
                        if not success:
                            return None

                    # Load data and return raw state dict
                    loaded_data = torch.load(
                        temp_file_path,
                        map_location=self.config.device,
                        weights_only=True,
                    )
                    return loaded_data

                except asyncio.TimeoutError:
                    tplr.logger.debug(f"Timeout downloading {key}")
                    return None

        except Exception as e:
            tplr.logger.error(f"Error in s3_get_object for {key}: {e}")
            return None
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    async def download_large_file(
        self, s3_client, bucket_name: str, key: str, file_size: int, temp_file_path: str
    ):
        """Download large file using multipart download with concurrent chunks."""
        try:
            # Determine optimal chunk size and concurrency
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                max_workers = min(torch.cuda.device_count() * 4, 16)
                chunk_size = min(
                    max(
                        5 * 1024 * 1024,  # Minimum 5MB for S3 multipart
                        gpu_mem // (max_workers * 4),
                    ),
                    5 * 1024 * 1024 * 1024,  # Maximum 5GB
                )
            else:
                cpu_count = os.cpu_count() or 1
                max_workers = min(cpu_count * 4, 16)
                chunk_size = min(
                    max(
                        5 * 1024 * 1024,
                        file_size // (max_workers * 2),
                    ),
                    5 * 1024 * 1024 * 1024,
                )

            total_chunks = math.ceil(file_size / chunk_size)
            max_workers = min(max_workers, total_chunks)
            semaphore = asyncio.Semaphore(max_workers)

            # Create the file with the correct size
            async with aiofiles.open(temp_file_path, "wb") as f:
                await f.truncate(file_size)

            # Create progress bar
            pbar = std_tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {key} ({max_workers} workers)",
            )

            downloaded_chunks = {}

            async def download_chunk(chunk_number: int, max_retries: int = 3):
                """Download a specific chunk with retries."""
                for attempt in range(max_retries):
                    async with semaphore:
                        start = chunk_number * chunk_size
                        end = min(start + chunk_size, file_size) - 1

                        try:
                            response = await s3_client.get_object(
                                Bucket=bucket_name,
                                Key=key,
                                Range=f"bytes={start}-{end}",
                            )

                            async with response["Body"] as stream:
                                chunk_data = await stream.read()

                            # Verify chunk size matches expected
                            chunk_len = len(chunk_data)
                            expected_len = end - start + 1
                            if chunk_len != expected_len:
                                raise Exception(
                                    f"Chunk size mismatch: got {chunk_len}, expected {expected_len}"
                                )

                            async with aiofiles.open(temp_file_path, "rb+") as f:
                                await f.seek(start)
                                await f.write(chunk_data)

                            pbar.update(chunk_len)
                            downloaded_chunks[chunk_number] = {
                                "start": start,
                                "end": end + 1,
                                "size": chunk_len,
                            }

                            return chunk_number

                        except Exception as e:
                            tplr.logger.error(
                                f"Error downloading chunk {chunk_number} (attempt {attempt + 1}/{max_retries}): {e}"
                            )
                            if attempt == max_retries - 1:  # Last attempt
                                raise
                            await asyncio.sleep(
                                1 * (attempt + 1)
                            )  # Exponential backoff

            try:
                tasks = [
                    asyncio.create_task(download_chunk(i)) for i in range(total_chunks)
                ]
                await asyncio.gather(*tasks)

                if len(downloaded_chunks) != total_chunks:
                    missing_chunks = set(range(total_chunks)) - set(
                        downloaded_chunks.keys()
                    )
                    raise Exception(f"Missing chunks: {missing_chunks}")

                downloaded_size = sum(
                    chunk["size"] for chunk in downloaded_chunks.values()
                )
                if downloaded_size != file_size:
                    raise Exception(
                        f"Downloaded size ({downloaded_size}) does not match expected size ({file_size})"
                    )

                return True

            finally:
                pbar.close()

        except Exception as e:
            tplr.logger.error(f"Error in download_large_file for {key}: {e}")
            return False

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
    ) -> Optional[tuple[dict, int]]:
        """GET operation."""
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        tplr.logger.debug(f"GET {filename} -->")

        try:
            if local:
                # Local storage logic remains unchanged
                await self.cleanup_local_data(
                    uid=uid, current_window=window, stale_retention=stale_retention
                )
                local_path = os.path.join(
                    LOCAL_TMP_DIR, str(uid), str(window), filename
                )
                if not os.path.exists(local_path):
                    tplr.logger.debug(f"Local file not found: {local_path}")
                    return None
                loaded_data = torch.load(local_path, weights_only=True)
                if key == "checkpoint":
                    return loaded_data, None
                state_dict = loaded_data.get("state_dict")
                global_step = loaded_data.get("global_step", 0)
                return state_dict, global_step

            # Remote storage logic
            peer_bucket = self.commitments.get(int(uid))
            tplr.logger.info(f"Peer bucket : {peer_bucket}")
            if not peer_bucket:
                return None

            loaded_data = await self.s3_get_object(
                key=filename, bucket=peer_bucket, timeout=timeout
            )

            if loaded_data is None:
                return None

            if key == "checkpoint":
                return loaded_data, None

            state_dict = loaded_data.get("state_dict")
            global_step = loaded_data.get("global_step", 0)
            return state_dict, global_step

        except Exception as e:
            tplr.logger.debug(f"GET error {filename}: {e}")
            return None

        finally:
            tplr.logger.debug(f"GET {filename} <--")

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

            # Skip if no response or already known bad peer
            if response is None or self.is_bad_peer(uid):
                tplr.logger.debug(
                    f"Skipping UID {uid}: {'No response' if response is None else 'Bad peer'}"
                )
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

            # Calculate gradient norm and update bad peers tracking
            total_norm = 0.0
            try:
                for tensor in state_dict_resp.values():
                    # Convert tensor to float if needed
                    if not tensor.is_floating_point():
                        tensor = tensor.to(torch.float32)
                    param_norm = tensor.norm(p=2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm**0.5

                # Update bad peers tracking
                self.update_bad_peers(uid, total_norm)

                # Skip if peer is now considered bad
                if self.is_bad_peer(uid):
                    tplr.logger.warning(f"UID {uid} exceeded bad peer threshold")
                    continue

                valid_uids.append(uid)
                global_steps.append(global_step_resp)

                # Normalize tensors and add to aggregated_state_dict
                for param_name, tensor in state_dict_resp.items():
                    # Ensure tensor is float
                    if not tensor.is_floating_point():
                        tensor = tensor.to(torch.float32)
                    tensor = tensor.to(device)
                    tensor_norm = tensor.norm(p=2)
                    if tensor_norm > 0:
                        tensor = tensor / tensor_norm
                    else:
                        tplr.logger.debug(
                            f"Tensor {param_name} from UID {uid} has zero norm and cannot be normalized."
                        )

                    if param_name not in aggregated_state_dict:
                        aggregated_state_dict[param_name] = []
                    aggregated_state_dict[param_name].append(tensor)
                    metrics["download_bytes"] += (
                        tensor.element_size() * tensor.nelement()
                    )

            except Exception as e:
                tplr.logger.error(f"Error processing tensors from UID {uid}: {str(e)}")
                continue

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
                        if e.response["Error"]["Code"] not in ["404", "403"]:
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
        """Get the latest checkpoint: Returns (checkpoint_data, window) tuple."""
        try:
            # Get validator with highest stake
            validator_uid = self.metagraph.S.argmax().item()
            tplr.logger.info(f"Found validator with highest stake: {validator_uid}")

            if validator_uid is None:
                tplr.logger.info("No active validators found")
                return None
            validator_bucket = self.commitments.get(int(validator_uid))

            if not validator_bucket:
                return None

            tplr.logger.info(f"Validator Bucket: {validator_bucket}")
            # List checkpoint files from validator's bucket
            checkpoint_files = []
            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(validator_bucket.account_id),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=validator_bucket.access_key_id,
                aws_secret_access_key=validator_bucket.secret_access_key,
            ) as s3_client:
                # Use regex pattern to match checkpoint files
                pattern = re.compile(
                    rf"^checkpoint-(\d+)-{validator_uid}-v{__version__}\.pt$"
                )

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
            loaded_data = await self.s3_get_object(
                key=latest["key"],
                bucket_secrets={
                    "account_id": validator_bucket.account_id,
                    "bucket_name": validator_bucket.name,
                    "write": {
                        "access_key_id": validator_bucket.access_key_id,
                        "secret_access_key": validator_bucket.secret_access_key,
                    },
                },
            )

            if loaded_data is None:
                tplr.logger.error(f"Failed to download checkpoint {latest['key']}")
                return None

            return loaded_data, latest["window"]

        except Exception as e:
            tplr.logger.error(f"Error getting latest checkpoint: {e}")
            return None

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
    ) -> tuple[bool, dict, int]:
        """
        Load latest checkpoint and catch up through missed windows.

        Returns:
            tuple: (success: bool, momentum: dict, global_step: int)
        """
        result = await self.get_latest_checkpoint()
        if not result:
            tplr.logger.info("No valid checkpoints found")
            return False, {}, 0

        checkpoint_data, window = result
        try:
            # Load model state
            model.load_state_dict(
                {
                    k: v.to(device)
                    for k, v in checkpoint_data["model_state_dict"].items()
                }
            )
            model.to(device)

            # Load optimizer state
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

            # Load scheduler state
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

            # Get checkpoint metadata
            momentum = checkpoint_data["momentum"]
            global_step = checkpoint_data["global_step"]
            checkpoint_window = checkpoint_data.get("checkpoint_window")

            if not checkpoint_window:
                tplr.logger.warning(
                    "Checkpoint missing window info, cannot catch up properly"
                )
                return False, {}, 0

            window_difference = current_window - checkpoint_window
            tplr.logger.info(
                f"Current window: {current_window}, Checkpoint window: {checkpoint_window}, Difference: {window_difference}"
            )

            if window_difference < 0:
                tplr.logger.warning(
                    f"Current window ({current_window}) is behind checkpoint window ({checkpoint_window})"
                )
                return True, momentum, global_step
            elif window_difference == 0:
                tplr.logger.info("No catch up needed - at same window")
                return True, momentum, global_step

            tplr.logger.info(f"Need to catch up through {window_difference} windows...")

            # Catch up through missed windows
            for catch_up_window in range(checkpoint_window + 1, current_window + 1):
                tplr.logger.info(
                    f"Catching up window {catch_up_window} (Progress: {catch_up_window - checkpoint_window}/{window_difference})"
                )
                # Gather gradients from peers for this historical window
                gather_result = await self.gather(
                    state_dict={},  # Empty dict since we're just catching up
                    my_uid=uid,
                    uids=peers,
                    window=catch_up_window,
                    key="gradient",
                    timeout=30,
                    device=device,
                    local=False,
                    stale_retention=100,
                    global_step=global_step,
                )

                if gather_result:
                    # Apply gathered gradients
                    for n, p in model.named_parameters():
                        idxs_key = n + "idxs"
                        vals_key = n + "vals"
                        idxs = getattr(gather_result.state_dict, idxs_key, None)
                        vals = getattr(gather_result.state_dict, vals_key, None)

                        if idxs is not None and vals is not None:
                            if not isinstance(idxs, (list, tuple)):
                                idxs = [idxs]
                            if not isinstance(vals, (list, tuple)):
                                vals = [vals]

                            new_grad = transformer.decode(
                                compressor.batch_decompress(
                                    p.to(device),
                                    idxs,
                                    vals,
                                    transformer.shapes[n],
                                    transformer.totalks[n],
                                )
                            )

                            if p.grad is None:
                                p.grad = new_grad
                            else:
                                p.grad.copy_(new_grad)
                            p.grad.sign_()

                    # Step optimizer and scheduler
                    optimizer.step()
                    scheduler.step()
                    global_step += 1

                    tplr.logger.info(
                        f"Caught up window {catch_up_window}, global_step={global_step}"
                    )
                else:
                    tplr.logger.warning(
                        f"No gradients found for window {catch_up_window}"
                    )

            tplr.logger.info(
                f"Successfully loaded checkpoint and caught up {window_difference} windows"
            )
            return True, momentum, global_step

        except KeyError as e:
            tplr.logger.error(f"Invalid checkpoint format: missing key {e}")
            return False, {}, 0
        except Exception as e:
            tplr.logger.error(f"Failed to load checkpoint: {e}")
            return False, {}, 0
