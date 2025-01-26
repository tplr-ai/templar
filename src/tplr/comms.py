# type: ignore
import os
import re
import math
import json
import time
import torch
import asyncio
import aiofiles
import botocore
import numpy as np
import bittensor as bt
from tqdm import tqdm as std_tqdm
from types import SimpleNamespace
from typing import List, Dict, Optional, TypeVar, Any
from aiobotocore.session import get_session

from . import __version__
from .config import client_config, BUCKET_SECRETS
from .chain import ChainManager
from .schemas import Bucket

import tplr as tplr


# Constants
CF_REGION_NAME: str = "enam"
LOCAL_TMP_DIR = "/tmp/local_store"

T = TypeVar("T", bound=Any)
FixtureFunction = TypeVar("FixtureFunction", bound=Any)


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
        self.bucket = self.get_own_bucket("gradients", "write")
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

    def get_own_bucket(self, bucket_type, access_type=None) -> Bucket:
        """Gets bucket configuration from environment variables via config.BUCKET_SECRETS.

        Args:
            bucket_type: Either "gradients" or "dataset" to determine which bucket to use
            access_type: For gradients bucket, either "read" or "write" to determine access level
        """
        try:
            if bucket_type not in ["gradients", "dataset"]:
                raise ValueError("bucket_type must be either 'gradients' or 'dataset'")

            if bucket_type == "gradients":
                if access_type not in ["read", "write"]:
                    raise ValueError(
                        "For gradients bucket, access_type must be either 'read' or 'write'"
                    )

                bucket_config = BUCKET_SECRETS["gradients"]
                credentials = bucket_config["credentials"][access_type]  # type: ignore
            else:  # dataset bucket
                bucket_config = BUCKET_SECRETS["dataset"]
                # For dataset, we'll use read credentials by default
                credentials = bucket_config["credentials"]["read"]  # type: ignore

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
            endpoint_url=self.get_base_url(BUCKET_SECRETS["gradients"]["account_id"]),
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=BUCKET_SECRETS["gradients"]["credentials"]["write"][  # type: ignore
                "access_key_id"
            ],
            aws_secret_access_key=BUCKET_SECRETS["gradients"]["credentials"]["write"][  # type: ignore
                "secret_access_key"
            ],
        ) as s3_client:
            continuation_token = None

            while True:
                list_args = {
                    "Bucket": BUCKET_SECRETS["gradients"]["name"],
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
                    key: str = obj["Key"]  # type: ignore
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
                        Bucket=BUCKET_SECRETS["gradients"]["name"],
                        Delete={"Objects": stale_objects},
                    )

                if response.get("IsTruncated"):
                    continuation_token = response.get("NextContinuationToken")
                else:
                    break

    async def s3_put_object(
        self,
        key: str,
        file_path: Optional[str] = None,
        bucket: Optional[Bucket] = None,
    ):
        """
        Puts an object into S3 storage, handling different file types appropriately.

        Args:
            key (str): The key/path to store the data under
            file_path (str, optional): The local file path to upload
            bucket (Bucket, optional): The bucket to use. Defaults to self.bucket
        """
        try:
            bucket = self.bucket

            # Handle JSON files
            if key.endswith(".json") or "start_window" in key:
                if file_path:
                    async with aiofiles.open(file_path, "r") as f:
                        data = await f.read()
                        data_bytes = json.dumps(json.loads(data)).encode("utf-8")
                else:
                    raise ValueError(f"file_path required for JSON file: {key}")

                async with self.session.create_client(
                    "s3",
                    endpoint_url=self.get_base_url(bucket.account_id),
                    region_name=CF_REGION_NAME,
                    config=client_config,
                    aws_access_key_id=bucket.access_key_id,
                    aws_secret_access_key=bucket.secret_access_key,
                ) as s3_client:
                    await s3_client.put_object(
                        Bucket=bucket.name, Key=key, Body=data_bytes
                    )
                return

            # Handle PyTorch files
            file_size = os.path.getsize(file_path)
            multipart_threshold = 100 * 1024 * 1024  # 100MB

            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(bucket.account_id),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=bucket.access_key_id,
                aws_secret_access_key=bucket.secret_access_key,
            ) as s3_client:
                if file_size <= multipart_threshold:
                    # Simple upload for small files
                    async with aiofiles.open(file_path, "rb") as f:
                        data = await f.read()
                        await s3_client.put_object(
                            Bucket=bucket.name, Key=key, Body=data
                        )
                else:
                    # Multipart upload for large files
                    await self.upload_large_file(file_path, key, s3_client)

        except Exception as e:
            tplr.logger.error(f"Error uploading {key} to S3: {e}")
            raise

    async def s3_get_object(
        self,
        key: str,
        bucket: Bucket = None,
        timeout: int = 5,
    ):
        """Download object from S3 using asynchronous streaming."""
        temp_file_path = None
        try:
            # Create temp directory if it doesn't exist
            os.makedirs(self.temp_dir, exist_ok=True)
            temp_file_path = os.path.join(self.temp_dir, f"temp_{key}")

            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(bucket.name),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=bucket.access_key_id,
                aws_secret_access_key=bucket.secret_access_key,
            ) as s3_client:
                try:
                    response = await asyncio.wait_for(
                        s3_client.head_object(Bucket=bucket.name, Key=key),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    tplr.logger.debug(f"Timeout checking for {key}")
                    return None
                except Exception as e:
                    if "404" in str(e):
                        tplr.logger.debug(
                            f"Object {key} not found in bucket {bucket.name}"
                        )
                        return None
                    raise

                file_size = response["ContentLength"]  # type: ignore

                try:
                    if (
                        file_size <= 5 * 1024 * 1024 * 1024
                    ):  # 5GB threshold (i.e. gradient)
                        response = await asyncio.wait_for(
                            s3_client.get_object(Bucket=bucket.name, Key=key),
                            timeout=timeout,
                        )
                        async with aiofiles.open(temp_file_path, "wb") as f:
                            async with response["Body"] as stream:  # type: ignore
                                data = await asyncio.wait_for(
                                    stream.read(), timeout=timeout
                                )
                                await f.write(data)
                    else:
                        success = await self.download_large_file(
                            s3_client=s3_client,
                            bucket_name=bucket.name,
                            key=key,
                            file_size=file_size,
                            temp_file_path=temp_file_path,
                        )
                        if not success:
                            return None

                    # Load data based on file type
                    if key.endswith(".json") or "start_window" in key:
                        # For JSON files
                        async with aiofiles.open(temp_file_path, "r") as f:
                            data = await f.read()
                            loaded_data = json.loads(data)
                    else:
                        # For PyTorch checkpoint files
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
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    #  Large File Operations

    async def upload_large_file(self, file_path: str, key: str, s3_client):
        """Uploads a large file to S3 using asynchronous multipart upload."""
        try:
            # Initiate multipart upload
            response = await s3_client.create_multipart_upload(
                Bucket=self.bucket.name, Key=key
            )
            upload_id = response["UploadId"]  # type: ignore

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
                                {"ETag": response["ETag"], "PartNumber": part_number}  # type: ignore
                            )
                    part_queue.task_done()
                return part_results

            # Start worker tasks
            workers = [upload_part() for _ in range(min(8, total_parts))]
            results_nested = await asyncio.gather(*workers)

            # Flatten the results
            parts = [item for sublist in results_nested for item in sublist]
            parts.sort(key=lambda x: x["PartNumber"])  # type: ignore

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

                            async with response["Body"] as stream:  # type: ignore
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
                    chunk["size"]
                    for chunk in downloaded_chunks.values()  # type: ignore
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
            tplr.logger.debug(f"Peer bucket : {peer_bucket}")
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
        store_gathers: bool = False,  # New parameter
    ) -> Optional[SimpleNamespace]:
        """Gather operation with individual gradient normalization."""
        start_time = time.time()
        metrics = {"upload_bytes": 0, "download_bytes": 0, "successes": []}

        # Put own state if provided
        if state_dict is not None:
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

            if response is None:
                tplr.logger.debug(f"No data received from UID {uid}")
                continue

            try:
                state_dict_resp, global_step_resp = response
            except (TypeError, ValueError) as e:
                tplr.logger.debug(f"Invalid response format from UID {uid}: {e}")
                continue

            if state_dict_resp is None:
                tplr.logger.debug(f"Empty state dict from UID {uid}")
                continue

            # Store raw gradients if enabled
            if store_gathers:
                try:
                    gradient_data = {
                        "state_dict": {
                            k: v.cpu().numpy() for k, v in state_dict_resp.items()
                        },
                        "metadata": {
                            "uid": uid,
                            "window": window,
                            "global_step": global_step_resp,
                            "timestamp": time.time(),
                            "version": __version__,
                        },
                    }

                    # Create temporary file
                    temp_file = os.path.join(
                        self.temp_dir, f"gradient_{uid}_{window}_{global_step}.npz"
                    )
                    try:
                        np.savez_compressed(temp_file, **gradient_data)
                        key = f"gathers/{__version__}/{uid}/{window}/{global_step}.npz"

                        await self.s3_put_object(
                            key=key, file_path=temp_file, bucket=self.bucket
                        )
                    finally:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

                except Exception as e:
                    tplr.logger.error(
                        f"Failed to store gradient from UID {uid}: {str(e)}"
                    )

            # Normalize each gradient value individually
            normalized_dict = {}
            for param_name, tensor in state_dict_resp.items():
                if param_name.endswith("vals"):
                    tensor = tensor.to(device)
                    orig_dtype = tensor.dtype
                    # Convert to float32 for normalization
                    tensor_f = tensor.to(torch.float32)
                    # Compute norm
                    norm = torch.norm(tensor_f)
                    # Normalize and keep as float32
                    normalized = tensor_f / (norm + 1e-8)
                    normalized_dict[param_name] = normalized
                else:
                    # Keep indices unchanged
                    normalized_dict[param_name] = tensor.to(device)
                metrics["download_bytes"] += tensor.element_size() * tensor.nelement()

            # Move these outside the parameter loop to ensure they are executed once per UID
            valid_uids.append(uid)
            global_steps.append(global_step_resp)

            # Add normalized tensors to aggregated_state_dict
            for param_name, tensor in normalized_dict.items():
                if param_name not in aggregated_state_dict:
                    aggregated_state_dict[param_name] = []
                aggregated_state_dict[param_name].append(tensor)

        # If no valid responses, return None
        if not valid_uids:
            tplr.logger.info("No valid gradients received from any UID")
            return None

        # Convert normalized gradients back to original dtype
        final_state_dict = {}
        for param_name, tensors in aggregated_state_dict.items():
            # Get original dtype
            orig_dtype = (
                state_dict_resp[param_name].dtype if state_dict_resp else torch.float32  # type: ignore
            )
            # Convert back to original dtype
            final_state_dict[param_name] = tensors[0].to(orig_dtype)  # type: ignore

        # Create result namespace
        result = SimpleNamespace(
            time=time.time() - start_time,
            upload_bytes=metrics["upload_bytes"],
            download_bytes=metrics["download_bytes"],
            success_rate=len(valid_uids) / len(uids),
            state_dict=SimpleNamespace(**final_state_dict),
            uids=valid_uids,
            global_steps=global_steps,
        )

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
                        if obj["Key"].startswith("checkpoint"):  # type: ignore
                            checkpoint_files.append(obj)

                # Sort by last modified time
                checkpoint_files.sort(key=lambda x: x["LastModified"], reverse=True)  # type: ignore

                # Delete older checkpoints
                if len(checkpoint_files) > keep_last:
                    to_delete = checkpoint_files[keep_last:]
                    await s3_client.delete_objects(
                        Bucket=self.bucket.name,
                        Delete={"Objects": [{"Key": obj["Key"]} for obj in to_delete]},  # type: ignore
                    )
                    tplr.logger.info(f"Deleted {len(to_delete)} old checkpoints")

        except Exception as e:
            tplr.logger.error(f"Error cleaning up old checkpoints: {e}")

    ## Peer Management

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
                        if e.response["Error"]["Code"] not in ["404", "403"]:  # type: ignore
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

    # Checkpoint Operations

    async def _get_highest_stake_validator_bucket(self):
        """Get the bucket for the validator with highest stake."""
        # Get validator with highest stake
        validator_uid = self.metagraph.S.argmax().item()
        tplr.logger.info(f"Found validator with highest stake: {validator_uid}")

        if validator_uid is None:
            tplr.logger.info("No active validators found")
            return None, None

        validator_bucket = self.commitments.get(int(validator_uid))
        if not validator_bucket:
            return None, None

        tplr.logger.info(f"Validator Bucket: {validator_bucket}")
        return validator_bucket, validator_uid

    async def get_latest_checkpoint(self):
        """Get the latest checkpoint: Returns (checkpoint_data, window) tuple."""
        try:
            (
                validator_bucket,
                validator_uid,
            ) = await self._get_highest_stake_validator_bucket()
            if not validator_bucket:
                return None

            # List checkpoint files efficiently
            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(validator_bucket.account_id),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=validator_bucket.access_key_id,
                aws_secret_access_key=validator_bucket.secret_access_key,
            ) as s3_client:
                pattern = re.compile(
                    rf"^checkpoint-(\d+)-{validator_uid}-v{__version__}\.pt$"
                )

                # Get the most recent objects
                response = await s3_client.list_objects_v2(
                    Bucket=validator_bucket.name,
                    Prefix="checkpoint",
                    MaxKeys=50,  # Limit to recent checkpoints
                )

                if not response.get("Contents"):
                    tplr.logger.info("No checkpoint files found")
                    return None

                # Find latest valid checkpoint that matches pattern
                latest = None
                valid_checkpoints = []

                for obj in response.get("Contents", []):
                    key: str = obj.get("Key", "")
                    match = pattern.match(key)
                    if match:
                        valid_checkpoints.append(
                            {
                                "key": key,
                                "window": int(match.group(1)),
                                "size": obj["Size"],  # type: ignore
                                "last_modified": obj["LastModified"],  # type: ignore
                            }
                        )

                if valid_checkpoints:
                    # Sort by LastModified timestamp (most recent first)
                    latest = max(valid_checkpoints, key=lambda x: x["last_modified"])  # type: ignore
                else:
                    tplr.logger.info("No valid checkpoint files found")
                    return None

                tplr.logger.info(
                    f"Found latest checkpoint: {latest['key']} from window {latest['window']}, "  # type: ignore
                    f"modified at {latest['last_modified']}"  # type: ignore
                )

                # Get the checkpoint data using the window from the latest checkpoint
                loaded_data = await self.s3_get_object(
                    key=latest["key"],
                    bucket=validator_bucket,  # type: ignore
                )

                if loaded_data is None:
                    tplr.logger.error(f"Failed to download checkpoint {latest['key']}")  # type: ignore
                    return None

                return loaded_data, latest["window"]  # type: ignore

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
    ) -> tuple[
        bool, dict, int, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler
    ]:
        """
        Loads the latest checkpoint and catches up across missed windows.
        In this version, we parallelize gathers but still apply updates strictly in ascending window order.

        Returns:
            tuple: (success: bool, momentum: dict, global_step: int, optimizer: Optimizer, scheduler: LRScheduler)
        """
        result = await self.get_latest_checkpoint()
        if not result:
            tplr.logger.info("No valid checkpoints found")
            return False, {}, 0, optimizer, scheduler

        checkpoint_data, checkpoint_window = result
        try:
            # 1) Load checkpoint data
            model.load_state_dict(
                {
                    k: v.to(device)
                    for k, v in checkpoint_data["model_state_dict"].items()
                }
            )
            model.to(device)

            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
            momentum = checkpoint_data["momentum"]

            checkpoint_start_window = checkpoint_data.get("start_window")
            checkpoint_current_window = checkpoint_data.get("current_window")
            if checkpoint_start_window is None or checkpoint_current_window is None:
                tplr.logger.warning(
                    "Checkpoint missing start_window or current_window info"
                )
                return False, {}, 0, optimizer, scheduler

            window_difference = current_window - checkpoint_current_window
            global_step = current_window - checkpoint_start_window

            tplr.logger.info(
                f"Checkpoint windows (start={checkpoint_start_window}, checkpoint_current={checkpoint_current_window}), "
                f"local_current={current_window}, window_diff={window_difference}, global_step={global_step}"
            )

            # 2) Sync scheduler with global_step if needed
            steps_needed = global_step - scheduler.last_epoch
            if steps_needed > 0:
                tplr.logger.info(
                    f"Syncing optimizer/scheduler by stepping {steps_needed} times…"
                )
                for _ in range(steps_needed):
                    optimizer.step()
                    scheduler.step()

            # 3) Return early if no catch-up or behind
            if window_difference < 0:
                tplr.logger.warning(
                    "Local current_window is behind checkpoint; using checkpoint without catch-up."
                )
                return True, momentum, global_step, optimizer, scheduler
            if window_difference == 0:
                tplr.logger.info("No catch-up needed — aligned with checkpoint.")
                return True, momentum, global_step, optimizer, scheduler

            tplr.logger.info(f"Performing catch-up for {window_difference} windows…")

            # 4) Option: Parallel gather in batches, but apply in ascending order
            BATCH_SIZE = 5  # tweak based on memory/time constraints
            windows_to_catch_up = range(
                checkpoint_current_window + 1, current_window + 1
            )

            for i in range(0, len(windows_to_catch_up), BATCH_SIZE):
                batch_windows = list(windows_to_catch_up)[i : i + BATCH_SIZE]

                # Launch gathers in parallel
                tasks = [
                    self.gather(
                        state_dict={},
                        my_uid=uid,
                        uids=peers,
                        window=w,
                        key="gradient",
                        timeout=30,
                        device=device,
                        local=False,
                        stale_retention=100,
                        global_step=global_step,
                    )
                    for w in batch_windows
                ]
                batch_results = await asyncio.gather(*tasks)

                # Store results in dict so we can apply them in correct ascending order
                gathered_data = dict(zip(batch_windows, batch_results))

                # 5) Apply each window's updates in ascending order
                for w in sorted(gathered_data.keys()):
                    gather_result = gathered_data[w]
                    if not gather_result:
                        tplr.logger.info(
                            f"No valid gather data for window {w}, skipping."
                        )
                        continue

                    # Build param updates
                    param_updates = {}
                    for n, p in model.named_parameters():
                        idxs = getattr(gather_result.state_dict, f"{n}idxs", None)
                        vals = getattr(gather_result.state_dict, f"{n}vals", None)
                        if idxs is not None and vals is not None:
                            # Convert to lists if necessary
                            if not isinstance(idxs, (list, tuple)):
                                idxs = [idxs]
                            if not isinstance(vals, (list, tuple)):
                                vals = [vals]

                            # Decode + decompress + sign
                            new_grad = transformer.decode(
                                compressor.batch_decompress(
                                    p.to(device),
                                    idxs,
                                    vals,
                                    transformer.shapes[n],
                                    transformer.totalks[n],
                                )
                            )
                            param_updates[n] = new_grad.sign_()

                    # Apply updates, step optimizer/scheduler
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            if n in param_updates:
                                p.grad = param_updates[n]

                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    tplr.logger.info(
                        f"Caught up window {w}, global_step => {global_step}"
                    )

            tplr.logger.info(
                f"Finished catch-up. Final global_step={global_step}, "
                f"optimizer steps={optimizer.state_dict()['state'].get(0, {}).get('step', 0)}, "  # type: ignore
                f"scheduler last_epoch={scheduler.last_epoch}"
            )
            return True, momentum, global_step, optimizer, scheduler

        except KeyError as e:
            tplr.logger.error(f"Invalid checkpoint format: missing key {e}")
            return False, {}, 0, optimizer, scheduler
        except Exception as e:
            tplr.logger.error(f"Failed to load checkpoint: {e}")
            return False, {}, 0, optimizer, scheduler
        # TODO: Add more robust error handling for large window_difference or gather failures.

    # Start Window Operations

    async def post_start_window(self, start_window: int):
        """Upload the start window as a JSON object to the node's R2 bucket."""
        key = f"start_window_v{__version__}.json"
        start_window_data = {"start_window": start_window}

        # Create temporary JSON file
        temp_file = os.path.join(self.temp_dir, f"temp_{key}")
        try:
            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(start_window_data))

            validator_bucket = self.get_own_bucket("gradients", "write")
            print(f"Validator Access Key : {validator_bucket.access_key_id}")
            await self.s3_put_object(
                key=key, file_path=temp_file, bucket=validator_bucket
            )
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    async def get_start_window(self) -> int:
        while True:
            try:
                (
                    validator_bucket,
                    validator_uid,
                ) = await self._get_highest_stake_validator_bucket()
                if validator_bucket is None:
                    tplr.logger.warning(
                        "No highest staked validator bucket found. Retrying in 10 seconds."
                    )
                    await asyncio.sleep(10)
                    continue

                tplr.logger.info(
                    f"Attempting to fetch start_window from UID {validator_uid} bucket {validator_bucket.name}"
                )

                # Fetch 'start_window.json' using s3_get_object
                start_window_data = await self.s3_get_object(
                    key=f"start_window_v{__version__}.json", bucket=validator_bucket
                )
                if start_window_data is not None:
                    # Check if start_window_data is already a dict
                    if isinstance(start_window_data, dict):
                        start_window_json = start_window_data
                    else:
                        # If it's bytes, decode and load JSON
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
