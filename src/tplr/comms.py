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
import random
import re
import math
import json
import time
from aiobotocore.client import AioBaseClient
from botocore.exceptions import ClientError, ConnectionClosedError
import numpy as np
import torch
import asyncio
import aiofiles
import botocore
from datetime import datetime, timezone
import bittensor as bt

from tqdm import tqdm as std_tqdm
from typing import List, Dict, Literal, Optional
from aiobotocore.session import get_session

from . import __version__
from .config import client_config, BUCKET_SECRETS
from .chain import ChainManager
from .schemas import Bucket

import tplr as tplr
# from .hparams import HParams

from types import SimpleNamespace
from typing import Any, Tuple
from transformers import LlamaForCausalLM
from torch.optim import SGD
from torch.optim.lr_scheduler import SequentialLR
from .compress import TransformDCT, CompressDCT


# Constants
CF_REGION_NAME: str = "enam"
LOCAL_TMP_DIR = "/tmp/local_store"
PEERS_FILE_PREFIX = "peers_"

# Types
PeerArray = np.ndarray[Any, np.dtype[np.int64]]


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

        ## a single aiobotocore session and a dictionary of clients
        self.session = get_session()
        self._s3_clients: dict[
            tuple[str, str, str], AioBaseClient
        ] = {}  # (acc_key, sec_key, account_id) -> s3_client

        self.lock = asyncio.Lock()
        self.active_peers = set()  # Set to store active peers
        self.active_check_interval = (
            self.hparams.active_check_interval
        )  # Interval in seconds
        self.recent_windows = (
            self.hparams.recent_windows
        )  # Number of recent windows to check

        # Add connection management
        self.client_semaphore = asyncio.Semaphore(30)  # Limit concurrent connections
        self.retry_config = {"max_attempts": 3, "backoff_base": 1.5}

    async def _get_s3_client(self, bucket: Bucket):
        """
        Returns a persistent s3_client for the given bucket credentials.
        We create it if we haven't already, else reuse the existing client.
        """
        key = (bucket.access_key_id, bucket.secret_access_key, bucket.account_id)
        if key in self._s3_clients:
            return self._s3_clients[key]

        # Create the base URL
        endpoint_url = self.get_base_url(bucket.account_id)

        # Create the client (equivalent to `async with`, but we store the client persistently)
        new_client = self.session.create_client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=bucket.access_key_id,
            aws_secret_access_key=bucket.secret_access_key,
        )
        # We must manually do an async enter
        new_client = await new_client.__aenter__()

        self._s3_clients[key] = new_client
        return new_client

    async def close_all_s3_clients(self):
        """
        Closes all S3 clients that have been created and stored
        """
        for key, client in list(self._s3_clients.items()):
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                tplr.logger.warning(f"Error closing s3_client {key}: {e}")
        self._s3_clients.clear()

    async def _purge_s3_client(self, bucket: Bucket) -> None:
        key = (bucket.access_key_id, bucket.secret_access_key, bucket.account_id)
        if key in self._s3_clients:
            del self._s3_clients[key]

    def start_background_tasks(self):
        self.loop = asyncio.get_running_loop()
        # Start background tasks
        self.loop.create_task(self.track_active_peers())

    def get_own_bucket(
        self,
        bucket_type: Literal["gradients", "dataset", "aggregator"],
        access_type=None,
    ) -> Bucket:
        """Gets bucket configuration from environment variables via config.BUCKET_SECRETS.

        Args:
            bucket_type: Either "gradients" or "dataset" to determine which bucket to use
            access_type: For gradients bucket, either "read" or "write" to determine access level
        """
        try:
            if bucket_type not in ["gradients", "dataset", "aggregator"]:
                raise ValueError("bucket_type must be either 'gradients' or 'dataset'")

            if bucket_type in ["gradients", "aggregator"]:
                if access_type not in ["read", "write"]:
                    raise ValueError(
                        f"For {bucket_type} bucket, access_type must be either 'read' or 'write'"
                    )

                bucket_config = BUCKET_SECRETS[bucket_type]
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
        try:
            min_allowed_window = current_window - stale_retention

            # Regex pattern to match filenames of the form:
            # gradient-<window>-<uid>-v<version>.pt
            pattern = re.compile(rf"^gradient-(\d+)-{uid}-v{tplr.__version__}.pt$")

            prefix = "gradient"

            # so we get the same credentials as `self.bucket`
            s3_client = await self._get_s3_client(self.bucket)

            continuation_token = None
            while True:
                list_args = {
                    "Bucket": self.bucket.name,
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

                    # Attempt to match our known filename pattern
                    match = pattern.match(key)
                    if match is None:
                        continue

                    try:
                        file_window = int(match.group(1))
                    except ValueError:
                        continue

                    if file_window < min_allowed_window:
                        stale_objects.append({"Key": key})

                # Batch delete stale objects
                if len(stale_objects) > 0:
                    tplr.logger.debug(
                        f"Removing stale S3 objects for {uid}: {stale_objects}"
                    )
                    await s3_client.delete_objects(
                        Bucket=self.bucket.name,
                        Delete={"Objects": stale_objects},
                    )

                if response.get("IsTruncated"):
                    continuation_token = response.get("NextContinuationToken")
                else:
                    break
        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(self.bucket)

    async def s3_put_object(
        self,
        key: str,
        file_path: Optional[str] = None,
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
            s3_client = await self._get_s3_client(bucket)

            # Handle JSON files
            if key.endswith(".json") or "start_window" in key:
                if file_path:
                    async with aiofiles.open(file_path, "r") as f:
                        data = await f.read()
                        data_bytes = json.dumps(json.loads(data)).encode("utf-8")
                else:
                    raise ValueError(f"file_path required for JSON file: {key}")

                await s3_client.put_object(Bucket=bucket.name, Key=key, Body=data_bytes)
                return

            # Otherwise, likely PyTorch files
            file_size = os.path.getsize(file_path)
            multipart_threshold = 100 * 1024 * 1024  # 100MB

            if file_size <= multipart_threshold:
                # Simple upload for small files
                async with aiofiles.open(file_path, "rb") as f:
                    data = await f.read()
                    await s3_client.put_object(Bucket=bucket.name, Key=key, Body=data)
            else:
                # Multipart upload for large files
                await self.upload_large_file(file_path, key, s3_client)

        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
        except Exception as e:
            tplr.logger.error(f"Error uploading {key} to S3: {e}")
            raise

    async def s3_get_object(
        self,
        key: str,
        bucket: Bucket = None,
        timeout: int = 10,
        time_min: datetime = None,
        time_max: datetime = None,
    ):
        """Download object from S3 using asynchronous streaming."""
        import uuid

        temp_file_path = os.path.join(
            self.temp_dir, f"temp_{key}_{uuid.uuid4().hex}.pt"
        )

        s3_client = await self._get_s3_client(bucket)
        try:
            # Normalize timezone info
            if time_min is not None and not time_min.tzinfo:
                time_min = time_min.replace(tzinfo=timezone.utc)
            if time_max is not None and not time_max.tzinfo:
                time_max = time_max.replace(tzinfo=timezone.utc)

            # HEAD the object
            try:
                response = await asyncio.wait_for(
                    s3_client.head_object(Bucket=bucket.name, Key=key),
                    timeout=timeout,
                )
                last_modified = response.get("LastModified")
                if last_modified is None:
                    tplr.logger.info(f"Object does not exist: {key}")
                    return None

                if time_min is not None and last_modified < time_min:
                    time_diff = (time_min - last_modified).total_seconds()
                    tplr.logger.info(
                        f"Object {key} was uploaded {time_diff:.2f}s before time_min."
                    )
                    return {"__status": "TOO_EARLY"}

                if time_max is not None and last_modified > time_max:
                    time_diff = (last_modified - time_max).total_seconds()
                    tplr.logger.info(
                        f"Object {key} was uploaded {time_diff:.2f}s after time_max."
                    )
                    return {"__status": "TOO_LATE"}

            except asyncio.TimeoutError:
                tplr.logger.debug(f"Timeout checking for {key}")
                return None
            except (ConnectionClosedError, ClientError) as e:
                await self._purge_s3_client(bucket)
                if "404" in str(e):
                    tplr.logger.debug(f"Object {key} not found in bucket {bucket.name}")
                    return None

            file_size = response["ContentLength"]  # type: ignore

            # Download the object
            if file_size <= 5 * 1024 * 1024 * 1024:  # 5GB
                response = await asyncio.wait_for(
                    s3_client.get_object(Bucket=bucket.name, Key=key),
                    timeout=timeout,
                )
                async with aiofiles.open(temp_file_path, "wb") as f:
                    async with response["Body"] as stream:
                        data = await asyncio.wait_for(stream.read(), timeout=timeout)
                        await f.write(data)
            else:
                success = await self.download_large_file(
                    s3_client=s3_client,
                    bucket=bucket,
                    key=key,
                    file_size=file_size,
                    temp_file_path=temp_file_path,
                )
                if not success:
                    return None

            # Now load the data
            if key.endswith(".json") or "start_window" in key:
                async with aiofiles.open(temp_file_path, "r") as f:
                    data = await f.read()
                    loaded_data = json.loads(data)
            else:
                loaded_data = torch.load(
                    temp_file_path, map_location=self.config.device
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

    #  Large File Operations

    async def upload_large_file(self, file_path: str, key: str, s3_client):
        """Uploads a large file to S3 using asynchronous multipart upload with 5MB chunks."""
        upload_id = None
        MAX_RETRIES = 3
        PART_SIZE = 5 * 1024 * 1024  # 5MB

        try:
            async with self.client_semaphore:
                for attempt in range(MAX_RETRIES):
                    try:
                        response = await s3_client.create_multipart_upload(
                            Bucket=self.bucket.name, Key=key
                        )
                        upload_id = response["UploadId"]
                        break
                    except Exception:
                        if attempt == MAX_RETRIES - 1:
                            raise
                        await asyncio.sleep(2**attempt)

                file_size = os.path.getsize(file_path)
                total_parts = math.ceil(file_size / PART_SIZE)
                parts = []

                async def upload_part(part_number: int):
                    byte_range_start = (part_number - 1) * PART_SIZE
                    byte_range_end = min(byte_range_start + PART_SIZE, file_size)

                    for attempt in range(MAX_RETRIES):
                        try:
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
                            return {
                                "ETag": response["ETag"],
                                "PartNumber": part_number,
                            }
                        except Exception as e:
                            if attempt == MAX_RETRIES - 1:
                                tplr.logger.error(
                                    f"Failed to upload part {part_number} after {MAX_RETRIES} attempts: {e}"
                                )
                                raise
                            await asyncio.sleep(2**attempt)

                # Launch one coroutine per part
                try:
                    part_results = await asyncio.gather(
                        *[
                            upload_part(part_number)
                            for part_number in range(1, total_parts + 1)
                        ]
                    )
                    parts.extend(part_results)
                except Exception as e:
                    tplr.logger.error(f"Multipart upload failed: {e}")
                    raise

                parts.sort(key=lambda x: x["PartNumber"])

                for attempt in range(MAX_RETRIES):
                    try:
                        await s3_client.complete_multipart_upload(
                            Bucket=self.bucket.name,
                            Key=key,
                            UploadId=upload_id,
                            MultipartUpload={"Parts": parts},
                        )
                        tplr.logger.info(f"Successfully uploaded {key}")
                        break
                    except Exception:
                        if attempt == MAX_RETRIES - 1:
                            raise
                        await asyncio.sleep(2**attempt)

        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(self.bucket)
        except Exception as e:
            tplr.logger.error(f"Error during multipart upload of {key}: {e}")
            if upload_id:
                try:
                    await s3_client.abort_multipart_upload(
                        Bucket=self.bucket.name, Key=key, UploadId=upload_id
                    )
                except Exception as abort_e:
                    tplr.logger.error(f"Failed to abort multipart upload: {abort_e}")
            raise

    async def download_large_file(
        self, s3_client, bucket: Bucket, key: str, file_size: int, temp_file_path: str
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
                                Bucket=bucket.name,
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

                            async with aiofiles.open(temp_file_path, "rb+") as f2:
                                await f2.seek(start)
                                await f2.write(chunk_data)

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
                        f"Downloaded size ({downloaded_size}) != expected size ({file_size})"
                    )

                return True

            finally:
                pbar.close()

        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
        except Exception as e:
            tplr.logger.error(f"Error in download_large_file for {key}: {e}")
            return False

    async def put(
        self,
        state_dict: dict,
        uid: str | None,
        window: int,
        key: Literal["checkpoint", "debug", "gradient", "aggregator"],
        global_step: int = 0,
        local: bool = True,
        stale_retention: int = 10,
    ) -> float:
        """
        Saves the data locally or uploads to S3, then cleans up stale files.

        Args:
            state_dict (dict): Data to save.
            uid (str): Target user/miner identifier.
            window (int): Current training window.
            key (str): Label for the data (e.g., "gradient").
            global_step (int, optional): Global step counter. Defaults to 0.
            local (bool, optional): If True, store locally; otherwise upload to S3. Defaults to True.
            stale_retention (int, optional): Number of windows to keep before cleanup. Defaults to 10.

        Returns:
            float: The elapsed time (in seconds) for the PUT operation.
        """
        if key == "aggregator":
            filename = f"{key}-{window}-v{__version__}.pt"
        else:
            filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        tplr.logger.debug(f"PUT {filename} -->")

        put_start = tplr.T()

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
                await self.s3_put_object(filename, temp_file_path)
                # Remote storage with automatic handling of large files
                asyncio.create_task(
                    self.cleanup_s3_data(
                        uid=uid, current_window=window, stale_retention=stale_retention
                    )
                )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        put_end = tplr.T()
        tplr.logger.info(f"{tplr.P(window, put_end - put_start)} PUT {filename} <--")
        return put_end - put_start

    async def get(
        self,
        uid: str,
        window: int,
        key: str,
        local: bool = True,
        stale_retention: int = 10,
        time_min: datetime = None,
        time_max: datetime = None,
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
                key=filename,
                bucket=peer_bucket,
                time_min=time_min,
                time_max=time_max,
            )

            if loaded_data is None:
                return None

            # Check for TOO_LATE/TOO_EARLY marker
            if isinstance(loaded_data, dict):
                if loaded_data.get("__status") == "TOO_LATE":
                    tplr.logger.info(
                        f"Object for UID {uid}, window {window}, key {key} was uploaded too late. Skipping."
                    )
                    return {"__status": "TOO_LATE"}
                elif loaded_data.get("__status") == "TOO_EARLY":
                    tplr.logger.info(
                        f"Object for UID {uid}, window {window}, key {key} was uploaded too early. Skipping."
                    )
                    return {"__status": "TOO_EARLY"}

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
        time_min: datetime = None,
        time_max: datetime = None,
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
                time_min=time_min,
                time_max=time_max,
            )

            # Check for TOO_LATE/TOO_EARLY markers - stop retrying immediately
            if isinstance(state_dict, dict):
                if state_dict.get("__status") == "TOO_LATE":
                    tplr.logger.info(
                        f"Gradient for UID {uid}, window {window} exists but was uploaded too late. Skipping."
                    )
                    return None
                elif state_dict.get("__status") == "TOO_EARLY":
                    tplr.logger.info(
                        f"Gradient for UID {uid}, window {window} exists but was uploaded too early. Skipping."
                    )
                    return None

            if state_dict is not None:
                return state_dict

            # Retry after a short delay
            await asyncio.sleep(0.1)

    async def gather(
        self,
        my_uid: int | None,
        uids: List[int],
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
        """Gather operation with individual gradient normalization and connection management."""
        start_time = time.time()
        metrics = {"upload_bytes": 0, "download_bytes": 0, "successes": []}

        tplr.logger.debug(
            f"Starting gather for window {window} with time window: {time_min} to {time_max}"
        )
        tplr.logger.debug(
            f"Gather operation - my_uid: {my_uid}, window: {window}, key: {key}, timeout: {timeout}"
        )
        tplr.logger.debug(f"Target UIDs for gathering: {uids}")

        aggregated_state_dict = {}
        valid_uids = []
        skipped_uids = []  # Retain UIDs that are skipped.
        global_steps = []

        async with self.client_semaphore:
            batch_tasks = [
                self.get_with_retry(
                    uid=uid,
                    window=window,
                    key=key,
                    timeout=timeout,
                    local=local,
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
                        tplr.logger.info(f"Skipped UID {uid} - gradient not found.")
                        skipped_uids.append(uid)
                        continue

                    try:
                        state_dict_resp, global_step_resp = response
                        tplr.logger.debug(
                            f"Received state dict and global step {global_step_resp} from UID {uid}"
                        )
                    except (TypeError, ValueError) as e:
                        tplr.logger.debug(f"Invalid response from UID {uid}: {e}")
                        skipped_uids.append(uid)
                        continue

                    if state_dict_resp is None:
                        tplr.logger.debug(f"Empty state dict from UID {uid}")
                        skipped_uids.append(uid)
                        continue

                    # ---------- Begin Compressed Indices and Values Check ----------
                    valid_response = True
                    for param_name, tensor in state_dict_resp.items():
                        if param_name.endswith("idxs"):
                            base_name = param_name[:-4]
                            totalk = totalks.get(base_name)
                            if totalk is None:
                                tplr.logger.warning(
                                    f"Missing totalk for parameter {base_name} from UID {uid}, skipping UID."
                                )
                                valid_response = False
                                break
                            try:
                                self.check_compressed_indices(
                                    param_name,
                                    tensor.to(device),
                                    totalk,
                                    allowed_topk=self.hparams.topk_compression,
                                )
                            except Exception as e:
                                tplr.logger.warning(
                                    f"Compressed indices check failed for parameter {param_name} from UID {uid}: {e}"
                                )
                                valid_response = False
                                break
                        # Check if values are valid (not NaN, not Inf)
                        elif param_name.endswith("vals"):
                            tensor_to_check = tensor.to(device)
                            if (
                                torch.isnan(tensor_to_check).any()
                                or torch.isinf(tensor_to_check).any()
                            ):
                                tplr.logger.warning(
                                    f"NaN/Inf in {param_name} from UID {uid}, skipping"
                                )
                                valid_response = False
                                break

                    # If any check failed, skip this UID entirely
                    if not valid_response:
                        tplr.logger.info(
                            f"Skipping UID {uid} due to validation failures"
                        )
                        skipped_uids.append(uid)
                        continue
                    # ---------- End Compressed Indices and Values Check ----------

                    # Process tensors (with normalization on 'vals' keys).
                    for param_name, tensor in state_dict_resp.items():
                        if isinstance(tensor, torch.Tensor):
                            if param_name.endswith("vals"):
                                tensor = tensor.to(device)
                                norm = torch.norm(tensor)
                                normalized = tensor / (norm + 1e-8)
                                aggregated_state_dict.setdefault(param_name, []).append(
                                    normalized
                                )
                            else:
                                aggregated_state_dict.setdefault(param_name, []).append(
                                    tensor.to(device)
                                )
                            metrics["download_bytes"] += (
                                tensor.element_size() * tensor.nelement()
                            )

                    valid_uids.append(uid)
                    global_steps.append(global_step_resp)

            except Exception as e:
                tplr.logger.error(f"Error processing uid batch: {str(e)}")

        if not valid_uids:
            tplr.logger.info("No valid gradients received from any UID")
            return None

        total_time = time.time() - start_time
        tplr.logger.info(
            f"Gather done in {total_time:.2f}s. Success rate: {len(valid_uids)}/{len(uids)}, "
            f"Upload: {metrics['upload_bytes']} bytes, Download: {metrics['download_bytes']} bytes"
        )

        result = SimpleNamespace(
            time=total_time,
            upload_bytes=metrics["upload_bytes"],
            download_bytes=metrics["download_bytes"],
            success_rate=len(valid_uids) / len(uids),
            state_dict=SimpleNamespace(**aggregated_state_dict),
            uids=valid_uids,
            global_steps=global_steps,
            skipped_uids=skipped_uids,
        )
        return result

    async def cleanup_old_checkpoints(self, keep_last: int = 3):
        """
        Removes old checkpoints from storage, keeping only the most recent ones.
        """
        try:
            s3_client = await self._get_s3_client(self.bucket)

            paginator = s3_client.get_paginator("list_objects_v2")
            checkpoint_files = []

            async for page in paginator.paginate(
                Bucket=self.bucket.name, Prefix="checkpoint"
            ):
                for obj in page.get("Contents", []):
                    if obj["Key"].startswith("checkpoint"):
                        checkpoint_files.append(obj)

            checkpoint_files.sort(key=lambda x: x["LastModified"], reverse=True)

            if len(checkpoint_files) > keep_last:
                to_delete = checkpoint_files[keep_last:]
                await s3_client.delete_objects(
                    Bucket=self.bucket.name,
                    Delete={"Objects": [{"Key": obj["Key"]} for obj in to_delete]},
                )
                tplr.logger.info(f"Deleted {len(to_delete)} old checkpoints")

        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(self.bucket)
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
            s3_client = await self._get_s3_client(peer_bucket)

            if not hasattr(self, "current_window") or self.current_window is None:
                tplr.logger.error(
                    "current_window is not set in comms. Please set comms.current_window."
                )
                return False

            current_window = self.current_window
            for window in range(current_window - recent_windows, current_window + 1):
                filename = f"gradient-{window}-{uid}-v{__version__}.pt"
                tplr.logger.debug(f"Checking for {filename} in {peer_bucket.name}")
                try:
                    await s3_client.head_object(Bucket=peer_bucket.name, Key=filename)
                    tplr.logger.debug(f"Found {filename} for UID {uid}")
                    return True
                except botocore.exceptions.ClientError as e:
                    if e.response["Error"]["Code"] not in ["404", "403", "401"]:
                        tplr.logger.error(f"Error checking activity for {uid}: {e}")
                        return False
                    tplr.logger.debug(f"{filename} not found for UID {uid}")

        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(peer_bucket)
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
        """
        Sequentially check:
        1. Whether the highest-staked validator has a checkpoint.
        2. Whether the R2 bucket of this instance has a checkpoint.
        3. Whether a checkpoint exists locally.
        If none are found, return None.
        """
        try:
            # 1. Check validator bucket
            (
                validator_bucket,
                validator_uid,
            ) = await self._get_highest_stake_validator_bucket()
            if validator_bucket:
                result = await self._get_bucket_checkpoint(
                    validator_bucket, validator_uid
                )
                if result:
                    # If successfully retrieved, return immediately.
                    return result

            # 2. Check self R2 bucket
            self_bucket = self.bucket  # Use self.bucket saved in __init__
            if self_bucket:
                result = await self._get_bucket_checkpoint(self_bucket, self.uid)
                if result:
                    return result

            # 3. Check local storage
            local_result = self._load_latest_local_checkpoint()
            if local_result:
                return local_result

            tplr.logger.info(
                "No checkpoint found in validator / self R2 / local storage"
            )
            return None

        except Exception as e:
            tplr.logger.error(f"Error getting latest checkpoint: {e}")
            return None

    def _load_latest_local_checkpoint(self):
        try:
            local_dir = os.path.join(LOCAL_TMP_DIR, str(self.uid))
            pattern = rf"checkpoint-(\d+)-{self.uid}-v{__version__}\.pt$"

            if not os.path.exists(local_dir):
                return None

            checkpoints = []
            for window_dir in os.listdir(local_dir):
                path = os.path.join(local_dir, window_dir)
                if not os.path.isdir(path):
                    continue

                for file in os.listdir(path):
                    match = re.match(pattern, file)
                    if match:
                        # window number comes from match.group(1)
                        w = int(match.group(1))
                        file_path = os.path.join(path, file)
                        checkpoints.append(
                            {
                                "path": file_path,
                                "window": w,
                                "modified": os.path.getmtime(file_path),
                            }
                        )

            if checkpoints:
                # choose the last modified checkpoint
                latest = max(checkpoints, key=lambda x: x["modified"])
                checkpoint_data = torch.load(latest["path"], weights_only=True)
                return checkpoint_data, latest["window"]
            else:
                return None
        except Exception as e:
            tplr.logger.error(f"Error in local checkpoint loading: {e}")
            return None

    async def _get_bucket_checkpoint(self, bucket, uid):
        """Helper to get checkpoint from a specific bucket."""
        try:
            s3_client = await self._get_s3_client(bucket)

            pattern = re.compile(rf"^checkpoint-(\d+)-{uid}-v{__version__}\.pt$")

            response = await s3_client.list_objects_v2(
                Bucket=bucket.name, Prefix="checkpoint", MaxKeys=1000
            )
            if not response.get("Contents"):
                return None

            valid_checkpoints = []
            for obj in response.get("Contents", []):
                key = obj.get("Key", "")
                match = pattern.match(key)
                if match:
                    valid_checkpoints.append(
                        {
                            "key": key,
                            "window": int(match.group(1)),
                            "last_modified": obj["LastModified"],
                        }
                    )

            if valid_checkpoints:
                latest = max(valid_checkpoints, key=lambda x: int(x["window"]))
                loaded_data = await self.s3_get_object(key=latest["key"], bucket=bucket)
                if loaded_data:
                    return loaded_data, latest["window"]
            return None
        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)

    async def load_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        current_window: int,
        device: str,
    ) -> tuple[
        bool, dict, int, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler
    ]:
        """
        Loads the latest checkpoint. No catchup or step simulation happens here.
        Returns:
            tuple: (success: bool, momentum: dict, checkpoint_current_window: int,
                    optimizer: Optimizer, scheduler: LRScheduler)
        """
        result = await self.get_latest_checkpoint()
        if not result:
            tplr.logger.info("No valid checkpoints found")
            return False, {}, 0, optimizer, scheduler

        checkpoint_data, checkpoint_window = result
        try:
            # 1) Load model and optimizer state
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
            checkpoint_sync_window = checkpoint_data.get("sync_window")
            if checkpoint_start_window is None or checkpoint_current_window is None:
                tplr.logger.warning(
                    "Checkpoint missing start_window or current_window info"
                )
                return False, {}, 0, optimizer, scheduler

            tplr.logger.info(
                f"Checkpoint loaded. start_window={checkpoint_start_window}, "
                f"checkpoint_current_window={checkpoint_current_window}, "
                f"checkpoint_sync_window={checkpoint_sync_window}, "
                f"local_current_window={current_window}"
            )

            return True, momentum, checkpoint_sync_window, optimizer, scheduler

        except KeyError as e:
            tplr.logger.error(f"Invalid checkpoint format: missing key {e}")
            return False, {}, 0, optimizer, scheduler
        except Exception as e:
            tplr.logger.error(f"Failed to load checkpoint: {e}")
            return False, {}, 0, optimizer, scheduler

    async def post_peer_list(
        self,
        peers: PeerArray,
        first_effective_window: int,
        sync_window: int,
        weights: torch.Tensor,
        initial_selection: bool,
    ):
        """Upload peer list and debug data as JSON to the node's R2 bucket.

        The first_effective_window is a future window (>current_window) from
        which this list peer list will be used.

        The following debugging fields are included in the JSON:
        - sync_window: when the peer list was updated in "validator time"
        - weights: weights for all UIDs, which were used to update the peer
          list (except for during the initial peer selection)
        - initial selection: whether this peer list is the first one in the
          current run
        """
        key = f"{PEERS_FILE_PREFIX}{first_effective_window}_v{__version__}.json"
        peers_and_weights = {
            "peers": peers.tolist(),
            "weights": weights.tolist(),
            "initial_selection": initial_selection,
            "sync_window": sync_window,
            "first_effective_window": first_effective_window,
        }

        # Create temporary JSON file
        temp_file = os.path.join(self.temp_dir, f"temp_{key}")
        try:
            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(peers_and_weights))

            await self.s3_put_object(key=key, file_path=temp_file)
            tplr.logger.info(f"PUT {key} <--")
        except Exception as e:
            tplr.logger.info(f"Failed to upload peer list: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

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

            await self.s3_put_object(key=key, file_path=temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    async def get_peer_list(self) -> tuple[PeerArray, int] | None:
        tplr.logger.info("Starting to look for a peer list on a validator bucket")
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
                    f"Attempting to fetch peer list from UID {validator_uid} bucket {validator_bucket.name}"
                )

                s3_client = await self._get_s3_client(validator_bucket)
                list_args = {
                    "Bucket": validator_bucket.name,
                    "Prefix": PEERS_FILE_PREFIX,
                }
                response = await s3_client.list_objects_v2(**list_args)

                pattern = rf"^{PEERS_FILE_PREFIX}(?P<window>\d+)_v{__version__}\.json$"
                # Filter keys that match the pattern
                keys = [
                    obj["Key"]
                    for obj in response.get("Contents", [])
                    if re.match(pattern, obj["Key"])
                ]
                if len(keys) == 0:
                    tplr.logger.info("No peer list files found")
                    return None
                max_window = -1
                selected_key = None
                for key in keys:
                    match = re.match(pattern, key)
                    if match:
                        window = int(match.group("window"))
                        if window > max_window:
                            max_window = window
                            selected_key = key

                if selected_key is None:
                    tplr.logger.error(
                        f"Failed to select most recent peers file on bucket. First "
                        f"{len(keys[:5])} peer list files are {keys[:5]}"
                    )
                    return None

                peers_data = await self.s3_get_object(
                    key=selected_key, bucket=validator_bucket
                )
                if isinstance(peers_data, dict):
                    peers_dict = peers_data
                else:
                    peers_dict = json.loads(peers_data.decode("utf-8"))
                return np.array(peers_dict["peers"]), peers_dict[
                    "first_effective_window"
                ]

            except (ConnectionClosedError, ClientError):
                await self._purge_s3_client(validator_bucket)
            except Exception as e:
                tplr.logger.error(f"Error fetching peer list: {e}")
                await asyncio.sleep(10)

    async def get_start_window(self) -> int:
        while True:
            try:
                (
                    validator_bucket,
                    validator_uid,
                ) = await self._get_highest_stake_validator_bucket()
                if validator_bucket is None:
                    tplr.logger.warning(
                        "No highest staked validator bucket found. Retrying in 10 seconds"
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
                    "start_window.json not found or empty. Retrying in 10 seconds"
                )
                await asyncio.sleep(10)

            except Exception as e:
                tplr.logger.error(f"Error fetching start_window: {e}")
                await asyncio.sleep(10)

    async def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        momentum,
        global_step,
        current_window,
        start_window,
    ):
        """Save checkpoint to R2 and local storage."""
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

        # save locally
        await self.put(
            state_dict=checkpoint_data,
            uid=str(self.uid),
            window=current_window,
            key="checkpoint",
            global_step=global_step,
            local=True,
        )

        # upload to R2
        await self.put(
            state_dict=checkpoint_data,
            uid=str(self.uid),
            window=current_window,
            key="checkpoint",
            global_step=global_step,
            local=False,
        )

        return True

    async def _gather_window_batch(
        self,
        batch_windows: List[int],
        uid: str,
        peers: List[int],
        device: str,
        totalks: dict,
        global_step: int,
    ) -> Dict[int, SimpleNamespace]:
        """Gather gradients for multiple windows in parallel."""
        try:
            gather_tasks = [
                self.gather(
                    my_uid=uid,
                    uids=peers,
                    window=w,
                    key="gradient",
                    timeout=30,
                    device=device,
                    totalks=totalks,
                    local=False,
                    stale_retention=100,
                )
                for w in batch_windows
            ]
            # Wait for all gather tasks to complete
            batch_results = await asyncio.gather(*gather_tasks, return_exceptions=True)

            # Filter out exceptions and create window->result mapping
            result_dict = {w: None for w in batch_windows}  # Initialize with None
            for window, result in zip(batch_windows, batch_results):
                if not isinstance(result, Exception) and result is not None:
                    result_dict[window] = result

            return result_dict

        except Exception as e:
            tplr.logger.error(
                f"Failed to gather window batch {batch_windows}: {str(e)}"
            )
            return {
                w: None for w in batch_windows
            }  # Return dict with None values on failure

    async def _apply_gathered_gradients(
        self,
        gather_result,
        model: LlamaForCausalLM,
        optimizer: SGD,
        scheduler: SequentialLR,
        transformer: TransformDCT,
        compressor: CompressDCT,
        device: str,
        window: int,
        global_step: int,
    ) -> Tuple[bool, int]:
        """Apply gathered gradients to model parameters.

        Args:
            gather_result: Gathered gradient data
            model: The model to update
            optimizer: SGD optimizer
            scheduler: Learning rate scheduler
            transformer: DCT transformer
            compressor: Gradient compressor
            device: Computing device
            window: Current window number
            global_step: Global step counter

        Returns:
            Tuple[bool, int]: (success, new_global_step)
        """
        try:
            if not gather_result or not gather_result.state_dict:
                return False, global_step

            model.train()
            optimizer.zero_grad()
            model.zero_grad()

            # Apply gradients
            for n, p in model.named_parameters():
                idxs = getattr(gather_result.state_dict, f"{n}idxs", None)
                vals = getattr(gather_result.state_dict, f"{n}vals", None)

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

            optimizer.step()
            scheduler.step()
            global_step += 1

            tplr.logger.info(
                f"Applied gradients for window {window}, global_step => {global_step}"
            )
            return True, global_step

        except Exception as e:
            tplr.logger.error(
                f"Failed to apply gradients for window {window}: {str(e)}"
            )
            return False, global_step

    def check_compressed_indices(
        self, param_name: str, idxs, totalk: int, allowed_topk: int | None = None
    ) -> None:
        """
        Validates that the compressed indices for a given parameter meet the conditions:
          1. If indices are provided as a flat list/tensor, the length must equal min(self.hparams.topk_compression, totalk).
          2. If the indices are multi-dimensional (typically when compressed per row),
             then the size of the last dimension must equal min(self.hparams.topk_compression, totalk).
          3. Every index must be in the valid range [0, totalk-1].

        This function handles both flat and nested (e.g. per-row) indices.
        """
        if allowed_topk is None:
            allowed_topk = self.hparams.topk_compression
        # Only allow up to the maximum available columns.
        allowed_topk = min(allowed_topk, totalk)

        def validate_list(indices):
            # Expected flat list length must equal allowed_topk.
            if len(indices) != allowed_topk:
                raise ValueError(
                    f"[{param_name}] Invalid number of indices: got {len(indices)} but expected {allowed_topk}"
                )
            for idx in indices:
                try:
                    idx_int = int(idx)
                except Exception as e:
                    raise ValueError(
                        f"[{param_name}] Failed to convert index {idx} to int: {e}"
                    )
                if idx_int < 0 or idx_int >= totalk:
                    raise ValueError(
                        f"[{param_name}] Index {idx_int} out of bounds (totalk = {totalk})"
                    )

        # If idxs is a tensor:
        if torch.is_tensor(idxs):
            if idxs.ndim == 1:
                # Flat tensor: expect exactly allowed_topk elements.
                if idxs.size(0) != allowed_topk:
                    raise ValueError(
                        f"[{param_name}] Invalid number of indices: got {idxs.size(0)} but expected {allowed_topk}"
                    )
                for idx in idxs.tolist():
                    if not (0 <= int(idx) < totalk):
                        raise ValueError(
                            f"[{param_name}] Index {int(idx)} out of bounds (totalk = {totalk})"
                        )
            else:
                # Multi-dimensional: check that the last dimension equals allowed_topk.
                if idxs.size(-1) != allowed_topk:
                    raise ValueError(
                        f"[{param_name}] Last dimension size invalid: got {idxs.size(-1)} but expected {allowed_topk}"
                    )
                # Check all indices in the tensor.
                for idx in idxs.flatten().tolist():
                    if not (0 <= int(idx) < totalk):
                        raise ValueError(
                            f"[{param_name}] Index {int(idx)} out of bounds (totalk = {totalk})"
                        )
        # If idxs is a list or tuple
        elif isinstance(idxs, (list, tuple)):
            if idxs and isinstance(idxs[0], (list, tuple)):
                # Nested structure: check each sub-list.
                for sublist in idxs:
                    validate_list(sublist)
            else:
                # Flat list.
                validate_list(list(idxs))
        else:
            # Single value provided.
            try:
                idx_int = int(idxs)
            except Exception as e:
                raise ValueError(
                    f"[{param_name}] Failed to convert index {idxs} to int: {e}"
                )
            if idx_int < 0 or idx_int >= totalk:
                raise ValueError(
                    f"[{param_name}] Index {idx_int} out of bounds (totalk = {totalk})"
                )

    async def load_aggregation(self, window: int):
        """
        Load aggregated gradients for a specified window from the aggregation server.

        Args:
            window: Window number to load

        Returns:
            Processed aggregation data or None if failed
        """
        try:
            bucket_config = BUCKET_SECRETS["aggregator"]
            credentials = bucket_config["credentials"]["read"]

            # Create a Bucket object using specified credentials
            bucket = Bucket(
                name=bucket_config["name"],
                account_id=bucket_config["account_id"],
                access_key_id=credentials["access_key_id"],
                secret_access_key=credentials["secret_access_key"],
            )

            filename = f"aggregator-{window}-v{tplr.__version__}.pt"

            tplr.logger.info(f"Attempting to download aggregation file: {filename}")

            # Use shared async S3 logic instead of boto3 manually
            result = await self.s3_get_object(
                key=filename,
                bucket=bucket,
                timeout=20,
            )

            if result is None:
                tplr.logger.warning(f"No aggregation file found for window {window}")
                return None

            tplr.logger.info(
                f"Successfully loaded aggregation data for window {window}"
            )
            return result

        except Exception as e:
            tplr.logger.error(
                f"Error loading aggregation file for window {window}: {e}"
            )
            return None

    async def get_debug_dict(self, window: int):
        """
        Get debug dictionary from validator bucket for a specific window.

        Args:
            window: Specific window to retrieve debug data for

        Returns:
            Debug dictionary or None if not found
        """
        try:
            (
                validator_bucket,
                validator_uid,
            ) = await self._get_highest_stake_validator_bucket()
            if not validator_bucket or validator_uid is None:
                tplr.logger.warning(
                    "No validator bucket - cannot proceed with debug fetch"
                )
                return

            key = f"debug-{window}-{validator_uid}-v{tplr.__version__}.pt"
            tplr.logger.info(
                f"Attempting to retrieve debug dictionary for window {window} from validator {validator_uid}"
            )

            result = await self.s3_get_object(
                key=key,
                bucket=validator_bucket,
                timeout=20,
            )

            if result is None:
                tplr.logger.warning(f"No debug dictionary found for window {window}")
                return None

            tplr.logger.info(
                f"Successfully retrieved debug dictionary for window {window}"
            )
            return result

        except Exception as e:
            tplr.logger.error(
                f"Error getting debug dictionary for window {window}: {e}"
            )
            import traceback

            tplr.logger.error(traceback.format_exc())
            return None

    def weighted_random_sample_no_replacement(
        self, candidates: list[int], weights: list[int], k: int
    ) -> list[int]:
        """
        Perform a weighted random sample (without replacement) of size k.
        candidates: list of items (uids).
        weights:    list of corresponding weights (integers or floats).
        k:          number of items to sample.
        Returns a list of selected items.
        """
        tplr.logger.debug("Starting weighted random sampling")
        tplr.logger.debug(f"Candidates: {candidates}")
        tplr.logger.debug(f"Weights: {weights}")
        tplr.logger.debug(f"Sample size (k): {k}")

        # Safety checks
        if not candidates or not weights or k <= 0:
            tplr.logger.warning("Invalid input detected. Returning empty list.")
            return []

        # Pair up each candidate with its weight
        pool = list(zip(candidates, weights))
        total_w = float(sum(weights))
        selected = []

        # If total weight is 0, return empty
        if total_w <= 0:
            tplr.logger.warning("Total weight is zero. Returning empty list")
            return []

        tplr.logger.debug(f"Initial total weight: {total_w}")

        for _ in range(min(k, len(candidates))):
            if total_w <= 0 or len(pool) == 0:
                tplr.logger.info("No more items to sample. Stopping early.")
                break

            # Draw a uniform sample in [0, total_w]
            r = random.uniform(0.0, total_w)
            tplr.logger.debug(f"Random threshold: {r}")
            cumulative = 0.0
            for idx, (uid, w) in enumerate(pool):
                cumulative += w
                if cumulative >= r:
                    # Found our pick
                    selected.append(uid)
                    tplr.logger.info(f"Selected item: {uid} with weight {w}")
                    # Remove from pool and subtract from total_w
                    total_w -= w
                    pool.pop(idx)
                    tplr.logger.debug(f"Updated total weight: {total_w}")
                    break

        tplr.logger.debug(f"Final selected items: {selected}")
        return selected
