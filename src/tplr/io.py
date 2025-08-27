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
import asyncio
import concurrent.futures
import functools
import json
import math
import os
import random
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial

# from .hparams import HParams
from types import SimpleNamespace
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
)

import aiofiles
import bittensor as bt
import botocore
import torch
from aiobotocore.client import AioBaseClient
from aiobotocore.session import get_session
from botocore import exceptions
from botocore.exceptions import ClientError, ConnectionClosedError
from tqdm import tqdm as std_tqdm

import tplr
from tplr import decos
from tplr import __version__
from tplr.chain import ChainManager
from tplr.compress import TopKCompressor, unpack_12bit_indices
from tplr.config import BUCKET_SECRETS, client_config
from tplr.schemas import Bucket, CommsGetResult

# Constants
CF_REGION_NAME: str = "enam"
LOCAL_TMP_DIR = "/tmp/local_store"
PEERS_FILE_PREFIX = "peers_"
CPU_COUNT = os.cpu_count() or 4
CPU_MAX_CONNECTIONS = min(100, max(30, CPU_COUNT * 4))
MAX_RETRIES = 3


class S3Manager:
    """A centralized s3 manager"""

    def __init__(
        self,
        # wallet,
        # key_prefix,
        # config,
        # netuid,
        # metagraph,
        # hparams,
        # uid,
        save_location,
    ):
        """Initializes the S3Manager class."""
        self._s3_clients: dict[
            tuple[str, str, str], AioBaseClient
        ] = {}  # (acc_key, sec_key, account_id) -> s3_client

        ## a single aiobotocore session and a dictionary of clients
        self.session = get_session()

        # Get the bucket directly
        self.bucket = get_own_bucket("gradients", "write")

        # set a path for tmp - MUST BE MOUNTED in docker compose!
        self.default_dir = save_location or "/tmp"

        # create s3 client workers
        self.client_semaphore = asyncio.Semaphore(CPU_MAX_CONNECTIONS)

    def __del__(self):
        """Destructor for the S3Manager class."""
        cleanup_temp_dir(self.default_dir)

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
        self._s3_clients.pop(key, None)

    @decos.async_s3_exception_catcher()
    async def cleanup_s3_data(
        self, uid: str, current_window: int, stale_retention: int
    ):
        """Clean up stale S3 data for a given uid."""
        min_allowed_window = current_window - stale_retention

        # Regex pattern to match filenames of the form:
        # gradient-<window>-<uid>-v<version>.pt
        pattern = re.compile(
            rf"^gradient-(\d+)-{uid}-v{re.escape(tplr.__version__)}.pt$"
        )

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

    @decos.async_s3_exception_catcher()
    async def s3_put_object(
        self,
        key: str,
        file_path: str | None = None,
        bucket: Bucket | None = None,
    ):
        """
        Puts an object into S3 storage, handling different file types appropriately.

        Args:
            key (str): The key/path to store the data under
            file_path (str, optional): The local file path to upload
            bucket (Bucket, optional): The bucket to use. Defaults to self.bucket
        """
        if bucket is None:
            bucket = self.bucket

        s3_client = await self._get_s3_client(bucket)

        # Handle JSON files
        if key.endswith(".json") or "start_window" in key:
            if file_path:
                async with aiofiles.open(file_path, "r") as f:
                    data = await f.read()
                    data_bytes = data.encode("utf-8")
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
            await self.upload_large_file(file_path, key, s3_client, bucket)

    @decos.async_s3_exception_catcher()
    async def s3_get_object(
        self,
        key: str,
        bucket: Bucket = None,
        timeout: int = 30,
        time_min: datetime = None,
        time_max: datetime = None,
        load_data: bool = True,
    ):
        """Download object from S3 using asynchronous streaming."""

        unique_path = uuid.uuid4().hex
        temp_dir = make_temp_dir(self.default_dir, unique_path)
        safe_key = key.replace("/", "_")
        filepath = f"temp_{safe_key}_{unique_path}.pt"
        temp_file_path = os.path.join(temp_dir, filepath)

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
            except Exception as e:
                tplr.logger.debug(f"Some other exception occurred: {e=}")
                raise e

            file_size = response["ContentLength"]  # type: ignore

            # Download the object
            if file_size <= 4 * 1024 * 1024 * 1024:  # 4GB
                response = await asyncio.wait_for(
                    s3_client.get_object(Bucket=bucket.name, Key=key),
                    timeout=timeout,
                )
                async with aiofiles.open(temp_file_path, "wb") as f:
                    async with response["Body"] as stream:
                        data = await asyncio.wait_for(stream.read(), timeout=timeout)
                        await f.write(data)
                success = True
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

            loaded_data = None
            if load_data:
                # Now load the data
                if key.endswith(".json") or "start_window" in key:
                    async with aiofiles.open(temp_file_path, "r") as f:
                        data = await f.read()
                        loaded_data = json.loads(data)
                else:
                    loaded_data = torch.load(
                        temp_file_path,
                        map_location="cpu",  # self.config.device,?
                        weights_only=True,
                    )
            else:
                if success:
                    target_directory = os.path.dirname(key)
                    if target_directory:
                        os.makedirs(target_directory, exist_ok=True)

                    # with a cross-device-safe move:
                    loaded_data = shutil.move(temp_file_path, key)
                else:
                    raise FileNotFoundError(
                        f"Download not successful for file at: {temp_file_path}"
                    )

            return loaded_data

        finally:  # Except handled in deco
            cleanup_temp_dir(temp_dir)

    #  Large File Operations
    @decos.async_s3_exception_catcher()
    async def upload_large_file(
        self, file_path: str, key: str, s3_client, bucket: Bucket | None = None
    ):
        """Uploads a large file to S3 using asynchronous multipart upload with 5MB chunks."""
        upload_id = None

        # maybe consider removing this part_size at 32 when the other
        part_size = 32 * 1024 * 1024
        file_size_gb = os.path.getsize(file_path) / (1024 * 1024 * 1024)
        if file_size_gb > 10:
            part_size = 128 * 1024 * 1024
        elif file_size_gb > 1:  # else?
            part_size = 64 * 1024 * 1024

        if bucket is None:
            bucket = self.bucket

        try:
            async with self.client_semaphore:
                for attempt in range(MAX_RETRIES):
                    try:
                        response = await s3_client.create_multipart_upload(
                            Bucket=bucket.name, Key=key
                        )
                        upload_id = response["UploadId"]
                        break
                    except Exception:
                        if attempt == MAX_RETRIES - 1:
                            raise
                        await asyncio.sleep(2**attempt)

                file_size = os.path.getsize(file_path)
                total_parts = math.ceil(file_size / part_size)
                parts = []

                @decos.retry_on_failure(retries=MAX_RETRIES, delay=2.0)
                @decos.async_s3_exception_catcher(on_error_return=lambda x: False)
                async def upload_part(part_number: int):
                    byte_range_start = (part_number - 1) * part_size
                    byte_range_end = min(byte_range_start + part_size, file_size)

                    async with aiofiles.open(file_path, "rb") as f:
                        await f.seek(byte_range_start)
                        data = await f.read(byte_range_end - byte_range_start)

                    response = await s3_client.upload_part(
                        Bucket=bucket.name,
                        Key=key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=data,
                    )
                    return {
                        "ETag": response["ETag"],
                        "PartNumber": part_number,
                    }

                # Launch one coroutine per part
                part_results = await asyncio.gather(
                    *[
                        upload_part(part_number)
                        for part_number in range(1, total_parts + 1)
                    ]
                )
                
                for part_number, part in enumerate(part_results):
                    if part is False:
                        message = f"Failed to upload part {part_number} after {MAX_RETRIES} attempts"
                        tplr.logger.exception(message)
                        raise Exception(message)
                
                parts.extend(part_results)
                parts.sort(key=lambda x: x["PartNumber"])

                for attempt in range(MAX_RETRIES):
                    try:
                        await s3_client.complete_multipart_upload(
                            Bucket=bucket.name,
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

        except Exception as e:
            tplr.logger.error(f"Error during multipart upload of {key}: {e}")
            if upload_id:
                try:
                    await s3_client.abort_multipart_upload(
                        Bucket=bucket.name, Key=key, UploadId=upload_id
                    )
                except Exception as abort_e:
                    tplr.logger.error(f"Failed to abort multipart upload: {abort_e}")
            raise

    @decos.async_s3_exception_catcher(on_error_return=lambda x: False)
    async def download_large_file(
        self, s3_client, bucket: Bucket, key: str, file_size: int, temp_file_path: str
    ):
        """Download large file using multipart download with concurrent chunks."""
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
        target_directory = os.path.dirname(temp_file_path)
        if target_directory:
            os.makedirs(target_directory, exist_ok=True)

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

        @decos.retry_on_failure(retries=3, delay=2.0)
        @decos.async_s3_exception_catcher(on_error_return=lambda x: False)
        async def download_chunk(chunk_number: int):
            """Download a specific chunk with retries."""
            async with semaphore:
                start = chunk_number * chunk_size
                end = min(start + chunk_size, file_size) - 1

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

        try:
            tasks = [
                asyncio.create_task(download_chunk(i)) for i in range(total_chunks)
            ]
            await asyncio.gather(*tasks)

            for n, task in enumerate(tasks):
                if task is False:
                    raise Exception(f"Downloading chunk part {n} failed") 

            if len(downloaded_chunks) != total_chunks:
                missing_chunks = set(range(total_chunks)) - set(
                    downloaded_chunks.keys()
                )
                raise Exception(f"Missing chunks: {missing_chunks}")

            downloaded_size = sum(chunk["size"] for chunk in downloaded_chunks.values())
            if downloaded_size != file_size:
                raise Exception(
                    f"Downloaded size ({downloaded_size}) != expected size ({file_size})"
                )

            return True

        finally:
            pbar.close()

    async def put(
        self,
        state_dict: dict,
        window: int,
        key: Literal["checkpoint", "debug", "gradient", "aggregator"],
        uid: str | None,
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
            bucket_config = BUCKET_SECRETS["aggregator"]
            credentials = bucket_config["credentials"]["write"]

            # Create a Bucket object using specified credentials
            bucket = Bucket(
                name=bucket_config["name"],
                account_id=bucket_config["account_id"],
                access_key_id=credentials["access_key_id"],
                secret_access_key=credentials["secret_access_key"],
            )
        else:
            filename = f"{key}-{window}-{uid}-v{__version__}.pt"
            bucket = None
        tplr.logger.debug(f"PUT {filename} -->")

        put_start = tplr.T()

        # Create per-uid temp directory
        unique_path = uuid.uuid4().hex
        base_dir = os.path.join(self.default_dir, unique_path)
        temp_dir = make_temp_dir(base_dir, uid)
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
            await asyncio.to_thread(torch.save, save_data, temp_file_path)

            if local:
                # Local storage with per-uid directories
                await self.cleanup_local_data(
                    uid=uid, current_window=window, stale_retention=stale_retention
                )
                local_dir = os.path.join(self.default_dir, str(uid), str(window))
                os.makedirs(local_dir, exist_ok=True)
                final_path = os.path.join(local_dir, filename)
                os.replace(temp_file_path, final_path)
            else:
                await self.s3_put_object(filename, temp_file_path, bucket)
                # Remote storage with automatic handling of large files
                asyncio.create_task(
                    self.cleanup_s3_data(
                        uid=uid, current_window=window, stale_retention=stale_retention
                    )
                )

        finally:
            cleanup_temp_dir(temp_dir)

        put_end = tplr.T()
        tplr.logger.info(f"{tplr.P(window, put_end - put_start)} PUT {filename} <--")
        return put_end - put_start

    @decos.async_s3_exception_catcher()
    async def cleanup_old_checkpoints(self, keep_last: int = 3):
        """
        Removes old checkpoints from storage, keeping only the most recent ones.
        """
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

    @decos.async_s3_exception_catcher()
    async def s3_get_object_size(self, bucket: Bucket, key: str) -> Optional[int]:
        """Get the size of an S3 object without downloading it using HEAD request."""
        s3_client = await self._get_s3_client(bucket)

        response = await s3_client.head_object(Bucket=bucket.name, Key=key)
        file_size = response["ContentLength"]

        tplr.logger.debug(f"Object {key} size: {file_size} bytes")
        return file_size

    @decos.async_s3_exception_catcher()
    async def s3_get_object_range(
        self, bucket: Bucket, key: str, start: int, end: int, timeout: int = 30
    ) -> Optional[bytes]:
        """Download a specific byte range from S3 object."""
        s3_client = await self._get_s3_client(bucket)

        response = await asyncio.wait_for(
            s3_client.get_object(
                Bucket=bucket.name, Key=key, Range=f"bytes={start}-{end}"
            ),
            timeout=timeout,
        )

        # Read the chunk data
        async with response["Body"] as stream:
            chunk_data = await asyncio.wait_for(stream.read(), timeout=timeout)

        # Verify chunk size
        expected_size = end - start + 1
        if len(chunk_data) != expected_size:
            raise Exception(
                f"Chunk size mismatch: got {len(chunk_data)}, expected {expected_size}"
            )

        return chunk_data

    def get_base_url(self, account_id: str) -> str:
        """Constructs the base URL for the R2 storage endpoint."""
        return f"https://{account_id}.r2.cloudflarestorage.com"

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
                        delete_local_directory(old_path)
                    except Exception as e:
                        tplr.logger.debug(
                            f"Error removing stale directory {old_path}: {e}"
                        )


@decos.general_exception_catcher()
def get_own_bucket(
    bucket_type: Literal["gradients", "dataset", "aggregator"],
    access_type=None,
) -> Bucket:
    """Gets bucket configuration from environment variables via config.BUCKET_SECRETS.

    Args:
        bucket_type: Either "gradients" or "dataset" to determine which bucket to use
        access_type: For gradients bucket, either "read" or "write" to determine access level
    """
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


def make_temp_dir(temp_dir, modifier) -> str:
    """Create configurable temp directory for this instance"""
    temp_dir = os.path.join(temp_dir, modifier)
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def delete_local_directory(path: str) -> None:
    """Safely remove a local directory and all its contents."""
    if not os.path.exists(path):
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(path)
    return


@decos.s3_exception_catcher()
def cleanup_temp_dir(directory: str | os.PathLike) -> None:
    """Remove dir at the path specified"""
    shutil.rmtree(directory)
    return