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
import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone

# from .hparams import HParams
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Tuple

import aiofiles
import bittensor as bt
import botocore
import numpy as np
import torch
from aiobotocore.client import AioBaseClient
from aiobotocore.session import get_session
from botocore.exceptions import ClientError, ConnectionClosedError
from torch.optim import SGD
from torch.optim.lr_scheduler import SequentialLR
from tqdm import tqdm as std_tqdm
from transformers import LlamaForCausalLM

import tplr as tplr
import tplr.distrib as distrib

from . import __version__
from .chain import ChainManager
from .compress import CompressDCT, TransformDCT
from .config import BUCKET_SECRETS, client_config
from .schemas import Bucket

# Constants
CF_REGION_NAME: str = "enam"
LOCAL_TMP_DIR = "/tmp/local_store"
PEERS_FILE_PREFIX = "peers_"
CPU_COUNT = os.cpu_count() or 4
CPU_MAX_CONNECTIONS = min(100, max(30, CPU_COUNT * 4))

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
        local_rank: int = 0,
        world_size: int = 1,
        **kwargs,
    ):
        self.wallet = wallet
        self.uid = uid
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_ddp = world_size > 1

        # Create temp directory for this instance/rank
        self.temp_dir = os.path.join("/tmp", f"templar_{self.uid}_rank_{self.local_rank}")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Get the bucket directly (assuming rank 0 needs write access primarily)
        access_type = "write" if distrib.is_rank0(self.local_rank) else "read"
        try:
            self.bucket = self.get_own_bucket("gradients", access_type)
        except ValueError:
            if distrib.is_rank0(self.local_rank):
                self.bucket = self.get_own_bucket("gradients", "write")
            else:
                self.bucket = None
                tplr.logger.info(f"[Rank {self.local_rank}] Non-rank 0 process, bucket set to None initially.")

        # Now initialize ChainManager with the bucket (or None for non-rank0)
        super().__init__(
            config=config,
            netuid=netuid,
            metagraph=metagraph,
            hparams=hparams,
            wallet=self.wallet,
            bucket=self.bucket,
            local_rank=self.local_rank,
            world_size=self.world_size,
        )

        # Use the hotkey directly in the save_location
        hotkey = self.wallet.hotkey.ss58_address
        self.save_location = os.path.join("/tmp", f"hotkey_{hotkey}")
        if distrib.is_rank0(self.local_rank):
            os.makedirs(self.save_location, exist_ok=True)
        self.key_prefix = key_prefix

        ## a single aiobotocore session and a dictionary of clients per process
        self.session = get_session()
        self._s3_clients: dict[
            tuple[str, str, str], AioBaseClient
        ] = {}  # (acc_key, sec_key, account_id) -> s3_client

        self.lock = asyncio.Lock()
        self.active_peers = set()
        self.active_check_interval = (
            self.hparams.active_check_interval
        )
        self.recent_windows = (
            self.hparams.recent_windows
        )

        self.client_semaphore = asyncio.Semaphore(CPU_MAX_CONNECTIONS)

        # keep a reference to the *whole* ckpt that was loaded once, so
        # miners / validators can consult keys such as `start_window`.        
        self.last_checkpoint_data: dict[str, Any] | None = None

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
        if bucket is None:
            return
        key = (bucket.access_key_id, bucket.secret_access_key, bucket.account_id)
        if key in self._s3_clients:
            del self._s3_clients[key]

    def start_background_tasks(self):
        if not distrib.is_rank0(self.local_rank):
            tplr.logger.debug(f"[Rank {self.local_rank}] Skipping background task startup.")
            return

        tplr.logger.info("[Rank 0] Starting background tasks...")
        self.loop = asyncio.get_running_loop()

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
        self.loop.set_default_executor(self.executor)

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
        if not distrib.is_rank0(self.local_rank):
            tplr.logger.warning(f"[Rank {self.local_rank}] Attempted to delete directory {path}. Should be Rank 0.")
            return
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
        if not distrib.is_rank0(self.local_rank):
            return
        tplr.logger.debug(f"[Rank 0] Running cleanup_local_data for UID {uid}")
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
        if not distrib.is_rank0(self.local_rank):
            return
        tplr.logger.debug(f"[Rank 0] Running cleanup_s3_data for UID {uid}")
        try:
            min_allowed_window = current_window - stale_retention

            # Regex pattern to match filenames of the form:
            # gradient-<window>-<uid>-v<version>.pt            
            pattern = re.compile(rf"^gradient-(\d+)-{uid}-v{tplr.__version__}.pt$")

            prefix = "gradient"

            # so we get the same credentials as `self.bucket`            
            s3_client = await self._get_s3_client(self.bucket)
            if s3_client is None:
                tplr.logger.error("[Rank 0] Failed to get S3 client for cleanup.")
                return

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
        except Exception as e:
            tplr.logger.error(f"[Rank 0] Error during S3 cleanup for UID {uid}: {e}")

    async def s3_put_object(
        self,
        key: str,
        file_path: Optional[str] = None,
    ):
        """
        Puts an object into S3 storage, handling different file types appropriately.
        Rank 0 only.
        """
        if not distrib.is_rank0(self.local_rank):
            tplr.logger.error(f"[Rank {self.local_rank}] Attempted s3_put_object for key {key}. Should be Rank 0.")
            raise RuntimeError("s3_put_object called by non-rank 0 process")

        try:
            bucket = self.bucket
            if bucket is None:
                raise ValueError("Rank 0 bucket is not configured.")
            s3_client = await self._get_s3_client(bucket)
            if s3_client is None:
                raise ValueError("Failed to get S3 client for Rank 0.")

            if key.endswith(".json") or "start_window" in key:
                if file_path:
                    async with aiofiles.open(file_path, "r") as f:
                        data = await f.read()
                        data_bytes = json.dumps(json.loads(data)).encode("utf-8")
                else:
                    raise ValueError(f"file_path required for JSON file: {key}")

                await s3_client.put_object(Bucket=bucket.name, Key=key, Body=data_bytes)
                return

            file_size = os.path.getsize(file_path)
            multipart_threshold = 100 * 1024 * 1024  # 100MB

            if file_size <= multipart_threshold:
                async with aiofiles.open(file_path, "rb") as f:
                    data = await f.read()
                    await s3_client.put_object(Bucket=bucket.name, Key=key, Body=data)
            else:
                await self.upload_large_file(file_path, key, s3_client)

        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
            raise
        except Exception as e:
            tplr.logger.error(f"[Rank 0] Error uploading {key} to S3: {e}")
            raise

    async def s3_get_object(
        self,
        key: str,
        bucket: Bucket | None = None,
        timeout: int = 15,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
    ):
        """
        Download object from S3 using asynchronous streaming.

        – Rank-0 normally fetches and later broadcasts, but any rank
          can call this as long as it owns valid credentials (`bucket`).

        – If `time_min` / `time_max` are given, the object's
          Last-Modified must fall inside that window, otherwise special
          dicts `{ "__status": "TOO_EARLY" | "TOO_LATE" }` are returned
          (caller inspects and decides).

        Returns: deserialized object or `None`.
        """
        import uuid

        # -------------------------------------------------------------
        #  pick bucket
        # -------------------------------------------------------------
        target_bucket = bucket or self.bucket
        if target_bucket is None:
            tplr.logger.error(
                f"[Rank {self.local_rank}] s3_get_object({key}) – no bucket specified."
            )
            return None

        # -------------------------------------------------------------
        #  temp file
        # -------------------------------------------------------------
        tmp_path = os.path.join(
            self.temp_dir, f"tmp_{uuid.uuid4().hex}_{os.path.basename(key)}"
        )

        # -------------------------------------------------------------
        #  grab client
        # -------------------------------------------------------------
        s3 = await self._get_s3_client(target_bucket)
        if s3 is None:
            tplr.logger.error(
                f"[Rank {self.local_rank}] s3_get_object({key}) – failed to obtain client."
            )
            return None

        try:
            # ---------------------------------------------------------
            #  timezone normalisation
            # ---------------------------------------------------------
            if time_min and not time_min.tzinfo:
                time_min = time_min.replace(tzinfo=timezone.utc)
            if time_max and not time_max.tzinfo:
                time_max = time_max.replace(tzinfo=timezone.utc)

            # ---------------------------------------------------------
            #  HEAD (existence + timestamp sanity)
            # ---------------------------------------------------------
            try:
                head = await asyncio.wait_for(
                    s3.head_object(Bucket=target_bucket.name, Key=key), timeout=timeout
                )
            except asyncio.TimeoutError:
                tplr.logger.debug(f"[Rank {self.local_rank}] head-timeout {key}")
                return None
            except (ConnectionClosedError, ClientError) as e:
                await self._purge_s3_client(target_bucket)
                if "404" in str(e):
                    return None
                raise

            last_mod = head.get("LastModified")
            if last_mod is None:
                return None

            if time_min and last_mod < time_min:
                return {"__status": "TOO_EARLY"}
            if time_max and last_mod > time_max:
                return {"__status": "TOO_LATE"}

            size = head["ContentLength"]  # type: ignore[attr-defined]

            # ---------------------------------------------------------
            #  download
            # ---------------------------------------------------------
            SMALL_THRESHOLD = 5 * 1024 * 1024 * 1024  # 5 GB
            if size <= SMALL_THRESHOLD:
                # single-shot
                obj = await asyncio.wait_for(
                    s3.get_object(Bucket=target_bucket.name, Key=key), timeout=timeout
                )
                async with aiofiles.open(tmp_path, "wb") as fh, obj["Body"] as body:  # type: ignore[arg-type]
                    data = await asyncio.wait_for(body.read(), timeout=timeout)
                    await fh.write(data)
            else:
                ok = await self.download_large_file(
                    s3_client=s3,
                    bucket=target_bucket,
                    key=key,
                    file_size=size,
                    temp_file_path=tmp_path,
                )
                if not ok:
                    return None

            # ---------------------------------------------------------
            #  deserialize
            # ---------------------------------------------------------
            if key.endswith(".json") or "start_window" in key:
                async with aiofiles.open(tmp_path, "r") as fh:
                    loaded = json.loads(await fh.read())
            else:
                map_loc = (
                    f"cuda:{self.local_rank}"
                    if torch.cuda.is_available()
                    else "cpu"
                )
                loaded = torch.load(tmp_path, map_location=map_loc, weights_only=False)

            return loaded

        except asyncio.TimeoutError:
            tplr.logger.debug(f"[Rank {self.local_rank}] get-timeout {key}")
            return None
        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(target_bucket)
            return None
        except Exception as e:
            tplr.logger.error(
                f"[Rank {self.local_rank}] s3_get_object({key}) failed: {e}",
                exc_info=True,
            )
            return None
        finally:
            # clean tmp artefact
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    #  Large File Operations
    async def upload_large_file(self, file_path: str, key: str, s3_client):
        """Uploads a large file to S3 using asynchronous multipart upload. Rank 0 only."""
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
                            tplr.logger.error(f"[Rank 0] Failed create_multipart_upload for {key} after {MAX_RETRIES} attempts.")
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
                                    f"[Rank 0] Failed to upload part {part_number} for {key} after {MAX_RETRIES} attempts: {e}"
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
                            tplr.logger.error(f"[Rank 0] Failed complete_multipart_upload for {key} after {MAX_RETRIES} attempts.")
                            raise
                        await asyncio.sleep(2**attempt)

        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(self.bucket)
        except Exception as e:
            tplr.logger.error(f"[Rank 0] Error during multipart upload of {key}: {e}")
            if upload_id:
                try:
                    await s3_client.abort_multipart_upload(
                        Bucket=self.bucket.name, Key=key, UploadId=upload_id
                    )
                except Exception as abort_e:
                    tplr.logger.error(f"[Rank 0] Failed to abort multipart upload for {key}: {abort_e}")
            raise

    async def download_large_file(
        self, s3_client, bucket: Bucket, key: str, file_size: int, temp_file_path: str
    ):
        """Download large file using multipart download. Can be called by any rank."""
        try:
            gpu_available = torch.cuda.is_available()
            if gpu_available and torch.cuda.device_count() > self.local_rank:
                gpu_mem = torch.cuda.get_device_properties(self.local_rank).total_memory
                max_workers = min(4, CPU_COUNT // self.world_size if self.is_ddp else CPU_COUNT)
                chunk_size = min(
                    max(
                        5 * 1024 * 1024,
                        gpu_mem // (max_workers * 4),
                    ),
                    5 * 1024 * 1024 * 1024,
                )
            else:
                cpu_count = os.cpu_count() or 1
                max_workers = min(4, cpu_count // self.world_size if self.is_ddp else cpu_count)
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

            pbar = None
            if distrib.is_rank0(self.local_rank):
                pbar = std_tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {key} (Rank 0, {max_workers} workers)",
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

                            chunk_len = len(chunk_data)
                            # Verify chunk size matches expected
                            expected_len = end - start + 1
                            if chunk_len != expected_len:
                                raise Exception(
                                    f"Chunk size mismatch: got {chunk_len}, expected {expected_len}"
                                )
                            
                            async with aiofiles.open(temp_file_path, "rb+") as f2:
                                await f2.seek(start)
                                await f2.write(chunk_data)

                            if pbar:
                                pbar.update(chunk_len)
                            downloaded_chunks[chunk_number] = {
                                "start": start,
                                "end": end + 1,
                                "size": chunk_len,
                            }

                            return chunk_number

                        except Exception as e:
                            tplr.logger.error(
                                f"[Rank {self.local_rank}] Error downloading chunk {chunk_number} for {key} (attempt {attempt + 1}/{max_retries}): {e}"
                            )
                            if attempt == max_retries - 1:
                                raise
                            await asyncio.sleep(
                                1 * (attempt + 1)
                            )

            try:
                tasks = [
                    asyncio.create_task(download_chunk(i)) for i in range(total_chunks)
                ]
                await asyncio.gather(*tasks)

                if len(downloaded_chunks) != total_chunks:
                    missing_chunks = set(range(total_chunks)) - set(
                        downloaded_chunks.keys()
                    )
                    raise Exception(f"[Rank {self.local_rank}] Missing chunks for {key}: {missing_chunks}")

                downloaded_size = sum(
                    chunk["size"] for chunk in downloaded_chunks.values()
                )
                if downloaded_size != file_size:
                    raise Exception(
                        f"[Rank {self.local_rank}] Downloaded size ({downloaded_size}) != expected size ({file_size}) for {key}"
                    )

                return True

            finally:
                if pbar:
                    pbar.close()

        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
        except Exception as e:
            tplr.logger.error(f"[Rank {self.local_rank}] Error in download_large_file for {key}: {e}")

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
        Saves the data locally or uploads to S3, then cleans up stale files. Rank 0 only.
        """
        if not distrib.is_rank0(self.local_rank):
            return None
        
        if key == "aggregator":
            filename = f"{key}-{window}-v{__version__}.pt"
        else:
            filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        tplr.logger.debug(f"[Rank 0] PUT {filename} -->")

        put_start = tplr.T()

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
        tplr.logger.info(f"[Rank 0] {tplr.P(window, put_end - put_start)} PUT {filename} <--")
        return put_end - put_start

    async def get(
        self,
        uid: str,
        window: int,
        key: str,
        local: bool = True,
        stale_retention: int = 10,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
    ) -> Optional[tuple[dict, int]]:
        """
        Retrieves gradients / checkpoints.

        – Rank-0 normally calls this; any rank may call it if it has the
          right creds and `local=False`.
        """
        if not distrib.is_rank0(self.local_rank):
            return None

        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        tplr.logger.debug(f"[Rank 0] GET {filename} -->")
        get_start = tplr.T()                                   # ← now used later

        try:
            # -------------------- local --------------------
            if local:
                await self.cleanup_local_data(
                    uid=uid, current_window=window, stale_retention=stale_retention
                )
                local_path = os.path.join(LOCAL_TMP_DIR, str(uid), str(window), filename)
                if not os.path.exists(local_path):
                    tplr.logger.debug(f"Local file not found: {local_path}")
                    return None

                loaded = torch.load(local_path, weights_only=True)
                return (loaded, None) if key == "checkpoint" else (
                    loaded.get("state_dict"),
                    loaded.get("global_step", 0),
                )

            # -------------------- remote -------------------
            peer_bucket = self.commitments.get(int(uid))
            if not peer_bucket:
                return None

            loaded = await self.s3_get_object(
                key=filename,
                bucket=peer_bucket,
                time_min=time_min,
                time_max=time_max,
            )
            if loaded is None:
                return None

            # time-window markers
            if isinstance(loaded, dict) and loaded.get("__status") in ("TOO_LATE", "TOO_EARLY"):
                return loaded

            return (loaded, None) if key == "checkpoint" else (
                loaded.get("state_dict"),
                loaded.get("global_step", 0),
            )

        except Exception as e:
            tplr.logger.debug(f"[Rank 0] GET error {filename}: {e}")
            return None
        finally:
            get_end = tplr.T()
            tplr.logger.info(
                f"[Rank 0] {tplr.P(window, get_end - get_start)} GET {filename} <--"
            )

    async def get_with_retry(
        self,
        uid: str,
        window: int,
        key: str,
        timeout: int,
        local: bool = True,
        stale_retention: int = 10,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
    ) -> Optional[dict]:
        """GET with retry operation."""
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"  # ← for logging                            

        start_time = time.time()
        end_time = start_time + timeout
        tried_after_time_max = False  # Track if we've tried once after passing time_max

        while True:
            # Check if we've timed out
            if time.time() >= end_time:
                tplr.logger.debug(f"GET {uid}/{window}/{key} timed out.")
                return None

            # Check if we're past time_max
            now = datetime.now(timezone.utc)
            past_time_max = time_max is not None and now > time_max

            # If we're past time_max and already tried once, don't retry again
            if past_time_max and tried_after_time_max:
                tplr.logger.debug(
                    f"Already tried once after time_max for UID {uid}, window {window}. Stopping retries."
                )
                return None

            # Make the request
            state_dict = await self.get(
                uid=uid,
                window=window,
                key=key,
                local=local,
                stale_retention=stale_retention,
                time_min=time_min,
                time_max=time_max,
            )

            # Check for TOO_LATE/TOO_EARLY markers
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

            # If we got a result, return it
            if state_dict is not None:
                return state_dict

            # If we're past time_max, mark that we've tried once
            if past_time_max:
                tried_after_time_max = True
                tplr.logger.debug(
                    f"Past time_max for UID {uid}, window {window}. This is the final retry."
                )

            # Short delay before retrying
            await asyncio.sleep(0.1)            
            tplr.logger.debug(f"[Rank 0] RETRY GET {filename} <--")

    async def gather(
        self,
        my_uid: str,
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
        """Gather gradients for multiple windows in parallel. Rank 0 only."""
        if not distrib.is_rank0(self.local_rank):
            return None
        
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
        if not distrib.is_rank0():
            return
        
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
        if not distrib.is_rank0():
            return False
        
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
            max_concurrent = min(30, len(self.commitments) if self.commitments else 10)
            semaphore = asyncio.Semaphore(max_concurrent)
            tplr.logger.debug(f"Commitments: {self.commitments}")
            async def check_peer(uid):
                async with semaphore:
                    is_active = await self.is_miner_active(
                        uid, recent_windows=self.recent_windows
                    )
                    if is_active:
                        active_peers.add(uid)
            # Buffer the processing of commitments to avoid blocking the event loop (over resourcing)
            batch_size = 50
            commitment_uids = list(self.commitments.keys())
            for i in range(0, len(commitment_uids), batch_size):
                batch_uids = commitment_uids[i : i + batch_size]
                batch_tasks = [check_peer(uid) for uid in batch_uids]
                await asyncio.gather(*batch_tasks)
            self.active_peers = active_peers
            tplr.logger.info(
                f"Updated active peers: {[int(uid) for uid in self.active_peers]}"
            )
            await asyncio.sleep(self.active_check_interval)

    # Checkpoint Operations
    async def _get_highest_stake_validator_bucket(self):
        """Get the bucket for the validator with highest stake."""
        if not distrib.is_rank0():
            return None, None
        
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
    async def get_latest_checkpoint(self, version):
        """
        Sequentially check:
        1. Whether the highest-staked validator has a checkpoint.
        2. Whether the R2 bucket of this instance has a checkpoint.
        3. Whether a checkpoint exists locally.
        If none are found, return None.
        """
        if not distrib.is_rank0():
            return None
        
        try:
            # 1. Check validator bucket
            (
                validator_bucket,
                validator_uid,
            ) = await self._get_highest_stake_validator_bucket()
            if validator_bucket:
                result = await self._get_bucket_checkpoint(
                    validator_bucket, validator_uid, version
                )
                if result:
                    # If successfully retrieved, return immediately.
                    return result
            # 2. Check self R2 bucket
            self_bucket = self.bucket  # Use self.bucket saved in __init__
            if self_bucket:
                result = await self._get_bucket_checkpoint(
                    self_bucket, self.uid, version
                )
                if result:
                    return result
            # 3. Check local storage
            local_result = self._load_latest_local_checkpoint(version)
            if local_result:
                return local_result
            tplr.logger.info(
                "No checkpoint found in validator / self R2 / local storage"
            )
            return None
        except Exception as e:
            tplr.logger.error(f"Error getting latest checkpoint: {e}")
            return None
    def _load_latest_local_checkpoint(self, version: str):
        if not distrib.is_rank0():
            return None
        
        try:
            local_dir = os.path.join(LOCAL_TMP_DIR, str(self.uid))
            pattern = rf"checkpoint-(\d+)-{self.uid}-v{re.escape(version)}\.pt$"
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

    async def _get_bucket_checkpoint(self, bucket, uid, version: str):
        """Helper to get checkpoint from a specific bucket."""
        if not distrib.is_rank0():
            return None
        try:
            s3_client = await self._get_s3_client(bucket)

            pat = re.compile(rf"^checkpoint-(\d+)-{uid}-v{re.escape(version)}\.pt$")

            latest_checkpoint = None
            max_window = -1

            continuation_token = None

            while True:
                list_kwargs = {
                    "Bucket": bucket.name,
                    "Prefix": "checkpoint",
                }
                if continuation_token:
                    list_kwargs["ContinuationToken"] = continuation_token

                response = await s3_client.list_objects_v2(**list_kwargs)

                # If no objects returned, stop checking
                if not response.get("Contents"):
                    break

                # Iterate through returned objects to find valid checkpoints
                for obj in response["Contents"]:
                    key = obj.get("Key", "")
                    match = pat.match(key)
                    if match:
                        window_number = int(match.group(1))
                        if window_number > max_window:
                            max_window = window_number
                            latest_checkpoint = key


                if response.get("IsTruncated"):
                    continuation_token = response.get("NextContinuationToken")
                else:
                    break

            if latest_checkpoint:
                tplr.logger.info(f"[Rank 0] Found latest checkpoint key: {latest_checkpoint}")
                loaded_data = await self.s3_get_object(
                    key=latest_checkpoint, bucket=bucket
                )
                if loaded_data:
                    self.last_checkpoint_data = loaded_data
                    return loaded_data, max_window
                else:
                    tplr.logger.warning(f"[Rank 0] Failed to load data for checkpoint key: {latest_checkpoint}")

            tplr.logger.info("[Rank 0] No suitable checkpoint found.")
            return None

        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
            tplr.logger.warning("[Rank 0] S3 connection error while getting latest checkpoint.")
            return None
        except Exception as e:
            tplr.logger.error(f"[Rank 0] Error getting latest checkpoint: {e}", exc_info=True)
            return None

    async def load_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        current_window: int,
        device: str,
        init_version: Optional[str] = None,
    ) -> tuple[
        bool, dict, int, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler
    ]:
        """
        Loads the latest checkpoint. Rank 0 loads, broadcasts, others receive.
        This function itself *only* loads on Rank 0. Broadcasting happens externally.
        Returns:
            tuple: (load_success_on_rank0: bool, state_data_for_broadcast: dict, checkpoint_sync_window: int,
                    optimizer: Optimizer, scheduler: LRScheduler)
        """
        if not distrib.is_rank0(self.local_rank):
            return False, {}, 0, optimizer, scheduler

        tplr.logger.info("[Rank 0] Attempting to load checkpoint...")
        device = f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'

        init_version = init_version if init_version is not None else __version__
        result = await self.get_latest_checkpoint(init_version)
        if not result:
            tplr.logger.info("[Rank 0] No valid checkpoints found via get_latest_checkpoint.")
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
            self.last_checkpoint_data = checkpoint_data

            return True, momentum, checkpoint_sync_window, optimizer, scheduler



        except KeyError as e:
            tplr.logger.error(f"[Rank 0] Invalid checkpoint format: missing key {e}")
            return False, {}, 0, optimizer, scheduler
        except Exception as e:
            tplr.logger.error(f"[Rank 0] Failed to load checkpoint state locally: {e}", exc_info=True)
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
        if not distrib.is_rank0(self.local_rank):
            return
        key = f"{PEERS_FILE_PREFIX}{first_effective_window}_v{__version__}.json"
        peers_and_weights = {
            "peers": peers.tolist(),
            "weights": weights.tolist(),
            "initial_selection": initial_selection,
            "sync_window": sync_window,
            "first_effective_window": first_effective_window,
        }

        temp_file = os.path.join(self.temp_dir, f"temp_{key}")
        try:
            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(peers_and_weights))

            await self.s3_put_object(key=key, file_path=temp_file)
            tplr.logger.info(f"[Rank 0] PUT {key} <--")
        except Exception as e:
            tplr.logger.info(f"[Rank 0] Failed to upload peer list {key}: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    async def post_start_window(self, start_window: int):
        """Upload the start window. Rank 0 only."""
        if not distrib.is_rank0(self.local_rank):
            return

        key = f"start_window_v{__version__}.json"
        start_window_data = {"start_window": start_window}

        temp_file = os.path.join(self.temp_dir, f"temp_{key}")
        try:
            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(start_window_data))

            await self.s3_put_object(key=key, file_path=temp_file)
            tplr.logger.info(f"[Rank 0] PUT {key} <--")
        except Exception as e:
            tplr.logger.error(f"[Rank 0] Failed to post start window {key}: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    async def get_peer_list(
        self, fetch_previous: bool = False
    ) -> tuple[PeerArray, int] | None:
        """
        Fetch the current or (optionally) previous peer-list JSON from the
        highest-staked validator's bucket. Rank-0 only.

        – returns (np.ndarray peers, first_effective_window) or None
        – retries forever (10 s back-off) unless a fatal/logic error occurs
        """
        if not distrib.is_rank0(self.local_rank):
            return None

        pat = re.compile(
            rf"^{PEERS_FILE_PREFIX}(?P<window>\d+)_v{__version__}\.json$"
        )

        tplr.logger.info(
            f"[Rank 0] Looking for a "
            f"{'previous' if fetch_previous else 'current'} peer list on a validator bucket"
        )

        while True:
            try:
                validator_bucket, validator_uid = await self._get_highest_stake_validator_bucket()
                if validator_bucket is None:
                    tplr.logger.warning(
                        "[Rank 0] No highest-staked validator bucket found. Retrying in 10 s."
                    )
                    await asyncio.sleep(10)
                    continue

                tplr.logger.info(
                    f"[Rank 0] Fetching peer list from UID {validator_uid} – bucket {validator_bucket.name}"
                )

                s3 = await self._get_s3_client(validator_bucket)
                if s3 is None:
                    tplr.logger.error(
                        f"[Rank 0] Failed to get S3 client for {validator_bucket.name}. Retrying in 10 s."
                    )
                    await asyncio.sleep(10)
                    continue

                # ---------- enumerate all peer-list files ----------
                keys: list[str] = []
                token = None
                while True:
                    list_args = {
                        "Bucket": validator_bucket.name,
                        "Prefix": PEERS_FILE_PREFIX,
                        "MaxKeys": 1000,
                    }
                    if token:
                        list_args["ContinuationToken"] = token

                    resp = await s3.list_objects_v2(**list_args)
                    for obj in resp.get("Contents", []):
                        if pat.match(obj["Key"]):
                            keys.append(obj["Key"])

                    if resp.get("IsTruncated"):
                        token = resp.get("NextContinuationToken")
                    else:
                        break

                if not keys:
                    tplr.logger.info("[Rank 0] No peer-list files found.")
                    return None

                # ---------- pick window ----------
                window_to_key: dict[int, str] = {}
                for k in keys:
                    m = pat.match(k)
                    if m:
                        window_to_key[int(m.group("window"))] = k

                if not window_to_key:
                    tplr.logger.error(
                        "[Rank 0] Failed to parse windows from peer-list filenames; "
                        f"sample keys: {keys[:5]}"
                    )
                    return None

                sorted_windows = sorted(window_to_key.keys(), reverse=True)
                if fetch_previous:
                    if len(sorted_windows) < 2:
                        tplr.logger.info("[Rank 0] No previous peer-list window available.")
                        return None
                    selected_window = sorted_windows[1]
                else:
                    selected_window = sorted_windows[0]

                selected_key = window_to_key[selected_window]
                tplr.logger.info(f"[Rank 0] Selected window {selected_window} – key {selected_key}")

                # ---------- download & decode ----------
                raw = await self.s3_get_object(key=selected_key, bucket=validator_bucket)
                if raw is None:
                    tplr.logger.warning(
                        f"[Rank 0] Failed to download peer-list {selected_key}. Retrying in 10 s."
                    )
                    await asyncio.sleep(10)
                    continue

                peers_dict = raw if isinstance(raw, dict) else json.loads(raw.decode("utf-8"))
                return np.array(peers_dict["peers"]), peers_dict["first_effective_window"]

            except (ConnectionClosedError, ClientError):
                await self._purge_s3_client(validator_bucket)
                tplr.logger.warning(
                    "[Rank 0] S3 connection error while fetching peer list. Retrying in 5 s..."
                )
                await asyncio.sleep(5)
            except Exception as e:
                tplr.logger.error(f"[Rank 0] Error fetching peer list: {e}", exc_info=True)
                await asyncio.sleep(10)


    async def get_start_window(self, retries: int = -1) -> int | None:
        """Get start window from validator bucket. Rank 0 only."""
        if not distrib.is_rank0(self.local_rank):
            return None

        attempt = 0
        while retries == -1 or attempt < retries:
            try:
                (
                    validator_bucket,
                    validator_uid,
                ) = await self._get_highest_stake_validator_bucket()
                if validator_bucket is None:
                    tplr.logger.warning(
                        "[Rank 0] No highest staked validator bucket found. Retrying in 10 seconds"
                    )
                    attempt += 1
                    await asyncio.sleep(10)
                    continue

                tplr.logger.info(
                    f"[Rank 0] Attempting to fetch start_window from UID {validator_uid} bucket {validator_bucket.name}"
                )
                start_window_key = f"start_window_v{__version__}.json"
                start_window_data = await self.s3_get_object(
                    key=start_window_key, bucket=validator_bucket
                )

                if start_window_data is not None:
                    if isinstance(start_window_data, dict):
                        start_window_json = start_window_data
                    else:
                        start_window_json = json.loads(start_window_data.decode("utf-8"))

                    start_window = start_window_json["start_window"]
                    tplr.logger.info(f"[Rank 0] Fetched start_window: {start_window}")
                    return start_window

                tplr.logger.warning(
                    f"[Rank 0] {start_window_key} not found or empty. Retrying in 10 seconds"
                )
                attempt += 1
                await asyncio.sleep(10)

            except (ConnectionClosedError, ClientError):
                await self._purge_s3_client(validator_bucket)
                tplr.logger.warning("[Rank 0] S3 connection error while fetching start window. Retrying...")
                attempt += 1
                await asyncio.sleep(5)
            except Exception as e:
                tplr.logger.error(f"[Rank 0] Error fetching start_window: {e}", exc_info=True)
                attempt += 1
                await asyncio.sleep(10)

        tplr.logger.warning("[Rank 0] Max retries exceeded while trying to fetch start_window")
        return None

    async def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        momentum,
        global_step,
        current_window,
        start_window,
        sync_window,
    ):
        """Save checkpoint to R2 and local storage. Rank 0 only."""
        if not distrib.is_rank0(self.local_rank):
            return False

        tplr.logger.info(f"[Rank 0] Saving checkpoint for window {current_window} (sync_window: {sync_window})...")
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

        optimizer_state_cpu = {'state': {}, 'param_groups': optimizer.state_dict()['param_groups']}
        for group_id, group_state in optimizer.state_dict()['state'].items():
            optimizer_state_cpu['state'][group_id] = {}
            for k, v in group_state.items():
                optimizer_state_cpu['state'][group_id][k] = v.cpu() if torch.is_tensor(v) else v

        checkpoint_data = {
            "model_state_dict": {
                k: v.cpu().clone() for k, v in model_to_save.state_dict().items()
            },
            "optimizer_state_dict": optimizer_state_cpu,
            "scheduler_state_dict": scheduler.state_dict(),
            "momentum": {k: v.cpu().clone() for k, v in momentum.items()},
            "start_window": start_window,
            "current_window": current_window,
            "sync_window": sync_window,
            "version": __version__,
            "global_step": global_step,
        }

        save_success = True
        try:
            await self.put(
                state_dict=checkpoint_data,
                uid=str(self.uid),
                window=sync_window,
                key="checkpoint",
                global_step=global_step,
                local=True,
            )
            tplr.logger.info(f"[Rank 0] Checkpoint saved locally for sync_window {sync_window}.")
        except Exception as e:
            tplr.logger.error(f"[Rank 0] Failed to save checkpoint locally for sync_window {sync_window}: {e}", exc_info=True)
            save_success = False

        try:
            await self.put(
                state_dict=checkpoint_data,
                uid=str(self.uid),
                window=sync_window,
                key="checkpoint",
                global_step=global_step,
                local=False,
            )
            tplr.logger.info(f"[Rank 0] Checkpoint uploaded to S3 for sync_window {sync_window}.")
        except Exception as e:
            tplr.logger.error(f"[Rank 0] Failed to upload checkpoint to S3 for sync_window {sync_window}: {e}", exc_info=True)
            save_success = False

        return save_success

    async def _gather_window_batch(
        self,
        batch_windows: List[int],
        uid: str,
        peers: List[int],
        device: str,
        totalks: dict,
        global_step: int,
    ) -> Dict[int, SimpleNamespace | None]:
        """Gather gradients for multiple windows in parallel. Rank 0 only."""
        if not distrib.is_rank0(self.local_rank):
            return {w: None for w in batch_windows}
        
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
            batch_results = await asyncio.gather(*gather_tasks, return_exceptions=True)

            result_dict = {w: None for w in batch_windows}
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
            }

    async def _apply_gathered_gradients(
        self,
        gather_result,
        model: LlamaForCausalLM,
        optimizer: SGD,
        scheduler: SequentialLR,
        transformer: TransformDCT,
        compressor: CompressDCT,
        window: int,              
        global_step: int,         
        device: str | None = None,
    ) -> Tuple[bool, int]:
        """Apply gathered gradients to model parameters. Rank-0 only."""
        if not distrib.is_rank0(self.local_rank):
            return False, global_step

        # ----------- device sanity -----------
        if device is None:                                   # allow caller to omit
            device = (
                f"cuda:{self.local_rank}"
                if torch.cuda.is_available()
                else "cpu"
            )

        try:
            if not gather_result or not gather_result.state_dict:
                tplr.logger.warning(f"[Rank 0] No gathered result/state_dict to apply for window {window}.")
                return False, global_step

            model_to_update = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

            model_to_update.train()
            optimizer.zero_grad()

            for n, p in model_to_update.named_parameters():
                idxs_key = f"{n}idxs"
                vals_key = f"{n}vals"

                idxs = getattr(gather_result.state_dict, idxs_key, None)
                vals = getattr(gather_result.state_dict, vals_key, None)

                if idxs is not None and vals is not None:
                    # ensure LongTensor on correct device
                    if torch.is_tensor(idxs):
                        idxs_tensor = idxs.to(device)
                    else:
                        idxs_tensor = torch.as_tensor(idxs, device=device, dtype=torch.long)

                    if torch.is_tensor(vals):
                        vals_tensor = vals.to(device)
                    else:
                        vals_tensor = torch.as_tensor(vals, device=device, dtype=p.dtype)

                    new_grad = transformer.decode(
                        compressor.batch_decompress(
                            p.to(device),
                            idxs_tensor,
                            vals_tensor,
                            transformer.shapes[n],
                            transformer.totalks[n],
                        )
                    ).to(device)

                    if p.grad is None:
                        p.grad = new_grad
                    else:
                        p.grad.copy_(new_grad)

            optimizer.step()
            scheduler.step()
            global_step += 1

            tplr.logger.info(
                f"[Rank 0] Applied gathered gradients for window {window}, global_step => {global_step}"
            )
            return True, global_step

        except Exception as e:
            tplr.logger.error(
                f"[Rank 0] Failed to apply gathered gradients for window {window}: {str(e)}"
            )
            optimizer.zero_grad()
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
        Load aggregated gradients for a specified window from the aggregation server. Rank 0 only.
        """
        if not distrib.is_rank0(self.local_rank):
            return None
        
        try:
            bucket_config = BUCKET_SECRETS["aggregator"]
            credentials = bucket_config["credentials"]["read"]

            bucket = Bucket(
                name=bucket_config["name"],
                account_id=bucket_config["account_id"],
                access_key_id=credentials["access_key_id"],
                secret_access_key=credentials["secret_access_key"],
            )

            filename = f"aggregator-{window}-v{tplr.__version__}.pt"

            tplr.logger.info(f"Attempting to download aggregation file: {filename}")

            result = await self._get_bucket_checkpoint(bucket, self.uid, __version__)
            if result:
                self.last_checkpoint_data = result[0]
                return result[0], result[1]

            tplr.logger.info("No suitable checkpoint found.")
            return None

        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
            tplr.logger.warning("S3 connection error while getting latest checkpoint.")
            return None
        except Exception as e:
            tplr.logger.error(f"Error getting latest checkpoint: {e}", exc_info=True)
            return None

    async def get_debug_dict(self, window: int):
        """
        Get debug dictionary from validator bucket for a specific window. Rank 0 only.
        """
        if not distrib.is_rank0(self.local_rank):
            return None
        
        # -------------------------------------------------------------
        #  discover bucket
        # -------------------------------------------------------------
        validator_bucket: Bucket | None = None   # for later purge
        try:
            validator_bucket, validator_uid = await self._get_highest_stake_validator_bucket()
            if validator_bucket is None or validator_uid is None:
                tplr.logger.warning("[Rank 0] No validator bucket – cannot fetch debug dict.")
                return None

            key = f"debug-{window}-{validator_uid}-v{tplr.__version__}.pt"
            tplr.logger.info(
                f"[Rank 0] Attempting debug-dict download: {key}  (window={window})"
            )

            debug_data = await self.s3_get_object(
                key=key,
                bucket=validator_bucket,
                timeout=20,
            )

            if debug_data is None:
                tplr.logger.warning(f"[Rank 0] No debug dictionary found for window {window}")
                return None

            tplr.logger.info(f"[Rank 0] Successfully retrieved debug dictionary for window {window}")
            return debug_data

        except (ConnectionClosedError, ClientError):
            if validator_bucket is not None:
                await self._purge_s3_client(validator_bucket)
            tplr.logger.warning("[Rank 0] S3 connection error while getting debug dict; returning None.")
            return None
        except Exception as e:
            tplr.logger.error(f"[Rank 0] Error getting debug dictionary for window {window}: {e}", exc_info=True)
            return None

    def weighted_random_sample_no_replacement(
        self, candidates: list[int], weights: list[int], k: int
    ) -> list[int]:
        """
        Perform a weighted random sample (without replacement) of size k. Rank 0 only.
        """
        if not distrib.is_rank0(self.local_rank):
            return []
        
        tplr.logger.debug("Starting weighted random sampling")
        tplr.logger.debug(f"Candidates: {candidates}")
        tplr.logger.debug(f"Weights: {weights}")
        tplr.logger.debug(f"Sample size (k): {k}")

        if not candidates or not weights or k <= 0:
            tplr.logger.warning("Invalid input detected. Returning empty list.")
            return []

        pool = list(zip(candidates, weights))
        total_w = float(sum(weights))
        selected = []

        if total_w <= 0:
            tplr.logger.warning("Total weight is zero. Returning empty list")
            return []

        tplr.logger.debug(f"Initial total weight: {total_w}")

        for _ in range(min(k, len(candidates))):
            if total_w <= 0 or len(pool) == 0:
                tplr.logger.info("No more items to sample. Stopping early.")
                break

            r = random.uniform(0.0, total_w)
            tplr.logger.debug(f"Random threshold: {r}")
            cumulative = 0.0
            for idx, (uid, w) in enumerate(pool):
                cumulative += w
                if cumulative >= r:
                    selected.append(uid)
                    total_w -= w
                    pool.pop(idx)
                    break

        tplr.logger.debug(f"Final selected items: {selected}")
        return selected
