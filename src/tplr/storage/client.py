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

import asyncio
import json
import math
import os
from datetime import datetime, timezone
from typing import Optional, List

import aiofiles
import torch
from aiobotocore.client import AioBaseClient
from aiobotocore.session import get_session
from botocore.exceptions import ClientError, ConnectionClosedError
from tqdm import tqdm as std_tqdm

import tplr
from ..config import client_config
from ..schemas import Bucket

# Constants
CF_REGION_NAME: str = "enam"
CPU_COUNT = os.cpu_count() or 4
CPU_MAX_CONNECTIONS = min(100, max(30, CPU_COUNT * 4))


class StorageClient:
    """Handles all S3/R2 storage operations"""
    
    def __init__(self, temp_dir: str):
        """Initialize with aiobotocore session and temp directory"""
        self.session = get_session()
        self._s3_clients: dict[tuple[str, str, str], AioBaseClient] = {}
        self.temp_dir = temp_dir
        self.client_semaphore = asyncio.Semaphore(CPU_MAX_CONNECTIONS)
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)

    def get_base_url(self, account_id: str) -> str:
        """Constructs the base URL for the R2 storage endpoint."""
        return f"https://{account_id}.r2.cloudflarestorage.com"

    async def _get_s3_client(self, bucket: Bucket) -> AioBaseClient:
        """Returns a persistent s3_client for the given bucket credentials."""
        key = (bucket.access_key_id, bucket.secret_access_key, bucket.account_id)
        if key in self._s3_clients:
            return self._s3_clients[key]

        endpoint_url = self.get_base_url(bucket.account_id)
        
        new_client = self.session.create_client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=bucket.access_key_id,
            aws_secret_access_key=bucket.secret_access_key,
        )
        new_client = await new_client.__aenter__()
        
        self._s3_clients[key] = new_client
        return new_client

    async def _purge_s3_client(self, bucket: Bucket) -> None:
        """Remove S3 client from cache"""
        key = (bucket.access_key_id, bucket.secret_access_key, bucket.account_id)
        if key in self._s3_clients:
            del self._s3_clients[key]

    async def get_object(self, key: str, bucket: Bucket, timeout: int = 15) -> Optional[bytes]:
        """Download object from S3 using asynchronous streaming."""
        import uuid
        
        temp_file_path = os.path.join(self.temp_dir, f"temp_{key}_{uuid.uuid4().hex}.pt")
        
        try:
            s3_client = await self._get_s3_client(bucket)
            
            # HEAD the object first
            try:
                response = await asyncio.wait_for(
                    s3_client.head_object(Bucket=bucket.name, Key=key),
                    timeout=timeout,
                )
                file_size = response["ContentLength"]
            except asyncio.TimeoutError:
                tplr.logger.debug(f"Timeout checking for {key}")
                return None
            except (ConnectionClosedError, ClientError) as e:
                await self._purge_s3_client(bucket)
                if "404" in str(e):
                    tplr.logger.debug(f"Object {key} not found in bucket {bucket.name}")
                    return None
                raise

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
                success = await self.multipart_download(key, temp_file_path, bucket)
                if not success:
                    return None

            # Read the file back
            async with aiofiles.open(temp_file_path, "rb") as f:
                return await f.read()

        except asyncio.TimeoutError:
            tplr.logger.debug(f"Timeout downloading {key}")
            return None
        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
            return None
        except Exception as e:
            tplr.logger.error(f"Error in get_object for {key}: {e}")
            return None
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    async def put_object(self, key: str, data: bytes, bucket: Bucket) -> bool:
        """Upload object to S3 storage."""
        import uuid
        
        temp_file_path = os.path.join(self.temp_dir, f"temp_put_{uuid.uuid4().hex}")
        
        try:
            # Write data to temp file
            async with aiofiles.open(temp_file_path, "wb") as f:
                await f.write(data)
            
            file_size = len(data)
            multipart_threshold = 100 * 1024 * 1024  # 100MB
            
            s3_client = await self._get_s3_client(bucket)
            
            if file_size <= multipart_threshold:
                # Simple upload for small files
                await s3_client.put_object(Bucket=bucket.name, Key=key, Body=data)
            else:
                # Multipart upload for large files
                await self.multipart_upload(key, temp_file_path, bucket)
            
            return True
            
        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
            return False
        except Exception as e:
            tplr.logger.error(f"Error uploading {key} to S3: {e}")
            return False
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    async def delete_object(self, key: str, bucket: Bucket) -> bool:
        """Delete object from S3 storage."""
        try:
            s3_client = await self._get_s3_client(bucket)
            await s3_client.delete_object(Bucket=bucket.name, Key=key)
            return True
        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
            return False
        except Exception as e:
            tplr.logger.error(f"Error deleting {key}: {e}")
            return False

    async def list_objects(self, prefix: str, bucket: Bucket) -> List[str]:
        """List objects with given prefix."""
        try:
            s3_client = await self._get_s3_client(bucket)
            keys = []
            continuation_token = None
            
            while True:
                list_args = {
                    "Bucket": bucket.name,
                    "Prefix": prefix,
                    "MaxKeys": 1000,
                }
                if continuation_token:
                    list_args["ContinuationToken"] = continuation_token
                
                response = await s3_client.list_objects_v2(**list_args)
                contents = response.get("Contents", [])
                
                for obj in contents:
                    keys.append(obj["Key"])
                
                if response.get("IsTruncated"):
                    continuation_token = response.get("NextContinuationToken")
                else:
                    break
            
            return keys
            
        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
            return []
        except Exception as e:
            tplr.logger.error(f"Error listing objects with prefix {prefix}: {e}")
            return []

    async def get_object_size(self, key: str, bucket: Bucket) -> Optional[int]:
        """Get the size of an S3 object without downloading it."""
        try:
            s3_client = await self._get_s3_client(bucket)
            response = await s3_client.head_object(Bucket=bucket.name, Key=key)
            return response["ContentLength"]
        except (ConnectionClosedError, ClientError) as e:
            await self._purge_s3_client(bucket)
            if "404" in str(e):
                return None
            tplr.logger.error(f"Error getting object size for {key}: {e}")
            return None
        except Exception as e:
            tplr.logger.error(f"Error getting object size for {key}: {e}")
            return None

    async def get_object_range(self, key: str, start: int, end: int, bucket: Bucket) -> Optional[bytes]:
        """Download a specific byte range from S3 object."""
        try:
            s3_client = await self._get_s3_client(bucket)
            
            response = await asyncio.wait_for(
                s3_client.get_object(
                    Bucket=bucket.name, Key=key, Range=f"bytes={start}-{end}"
                ),
                timeout=30,
            )
            
            async with response["Body"] as stream:
                chunk_data = await asyncio.wait_for(stream.read(), timeout=30)
            
            expected_size = end - start + 1
            if len(chunk_data) != expected_size:
                raise Exception(f"Chunk size mismatch: got {len(chunk_data)}, expected {expected_size}")
            
            return chunk_data
            
        except asyncio.TimeoutError:
            tplr.logger.error(f"Timeout downloading range {start}-{end} for {key}")
            return None
        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
            return None
        except Exception as e:
            tplr.logger.error(f"Error downloading range {start}-{end} for {key}: {e}")
            return None

    async def multipart_upload(self, key: str, file_path: str, bucket: Bucket) -> bool:
        """Uploads a large file to S3 using asynchronous multipart upload."""
        upload_id = None
        MAX_RETRIES = 3
        PART_SIZE = 5 * 1024 * 1024  # 5MB
        
        try:
            async with self.client_semaphore:
                s3_client = await self._get_s3_client(bucket)
                
                # Create multipart upload
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
                        except Exception as e:
                            if attempt == MAX_RETRIES - 1:
                                tplr.logger.error(f"Failed to upload part {part_number}: {e}")
                                raise
                            await asyncio.sleep(2**attempt)
                
                # Upload all parts
                part_results = await asyncio.gather(
                    *[upload_part(part_number) for part_number in range(1, total_parts + 1)]
                )
                parts.extend(part_results)
                parts.sort(key=lambda x: x["PartNumber"])
                
                # Complete multipart upload
                for attempt in range(MAX_RETRIES):
                    try:
                        await s3_client.complete_multipart_upload(
                            Bucket=bucket.name,
                            Key=key,
                            UploadId=upload_id,
                            MultipartUpload={"Parts": parts},
                        )
                        tplr.logger.info(f"Successfully uploaded {key}")
                        return True
                    except Exception:
                        if attempt == MAX_RETRIES - 1:
                            raise
                        await asyncio.sleep(2**attempt)
                        
        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
            return False
        except Exception as e:
            tplr.logger.error(f"Error during multipart upload of {key}: {e}")
            if upload_id:
                try:
                    s3_client = await self._get_s3_client(bucket)
                    await s3_client.abort_multipart_upload(
                        Bucket=bucket.name, Key=key, UploadId=upload_id
                    )
                except Exception as abort_e:
                    tplr.logger.error(f"Failed to abort multipart upload: {abort_e}")
            return False

    async def multipart_download(self, key: str, file_path: str, bucket: Bucket) -> bool:
        """Download large file using multipart download with concurrent chunks."""
        try:
            s3_client = await self._get_s3_client(bucket)
            
            # Get file size
            file_size = await self.get_object_size(key, bucket)
            if file_size is None:
                return False
            
            # Determine optimal chunk size and concurrency
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                max_workers = min(torch.cuda.device_count() * 4, 16)
                chunk_size = min(
                    max(5 * 1024 * 1024, gpu_mem // (max_workers * 4)),
                    5 * 1024 * 1024 * 1024,
                )
            else:
                cpu_count = os.cpu_count() or 1
                max_workers = min(cpu_count * 4, 16)
                chunk_size = min(
                    max(5 * 1024 * 1024, file_size // (max_workers * 2)),
                    5 * 1024 * 1024 * 1024,
                )
            
            total_chunks = math.ceil(file_size / chunk_size)
            max_workers = min(max_workers, total_chunks)
            semaphore = asyncio.Semaphore(max_workers)
            
            # Create the file with correct size
            async with aiofiles.open(file_path, "wb") as f:
                await f.truncate(file_size)
            
            # Progress tracking
            pbar = std_tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {key} ({max_workers} workers)",
            )
            
            downloaded_chunks = {}
            
            async def download_chunk(chunk_number: int, max_retries: int = 3):
                for attempt in range(max_retries):
                    async with semaphore:
                        start = chunk_number * chunk_size
                        end = min(start + chunk_size, file_size) - 1
                        
                        try:
                            chunk_data = await self.get_object_range(key, start, end, bucket)
                            if chunk_data is None:
                                raise Exception("Failed to download chunk")
                            
                            async with aiofiles.open(file_path, "rb+") as f2:
                                await f2.seek(start)
                                await f2.write(chunk_data)
                            
                            pbar.update(len(chunk_data))
                            downloaded_chunks[chunk_number] = {
                                "start": start,
                                "end": end + 1,
                                "size": len(chunk_data),
                            }
                            return chunk_number
                            
                        except Exception as e:
                            tplr.logger.error(f"Error downloading chunk {chunk_number} (attempt {attempt + 1}): {e}")
                            if attempt == max_retries - 1:
                                raise
                            await asyncio.sleep(1 * (attempt + 1))
            
            try:
                tasks = [asyncio.create_task(download_chunk(i)) for i in range(total_chunks)]
                await asyncio.gather(*tasks)
                
                if len(downloaded_chunks) != total_chunks:
                    missing_chunks = set(range(total_chunks)) - set(downloaded_chunks.keys())
                    raise Exception(f"Missing chunks: {missing_chunks}")
                
                downloaded_size = sum(chunk["size"] for chunk in downloaded_chunks.values())
                if downloaded_size != file_size:
                    raise Exception(f"Downloaded size ({downloaded_size}) != expected size ({file_size})")
                
                return True
                
            finally:
                pbar.close()
                
        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
            return False
        except Exception as e:
            tplr.logger.error(f"Error in multipart_download for {key}: {e}")
            return False

    async def close_all_clients(self) -> None:
        """Closes all S3 clients that have been created and stored"""
        for key, client in list(self._s3_clients.items()):
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                tplr.logger.warning(f"Error closing s3_client {key}: {e}")
        self._s3_clients.clear() 