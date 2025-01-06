import os
import time
import torch
import asyncio
import aiofiles
import tempfile
import bittensor as bt
from typing import List, Dict, Optional, Tuple
from types import SimpleNamespace
from aiobotocore.session import get_session
from . import __version__
from .config import client_config, BUCKET_SECRETS
from .chain import ChainManager
from .schemas import Bucket

import tplr as tplr
import botocore

# Constants
CF_REGION_NAME: str = "enam"
LOCAL_TMP_DIR = "/tmp/local_store"


class Comms(ChainManager):
    def __init__(
        self,
        wallet: "bt.wallet",
        save_location: str = "/tmp",
        key_prefix: str = "model",
        **kwargs,
    ):
        self.wallet = wallet
        # Get the bucket directly
        self.bucket = self.get_own_bucket()
        # Now initialize ChainManager with the bucket
        super().__init__(
            config=kwargs.get("config"),
            netuid=kwargs.get("netuid"),
            metagraph=kwargs.get("metagraph"),
            hparams=kwargs.get("hparams"),
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

    async def s3_put_object(self, key: str, data: bytes):
        """Upload object to S3."""
        session = get_session()
        async with session.create_client(
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

    async def s3_get_object(self, key: str, timeout: int) -> Optional[dict]:
        """Download object from S3."""
        session = get_session()
        async with session.create_client(
            "s3",
            endpoint_url=self.get_base_url(BUCKET_SECRETS["account_id"]),
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
            aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
        ) as s3_client:
            try:
                # Check if file exists first
                try:
                    await s3_client.head_object(
                        Bucket=BUCKET_SECRETS["bucket_name"], Key=key
                    )
                except (
                    botocore.exceptions.ClientError,
                    botocore.exceptions.BotoCoreError,
                ) as e:
                    tplr.logger.debug(f"Object not found or access denied: {e}")
                    return None

                response = await asyncio.wait_for(
                    s3_client.get_object(Bucket=BUCKET_SECRETS["bucket_name"], Key=key),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                tplr.logger.debug(f"Timeout occurred while downloading {key}.")
                return None
            except Exception as e:
                tplr.logger.debug(f"An error occurred during GET {key}: {e}")
                return None

            # Save to a temporary file and load
            with tempfile.NamedTemporaryFile(delete=True, suffix=".pt") as temp_file:
                temp_file_path = temp_file.name
                async with aiofiles.open(temp_file_path, "wb") as outfile:
                    while True:
                        chunk = await response["Body"].read(1 * 1024 * 1024)
                        if not chunk:
                            break
                        await outfile.write(chunk)

                # Load the object
                try:
                    with open(temp_file_path, "rb") as f:
                        state_dict = torch.load(f, weights_only=True)
                    return state_dict
                except Exception as e:
                    tplr.logger.debug(f"Error loading state_dict from {key}: {e}")
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
        tplr.logger.debug(f"PUT {uid}/{window}/{key} -->")

        # Create versioned filename
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"

        # Create a temporary file path
        temp_file_path = tempfile.mktemp(suffix=".pt")

        try:
            # Prepare the data to be saved
            if key == "checkpoint":
                save_data = (
                    state_dict  # state_dict already contains all checkpoint data
                )
            else:
                save_data = {
                    "state_dict": state_dict,
                    "global_step": global_step,
                }

            # Save the combined data
            torch.save(save_data, temp_file_path)

            if local:
                # Local storage logic remains unchanged
                await self.cleanup_local_data(
                    uid=uid, current_window=window, stale_retention=stale_retention
                )
                local_dir = os.path.join(LOCAL_TMP_DIR, str(uid), str(window))
                os.makedirs(local_dir, exist_ok=True)
                final_path = os.path.join(local_dir, filename)
                os.replace(temp_file_path, final_path)
            else:
                # Cleanup old S3 data
                await self.cleanup_s3_data(
                    uid=uid, current_window=window, stale_retention=stale_retention
                )

                # Check file size
                file_size = os.path.getsize(temp_file_path)
                object_key = filename

                if file_size > 5 * 1024 * 1024 * 1024:  # 5GB
                    # Use multipart upload for large files
                    success = await self.upload_large_file(temp_file_path, object_key)
                    if not success:
                        raise Exception("Large file upload failed")
                else:
                    # Use regular upload for smaller files
                    async with aiofiles.open(temp_file_path, "rb") as f:
                        data = await f.read()
                        await self.s3_put_object(object_key, data)

            # Remove temporary file after successful upload
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        except Exception as e:
            tplr.logger.debug(f"PUT error {uid}/{window}/{key}: {e}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        tplr.logger.debug(f"PUT {uid}/{window}/{key} <--")

    async def get(
        self,
        uid: str,
        window: int,
        key: str,
        timeout: int = 30,
        local: bool = True,
        stale_retention: int = 10,
    ) -> Optional[Tuple[dict, int]]:
        """GET operation: Retrieve state_dict and global_step."""
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        # full_key = f"{uid}/{window}/{filename}"
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
            else:
                # Cleanup old S3 data
                await self.cleanup_s3_data(
                    uid=uid, current_window=window, stale_retention=stale_retention
                )

                # Get the peer's bucket from commitments
                peer_bucket = self.commitments.get(int(uid))
                tplr.logger.debug(f"getting {key} from peer : {peer_bucket}")
                if not peer_bucket:
                    tplr.logger.debug(f"No bucket found for UID {uid}")
                    return None

                async with self.session.create_client(
                    "s3",
                    endpoint_url=self.get_base_url(peer_bucket.account_id),
                    region_name=CF_REGION_NAME,
                    config=client_config,
                    aws_access_key_id=peer_bucket.access_key_id,
                    aws_secret_access_key=peer_bucket.secret_access_key,
                ) as s3_client:
                    try:
                        # Check if file exists first
                        await s3_client.head_object(
                            Bucket=peer_bucket.name, Key=filename
                        )
                    except (
                        botocore.exceptions.ClientError,
                        botocore.exceptions.BotoCoreError,
                    ) as e:
                        error_code = e.response["Error"]["Code"]
                        if error_code == "404":
                            tplr.logger.debug(
                                f"Data not found for uid {uid} at window {window}. Skipping."
                            )
                            return None
                        else:
                            raise  # Re-raise if it's a different exception

                    # Proceed to get the object if it exists
                    response = await asyncio.wait_for(
                        s3_client.get_object(Bucket=peer_bucket.name, Key=filename),
                        timeout=timeout,
                    )

                    # Save to a temporary file and load
                    with tempfile.NamedTemporaryFile(
                        delete=True, suffix=".pt"
                    ) as temp_file:
                        temp_file_path = temp_file.name
                        async with aiofiles.open(temp_file_path, "wb") as outfile:
                            while True:
                                chunk = await response["Body"].read(1 * 1024 * 1024)
                                if not chunk:
                                    break
                                await outfile.write(chunk)

                        # Load the object
                        try:
                            with open(temp_file_path, "rb") as f:
                                loaded_data = torch.load(f, weights_only=True)
                            if key == "checkpoint":
                                return loaded_data, None
                            state_dict = loaded_data.get("state_dict")
                            global_step = loaded_data.get("global_step", 0)
                            return state_dict, global_step
                        except Exception as e:
                            tplr.logger.debug(
                                f"Error loading data from {filename}: {e}"
                            )
                            return None

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
        Uploads a large file to R2 using multipart upload.

        Args:
            file_path (str): Path to the local file to upload
            filename (str): Destination filename in R2

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use 16MB chunks for multipart upload
            chunk_size = 16 * 1024 * 1024

            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(self.bucket.account_id),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=self.bucket.access_key_id,
                aws_secret_access_key=self.bucket.secret_access_key,
            ) as s3_client:
                # Initialize multipart upload
                response = await s3_client.create_multipart_upload(
                    Bucket=self.bucket.name, Key=filename
                )
                upload_id = response["UploadId"]

                # Upload parts
                parts = []
                part_number = 1

                async with aiofiles.open(file_path, "rb") as f:
                    while True:
                        data = await f.read(chunk_size)
                        if not data:
                            break

                        response = await s3_client.upload_part(
                            Bucket=self.bucket.name,
                            Key=filename,
                            PartNumber=part_number,
                            UploadId=upload_id,
                            Body=data,
                        )

                        parts.append(
                            {"PartNumber": part_number, "ETag": response["ETag"]}
                        )
                        part_number += 1

                # Complete multipart upload
                await s3_client.complete_multipart_upload(
                    Bucket=self.bucket.name,
                    Key=filename,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )

            tplr.logger.debug(f"Successfully uploaded large file {filename}")
            return True

        except Exception as e:
            tplr.logger.error(f"Error uploading large file {filename}: {e}")
            return False

    async def download_large_file(self, filename: str, destination_path: str) -> bool:
        """
        Downloads a large file from R2 using multipart download.

        Args:
            filename (str): File to download from R2
            destination_path (str): Local path to save the file

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

                # Use 16MB chunks for multipart download
                chunk_size = 16 * 1024 * 1024
                total_parts = (file_size + chunk_size - 1) // chunk_size

                async with aiofiles.open(destination_path, "wb") as f:
                    for part in range(total_parts):
                        start = part * chunk_size
                        end = min(start + chunk_size, file_size)

                        response = await s3_client.get_object(
                            Bucket=self.bucket.name,
                            Key=filename,
                            Range=f"bytes={start}-{end-1}",
                        )

                        chunk = await response["Body"].read()
                        await f.write(chunk)

                tplr.logger.debug(f"Successfully downloaded large file {filename}")
                return True

        except Exception as e:
            tplr.logger.error(f"Error downloading large file {filename}: {e}")
            return False

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
