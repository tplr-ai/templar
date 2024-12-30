# The MIT License (MIT)
# © 2024 templar.tech

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
# fmt: off

# Global imports
import os
import re
import time
import yaml
import torch
import asyncio
import aiofiles
import numpy as np
import bittensor as bt
from typing import List, Dict, Optional, Tuple
from aiobotocore.session import get_session

# Local imports
from . import __version__
from .schemas import Bucket
from .logging import logger
from .chain import ChainManager
from .config import client_config, BUCKET_SECRETS

CF_REGION_NAME: str = "enam"


def get_base_url(account_id):
    """Constructs the base URL for the R2 storage endpoint."""
    return f"https://{account_id}.r2.cloudflarestorage.com"


class Comms(ChainManager):
    def __init__(
        self,
        wallet: "bt.wallet",
        save_location: str = "/tmp",
        key_prefix: str = "slice",
        **kwargs
    ):
        self.wallet = wallet
        self.bucket = self.get_own_bucket()
        super().__init__(
            config=kwargs.get('config'),
            netuid=kwargs.get('netuid'),
            metagraph=kwargs.get('metagraph'),
            hparams=kwargs.get('hparams'),
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
        # Load bucket secrets
        self.bucket_secrets = BUCKET_SECRETS

    def get_own_bucket(self) -> Bucket:
        """Parses the credentials from .env.yaml to create a Bucket object."""
        env_file = ".env.yaml"
        if not os.path.isfile(env_file):
            logger.error(f"The {env_file} file was not found.")
            raise FileNotFoundError(f"The {env_file} file was not found.")

        try:
            with open(env_file, "r") as file:
                credentials = yaml.safe_load(file)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing {env_file}: {e}")
            raise e

        try:
            account_id = credentials["account_id"]
            read_access_key_id = credentials["read"]["access_key_id"]
            read_secret_access_key = credentials["read"]["secret_access_key"]

            # Create a Bucket object
            bucket = Bucket(
                name=account_id,
                account_id=account_id,
                access_key_id=read_access_key_id,
                secret_access_key=read_secret_access_key,
            )
            logger.debug(f"Parsed bucket from {env_file}: {bucket}")
            return bucket
        except KeyError as e:
            logger.error(f"Missing key in {env_file}: {e}")
            raise e

    async def put(
        self,
        state_dict_or_path,
        uid: str,
        window_or_block: int,
        key: Optional[str] = None,
    ):
        """
        Uploads data to the R2 bucket. Handles both small state_dicts and large checkpoint files.

        Args:
            state_dict_or_path (dict or str): The state dictionary to upload or the path to the checkpoint file.
            uid (str): Unique identifier for the upload (e.g., hotkey or user ID).
            window_or_block (int): The window number or block number.
            key (str, optional): Custom key for the filename. Defaults to self.key_prefix.
        """
        key = key or self.key_prefix
        hotkey = self.wallet.hotkey.ss58_address

        if isinstance(state_dict_or_path, dict):
            # Handle state_dict upload
            filename = f"{key}-{window_or_block}-{hotkey}-v{__version__}.pt"
            temp_file_path = os.path.join(self.save_location, filename)

            # Ensure the save directory exists
            os.makedirs(self.save_location, exist_ok=True)

            try:
                # Save the state_dict to a temporary file
                torch.save(state_dict_or_path, temp_file_path)
                logger.debug(f"Temporary file saved at {temp_file_path}")
            except Exception as e:
                logger.error(f"Error saving temporary file: {e}")
                raise

            file_path = temp_file_path
        elif isinstance(state_dict_or_path, str):
            # Handle checkpoint file upload
            file_path = state_dict_or_path
            filename = os.path.basename(file_path)
        else:
            raise ValueError("state_dict_or_path must be a state_dict or a file path.")

        # Determine if multipart upload is needed based on file size
        file_size = os.path.getsize(file_path)
        use_multipart = file_size > 5 * 1024 * 1024 * 1024  # 5 GB threshold

        # Upload the file to R2 bucket
        try:
            async with self.session.create_client(
                "s3",
                endpoint_url=get_base_url(BUCKET_SECRETS["account_id"]),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
                aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
            ) as s3_client:
                if use_multipart:
                    await self._multipart_upload(s3_client, filename, file_path)
                else:
                    async with aiofiles.open(file_path, "rb") as f:
                        data = await f.read()
                        await s3_client.put_object(
                            Bucket=self.bucket.name, Key=filename, Body=data
                        )
                    logger.debug(f"Successfully uploaded {filename} to R2 bucket.")
        except Exception as e:
            logger.error(f"Failed to upload {filename} to R2 bucket: {e}")
            raise
        finally:
            # Clean up the temporary file if it exists and was created
            if isinstance(state_dict_or_path, dict):
                logger.debug(f"Attempting to delete temporary file at {temp_file_path}")
                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        logger.debug(f"Deleted temporary file at {temp_file_path}")
                    else:
                        logger.debug(f"Temporary file does not exist at {temp_file_path}")
                except Exception as e:
                    logger.error(f"Error during cleanup of temporary file: {e}")

    async def _multipart_upload(self, s3_client, filename, file_path):
        """Handles multipart upload for large files."""
        bucket = BUCKET_SECRETS["bucket_name"].split("/")[-1]
        chunk_size = 16 * 1024 * 1024  # 16MB chunks
        max_concurrent_uploads = 10
        max_retries = 3
        retry_delay = 5

        # Initialize multipart upload
        response = await s3_client.create_multipart_upload(
            Bucket=bucket,
            Key=filename,
            CacheControl="no-cache, no-store, must-revalidate",
        )
        upload_id = response["UploadId"]
        logger.info(f"Initiated multipart upload with ID: {upload_id}")

        try:
            total_size = os.path.getsize(file_path)
            total_parts = (total_size + chunk_size - 1) // chunk_size
            parts = {}
            semaphore = asyncio.Semaphore(max_concurrent_uploads)
            upload_tasks = []

            async def upload_part(part_number: int, offset: int):
                """Upload a single part with retries."""
                for attempt in range(max_retries):
                    try:
                        async with semaphore:
                            async with aiofiles.open(file_path, "rb") as f:
                                await f.seek(offset)
                                chunk = await f.read(min(chunk_size, total_size - offset))

                                response = await s3_client.upload_part(
                                    Bucket=bucket,
                                    Key=filename,
                                    PartNumber=part_number,
                                    UploadId=upload_id,
                                    Body=chunk,
                                )

                                return {
                                    "PartNumber": part_number,
                                    "ETag": response["ETag"],
                                }
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Retry {attempt + 1}/{max_retries} for part {part_number}: {str(e)}")
                            await asyncio.sleep(retry_delay)
                        else:
                            raise

            # Create upload tasks for all parts
            for part_number in range(1, total_parts + 1):
                offset = (part_number - 1) * chunk_size
                task = asyncio.create_task(upload_part(part_number, offset))
                upload_tasks.append(task)

            # Wait for all uploads and collect results
            completed_parts = await asyncio.gather(*upload_tasks)
            parts = [part for part in completed_parts if part is not None]
            parts.sort(key=lambda x: x["PartNumber"])

            # Complete multipart upload
            await s3_client.complete_multipart_upload(
                Bucket=bucket,
                Key=filename,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
            logger.info(f"Successfully uploaded checkpoint {filename}")
        except Exception as e:
            logger.error(f"Error during multipart upload: {str(e)}")
            await s3_client.abort_multipart_upload(
                Bucket=bucket, Key=filename, UploadId=upload_id
            )
            logger.info(f"Aborted multipart upload {upload_id}")
            raise

    async def get(
        self,
        uid: str,
        window: int,
        key: Optional[str] = None,
        timeout: int = 30,
        local: bool = False
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Downloads data from the R2 bucket. Handles both model state dicts and large checkpoint files.
        
        Args:
            uid (str): Unique identifier for the download.
            window (int): The window number for synchronization.
            key (str, optional): Custom key for the filename.
            timeout (int): Timeout in seconds for the download operation.
            local (bool): If True, keeps the downloaded file on disk.

        Returns:
            Optional[Dict[str, torch.Tensor]]: The downloaded state dictionary.
        """
        key = key or self.key_prefix
        hotkey = self.get_hotkey(int(uid))
        if hotkey is None:
            logger.error(f"No hotkey found for uid {uid}")
            return None

        filename = f"{key}-{window}-{hotkey}-v{__version__}.pt"
        temp_file_path = os.path.join(self.save_location, filename)

        bucket = self.get_bucket(int(uid))
        if bucket is None:
            logger.debug(f"Bucket for uid {uid} not found. Skipping...")
            return None

        try:
            async with self.session.create_client(
                "s3",
                endpoint_url=get_base_url(bucket.account_id),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=bucket.access_key_id,
                aws_secret_access_key=bucket.secret_access_key,
            ) as s3_client:
                # Check file size first
                response = await s3_client.head_object(
                    Bucket=bucket.name,
                    Key=filename
                )
                file_size = response['ContentLength']
                
                # Use multipart download for files larger than 100MB
                if file_size > 100 * 1024 * 1024:
                    success = await self._download_large_file(s3_client, filename, temp_file_path)
                    if not success:
                        return None
                else:
                    async def download():
                        response = await s3_client.get_object(
                            Bucket=bucket.name,
                            Key=filename
                        )
                        async with aiofiles.open(temp_file_path, "wb") as f:
                            await f.write(await response["Body"].read())

                    await asyncio.wait_for(download(), timeout=timeout)

                # Load the state_dict
                state_dict = torch.load(temp_file_path, map_location="cpu", weights_only=True)
                logger.debug(f"Successfully downloaded {filename} from R2 bucket.")
                
                # Clean up unless local=True
                if not local:
                    os.remove(temp_file_path)
                    
                return state_dict

        except asyncio.TimeoutError:
            logger.error(f"Timeout while downloading {filename} from R2 bucket.")
        except Exception as e:
            logger.error(f"Failed to download {filename} from R2 bucket: {e}")
        finally:
            if not local and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        return None

    async def get_with_retry(
        self,
        uid: str,
        window: int,
        key: Optional[str] = None,
        timeout: int = 30,
        retry_interval: float = 0.1,
    ):
        """
        Attempts to download data from the R2 bucket, retrying until success or timeout.

        Args:
            uid (str): Unique identifier for the download.
            window (int): The window number for synchronization.
            key (str, optional): Custom key for the filename.
            timeout (int): Total timeout duration for retries.
            retry_interval (float): Time to wait between retries.

        Returns:
            dict: The state dictionary downloaded from the bucket.
        """
        start_time = time.time()
        while True:
            state_dict = await self.get(uid, window, key, timeout)
            if state_dict is not None:
                return state_dict
            if time.time() - start_time > timeout:
                logger.error(f"Exceeded timeout while downloading data for UID {uid}.")
                return None
            await asyncio.sleep(retry_interval)

    async def gather(
        self,
        state_dict: Dict[str, torch.Tensor],
        my_uid: str,
        uids: List[str],
        window: int,
        key: Optional[str] = None,
        timeout: int = 30,
        device: str = "cpu",
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Gathers slices from multiple peers and assembles them for aggregation.

        Args:
            state_dict (Dict[str, torch.Tensor]): Local state dictionary.
            my_uid (str): This node's unique identifier.
            uids (List[str]): List of peer UIDs to gather data from.
            window (int): The window number for synchronization.
            key (str, optional): Custom key for filenames.
            timeout (int): Timeout for gathering data from each peer.
            device (str): Device to map tensors onto.

        Returns:
            Dict[str, List[torch.Tensor]]: Aggregated state dictionaries from all peers.
        """
        key = key or self.key_prefix
        # Put own state_dict to the bucket
        await self.put(state_dict, my_uid, window, key)

        time.sleep(5)

        # Gather state_dicts from peers
        gather_tasks = [
            self.get_with_retry(uid=uid, window=window, key=key, timeout=timeout)
            for uid in uids
        ]

        responses = await asyncio.gather(*gather_tasks)
        # Initialize the gather_result dictionary
        gather_result = {param_name: [] for param_name in state_dict.keys()}
        # Assemble the results
        for idx, peer_state in enumerate(responses):
            if peer_state is None:
                # Handle missing peer data, e.g., fill with zeros or skip
                for param_name in state_dict.keys():
                    gather_result[param_name].append(
                        torch.zeros_like(state_dict[param_name]).to(device)
                    )
            else:
                for param_name in state_dict.keys():
                    gather_result[param_name].append(peer_state[param_name].to(device))

        return gather_result

    def get_highest_stake_validator(self) -> Tuple[Optional[int], float]:
        """Returns the UID and stake of the neuron with the highest stake."""
        stakes = self.metagraph.S
        logger.info(stakes)
        
        # Convert numpy array to torch tensor if needed
        if isinstance(stakes, np.ndarray):
            stakes = torch.from_numpy(stakes)
        
        # Check if any stakes are non-zero
        if torch.all(stakes == 0):
            return None, 0.0
            
        highest_stake_uid = torch.argmax(stakes).item()
        stake = stakes[highest_stake_uid].item()

        # Validate the stake is actually non-zero
        if stake == 0:
            return None, 0.0

        return highest_stake_uid, stake

    async def get_latest_checkpoint(self) -> Optional[str]:
        """
        Attempts to get the latest checkpoint from the highest stake validator.
        Returns the checkpoint path if successful, None otherwise.
        """
        validator_uid, stake = self.get_highest_stake_validator()
        if stake == 0:
            logger.warning("No active validators found")
            return None

        # Get the current block and calculate window
        current_block = self.subtensor.block
        current_window = int(current_block / self.hparams.blocks_per_window)
        
        # Try last 5 windows to find most recent checkpoint
        for window in range(current_window, max(0, current_window - 5), -1):
            try:
                checkpoint = await self.get(
                    uid=str(validator_uid),
                    window=window,
                    key='checkpoint',
                    timeout=30  # Longer timeout for checkpoint downloads
                )
                if checkpoint is not None:
                    # Save checkpoint to disk
                    checkpoint_path = os.path.join(self.save_location, f'checkpoint-{window}.pt')
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Downloaded checkpoint from validator {validator_uid} at window {window}")
                    return checkpoint_path
            except Exception as e:
                logger.warning(f"Failed to get checkpoint from window {window}: {e}")
                continue
        
        return None

    async def _download_large_file(self, s3_client, filename: str, temp_file_path: str):
        """Handles downloading large files using multipart download."""
        try:
            # Get file size
            response = await s3_client.head_object(
                Bucket=self.bucket.name,
                Key=filename
            )
            file_size = response['ContentLength']
            
            # Use 16MB chunks for multipart download
            chunk_size = 16 * 1024 * 1024
            total_parts = (file_size + chunk_size - 1) // chunk_size
            
            async with aiofiles.open(temp_file_path, 'wb') as f:
                for part in range(total_parts):
                    start = part * chunk_size
                    end = min(start + chunk_size, file_size)
                    
                    response = await s3_client.get_object(
                        Bucket=self.bucket.name,
                        Key=filename,
                        Range=f'bytes={start}-{end-1}'
                    )
                    
                    chunk = await response['Body'].read()
                    await f.write(chunk)
                    
            logger.debug(f"Successfully downloaded large file {filename}")
            return True
        except Exception as e:
            logger.error(f"Error downloading large file {filename}: {e}")
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
                endpoint_url=get_base_url(self.bucket.account_id),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=self.bucket.access_key_id,
                aws_secret_access_key=self.bucket.secret_access_key,
            ) as s3_client:
                # List all checkpoint files
                paginator = s3_client.get_paginator("list_objects_v2")
                checkpoint_files = []
                
                async for page in paginator.paginate(
                    Bucket=self.bucket.name,
                    Prefix='checkpoint'
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
                        Delete={"Objects": [{"Key": obj["Key"]} for obj in to_delete]}
                    )
                    logger.info(f"Deleted {len(to_delete)} old checkpoints")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old checkpoints: {e}")


async def delete_old_version_files(bucket_name: str, current_version: str):
    """
    Deletes files from the S3 bucket that do not match the current version.

    Args:
        bucket_name (str): The name of the S3 bucket.
        current_version (str): The current version string.
    """
    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=get_base_url(BUCKET_SECRETS["account_id"]),
        region_name=CF_REGION_NAME,
        config=client_config,
        aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
        aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
    ) as s3_client:
        paginator = s3_client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket_name):
            to_delete = []
            for obj in page.get("Contents", []):
                filename = obj["Key"]
                # Check if the file version matches the current version
                match = re.match(r".+-v(.+)\.pt$", filename)
                if match:
                    file_version = match.group(1)
                    if file_version != current_version:
                        to_delete.append({"Key": filename})
                        logger.debug(f"Scheduled for deletion: {filename}")
            # Delete old versions in batches of 1000 (S3 limit for delete_objects)
            if to_delete:
                response = await s3_client.delete_objects(
                    Bucket=bucket_name, Delete={"Objects": to_delete}
                )
                deleted = response.get("Deleted", [])
                logger.info(
                    f"Deleted {len(deleted)} old version files from bucket {bucket_name}"
                )
