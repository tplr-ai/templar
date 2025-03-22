import os
import time
import asyncio
import aiohttp
import torch
from typing import Optional
from datetime import timezone

# Fix circular import by using direct imports
from tplr.logging import logger  # Import logger directly from logging module
from tplr import __version__  # This can still be imported directly
from aiobotocore.session import get_session


class StorageManager:
    """Handles all storage operations, both local and remote"""

    def __init__(self, temp_dir, save_location, wallet=None):
        self.temp_dir = temp_dir
        self.save_location = save_location
        self.wallet = wallet
        self.session = get_session()
        self.lock = asyncio.Lock()

    async def store_local(self, state_dict, uid, window, key):
        """Store data in local filesystem"""
        path = os.path.join(self.save_location, "local_store", uid, str(window))
        os.makedirs(path, exist_ok=True)

        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        file_path = os.path.join(path, filename)

        try:
            torch.save(state_dict, file_path)
            await self.cleanup_local_data()
            return True
        except Exception as e:
            logger.error(f"Error storing local data: {e}")
            return False

    async def store_remote(self, state_dict, uid, window, key, bucket, global_step=0):
        """Store data in remote bucket"""
        if bucket is None:
            logger.error("Cannot store remotely: no bucket provided")
            return False

        # Save to temp file first
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        temp_path = os.path.join(self.temp_dir, filename)

        try:
            torch.save(state_dict, temp_path)

            # Upload to S3
            success = await self.s3_put_object(
                key=filename, file_path=temp_path, bucket=bucket
            )

            # Clean up temp file
            asyncio.create_task(self._cleanup_temp_file(temp_path))

            return success
        except Exception as e:
            logger.error(f"Error storing remote data: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    async def store_bytes(self, data, key, bucket):
        """Store raw bytes in remote bucket"""
        if bucket is None:
            logger.error("Cannot store bytes: no bucket provided")
            return False

        try:
            # Create a temp file with the data
            temp_path = os.path.join(self.temp_dir, key)
            with open(temp_path, "wb") as f:
                f.write(data)

            # Upload to S3
            success = await self.s3_put_object(
                key=key, file_path=temp_path, bucket=bucket
            )

            # Clean up temp file
            asyncio.create_task(self._cleanup_temp_file(temp_path))

            return success
        except Exception as e:
            logger.error(f"Error storing bytes: {e}")
            return False

    async def get_local(self, uid, window, key):
        """Get data from local filesystem"""
        path = os.path.join(self.save_location, "local_store", uid, str(window))
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        file_path = os.path.join(path, filename)

        if os.path.exists(file_path):
            try:
                data = torch.load(file_path)
                await self.cleanup_local_data()
                return data
            except Exception as e:
                logger.error(f"Error loading local data: {e}")

        return None

    async def get_remote(self, uid, window, key, bucket, timeout=30):
        """Get data from remote bucket"""
        if bucket is None:
            logger.error("Cannot get remote data: no bucket provided")
            return None

        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        temp_path = os.path.join(self.temp_dir, filename)

        try:
            # Download from S3
            success = await self.s3_get_object(
                key=filename, bucket=bucket, file_path=temp_path, timeout=timeout
            )

            # If s3_get_object returns a status marker, propagate it.
            if isinstance(success, str) and success in ["TOO_EARLY", "TOO_LATE"]:
                return success
            elif success is True:
                data = torch.load(temp_path)
                # Clean up temp file asynchronously.
                asyncio.create_task(self._cleanup_temp_file(temp_path))
                return data
        except Exception as e:
            logger.error(f"Error getting remote data: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return None

    async def get_bytes(self, key, bucket, timeout=30):
        """Get raw bytes from remote bucket"""
        if bucket is None:
            logger.error("Cannot get bytes: no bucket provided")
            return None

        try:
            # Use direct S3 get operation
            return await self.s3_get_object(key=key, bucket=bucket, timeout=timeout)
        except Exception as e:
            logger.error(f"Error getting bytes: {e}")
            return None

    async def load_latest_checkpoint(self, uid):
        """Load the latest checkpoint from local storage"""
        checkpoints_dir = os.path.join(self.save_location, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            return None

        # Find all checkpoint files
        checkpoint_files = []
        for filename in os.listdir(checkpoints_dir):
            if filename.startswith("checkpoint-") and filename.endswith(".pt"):
                file_path = os.path.join(checkpoints_dir, filename)
                checkpoint_files.append((file_path, os.path.getmtime(file_path)))

        if not checkpoint_files:
            return None

        # Get the most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: x[1])[0]

        try:
            return torch.load(latest_checkpoint)
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None

    async def load_remote_checkpoint(self, uid, device, bucket):
        """Load the latest checkpoint from remote storage"""
        if bucket is None:
            return None

        # First get the checkpoint index
        try:
            checkpoint_index = await self.s3_get_object(
                key=f"checkpoint_index_{uid}.json", bucket=bucket
            )

            if checkpoint_index:
                if isinstance(checkpoint_index, bytes):
                    import json

                    checkpoint_index = json.loads(checkpoint_index.decode("utf-8"))

                latest_checkpoint = checkpoint_index.get("latest_checkpoint")
                if latest_checkpoint:
                    # Download the checkpoint file
                    temp_path = os.path.join(
                        self.temp_dir, f"remote_checkpoint_{uid}.pt"
                    )
                    success = await self.s3_get_object(
                        key=latest_checkpoint, bucket=bucket, file_path=temp_path
                    )

                    if success:
                        checkpoint = torch.load(temp_path, map_location=device)
                        # Clean up temp file
                        asyncio.create_task(self._cleanup_temp_file(temp_path))
                        return checkpoint
        except Exception as e:
            logger.error(f"Error loading remote checkpoint: {e}")

        return None

    async def cleanup_local_data(self, max_age_days=7):
        """Remove old local data files"""
        base_path = os.path.join(self.save_location, "local_store")
        if not os.path.exists(base_path):
            return

        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60

        try:
            for uid_dir in os.listdir(base_path):
                uid_path = os.path.join(base_path, uid_dir)
                if not os.path.isdir(uid_path):
                    continue

                for window_dir in os.listdir(uid_path):
                    window_path = os.path.join(uid_path, window_dir)
                    if not os.path.isdir(window_path):
                        continue

                    for filename in os.listdir(window_path):
                        file_path = os.path.join(window_path, filename)
                        file_age = current_time - os.path.getmtime(file_path)

                        if file_age > max_age_seconds:
                            os.remove(file_path)

                    # Remove empty directories
                    if not os.listdir(window_path):
                        os.rmdir(window_path)

                if not os.listdir(uid_path):
                    os.rmdir(uid_path)
        except Exception as e:
            logger.error(f"Error cleaning up local data: {e}")

    async def _cleanup_temp_file(self, file_path):
        """Safely remove a temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error removing temp file {file_path}: {e}")

    # S3 operations
    async def s3_put_object(self, key, file_path, bucket):
        """Upload a file to S3"""
        import boto3
        from botocore.client import Config

        s3_config = Config(
            region_name="auto", retries={"max_attempts": 3, "mode": "standard"}
        )

        s3_client = boto3.client(
            "s3",
            endpoint_url=f"https://{bucket.account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=bucket.access_key_id,
            aws_secret_access_key=bucket.secret_access_key,
            config=s3_config,
        )

        try:
            s3_client.upload_file(Filename=file_path, Bucket=bucket.name, Key=key)
            return True
        except Exception as e:
            logger.error(f"S3 put error: {e}")
            return False

    async def s3_get_object(
        self,
        key: str,
        bucket,
        file_path: Optional[str] = None,
        timeout=10,
        time_min=None,
        time_max=None,
    ):
        """Download an object from S3"""
        if bucket is None:
            logger.error("Cannot get remote data: no bucket provided")
            return None

        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)

        # Normalize timezone information BEFORE comparisons
        if time_min is not None and not time_min.tzinfo:
            time_min = time_min.replace(tzinfo=timezone.utc)
        if time_max is not None and not time_max.tzinfo:
            time_max = time_max.replace(tzinfo=timezone.utc)

        # Construct URL for S3 GET
        url = (
            f"https://{bucket.account_id}.r2.cloudflarestorage.com/{bucket.name}/{key}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={}, timeout=timeout) as response:
                    if response.status == 200:
                        loaded_data = await response.read()
                        import json

                        json_data = None
                        try:
                            if isinstance(loaded_data, bytes):
                                decoded = loaded_data.decode("utf-8")
                                json_data = json.loads(decoded)
                            elif isinstance(loaded_data, str):
                                json_data = json.loads(loaded_data)
                            elif isinstance(loaded_data, dict):
                                json_data = loaded_data
                        except Exception:
                            json_data = None

                        if json_data is not None:
                            status = json_data.get("__status")
                            if status in ["TOO_EARLY", "TOO_LATE"]:
                                # Instead of returning None, return status so that miners/validators can use it.
                                return status

                        if file_path is not None:
                            with open(file_path, "wb") as f:
                                if isinstance(loaded_data, bytes):
                                    f.write(loaded_data)
                                else:
                                    f.write(bytes(str(loaded_data), "utf-8"))
                            return True
                        return loaded_data
                    else:
                        logger.debug(
                            f"S3 get error: {response.status} - {await response.text()}"
                        )
                        return None
        except Exception as e:
            logger.error(f"Error in s3_get_object for {key}: {e}")
            return None
