import os
import time
import asyncio
import torch
from typing import Optional
from datetime import timezone
import re

# Fix circular import by using direct imports
from tplr.logging import logger  # Import logger directly from logging module
from tplr import __version__  # This can still be imported directly
from aiobotocore.session import get_session


class StorageManager:
    """Handles all storage operations, both local and remote"""

    def __init__(self, temp_dir, save_location, wallet=None, s3_client=None):
        self.temp_dir = temp_dir
        self.save_location = save_location
        self.wallet = wallet
        self.session = get_session()
        self.lock = asyncio.Lock()
        # Allow injection of an S3 client (useful for tests).
        self.s3_client = s3_client
        # Persistent dictionary of S3 clients for explicit lifecycle management.
        self._s3_clients = {}

    async def _get_s3_client(self, bucket) -> any:
        """
        Returns a persistent S3 client for the given bucket credentials.
        """
        key = (bucket.access_key_id, bucket.secret_access_key, bucket.account_id)
        if key in self._s3_clients:
            return self._s3_clients[key]

        from botocore.client import Config

        s3_config = Config(
            region_name="auto",
            signature_version="s3v4",
            max_pool_connections=256,
            retries={"max_attempts": 3, "mode": "standard"},
        )
        client_params = dict(
            endpoint_url=f"https://{bucket.account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=bucket.access_key_id,
            aws_secret_access_key=bucket.secret_access_key,
            config=s3_config,
        )
        new_client = self.session.create_client("s3", **client_params)
        new_client = await new_client.__aenter__()
        self._s3_clients[key] = new_client
        return new_client

    async def close_all_s3_clients(self):
        """
        Closes all persistent S3 clients.
        """
        for key, client in list(self._s3_clients.items()):
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing s3_client {key}: {e}")
        self._s3_clients.clear()

    async def _purge_s3_client(self, bucket) -> None:
        """
        Purges the persistent S3 client for the given bucket.
        """
        key = (bucket.access_key_id, bucket.secret_access_key, bucket.account_id)
        if key in self._s3_clients:
            client = self._s3_clients.pop(key)
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error purging s3_client {key}: {e}")

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

    async def store_remote(self, state_dict, uid, window, key, bucket, global_step=None):
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
            success = await self.s3_put_object(key=filename, file_path=temp_path, bucket=bucket)

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

    async def get_local(
        self, uid, window, key, stale_retention=10, time_min=None, time_max=None
    ):
        """Get data from local filesystem"""
        path = os.path.join(self.save_location, "local_store", uid, str(window))
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        file_path = os.path.join(path, filename)

        if os.path.exists(file_path):
            try:
                data = torch.load(file_path)
                # Trigger cleanup for local gradients based on stale retention.
                await self.cleanup_local_gradients(uid, key, retention=stale_retention)
                return data
            except Exception as e:
                logger.error(f"Error loading local data: {e}")

        return None

    async def get_remote(
        self,
        uid,
        window,
        key,
        bucket,
        timeout=30,
        stale_retention=10,
        time_min=None, 
        time_max=None,
    ):
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
                # Trigger cleanup for remote gradients based on stale retention.
                await self.cleanup_remote_gradients(
                    uid, key, retention=stale_retention, bucket=bucket
                )
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

                latest_checkpoint = (
                    checkpoint_index["latest_checkpoint"]
                    if isinstance(checkpoint_index, dict)
                    else None
                )
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
    async def s3_head_object(self, key: str, bucket, timeout=10) -> bool:
        """Check if an object exists in S3 using a HEAD request asynchronously."""
        if bucket is None:
            logger.error("Cannot perform head: no bucket provided")
            return False

        try:
            from botocore.client import Config

            s3_config = Config(
                region_name="auto",
                signature_version="s3v4",
                max_pool_connections=256,
                retries={"max_attempts": 3, "mode": "standard"},
            )
            client_params = dict(
                endpoint_url=f"https://{bucket.account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=bucket.access_key_id,
                aws_secret_access_key=bucket.secret_access_key,
                config=s3_config,
            )

            async with self.session.create_client("s3", **client_params) as s3_client:
                await s3_client.head_object(Bucket=bucket.name, Key=key)
            return True
        except Exception as e:
            logger.error(f"Error in s3_head_object for key {key}: {e}")
            return False

    async def s3_put_object(self, key: str, file_path: str, bucket) -> bool:
        """Upload an object to S3 asynchronously using a persistent client."""
        if bucket is None:
            logger.error("Cannot put remote data: no bucket provided")
            return False

        try:
            from botocore.client import Config

            s3_config = Config(
                region_name="auto",
                signature_version="s3v4",
                max_pool_connections=256,
                retries={"max_attempts": 3, "mode": "standard"},
            )
            client_params = dict(
                endpoint_url=f"https://{bucket.account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=bucket.access_key_id,
                aws_secret_access_key=bucket.secret_access_key,
                config=s3_config,
            )
            s3_client = await self._get_s3_client(bucket)
            with open(file_path, "rb") as f:
                data = f.read()
            await s3_client.put_object(Bucket=bucket.name, Key=key, Body=data)
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
        """Download an object from S3 asynchronously using a persistent client."""
        if bucket is None:
            logger.error("Cannot get remote data: no bucket provided")
            return None

        os.makedirs(self.temp_dir, exist_ok=True)
        if time_min is not None and not time_min.tzinfo:
            time_min = time_min.replace(tzinfo=timezone.utc)
        if time_max is not None and not time_max.tzinfo:
            time_max = time_max.replace(tzinfo=timezone.utc)

        try:
            # Use persistent S3 client.
            s3_client = await self._get_s3_client(bucket)
            response = await s3_client.get_object(Bucket=bucket.name, Key=key)
            loaded_data = await response["Body"].read()

            # Check if the loaded data contains JSON status flags.
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
                    return status

            if file_path is not None:
                with open(file_path, "wb") as f:
                    f.write(loaded_data)
                return True

            return loaded_data

        except Exception as e:
            logger.error(f"Error in s3_get_object for {key}: {e}")
            return None

    async def cleanup_local_gradients(self, uid: str, key: str, retention: int = 10):
        """
        Cleanup local gradient files for a given uid and key,
        keeping only the latest 'retention' number of files.
        Expected filename structure: {key}-{window}-{uid}-v{__version__}.pt
        """
        base_dir = os.path.join(self.save_location, "local_store", uid)
        if not os.path.exists(base_dir):
            return

        gradient_files = []  # List of tuples: (window, file_path)
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.startswith(f"{key}-") and file.endswith(
                    f"-{uid}-v{__version__}.pt"
                ):
                    parts = file.split("-")
                    if len(parts) >= 4:
                        try:
                            window = int(parts[1])
                            gradient_files.append((window, os.path.join(root, file)))
                        except Exception as e:
                            logger.error(f"Error parsing window from file {file}: {e}")
        gradient_files.sort(key=lambda x: x[0])
        # Remove oldest files if exceeding retention count.
        if len(gradient_files) > retention:
            for window, file_path in gradient_files[: len(gradient_files) - retention]:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed stale local gradient file: {file_path}")
                except Exception as e:
                    logger.error(
                        f"Error removing stale local gradient file {file_path}: {e}"
                    )

    async def cleanup_remote_gradients(
        self, uid: str, key: str, retention: int = 10, bucket=None
    ):
        """Clean up stale remote gradient files asynchronously."""
        try:
            from botocore.client import Config

            s3_config = Config(
                region_name="auto", retries={"max_attempts": 3, "mode": "standard"}
            )
            client_params = dict(
                endpoint_url=f"https://{bucket.account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=bucket.access_key_id,
                aws_secret_access_key=bucket.secret_access_key,
                config=s3_config,
            )

            async with self.session.create_client("s3", **client_params) as s3_client:
                prefix = f"{key}-"
                response = await s3_client.list_objects_v2(
                    Bucket=bucket.name, Prefix=prefix
                )
                gradient_files = []
                pattern = re.compile(rf"^{key}-(\d+)-{uid}-v{__version__}\.pt$")
                if "Contents" in response:
                    for obj in response["Contents"]:
                        file_key = obj["Key"]
                        m = pattern.match(file_key)
                        if m:
                            try:
                                window = int(m.group(1))
                                gradient_files.append((window, file_key))
                            except Exception as e:
                                logger.error(
                                    f"Error parsing window from remote file {file_key}: {e}"
                                )
                gradient_files.sort(key=lambda x: x[0])
                if len(gradient_files) > retention:
                    for window, file_key in gradient_files[: len(gradient_files) - retention]:
                        try:
                            await s3_client.delete_object(Bucket=bucket.name, Key=file_key)
                            logger.info(
                                f"Removed stale remote gradient file: {file_key}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error removing stale remote gradient file {file_key}: {e}"
                            )
        except Exception as e:
            logger.error(f"Error during remote cleanup: {e}")

def create_storage_manager(temp_dir, save_location, wallet=None, use_mock: bool = False):
    """
    Factory function to create a StorageManager or MockStorageManager instance.
    In testing, set use_mock=True to inject the mock version.
    """
    if use_mock:
        from tests.mocks.storage import MockStorageManager
        return MockStorageManager(temp_dir=temp_dir, save_location=save_location, wallet=wallet)
    return StorageManager(temp_dir, save_location, wallet)
