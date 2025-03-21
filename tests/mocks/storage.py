"""Mock storage manager for managing model and gradient checkpoints"""

from .base import BaseMock
import asyncio
from unittest.mock import AsyncMock


class MockStorageManager(BaseMock):
    """Mock implementation of StorageManager for testing"""

    def __init__(self, temp_dir=None, save_location=None, wallet=None):
        super().__init__()
        self.temp_dir = temp_dir or "/tmp/mock_temp"
        self.save_location = save_location or "/tmp/mock_save"
        self.wallet = wallet
        self.lock = asyncio.Lock()

        # Mock methods
        self.store_local = AsyncMock(return_value=True)
        self.store_remote = AsyncMock(return_value=True)
        self.store_bytes = AsyncMock(return_value=True)
        self.get_local = AsyncMock(return_value=None)
        self.get_remote = AsyncMock(return_value=None)
        self.get_bytes = AsyncMock(return_value=None)
        self.load_latest_checkpoint = AsyncMock(return_value=None)
        self.load_remote_checkpoint = AsyncMock(return_value=None)
        self.cleanup_local_data = AsyncMock()
        self._cleanup_temp_file = AsyncMock()
        self.s3_put_object = AsyncMock(return_value=True)
        self.s3_get_object = AsyncMock(return_value=None)
        self.s3_head_object = AsyncMock(return_value=False)

    async def s3_put_object(self, key, data, bucket=None, content_type=None):
        """Mock S3 upload operation"""
        return True

    async def s3_get_object(
        self, key, bucket=None, timeout=10, time_min=None, time_max=None
    ):
        """Mock S3 download operation with time window support"""
        # Return success status and empty dict to simulate download
        import torch

        return {"success": True, "data": {"test": torch.tensor([1.0])}}, 1
