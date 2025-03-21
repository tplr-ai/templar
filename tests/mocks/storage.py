"""Mock storage manager for managing model and gradient checkpoints"""

from .base import BaseMock


class MockStorageManager(BaseMock):
    """Mock storage manager that simulates S3 operations."""

    def __init__(self):
        super().__init__()
        self.temp_dir = "/tmp/mock_storage"

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
