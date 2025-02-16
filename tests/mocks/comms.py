"""Mock communications components"""
from unittest.mock import AsyncMock, MagicMock
import torch
from types import SimpleNamespace
from .base import BaseMock

class MockComms(BaseMock):
    """Mock communications with async operations"""
    def __init__(self):
        super().__init__()
        self.bucket = SimpleNamespace(
            name="test-bucket",
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret"
        )
        self.temp_dir = "/tmp/test_comms"
        self.save_location = "/tmp/test_save"
        self.lock = AsyncMock()
        self.active_peers = set()
        self.session = MagicMock()
        self.evaluated_uids = set()  # Track evaluated UIDs
        self.moving_avg_scores = torch.zeros(100)  # Match validator size
        
    async def get_with_retry(self, *args, **kwargs):
        """Mock get operation with retry"""
        return {
            "layer1.weightsidxs": torch.tensor([0, 1]),
            "layer1.weightsvals": torch.tensor([0.1, 0.2])
        }, 1
        
    async def gather(self, *args, **kwargs):
        """Mock gather operation"""
        return SimpleNamespace(
            state_dict=SimpleNamespace(
                **{
                    "layer1.weightidxs": [torch.arange(5)],
                    "layer1.weightvals": [torch.ones(5) * 0.1],
                    "layer1.weightshape": [(10, 10)],
                    "layer1.weighttotalk": [50],
                    "layer1.biasidxs": [torch.arange(2)],
                    "layer1.biasvals": [torch.ones(2) * 0.1],
                    "layer1.biasshape": [(10,)],
                    "layer1.biastotalk": [5]
                }
            ),
            uids=["1"],
            global_steps=[1]
        ) 