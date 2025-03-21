"""Mock communications components"""

from unittest.mock import AsyncMock, MagicMock
import torch
from types import SimpleNamespace
from .base import BaseMock
from .storage import MockStorageManager
from .peer_manager import MockPeerManager
from .chain_sync import MockChainSync


class MockComms(BaseMock):
    """Mock communications with async operations"""

    def __init__(
        self,
        wallet=None,
        save_location="/tmp",
        key_prefix="model",
        config=None,
        netuid=1,
        metagraph=None,
        hparams=None,
        uid=0,
    ):
        super().__init__()

        # Initialize base components
        self.wallet = wallet or MagicMock()
        self.save_location = save_location
        self.key_prefix = key_prefix
        self.config = config or MagicMock()
        self.netuid = netuid
        self.metagraph = metagraph or MagicMock()
        self.hparams = hparams or MagicMock()
        self.uid = uid

        # Set up temp directory
        self.temp_dir = "/tmp/test_comms"

        # Create specialized component mocks
        self.storage = MockStorageManager()
        self.peer_manager = MockPeerManager()
        self.chain_sync = MockChainSync(
            config=self.config,
            netuid=self.netuid,
            metagraph=self.metagraph,
            hparams=self.hparams,
            wallet=self.wallet,
        )

        # Copy over some useful attributes from chain_sync for convenience
        self.active_peers = self.chain_sync.active_peers
        self.eval_peers = self.chain_sync.eval_peers
        self.inactive_peers = self.chain_sync.inactive_peers
        self.peers = self.chain_sync.peers
        self.commitments = self.chain_sync.commitments

        # Simulated data
        self.evaluated_uids = set()
        self.moving_avg_scores = torch.zeros(100)  # Match validator size

        # Add mock methods that aren't in the components
        self.commit_own_bucket = AsyncMock(return_value=True)
        self.fetch_commitments = AsyncMock(
            return_value={"1": "bucket-1", "2": "bucket-2"}
        )

    async def get_with_retry(self, key, uid, window, max_retries=3, **kwargs):
        """Mock get operation with retry that properly handles time window params"""
        # Optional time parameters are ignored in the mock.
        # (Previous code extracted time_min and time_max, but they are not used.)

        # Return mock gradient data
        return {
            "layer1.weightsidxs": torch.tensor([0, 1]),
            "layer1.weightsvals": torch.tensor([0.1, 0.2]),
        }, 1

    async def get(self, key, uid, window, **kwargs):
        """Get a state dict from storage."""
        # Optional time parameters are ignored in the mock.
        # (Previous code extracted time_min and time_max, but they are not used.)

        # Use storage component's method
        return await self.storage.s3_get_object(
            key=f"{key}-{window}-{uid}",
            time_min=kwargs.get("time_min"),
            time_max=kwargs.get("time_max"),
        )

    async def gather(self, my_uid, uids, window, key, **kwargs):
        """Mock gather operation with time window support"""
        # Optional time parameters are ignored in the mock.
        # (Previous code extracted time_min and time_max, but they are not used.)

        # Return simulated gradient data
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
                    "layer1.biastotalk": [5],
                }
            ),
            uids=["1"],
            global_steps=[1],
        )

    def mock_get_operation(self, *args, **kwargs):
        """Mock get operation with retry that properly handles time window params"""
        # Optional time parameters are ignored in the mock.
        # (Previous code extracted time_min and time_max, but they are not used.)

        # Return mock gradient data
        return {"gradient": "mock_data"}

    def mock_gather_operation(self, *args, **kwargs):
        """Mock gather operation with time window support"""

        # Return simulated gradient data
        return [{"gradient": "mock_data"}]
