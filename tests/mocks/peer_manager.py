"""Mock peer manager for tracking active peers"""

from unittest.mock import AsyncMock
from collections import defaultdict
from .base import BaseMock


class MockPeerManager(BaseMock):
    """Mock peer manager that tracks active and inactive peers."""

    def __init__(self, chain=None, hparams=None, metagraph=None):
        super().__init__()
        self.chain = chain
        self.hparams = hparams
        self.metagraph = metagraph
        self.active_peers = set([1, 2, 3])  # Some default active peers
        self.inactive_peers = set()
        self.eval_peers = defaultdict(int)

        # Mock methods
        self.track_active_peers = AsyncMock()
        self.is_miner_active = AsyncMock(return_value=True)
