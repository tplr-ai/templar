"""Mock peer manager for tracking active peers"""

from .base import BaseMock


class MockPeerManager(BaseMock):
    """Mock peer manager that tracks active and inactive peers."""

    def __init__(self):
        super().__init__()
        self.active_peers = set([1, 2, 3])  # Some default active peers

    async def track_active_peers(self):
        """Mock implementation of active peer tracking."""
        pass

    async def is_miner_active(self, uid, window):
        """Check if a miner is active within recent windows."""
        return uid in self.active_peers

    def weighted_sample(self, candidates, weights, k):
        """Sample k peers weighted by their scores."""
        return candidates[: min(k, len(candidates))]
