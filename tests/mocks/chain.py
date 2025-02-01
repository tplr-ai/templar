"""Mock chain components"""
import torch
from .base import BaseMock
from unittest.mock import Mock

class MockChain(BaseMock):
    """Mock chain with active/inactive peer tracking"""
    def __init__(self):
        self.active_peers = set()
        self.eval_peers = []
        self.inactive_peers = set()
        self.metagraph = Mock()
        self.metagraph.uids = torch.tensor(range(100))
        self.metagraph.S = torch.ones(100)
        self.metagraph.I = torch.ones(100)

    def update_peers_with_buckets(self):
        """Update inactive peers based on active peers"""
        self.inactive_peers = set(self.eval_peers) - self.active_peers 