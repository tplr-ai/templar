"""Mock metagraph and network components"""
import torch
from .base import BaseMock

class MockMetagraph(BaseMock):
    """Unified mock metagraph for all tests"""
    def __init__(self, n_validators=10):
        # Include our test wallet's address as first hotkey
        self.hotkeys = [
            '5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty',  # Test wallet address
            *[f"hotkey{i}" for i in range(n_validators-1)]  # Other validators
        ]
        self.uids = list(range(n_validators))
        self.n = len(self.uids)
        self.S = torch.ones(self.n)  # Stake values
        self.block = 1000
        self.netuid = 1
        self.name = "mock_network"
        self.I = torch.ones(self.n)  # Incentive values 

    def __call__(self, netuid=1, lite=False):
        """Make metagraph callable with netuid and lite params"""
        self.netuid = netuid
        return self 