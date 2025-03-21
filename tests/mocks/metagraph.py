"""Mock metagraph and network components"""

import torch
from .base import BaseMock


class MockMetagraph(BaseMock):
    """Unified mock metagraph for all tests"""

    def __init__(self, n_validators=10):
        super().__init__()
        # Include our test wallet's address as first hotkey
        self.hotkeys = [
            "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # Test wallet address
            *[f"hotkey{i}" for i in range(n_validators - 1)],  # Other validators
            "default",  # Include the wallet's hotkey
        ]

        # Create UIDs as tensor to match real implementation
        self.uids = torch.tensor(list(range(n_validators)))
        self.n = len(self.uids)

        # Create stake values as tensor (varies for testing validator selection)
        self.S = torch.tensor([100.0 + (i * 100.0) for i in range(n_validators)])

        # Trust values - identifies validators (> 0 = validator)
        # Make 50% of nodes validators (trust > 0)
        self.T = torch.zeros(self.n)
        for i in range(0, self.n, 2):  # Every other node is a validator
            self.T[i] = 1.0

        # Incentive values - varies for testing peer selection by incentive
        self.I = torch.tensor([0.5 + (i * 0.1) for i in range(n_validators)])

        # Other common attributes
        self.block = 1000
        self.netuid = 1
        self.name = "mock_network"

    def __call__(self, netuid=1, lite=False):
        """Make metagraph callable with netuid and lite params"""
        self.netuid = netuid
        return self

    # Add item method for tensor-like access
    def __getitem__(self, idx):
        """Allow tensor-like indexing into the metagraph"""
        return {
            "uid": self.uids[idx],
            "stake": self.S[idx],
            "trust": self.T[idx],
            "incentive": self.I[idx],
            "hotkey": self.hotkeys[idx] if idx < len(self.hotkeys) else f"hotkey{idx}",
        }
