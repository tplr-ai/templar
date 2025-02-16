"""Mock wallet and subtensor components"""
from unittest.mock import MagicMock
from types import SimpleNamespace

class MockWallet:
    """Mock wallet with configurable hotkey"""
    def __init__(self, config=None, **kwargs):
        self.config = config
        self.hotkey = SimpleNamespace(
            ss58_address='5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty'  # Fixed test address
        )
        self.coldkey = MagicMock()
        self.coldkey.ss58_address = "test_coldkey_address"
        
        # Add class method for adding args
        self.add_args = MagicMock()
    
    @classmethod
    def add_args(cls, parser):
        """Mock add_args classmethod"""
        pass
    
    @classmethod
    def config(cls):
        """Mock config classmethod"""
        return MagicMock()
    
    @classmethod
    def create_from_config(cls, config):
        """Create wallet from config"""
        return cls()

