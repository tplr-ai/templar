from unittest.mock import AsyncMock, MagicMock
from .metagraph import MockMetagraph

class MockSubtensor:
    def __init__(self, config=None):
        self.config = config
        self.block = MagicMock(return_value=1000)
        self.get_current_block = MagicMock(return_value=1000)
        self.get_balance = MagicMock(return_value=1000)
        
        # Mock weight setting
        self.set_weights = AsyncMock()
        
        # Add metagraph instance
        self.metagraph = MockMetagraph()
        
        # Mock network info
        self.network = self.metagraph.name
        self.chain_endpoint = "mock_endpoint"
        
        # Mock difficulty
        self.difficulty = MagicMock(return_value=1.0)
        
        # Mock registration
        self.is_hotkey_registered = MagicMock(return_value=True)
        self.register = AsyncMock()
        
        # Mock stake operations
        self.get_stake = MagicMock(return_value=1000)
        self.add_stake = AsyncMock()
        self.remove_stake = AsyncMock()
        
        # Mock neuron info
        self.get_neuron_for_pubkey = MagicMock(return_value={"uid": 1})
        self.get_neuron_for_uid = MagicMock(return_value={"hotkey": "test_hotkey"}) 