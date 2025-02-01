"""Mock bittensor module"""
from unittest.mock import MagicMock
import sys
from types import ModuleType
from .wallet import MockWallet, MockSubtensor
from .metagraph import MockMetagraph

class MockBittensor(ModuleType):
    """Mock bittensor module with required components"""
    def __init__(self):
        super().__init__('bt')
        
        # Create a new instance for module-level wallet
        self._wallet_instance = MockWallet()
        
        # Keep MockWallet class for instantiation
        self.Wallet = MockWallet  # Class for instantiation
        self.wallet = MockWallet  
        
        # Mock subtensor instance and class
        self._subtensor_instance = MockSubtensor()
        self.subtensor = MockSubtensor 
        
        # Mock metagraph
        self._metagraph = None
        
        # Mock logging
        self.logging = MagicMock()
        self.logging.add_args = MagicMock()
        
        # Mock config
        self.config = MagicMock(return_value=MagicMock())
        
        # Add other required attributes
        self.trace = MagicMock()
        self.debug = MagicMock()
        
        # Mock argparse additions
        self.wallet.add_args = MagicMock()
        self.subtensor.add_args = MagicMock()
    
    @property
    def metagraph(self):
        """Lazy load metagraph mock"""
        if self._metagraph is None:
            self._metagraph = MockMetagraph()
        return self._metagraph
    
    def __call__(self, config=None):
        """Support bt() call pattern"""
        return self._subtensor_instance  # Return instance instead of class

# Create mock module
mock_bt = MockBittensor()

# Add to sys.modules
sys.modules['bt'] = mock_bt
sys.modules['bittensor'] = mock_bt
