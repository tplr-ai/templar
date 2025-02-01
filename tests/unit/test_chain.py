"""Unit tests for chain functionality"""
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from ..utils.assertions import assert_tensor_equal

# Mark all tests as async
pytestmark = pytest.mark.asyncio

class TestChainBasics:
    """Test basic chain functionality"""
    
    @pytest.fixture
    async def chain_instance(self, mock_wallet, mock_metagraph, mock_subtensor):
        """Create chain instance with standard mocks"""
        from tplr.chain import ChainManager
        
        hparams = SimpleNamespace(
            blocks_per_window=100,
            active_check_interval=60,
            recent_windows=5
        )
        
        return ChainManager(
            wallet=mock_wallet,
            metagraph=mock_metagraph,
            subtensor=mock_subtensor,
            hparams=hparams,
            netuid=1
        )

    async def test_initialization(self, chain_instance):
        """Test chain initialization"""
        assert chain_instance.active_peers == set()
        assert chain_instance.eval_peers == []
        assert chain_instance.inactive_peers == set()
        assert chain_instance.netuid == 1

    async def test_block_tracking(self, chain_instance, mock_subtensor):
        """Test block tracking functionality"""
        # Setup mock block values
        mock_subtensor.block.return_value = 1000
        
        # Get current block
        block = await chain_instance.get_current_block()
        assert block == 1000
        
        # Verify window calculation
        window = chain_instance.get_window_from_block(block)
        expected_window = block // chain_instance.hparams.blocks_per_window
        assert window == expected_window

class TestPeerTracking:
    """Test peer tracking functionality"""
    
    @pytest.fixture
    async def chain_instance(self, mock_wallet, mock_metagraph, mock_subtensor):
        """Create chain instance with peer tracking"""
        from neurons.validator.chain import Chain
        
        hparams = SimpleNamespace(
            blocks_per_window=100,
            active_check_interval=60,
            recent_windows=5
        )
        
        chain = Chain(
            wallet=mock_wallet,
            metagraph=mock_metagraph,
            subtensor=mock_subtensor,
            hparams=hparams,
            netuid=1
        )
        
        # Setup initial peers
        chain.eval_peers = [1, 2, 3]
        chain.active_peers = {2, 3}
        
        return chain

    async def test_peer_status_tracking(self, chain_instance):
        """Test tracking of peer status"""
        # Initial state
        assert len(chain_instance.eval_peers) == 3
        assert len(chain_instance.active_peers) == 2
        
        # Update inactive peers
        chain_instance.update_peers_with_buckets()
        assert chain_instance.inactive_peers == {1}
        
        # Add new active peer
        chain_instance.active_peers.add(1)
        chain_instance.update_peers_with_buckets()
        assert len(chain_instance.inactive_peers) == 0

    async def test_peer_activity_windows(self, chain_instance):
        """Test peer activity across windows"""
        # Mock current window
        current_window = 10
        chain_instance.get_current_window = AsyncMock(return_value=current_window)
        
        # Track activity
        await chain_instance.track_peer_activity(uid=1, window=current_window)
        assert 1 in chain_instance.active_peers
        
        # Track inactivity
        chain_instance.active_peers.remove(1)
        chain_instance.update_peers_with_buckets()
        assert 1 in chain_instance.inactive_peers

class TestChainSyncing:
    """Test chain syncing functionality"""
    
    @pytest.fixture
    async def chain_instance(self, mock_wallet, mock_metagraph, mock_subtensor):
        """Create chain instance for sync testing"""
        from neurons.validator.chain import Chain
        
        hparams = SimpleNamespace(
            blocks_per_window=100,
            active_check_interval=60,
            recent_windows=5,
            catch_up_threshold=5,
            catch_up_min_peers=1
        )
        
        return Chain(
            wallet=mock_wallet,
            metagraph=mock_metagraph,
            subtensor=mock_subtensor,
            hparams=hparams,
            netuid=1
        )

    async def test_sync_status(self, chain_instance):
        """Test sync status determination"""
        # Mock current window
        current_window = 10
        chain_instance.get_current_window = AsyncMock(return_value=current_window)
        
        # Test in-sync
        assert not await chain_instance.should_sync(sync_window=current_window-1)
        
        # Test out of sync
        assert await chain_instance.should_sync(sync_window=current_window-6)

    async def test_sync_peer_selection(self, chain_instance):
        """Test selection of peers for syncing"""
        # Setup active peers
        chain_instance.active_peers = {1, 2, 3}
        chain_instance.eval_peers = [1, 2, 3, 4]
        
        # Get sync peers
        sync_peers = chain_instance.get_sync_peers()
        
        # Verify selection
        assert len(sync_peers) >= chain_instance.hparams.catch_up_min_peers
        assert all(p in chain_instance.active_peers for p in sync_peers)

class TestChainEdgeCases:
    """Test chain edge cases and error handling"""
    
    @pytest.fixture
    async def chain_instance(self, mock_wallet, mock_metagraph, mock_subtensor):
        """Create chain instance for edge case testing"""
        from neurons.validator.chain import Chain
        
        hparams = SimpleNamespace(
            blocks_per_window=100,
            active_check_interval=60,
            recent_windows=5
        )
        
        return Chain(
            wallet=mock_wallet,
            metagraph=mock_metagraph,
            subtensor=mock_subtensor,
            hparams=hparams,
            netuid=1
        )

    async def test_block_rollback(self, chain_instance, mock_subtensor):
        """Test handling of block rollbacks"""
        # Setup initial block
        mock_subtensor.block.return_value = 1000
        initial_window = await chain_instance.get_current_window()
        
        # Simulate rollback
        mock_subtensor.block.return_value = 900
        rollback_window = await chain_instance.get_current_window()
        
        assert rollback_window < initial_window

    async def test_peer_churn(self, chain_instance):
        """Test handling of rapid peer changes"""
        # Setup initial peers
        chain_instance.eval_peers = [1, 2, 3]
        chain_instance.active_peers = {2, 3}
        chain_instance.update_peers_with_buckets()
        
        # Rapid changes
        for _ in range(10):
            # Remove random peer
            if chain_instance.active_peers:
                peer = next(iter(chain_instance.active_peers))
                chain_instance.active_peers.remove(peer)
            
            # Add random peer
            new_peer = max(chain_instance.eval_peers) + 1
            chain_instance.eval_peers.append(new_peer)
            chain_instance.active_peers.add(new_peer)
            
            # Update tracking
            chain_instance.update_peers_with_buckets()
            
            # Verify consistency
            assert chain_instance.inactive_peers.isdisjoint(chain_instance.active_peers) 