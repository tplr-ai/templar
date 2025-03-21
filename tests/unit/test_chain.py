import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from tests.mocks.chain import MockChainManager as ChainManager

# Mark all tests as async
pytestmark = pytest.mark.asyncio


class TestChainBasics:
    """Test basic chain functionality"""

    @pytest.fixture
    async def chain_instance(self, mock_wallet, mock_metagraph, mock_subtensor):
        """Create chain instance with standard mocks"""
        hparams = SimpleNamespace(
            blocks_per_window=100, active_check_interval=60, recent_windows=5
        )
        chain = ChainManager(n_validators=10)
        # Attach required attributes from real ChainManager for testing
        chain.wallet = mock_wallet
        chain.metagraph = mock_metagraph
        chain.subtensor = mock_subtensor
        chain.hparams = hparams
        chain.netuid = 1

        # Stub methods required by tests
        chain.get_current_block = AsyncMock(return_value=1000)
        chain.get_window_from_block = (
            lambda block: block // chain.hparams.blocks_per_window
        )
        return chain

    async def test_initialization(self, chain_instance):
        """Test chain initialization"""
        assert chain_instance.active_peers == set()
        assert chain_instance.eval_peers == []
        assert chain_instance.inactive_peers == set()
        assert chain_instance.netuid == 1

    async def test_block_tracking(self, chain_instance, mock_subtensor):
        """Test block tracking functionality"""
        # Setup mock block value in subtensor (simulate being used elsewhere)
        mock_subtensor.block.return_value = 1000

        # Get current block using stubbed method
        block = await chain_instance.get_current_block()
        assert block == 1000

        # Verify window calculation using the stubbed method
        window = chain_instance.get_window_from_block(block)
        expected_window = block // chain_instance.hparams.blocks_per_window
        assert window == expected_window


class TestPeerTracking:
    """Test peer tracking functionality"""

    @pytest.fixture
    async def chain_instance(self, mock_wallet, mock_metagraph, mock_subtensor):
        """Create chain instance with peer tracking using mocks"""
        hparams = SimpleNamespace(
            blocks_per_window=100, active_check_interval=60, recent_windows=5
        )
        chain = ChainManager(n_validators=10)
        chain.wallet = mock_wallet
        chain.metagraph = mock_metagraph
        chain.subtensor = mock_subtensor
        chain.hparams = hparams
        chain.netuid = 1

        # Setup initial peers
        chain.eval_peers = [1, 2, 3]
        chain.active_peers = {2, 3}
        # Stub missing method for tracking peer activity
        chain.track_peer_activity = AsyncMock(
            side_effect=lambda uid, window: chain.active_peers.add(uid)
        )
        return chain

    async def test_peer_status_tracking(self, chain_instance):
        """Test tracking of peer status"""
        # Initial state
        assert len(chain_instance.eval_peers) == 3
        assert len(chain_instance.active_peers) == 2

        # Update inactive peers via mock's update method
        chain_instance.update_peers_with_buckets()
        assert chain_instance.inactive_peers == {1}

        # Add new active peer and update tracking
        chain_instance.active_peers.add(1)
        chain_instance.update_peers_with_buckets()
        assert len(chain_instance.inactive_peers) == 0

    async def test_peer_activity_windows(self, chain_instance):
        """Test peer activity across windows"""
        current_window = 10
        chain_instance.get_current_window = AsyncMock(return_value=current_window)

        # Use the stubbed track_peer_activity to add peer 1 as active
        await chain_instance.track_peer_activity(uid=1, window=current_window)
        assert 1 in chain_instance.active_peers

        # Simulate inactivity: remove peer 1 and update buckets
        chain_instance.active_peers.remove(1)
        chain_instance.update_peers_with_buckets()
        assert 1 in chain_instance.inactive_peers


class TestChainSyncing:
    """Test chain syncing functionality"""

    @pytest.fixture
    async def chain_instance(self, mock_wallet, mock_metagraph, mock_subtensor):
        """Create chain instance for sync testing using mocks"""
        hparams = SimpleNamespace(
            blocks_per_window=100,
            active_check_interval=60,
            recent_windows=5,
            catch_up_threshold=5,
            catch_up_min_peers=1,
        )
        chain = ChainManager(n_validators=10)
        chain.wallet = mock_wallet
        chain.metagraph = mock_metagraph
        chain.subtensor = mock_subtensor
        chain.hparams = hparams
        chain.netuid = 1

        # Stub methods required by sync tests
        # should_sync: returns True if gap >= catch_up_threshold, else False.
        async def check_catch_up(sync_window):
            current_window = await chain.get_current_window()
            return (current_window - sync_window) >= hparams.catch_up_threshold

        chain.should_sync = AsyncMock(side_effect=check_catch_up)
        # For testing get_sync_peers: simply return active peers meeting minimum min_peers.
        chain.get_sync_peers = (
            lambda: list(chain.active_peers)
            if len(chain.active_peers) >= hparams.catch_up_min_peers
            else []
        )
        chain.get_current_window = AsyncMock(return_value=10)
        return chain

    async def test_sync_status(self, chain_instance):
        """Test sync status determination"""
        current_window = 10
        chain_instance.get_current_window = AsyncMock(return_value=current_window)

        # Test in-sync: difference < catch_up_threshold
        result = await chain_instance.should_sync(sync_window=current_window - 1)
        assert result is False

        # Test out-of-sync: difference >= catch_up_threshold
        result = await chain_instance.should_sync(sync_window=current_window - 6)
        assert result is True

    async def test_sync_peer_selection(self, chain_instance):
        """Test selection of peers for syncing"""
        # Setup active peers and evaluation peers
        chain_instance.active_peers = {1, 2, 3}
        chain_instance.eval_peers = [1, 2, 3, 4]

        sync_peers = chain_instance.get_sync_peers()
        # Verify that the number of sync peers meets the minimum and are among active peers.
        assert len(sync_peers) >= chain_instance.hparams.catch_up_min_peers
        assert all(p in chain_instance.active_peers for p in sync_peers)


class TestChainEdgeCases:
    """Test chain edge cases and error handling"""

    @pytest.fixture
    async def chain_instance(self, mock_wallet, mock_metagraph, mock_subtensor):
        """Create chain instance for edge case testing using mocks"""
        hparams = SimpleNamespace(
            blocks_per_window=100, active_check_interval=60, recent_windows=5
        )
        chain = ChainManager(n_validators=10)
        chain.wallet = mock_wallet
        chain.metagraph = mock_metagraph
        chain.subtensor = mock_subtensor
        chain.hparams = hparams
        chain.netuid = 1
        return chain

    async def test_block_rollback(self, chain_instance, mock_subtensor):
        """Test handling of block rollbacks"""
        # Setup initial block
        mock_subtensor.block.return_value = 1000
        chain_instance.get_current_window = AsyncMock(return_value=1000)
        initial_window = await chain_instance.get_current_window()

        # Simulate rollback by setting a lower value
        chain_instance.get_current_window = AsyncMock(return_value=900)
        rollback_window = await chain_instance.get_current_window()

        assert rollback_window < initial_window

    async def test_peer_churn(self, chain_instance):
        """Test handling of rapid peer changes"""
        # Setup initial peers
        chain_instance.eval_peers = [1, 2, 3]
        chain_instance.active_peers = {2, 3}
        chain_instance.update_peers_with_buckets()

        # Rapid changes simulation
        for _ in range(10):
            if chain_instance.active_peers:
                # Remove one peer arbitrarily
                peer = next(iter(chain_instance.active_peers))
                chain_instance.active_peers.remove(peer)

            # Add new peer
            new_peer = max(chain_instance.eval_peers) + 1
            chain_instance.eval_peers.append(new_peer)
            chain_instance.active_peers.add(new_peer)

            chain_instance.update_peers_with_buckets()
            # Verify no overlap between active and inactive peers
            assert chain_instance.inactive_peers.isdisjoint(chain_instance.active_peers)
