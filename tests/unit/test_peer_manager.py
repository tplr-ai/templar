import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from collections import defaultdict

from tplr.peer_manager import PeerManager
from tests.mocks import MockChainSync, MockStorageManager, BaseMock

# Mark all tests as async
pytestmark = pytest.mark.asyncio


class MockHparams(BaseMock):
    """Mock hyperparameters for testing PeerManager"""

    def __init__(self):
        super().__init__()
        self.active_check_interval = 60
        self.recent_windows = 3


@pytest.fixture
def mock_storage():
    """Mock storage manager with configurable s3_head_object response"""
    storage = MockStorageManager()
    storage.s3_head_object = AsyncMock()
    return storage


@pytest.fixture
def mock_chain():
    """Create a mock chain with commitments"""
    chain = MockChainSync()
    chain.commitments = {1: "bucket1", 2: "bucket2", 3: "bucket3"}
    chain.current_window = 10
    chain.update_peers_with_buckets = MagicMock()
    return chain


@pytest.fixture
def mock_metagraph():
    """Create a mock metagraph"""
    metagraph = MagicMock()
    metagraph.n = 10
    metagraph.uids = list(range(10))
    return metagraph


@pytest.fixture
def peer_manager(mock_chain, mock_metagraph):
    """Create a PeerManager instance for testing"""
    hparams = MockHparams()
    return PeerManager(mock_chain, hparams, mock_metagraph)


class TestPeerManagerInit:
    """Test initialization of PeerManager"""

    def test_initialization(self, mock_chain, mock_metagraph):
        """Test basic initialization"""
        hparams = MockHparams()

        # Create PeerManager
        manager = PeerManager(mock_chain, hparams, mock_metagraph)

        # Verify attributes
        assert manager.chain == mock_chain
        assert manager.hparams == hparams
        assert manager.metagraph == mock_metagraph
        assert isinstance(manager.active_peers, set)
        assert isinstance(manager.inactive_peers, set)
        assert isinstance(manager.eval_peers, defaultdict)


class TestMinerActivity:
    """Test miner activity checking"""

    @patch("tplr.storage.StorageManager")
    async def test_is_miner_active_true(
        self, mock_storage_class, peer_manager, mock_chain
    ):
        """Test when a miner is active (has gradient files)"""
        # Setup mock storage that returns True for s3_head_object
        mock_storage_instance = AsyncMock()
        mock_storage_instance.s3_head_object = AsyncMock(return_value=True)
        mock_storage_class.return_value = mock_storage_instance

        # Test
        result = await peer_manager.is_miner_active(1, recent_windows=3)

        # Verify
        assert result is True
        assert mock_storage_instance.s3_head_object.call_count == 1

    @patch("tplr.storage.StorageManager")
    async def test_is_miner_active_false(
        self, mock_storage_class, peer_manager, mock_chain
    ):
        """Test when a miner is inactive (no gradient files)"""
        # Setup mock storage that returns False for s3_head_object
        mock_storage_instance = AsyncMock()
        mock_storage_instance.s3_head_object = AsyncMock(return_value=False)
        mock_storage_class.return_value = mock_storage_instance

        # Test
        result = await peer_manager.is_miner_active(1, recent_windows=3)

        # Verify
        assert result is False
        assert mock_storage_instance.s3_head_object.call_count >= 1

    @patch("tplr.storage.StorageManager")
    async def test_is_miner_active_exception(
        self, mock_storage_class, peer_manager, mock_chain
    ):
        """Test error handling during miner activity check"""
        # Setup mock storage that raises exception
        mock_storage_instance = AsyncMock()
        mock_storage_instance.s3_head_object = AsyncMock(
            side_effect=Exception("Test error")
        )
        mock_storage_class.return_value = mock_storage_instance

        # Test
        result = await peer_manager.is_miner_active(1, recent_windows=3)

        # Verify method handles exception and returns False
        assert result is False
        # Should check all windows (3) since exceptions don't stop the loop
        assert mock_storage_instance.s3_head_object.call_count == 3

    @patch("tplr.storage.StorageManager")
    async def test_is_miner_active_no_bucket(
        self, mock_storage_class, peer_manager, mock_chain
    ):
        """Test when bucket is not found for UID"""
        # Setup chain to return no bucket for UID
        peer_manager.chain.get_bucket = MagicMock(return_value=None)
        mock_storage_instance = AsyncMock()
        mock_storage_class.return_value = mock_storage_instance

        # Test
        result = await peer_manager.is_miner_active(99, recent_windows=3)

        # Verify
        assert result is False
        assert mock_storage_instance.s3_head_object.call_count == 0

    @patch("tplr.storage.StorageManager")
    async def test_is_miner_active_with_first_window_success(
        self, mock_storage_class, peer_manager, mock_chain
    ):
        """Test when miner is active in the first window checked"""
        # Setup mock storage that returns True for first window, False for others
        mock_storage_instance = AsyncMock()

        # Return True only for the current window, False for others
        async def head_object_side_effect(*args, **kwargs):
            key = kwargs.get("key", "")
            if f"gradient-{peer_manager.chain.current_window}" in key:
                return True
            return False

        mock_storage_instance.s3_head_object = AsyncMock(
            side_effect=head_object_side_effect
        )
        mock_storage_class.return_value = mock_storage_instance

        # Test
        result = await peer_manager.is_miner_active(1, recent_windows=3)

        # Verify
        assert result is True
        assert (
            mock_storage_instance.s3_head_object.call_count == 1
        )  # Should stop checking after first True

    @patch("tplr.storage.StorageManager")
    async def test_is_miner_active_below_zero_window(
        self, mock_storage_class, peer_manager, mock_chain
    ):
        """Test handling of window numbers below zero"""
        # Set current window to 1 to test below-zero window handling
        peer_manager.chain.current_window = 1

        mock_storage_instance = AsyncMock()
        mock_storage_instance.s3_head_object = AsyncMock(return_value=False)
        mock_storage_class.return_value = mock_storage_instance

        # Test
        result = await peer_manager.is_miner_active(1, recent_windows=3)

        # Verify
        assert result is False
        assert (
            mock_storage_instance.s3_head_object.call_count == 1
        )  # Only one valid window (1)


class TestTrackActivePeers:
    """Test active peer tracking functionality"""

    @patch("tplr.peer_manager.asyncio.Semaphore")
    async def test_track_active_peers_empty(self, mock_semaphore, peer_manager):
        """Test tracking with no peers"""
        # Setup empty commitments
        peer_manager.chain.commitments = {}

        # Create a mock for sleep to control loop execution
        with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            # Make sleep raise exception on first call to exit the infinite loop immediately
            mock_sleep.side_effect = Exception("Break loop")

            # Run with exception handling to simulate one iteration
            try:
                await peer_manager.track_active_peers()
            except Exception as e:
                assert str(e) == "Break loop"

            # Verify empty active_peers and update_peers_with_buckets was called
            assert peer_manager.active_peers == set()
            assert peer_manager.chain.update_peers_with_buckets.call_count == 1
            mock_sleep.assert_called_once_with(
                peer_manager.hparams.active_check_interval
            )

    @patch("tplr.peer_manager.PeerManager.is_miner_active")
    @patch("tplr.peer_manager.asyncio.Semaphore")
    async def test_track_active_peers_all_active(
        self, mock_semaphore, mock_is_active, peer_manager
    ):
        """Test tracking when all peers are active"""
        # Setup mock for is_miner_active to always return True
        mock_is_active.return_value = True

        # Run one iteration of the loop
        with patch("asyncio.sleep", AsyncMock(side_effect=Exception("Break loop"))):
            # Run with exception handling to simulate one iteration
            try:
                # Use the real gather function
                with patch("tplr.peer_manager.asyncio.gather", asyncio.gather):
                    await peer_manager.track_active_peers()
            except Exception as e:
                assert str(e) == "Break loop"

            # Verify active_peers contains all peers from commitments
            assert peer_manager.active_peers == set(
                peer_manager.chain.commitments.keys()
            )
            assert peer_manager.chain.update_peers_with_buckets.call_count == 1

    @patch("tplr.peer_manager.PeerManager.is_miner_active")
    @patch("tplr.peer_manager.asyncio.Semaphore")
    async def test_track_active_peers_mixed(
        self, mock_semaphore, mock_is_active, peer_manager
    ):
        """Test tracking with mix of active and inactive peers"""

        # Setup is_miner_active to return True only for uid 1
        async def is_active_side_effect(uid, recent_windows):
            return uid == 1

        mock_is_active.side_effect = is_active_side_effect

        # Run one iteration
        with patch("asyncio.sleep", AsyncMock(side_effect=Exception("Break loop"))):
            try:
                # Use the real gather function
                with patch("tplr.peer_manager.asyncio.gather", asyncio.gather):
                    await peer_manager.track_active_peers()
            except Exception as e:
                assert str(e) == "Break loop"

            # Verify only uid 1 is active
            assert peer_manager.active_peers == {1}
            assert peer_manager.chain.update_peers_with_buckets.call_count == 1

    @patch(
        "tplr.peer_manager.PeerManager.is_miner_active",
        side_effect=Exception("Check failed"),
    )
    @patch("tplr.peer_manager.asyncio.Semaphore")
    async def test_track_active_peers_error_handling(
        self, mock_semaphore, mock_is_active, peer_manager
    ):
        """Test error handling during peer tracking"""
        # Create a mock for gather that returns a completed future
        mock_gather = AsyncMock()
        mock_gather.return_value = []  # Empty list of results

        # Run one iteration
        with patch("asyncio.sleep", AsyncMock(side_effect=Exception("Break loop"))):
            try:
                # Use a mock that doesn't cause recursion
                with patch("tplr.peer_manager.asyncio.gather", mock_gather):
                    await peer_manager.track_active_peers()
            except Exception as e:
                assert str(e) == "Break loop"

            # Verify no peers marked active due to errors
            assert len(peer_manager.active_peers) == 0
            assert peer_manager.chain.update_peers_with_buckets.call_count == 1
            # Verify gather was called
            assert mock_gather.called

    @patch("tplr.peer_manager.logger")
    @patch("tplr.peer_manager.PeerManager.is_miner_active")
    @patch("tplr.peer_manager.asyncio.Semaphore")
    async def test_track_active_peers_logging(
        self, mock_semaphore, mock_is_active, mock_logger, peer_manager
    ):
        """Test proper logging during active peer tracking"""
        # Setup mock for is_miner_active
        mock_is_active.return_value = True

        # Run one iteration
        with patch("asyncio.sleep", AsyncMock(side_effect=Exception("Break loop"))):
            try:
                # Use the real gather function
                with patch("tplr.peer_manager.asyncio.gather", asyncio.gather):
                    await peer_manager.track_active_peers()
            except Exception as e:
                assert str(e) == "Break loop"

            # Verify logging calls
            assert mock_logger.debug.call_count >= 1
            assert mock_logger.info.call_count >= 1


class TestSemaphoreUsage:
    """Test semaphore usage for limiting concurrent requests"""

    @patch("tplr.storage.StorageManager")
    async def test_semaphore_limits_concurrent_requests(
        self, mock_storage_class, peer_manager
    ):
        """Test that semaphore properly limits concurrent requests"""

        # Create a custom tracking semaphore
        class TrackingSemaphore:
            def __init__(self, value=1):
                self.semaphore = asyncio.Semaphore(value)
                self.active = 0
                self.max_concurrent = 0

            async def __aenter__(self):
                await self.semaphore.__aenter__()
                self.active += 1
                self.max_concurrent = max(self.max_concurrent, self.active)
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.active -= 1
                return await self.semaphore.__aexit__(exc_type, exc_val, exc_tb)

        # Create our tracking semaphore with limit of 3
        tracking_sem = TrackingSemaphore(3)

        # Mock storage with delayed response
        mock_storage_instance = AsyncMock()

        async def delayed_head_object(*args, **kwargs):
            await asyncio.sleep(0.05)  # Small delay to ensure concurrent calls
            return False

        mock_storage_instance.s3_head_object = AsyncMock(
            side_effect=delayed_head_object
        )
        mock_storage_class.return_value = mock_storage_instance

        # Mock Semaphore to return our tracking instance
        with patch("tplr.peer_manager.asyncio.Semaphore", return_value=tracking_sem):
            # Setup many peers to check concurrently
            peer_manager.chain.commitments = {i: f"bucket{i}" for i in range(10)}

            # Run one iteration with exception to break the loop
            with patch("asyncio.sleep", AsyncMock(side_effect=Exception("Break loop"))):
                try:
                    await peer_manager.track_active_peers()
                except Exception as e:
                    assert str(e) == "Break loop"

                # Verify concurrency was limited by semaphore
                assert tracking_sem.max_concurrent <= 3
                # Verify all peers were checked
                assert mock_storage_instance.s3_head_object.call_count > 0
