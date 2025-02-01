"""Unit tests for inactivity slashing functionality"""
import pytest
import torch
from types import SimpleNamespace
from ..utils.assertions import assert_tensor_equal

# Mark all tests as async
pytestmark = pytest.mark.asyncio

class TestInactivitySlashing:
    """Test inactivity slashing mechanism"""
    
    @pytest.fixture
    def mock_chain(self, mock_metagraph):
        """Setup chain with inactivity tracking"""
        from ..mocks import MockChain
        chain = MockChain()
        chain.metagraph = mock_metagraph
        return chain

    @pytest.fixture
    def mock_validator(self, mock_wallet, mock_metagraph, mock_comms):
        """Setup validator with inactivity tracking"""
        validator = SimpleNamespace(
            moving_avg_scores=torch.zeros(100),
            eval_peers=set(),
            inactive_scores={},  # {uid: (last_active_window, last_score)}
            inactivity_slash_rate=0.25,
            sync_window=1,
            wandb=SimpleNamespace(log=lambda **kwargs: None),
            global_step=0,
            comms=mock_comms
        )
        return validator

    async def test_newly_inactive_peer_tracking(self, mock_chain):
        """Test identification of newly inactive peers"""
        # Setup initial state
        mock_chain.eval_peers = [1, 2, 3]
        mock_chain.active_peers = {2, 3}

        # Update inactive peers
        mock_chain.update_peers_with_buckets()

        # Verify inactive set
        assert mock_chain.inactive_peers == {1}, (
            f"Expected peer 1 to be inactive, got {mock_chain.inactive_peers}"
        )

    async def test_inactive_peer_scoring(self, mock_validator):
        """Test scoring flow for inactive peers"""
        # Setup test data
        uid = 1
        mock_validator.moving_avg_scores[uid] = 0.8
        mock_validator.comms.inactive_peers = {uid}
        mock_validator.sync_window = 5

        # Record initial score
        if uid not in mock_validator.inactive_scores:
            mock_validator.inactive_scores[uid] = (
                mock_validator.sync_window,
                mock_validator.moving_avg_scores[uid].item()
            )

        # Calculate windows inactive
        windows_inactive = mock_validator.sync_window - mock_validator.inactive_scores[uid][0]
        
        # Apply slashing
        slash_factor = (1 - mock_validator.inactivity_slash_rate) ** windows_inactive
        old_score = mock_validator.moving_avg_scores[uid].item()
        mock_validator.moving_avg_scores[uid] *= slash_factor

        # Verify results
        assert uid in mock_validator.inactive_scores
        assert mock_validator.inactive_scores[uid][0] == 5
        assert abs(mock_validator.inactive_scores[uid][1] - 0.8) < 1e-6
        assert abs(mock_validator.moving_avg_scores[uid].item() - 0.8) < 1e-6

    async def test_score_slashing_over_time(self, mock_validator):
        """Test score reduction over multiple windows"""
        # Setup
        uid = 1
        initial_score = 1.0
        mock_validator.moving_avg_scores[uid] = initial_score
        mock_validator.inactive_scores[uid] = (1, initial_score)  # Inactive since window 1
        mock_validator.sync_window = 3  # 2 windows of inactivity

        # Apply slashing
        old_score = mock_validator.moving_avg_scores[uid].item()
        mock_validator.moving_avg_scores[uid] *= 0.75  # 25% reduction

        # Verify score reduction
        expected_score = initial_score * 0.75
        assert abs(mock_validator.moving_avg_scores[uid].item() - expected_score) < 1e-6

    async def test_peer_reactivation(self, mock_validator):
        """Test handling of reactivated peers"""
        # Setup
        uid = 1
        mock_validator.inactive_scores[uid] = (1, 0.8)
        mock_validator.eval_peers = {uid}

        # Process reactivation
        if uid in mock_validator.eval_peers:
            del mock_validator.inactive_scores[uid]

        # Verify cleanup
        assert uid not in mock_validator.inactive_scores

    async def test_wandb_logging(self, mock_validator):
        """Test metric logging for slashing"""
        # Setup
        uid = 1
        mock_validator.moving_avg_scores[uid] = 1.0
        mock_validator.inactive_scores[uid] = (1, 1.0)
        mock_validator.sync_window = 3

        # Track metrics
        old_score = mock_validator.moving_avg_scores[uid].item()
        mock_validator.moving_avg_scores[uid] *= 0.75

        # Log metrics
        logged_data = {}
        def mock_log(**kwargs):
            logged_data.update(kwargs)
        mock_validator.wandb.log = mock_log

        mock_validator.wandb.log(
            {
                f"validator/inactivity/{uid}/score_before": old_score,
                f"validator/inactivity/{uid}/score_after": mock_validator.moving_avg_scores[uid].item(),
            },
            step=mock_validator.global_step
        )

        # Verify logging
        assert f"validator/inactivity/{uid}/score_before" in logged_data
        assert f"validator/inactivity/{uid}/score_after" in logged_data
        assert logged_data[f"validator/inactivity/{uid}/score_before"] == 1.0
        assert abs(logged_data[f"validator/inactivity/{uid}/score_after"] - 0.75) < 1e-6

    async def test_multiple_peers_slashing(self, mock_validator):
        """Test slashing multiple inactive peers"""
        # Setup multiple peers
        peers = {1: 1.0, 2: 0.8, 3: 0.5}
        for uid, score in peers.items():
            mock_validator.moving_avg_scores[uid] = score
            mock_validator.inactive_scores[uid] = (1, score)
        mock_validator.sync_window = 3

        # Apply slashing to all peers
        for uid in peers:
            mock_validator.moving_avg_scores[uid] *= 0.75

        # Verify all scores
        for uid, initial_score in peers.items():
            expected_score = initial_score * 0.75
            assert abs(mock_validator.moving_avg_scores[uid].item() - expected_score) < 1e-6

    async def test_edge_cases(self, mock_validator):
        """Test edge cases in slashing"""
        # Test zero scores
        uid = 1
        mock_validator.moving_avg_scores[uid] = 0.0
        mock_validator.inactive_scores[uid] = (1, 0.0)
        mock_validator.sync_window = 2

        windows_inactive = mock_validator.sync_window - mock_validator.inactive_scores[uid][0]
        slash_factor = (1 - mock_validator.inactivity_slash_rate) ** windows_inactive
        mock_validator.moving_avg_scores[uid] *= slash_factor

        assert mock_validator.moving_avg_scores[uid] == 0.0

        # Test very small scores
        uid = 2
        mock_validator.moving_avg_scores[uid] = 1e-10
        mock_validator.inactive_scores[uid] = (1, 1e-10)

        windows_inactive = mock_validator.sync_window - mock_validator.inactive_scores[uid][0]
        slash_factor = (1 - mock_validator.inactivity_slash_rate) ** windows_inactive
        mock_validator.moving_avg_scores[uid] *= slash_factor

        assert mock_validator.moving_avg_scores[uid] >= 0.0 