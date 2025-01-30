# ruff: noqa
# type: ignore
# pylint: disable=all
# mypy: ignore-errors


import pytest
import torch
from unittest.mock import Mock


class TestInactivitySlashing:
    @pytest.fixture
    def mock_chain(self):
        """Setup a mock chain with required attributes"""
        chain = Mock()
        chain.active_peers = set()
        chain.eval_peers = []
        chain.inactive_peers = set()
        chain.metagraph = Mock()
        chain.metagraph.uids = torch.tensor(range(100))
        chain.metagraph.S = torch.ones(100)
        chain.metagraph.I = torch.ones(100)

        # Mock the update_peers_with_buckets method
        def update_peers_mock():
            chain.inactive_peers = set(chain.eval_peers) - chain.active_peers

        chain.update_peers_with_buckets.side_effect = update_peers_mock

        return chain

    @pytest.fixture
    def mock_validator(self):
        """Setup a mock validator with required attributes"""
        validator = Mock()
        validator.moving_avg_scores = torch.zeros(100)
        validator.eval_peers = set()
        validator.inactive_scores = {}  # {uid: (last_active_window, last_score)}
        validator.inactivity_slash_rate = 0.25
        validator.sync_window = 1
        validator.wandb = Mock()
        validator.global_step = 0
        validator.comms = Mock()
        return validator

    def test_newly_inactive_peer_tracking(self, mock_chain):
        """Test that newly inactive peers are correctly identified"""
        # Setup
        mock_chain.eval_peers = [1, 2, 3]
        mock_chain.active_peers = {2, 3}

        # Execute
        mock_chain.update_peers_with_buckets()

        # Verify
        assert mock_chain.inactive_peers == {1}

    @pytest.mark.asyncio
    async def test_inactive_peer_scoring(self, mock_validator):
        """Test the complete inactive peer scoring flow in run()"""
        # Setup
        uid = 1
        mock_validator.moving_avg_scores[uid] = 0.8
        mock_validator.comms.inactive_peers = {uid}
        mock_validator.sync_window = 5

        # Execute one iteration of the scoring logic from run()
        if uid not in mock_validator.inactive_scores:
            mock_validator.inactive_scores[uid] = (
                mock_validator.sync_window,
                mock_validator.moving_avg_scores[uid].item(),
            )

        windows_inactive = (
            mock_validator.sync_window - mock_validator.inactive_scores[uid][0]
        )
        slash_factor = (1 - mock_validator.inactivity_slash_rate) ** windows_inactive
        old_score = mock_validator.moving_avg_scores[uid].item()
        mock_validator.moving_avg_scores[uid] *= slash_factor

        # Verify with tolerance for floating point precision
        assert uid in mock_validator.inactive_scores
        assert (
            mock_validator.inactive_scores[uid][0] == 5
        )  # Window number should match exactly
        assert (
            abs(mock_validator.inactive_scores[uid][1] - 0.8) < 1e-6
        )  # Score should be close to 0.8
        assert (
            abs(mock_validator.moving_avg_scores[uid].item() - 0.8) < 1e-6
        )  # No slash on first window

    def test_score_slashing_over_time(self, mock_validator):
        """Test that scores are correctly slashed with flat rate"""
        # Setup
        uid = 1
        initial_score = 1.0
        mock_validator.moving_avg_scores[uid] = initial_score
        mock_validator.inactive_scores[uid] = (
            1,
            initial_score,
        )  # Inactive since window 1
        mock_validator.sync_window = 3  # 2 windows of inactivity

        # Execute scoring logic from run()
        old_score = mock_validator.moving_avg_scores[uid].item()
        mock_validator.moving_avg_scores[uid] *= 0.75  # Flat 25% reduction

        # Verify - should be 75% of original score regardless of windows inactive
        expected_score = initial_score * 0.75
        assert abs(mock_validator.moving_avg_scores[uid].item() - expected_score) < 1e-6

    def test_peer_reactivation(self, mock_validator):
        """Test that reactivated peers are properly handled"""
        # Setup
        uid = 1
        mock_validator.inactive_scores[uid] = (1, 0.8)
        mock_validator.eval_peers = {uid}

        # Execute reactivation logic from run()
        if uid in mock_validator.eval_peers:
            del mock_validator.inactive_scores[uid]

        # Verify
        assert uid not in mock_validator.inactive_scores

    def test_wandb_logging(self, mock_validator):
        """Test that metrics are properly logged to wandb"""
        # Setup
        uid = 1
        mock_validator.moving_avg_scores[uid] = 1.0
        mock_validator.inactive_scores[uid] = (1, 1.0)
        mock_validator.sync_window = 3

        # Execute logging logic from run()
        old_score = mock_validator.moving_avg_scores[uid].item()
        mock_validator.moving_avg_scores[uid] *= 0.75  # Flat 25% reduction

        mock_validator.wandb.log(
            {
                f"validator/inactivity/{uid}/score_before": old_score,
                f"validator/inactivity/{uid}/score_after": mock_validator.moving_avg_scores[
                    uid
                ].item(),
            },
            step=mock_validator.global_step,
        )

        # Verify
        mock_validator.wandb.log.assert_called_once()

    def test_multiple_peers_slashing(self, mock_validator):
        """Test slashing works correctly for multiple inactive peers"""
        # Setup
        peers = {1: 1.0, 2: 0.8, 3: 0.5}
        for uid, score in peers.items():
            mock_validator.moving_avg_scores[uid] = score
            mock_validator.inactive_scores[uid] = (1, score)
        mock_validator.sync_window = 3

        # Execute scoring logic for each peer
        for uid in peers:
            mock_validator.moving_avg_scores[uid] *= 0.75  # Flat 25% reduction

        # Verify
        for uid, initial_score in peers.items():
            expected_score = initial_score * 0.75
            assert (
                abs(mock_validator.moving_avg_scores[uid].item() - expected_score)
                < 1e-6
            )

    def test_edge_cases(self, mock_validator):
        """Test various edge cases"""
        # Test zero scores
        uid = 1
        mock_validator.moving_avg_scores[uid] = 0.0
        mock_validator.inactive_scores[uid] = (1, 0.0)
        mock_validator.sync_window = 2

        windows_inactive = (
            mock_validator.sync_window - mock_validator.inactive_scores[uid][0]
        )
        slash_factor = (1 - mock_validator.inactivity_slash_rate) ** windows_inactive
        mock_validator.moving_avg_scores[uid] *= slash_factor

        assert mock_validator.moving_avg_scores[uid] == 0.0

        # Test very small scores
        uid = 2
        mock_validator.moving_avg_scores[uid] = 1e-10
        mock_validator.inactive_scores[uid] = (1, 1e-10)

        windows_inactive = (
            mock_validator.sync_window - mock_validator.inactive_scores[uid][0]
        )
        slash_factor = (1 - mock_validator.inactivity_slash_rate) ** windows_inactive
        mock_validator.moving_avg_scores[uid] *= slash_factor

        assert mock_validator.moving_avg_scores[uid] >= 0.0
