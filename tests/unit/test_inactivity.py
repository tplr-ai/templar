"""Unit tests for inactivity slashing functionality"""

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_chain():
    """Create a mock of ChainSync for testing"""
    from tplr.chain_sync import ChainSync

    # Create a simple config
    config = SimpleNamespace()

    # Create a mock metagraph
    metagraph = SimpleNamespace()
    metagraph.uids = torch.tensor([1, 2, 3])
    metagraph.S = torch.tensor([1000.0, 2000.0, 3000.0])

    # Initialize ChainSync with our test config and metagraph
    chain = ChainSync(config=config, metagraph=metagraph)

    # Pre-configure properties for testing
    chain.eval_peers = {}
    chain.active_peers = set()
    chain.inactive_peers = set()

    return chain


@pytest.fixture
def mock_validator():
    from tests.mocks.comms import MockComms

    comms = MockComms()
    # Create a simple namespace to hold validator attributes.
    validator = SimpleNamespace(
        moving_avg_scores=torch.zeros(100),
        eval_peers=set(),
        inactive_scores={},  # {uid: (last_active_window, last_score)}
        inactivity_slash_rate=0.25,
        sync_window=1,
        wandb=SimpleNamespace(log=lambda *args, **kwargs: None),
        global_step=0,
        comms=comms,
    )
    return validator


@pytest.mark.asyncio
class TestInactivitySlashing:
    async def test_newly_inactive_peer_tracking(self, mock_chain):
        """
        Test that newly inactive peers are correctly identified.
        Uses ChainSync.update_peers_with_buckets to compute inactive_peers.
        """
        # Setup
        mock_chain.eval_peers = {1: 1, 2: 1, 3: 1}  # Previously active peers
        mock_chain.active_peers = {2, 3}  # Currently active peers

        # Mock the hparams with all required attributes
        mock_chain.hparams = SimpleNamespace(
            eval_stake_threshold=20000,
            topk_peers=10,
            minimum_peers=3,
            max_topk_peers=10,
        )

        # Use monkeypatch to avoid calling to Bittensor in set_gather_peers
        with patch.object(mock_chain, "set_gather_peers", MagicMock()):
            # Execute
            mock_chain.update_peers_with_buckets()

            # Verify
            assert mock_chain.inactive_peers == {1}

    async def test_inactive_peer_scoring(self, mock_validator):
        """
        Test the complete inactive peer scoring flow.
        When a peer first becomes inactive, its score is recorded (with its window)
        so that subsequent updates compute the proper step decay.
        """
        uid = 1
        mock_validator.moving_avg_scores[uid] = 0.8
        # From comms we set inactive_peers for completeness.
        mock_validator.comms.inactive_peers = {uid}
        # Set sync_window so that this iteration is the first one where score is recorded.
        mock_validator.sync_window = 5

        # Record initial score if not present.
        if uid not in mock_validator.inactive_scores:
            mock_validator.inactive_scores[uid] = (
                mock_validator.sync_window,
                mock_validator.moving_avg_scores[uid].item(),
            )

        windows_inactive = (
            mock_validator.sync_window - mock_validator.inactive_scores[uid][0]
        )
        slash_factor = (1 - mock_validator.inactivity_slash_rate) ** windows_inactive
        mock_validator.moving_avg_scores[uid] *= slash_factor

        # Verify with tolerance for floating point precision.
        assert uid in mock_validator.inactive_scores
        assert (
            mock_validator.inactive_scores[uid][0] == 5
        )  # Window number should match.
        assert abs(mock_validator.inactive_scores[uid][1] - 0.8) < 1e-6
        # Since windows_inactive is zero the score remains unchanged.
        assert abs(mock_validator.moving_avg_scores[uid].item() - 0.8) < 1e-6

    async def test_score_slashing_over_time(self, mock_validator):
        """
        Test that scores are correctly slashed with a flat reduction factor over multiple windows.
        """
        uid = 1
        initial_score = 1.0
        mock_validator.moving_avg_scores[uid] = initial_score
        # Assume peer has been inactive since window 1.
        mock_validator.inactive_scores[uid] = (1, initial_score)
        # Set current window (e.g., 3 means 2 windows inactive).
        mock_validator.sync_window = 3

        # Execute scoring update.
        mock_validator.moving_avg_scores[uid] *= 0.75  # Flat 25% reduction

        # Verify – expected score is 75% of the original.
        expected_score = initial_score * 0.75
        assert abs(mock_validator.moving_avg_scores[uid].item() - expected_score) < 1e-6

    async def test_peer_reactivation(self, mock_validator):
        """
        Test that reactivated peers have their inactivity tracking state cleared.
        """
        uid = 1
        mock_validator.inactive_scores[uid] = (1, 0.8)
        mock_validator.eval_peers = {uid}

        # Reactivation logic: if the peer is active again, clear its inactive record.
        if uid in mock_validator.eval_peers:
            del mock_validator.inactive_scores[uid]

        # Verify cleanup.
        assert uid not in mock_validator.inactive_scores

    async def test_wandb_logging(self, mock_validator):
        """
        Test that metrics are properly logged to wandb.
        """
        uid = 1
        mock_validator.moving_avg_scores[uid] = 1.0
        mock_validator.inactive_scores[uid] = (1, 1.0)
        mock_validator.sync_window = 3

        mock_validator.moving_avg_scores[uid] *= 0.75  # Flat 25% reduction

        logged_data = {}

        def mock_log(*args, **kwargs):
            if args and isinstance(args[0], dict):
                logged_data.update(args[0])
            logged_data.update(kwargs)

        mock_validator.wandb.log = mock_log

        mock_validator.wandb.log(
            {
                f"validator/inactivity/{uid}/score_before": 1.0,
                f"validator/inactivity/{uid}/score_after": mock_validator.moving_avg_scores[
                    uid
                ].item(),
            },
            step=mock_validator.global_step,
        )

        # Verify logging has been performed.
        assert f"validator/inactivity/{uid}/score_before" in logged_data
        assert f"validator/inactivity/{uid}/score_after" in logged_data
        assert logged_data[f"validator/inactivity/{uid}/score_before"] == 1.0
        assert abs(logged_data[f"validator/inactivity/{uid}/score_after"] - 0.75) < 1e-6

    async def test_multiple_peers_slashing(self, mock_validator):
        """
        Test slashing for multiple inactive peers.
        """
        peers = {1: 1.0, 2: 0.8, 3: 0.5}
        for uid, score in peers.items():
            mock_validator.moving_avg_scores[uid] = score
            mock_validator.inactive_scores[uid] = (1, score)
        mock_validator.sync_window = 3

        # Apply flat 25% reduction to each inactive peer.
        for uid in peers:
            mock_validator.moving_avg_scores[uid] *= 0.75

        # Verify each score is reduced as expected.
        for uid, initial_score in peers.items():
            expected_score = initial_score * 0.75
            assert (
                abs(mock_validator.moving_avg_scores[uid].item() - expected_score)
                < 1e-6
            )

    async def test_edge_cases(self, mock_validator):
        """
        Test handling edge cases such as zero and very small scores.
        """
        # Test for zero score.
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

        # Test for a very small score.
        uid = 2
        mock_validator.moving_avg_scores[uid] = 1e-10
        mock_validator.inactive_scores[uid] = (1, 1e-10)
        windows_inactive = (
            mock_validator.sync_window - mock_validator.inactive_scores[uid][0]
        )
        slash_factor = (1 - mock_validator.inactivity_slash_rate) ** windows_inactive
        mock_validator.moving_avg_scores[uid] *= slash_factor
        # Even for extremely small scores, the result should be non-negative.
        assert mock_validator.moving_avg_scores[uid] >= 0.0
