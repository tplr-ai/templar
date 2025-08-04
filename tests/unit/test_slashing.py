import pytest
import torch

from neurons.validator import Validator
from tplr.neurons import determine_slash_egregiousness, instantiate_slashing_multiplier


# Mock the Validator's __init__ method to avoid complex setup
def mock_validator_init(self):
    pass


@pytest.fixture
def validator_instance(monkeypatch):
    """Test validator instance"""
    monkeypatch.setattr(Validator, "__init__", mock_validator_init)
    validator = Validator()
    validator.final_scores = torch.ones(10, dtype=torch.float32)
    validator.weights = torch.zeros(10, dtype=torch.float32)
    validator.gradient_scores = torch.zeros(10, dtype=torch.float32)
    validator.binary_moving_averages = torch.ones(10, dtype=torch.float32)
    validator.binary_indicator_scores = torch.zeros(10, dtype=torch.float32)
    validator.sync_scores = torch.zeros(10, dtype=torch.float32)
    validator.openskill_ratings = {}
    validator.eval_peers = {}
    validator.inactive_scores = {}
    validator.idx_similarity_slashing_rate = instantiate_slashing_multiplier()
    validator.naughty_peers = {}
    validator.naughty_peer_timeout = 200
    validator.sync_window = 0
    validator.current_window = 0
    return validator


def test_determine_slash_egregiousness():
    assert determine_slash_egregiousness(0.0) == "high"
    assert determine_slash_egregiousness(0.8) == "high"
    assert determine_slash_egregiousness(0.949) == "high"
    assert determine_slash_egregiousness(0.951) == "max"
    assert determine_slash_egregiousness(0.99) == "max"
    assert determine_slash_egregiousness(1.0) == "mega"

    # Test invalid inputs if validation is added
    with pytest.raises(ValueError):
        determine_slash_egregiousness(-0.1)
    with pytest.raises(ValueError):
        determine_slash_egregiousness(1.1)


def test_instantiate_slashing_multiplier():
    output_dict = instantiate_slashing_multiplier()
    assert isinstance(output_dict, dict)
    assert all(isinstance(val, float) for val in output_dict.values())
    assert all(isinstance(key, str) for key in output_dict)


def test_slash_from_overlap(validator_instance):
    validator = validator_instance

    # Test case 1: High overlap
    idx_overlap_high = {"uids_over_thresh": {1: "high"}}
    validator.slash_from_overlap(idx_overlap_high)
    assert validator.final_scores[1] == 0.5
    assert validator.binary_moving_averages[1] == 0.5

    # Test case 2: Max overlap
    idx_overlap_max = {"uids_over_thresh": {2: "max"}}
    validator.final_scores[2] = 1.0  # reset score
    validator.binary_moving_averages[2] = 1.0  # reset score
    validator.inactive_scores[2] = (0, 1.0)  # Ensure peer exists in inactive_scores
    validator.slash_from_overlap(idx_overlap_max)
    assert validator.final_scores[2] == 0.0
    assert validator.binary_moving_averages[2] == 0.0

    # Test case 3: Mega overlap
    idx_overlap_mega = {"uids_over_thresh": {3: "mega"}}
    validator.final_scores[3] = 1.0  # reset score
    validator.binary_moving_averages[3] = 1.0  # reset score
    validator.inactive_scores[3] = (0, 1.0)  # Ensure peer exists in inactive_scores
    validator.slash_from_overlap(idx_overlap_mega)
    assert validator.final_scores[3] == 0.0
    assert validator.binary_moving_averages[3] == 0.0
    assert 3 in validator.naughty_peers
    expected_timeout = validator.sync_window + validator.naughty_peer_timeout - 1
    assert validator.naughty_peers[3] == expected_timeout

    # Test case 4: Naughty peer timeout
    validator.naughty_peers = {4: 1}
    validator.inactive_scores[4] = (0, 1.0)  # Ensure peer exists in inactive_scores
    idx_overlap_empty = {"uids_over_thresh": {}}
    validator.slash_from_overlap(idx_overlap_empty)
    assert 4 not in validator.naughty_peers

    # Test case 5: No overlap
    validator.final_scores[5] = 1.0  # reset score
    validator.binary_moving_averages[5] = 1.0  # reset score
    idx_overlap_none = {"uids_over_thresh": {}}
    validator.slash_from_overlap(idx_overlap_none)
    assert validator.final_scores[5] == 1.0
    assert validator.binary_moving_averages[5] == 1.0
