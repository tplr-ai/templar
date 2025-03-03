"""Integration test for the Evaluator.

1. Properly detecting new checkpoints by window number
2. Skipping previously evaluated checkpoints
3. Loading and evaluating new checkpoints
4. Updating tracking state correctly
"""

import sys
import os
import pytest
import torch
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluator import Evaluator


def setup_evaluator_with_mocks():
    """Setup evaluator with necessary mocks for testing"""
    with (
        patch(
            "scripts.evaluator.config", return_value=MagicMock(netuid=3, device="cpu")
        ),
        patch("bittensor.subtensor"),
        patch("bittensor.metagraph"),
        patch("tplr.load_hparams"),
        patch("bittensor.wallet"),
        patch("transformers.models.llama.LlamaForCausalLM"),
        patch("tplr.compress.CompressDCT"),
        patch("torch.optim.SGD"),
        patch("torch.optim.lr_scheduler.LinearLR"),
        patch("torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"),
        patch("torch.optim.lr_scheduler.SequentialLR"),
        patch("tplr.compress.TransformDCT"),
        patch("tplr.comms.Comms"),
        patch("wandb.init"),
    ):
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.last_eval_window = 100
        evaluator.config = MagicMock(device="cpu")
        evaluator.model = MagicMock()
        evaluator.metrics_logger = MagicMock()
        evaluator.comms = MagicMock()

        yield evaluator


@pytest.fixture
def evaluator():
    """
    Fixture to return a properly mocked evaluator instance
    """
    yield from setup_evaluator_with_mocks()


@pytest.mark.asyncio
async def test_evaluator_skips_old_checkpoints(evaluator):
    """
    Test that load_latest_model skips checkpoints with window_number <= last_eval_window
    """
    # Create checkpoint with window equal to last_eval_window (should skip)
    old_checkpoint_data = {
        "start_window": 50,
        "current_window": 100,  # Same as last_eval_window
        "model_state_dict": {"layer.weight": torch.zeros(10, 10)},
        "momentum": {"layer.weight": torch.zeros(10, 10)},
    }

    evaluator.comms.get_latest_checkpoint = AsyncMock(
        return_value=(old_checkpoint_data, None)
    )

    success, data, window, step = await evaluator.load_latest_model()

    assert not success, "Should not load when checkpoint window equals last_eval_window"
    assert window == 100, "Should return correct window number"
    assert evaluator.last_eval_window == 100, "last_eval_window should not change"
    evaluator.model.load_state_dict.assert_not_called()


@pytest.mark.asyncio
async def test_evaluator_loads_new_checkpoints(evaluator):
    """
    Test that load_latest_model loads checkpoints with window_number > last_eval_window
    and calculates global step correctly
    """
    # Create checkpoint with window > last_eval_window (should load)
    model_state_dict = {
        "layer.weight": torch.ones(10, 10),
        "layer.bias": torch.ones(10),
    }
    momentum_data = {
        "layer.weight": torch.ones(10, 10) * 0.5,
        "layer.bias": torch.ones(10) * 0.5,
    }

    new_checkpoint_data = {
        "start_window": 50,
        "current_window": 110,
        "model_state_dict": model_state_dict,
        "momentum": momentum_data,
    }

    loaded_model_state = None

    def capture_model_load(state_dict):
        nonlocal loaded_model_state
        loaded_model_state = state_dict
        return None

    evaluator.model.load_state_dict.side_effect = capture_model_load
    evaluator.comms.get_latest_checkpoint = AsyncMock(
        return_value=(new_checkpoint_data, None)
    )

    success, data, window, step = await evaluator.load_latest_model()

    assert success, "Should load when checkpoint window > last_eval_window"
    assert window == 110, "Should return correct window number"
    assert step == 60, "Should calculate global step as 110-50=60"

    evaluator.model.load_state_dict.assert_called_once()

    assert (
        evaluator.momentum == momentum_data
    ), "Should load the exact momentum data from checkpoint"

    evaluator.model.to.assert_called_once_with(evaluator.config.device)

    assert loaded_model_state is not None, "Model state dict should be loaded"
    assert len(loaded_model_state) == len(
        model_state_dict
    ), "Model state dict should have same number of keys"
    for key in model_state_dict:
        assert key in loaded_model_state, f"Key {key} should be in loaded state dict"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
