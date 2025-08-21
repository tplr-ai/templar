"""Integration test for the Evaluator.

1. Properly detecting new checkpoints by window number
2. Skipping previously evaluated checkpoints
3. Loading and evaluating new checkpoints
4. Updating tracking state correctly
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurons.evaluator import Evaluator


def setup_evaluator_with_mocks():
    """Setup evaluator with necessary mocks for testing"""
    with (
        patch(
            "neurons.evaluator.Evaluator.evaluator_config",
            return_value=MagicMock(netuid=3, device="cpu"),
        ),
        patch("bittensor.subtensor"),
        patch("bittensor.metagraph"),
        patch("tplr.load_hparams"),
        patch("bittensor.wallet"),
        patch("transformers.models.llama.LlamaForCausalLM"),
        patch("tplr.compress.TopKCompressor"),
        patch("torch.optim.SGD"),
        patch("torch.optim.lr_scheduler.LinearLR"),
        patch("torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"),
        patch("torch.optim.lr_scheduler.SequentialLR"),
        patch("tplr.compress.ChunkingTransformer"),
        patch("tplr.comms.Comms"),
    ):
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.last_eval_window = 100
        evaluator.config = MagicMock(device="cpu")
        evaluator.model = MagicMock()
        evaluator.metrics_logger = MagicMock()
        evaluator.comms = MagicMock()
        evaluator.version = "test_version"
        evaluator.subtensor = MagicMock()
        evaluator.hparams = MagicMock(blocks_per_window=100)
        evaluator.is_master = True

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
    # Mock subtensor to return current block
    evaluator.subtensor.get_current_block = MagicMock(return_value=10100)  # window 101

    # Mock load_checkpoint to return failure (window <= last_eval_window)
    mock_load_checkpoint = AsyncMock(return_value=(False, 100))
    evaluator.comms.load_checkpoint = mock_load_checkpoint

    success, window, step = await evaluator.load_latest_model()

    # Verify that load_checkpoint was called with correct parameters
    mock_load_checkpoint.assert_called_once_with(
        model=evaluator.model,
        current_window=101,
        init_version=evaluator.version,
        is_master=True,
    )

    assert not success, "Should not load when checkpoint window equals last_eval_window"
    assert window == 0, "Should return 0 for window when checkpoint loading fails"
    assert evaluator.last_eval_window == 100, "last_eval_window should not change"


@pytest.mark.asyncio
async def test_evaluator_loads_new_checkpoints(evaluator):
    """
    Test that load_latest_model loads checkpoints with window_number > last_eval_window
    and calculates global step correctly
    """
    # Mock subtensor to return current block
    evaluator.subtensor.get_current_block = MagicMock(return_value=11100)  # window 111

    # Mock load_checkpoint to return success with new checkpoint window
    mock_load_checkpoint = AsyncMock(return_value=(True, 110))
    evaluator.comms.load_checkpoint = mock_load_checkpoint

    success, window, step = await evaluator.load_latest_model()

    # Verify that load_checkpoint was called with correct parameters
    mock_load_checkpoint.assert_called_once_with(
        model=evaluator.model,
        current_window=111,
        init_version=evaluator.version,
        is_master=True,
    )

    assert success, "Should load when checkpoint window > last_eval_window"
    assert window == 110, "Should return correct window number"
    assert step == 110, "Global step should match checkpoint window"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
