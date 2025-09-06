"""Integration test for the Evaluator.

1. Properly detecting new checkpoints by window number
2. Skipping previously evaluated checkpoints
3. Discovering new checkpoints
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
        patch("tplr.distributed.dist_helper.init_process_group"),
        patch("tplr.distributed.dist_helper.is_master", return_value=True),
        patch("tplr.distributed.dist_helper.rank", 0),
        patch("tplr.distributed.dist_helper.world_size", 2),
        patch("tplr.distributed.dist_helper.local_rank", 0),
        patch("tplr.distributed.dist_helper.device", "cpu"),
        patch("tplr.load_hparams"),
        patch("tplr.comms.Comms"),
        patch("tplr.DCPCheckpointer"),
        patch("tplr.initialize_wandb"),
        patch("tplr.metrics.MetricsLogger"),
        patch("neurons.evaluator.Evaluator._initialize_or_load_model"),
        patch("torch.distributed.broadcast"),
    ):
        evaluator = Evaluator.__new__(Evaluator)

        # Setup basic attributes
        evaluator.config = MagicMock(
            netuid=3,
            device="cpu",
            eval_interval=30,
            version="test_version",
            cache_dir="/tmp/cache",
            tasks="arc_easy",
        )
        evaluator.netuid = 3
        evaluator.version = "test_version"
        evaluator.eval_interval = 30

        # Setup distributed attributes
        evaluator.rank = 0
        evaluator.world_size = 2
        evaluator.local_rank = 0
        evaluator.is_master = True
        evaluator.device = "cpu"

        # Setup model and components
        evaluator.model = MagicMock()
        evaluator.metrics_logger = MagicMock()
        evaluator.comms = MagicMock()
        evaluator.hparams = MagicMock()
        evaluator.hparams.blocks_per_window = 100
        evaluator.hparams.checkpoint_init_version = (
            None  # No bootstrap version by default
        )
        evaluator.ckpt = MagicMock()

        # Setup state tracking
        evaluator.evaluated_windows = set()
        evaluator.baseline_evaluated = False
        evaluator.last_discovered_window = None
        evaluator.start_window = 0

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
    Test that check_latest_checkpoint skips already evaluated checkpoints
    """
    # Mark window 100 as already evaluated
    evaluator.evaluated_windows.add(100)

    # Mock _discover_latest to return window 100 (already evaluated)
    evaluator.ckpt._discover_latest = AsyncMock(return_value=100)

    # Call check_latest_checkpoint
    result = await evaluator.check_latest_checkpoint()

    # Verify that _discover_latest was called with correct parameters
    evaluator.ckpt._discover_latest.assert_called_once_with(prefer_highest_staked=True)

    # Should return None since window 100 is already evaluated
    assert result is None, "Should return None for already evaluated windows"
    assert evaluator.last_discovered_window == 100, (
        "Should update last_discovered_window"
    )


@pytest.mark.asyncio
async def test_evaluator_loads_new_checkpoints(evaluator):
    """
    Test that check_latest_checkpoint discovers new checkpoints not yet evaluated
    """
    # No windows evaluated yet
    evaluator.evaluated_windows = set()
    evaluator.last_discovered_window = None

    # Mock _discover_latest to return a new checkpoint at window 110
    evaluator.ckpt._discover_latest = AsyncMock(return_value=110)

    # Call check_latest_checkpoint
    result = await evaluator.check_latest_checkpoint()

    # Verify that _discover_latest was called with correct parameters
    evaluator.ckpt._discover_latest.assert_called_once_with(prefer_highest_staked=True)

    assert result == 110, "Should return the new window number"
    assert evaluator.last_discovered_window == 110, (
        "Should update last_discovered_window"
    )
    assert 110 not in evaluator.evaluated_windows, "Should not mark as evaluated yet"


@pytest.mark.asyncio
async def test_evaluate_window_successful(evaluator):
    """
    Test successful evaluation of a checkpoint window
    """
    # Setup mocks
    evaluator.ckpt.download_and_load = AsyncMock(return_value=(120, 20))
    evaluator.save_model_for_eval = MagicMock(return_value="/tmp/model_120")
    evaluator.run_custom_eval = MagicMock(return_value=(5.2, 1.8))
    evaluator.run_lm_eval_multi_gpu = MagicMock(return_value={"arc_easy": 0.75})
    evaluator.model_cache = MagicMock()
    evaluator.tasks = ["arc_easy"]

    # Mock pause_ddp_for_lm_eval context manager
    with patch("neurons.evaluator.pause_ddp_for_lm_eval"):
        success = await evaluator.evaluate_window(120, is_baseline=False)

    assert success, "Should return True for successful evaluation"
    assert 120 in evaluator.evaluated_windows, "Should mark window as evaluated"
    evaluator.ckpt.download_and_load.assert_called_once()
    evaluator.save_model_for_eval.assert_called_once_with(120)
    evaluator.run_custom_eval.assert_called_once_with(120, 20)


@pytest.mark.asyncio
async def test_evaluate_window_skips_already_evaluated(evaluator):
    """
    Test that evaluate_window skips already evaluated windows
    """
    # Mark window 100 as already evaluated
    evaluator.evaluated_windows.add(100)

    success = await evaluator.evaluate_window(100)

    assert success, "Should return True for already evaluated window"
    # Should not call any evaluation methods
    evaluator.ckpt.download_and_load.assert_not_called()


@pytest.mark.asyncio
async def test_evaluate_window_baseline(evaluator):
    """
    Test baseline evaluation (no checkpoint loading)
    """
    evaluator.save_model_for_eval = MagicMock(return_value="/tmp/model_baseline")
    evaluator.run_custom_eval = MagicMock(return_value=(10.5, 2.3))
    evaluator.run_lm_eval_multi_gpu = MagicMock(return_value={"arc_easy": 0.25})
    evaluator.model_cache = MagicMock()
    evaluator.tasks = ["arc_easy"]
    evaluator.start_window = 0

    with patch("neurons.evaluator.pause_ddp_for_lm_eval"):
        success = await evaluator.evaluate_window(0, is_baseline=True)

    assert success, "Should return True for successful baseline evaluation"
    assert 0 in evaluator.evaluated_windows, "Should mark window 0 as evaluated"
    # Should not try to download checkpoint for baseline
    evaluator.ckpt.download_and_load.assert_not_called()
    evaluator.save_model_for_eval.assert_called_once_with(0)


@pytest.mark.asyncio
async def test_evaluate_window_download_failure(evaluator):
    """
    Test handling of checkpoint download failure
    """
    evaluator.ckpt.download_and_load = AsyncMock(return_value=None)

    success = await evaluator.evaluate_window(130, is_baseline=False)

    assert not success, "Should return False when checkpoint download fails"
    assert 130 not in evaluator.evaluated_windows, (
        "Should not mark as evaluated on failure"
    )


@pytest.mark.asyncio
async def test_model_cache_exists(evaluator):
    """
    Test model cache functionality when model already cached
    """
    from pathlib import Path

    from neurons.evaluator import ModelCache

    cache = ModelCache(Path("/tmp/test_cache"))
    evaluator.model_cache = cache

    # Mock cache.exists to return True
    with patch.object(cache, "exists", return_value=True):
        with patch.object(cache, "get_path", return_value=Path("/tmp/cached_model")):
            result = evaluator.save_model_for_eval(100)

    assert result == Path("/tmp/cached_model"), "Should return cached path"


@pytest.mark.asyncio
async def test_run_initial_baseline(evaluator):
    """
    Test run() method with no checkpoints (baseline evaluation)
    """
    evaluator.comms.get_commitments = AsyncMock(return_value={})
    evaluator.comms.get_start_window = AsyncMock(return_value=0)
    evaluator.check_latest_checkpoint = AsyncMock(return_value=None)
    evaluator.evaluate_window = AsyncMock(return_value=True)
    evaluator.baseline_evaluated = False

    # Mock asyncio.sleep to prevent infinite loop
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        # Set side effect to raise exception after first iteration
        mock_sleep.side_effect = [None, KeyboardInterrupt()]

        try:
            await evaluator.run()
        except KeyboardInterrupt:
            pass

    # Should evaluate baseline
    evaluator.evaluate_window.assert_called_with(0, is_baseline=True)
    assert evaluator.start_window == 0


@pytest.mark.asyncio
async def test_run_with_checkpoint(evaluator):
    """
    Test run() method with existing checkpoint
    """
    evaluator.comms.get_commitments = AsyncMock(return_value={})
    evaluator.comms.get_start_window = AsyncMock(return_value=0)
    evaluator.check_latest_checkpoint = AsyncMock(return_value=150)
    evaluator.evaluate_window = AsyncMock(return_value=True)
    evaluator.baseline_evaluated = False

    # Mock asyncio.sleep to prevent infinite loop
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = [None, KeyboardInterrupt()]

        try:
            await evaluator.run()
        except KeyboardInterrupt:
            pass

    # Should evaluate checkpoint 150
    evaluator.evaluate_window.assert_called_with(150)


@pytest.mark.asyncio
async def test_run_loop_discovers_new_checkpoint(evaluator):
    """
    Test main loop discovering and evaluating new checkpoint
    """
    evaluator.comms.get_commitments = AsyncMock(return_value={})
    evaluator.comms.get_start_window = AsyncMock(return_value=0)
    evaluator.evaluate_window = AsyncMock(return_value=True)
    evaluator.config.eval_interval = 1  # Short interval for testing

    # First check returns None, second check returns 160
    evaluator.check_latest_checkpoint = AsyncMock(
        side_effect=[None, 160, KeyboardInterrupt()]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with patch("tplr.distributed.dist_helper.safe_barrier"):
            try:
                await evaluator.run()
            except KeyboardInterrupt:
                pass

    # Should have evaluated window 160
    assert evaluator.evaluate_window.call_count >= 1
    calls = evaluator.evaluate_window.call_args_list
    assert any(call[0][0] == 160 for call in calls), "Should evaluate window 160"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
