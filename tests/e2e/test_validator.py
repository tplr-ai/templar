import asyncio

import pytest

import tplr
from neurons.validator import Validator


@pytest.fixture
def mock_args(mocker):
    """Setup mock command line arguments"""
    args = [
        "validator.py",
        "--netuid",
        "2",
        "--local",  # Use local/small model
        "--wallet.name",
        "validator",
        "--wallet.hotkey",
        "default",
        "--subtensor.network",
        "local",  # Local subtensor
        "--device",
        "cpu",
        "--log-to-private-wandb",
        "--project",
        "templar-test",
    ]
    mocker.patch("sys.argv", args)


async def test_validator(mocker, mock_args):
    """Test that the validator properly updates its peer list"""
    tplr.__version__ = "0.0.0test"
    # Setup mocks
    mock_update_peers = mocker.patch("tplr.neurons.update_peers", autospec=True)
    mock_update_peers_with_buckets = mocker.patch(
        "tplr.comms.Comms.update_peers_with_buckets", autospec=True
    )

    # Create and run validator
    tplr.logger.info("Creating validator")
    validator = Validator()
    tplr.logger.info("Created validator")
    initial_window = validator.current_window

    # Start validator as a task
    tplr.logger.info("Running validator")
    validator_task = asyncio.create_task(validator.run())

    # Wait for next window
    while (
        validator.current_window
        < initial_window + 2 + validator.hparams.validator_offset
    ):
        await asyncio.sleep(1)

    # Stop the validator
    validator_task.cancel()
    try:
        await validator_task
    except asyncio.CancelledError:
        pass

    tplr.logger.info("Stopped validator")

    # Assert that both peer update methods were called
    mock_update_peers.assert_called_once()
    mock_update_peers_with_buckets.assert_called_once()
