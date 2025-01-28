# ruff: noqa

import os
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import torch
import tempfile
from types import SimpleNamespace
from dotenv import load_dotenv
import pytest_asyncio

# Load environment variables from .env file
load_dotenv()

from tplr.comms import Comms
import tplr

# Setup pytest-asyncio
pytestmark = [pytest.mark.asyncio]


# Existing mock functions
def mock_bittensor_wallet():
    wallet = MagicMock()
    wallet.hotkey.ss58_address = "test_hotkey_address"
    return wallet


def mock_bittensor_subtensor():
    subtensor = MagicMock()
    subtensor.block = MagicMock(return_value=1000)
    return subtensor


def mock_metagraph():
    metagraph = MagicMock()
    metagraph.hotkeys = ["test_hotkey_address"]
    metagraph.uids = [0]
    metagraph.S = torch.tensor([100.0])  # Stake
    return metagraph


@pytest_asyncio.fixture
async def comms_instance():
    # Create a Comms instance with mocked dependencies
    wallet = mock_bittensor_wallet()
    metagraph = mock_metagraph()
    hparams = MagicMock()
    hparams.blocks_per_window = 100
    config = MagicMock()
    config.netuid = 1
    config.device = "cpu"

    comms = Comms(
        wallet=wallet,
        save_location="/tmp",
        key_prefix="model",
        config=config,
        netuid=config.netuid,
        metagraph=metagraph,
        hparams=hparams,
        uid=0,  # Using 0 as test UID
    )

    # Mock bucket and commitment methods
    comms.get_own_bucket = MagicMock(return_value="test-bucket")
    comms.try_commit = AsyncMock()
    comms.fetch_commitments = AsyncMock()
    comms.get_commitments_sync = MagicMock(return_value={})
    comms.update_peers_with_buckets = MagicMock()

    # Set up temp directory
    comms.temp_dir = tempfile.mkdtemp()

    yield comms

    # Cleanup temp dir
    if os.path.exists(comms.temp_dir):
        for f in os.listdir(comms.temp_dir):
            os.remove(os.path.join(comms.temp_dir, f))
        os.rmdir(comms.temp_dir)


# Existing tests remain unchanged
async def test_put_local(comms_instance):
    # Test putting data to local storage
    test_state_dict = {"param": torch.tensor([1, 2, 3])}
    uid = "0"
    window = 1
    key = "gradient"

    # Clean up test directory first
    expected_dir = os.path.join("/tmp/local_store", uid, str(window))
    base_dir = os.path.dirname(expected_dir)  # /tmp/local_store/0

    # Recursive cleanup of the uid directory
    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(base_dir)

    # Ensure local directory cleanup is called
    with patch.object(comms_instance, "cleanup_local_data") as mock_cleanup:
        await comms_instance.put(
            state_dict=test_state_dict,
            uid=uid,
            window=window,
            key=key,
            local=True,
        )
        mock_cleanup.assert_called_once()

    # Check that the file was saved locally
    files = os.listdir(expected_dir)
    assert len(files) == 1
    assert files[0].startswith(key)


async def test_get_local(comms_instance):
    # Prepare local file
    test_state_dict = {
        "state_dict": {"param": torch.tensor([1, 2, 3])},
        "global_step": 10,
    }
    uid = "0"
    window = 1
    key = "gradient"
    filename = f"{key}-{window}-{uid}-v{tplr.__version__}.pt"
    local_dir = os.path.join("/tmp/local_store", uid, str(window))
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    torch.save(test_state_dict, local_path)

    # Test getting data from local storage
    with patch.object(comms_instance, "cleanup_local_data") as mock_cleanup:
        state_dict, global_step = await comms_instance.get(
            uid=uid,
            window=window,
            key=key,
            local=True,
        )
        mock_cleanup.assert_called_once()

    assert torch.equal(state_dict["param"], test_state_dict["state_dict"]["param"])
    assert global_step == test_state_dict["global_step"]


# New tests for gather functionality
async def test_gather_basic_functionality(comms_instance):
    # Setup test data
    state_dict = {
        "layer1.weightsidxs": torch.tensor([0, 1, 2]),
        "layer1.weightsvals": torch.tensor([0.1, 0.2, 0.3]),
    }

    # Mock get_with_retry
    comms_instance.get_with_retry = AsyncMock()

    # Mock responses from peers
    peer1_response = (
        {
            "layer1.weightsidxs": torch.tensor([0, 1, 2]),
            "layer1.weightsvals": torch.tensor([0.4, 0.5, 0.6]),
        },
        1,  # global_step
    )
    peer2_response = (
        {
            "layer1.weightsidxs": torch.tensor([0, 1, 2]),
            "layer1.weightsvals": torch.tensor([0.7, 0.8, 0.9]),
        },
        2,  # global_step
    )

    comms_instance.get_with_retry.side_effect = [peer1_response, peer2_response]

    # Call gather
    result = await comms_instance.gather(
        state_dict=state_dict,
        my_uid="0",
        uids=["1", "2"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        global_step=0,
        local=True,
        stale_retention=10,
        store_gathers=False,
    )

    # Verify basic structure
    assert result is not None
    assert isinstance(result, SimpleNamespace)
    assert hasattr(result, "state_dict")
    assert hasattr(result, "uids")
    assert hasattr(result, "global_steps")


async def test_gather_normalization(comms_instance):
    # Test that values are properly normalized
    vals = torch.tensor([3.0, 4.0])  # norm should be 5
    state_dict = {
        "layer.idxs": torch.tensor([0, 1]),
        "layer.vals": vals,
    }

    comms_instance.get_with_retry = AsyncMock()
    peer_response = (state_dict, 1)
    comms_instance.get_with_retry.side_effect = [peer_response]

    result = await comms_instance.gather(
        state_dict=None,
        my_uid="0",
        uids=["1"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        global_step=0,
    )

    # Check normalization
    normalized_vals = getattr(result.state_dict, "layer.vals")[
        0
    ]  # Get first tensor from list
    expected_norm = torch.tensor([0.6, 0.8])  # [3/5, 4/5]
    assert torch.allclose(normalized_vals, expected_norm, rtol=1e-5)


async def test_gather_empty_responses(comms_instance):
    comms_instance.get_with_retry = AsyncMock(side_effect=[None, (None, 0)])

    result = await comms_instance.gather(
        state_dict=None,
        my_uid="0",
        uids=["1", "2"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        global_step=0,
    )

    assert result is None


async def test_gather_store_gathers(comms_instance):
    state_dict = {
        "layer.idxs": torch.tensor([0, 1]),
        "layer.vals": torch.tensor([0.1, 0.2]),
    }

    comms_instance.get_with_retry = AsyncMock()
    peer_response = (state_dict, 1)
    comms_instance.get_with_retry.side_effect = [peer_response]
    comms_instance.s3_put_object = AsyncMock()

    await comms_instance.gather(
        state_dict=None,
        my_uid="0",
        uids=["1"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        global_step=0,
        store_gathers=True,
    )

    assert comms_instance.s3_put_object.called


async def test_gather_averaging(comms_instance):
    # Test with simpler values that are already normalized
    peer1_response = (
        {
            "layer.idxs": torch.tensor([0, 1]),
            "layer.vals": torch.tensor(
                [0.6, 0.8]
            ),  # Already normalized (3-4-5 triangle)
        },
        1,
    )
    peer2_response = (
        {
            "layer.idxs": torch.tensor([0, 1]),
            "layer.vals": torch.tensor([0.6, 0.8]),  # Same values
        },
        2,
    )

    comms_instance.get_with_retry = AsyncMock()
    comms_instance.get_with_retry.side_effect = [peer1_response, peer2_response]

    result = await comms_instance.gather(
        state_dict=None,
        my_uid="0",
        uids=["1", "2"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        global_step=0,
    )

    assert result is not None
    actual_vals = getattr(result.state_dict, "layer.vals")[0]
    expected_vals = torch.tensor(
        [0.6, 0.8]
    )  # Should be same since inputs are identical

    # Print values for debugging
    print(f"Actual: {actual_vals}")
    print(f"Expected: {expected_vals}")

    assert torch.allclose(actual_vals, expected_vals, rtol=1e-3, atol=1e-3)


# TODO:
# - Add tests for edge cases with very large tensors
# - Add tests for different device placements
# - Add tests for timeout scenarios
# - Add tests for concurrent access patterns
# - Add tests for different compression scenarios
