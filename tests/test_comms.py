# ruff: noqa

import os
from unittest.mock import patch, MagicMock
import pytest
import torch

# Now import tplr modules

# from tplr.schemas import Bucket
# from tplr.config import BUCKET_SECRETS

from dotenv import load_dotenv
import pytest_asyncio

# Load environment variables from .env file
load_dotenv()

from tplr.comms import Comms
import tplr


# Setup pytest-asyncio
pytestmark = [pytest.mark.asyncio]


# Mock Bittensor wallet and subtensor
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
    subtensor = mock_bittensor_subtensor()
    metagraph = mock_metagraph()
    hparams = MagicMock()
    hparams.blocks_per_window = 100

    comms = Comms(
        wallet=wallet,
        config=None,
        netuid=1,
        metagraph=metagraph,
        hparams=hparams,
    )
    return comms


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


async def test_gather(comms_instance):
    # Test gathering data from multiple UIDs
    state_dict = {"param": torch.tensor([1, 2, 3])}
    my_uid = "0"
    uids = ["1", "2"]
    window = 1
    key = "gradient"
    device = "cpu"

    # Mock get_with_retry to return test state dicts
    async def mock_get_with_retry(uid, window, key, timeout, local, stale_retention):
        test_state_dict = {
            "param": torch.tensor([int(uid), int(uid) + 1, int(uid) + 2])
        }
        global_step = int(uid) * 10
        return test_state_dict, global_step

    comms_instance.get_with_retry = MagicMock(side_effect=mock_get_with_retry)

    result = await comms_instance.gather(
        state_dict=state_dict,
        my_uid=my_uid,
        uids=uids,
        window=window,
        key=key,
        timeout=5,
        device=device,
        global_step=10,
    )

    assert result is not None
    assert len(result.uids) == len(uids)
    assert len(result.state_dict.param) == len(uids)
    assert torch.equal(result.state_dict.param[0], torch.tensor([1, 2, 3]))
    assert torch.equal(result.state_dict.param[1], torch.tensor([2, 3, 4]))
    assert result.global_steps == [10, 20]


async def test_checkpoint_saving_loading(comms_instance):
    # Test checkpoint saving and loading
    test_checkpoint = {
        "model_state_dict": {"param": torch.tensor([1, 2, 3])},
        "optimizer_state_dict": {"lr": 0.01},
        "scheduler_state_dict": {"last_epoch": 10},
        "momentum": {"param": torch.tensor([0.1, 0.1, 0.1])},
        "global_step": 100,
    }
    uid = "0"
    window = 1
    key = "checkpoint"

    # Save checkpoint
    with patch.object(comms_instance, "cleanup_local_data") as mock_cleanup:
        await comms_instance.put(
            state_dict=test_checkpoint,
            uid=uid,
            window=window,
            key=key,
            local=True,
        )
        mock_cleanup.assert_called_once()

    # Load checkpoint
    with patch.object(comms_instance, "cleanup_local_data") as mock_cleanup:
        loaded_checkpoint, _ = await comms_instance.get(
            uid=uid,
            window=window,
            key=key,
            local=True,
        )
        mock_cleanup.assert_called_once()

    # Compare non-tensor values
    assert (
        loaded_checkpoint["optimizer_state_dict"]
        == test_checkpoint["optimizer_state_dict"]
    )
    assert (
        loaded_checkpoint["scheduler_state_dict"]
        == test_checkpoint["scheduler_state_dict"]
    )
    assert loaded_checkpoint["global_step"] == test_checkpoint["global_step"]

    # Compare tensors using torch.equal
    assert torch.equal(
        loaded_checkpoint["model_state_dict"]["param"],
        test_checkpoint["model_state_dict"]["param"],
    )
    assert torch.equal(
        loaded_checkpoint["momentum"]["param"], test_checkpoint["momentum"]["param"]
    )


async def test_get_timeout(comms_instance):
    # Test get operation timeout
    uid = "0"
    window = 1
    key = "gradient"

    # Mock get_with_retry to always return None
    async def mock_get_with_retry(*args, **kwargs):
        return None

    comms_instance.get_with_retry = MagicMock(side_effect=mock_get_with_retry)

    result = await comms_instance.get_with_retry(
        uid=uid,
        window=window,
        key=key,
        timeout=1,
        local=False,
        stale_retention=10,
    )
    assert result is None


async def test_cleanup_local_data(comms_instance):
    # Test cleanup of local data
    uid = "0"
    current_window = 10
    stale_retention = 5

    # Create dummy directories
    base_dir = os.path.join("/tmp/local_store", uid)

    # Clean up any existing directories first
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # Recursively remove directory contents
                for root, dirs, files in os.walk(item_path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(item_path)
        os.rmdir(base_dir)

    # Create fresh test directories
    os.makedirs(base_dir)
    for w in range(1, 20):
        os.makedirs(os.path.join(base_dir, str(w)), exist_ok=True)

    await comms_instance.cleanup_local_data(uid, current_window, stale_retention)

    # Check that directories older than current_window - stale_retention are deleted
    expected_windows = set(
        str(w) for w in range(current_window - stale_retention, current_window + 10)
    )
    actual_windows = set(os.listdir(base_dir))
    assert expected_windows == actual_windows


# async def test_checkpoint_versioning(comms_instance):
#     # Create two checkpoints with different global steps
#     checkpoint1 = {
#         'model_state_dict': {'param': torch.tensor([1, 2, 3])},
#         'optimizer_state_dict': {'lr': 0.01},
#         'scheduler_state_dict': {'last_epoch': 10},
#         'momentum': {'param': torch.tensor([0.1, 0.1, 0.1])},
#         'global_step': 100
#     }
#     checkpoint2 = {
#         'model_state_dict': {'param': torch.tensor([4, 5, 6])},
#         'optimizer_state_dict': {'lr': 0.02},
#         'scheduler_state_dict': {'last_epoch': 20},
#         'momentum': {'param': torch.tensor([0.2, 0.2, 0.2])},
#         'global_step': 200
#     }

#     # Save both checkpoints
#     await comms_instance.put(
#         state_dict=checkpoint1,
#         uid="0",
#         window=1,
#         key='checkpoint',
#         global_step=100,
#         local=True
#     )
#     await comms_instance.put(
#         state_dict=checkpoint2,
#         uid="0",
#         window=2,
#         key='checkpoint',
#         global_step=200,
#         local=True
#     )

#     # Load checkpoint and verify we get the latest one
#     loaded_checkpoint, global_step = await comms_instance.get(
#         uid="0",
#         window=2,
#         key='checkpoint',
#         local=True
#     )

#     assert global_step == 200
#     assert torch.equal(loaded_checkpoint['model_state_dict']['param'], checkpoint2['model_state_dict']['param'])
