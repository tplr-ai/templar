# ruff: noqa

import os
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import torch
import tempfile
from types import SimpleNamespace
from dotenv import load_dotenv
import pytest_asyncio
import asyncio
import botocore
from dataclasses import dataclass
import time
import numpy as np
from torch.optim import SGD
from torch.optim.lr_scheduler import SequentialLR
from transformers import LlamaForCausalLM
import math


# Set required environment variables
os.environ["R2_GRADIENTS_ACCOUNT_ID"] = "test_account"
os.environ["R2_GRADIENTS_BUCKET_NAME"] = "test-bucket"
os.environ["R2_GRADIENTS_READ_ACCESS_KEY_ID"] = "test_read_key"
os.environ["R2_GRADIENTS_READ_SECRET_ACCESS_KEY"] = "test_read_secret"
os.environ["R2_GRADIENTS_WRITE_ACCESS_KEY_ID"] = "test_write_key"
os.environ["R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY"] = "test_write_secret"
os.environ["R2_DATASET_BUCKET_NAME"] = "test-dataset-bucket"


# Mock Bucket class
@dataclass
class Bucket:
    name: str
    account_id: str
    access_key_id: str
    secret_access_key: str


# Mock the config module
@pytest.fixture(autouse=True)
def mock_config():
    with (
        patch(
            "tplr.config.BUCKET_SECRETS",
            {
                "gradients": {
                    "account_id": "test_account",
                    "bucket_name": "test-bucket",
                    "read": {
                        "access_key_id": "test_read_key",
                        "secret_access_key": "test_read_secret",
                    },
                    "write": {
                        "access_key_id": "test_write_key",
                        "secret_access_key": "test_write_secret",
                    },
                },
                "dataset": {"bucket_name": "test-dataset-bucket"},
            },
        ),
        patch("tplr.config.client_config", {}),
    ):
        yield


from tplr.schemas import Bucket
from tplr.compress import TransformDCT, CompressDCT

# Load environment variables from .env file
load_dotenv()

from tplr.comms import Comms
import tplr
from tplr import logger, debug

debug()

# Setup pytest-asyncio
pytestmark = [pytest.mark.asyncio]


# Test fixture for comms instance
@pytest.fixture
async def comms_instance():
    # Mock wallet
    mock_wallet = MagicMock()
    mock_wallet.hotkey.ss58_address = "test_address"

    # Mock config and other dependencies
    mock_config = MagicMock()
    mock_metagraph = MagicMock()
    mock_hparams = MagicMock()
    mock_hparams.active_check_interval = 60
    mock_hparams.recent_windows = 3

    # Create comms instance with mocked get_own_bucket
    with patch(
        "tplr.comms.Comms.get_own_bucket",
        return_value=Bucket(
            name="test-bucket",
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret",
        ),
    ):
        comms = tplr.comms.Comms(
            wallet=mock_wallet,
            save_location="/tmp",
            key_prefix="test",
            config=mock_config,
            netuid=1,
            metagraph=mock_metagraph,
            hparams=mock_hparams,
            uid="test_uid",
        )

        yield comms

        # Cleanup
        if os.path.exists(comms.temp_dir):
            import shutil

            shutil.rmtree(comms.temp_dir)
        if os.path.exists(comms.save_location):
            shutil.rmtree(comms.save_location)


# Existing mock functions
def mock_bittensor_wallet():
    wallet = MagicMock()
    wallet.hotkey.ss58_address = "test_hotkey_address"
    return wallet


def mock_bittensor_subtensor():
    subtensor = MagicMock()
    subtensor.block = MagicMock(return_value=1000)
    return subtensor


class MockMetagraph:
    """Unified mock metagraph for all tests"""

    def __init__(self):
        self.hotkeys = [f"hotkey{i}" for i in range(10)]
        self.uids = list(range(10))
        self.n = len(self.uids)
        self.S = torch.ones(self.n)  # Stake values
        self.block = 1000
        self.netuid = 1
        self.name = "mock_network"

    def __getattr__(self, name):
        """Handle any unexpected attribute access"""
        tplr.logger.debug(f"Accessing undefined metagraph attribute: {name}")
        return None


@pytest.fixture
def mock_metagraph():
    return MockMetagraph()


@pytest.fixture
async def comms_instance(mock_wallet, mock_metagraph):
    return Comms(
        wallet=mock_wallet,
        save_location="/tmp",
        key_prefix="test",
        config=SimpleNamespace(netuid=1),
        metagraph=mock_metagraph,
        hparams=MockHParams(),
        uid=1,
    )


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


#  TODO: Move to analyser when refactored

# async def test_gather_store_gathers(comms_instance):
#     """Test that gradients are stored when store_gathers=True"""
#     # Setup test data
#     state_dict = {
#         "layer.idxs": torch.tensor([0, 1]),
#         "layer.vals": torch.tensor([0.1, 0.2]),
#     }

#     # Mock methods
#     comms_instance.get_with_retry = AsyncMock()
#     peer_response = (state_dict, 1)
#     comms_instance.get_with_retry.side_effect = [peer_response]
#     comms_instance.s3_put_object = AsyncMock()

#     # Call gather with store_gathers=True
#     await comms_instance.gather(
#         state_dict=None,
#         my_uid="0",
#         uids=["1"],
#         window=1,
#         key="gradient",
#         timeout=5,
#         device="cpu",
#         global_step=0,
#         store_gathers=True,
#     )

#     # Wait a bit for async tasks to be created
#     await asyncio.sleep(0.1)

#     # Verify s3_put_object was called
#     assert comms_instance.s3_put_object.called

#     # Verify correct arguments
#     call_args = comms_instance.s3_put_object.call_args
#     assert call_args is not None
#     kwargs = call_args.kwargs
#     assert kwargs["bucket"] == comms_instance.bucket
#     assert kwargs["key"].startswith("gathers/")
#     assert kwargs["key"].endswith(".npz")


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


async def test_gather_complex_normalization(comms_instance):
    # Test multiple peers with different scales and patterns
    peer1_response = (
        {
            "layer.idxs": torch.tensor([0, 1, 2]),
            "layer.vals": torch.tensor([1.0, 2.0, 2.0]),  # norm ≈ 3
        },
        1,
    )
    peer2_response = (
        {
            "layer.idxs": torch.tensor([0, 1, 2]),
            "layer.vals": torch.tensor(
                [10.0, 20.0, 20.0]
            ),  # Same pattern, larger scale
        },
        2,
    )
    peer3_response = (
        {
            "layer.idxs": torch.tensor([0, 1, 2]),
            "layer.vals": torch.tensor([-5.0, 5.0, 5.0]),  # Different sign
        },
        3,
    )

    comms_instance.get_with_retry = AsyncMock()
    comms_instance.get_with_retry.side_effect = [
        peer1_response,
        peer2_response,
        peer3_response,
    ]

    result = await comms_instance.gather(
        state_dict=None,
        my_uid="0",
        uids=["1", "2", "3"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        global_step=0,
    )

    assert result is not None
    # Fix: Get all normalized tensors and average them
    normalized_tensors = getattr(result.state_dict, "layer.vals")
    actual_vals = torch.stack(normalized_tensors).mean(dim=0)

    # Calculate expected normalized values
    norm1 = torch.norm(peer1_response[0]["layer.vals"])
    norm2 = torch.norm(peer2_response[0]["layer.vals"])
    norm3 = torch.norm(peer3_response[0]["layer.vals"])

    # Add small epsilon to avoid division by zero
    eps = 1e-8
    normalized1 = peer1_response[0]["layer.vals"] / (norm1 + eps)
    normalized2 = peer2_response[0]["layer.vals"] / (norm2 + eps)
    normalized3 = peer3_response[0]["layer.vals"] / (norm3 + eps)

    # Average all three normalized values
    expected_vals = torch.stack([normalized1, normalized2, normalized3]).mean(dim=0)

    # Print values for debugging
    print(f"\nPeer 1 original: {peer1_response[0]['layer.vals']}, norm: {norm1}")
    print(f"Peer 1 normalized: {normalized1}")
    print(f"\nPeer 2 original: {peer2_response[0]['layer.vals']}, norm: {norm2}")
    print(f"Peer 2 normalized: {normalized2}")
    print(f"\nPeer 3 original: {peer3_response[0]['layer.vals']}, norm: {norm3}")
    print(f"Peer 3 normalized: {normalized3}")
    print(f"\nExpected average: {expected_vals}")
    print(f"Actual result: {actual_vals}")

    # Compare with higher tolerance due to floating point operations
    assert torch.allclose(actual_vals, expected_vals, rtol=1e-3, atol=1e-3)

    # Additional assertions to verify all peers were processed
    assert len(normalized_tensors) == 3, (
        f"Expected 3 normalized tensors, got {len(normalized_tensors)}"
    )
    assert len(result.uids) == 3, f"Expected 3 valid UIDs, got {len(result.uids)}"


# Test Initialization and Cleanup
async def test_comms_init(comms_instance):
    """Test proper initialization of Comms instance"""
    assert os.path.exists(comms_instance.temp_dir)
    assert os.path.exists(comms_instance.save_location)
    assert comms_instance.lock is not None
    assert isinstance(comms_instance.active_peers, set)


async def test_cleanup_local_data(comms_instance):
    """Test cleanup of stale local data"""
    # Setup test directories and files
    uid = "test_uid"
    test_dir = os.path.join("/tmp/local_store", uid)
    os.makedirs(os.path.join(test_dir, "10"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "20"), exist_ok=True)

    await comms_instance.cleanup_local_data(uid, 25, 5)
    assert not os.path.exists(os.path.join(test_dir, "10"))
    assert os.path.exists(os.path.join(test_dir, "20"))


# Test S3 Operations
async def test_s3_put_small_file(comms_instance):
    """Test uploading small file to S3"""
    # Create test file
    with open("test_file.txt", "w") as f:
        f.write("test data")

    # Mock S3 client with proper async context manager
    mock_client = AsyncMock()
    mock_client.put_object = AsyncMock()
    comms_instance.session.create_client = MagicMock(return_value=mock_client)

    # Create proper Bucket instance instead of string
    comms_instance.bucket = Bucket(
        name="test-bucket",
        account_id="test-account",
        access_key_id="test-key",
        secret_access_key="test-secret",
    )

    await comms_instance.s3_put_object("test_key", "test_file.txt")

    # Cleanup
    os.remove("test_file.txt")


@pytest.mark.asyncio
async def test_s3_put_large_file(comms_instance):
    """Test multipart upload for large files with batching and retries"""
    mock_client = AsyncMock()
    mock_client.create_multipart_upload = AsyncMock(
        return_value={"UploadId": "test_id"}
    )
    mock_client.upload_part = AsyncMock(return_value={"ETag": "test_etag"})
    mock_client.complete_multipart_upload = AsyncMock()
    mock_client.abort_multipart_upload = AsyncMock()
    comms_instance.session.create_client = MagicMock(return_value=mock_client)

    comms_instance.bucket = Bucket(
        name="test-bucket",
        account_id="test-account",
        access_key_id="test-key",
        secret_access_key="test-secret",
    )

    with open("large_file.txt", "wb") as f:
        f.write(os.urandom(100 * 1024 * 1024))

    await comms_instance.s3_put_object("test_key", "large_file.txt")

    upload_part_calls = mock_client.upload_part.call_args_list
    assert len(upload_part_calls) <= 20

    part_numbers = [call.kwargs["PartNumber"] for call in upload_part_calls]
    assert part_numbers == sorted(part_numbers)

    os.remove("large_file.txt")


async def test_download_large_file(comms_instance):
    """Test downloading large file with chunks"""
    # Mock S3 client with proper responses
    mock_client = AsyncMock()
    mock_client.head_object = AsyncMock(
        return_value={"ContentLength": 10 * 1024 * 1024}
    )

    # Mock get_object to return proper chunk data
    async def mock_get_object(**kwargs):
        range_header = kwargs.get("Range", "")
        start, end = map(int, range_header.replace("bytes=", "").split("-"))
        chunk_size = end - start + 1
        return {
            "Body": AsyncMock(
                **{
                    "__aenter__.return_value": AsyncMock(
                        **{"read.return_value": os.urandom(chunk_size)}
                    )
                }
            )
        }

    mock_client.get_object = AsyncMock(side_effect=mock_get_object)
    comms_instance.session.create_client = MagicMock(return_value=mock_client)

    success = await comms_instance.download_large_file(
        mock_client, "test-bucket", "test_key", 10 * 1024 * 1024, "test_output.txt"
    )
    mock_client.get_object.assert_called()


# Test Checkpoint Operations
@pytest.mark.asyncio
async def test_load_checkpoint_success(comms_instance):
    """Test successful checkpoint loading"""
    # Create mock model and parameters
    model = MagicMock()
    test_param = torch.nn.Parameter(torch.randn(10))
    model.named_parameters.return_value = [("layer1", test_param)]

    # Create mock optimizer and scheduler
    optimizer = MagicMock()
    optimizer.state = {}  # Add empty state dict
    scheduler = MagicMock()
    scheduler.last_epoch = 0  # Add last_epoch attribute
    transformer = MagicMock()
    compressor = MagicMock()

    # Mock checkpoint data with all required fields
    checkpoint_data = {
        "model_state_dict": {"layer1": torch.randn(10)},
        "optimizer_state_dict": {
            "state": {0: {"step": 100}},
            "param_groups": [{"lr": 0.001}],  # Add param_groups
        },
        "scheduler_state_dict": {"last_epoch": 0},
        "momentum": {"layer1": torch.randn(10)},
        "global_step": 100,
        "start_window": 1,
        "current_window": 5,
    }

    # Mock get_latest_checkpoint result
    comms_instance.get_latest_checkpoint = AsyncMock(
        return_value=(checkpoint_data, 5)  # Return tuple of (data, window)
    )

    # Mock model's load_state_dict
    model.load_state_dict = MagicMock()

    # Mock optimizer and scheduler load_state_dict
    optimizer.load_state_dict = MagicMock()
    scheduler.load_state_dict = MagicMock()

    # Mock gather for catch-up phase
    comms_instance.gather = AsyncMock(
        return_value=SimpleNamespace(
            state_dict=SimpleNamespace(
                **{
                    "layer1idxs": [torch.tensor([0, 1])],
                    "layer1vals": [torch.tensor([0.1, 0.2])],
                }
            ),
            uids=["1"],
            global_steps=[100],
        )
    )

    # Add shape information to transformer mock
    transformer.shapes = {"layer1": torch.Size([10])}
    transformer.totalks = {"layer1": 10}
    transformer.decode.return_value = torch.randn(10)

    # Add debug prints
    print("\nBefore loading checkpoint...")

    with (
        patch("tplr.logger.error") as mock_error,
        patch("tplr.logger.info") as mock_info,
        patch("tplr.logger.debug") as mock_debug,
        patch("tplr.logger.warning") as mock_warning,
    ):
        success, momentum, step, opt, sched = await comms_instance.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            transformer=transformer,
            compressor=compressor,
            current_window=10,
            device="cpu",
            peers=[1, 2],
            uid="0",
        )

        # Print any error logs that occurred
        print("\nError logs:")
        for call in mock_error.call_args_list:
            print(f"Error: {call.args[0]}")

        print("\nWarning logs:")
        for call in mock_warning.call_args_list:
            print(f"Warning: {call.args[0]}")

        print(f"\nSuccess: {success}")
        print(f"Step: {step}")

    assert success, "Checkpoint loading failed"
    assert isinstance(momentum, dict)
    assert "layer1" in momentum
    assert step > 0
    assert opt == optimizer
    assert sched == scheduler


@pytest.mark.asyncio
async def test_load_checkpoint_missing_data(comms_instance):
    """Test loading checkpoint when data is missing"""
    # Mock the get_latest_checkpoint method to return None without error
    comms_instance.get_latest_checkpoint = AsyncMock(return_value=None)

    # Mock get_validator_with_highest_stake to avoid bucket access
    comms_instance.get_validator_with_highest_stake = AsyncMock(return_value=(0, 1.0))

    # Create mock model and optimizer
    mock_model = MagicMock()
    mock_optimizer = MagicMock()
    mock_scheduler = MagicMock()

    # The function returns 5 values: success, momentum, global_step, optimizer, scheduler
    (
        success,
        momentum,
        global_step,
        optimizer,
        scheduler,
    ) = await comms_instance.load_checkpoint(
        model=mock_model,
        optimizer=mock_optimizer,
        scheduler=mock_scheduler,
        transformer=MagicMock(),
        compressor=MagicMock(),
        current_window=1,
        device="cpu",
        peers=[1, 2, 3],
        uid="test_uid",
    )

    assert not success
    assert momentum == {}
    assert global_step == 0
    assert (
        optimizer == mock_optimizer
    )  # Check it returns the same optimizer we passed in
    assert (
        scheduler == mock_scheduler
    )  # Check it returns the same scheduler we passed in


# Test Gather Operations
async def test_gather_basic(comms_instance):
    """Test basic gather functionality"""
    state_dict = {
        "layer.idxs": torch.tensor([0, 1]),
        "layer.vals": torch.tensor([0.1, 0.2]),
    }
    comms_instance.get_with_retry = AsyncMock(return_value=(state_dict, 1))

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
    assert result is not None
    assert hasattr(result.state_dict, "layer.vals")


async def test_gather_timeout(comms_instance):
    """Test gather operation with timeout"""

    async def slow_get(*args, **kwargs):
        await asyncio.sleep(2)
        return None

    comms_instance.get_with_retry = AsyncMock(side_effect=Exception("Test error"))

    # Mock logger to avoid actual logging
    with (
        patch("tplr.logger.error"),
        patch("tplr.logger.debug"),
        patch("tplr.logger.info"),
        patch("tplr.logger.warning"),
    ):
        result = await comms_instance.gather(
            state_dict=None,
            my_uid="0",
            uids=["1"],
            window=1,
            key="gradient",
            timeout=5,
            device="cpu",
            local=True,  # Use local=True to avoid S3 operations
            global_step=0,
            stale_retention=10,
        )

        # Should return None on error
        assert result is None


async def test_gather_timeout(comms_instance):
    """Test gather operation with timeout"""

    # Mock get_with_retry to simulate timeout
    async def mock_get_with_retry(*args, **kwargs):
        await asyncio.sleep(0.2)  # Sleep longer than timeout
        return None

    comms_instance.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

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
    assert result is None


# Test Start Window Operations
async def test_get_start_window(comms_instance):
    """Test fetching start window"""
    mock_bucket = MagicMock()
    comms_instance._get_highest_stake_validator_bucket = AsyncMock(
        return_value=(mock_bucket, "1")
    )
    comms_instance.s3_get_object = AsyncMock(return_value={"start_window": 100})

    start_window = await comms_instance.get_start_window()
    assert start_window == 100


async def test_get_start_window_retry(comms_instance):
    """Test start window fetch with retries"""
    mock_bucket = MagicMock()
    comms_instance._get_highest_stake_validator_bucket = AsyncMock(
        return_value=(mock_bucket, "1")
    )
    comms_instance.s3_get_object = AsyncMock(
        side_effect=[None, None, {"start_window": 100}]
    )

    start_window = await comms_instance.get_start_window()
    assert start_window == 100


# async def test_gather_store_gathers_non_blocking(comms_instance):
#     """Test that storing gradients doesn't block the gather operation"""
#     # Setup test data
#     state_dict = {
#         "layer.idxs": torch.tensor([0, 1]),
#         "layer.vals": torch.tensor([0.1, 0.2]),
#     }

#     # Mock methods
#     comms_instance.get_with_retry = AsyncMock()
#     peer_response = (state_dict, 1)
#     comms_instance.get_with_retry.side_effect = [peer_response]

#     # Mock s3_put_object to simulate slow upload
#     async def slow_upload(*args, **kwargs):
#         await asyncio.sleep(1)  # Simulate slow upload
#         return True

#     comms_instance.s3_put_object = AsyncMock(side_effect=slow_upload)

#     # Measure time taken
#     start_time = time.perf_counter()

#     await comms_instance.gather(
#         state_dict=None,
#         my_uid="0",
#         uids=["1"],
#         window=1,
#         key="gradient",
#         timeout=5,
#         device="cpu",
#         global_step=0,
#         store_gathers=True,
#     )

#     # Wait a bit for async tasks to be created
#     await asyncio.sleep(0.1)

#     end_time = time.perf_counter()
#     duration = end_time - start_time

#     # Verify gather completed quickly (much less than 1 second)
#     assert duration < 0.5, f"Gather took {duration:.2f}s, should be near-instant"

#     # Verify upload was initiated
#     assert comms_instance.s3_put_object.called


# async def test_store_gradient_data_success(comms_instance):
#     """Test successful gradient data storage"""
#     # Setup test data
#     uid = "1"
#     window = 10
#     global_step = 5
#     state_dict_resp = {
#         "layer1.weight": torch.tensor([1.0, 2.0, 3.0]),
#         "layer1.bias": torch.tensor([0.1, 0.2]),
#     }
#     global_step_resp = 5

#     # Mock s3_put_object
#     comms_instance.s3_put_object = AsyncMock()

#     # Call the function
#     await comms_instance._store_gradient_data(
#         uid=uid,
#         window=window,
#         global_step=global_step,
#         state_dict_resp=state_dict_resp,
#         global_step_resp=global_step_resp,
#     )

#     # Wait for all pending tasks to complete
#     tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
#     await asyncio.gather(*tasks)

#     # Additional wait for file cleanup
#     await asyncio.sleep(1.2)  # Slightly longer than cleanup delay

#     # Verify s3_put_object was called with correct parameters
#     assert comms_instance.s3_put_object.called
#     call_args = comms_instance.s3_put_object.call_args
#     assert call_args is not None

#     # Verify the key format
#     expected_key = f"gathers/{tplr.__version__}/{uid}/{window}/{global_step}.npz"
#     assert call_args.kwargs["key"] == expected_key

#     # Verify the file was created and then cleaned up
#     temp_file_pattern = f"gradient_{uid}_{window}_{global_step}.npz"
#     assert not any(temp_file_pattern in f for f in os.listdir(comms_instance.temp_dir))


# async def test_store_gradient_data_error_handling(comms_instance):
#     """Test error handling in gradient data storage"""
#     # Setup test data with invalid tensor to trigger error
#     uid = "1"
#     window = 10
#     global_step = 5
#     state_dict_resp = {
#         "layer1.weight": "invalid_tensor",  # This will cause an error
#     }
#     global_step_resp = 5

#     # Mock logger
#     with patch("tplr.logger.warning") as mock_warning:
#         await comms_instance._store_gradient_data(
#             uid=uid,
#             window=window,
#             global_step=global_step,
#             state_dict_resp=state_dict_resp,
#             global_step_resp=global_step_resp,
#         )

#         # Verify warning was logged
#         mock_warning.assert_called_once()
#         assert "Failed to store gradient" in mock_warning.call_args[0][0]


async def test_cleanup_temp_file(comms_instance):
    """Test temporary file cleanup"""
    # Create a test file
    test_file = os.path.join(comms_instance.temp_dir, "test_temp_file.npz")
    with open(test_file, "w") as f:
        f.write("test")

    # Call cleanup
    await comms_instance._cleanup_temp_file(test_file)

    # Wait for async cleanup
    await asyncio.sleep(1.1)  # Slightly longer than the sleep in cleanup

    # Verify file was removed
    assert not os.path.exists(test_file)


async def test_cleanup_temp_file_nonexistent(comms_instance):
    """Test cleanup with non-existent file"""
    nonexistent_file = os.path.join(comms_instance.temp_dir, "nonexistent.npz")

    # Mock logger
    with patch("tplr.logger.warning") as mock_warning:
        await comms_instance._cleanup_temp_file(nonexistent_file)

        # Verify no warning was logged (clean exit)
        mock_warning.assert_not_called()


# async def test_gather_with_store_gathers(comms_instance):
#     """Test gather operation with store_gathers enabled"""
#     # Setup test data
#     state_dict = {
#         "layer1.weightsidxs": torch.tensor([0, 1, 2]),
#         "layer1.weightsvals": torch.tensor([0.1, 0.2, 0.3]),
#     }

#     # Mock get_with_retry
#     peer_response = (state_dict, 1)
#     comms_instance.get_with_retry = AsyncMock(return_value=peer_response)

#     # Mock s3_put_object
#     comms_instance.s3_put_object = AsyncMock()

#     # Call gather with store_gathers=True
#     result = await comms_instance.gather(
#         state_dict=state_dict,
#         my_uid="0",
#         uids=["1"],
#         window=1,
#         key="gradient",
#         timeout=5,
#         device="cpu",
#         global_step=0,
#         local=True,
#         store_gathers=True,
#     )

#     # Wait for async tasks
#     await asyncio.sleep(0.1)

#     # Verify s3_put_object was called
#     assert comms_instance.s3_put_object.called

#     # Verify the key format in the call
#     call_args = comms_instance.s3_put_object.call_args
#     assert call_args is not None
#     assert call_args.kwargs["key"].startswith(f"gathers/{tplr.__version__}/")
#     assert call_args.kwargs["key"].endswith(".npz")


# async def test_gather_concurrent_store_gathers(comms_instance):
#     """Test concurrent gradient storage during gather"""
#     # Setup multiple peer responses
#     state_dict = {
#         "layer1.weightsidxs": torch.tensor([0, 1, 2]),
#         "layer1.weightsvals": torch.tensor([0.1, 0.2, 0.3]),
#     }

#     peer_responses = [(state_dict, i) for i in range(3)]
#     comms_instance.get_with_retry = AsyncMock(side_effect=peer_responses)

#     # Mock s3_put_object with delay to simulate upload
#     async def delayed_upload(*args, **kwargs):
#         await asyncio.sleep(0.1)
#         return True

#     comms_instance.s3_put_object = AsyncMock(side_effect=delayed_upload)

#     # Clean temp directory before test
#     if os.path.exists(comms_instance.temp_dir):
#         for f in os.listdir(comms_instance.temp_dir):
#             try:
#                 os.remove(os.path.join(comms_instance.temp_dir, f))
#             except Exception:
#                 pass

#     # Call gather with store_gathers=True
#     result = await comms_instance.gather(
#         state_dict=state_dict,
#         my_uid="0",
#         uids=["1", "2", "3"],
#         window=1,
#         key="gradient",
#         timeout=5,
#         device="cpu",
#         global_step=0,
#         local=True,
#         store_gathers=True,
#     )

#     # Wait for all pending tasks to complete
#     tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
#     await asyncio.gather(*tasks)

#     # Additional wait for file cleanup
#     await asyncio.sleep(2)  # Increased wait time

#     # Verify s3_put_object was called multiple times
#     assert comms_instance.s3_put_object.call_count == 3

#     # Verify all calls had correct key format
#     for call in comms_instance.s3_put_object.call_args_list:
#         assert call.kwargs["key"].startswith(f"gathers/{tplr.__version__}/")
#         assert call.kwargs["key"].endswith(".npz")

#     # Force cleanup of temp files
#     if os.path.exists(comms_instance.temp_dir):
#         for f in os.listdir(comms_instance.temp_dir):
#             try:
#                 os.remove(os.path.join(comms_instance.temp_dir, f))
#             except Exception:
#                 pass

#     # Verify temp directory is clean
#     assert len(os.listdir(comms_instance.temp_dir)) == 0


class TestCommsGradientOperations:
    """Test gradient application and gathering operations"""

    async def test_apply_gathered_gradients_empty_result(self):
        """Should handle empty gather results gracefully"""
        comms_instance = await setup_test_comms()
        model = setup_test_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = setup_test_scheduler(optimizer)
        transformer = MockTransformer()
        compressor = MockCompressor()

        success, new_global_step = await comms_instance._apply_gathered_gradients(
            gather_result=None,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            transformer=transformer,
            compressor=compressor,
            device="cpu",
            window=1,
            global_step=0,
        )

        assert not success
        assert new_global_step == 0

    async def test_apply_gathered_gradients_missing_params(self):
        """Should handle missing parameters gracefully by skipping them"""
        comms_instance = await setup_test_comms()
        model = setup_test_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = setup_test_scheduler(optimizer)

        class TestTransformer:
            def __init__(self):
                self.shapes = {}
                self.totalks = {}
                for name, param in model.named_parameters():
                    self.shapes[name] = param.shape
                    self.totalks[name] = param.numel()

            def decode(self, x):
                return x

        transformer = TestTransformer()
        compressor = MockCompressor()

        # Create gather result with partial parameter data
        gather_result = SimpleNamespace(state_dict=SimpleNamespace(), global_steps=[1])

        # Only add gradient data for weight, leaving bias missing
        weight_param_name = "0.weight"  # First layer's weight in sequential model
        param = next(p for n, p in model.named_parameters() if n == weight_param_name)
        setattr(
            gather_result.state_dict,
            f"{weight_param_name}idxs",
            torch.zeros(param.numel(), dtype=torch.long),
        )
        setattr(
            gather_result.state_dict,
            f"{weight_param_name}vals",
            torch.ones(param.numel()) * 0.1,
        )

        # Store initial parameters
        initial_params = {n: p.clone() for n, p in model.named_parameters()}

        success, new_global_step = await comms_instance._apply_gathered_gradients(
            gather_result=gather_result,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            transformer=transformer,
            compressor=compressor,
            device="cpu",
            window=1,
            global_step=0,
        )

        # Verify behavior:
        # 1. Operation should succeed even with missing parameters
        assert success
        assert new_global_step == 1

        # 2. Weight parameter should be updated
        assert not torch.equal(
            next(p for n, p in model.named_parameters() if n == weight_param_name),
            initial_params[weight_param_name],
        ), "Weight parameter should have been updated"

        # 3. Bias parameter should remain unchanged
        bias_param_name = "0.bias"  # First layer's bias
        assert torch.equal(
            next(p for n, p in model.named_parameters() if n == bias_param_name),
            initial_params[bias_param_name],
        ), "Bias parameter should not have been updated"

    async def test_apply_gathered_gradients_device_mismatch(self):
        """Should handle tensor device mismatches"""
        comms_instance = await setup_test_comms()
        model = setup_test_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = setup_test_scheduler(optimizer)
        transformer = MockTransformer()
        compressor = MockCompressor()

        # Create gather result with tensors on CPU
        gather_result = create_mock_gather_result(model, "cpu")

        success, new_global_step = await comms_instance._apply_gathered_gradients(
            gather_result=gather_result,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            transformer=transformer,
            compressor=compressor,
            device="cpu",
            window=1,
            global_step=0,
        )

        assert success
        assert new_global_step == 1

    async def test_apply_gathered_gradients_success(self):
        """Should successfully apply gathered gradients"""
        comms_instance = await setup_test_comms()
        model = setup_test_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = setup_test_scheduler(optimizer)

        class RealTransformer:
            def __init__(self):
                self.shapes = {}  # Required by validator.py
                self.totalks = {}  # Add totalks dict
                for name, param in model.named_parameters():
                    self.shapes[name] = param.shape  # Store by name instead of shape
                    self.totalks[name] = param.numel()  # Store total elements

            def decode(self, x):
                return torch.ones_like(x) * 0.1  # Return non-zero gradient

        class RealCompressor:
            def batch_decompress(self, p, idxs, vals, xshape, totalk):
                # Match validator.py behavior
                if not isinstance(idxs, list):
                    idxs = [idxs]
                if not isinstance(vals, list):
                    vals = [vals]
                return torch.ones_like(p) * 0.1

            def decompress(self, p, idxs, vals, xshape, totalk):
                return torch.ones_like(p) * 0.1

        transformer = RealTransformer()
        compressor = RealCompressor()

        # Store initial parameters
        initial_params = {n: p.clone() for n, p in model.named_parameters()}

        # Create gather result with actual gradients
        gather_result = SimpleNamespace(
            state_dict=SimpleNamespace(), global_steps=[1], uids=[0]
        )

        # Add compressed gradient info for each parameter exactly as validator expects
        for name, param in model.named_parameters():
            # Add both idxs and vals with correct shapes
            setattr(
                gather_result.state_dict,
                f"{name}idxs",
                torch.zeros(param.numel(), dtype=torch.long),
            )
            setattr(
                gather_result.state_dict, f"{name}vals", torch.ones(param.numel()) * 0.1
            )

        success, new_global_step = await comms_instance._apply_gathered_gradients(
            gather_result=gather_result,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            transformer=transformer,
            compressor=compressor,
            device="cpu",
            window=1,
            global_step=0,
        )

        assert success
        assert new_global_step == 1

        # Verify parameters were actually updated
        for name, param in model.named_parameters():
            assert not torch.equal(param, initial_params[name])


class MockTransformer:
    def encode(self, tensor):
        # Return a modified version of input tensor
        return tensor + 1.0

    def decode(self, tensor):
        # Return a modified version to ensure parameter updates
        return tensor + 1.0


class MockCompressor:
    def compress(self, tensor, topk):
        # Return mock compression values that will cause parameter updates
        return [0], [1.0], tensor.shape, 1

    def decompress(self, p, idxs, vals, xshape, totalk):
        # Return tensor that will modify parameters
        return torch.ones_like(p) * 0.1

    def batch_decompress(self, p, idxs, vals, xshape, totalk):
        # Return tensor that will modify parameters
        return torch.ones_like(p) * 0.1


class MockWallet:
    def __init__(self):
        self.hotkey = SimpleNamespace(ss58_address="test_address")


class MockHParams:
    def __init__(self):
        self.blocks_per_window = 100
        self.target_chunk = 512
        self.topk_compression = 0.1
        self.catch_up_threshold = 5
        self.catch_up_min_peers = 1
        self.catch_up_batch_size = 10
        self.catch_up_timeout = 300
        self.active_check_interval = 60
        self.recent_windows = 5


def create_mock_gather_result(model, device, wrong_shape=False):
    """Create a mock gather result matching exact parameter structure"""
    state_dict = SimpleNamespace()

    # Print actual parameter names for debugging
    print("Model parameters:", [name for name, _ in model.named_parameters()])

    for name, param in model.named_parameters():
        if wrong_shape:
            shape = (5, 5)
        else:
            shape = param.shape

        # Create tensors matching parameter size
        idxs = torch.arange(param.numel(), dtype=torch.long, device=device)
        vals = torch.ones(param.numel(), device=device)

        # Use exact parameter names
        base_name = name.replace(".", "_")  # Convert '0.weight' to '0_weight'
        setattr(state_dict, f"{base_name}idxs", idxs)
        setattr(state_dict, f"{base_name}vals", vals)
        setattr(state_dict, f"{base_name}shape", shape)
        setattr(state_dict, f"{base_name}totalk", param.numel())

    return SimpleNamespace(state_dict=state_dict, global_steps=[1])


async def setup_test_comms():
    mock_wallet = MockWallet()
    mock_hparams = MockHParams()
    mock_metagraph = MockMetagraph()
    comms_instance = Comms(
        wallet=mock_wallet,
        save_location="/tmp",
        hparams=mock_hparams,
        config=SimpleNamespace(netuid=1),
        metagraph=mock_metagraph,
        uid=0,
    )
    # For testing, override the endpoint to avoid using the R2 endpoint.
    comms_instance.get_base_url = lambda account_id: "http://localhost:4566"
    return comms_instance


def setup_test_model():
    """Create a simple test model with predictable parameter names"""
    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    # The model will have parameter named '0.weight' and '0.bias'
    return model


def setup_test_scheduler(optimizer):
    """Create a simple test scheduler"""
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)


# Setup pytest fixtures
@pytest.fixture(scope="function")
async def comms_instance():
    return await setup_test_comms()


@pytest.fixture(scope="function")
def model():
    return setup_test_model()


@pytest.fixture(scope="function")
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)


@pytest.fixture(scope="function")
def scheduler(optimizer):
    return setup_test_scheduler(optimizer)


def _get_prime_divisors(n):
    divisors = []
    while n % 2 == 0:
        divisors.append(2)
        n //= 2
    while n % 3 == 0:
        divisors.append(3)
        n //= 3
    i = 5
    while i * i <= n:
        for k in (i, i + 2):
            while n % k == 0:
                divisors.append(k)
                n //= k
        i += 6
    if n > 1:
        divisors.append(n)
    return divisors


def _get_divisors(n):
    divisors = []
    if n == 1:
        divisors.append(1)
    elif n > 1:
        prime_factors = _get_prime_divisors(n)
        divisors = [1]
        last_prime = 0
        factor = 0
        slice_len = 0
        for prime in prime_factors:
            if last_prime != prime:
                slice_len = len(divisors)
                factor = prime
            else:
                factor *= prime
            for i in range(slice_len):
                divisors.append(divisors[i] * factor)
            last_prime = prime
        divisors.sort()
    return divisors


def _get_smaller_split(n, close_to):
    all_divisors = _get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if val == close_to:
            return val
        if val > close_to:
            if ix == 0:
                return val
            return all_divisors[ix - 1]
    return n


class TestGatherWindowBatch:
    @pytest.mark.asyncio
    async def test_gather_window_batch_success(self, comms_instance):
        """Test successful batch gathering"""
        # Mock gather responses
        mock_responses = [
            SimpleNamespace(state_dict={"param1": torch.randn(10)}),
            SimpleNamespace(state_dict={"param1": torch.randn(10)}),
        ]
        comms_instance.gather = AsyncMock(side_effect=mock_responses)

        result = await comms_instance._gather_window_batch(
            batch_windows=[1, 2],
            uid="test_uid",
            peers=[1, 2, 3],
            device="cpu",
            global_step=10,
        )

        assert len(result) == 2
        assert all(w in result for w in [1, 2])
        assert all(isinstance(r, SimpleNamespace) for r in result.values())

    @pytest.mark.asyncio
    async def test_gather_window_batch_partial_failure(self, comms_instance):
        """Test when some gathers fail"""

        # Create a mock gather function that returns success for window 1 and None for window 2
        async def mock_gather(*args, **kwargs):
            window = kwargs.get("window")
            if window == 1:
                return SimpleNamespace(state_dict={"param1": torch.randn(10)})
            return None

        comms_instance.gather = AsyncMock(side_effect=mock_gather)

        result = await comms_instance._gather_window_batch(
            batch_windows=[1, 2],
            uid="test_uid",
            peers=[1, 2, 3],
            device="cpu",
            global_step=10,
        )

        # Verify both windows are in result dict
        assert 1 in result
        assert 2 in result
        assert result[1] is not None  # First window should have data
        assert result[2] is None  # Second window should be None
        assert len(result) == 2  # Should have entries for both windows

    @pytest.mark.asyncio
    async def test_gather_window_batch_exception(self, comms_instance):
        """Test exception handling"""
        comms_instance.gather = AsyncMock(side_effect=Exception("Network error"))

        result = await comms_instance._gather_window_batch(
            batch_windows=[1, 2],
            uid="test_uid",
            peers=[1, 2, 3],
            device="cpu",
            global_step=10,
        )

        # The implementation returns a dict with None values for all windows on failure
        assert result == {1: None, 2: None}  # Updated assertion
        assert len(result) == 2
        assert all(v is None for v in result.values())


class TestApplyGatheredGradients:
    @pytest.fixture
    def mock_model(self):
        model = MagicMock(spec=LlamaForCausalLM)
        model.named_parameters.return_value = [
            ("param1", torch.nn.Parameter(torch.randn(10))),
            ("param2", torch.nn.Parameter(torch.randn(10))),
        ]
        return model

    @pytest.fixture
    def mock_transformer(self):
        transformer = MagicMock(spec=TransformDCT)
        transformer.shapes = {"param1": (10,), "param2": (10,)}
        transformer.totalks = {"param1": 5, "param2": 5}
        transformer.decode.return_value = torch.randn(10)
        return transformer

    @pytest.fixture
    def mock_compressor(self):
        compressor = MagicMock(spec=CompressDCT)
        compressor.batch_decompress.return_value = torch.randn(10)
        return compressor

    @pytest.mark.asyncio
    async def test_apply_gathered_gradients_success(
        self, comms_instance, mock_model, mock_transformer, mock_compressor
    ):
        """Test successful gradient application"""
        optimizer = MagicMock(spec=SGD)
        scheduler = MagicMock(spec=SequentialLR)

        gather_result = SimpleNamespace(
            state_dict=SimpleNamespace(
                param1idxs=[0, 1],
                param1vals=torch.randn(2),
                param2idxs=[0, 1],
                param2vals=torch.randn(2),
            )
        )

        success, new_step = await comms_instance._apply_gathered_gradients(
            gather_result=gather_result,
            model=mock_model,
            optimizer=optimizer,
            scheduler=scheduler,
            transformer=mock_transformer,
            compressor=mock_compressor,
            device="cpu",
            window=1,
            global_step=10,
        )

        assert success
        assert new_step == 11
        optimizer.step.assert_called_once()
        scheduler.step.assert_called_once()

    # @pytest.mark.asyncio
    # async def test_apply_gathered_gradients_invalid_data(self, comms_instance, mock_model, mock_transformer, mock_compressor):
    #     """Test handling of invalid gradient data"""
    #     optimizer = MagicMock(spec=SGD)
    #     scheduler = MagicMock(spec=SequentialLR)

    #     gather_result = SimpleNamespace(state_dict=SimpleNamespace())

    #     success, step = await comms_instance._apply_gathered_gradients(
    #         gather_result=gather_result,
    #         model=mock_model,
    #         optimizer=optimizer,
    #         scheduler=scheduler,
    #         transformer=mock_transformer,
    #         compressor=mock_compressor,
    #         device="cpu",
    #         window=1,
    #         global_step=10
    #     )

    #     assert not success
    #     assert step == 10
    #     optimizer.step.assert_not_called()
    #     scheduler.step.assert_not_called()


class TestCheckAndPerformCatchUp:
    @pytest.fixture
    def mock_hparams(self):
        return SimpleNamespace(
            catch_up_threshold=2,
            catch_up_min_peers=3,
            catch_up_batch_size=5,
            catch_up_timeout=300,
        )

    @pytest.mark.asyncio
    async def test_check_and_perform_catch_up_below_threshold(
        self, comms_instance, mock_hparams
    ):
        """Test when catch-up is not needed due to small window gap"""
        result = await comms_instance.check_and_perform_catch_up(
            model=MagicMock(),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            transformer=MagicMock(),
            compressor=MagicMock(),
            current_window=11,
            sync_window=10,
            device="cpu",
            peers=[1, 2, 3, 4],
            uid="test_uid",
            global_step=10,
            hparams=mock_hparams,
        )

        assert not result[0]
        assert result[1] == 10

    @pytest.mark.asyncio
    async def test_check_and_perform_catch_up_timeout(
        self, comms_instance, mock_hparams
    ):
        """Test catch-up timeout handling"""
        mock_hparams.catch_up_timeout = 0.1  # Short timeout

        async def slow_gather(*args, **kwargs):
            await asyncio.sleep(0.2)  # Sleep longer than timeout
            return {}

        comms_instance._gather_window_batch = AsyncMock(side_effect=slow_gather)

        result = await comms_instance.check_and_perform_catch_up(
            model=MagicMock(),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            transformer=MagicMock(),
            compressor=MagicMock(),
            current_window=20,
            sync_window=10,
            device="cpu",
            peers=[1, 2, 3, 4],
            uid="test_uid",
            global_step=10,
            hparams=mock_hparams,
        )

        assert not result[0]  # Should fail due to timeout
        assert result[1] == 10  # Global step should remain unchanged

    @pytest.mark.asyncio
    async def test_check_and_perform_catch_up_with_sleep(
        self, comms_instance, mock_hparams
    ):
        """Test catch-up with proper sleep handling"""
        mock_hparams.catch_up_timeout = 1.0  # Longer timeout

        async def gather_with_sleep(*args, **kwargs):
            await asyncio.sleep(0.1)  # Short sleep that should complete
            return {
                11: SimpleNamespace(state_dict={"param1": torch.randn(10)}),
                12: SimpleNamespace(state_dict={"param1": torch.randn(10)}),
            }

        comms_instance._gather_window_batch = AsyncMock(side_effect=gather_with_sleep)
        comms_instance._apply_gathered_gradients = AsyncMock(return_value=(True, 11))

        result = await comms_instance.check_and_perform_catch_up(
            model=MagicMock(),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            transformer=MagicMock(),
            compressor=MagicMock(),
            current_window=15,
            sync_window=10,
            device="cpu",
            peers=[1, 2, 3, 4],
            uid="test_uid",
            global_step=10,
            hparams=mock_hparams,
        )

        assert result[0]  # Should succeed
        assert result[1] > 10  # Global step should increase
        assert comms_instance._gather_window_batch.called
        assert comms_instance._apply_gathered_gradients.called


# New tests for upload_large_file retry behavior
@pytest.mark.asyncio
async def test_upload_large_file_retry_success(comms_instance):
    """Test retry logic in upload_large_file with exact implementation parameters"""
    mock_client = AsyncMock()

    # Track uploaded parts for verification
    uploaded_parts = []

    async def mock_upload_part(**kwargs):
        uploaded_parts.append(
            {"PartNumber": kwargs["PartNumber"], "ETag": f"etag_{kwargs['PartNumber']}"}
        )
        return {"ETag": f"etag_{kwargs['PartNumber']}"}

    mock_client.create_multipart_upload = AsyncMock(
        side_effect=[Exception("First attempt failed"), {"UploadId": "test_id"}]
    )
    mock_client.upload_part = AsyncMock(side_effect=mock_upload_part)
    mock_client.complete_multipart_upload = AsyncMock()
    mock_client.abort_multipart_upload = AsyncMock()

    async def mock_create_client(*args, **kwargs):
        return mock_client

    comms_instance.session.create_client = AsyncMock(side_effect=mock_create_client)

    # Create test file matching implementation's part size
    PART_SIZE = 50 * 1024 * 1024  # 50MB chunks from implementation
    MAX_CONCURRENT_PARTS = 5  # From implementation

    # Create a file that will generate multiple batches (120MB = 3 parts)
    file_size = 120 * 1024 * 1024
    with open("test_file.txt", "wb") as f:
        f.write(b"\0" * file_size)

    try:
        await comms_instance.upload_large_file(
            file_path="test_file.txt", key="test_key", s3_client=mock_client
        )

        # Verify create_multipart_upload retry behavior
        assert mock_client.create_multipart_upload.call_count == 2
        create_calls = mock_client.create_multipart_upload.call_args_list
        for call in create_calls:
            assert call.kwargs["Bucket"] == comms_instance.bucket.name
            assert call.kwargs["Key"] == "test_key"

        # Verify part upload behavior
        expected_parts = math.ceil(file_size / PART_SIZE)
        expected_batches = math.ceil(expected_parts / MAX_CONCURRENT_PARTS)

        print(f"\nFile size: {file_size / (1024 * 1024):.2f}MB")
        print(f"Part size: {PART_SIZE / (1024 * 1024):.2f}MB")
        print(f"Expected parts: {expected_parts}")
        print(f"Expected batches: {expected_batches}")
        print(f"Actual uploaded parts: {len(uploaded_parts)}")

        # Verify number of parts
        assert len(uploaded_parts) == expected_parts

        # Verify all required part numbers are present (order doesn't matter for uploads)
        uploaded_part_numbers = {part["PartNumber"] for part in uploaded_parts}
        expected_part_numbers = set(range(1, expected_parts + 1))
        assert uploaded_part_numbers == expected_part_numbers, (
            f"Missing part numbers. Got {uploaded_part_numbers}, expected {expected_part_numbers}"
        )

        # Verify complete_multipart_upload was called with correct parts
        assert mock_client.complete_multipart_upload.call_count == 1
        complete_call = mock_client.complete_multipart_upload.call_args
        assert complete_call.kwargs["Bucket"] == comms_instance.bucket.name
        assert complete_call.kwargs["Key"] == "test_key"
        assert complete_call.kwargs["UploadId"] == "test_id"

        # Verify parts in complete_multipart_upload are sorted by PartNumber
        complete_parts = complete_call.kwargs["MultipartUpload"]["Parts"]
        assert len(complete_parts) == expected_parts
        complete_part_numbers = [p["PartNumber"] for p in complete_parts]
        assert complete_part_numbers == sorted(complete_part_numbers), (
            f"Parts in complete_multipart_upload not properly sorted: {complete_part_numbers}"
        )

        # Verify abort was not called (successful upload)
        assert mock_client.abort_multipart_upload.call_count == 0

    finally:
        if os.path.exists("test_file.txt"):
            os.remove("test_file.txt")


# New test for gather batching
@pytest.mark.asyncio
async def test_gather_with_batching(comms_instance):
    """Test gather operation with batched processing"""
    state_dict = {
        "layer1.weightsidxs": torch.tensor([0, 1, 2]),
        "layer1.weightsvals": torch.tensor([0.1, 0.2, 0.3]),
    }

    peer_responses = [
        (
            {
                "layer1.weightsidxs": torch.tensor([i, i + 1]),
                "layer1.weightsvals": torch.tensor([0.1 * i, 0.2 * i]),
            },
            i,
        )
        for i in range(1, 8)
    ]

    call_count = 0

    async def mock_get_with_retry(*args, **kwargs):
        nonlocal call_count
        if call_count < len(peer_responses):
            response = peer_responses[call_count]
            call_count += 1
            return response
        return None

    comms_instance.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

    result = await comms_instance.gather(
        state_dict=state_dict,
        my_uid="0",
        uids=[str(i) for i in range(1, 8)],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        global_step=0,
        local=True,
    )

    assert result is not None
    assert len(result.uids) == 7
    assert hasattr(result.state_dict, "layer1.weightsvals")

    # Access the attributes using dictionary style access
    vals = getattr(result.state_dict, "layer1.weightsvals")
    idxs = getattr(result.state_dict, "layer1.weightsidxs")

    # Print debug info
    print("\nGathered state dict attributes:", vars(result.state_dict))
    print("Vals type:", type(vals))
    print("Vals:", vals)

    # Verify we got a list of tensors
    assert isinstance(vals, list), f"Expected list but got {type(vals)}"
    assert all(isinstance(v, torch.Tensor) for v in vals), (
        "All elements should be tensors"
    )
    assert len(vals) == 7, f"Expected 7 tensors but got {len(vals)}"

    # Verify tensor shapes
    for i, v in enumerate(vals):
        assert v.shape == torch.Size([2]), f"Tensor {i} has wrong shape: {v.shape}"

    # Verify we got responses from all peers
    assert len(result.uids) == 7
    assert all(str(uid) in result.uids for uid in range(1, 8))

    # Verify the global steps were collected
    assert len(result.global_steps) == 7
    assert result.global_steps == list(range(1, 8))


# New test for connection semaphore
@pytest.mark.asyncio
async def test_gather_connection_limiting(comms_instance):
    """Test connection limiting in gather operation"""

    async def delayed_response(*args, **kwargs):
        await asyncio.sleep(0.1)
        return (
            {
                "layer1.weightsidxs": torch.tensor([0]),
                "layer1.weightsvals": torch.tensor([0.1]),
            },
            1,
        )

    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def wrapped_get_with_retry(*args, **kwargs):
        nonlocal max_concurrent, current_concurrent
        async with lock:
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)

        result = await delayed_response(*args, **kwargs)

        async with lock:
            current_concurrent -= 1
        return result

    comms_instance.get_with_retry = AsyncMock(side_effect=wrapped_get_with_retry)

    result = await comms_instance.gather(
        state_dict=None,
        my_uid="0",
        uids=[str(i) for i in range(10)],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        global_step=0,
    )

    assert max_concurrent <= 20
    assert result is not None
    assert len(result.uids) == 10


@pytest.fixture
def monkeypatch():
    with patch(
        "tplr.comms.Comms.get_base_url", lambda account_id: "http://localhost:4566"
    ):
        yield


async def test_get_base_url(monkeypatch):
    comms = await setup_test_comms()
    base_url = comms.get_base_url("test_account")
    assert base_url == "http://localhost:4566"
