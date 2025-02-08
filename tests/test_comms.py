# ruff: noqa

import os
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import torch
from types import SimpleNamespace
from dotenv import load_dotenv
import pytest_asyncio
import asyncio
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


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def create_xshapes_totalks(model):
    xshapes, totalks = {}, {}
    for name, param in model.named_parameters():
        xshapes[name] = param.shape
        totalks[name] = param.numel()
    return xshapes, totalks


def create_valid_state_dict(model):
    state_dict = {}
    for name, _ in model.named_parameters():
        state_dict[name + "idxs"] = torch.tensor([0, 1], dtype=torch.long)
        state_dict[name + "vals"] = torch.tensor([0.1, 0.2], dtype=torch.float32)
    return state_dict


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


@pytest.mark.asyncio
async def test_gather_basic_functionality(comms_instance):
    """
    Test gather basic functionality:
        - Setup:
              • Simulate two valid peer responses via get_with_retry.
        - Expected Outcome:
              • The aggregated state_dict is correctly constructed.
              • Valid UIDs and global steps match the responses.
    """
    # Remove any external state_dict parameter.
    comms_instance.get_with_retry = AsyncMock()

    peer1_response = (
        {
            "layer1.weightsidxs": torch.tensor([0, 1, 2]),
            "layer1.weightsvals": torch.tensor([0.4, 0.5, 0.6]),
        },
        1,  # global_step for uid "1"
    )
    peer2_response = (
        {
            "layer1.weightsidxs": torch.tensor([0, 1, 2]),
            "layer1.weightsvals": torch.tensor([0.7, 0.8, 0.9]),
        },
        2,  # global_step for uid "2"
    )
    comms_instance.get_with_retry.side_effect = [peer1_response, peer2_response]

    # Call gather without the unsupported global_step keyword.
    result = await comms_instance.gather(
        my_uid="0",
        uids=["1", "2"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        local=True,
        stale_retention=10,
    )

    # Validate the outcome.
    assert result is not None, "Expected a non-None result"
    # Check that the aggregated valid UIDs match the responses provided.
    assert result.uids == ["1", "2"], (
        f"Expected valid_uids ['1', '2'], got {result.uids}"
    )
    # Check that the global steps are as expected.
    assert result.global_steps == [1, 2], (
        f"Expected global_steps [1, 2], got {result.global_steps}"
    )

    # Verify that the aggregated state_dict contains the expected keys from both responses.
    aggregated = result.state_dict.__dict__
    for key in ["layer1.weightsidxs", "layer1.weightsvals"]:
        assert key in aggregated, f"Expected key {key} in aggregated state_dict"
        # There should be one tensor per valid UID.
        assert len(aggregated[key]) == 2, (
            f"Expected 2 tensors for key {key}, got {len(aggregated[key])}"
        )


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
        my_uid="0",
        uids=["1"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
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
        my_uid="0",
        uids=["1", "2"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
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
        my_uid="0",
        uids=["1", "2"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
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
        my_uid="0",
        uids=["1", "2", "3"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
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
            my_uid="0",
            uids=["1"],
            window=1,
            key="gradient",
            timeout=5,
            device="cpu",
            local=True,  # Use local=True to avoid S3 operations
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
        my_uid="0",
        uids=["1"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
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


@pytest.mark.asyncio
async def test_valid_response_handling(comms_instance, model):
    """
    Test 1: Valid Response Handling
        - Setup:
              • Create a dummy reference model with a few parameters.
              • Precompute dummy xshapes and totalks for each parameter.
              • Use AsyncMock to simulate a get_with_retry response for a UID that returns a state_dict
                containing proper keys "<param_name>idxs" and "<param_name>vals", which when passed into
                compressor.batch_decompress and transformer.decode succeed without raising an exception.
        - Expected Outcome:
              • The UID appears in valid_uids.
              • The aggregated state_dict contains normalized tensors for each gradient.
              • The skipped_uids list is empty.
              • Global steps are aggregated correctly.
    """
    comms = comms_instance  # using the provided fixture
    device = "cpu"

    # If compressor is None, assign a dummy one; otherwise, patch its method.
    if comms.compressor is None:

        class DummyCompressor:
            def batch_decompress(self, p, idxs, vals, xshape, totalk):
                return torch.ones_like(p)

        comms.compressor = DummyCompressor()
    else:
        comms.compressor.batch_decompress = (
            lambda p, idxs, vals, xshape, totalk: torch.ones_like(p)
        )

    # If transformer is None, assign a dummy one; otherwise, patch its method.
    if comms.transformer is None:

        class DummyTransformer:
            def decode(self, x):
                return x

        comms.transformer = DummyTransformer()
    else:
        comms.transformer.decode = lambda x: x

    # Precompute dummy xshapes and totalks for each parameter.
    xshapes = {}
    totalks = {}
    for name, param in model.named_parameters():
        xshapes[name] = param.shape  # using the original shape as dummy xshape
        totalks[name] = param.numel()  # total number of elements

    # Define a list of dummy UIDs.
    uids = ["uid1", "uid2", "uid3"]

    # Create valid responses for each UID.
    def create_valid_state_dict():
        state_dict = {}
        for name, param in model.named_parameters():
            # Simulate proper gradient keys.
            state_dict[name + "idxs"] = torch.tensor([0, 1], dtype=torch.long)
            state_dict[name + "vals"] = torch.tensor([0.1, 0.2], dtype=torch.float32)
        return state_dict

    # Simulate global step responses.
    responses = [
        (create_valid_state_dict(), 10),
        (create_valid_state_dict(), 20),
        (create_valid_state_dict(), 30),
    ]
    call_count = 0

    async def mock_get_with_retry(*args, **kwargs):
        nonlocal call_count
        if call_count < len(responses):
            response = responses[call_count]
            call_count += 1
            return response
        return None

    # Replace the real get_with_retry with our mock.
    comms.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

    # Call gather providing the reference model and the dummy xshapes and totalks.
    result = await comms.gather(
        my_uid="dummy_uid",
        uids=uids,
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        ref_model=model,
        xshapes=xshapes,
        totalks=totalks,
    )

    # Validate the result.
    assert result is not None, "Expected gather result to be non-None"
    # Check that all UIDs appear in valid_uids.
    assert set(result.uids) == set(uids), (
        f"Expected valid_uids {uids}, got {result.uids}"
    )
    # Ensure no UID was skipped.
    assert result.skipped_uids == [], (
        f"Expected skipped_uids to be empty, got {result.skipped_uids}"
    )
    # Verify global steps are aggregated in order.
    assert result.global_steps == [10, 20, 30], (
        f"Unexpected global_steps: {result.global_steps}"
    )

    # Check that the aggregated state_dict contains normalized tensors.
    aggregated = result.state_dict.__dict__
    for name, param in model.named_parameters():
        key_vals = name + "vals"
        assert key_vals in aggregated, f"Missing aggregated key {key_vals}"
        tensor_list = aggregated[key_vals]
        # There should be one tensor per UID.
        assert len(tensor_list) == len(uids), (
            f"Expected {len(uids)} tensors in {key_vals}, got {len(tensor_list)}"
        )
        # Each tensor should be normalized (norm approx. 1).
        for tensor in tensor_list:
            norm = torch.norm(tensor)
            assert torch.isclose(norm, torch.tensor(1.0, device=device), atol=1e-5), (
                f"Tensor in {key_vals} is not normalized: norm = {norm}"
            )

    # Verify that download_bytes metric is computed (i.e. > 0).
    assert result.download_bytes > 0, "Expected download_bytes to be >0"


@pytest.mark.asyncio
async def test_missing_idxs_key(comms_instance, model):
    """
    Test 2: Missing "idxs" Key for a Parameter
        - Setup:
              • Simulate a UID response that includes "<param_name>vals" but is missing "<param_name>idxs".
        - Expected Outcome:
              • The gradient decoding check fails for that UID.
              • The UID is skipped and is added to skipped_uids.
              • This UID does not contribute to the aggregated state_dict.
    """
    comms = comms_instance  # use provided fixture
    device = "cpu"

    # Ensure compressor and transformer are available.
    if comms.compressor is None:

        class DummyCompressor:
            def batch_decompress(self, p, idxs, vals, xshape, totalk):
                return torch.ones_like(p)

        comms.compressor = DummyCompressor()
    else:
        comms.compressor.batch_decompress = (
            lambda p, idxs, vals, xshape, totalk: torch.ones_like(p)
        )

    if comms.transformer is None:

        class DummyTransformer:
            def decode(self, x):
                return x

        comms.transformer = DummyTransformer()
    else:
        comms.transformer.decode = lambda x: x

    # Precompute dummy xshapes and totalks for each model parameter.
    xshapes, totalks = {}, {}
    for name, param in model.named_parameters():
        xshapes[name] = param.shape
        totalks[name] = param.numel()

    # Define dummy UIDs.
    uids = ["uid1", "uid2", "uid3"]

    # Define helper to create valid state_dict.
    def create_valid_state_dict():
        state_dict = {}
        for name, param in model.named_parameters():
            state_dict[name + "idxs"] = torch.tensor([0, 1], dtype=torch.long)
            state_dict[name + "vals"] = torch.tensor([0.1, 0.2], dtype=torch.float32)
        return state_dict

    # Define helper to create an invalid state_dict missing the "<param_name>idxs" keys.
    def create_missing_idxs_state_dict():
        state_dict = {}
        for name, param in model.named_parameters():
            # Intentionally omit the "idxs" key.
            state_dict[name + "vals"] = torch.tensor([0.1, 0.2], dtype=torch.float32)
        return state_dict

    # Simulate responses: first UID returns invalid state_dict, others valid.
    responses = [
        (
            create_missing_idxs_state_dict(),
            10,
        ),  # UID "uid1" is missing "idxs" -> should be skipped.
        (create_valid_state_dict(), 20),  # UID "uid2" valid.
        (create_valid_state_dict(), 30),  # UID "uid3" valid.
    ]
    call_count = 0

    async def mock_get_with_retry(*args, **kwargs):
        nonlocal call_count
        if call_count < len(responses):
            resp = responses[call_count]
            call_count += 1
            return resp
        return None

    # Replace real get_with_retry with the mock.
    comms.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

    # Call gather() with our simulated responses.
    result = await comms.gather(
        my_uid="dummy_uid",
        uids=uids,
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        ref_model=model,
        xshapes=xshapes,
        totalks=totalks,
    )

    # Validate the result.
    # Expect only valid UIDs "uid2" and "uid3".
    assert result is not None, "Expected non-None result from gather()"
    assert result.uids == ["uid2", "uid3"], (
        f"Expected valid_uids ['uid2', 'uid3'], got {result.uids}"
    )
    assert result.skipped_uids == ["uid1"], (
        f"Expected skipped_uids ['uid1'], got {result.skipped_uids}"
    )
    # Global steps should match those from valid responses.
    assert result.global_steps == [20, 30], (
        f"Expected global_steps [20, 30], got {result.global_steps}"
    )

    # Check aggregated state_dict: only valid UIDs (2 responses) should be aggregated.
    aggregated = result.state_dict.__dict__
    for name, _ in model.named_parameters():
        key_vals = name + "vals"
        assert key_vals in aggregated, f"Missing aggregated key {key_vals}"
        tensor_list = aggregated[key_vals]
        assert len(tensor_list) == 2, (
            f"Expected 2 tensors in {key_vals}, got {len(tensor_list)}"
        )
        # Verify each tensor gets normalized (norm approx. 1).
        for tensor in tensor_list:
            norm = torch.norm(tensor)
            assert torch.isclose(norm, torch.tensor(1.0, device=device), atol=1e-5), (
                f"Tensor in {key_vals} is not normalized: norm = {norm}"
            )

    # Confirm the download_bytes metric is computed.
    assert result.download_bytes > 0, "Expected download_bytes to be > 0"


@pytest.mark.asyncio
async def test_missing_vals_key(comms_instance, model):
    """
    Test 3: Missing "vals" Key for a Parameter
        - Setup:
              • Simulate a UID response with "<param_name>idxs" present but missing "<param_name>vals".
        - Expected Outcome:
              • The UID is skipped and added to skipped_uids.
              • No gradient data from this UID is aggregated.
    """
    comms = comms_instance  # using the provided fixture
    device = "cpu"

    # Ensure compressor and transformer are available.
    if comms.compressor is None:

        class DummyCompressor:
            def batch_decompress(self, p, idxs, vals, xshape, totalk):
                return torch.ones_like(p)

        comms.compressor = DummyCompressor()
    else:
        comms.compressor.batch_decompress = (
            lambda p, idxs, vals, xshape, totalk: torch.ones_like(p)
        )

    if comms.transformer is None:

        class DummyTransformer:
            def decode(self, x):
                return x

        comms.transformer = DummyTransformer()
    else:
        comms.transformer.decode = lambda x: x

    # Precompute dummy xshapes and totalks for each model parameter.
    xshapes, totalks = {}, {}
    for name, param in model.named_parameters():
        xshapes[name] = param.shape
        totalks[name] = param.numel()

    # Define dummy UIDs.
    uids = ["uid1", "uid2", "uid3"]

    # Helper function to create a valid state_dict with both keys.
    def create_valid_state_dict():
        state_dict = {}
        for name, _ in model.named_parameters():
            state_dict[name + "idxs"] = torch.tensor([0, 1], dtype=torch.long)
            state_dict[name + "vals"] = torch.tensor([0.1, 0.2], dtype=torch.float32)
        return state_dict

    # Helper function to create an invalid state_dict missing the "vals" key.
    def create_missing_vals_state_dict():
        state_dict = {}
        for name, _ in model.named_parameters():
            state_dict[name + "idxs"] = torch.tensor([0, 1], dtype=torch.long)
            # Intentionally omit the "vals" key.
        return state_dict

    # Simulate responses:
    #  - "uid1" returns a state_dict missing "vals"
    #  - "uid2" returns a state_dict missing "vals"
    #  - "uid3" returns a valid state_dict.
    responses = [
        (create_missing_vals_state_dict(), 10),
        (create_valid_state_dict(), 20),
        (create_valid_state_dict(), 30),
    ]
    call_count = 0

    async def mock_get_with_retry(*args, **kwargs):
        nonlocal call_count
        if call_count < len(responses):
            resp = responses[call_count]
            call_count += 1
            return resp
        return None

    # Replace the real get_with_retry with our mock.
    comms.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

    # Invoke gather() with our simulated responses.
    result = await comms.gather(
        my_uid="dummy_uid",
        uids=uids,
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        ref_model=model,
        xshapes=xshapes,
        totalks=totalks,
    )

    # Validate the result.
    # We expect that only UIDs "uid2" and "uid3" are accepted as valid.
    assert result is not None, "Expected non-None result from gather()"
    assert result.uids == ["uid2", "uid3"], (
        f"Expected valid_uids ['uid2', 'uid3'], got {result.uids}"
    )
    assert result.skipped_uids == ["uid1"], (
        f"Expected skipped_uids ['uid1'], got {result.skipped_uids}"
    )
    # Global steps should be from uid2 and uid3.
    assert result.global_steps == [20, 30], (
        f"Expected global_steps [20, 30], got {result.global_steps}"
    )

    # Check aggregated state_dict: only valid UIDs (2 responses) should be aggregated.
    aggregated = result.state_dict.__dict__
    for name, _ in model.named_parameters():
        key_vals = name + "vals"
        assert key_vals in aggregated, f"Missing aggregated key {key_vals}"
        tensor_list = aggregated[key_vals]
        # There should be one entry per valid UID.
        assert len(tensor_list) == 2, (
            f"Expected 2 tensors in {key_vals}, got {len(tensor_list)}"
        )
        for tensor in tensor_list:
            norm = torch.norm(tensor)
            assert torch.isclose(norm, torch.tensor(1.0, device=device), atol=1e-5), (
                f"Tensor in {key_vals} is not normalized: norm = {norm}"
            )

    # Verify that the download_bytes metric is computed.
    assert result.download_bytes > 0, "Expected download_bytes to be > 0"


@pytest.mark.asyncio
async def test_empty_or_none_state_dict(comms_instance, model):
    """
    Test 4: Empty or None state_dict
      - Setup:
            • Use AsyncMock to have get_with_retry return a valid response for the first UID and
              None (or an empty dict) for subsequent UIDs.
      - Expected Outcome:
            • Only the UID that returns a valid state_dict is aggregated.
            • The remaining UIDs are skipped (i.e. collected in skipped_uids).
            • Global steps reflect only valid responses.
    """
    comms = comms_instance
    device = "cpu"
    xshapes, totalks = create_xshapes_totalks(model)

    # Patch transformer and compressor so decoding succeeds.
    if comms.transformer is None:

        class DummyTransformer:
            def decode(self, x):
                return x

        comms.transformer = DummyTransformer()
    else:
        comms.transformer.decode = lambda x: x

    if comms.compressor is None:

        class DummyCompressor:
            def batch_decompress(self, p, idxs, vals, xshape, totalk):
                return vals

        comms.compressor = DummyCompressor()
    else:
        comms.compressor.batch_decompress = lambda p, idxs, vals, xshape, totalk: vals

    # Define dummy UIDs.
    uids = ["uid1", "uid2", "uid3"]
    call_count = 0

    async def mock_get_with_retry(*args, **kwargs):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return (create_valid_state_dict(model), 10)
        elif call_count == 1:
            call_count += 1
            return None
        elif call_count == 2:
            call_count += 1
            return ({}, 30)
        return None

    comms.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

    result = await comms.gather(
        my_uid="dummy_uid",
        uids=uids,
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        ref_model=model,
        xshapes=xshapes,
        totalks=totalks,
    )

    # Since only the first UID returns a valid response,
    # valid_uids should be ["uid1"] and the others should be skipped.
    assert result is not None, "Expected a non-None result."
    assert result.uids == ["uid1"], f"Expected valid_uids ['uid1'], got {result.uids}"
    assert set(result.skipped_uids) == {"uid2", "uid3"}, (
        f"Expected skipped_uids {{'uid2', 'uid3'}}, got {result.skipped_uids}"
    )
    assert result.global_steps == [10], (
        f"Expected global_steps [10], got {result.global_steps}"
    )


@pytest.mark.asyncio
async def test_get_with_retry_exception(comms_instance, model, caplog):
    """
    Test 5: get_with_retry Exception
      - Setup:
            • Force get_with_retry (via AsyncMock) to throw an exception for a UID.
      - Expected Outcome:
            • That UID is skipped.
            • The UID is recorded in skipped_uids.
            • An appropriate warning message is logged.
    """
    comms = comms_instance
    device = "cpu"
    xshapes, totalks = create_xshapes_totalks(
        model
    )  # global helper from earlier in the file
    uids = ["uid1", "uid2", "uid3"]

    # Patch transformer and compressor so that gradient decoding succeeds.
    if comms.transformer is None:

        class DummyTransformer:
            def decode(self, x):
                return x

        comms.transformer = DummyTransformer()
    else:
        comms.transformer.decode = lambda x: x

    if comms.compressor is None:

        class DummyCompressor:
            def batch_decompress(self, p, idxs, vals, xshape, totalk):
                return vals

        comms.compressor = DummyCompressor()
    else:
        comms.compressor.batch_decompress = lambda p, idxs, vals, xshape, totalk: vals

    call_count = 0

    async def mock_get_with_retry(*args, **kwargs):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            raise Exception("Simulated exception in get_with_retry")
        elif call_count == 1:
            call_count += 1
            return (create_valid_state_dict(model), 20)
        elif call_count == 2:
            call_count += 1
            return (create_valid_state_dict(model), 30)
        return None

    caplog.set_level("DEBUG")
    comms.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

    result = await comms.gather(
        my_uid="dummy_uid",
        uids=uids,
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        ref_model=model,
        xshapes=xshapes,
        totalks=totalks,
    )

    # Expect that the first UID raised an exception and that UIDs uid2 and uid3 contributed.
    assert result is not None, "Expected non-None result."
    assert result.uids == ["uid2", "uid3"], (
        f"Expected valid_uids ['uid2', 'uid3'], got {result.uids}"
    )
    assert result.global_steps == [20, 30], (
        f"Expected global_steps [20, 30], got {result.global_steps}"
    )


@pytest.mark.asyncio
async def test_gradient_decoding_failure(comms_instance, model):
    """
    Test 6: Gradient Decoding Failure
      - Setup:
            • Simulate a UID response where transformer.decode raises an exception during processing.
      - Expected Outcome:
            • The exception is caught.
            • The entire UID response is skipped.
            • UID "uid1" is skipped, and UIDs "uid2" and "uid3" are aggregated.
    """
    comms = comms_instance
    device = "cpu"
    xshapes, totalks = create_xshapes_totalks(model)
    uids = ["uid1", "uid2", "uid3"]

    # Create a transformer that fails on the second decode call.
    # For a typical model with a Linear layer, the state dict may include weight and bias.
    # This ensures that the first UID (uid1) fails decoding.
    class FaultyTransformer:
        def __init__(self):
            self.call_count = 0

        def decode(self, x):
            self.call_count += 1
            if self.call_count == 2:
                raise Exception("Decoding failure")
            return x

    comms.transformer = FaultyTransformer()

    # Use a dummy compressor that simply returns the values.
    class DummyCompressor:
        def batch_decompress(self, p, idxs, vals, xshape, totalk):
            return vals

    comms.compressor = DummyCompressor()

    # Use the global create_valid_state_dict(model) helper here.
    responses = [
        (create_valid_state_dict(model), 10),
        (create_valid_state_dict(model), 20),
        (create_valid_state_dict(model), 30),
    ]

    call_count = 0

    async def mock_get_with_retry(*args, **kwargs):
        nonlocal call_count
        if call_count < len(responses):
            response = responses[call_count]
            call_count += 1
            return response
        return None

    comms.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

    result = await comms.gather(
        my_uid="dummy_uid",
        uids=uids,
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        ref_model=model,
        xshapes=xshapes,
        totalks=totalks,
    )

    # Expect that uid1 is skipped due to the decoding exception.
    assert result is not None, "Expected a non-None result"
    # UID "uid1" should be skipped, and "uid2" and "uid3" should be aggregated.
    assert result.uids == ["uid2", "uid3"], (
        f"Expected valid_uids ['uid2', 'uid3'], got {result.uids}"
    )
    assert set(result.skipped_uids) == {"uid1"}, (
        f"Expected skipped_uids {{'uid1'}}, got {result.skipped_uids}"
    )


@pytest.mark.asyncio
async def test_mixed_valid_and_invalid_uid_responses(comms_instance, model):
    """
    Test 7: Mixed Valid and Invalid UID Responses
      - Setup:
            • Simulate responses for multiple UIDs – some with valid gradient data and some with missing keys or causing decode errors.
      - Expected Outcome:
            • Only valid UID responses are aggregated.
            • valid_uids contains only the UIDs without decoding issues.
            • skipped_uids lists all UIDs with problems.
            • Global steps and metrics reflect only the valid responses.
    """
    comms = comms_instance
    device = "cpu"
    xshapes, totalks = create_xshapes_totalks(model)
    uids = ["uid1", "uid2", "uid3", "uid4"]

    def create_valid():
        return create_valid_state_dict(model)

    def create_missing_idxs():
        d = {}
        for name, _ in model.named_parameters():
            d[name + "vals"] = torch.tensor([0.1, 0.2], dtype=torch.float32)
        return d

    responses = [
        (create_valid(), 10),  # uid1 valid
        (create_missing_idxs(), 20),  # uid2 invalid (missing idxs)
        (create_valid(), 30),  # uid3 valid
        (create_valid(), 40),  # uid4 will trigger decode error
    ]
    call_count = 0

    async def mock_get_with_retry(*args, **kwargs):
        nonlocal call_count
        resp = responses[call_count]
        call_count += 1
        return resp

    comms.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

    # Ensure that comms.transformer is not None.
    if comms.transformer is None:

        class DummyTransformer:
            def decode(self, x):
                return x

        comms.transformer = DummyTransformer()

    # Patch transformer.decode to simulate a decode failure for uid4.
    p_count = sum(1 for _ in model.parameters())
    decode_call_count = 0

    def custom_decode(x):
        nonlocal decode_call_count, p_count
        decode_call_count += 1
        # For the fourth UID, simulate a decoding failure on its first decode call.
        # Given the processing order, we assume uid4's first call occurs when decode_call_count == (2 * p_count) + 1.
        if decode_call_count == (2 * p_count) + 1:
            raise Exception("Simulated gradient decoding failure for uid4")
        return x

    comms.transformer.decode = custom_decode

    # Similarly, if no compressor is set, create a dummy compressor.
    if comms.compressor is None:

        class DummyCompressor:
            def batch_decompress(self, p, idxs, vals, xshape, totalk):
                return vals

        comms.compressor = DummyCompressor()

    result = await comms.gather(
        my_uid="dummy_uid",
        uids=uids,
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        ref_model=model,
        xshapes=xshapes,
        totalks=totalks,
    )

    # Expect valid responses only from uid1 and uid3.
    assert result is not None, "Expected a non-None result"
    assert result.uids == ["uid1", "uid3"], (
        f"Expected valid_uids ['uid1', 'uid3'], got {result.uids}"
    )
    # Expect uid2 (missing idxs) and uid4 (decoding error) to be skipped.
    assert set(result.skipped_uids) == {"uid2", "uid4"}, (
        f"Expected skipped_uids {{'uid2', 'uid4'}}, got {result.skipped_uids}"
    )


@pytest.mark.asyncio
async def test_no_uids_provided(comms_instance, model):
    """
    Test 8: No UIDs Provided
      - Setup:
            • Call gather() with an empty UID list.
      - Expected Outcome:
            • The function returns None or an appropriate empty result.
            • No gradients are aggregated.
    """
    comms = comms_instance
    device = "cpu"
    xshapes, totalks = create_xshapes_totalks(model)
    result = await comms.gather(
        my_uid="dummy_uid",
        uids=[],  # empty UID list
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        ref_model=model,
        xshapes=xshapes,
        totalks=totalks,
    )
    if result is None:
        assert result is None, "Expected None when no UIDs provided."
    else:
        assert result.uids == []
        assert result.skipped_uids == []
        assert result.global_steps == []
        for key, tensor_list in result.state_dict.__dict__.items():
            assert tensor_list == [] or tensor_list is None, (
                f"Expected empty aggregation for {key}"
            )
        assert result.download_bytes == 0


@pytest.mark.asyncio
async def test_aggregation_without_reference_information(comms_instance, model):
    """
    Test 9: Aggregation Without Reference Information
      - Setup:
            • Call gather() without passing ref_model, xshapes, and totalks.
            • Simulate valid get_with_retry responses.
      - Expected Outcome:
            • Aggregation proceeds solely by converting tensors.
            • No UID is skipped due to decoding failures.
    """
    comms = comms_instance
    device = "cpu"
    uids = ["uid1", "uid2", "uid3"]

    # Setup transformer and compressor as identity decoders.
    if comms.transformer is None:

        class DummyTransformer:
            def decode(self, x):
                return x

        comms.transformer = DummyTransformer()
    else:
        comms.transformer.decode = lambda x: x
    if comms.compressor is None:

        class DummyCompressor:
            def batch_decompress(self, p, idxs, vals, xshape, totalk):
                return torch.tensor(p, dtype=torch.float32)

        comms.compressor = DummyCompressor()
    else:
        comms.compressor.batch_decompress = (
            lambda p, idxs, vals, xshape, totalk: torch.tensor(p, dtype=torch.float32)
        )

    async def mock_get_with_retry(*args, **kwargs):
        return (create_valid_state_dict(model), 10)

    comms.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

    result = await comms.gather(
        my_uid="dummy_uid",
        uids=uids,
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        # ref_model, xshapes, totalks omitted intentionally.
    )
    assert result is not None, "Expected non-None result."
    assert result.uids == uids, f"Expected valid_uids {uids}, got {result.uids}"
    assert result.skipped_uids == [], (
        f"Expected no skipped_uids, got {result.skipped_uids}"
    )
    assert result.global_steps == [10, 10, 10]


@pytest.mark.asyncio
async def test_tensor_device_conversion(comms_instance, model):
    """
    Test 10: Tensor Device Conversion
        - Setup:
              • Simulate responses with tensors initially on CPU.
              • Call gather() with device explicitly set (e.g., "cuda" or "cpu").
        - Expected Outcome:
              • All tensors in the aggregated state_dict end up on the specified device.
    """
    comms = comms_instance
    # Use "cuda" if available; otherwise fallback to "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    expected_device = torch.device(device)
    if expected_device.type == "cuda" and expected_device.index is None:
        expected_device = torch.device("cuda:0")

    # Patch transformer if not set.
    if comms.transformer is None:

        class DummyTransformer:
            def decode(self, x):
                return x

        comms.transformer = DummyTransformer()
    else:
        comms.transformer.decode = lambda x: x

    # Patch compressor.
    if comms.compressor is None:

        class DummyCompressor:
            def batch_decompress(self, p, idxs, vals, xshape, totalk):
                return p.clone().detach().to(dtype=torch.float32)

        comms.compressor = DummyCompressor()
    else:
        comms.compressor.batch_decompress = (
            lambda p, idxs, vals, xshape, totalk: p.clone()
            .detach()
            .to(dtype=torch.float32)
        )

    # Compute xshapes and totalks from the model.
    xshapes, totalks = create_xshapes_totalks(model)
    uids = ["uid1", "uid2"]

    def create_valid():
        return create_valid_state_dict(model)

    responses = [
        (create_valid(), 10),
        (create_valid(), 20),
    ]
    call_count = 0

    async def mock_get_with_retry(*args, **kwargs):
        nonlocal call_count
        resp = responses[call_count]
        call_count += 1
        return resp

    comms.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

    result = await comms.gather(
        my_uid="dummy_uid",
        uids=uids,
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        ref_model=model,
        xshapes=xshapes,
        totalks=totalks,
    )

    assert result is not None, "Expected non-None result."
    # Verify that every tensor in the aggregated state dict is on the expected device.
    aggregated = result.state_dict.__dict__
    for key, tensor_list in aggregated.items():
        for tensor in tensor_list:
            assert tensor.device == expected_device, (
                f"Tensor {key} is on {tensor.device} but expected {expected_device}"
            )

    # Also verify that download_bytes is non-zero.
    assert result.download_bytes > 0, "Expected download_bytes to be > 0"


@pytest.mark.asyncio
async def test_all_uids_skipped(comms_instance, model):
    """
    Test 13: All UIDs Skipped
      - Setup:
            • Force all simulated UID responses to fail.
      - Expected Outcome:
            • No valid responses are received.
            • The function returns None or an empty aggregation.
            • skipped_uids contains all provided UIDs.
    """
    comms = comms_instance
    device = "cpu"
    xshapes, totalks = create_xshapes_totalks(model)
    uids = ["uid1", "uid2", "uid3"]

    call_count = 0

    async def mock_get_with_retry(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return None  # Always fail

    comms.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)
    result = await comms.gather(
        my_uid="dummy_uid",
        uids=uids,
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        ref_model=model,
        xshapes=xshapes,
        totalks=totalks,
    )
    if result is None:
        assert result is None, "Expected None when all responses are invalid"
    else:
        assert result.uids == []
        assert set(result.skipped_uids) == set(uids), (
            f"Expected skipped_uids {set(uids)}, got {result.skipped_uids}"
        )


@pytest.mark.asyncio
async def test_extremely_large_tensor_normalization(comms_instance, model):
    """
    Test: Extremely Large Tensor Normalization
      - Setup:
            • Simulate responses with extremely large tensor gradients.
      - Expected Outcome:
            • The gradients are normalized correctly without transformer.decode errors.
    """
    comms = comms_instance
    device = "cpu"
    xshapes, totalks = create_xshapes_totalks(model)
    uids = ["uid_large1", "uid_large2"]

    # Ensure transformer is set to a dummy implementation to avoid 'NoneType' decode errors.
    if comms.transformer is None:

        class DummyTransformer:
            def decode(self, x):
                return x

        comms.transformer = DummyTransformer()
    else:
        comms.transformer.decode = lambda x: x

    # Likewise, patch compressor if needed.
    if comms.compressor is None:

        class DummyCompressor:
            def batch_decompress(self, p, idxs, vals, xshape, totalk):
                return vals

        comms.compressor = DummyCompressor()
    else:
        comms.compressor.batch_decompress = lambda p, idxs, vals, xshape, totalk: vals

    # Create state dicts with extremely large tensor values.
    def create_extremely_large_state_dict(scale):
        state_dict = {}
        for name, _ in model.named_parameters():
            state_dict[name + "idxs"] = torch.tensor([0, 1], dtype=torch.long)
            # Multiply by a large scale factor.
            state_dict[name + "vals"] = torch.tensor(
                [scale * 1e6, scale * 2e6], dtype=torch.float32
            )
        return state_dict

    responses = [
        (create_extremely_large_state_dict(1.0), 100),  # uid_large1
        (create_extremely_large_state_dict(2.0), 200),  # uid_large2
    ]
    call_count = 0

    async def mock_get_with_retry(*args, **kwargs):
        nonlocal call_count
        resp = responses[call_count]
        call_count += 1
        return resp

    comms.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

    result = await comms.gather(
        my_uid="dummy_uid",
        uids=uids,
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        ref_model=model,
        xshapes=xshapes,
        totalks=totalks,
    )

    # Verify that a result is returned and that all expected keys are present.
    assert result is not None, "Expected non-None gathered result"
    aggregated = result.state_dict.__dict__
    for name, _ in model.named_parameters():
        key_idxs = name + "idxs"
        key_vals = name + "vals"
        assert key_idxs in aggregated, f"Expected {key_idxs} in aggregated state_dict"
        assert key_vals in aggregated, f"Expected {key_vals} in aggregated state_dict"
        # Verify tensors are on the specified device.
        for tensor in aggregated[key_idxs]:
            assert tensor.device.type == device, (
                f"Tensor for {key_idxs} is on {tensor.device}, expected {device}"
            )
        for tensor in aggregated[key_vals]:
            assert tensor.device.type == device, (
                f"Tensor for {key_vals} is on {tensor.device}, expected {device}"
            )
