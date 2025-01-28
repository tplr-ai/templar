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


def mock_metagraph():
    metagraph = MagicMock()
    metagraph.hotkeys = ["test_hotkey_address"]
    metagraph.uids = [0]
    metagraph.S = torch.tensor([100.0])  # Stake
    return metagraph


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
    """Test that gradients are stored when store_gathers=True"""
    # Setup test data
    state_dict = {
        "layer.idxs": torch.tensor([0, 1]),
        "layer.vals": torch.tensor([0.1, 0.2]),
    }

    # Mock methods
    comms_instance.get_with_retry = AsyncMock()
    peer_response = (state_dict, 1)
    comms_instance.get_with_retry.side_effect = [peer_response]
    comms_instance.s3_put_object = AsyncMock()

    # Call gather with store_gathers=True
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

    # Wait a bit for async tasks to be created
    await asyncio.sleep(0.1)
    
    # Verify s3_put_object was called
    assert comms_instance.s3_put_object.called
    
    # Verify correct arguments
    call_args = comms_instance.s3_put_object.call_args
    assert call_args is not None
    kwargs = call_args.kwargs
    assert kwargs["bucket"] == comms_instance.bucket
    assert kwargs["key"].startswith("gathers/")
    assert kwargs["key"].endswith(".npz")


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


async def test_s3_put_large_file(comms_instance):
    """Test multipart upload for large files"""
    # Mock S3 client with proper responses
    mock_client = AsyncMock()
    mock_client.create_multipart_upload = AsyncMock(
        return_value={"UploadId": "test_id"}
    )
    mock_client.upload_part = AsyncMock(return_value={"ETag": "test_etag"})
    mock_client.complete_multipart_upload = AsyncMock()
    comms_instance.session.create_client = MagicMock(return_value=mock_client)

    # Create proper Bucket instance
    comms_instance.bucket = Bucket(
        name="test-bucket",
        account_id="test-account",
        access_key_id="test-key",
        secret_access_key="test-secret",
    )

    # Create large test file
    with open("large_file.txt", "wb") as f:
        f.write(os.urandom(10 * 1024 * 1024))  # 10MB file

    await comms_instance.s3_put_object("test_key", "large_file.txt")

    # Cleanup
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


async def test_load_checkpoint_missing_data(comms_instance):
    """Test checkpoint loading with missing data"""
    model = MagicMock()
    optimizer = MagicMock()
    scheduler = MagicMock()

    comms_instance.gather = AsyncMock(return_value=None)

    success, _, _, _, _ = await comms_instance.load_checkpoint(
        model,
        optimizer,
        scheduler,
        MagicMock(),
        MagicMock(),
        current_window=10,
        device="cpu",
        peers=[1, 2],
        uid="0",
    )
    assert not success


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


async def test_gather_store_gathers_non_blocking(comms_instance):
    """Test that storing gradients doesn't block the gather operation"""
    # Setup test data
    state_dict = {
        "layer.idxs": torch.tensor([0, 1]),
        "layer.vals": torch.tensor([0.1, 0.2]),
    }

    # Mock methods
    comms_instance.get_with_retry = AsyncMock()
    peer_response = (state_dict, 1)
    comms_instance.get_with_retry.side_effect = [peer_response]

    # Mock s3_put_object to simulate slow upload
    async def slow_upload(*args, **kwargs):
        await asyncio.sleep(1)  # Simulate slow upload
        return True
    comms_instance.s3_put_object = AsyncMock(side_effect=slow_upload)

    # Measure time taken
    start_time = time.perf_counter()

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

    # Wait a bit for async tasks to be created
    await asyncio.sleep(0.1)

    end_time = time.perf_counter()
    duration = end_time - start_time

    # Verify gather completed quickly (much less than 1 second)
    assert duration < 0.5, f"Gather took {duration:.2f}s, should be near-instant"

    # Verify upload was initiated
    assert comms_instance.s3_put_object.called
