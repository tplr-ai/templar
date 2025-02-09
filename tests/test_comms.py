# ruff: noqa

import os
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import torch
from types import SimpleNamespace
from dotenv import load_dotenv
import asyncio
from dataclasses import dataclass

from tplr import load_hparams

hparams = load_hparams()

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
    xshapes = {}
    totalks = {}
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


def create_missing_idxs(model):
    d = {}
    for name, _ in model.named_parameters():
        # Omit the "idxs" key intentionally.
        d[name + "vals"] = torch.tensor([0.1, 0.2], dtype=torch.float32)
    return d


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


"""
Tests for the Comms class functionality focusing on local storage, data retrieval,
and gradient gathering operations.
"""


async def test_put_local(comms_instance):
    """Test 1: Local Storage Functionality

    Tests the ability to store data locally by:
    - Verifying data can be correctly stored in local filesystem
    - Checking directory cleanup operations work properly
    - Ensuring correct file creation with proper naming
    - Validating storage location and structure
    """
    test_state_dict = {"param": torch.tensor([1, 2, 3])}
    uid = "0"
    window = 1
    key = "gradient"

    expected_dir = os.path.join("/tmp/local_store", uid, str(window))
    base_dir = os.path.dirname(expected_dir)  # /tmp/local_store/0

    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(base_dir)

    with patch.object(comms_instance, "cleanup_local_data") as mock_cleanup:
        await comms_instance.put(
            state_dict=test_state_dict,
            uid=uid,
            window=window,
            key=key,
            local=True,
        )
        mock_cleanup.assert_called_once()

    files = os.listdir(expected_dir)
    assert len(files) == 1
    assert files[0].startswith(key)


async def test_get_local(comms_instance):
    """Test 2: Local Data Retrieval

    Validates the retrieval of locally stored data by:
    - Testing correct loading of stored state dictionaries
    - Verifying proper handling of global step information
    - Ensuring cleanup operations are called during retrieval
    - Checking data integrity after retrieval
    """
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
    """Test 3: Basic Gradient Gathering

    Tests fundamental gradient gathering operations by:
    - Validating correct handling of multiple peer responses
    - Verifying proper aggregation of gradients
    - Checking accurate tracking of UIDs and global steps
    - Ensuring correct structure of aggregated results
    """
    comms_instance.check_compressed_indices = (
        lambda param_name, idxs, totalk, allowed_topk=None: None
    )

    comms_instance.get_with_retry = AsyncMock()

    totalk_value = 100
    peer1_response = (
        {
            "0.weightidxs": torch.tensor([0, 1, 2]),
            "0.weightvals": torch.tensor([0.4, 0.5, 0.6]),
            "totalks": {"0.weight": totalk_value},
        },
        1,
    )
    peer2_response = (
        {
            "0.weightidxs": torch.tensor([0, 1, 2]),
            "0.weightvals": torch.tensor([0.7, 0.8, 0.9]),
            "totalks": {"0.weight": totalk_value},
        },
        2,
    )
    comms_instance.get_with_retry.side_effect = [peer1_response, peer2_response]

    totalks = {"0.weight": totalk_value}
    result = await comms_instance.gather(
        my_uid="0",
        uids=["1", "2"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        local=True,
        stale_retention=10,
        totalks=totalks,
    )

    assert result is not None, "Expected a non-None result"
    assert result.uids == ["1", "2"], (
        f"Expected valid_uids ['1', '2'], got {result.uids}"
    )
    assert result.global_steps == [1, 2], (
        f"Expected global_steps [1, 2], got {result.global_steps}"
    )

    aggregated = result.state_dict.__dict__
    for key in ["0.weightidxs", "0.weightvals"]:
        assert key in aggregated, f"Expected key {key} in aggregated state_dict"
        assert len(aggregated[key]) == 2, (
            f"Expected 2 tensors for key {key}, got {len(aggregated[key])}"
        )


@pytest.mark.asyncio
async def test_gather_normalization(comms_instance):
    """Test 4: Gradient Normalization

    Validates gradient normalization functionality by:
    - Testing proper handling of normalized gradients
    - Verifying correct processing of single peer response
    - Ensuring normalization maintains data integrity
    """
    comms_instance.check_compressed_indices = (
        lambda param_name, idxs, totalk, allowed_topk=None: None
    )
    comms_instance.get_with_retry = AsyncMock()

    totalk_value = 100
    peer_response = (
        {
            "0.weightidxs": torch.tensor([0, 1, 2]),
            "0.weightvals": torch.tensor([0.4, 0.5, 0.6]),
            "totalks": {"0.weight": totalk_value},
        },
        1,
    )
    comms_instance.get_with_retry.side_effect = [peer_response]
    result = await comms_instance.gather(
        my_uid="0",
        uids=["1"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        local=True,
        stale_retention=10,
        totalks={"0.weight": totalk_value},
    )
    assert result is not None


@pytest.mark.asyncio
async def test_gather_empty_responses(comms_instance):
    """Test 5: Empty Response Handling

    Tests system behavior with empty responses by:
    - Verifying proper handling when peers return no data
    - Ensuring system gracefully handles null responses
    - Checking appropriate error states and return values
    """
    comms_instance.check_compressed_indices = (
        lambda param_name, idxs, totalk, allowed_topk=None: None
    )
    comms_instance.get_with_retry = AsyncMock(return_value=(None, None))
    result = await comms_instance.gather(
        my_uid="0",
        uids=["1"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        local=True,
        stale_retention=10,
        totalks={"0.weight": 100},
    )
    assert result is None


@pytest.mark.asyncio
async def test_gather_averaging(comms_instance):
    """Test 6: Gradient Averaging

    Validates gradient averaging functionality by:
    - Testing correct averaging of gradients from multiple peers
    - Verifying proper handling of global steps during averaging
    - Ensuring averaged gradients maintain mathematical correctness
    """
    comms_instance.check_compressed_indices = (
        lambda param_name, idxs, totalk, allowed_topk=None: None
    )
    comms_instance.get_with_retry = AsyncMock()
    totalk_value = 100
    peer1_response = (
        {
            "0.weightidxs": torch.tensor([0, 1, 2]),
            "0.weightvals": torch.tensor([0.4, 0.5, 0.6]),
            "totalks": {"0.weight": totalk_value},
        },
        1,
    )
    peer2_response = (
        {
            "0.weightidxs": torch.tensor([0, 1, 2]),
            "0.weightvals": torch.tensor([0.8, 0.9, 1.0]),
            "totalks": {"0.weight": totalk_value},
        },
        2,
    )
    comms_instance.get_with_retry.side_effect = [peer1_response, peer2_response]
    result = await comms_instance.gather(
        my_uid="0",
        uids=["1", "2"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        local=True,
        stale_retention=10,
        totalks={"0.weight": totalk_value},
    )
    assert result.global_steps == [1, 2]


@pytest.mark.asyncio
async def test_gather_complex_normalization(comms_instance):
    """Test 7: Complex Normalization Scenarios

    Tests advanced normalization cases by:
    - Validating handling of multiple keys in gradient responses
    - Verifying normalization behavior with complex data structures
    - Ensuring proper handling of multi-dimensional tensors
    """
    comms_instance.check_compressed_indices = (
        lambda param_name, idxs, totalk, allowed_topk=None: None
    )
    comms_instance.get_with_retry = AsyncMock()
    totalk_value = 100
    peer_response = (
        {
            "0.weightidxs": torch.tensor([0, 1, 2]),
            "0.weightvals": torch.tensor([0.3, 0.4, 0.5]),
            "totalks": {"0.weight": totalk_value},
        },
        3,
    )
    comms_instance.get_with_retry.side_effect = [peer_response]
    result = await comms_instance.gather(
        my_uid="0",
        uids=["1"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        local=True,
        stale_retention=10,
        totalks={"0.weight": totalk_value},
    )
    assert result is not None


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


@pytest.mark.asyncio
async def test_gather_averaging(comms_instance):
    """Test 8: Verify gradient averaging with multiple peers

    Tests that gradients from multiple peers are properly averaged during gather operation.
    Checks:
    - Proper handling of totalks parameter
    - Correct aggregation of peer responses
    - Validation of UIDs and global steps
    - Tensor shape and size validation
    """
    # Mock check_compressed_indices as specified.
    comms_instance.check_compressed_indices = (
        lambda param_name, idxs, totalk, allowed_topk=None: None
    )

    # Patch get_with_retry to simulate two peer responses.
    comms_instance.get_with_retry = AsyncMock()
    totalk_value = 2  # For key "layer.", allowed_topk = min(3, 2) == 2.
    # Use key with trailing dot so that stripping "idxs" from "layer.idxs" produces "layer."
    peer1_response = (
        {
            "layer.idxs": torch.tensor([0, 1]),
            "layer.vals": torch.tensor([0.6, 0.8]),
            "totalks": {"layer.": totalk_value},  # totalk keyed as "layer."
        },
        1,  # global_step for peer "1"
    )
    peer2_response = (
        {
            "layer.idxs": torch.tensor([0, 1]),
            "layer.vals": torch.tensor([0.6, 0.8]),
            "totalks": {"layer.": totalk_value},  # totalk keyed as "layer."
        },
        2,  # global_step for peer "2"
    )
    comms_instance.get_with_retry.side_effect = [peer1_response, peer2_response]

    # Pass totalks via the gather call with key "layer.".
    totalks_arg = {"layer.": totalk_value}
    result = await comms_instance.gather(
        my_uid="0",
        uids=["1", "2"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        totalks=totalks_arg,
    )

    # Validate the aggregated result.
    assert result is not None, "Expected a non-None gather result"
    assert result.uids == ["1", "2"], f"Expected UIDs ['1', '2'], got {result.uids}"
    assert result.global_steps == [1, 2], (
        f"Expected global_steps [1, 2] got {result.global_steps}"
    )

    aggregated = result.state_dict.__dict__
    for key in ["layer.idxs", "layer.vals"]:
        assert key in aggregated, f"Expected key {key} in state_dict"
        assert len(aggregated[key]) == 2, (
            f"Expected 2 tensors for key {key}, got {len(aggregated[key])}"
        )


async def test_gather_complex_normalization(comms_instance):
    """Test 8: Verify complex gradient normalization with multiple peers

    Tests normalization of gradients with different scales and signs.
    Checks:
    - Proper normalization of tensors with different magnitudes
    - Correct handling of different signs in gradients
    - Validation of aggregated results against expected values
    - Proper handling of multiple peer responses
    """
    # Bypass the compressed indices validation for this test.
    comms_instance.check_compressed_indices = (
        lambda param_name, idxs, totalk, allowed_topk=None: None
    )

    totalk_value = (
        3  # For three indices, allowed_topk = min(topk_compression, totalk_value) = 3
    )
    # Include totalks in each peer response using the key "layer." (so that stripping "idxs"/"vals" returns the same base key).
    peer1_response = (
        {
            "layer.idxs": torch.tensor([0, 1, 2]),
            "layer.vals": torch.tensor([1.0, 2.0, 2.0]),  # norm ≈ 3
            "totalks": {"layer.": totalk_value},
        },
        1,
    )
    peer2_response = (
        {
            "layer.idxs": torch.tensor([0, 1, 2]),
            "layer.vals": torch.tensor([10.0, 20.0, 20.0]),  # Larger scale
            "totalks": {"layer.": totalk_value},
        },
        2,
    )
    peer3_response = (
        {
            "layer.idxs": torch.tensor([0, 1, 2]),
            "layer.vals": torch.tensor([-5.0, 5.0, 5.0]),  # Different sign
            "totalks": {"layer.": totalk_value},
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
        totalks={"layer.": totalk_value},
    )

    assert result is not None
    # Get all normalized tensors from the aggregated state dictionary.
    normalized_tensors = getattr(result.state_dict, "layer.vals")
    actual_vals = torch.stack(normalized_tensors).mean(dim=0)

    # Calculate expected normalized values.
    eps = 1e-8  # Small epsilon to avoid division by zero.
    norm1 = torch.norm(peer1_response[0]["layer.vals"])
    norm2 = torch.norm(peer2_response[0]["layer.vals"])
    norm3 = torch.norm(peer3_response[0]["layer.vals"])

    normalized1 = peer1_response[0]["layer.vals"] / (norm1 + eps)
    normalized2 = peer2_response[0]["layer.vals"] / (norm2 + eps)
    normalized3 = peer3_response[0]["layer.vals"] / (norm3 + eps)
    expected_vals = torch.stack([normalized1, normalized2, normalized3]).mean(dim=0)

    # Debug prints (optional)
    print(f"Peer 1 normalized: {normalized1}")
    print(f"Peer 2 normalized: {normalized2}")
    print(f"Peer 3 normalized: {normalized3}")
    print(f"Expected average: {expected_vals}")
    print(f"Actual result: {actual_vals}")

    # Floating point comparisons with tolerances.
    assert torch.allclose(actual_vals, expected_vals, rtol=1e-3, atol=1e-3)
    # Additional assertions to verify that all peers were processed.
    assert len(normalized_tensors) == 3, (
        f"Expected 3 normalized tensors, got {len(normalized_tensors)}"
    )
    assert len(result.uids) == 3, f"Expected 3 valid UIDs, got {len(result.uids)}"


# Test Initialization and Cleanup
async def test_comms_init(comms_instance):
    """Test 10: Verify proper initialization of Comms instance

    Tests that all required components are properly initialized.
    Checks:
    - Temporary directory creation
    - Save location existence
    - Lock initialization
    - Active peers set initialization
    """
    assert os.path.exists(comms_instance.temp_dir)
    assert os.path.exists(comms_instance.save_location)
    assert comms_instance.lock is not None
    assert isinstance(comms_instance.active_peers, set)


async def test_cleanup_local_data(comms_instance):
    """Test 11: Verify cleanup of stale local data

    Tests the cleanup functionality for old local data.
    Checks:
    - Proper removal of old data based on window
    - Retention of recent data
    - Directory structure maintenance
    """
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
    """Test 12: Verify S3 upload for small files

    Tests the basic S3 upload functionality for small files.
    Checks:
    - Proper file creation
    - S3 client initialization
    - Upload operation execution
    - Cleanup after upload
    """
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
    """Test 13: Verify S3 multipart upload for large files

    Tests the multipart upload functionality for large files.
    Checks:
    - Multipart upload initialization
    - Proper part uploading
    - Upload completion
    - Part number ordering
    - Cleanup operations
    """
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
    """Test 14: Verify downloading of large files

    Tests the chunked download functionality for large files.
    Checks:
    - Proper content length handling
    - Chunk size calculations
    - Range request handling
    - Download completion
    """
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
# @pytest.mark.asyncio
# async def test_load_checkpoint_success(comms_instance):
#     """Test 15: Verify successful checkpoint loading

#     Tests the complete checkpoint loading process.
#     Checks:
#     - Model state dict loading
#     - Optimizer state loading
#     - Scheduler state loading
#     - Momentum handling
#     - Global step tracking
#     - Window management
#     """
#     # Create mock model and parameters
#     model = MagicMock()
#     test_param = torch.nn.Parameter(torch.randn(10))
#     model.named_parameters.return_value = [("layer1", test_param)]

#     # Create mock optimizer and scheduler
#     optimizer = MagicMock()
#     optimizer.state = {}  # Add empty state dict
#     scheduler = MagicMock()
#     scheduler.last_epoch = 0  # Add last_epoch attribute
#     transformer = MagicMock()
#     compressor = MagicMock()

#     # Mock checkpoint data with all required fields
#     checkpoint_data = {
#         "model_state_dict": {"layer1": torch.randn(10)},
#         "optimizer_state_dict": {
#             "state": {0: {"step": 100}},
#             "param_groups": [{"lr": 0.001}],  # Add param_groups
#         },
#         "scheduler_state_dict": {"last_epoch": 0},
#         "momentum": {"layer1": torch.randn(10)},
#         "global_step": 100,
#         "start_window": 1,
#         "current_window": 5,
#     }

#     # Mock get_latest_checkpoint result
#     comms_instance.get_latest_checkpoint = AsyncMock(
#         return_value=(checkpoint_data, 5)  # Return tuple of (data, window)
#     )

#     # Mock model's load_state_dict
#     model.load_state_dict = MagicMock()

#     # Mock optimizer and scheduler load_state_dict
#     optimizer.load_state_dict = MagicMock()
#     scheduler.load_state_dict = MagicMock()

#     # Mock gather for catch-up phase
#     comms_instance.gather = AsyncMock(
#         return_value=SimpleNamespace(
#             state_dict=SimpleNamespace(
#                 **{
#                     "layer1idxs": [torch.tensor([0, 1])],
#                     "layer1vals": [torch.tensor([0.1, 0.2])],
#                 }
#             ),
#             uids=["1"],
#             global_steps=[100],
#         )
#     )

#     # Add shape information to transformer mock
#     transformer.shapes = {"layer1": torch.Size([10])}
#     transformer.totalks = {"layer1": 10}
#     transformer.decode.return_value = torch.randn(10)

#     # Add debug prints
#     print("\nBefore loading checkpoint...")

#     with (
#         patch("tplr.logger.error") as mock_error,
#         patch("tplr.logger.info") as mock_info,
#         patch("tplr.logger.debug") as mock_debug,
#         patch("tplr.logger.warning") as mock_warning,
#     ):


#         totalks = {}
#         for n, p in model.named_parameters():
#             _, _, _, totalk = compressor.compress(
#                 transformer.encode(torch.zeros_like(p)),
#                 hparams.topk_compression
#             )
#             totalks[n] = totalk
#         success, momentum, step, opt, sched = await comms_instance.load_checkpoint(
#             model=model,
#             optimizer=optimizer,
#             scheduler=scheduler,
#             transformer=transformer,
#             compressor=compressor,
#             current_window=10,
#             device="cpu",
#             peers=[1, 2],
#             uid="0",
#             totalks=totalks,
#         )

#         # Print any error logs that occurred
#         print("\nError logs:")
#         for call in mock_error.call_args_list:
#             print(f"Error: {call.args[0]}")

#         print("\nWarning logs:")
#         for call in mock_warning.call_args_list:
#             print(f"Warning: {call.args[0]}")

#         print(f"\nSuccess: {success}")
#         print(f"Step: {step}")

#     assert success, "Checkpoint loading failed"
#     assert isinstance(momentum, dict)
#     assert "layer1" in momentum
#     assert step > 0
#     assert opt == optimizer
#     assert sched == scheduler


@pytest.mark.asyncio
async def test_load_checkpoint_missing_data(comms_instance):
    """Test 16: Verify checkpoint loading with missing data

    Tests the checkpoint loading behavior when data is missing.
    Checks:
    - Proper handling of missing checkpoint data
    - Default value returns
    - Error handling
    - State preservation
    """
    # Mock the get_latest_checkpoint method to return None without error
    comms_instance.get_latest_checkpoint = AsyncMock(return_value=None)

    # Mock get_validator_with_highest_stake to avoid bucket access
    comms_instance.get_validator_with_highest_stake = AsyncMock(return_value=(0, 1.0))

    # Create mock model and optimizer
    mock_model = MagicMock()
    mock_optimizer = MagicMock()
    mock_scheduler = MagicMock()

    # The function returns 5 values: success, momentum, global_step, optimizer, scheduler
    # Using empty totalks (or a dummy dict) for missing data test
    totalks = {}
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
        totalks=totalks,
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
    """Test 17: Verify gather operation timeout handling

    Tests the timeout mechanism in gather operations.
    Checks:
    - Proper timeout handling
    - Error response
    - Resource cleanup
    """

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
            totalks={},
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
        totalks={},
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
async def test_valid_response_handling(comms_instance):
    # Patch out the compressed indices check
    comms_instance.check_compressed_indices = (
        lambda param_name, idxs, totalk, allowed_topk=None: None
    )

    # Patch get_with_retry to simulate three valid peer responses.
    comms_instance.get_with_retry = AsyncMock()
    totalk_value = 100
    peer1_response = (
        {
            "0.weightidxs": torch.tensor([0, 1]),
            "0.weightvals": torch.tensor([0.3, 0.4]),
            "totalks": {"0.weight": totalk_value},
        },
        10,
    )
    peer2_response = (
        {
            "0.weightidxs": torch.tensor([0, 1]),
            "0.weightvals": torch.tensor([0.5, 0.6]),
            "totalks": {"0.weight": totalk_value},
        },
        20,
    )
    peer3_response = (
        {
            "0.weightidxs": torch.tensor([0, 1]),
            "0.weightvals": torch.tensor([0.7, 0.8]),
            "totalks": {"0.weight": totalk_value},
        },
        30,
    )
    comms_instance.get_with_retry.side_effect = [
        peer1_response,
        peer2_response,
        peer3_response,
    ]

    totalks_arg = {"0.weight": totalk_value}
    result = await comms_instance.gather(
        my_uid="dummy_uid",
        uids=["uid1", "uid2", "uid3"],
        window=1,
        key="gradient",
        timeout=5,
        device="cpu",
        totalks=totalks_arg,
    )

    assert result is not None, "Expected gather result to be non-None"


@pytest.mark.asyncio
async def test_missing_idxs_key(comms_instance, model):
    """
    Test 2: Missing "idxs" Key for a Parameter

    Setup:
      - Simulate a UID response that includes "<param_name>vals" but with "<param_name>idxs" explicitly set to None.
    Expected Outcome:
      - The gradient decoding check fails for that UID.
      - That UID is skipped and is added to skipped_uids.
      - This UID does not contribute to the aggregated state_dict.
    """
    comms = comms_instance  # use provided fixture
    device = "cpu"

    # Precompute dummy xshapes and totalks from model parameters.
    xshapes, totalks = {}, {}
    for name, param in model.named_parameters():
        xshapes[name] = param.shape
        totalks[name] = param.numel()

    # Define dummy UIDs.
    uids = ["uid1", "uid2", "uid3"]

    # Helper: create valid state_dict.
    def create_valid_state_dict():
        state_dict = {}
        for name, param in model.named_parameters():
            state_dict[name + "idxs"] = torch.tensor([0, 1], dtype=torch.long)
            state_dict[name + "vals"] = torch.tensor([0.1, 0.2], dtype=torch.float32)
        return state_dict

    # Helper: create state_dict with missing indices (set to None) instead of omitting the key.
    def create_missing_idxs_state_dict():
        state_dict = {}
        for name, param in model.named_parameters():
            # Explicitly set "idxs" key to None.
            state_dict[name + "idxs"] = None
            state_dict[name + "vals"] = torch.tensor([0.1, 0.2], dtype=torch.float32)
        return state_dict

    # Simulate responses: uid1 returns invalid state_dict (with None in "idxs"), others are valid.
    responses = [
        (
            create_missing_idxs_state_dict(),
            10,
        ),  # UID "uid1": missing indices → should be skipped.
        (create_valid_state_dict(), 20),  # UID "uid2": valid.
        (create_valid_state_dict(), 30),  # UID "uid3": valid.
    ]
    call_count = 0

    async def mock_get_with_retry(*args, **kwargs):
        nonlocal call_count
        if call_count < len(responses):
            resp = responses[call_count]
            call_count += 1
            return resp
        return None

    comms.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

    # Patch check_compressed_indices: Raise error if the provided indices object is None.
    def patched_check(param_name, idxs, totalk, allowed_topk=None):
        if idxs is None:
            raise ValueError(f"Missing indices for {param_name}")
        return None

    comms.check_compressed_indices = patched_check

    # Call gather() with our simulated responses.
    result = await comms.gather(
        my_uid="dummy_uid",
        uids=uids,
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        totalks=totalks,
    )

    # Validate the result.
    # Expect only valid UIDs "uid2" and "uid3" to be present.
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
            • Simulate a UID response with "<param_name>idxs" present but with
              "<param_name>vals" explicitly set to None.
      - Expected Outcome:
            • The UID with missing "vals" is skipped.
            • Only UIDs with valid state dicts contribute to the aggregated gradients.
    """
    comms = comms_instance
    device = "cpu"

    # Precompute totalks for each model parameter.
    totalks = {}
    for name, param in model.named_parameters():
        totalks[name] = param.numel()

    # Define dummy UIDs.
    uids = ["uid1", "uid2", "uid3"]

    # Helper: create a valid state_dict with both keys.
    def create_valid_state_dict():
        state_dict = {}
        for name, _ in model.named_parameters():
            state_dict[name + "idxs"] = torch.tensor([0, 1], dtype=torch.long)
            state_dict[name + "vals"] = torch.tensor([0.1, 0.2], dtype=torch.float32)
        return state_dict

    # Helper: create a state_dict where the "vals" key is explicitly set to None.
    def create_missing_vals_state_dict():
        state_dict = {}
        for name, _ in model.named_parameters():
            state_dict[name + "idxs"] = torch.tensor([0, 1], dtype=torch.long)
            state_dict[name + "vals"] = None  # Simulate missing values.
        return state_dict

    # Simulated responses:
    #  - uid1 returns an invalid state_dict (missing vals)
    #  - uid2 and uid3 return valid state_dicts.
    responses = [
        (create_missing_vals_state_dict(), 10),
        (create_valid_state_dict(), 20),
        (create_valid_state_dict(), 30),
    ]
    call_count = 0

    async def mock_get_with_retry(*args, **kwargs):
        nonlocal call_count
        if call_count < len(responses):
            state_dict, global_step = responses[call_count]
            try:
                for key in state_dict:
                    if key.endswith("vals") and state_dict[key] is None:
                        raise ValueError(f"Missing value for {key}")
            except ValueError as e:
                call_count += 1  # Ensure we advance even if an error occurs.
                raise e
            call_count += 1
            return state_dict, global_step
        return None

    comms.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)
    # Patch check_compressed_indices as a no-op.
    comms.check_compressed_indices = (
        lambda param_name, data, totalk, allowed_topk=None: None
    )

    result = await comms.gather(
        my_uid="dummy_uid",
        uids=uids,
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        totalks=totalks,
    )

    assert result is not None, "Expected non-None result from gather()"
    assert result.uids == ["uid2", "uid3"], (
        f"Expected valid_uids ['uid2', 'uid3'], got {result.uids}"
    )


@pytest.mark.asyncio
async def test_empty_or_none_state_dict(comms_instance, model):
    """
    Test 4: Empty or None state_dict
      - Setup:
            • Use AsyncMock to have get_with_retry return a valid response for the first UID and
              None (or an empty dict) for subsequent UIDs.
      - Expected Outcome:
            • Only the UID that returns a valid state_dict is aggregated.
            • The remaining UIDs are skipped.
            • Global steps reflect only valid responses.
    """
    comms = comms_instance
    device = "cpu"

    # Helper to compute xshapes and totalks from model parameters.
    def create_xshapes_totalks(model):
        xshapes = {}
        totalks = {}
        for name, param in model.named_parameters():
            xshapes[name] = param.shape
            totalks[name] = param.numel()
        return xshapes, totalks

    # Helper to create a valid state_dict.
    def create_valid_state_dict(model):
        state_dict = {}
        for name, _ in model.named_parameters():
            state_dict[name + "idxs"] = torch.tensor([0, 1], dtype=torch.long)
            state_dict[name + "vals"] = torch.tensor([0.1, 0.2], dtype=torch.float32)
        return state_dict

    xshapes, totalks = create_xshapes_totalks(model)

    # Patch check_compressed_indices to be a no-op so that valid responses won't be rejected.
    comms.check_compressed_indices = (
        lambda param_name, idxs, totalk, allowed_topk=None: None
    )

    # Define dummy UIDs.
    uids = ["uid1", "uid2", "uid3"]
    call_count = 0

    async def mock_get_with_retry(*args, **kwargs):
        nonlocal call_count
        if call_count == 0:
            ret = (create_valid_state_dict(model), 10)
        elif call_count == 1:
            ret = None
        elif call_count == 2:
            ret = None  # Instead of returning an empty dict, return None.
        else:
            ret = None
        call_count += 1
        return ret

    comms.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

    result = await comms.gather(
        my_uid="dummy_uid",
        uids=uids,
        window=1,
        key="gradient",
        timeout=5,
        device=device,
        local=False,
        totalks=totalks,
    )

    # Since only UID "uid1" returns a valid response,
    # valid_uids should be ["uid1"].
    assert result is not None, "Expected a non-None result."
    assert result.uids == ["uid1"], f"Expected valid_uids ['uid1'], got {result.uids}"


# Dummy hparams with topk_compression set to 3.
class DummyHParams:
    topk_compression = 3


# Dummy Comms instance that only supplies hparams for testing.
class DummyComms(Comms):
    def __init__(self):
        # Only initialization required for testing check_compressed_indices.
        self.hparams = DummyHParams()


def test_valid_flat_tensor():
    """
    Test Case: test_valid_flat_tensor
      - Input: A 1D tensor (torch.Tensor) with length equal to min(hparams.topk_compression, totalk).
      - Valid indices (all indices within [0, totalk-1]).
      - Expected Outcome: The function should complete without raising an error.
    """
    dummy_comms = DummyComms()

    # totalk is set to 10; allowed_topk is min(3, 10) == 3.
    totalk = 10
    valid_tensor = torch.tensor([1, 5, 9], dtype=torch.long)

    # This call should complete without any error.
    dummy_comms.check_compressed_indices("test_param", valid_tensor, totalk)


def test_valid_multi_dim_tensor():
    """
    Test that a multi-dimensional tensor (e.g., 2D tensor) where the last dimension equals min(hparams.topk_compression, totalk)
    and all indices are within the valid range completes without raising an error.
    """
    dummy_comms = DummyComms()
    totalk = 20  # allowed_topk = min(3, 20) = 3
    # Create a valid 2D tensor (shape: 2 x 3) with valid indices.
    valid_tensor = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    dummy_comms.check_compressed_indices("param", valid_tensor, totalk)


def test_valid_flat_list():
    """
    Test that a flat Python list of integers with length equal to allowed_topk
    and valid indices completes without raising an error.
    """
    dummy_comms = DummyComms()
    totalk = 20  # allowed_topk = min(3, 20) = 3
    valid_list = [0, 1, 19]  # All values are within [0, totalk)
    dummy_comms.check_compressed_indices("param", valid_list, totalk)


def test_valid_nested_list():
    """
    Test that a nested list (list of lists), where each inner list has length equal to allowed_topk
    and contains valid indices completes without raising an error.
    """
    dummy_comms = DummyComms()
    totalk = 20  # allowed_topk = 3
    valid_nested = [[0, 1, 2], [3, 4, 5]]
    dummy_comms.check_compressed_indices("param", valid_nested, totalk)


def test_invalid_flat_tensor_wrong_length():
    """
    Test that a 1D tensor whose length does not equal min(hparams.topk_compression, totalk) (e.g., too short)
    raises ValueError with message about invalid number of indices.
    """
    dummy_comms = DummyComms()
    totalk = 10  # allowed_topk = min(3, 10) = 3
    invalid_tensor = torch.tensor([0, 1], dtype=torch.long)  # Length is 2, should be 3.
    with pytest.raises(ValueError, match="Invalid number of indices"):
        dummy_comms.check_compressed_indices("param", invalid_tensor, totalk)


def test_invalid_multi_dim_tensor_wrong_last_dimension():
    """
    Test that a multi-dimensional tensor where the size of the last dimension is not equal to min(hparams.topk_compression, totalk)
    raises ValueError indicating the last dimension size is invalid.
    """
    dummy_comms = DummyComms()
    totalk = 20  # allowed_topk = min(3, 20) = 3
    # Create a 2D tensor with last dimension size 4 (should be 3)
    invalid_tensor = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
    with pytest.raises(ValueError, match="Last dimension size invalid"):
        dummy_comms.check_compressed_indices("param", invalid_tensor, totalk)


def test_invalid_flat_list_negative_index():
    """
    Test that a flat list with one or more indices being negative raises ValueError indicating that an index is out of bounds.
    """
    dummy_comms = DummyComms()
    totalk = 10  # allowed_topk = min(3, 10) = 3
    invalid_list = [0, -1, 2]  # Contains a negative index.
    with pytest.raises(ValueError, match="Index -1 out of bounds"):
        dummy_comms.check_compressed_indices("param", invalid_list, totalk)


def test_invalid_flat_tensor_out_of_range_index():
    """
    Test that a flat tensor with an index equal to or greater than totalk raises ValueError indicating index out of bounds.
    """
    dummy_comms = DummyComms()
    totalk = 10  # allowed_topk = min(3, 10) = 3
    # Index 10 is out-of-range because valid indices are 0 to 9.
    invalid_tensor = torch.tensor([0, 1, 10], dtype=torch.long)
    with pytest.raises(ValueError, match="Index 10 out of bounds"):
        dummy_comms.check_compressed_indices("param", invalid_tensor, totalk)


def test_invalid_flat_list_wrong_length():
    """
    Test that a flat list whose length is not equal to allowed_topk raises ValueError about the invalid number of indices.
    """
    dummy_comms = DummyComms()
    totalk = 20  # allowed_topk = min(3, 20) = 3
    invalid_list = [0, 1]  # Only 2 elements instead of 3.
    with pytest.raises(ValueError, match="Invalid number of indices"):
        dummy_comms.check_compressed_indices("param", invalid_list, totalk)


def test_invalid_nested_list_wrong_length():
    """
    Test that a nested list where at least one sublist has a length different from allowed_topk raises ValueError on the offending sublist.
    """
    dummy_comms = DummyComms()
    totalk = 20  # allowed_topk = 3
    # The second sublist has only 2 elements.
    invalid_nested = [[0, 1, 2], [3, 4]]
    with pytest.raises(ValueError, match="Invalid number of indices"):
        dummy_comms.check_compressed_indices("param", invalid_nested, totalk)


def test_valid_single_value():
    """
    Test that a single valid integer index (not wrapped in a list or tensor) within the range [0, totalk-1] completes without raising an error.
    """
    dummy_comms = DummyComms()
    totalk = 5
    valid_single = 3  # Valid since 0 <= 3 < 5.
    dummy_comms.check_compressed_indices("param", valid_single, totalk)


def test_invalid_single_value_out_of_bounds():
    """
    Test that a single integer index that is out-of-bounds raises ValueError indicating the index is out of bounds.
    """
    dummy_comms = DummyComms()
    totalk = 5
    # Index 5 is out-of-bounds because valid indices are 0 to 4.
    with pytest.raises(ValueError, match="Index 5 out of bounds"):
        dummy_comms.check_compressed_indices("param", 5, totalk)


def test_override_allowed_topk():
    """
    Test using the optional allowed_topk parameter to override hparams.topk_compression:
    - Call with a flat list matching the provided allowed_topk -> Should pass
    - Call with a list of a different length than allowed_topk -> Should raise ValueError
    """
    dummy_comms = DummyComms()
    totalk = 10
    # Override allowed_topk to 2.
    valid_list = [0, 9]  # Correct length: 2 elements.
    dummy_comms.check_compressed_indices("param", valid_list, totalk, allowed_topk=2)

    invalid_list = [0, 1, 2]  # Incorrect length: 3 elements instead of 2.
    with pytest.raises(ValueError, match="Invalid number of indices"):
        dummy_comms.check_compressed_indices(
            "param", invalid_list, totalk, allowed_topk=2
        )


def test_topk_auto_adjust_when_totalk_is_lower():
    """
    Test scenario where totalk is less than hparams.topk_compression:
    - Provide a flat list with length equal to totalk (which is the adjusted allowed_topk) -> Should pass
    - Test a list with fewer elements than totalk -> Should raise ValueError
    """
    dummy_comms = DummyComms()
    totalk = 2  # Now allowed_topk becomes min(hparams.topk_compression, totalk) = min(3,2) = 2.
    valid_list = [0, 1]  # Valid: length matches allowed_topk (which is 2).
    dummy_comms.check_compressed_indices("param", valid_list, totalk)

    invalid_list = [0]  # Too few elements.
    with pytest.raises(ValueError, match="Invalid number of indices"):
        dummy_comms.check_compressed_indices("param", invalid_list, totalk)
