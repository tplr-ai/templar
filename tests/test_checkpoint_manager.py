# === CheckpointManager.__init__ Tests ===
# Test initialization with all required parameters
# Test initialization with optional metagraph and commitments
# Test initialization with empty commitments dict
# Test last_checkpoint_data initializes to None

# === CheckpointManager._move_to_cpu Tests ===
# Test moving single tensor to CPU
# Test moving nested dict with tensors to CPU
# Test moving list of tensors to CPU
# Test moving tuple of tensors to CPU
# Test moving mixed nested structure (dict with lists of tensors)
# Test moving non-tensor objects (strings, ints, None) - should return unchanged
# Test moving empty dict/list/tuple
# Test moving deeply nested structures (dict of dict of tensors)

# === CheckpointManager.save_checkpoint Tests ===
# Test successful checkpoint save with all parameters
# Test CPU tensor conversion for model state dict
# Test CPU tensor conversion for optimizer state dict with nested tensors
# Test temp file creation and cleanup on success
# Test temp file cleanup on failure
# Test torch.save with pickle.HIGHEST_PROTOCOL
# Test R2 upload success
# Test R2 upload failure
# Test file read failure after torch.save
# Test filename generation with special characters in uid
# Test filename generation with version string
# Test exception during checkpoint data preparation
# Test exception during temp file operations
# Test logging for success and failure cases

# === CheckpointManager.load_checkpoint Tests ===
# Test successful checkpoint load with device movement
# Test no checkpoints found scenario
# Test invalid checkpoint format (missing keys)
# Test model.load_state_dict failure (key mismatch)
# Test model.load_state_dict failure (size mismatch)
# Test optimizer state loading and device movement
# Test scheduler state loading
# Test device movement for optimizer tensors in nested state
# Test missing start_window in checkpoint data
# Test missing current_window in checkpoint data
# Test missing sync_window in checkpoint data (should default to 0)
# Test last_checkpoint_data is set after successful load
# Test return values for success case
# Test return values for failure cases
# Test with custom init_version parameter
# Test with None init_version (should use __version__)

# === CheckpointManager.get_latest_checkpoint Tests ===
# Test validator checkpoint found and returned
# Test validator not found, self R2 checkpoint found
# Test validator not found, self R2 not found, returns None
# Test validator bucket lookup failure
# Test validator bucket exists but no checkpoint
# Test self bucket checkpoint fallback
# Test exception handling during validator lookup
# Test exception handling during bucket checkpoint retrieval
# Test logging for each checkpoint source

# === CheckpointManager._get_bucket_checkpoint Tests ===
# Test successful checkpoint retrieval with highest window number
# Test no matching checkpoint files found
# Test pattern matching with various uid formats
# Test pattern matching with version string escaping
# Test multiple checkpoints, returns highest window number
# Test corrupted checkpoint file, falls back to next valid one
# Test all checkpoints corrupted, returns None
# Test storage client list_objects failure
# Test storage client get_object failure
# Test torch.load failure with weights_only=False
# Test temp file creation failure
# Test temp file cleanup after successful load
# Test temp file cleanup after failed load
# Test regex pattern with special characters in uid
# Test regex pattern matching edge cases
# Test checkpoint with window number 0
# Test checkpoint with very large window numbers
# Test empty checkpoint list from storage

# === CheckpointManager.cleanup_old_checkpoints Tests ===
# Test successful cleanup keeping specified number of checkpoints
# Test cleanup with fewer checkpoints than keep_last (no deletion)
# Test cleanup with exact number of checkpoints as keep_last
# Test cleanup with pattern matching for current uid only
# Test cleanup with mixed checkpoint files (different uids/versions)
# Test partial deletion failures (some succeed, some fail)
# Test all deletion failures
# Test storage client list_objects failure
# Test storage client delete_object failure
# Test concurrent deletion execution
# Test logging for successful deletions count
# Test exception handling during cleanup process
# Test default keep_last=3 parameter
# Test custom keep_last parameter
# Test cleanup with no matching checkpoints

# === CheckpointManager._get_highest_stake_validator_bucket Tests ===
# Test successful validator bucket retrieval
# Test no metagraph available
# Test metagraph with no validators (empty S tensor)
# Test metagraph.S.argmax() returns None
# Test validator uid not in commitments dict
# Test validator uid in commitments with valid bucket
# Test exception during metagraph.S.argmax()
# Test exception during commitments lookup
# Test logging for validator not found
# Test logging for no bucket committed
# Test logging for successful validator bucket found
# Test return types (Bucket, int) vs (None, None)

# === Integration Tests ===
# Test complete save -> load cycle
# Test save -> cleanup -> load cycle
# Test load from validator -> fallback to self -> fallback to None
# Test checkpoint data persistence across save/load
# Test device consistency after load
# Test optimizer state preservation across save/load
# Test scheduler state preservation across save/load
# Test window number tracking across operations
# Test multiple concurrent checkpoint operations
# Test checkpoint manager with different storage backends
# Test checkpoint manager with different file managers

# === Error Handling and Edge Cases ===
# Test with malformed bucket objects
# Test with invalid uid types
# Test with extremely large checkpoint data
# Test with empty model state dict
# Test with empty optimizer state dict
# Test with corrupted temporary files
# Test with insufficient disk space scenarios
# Test with network failures during R2 operations
# Test with invalid device strings
# Test with missing torch dependencies
# Test memory cleanup after large checkpoint operations
# Test timeout scenarios for long-running operations
# Test concurrent access to same checkpoint files
# Test checkpoint manager destruction and cleanup

# === Mock and Fixture Requirements ===
# Mock storage_client with configurable success/failure responses
# Mock file_manager with temp file creation/deletion tracking
# Mock torch.nn.Module with controllable state_dict
# Mock torch.optim.Optimizer with nested tensor states
# Mock torch.optim.lr_scheduler.LRScheduler
# Mock metagraph with configurable stake values
# Mock commitments dict with validator buckets
# Mock torch.save/load with controllable success/failure
# Mock asyncio operations for concurrent testing
# Fixture for temporary directories and cleanup
# Fixture for sample checkpoint data with various formats
# Fixture for device testing (CPU/CUDA simulation)

import pytest
import torch
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import pickle

from src.tplr.training.checkpoint_manager import CheckpointManager
from src.tplr.schemas import Bucket


# === Core Fixtures ===


@pytest.fixture
def mock_storage_client():
    """Mock storage client with configurable success/failure responses"""
    client = AsyncMock()

    # Default successful responses
    client.put_object.return_value = True
    client.get_object.return_value = b"mock_checkpoint_data"
    client.list_objects.return_value = []
    client.delete_object.return_value = True

    # Add method to configure responses
    def configure_responses(**kwargs):
        for method, response in kwargs.items():
            getattr(client, method).return_value = response

    client.configure_responses = configure_responses
    return client


@pytest.fixture
def mock_file_manager():
    """Mock file manager with temp file creation/deletion tracking"""
    manager = Mock()

    # Track created and deleted files
    manager.created_files = []
    manager.deleted_files = []

    def create_temp_file(prefix="test", suffix=".tmp"):
        # Create actual temp file for realistic testing
        temp_file = tempfile.NamedTemporaryFile(
            prefix=prefix, suffix=suffix, delete=False
        )
        temp_path = temp_file.name
        temp_file.close()
        manager.created_files.append(temp_path)
        return temp_path

    def delete_file(filepath):
        manager.deleted_files.append(filepath)
        try:
            Path(filepath).unlink(missing_ok=True)
        except Exception:
            pass

    manager.create_temp_file.side_effect = create_temp_file
    manager.delete_file.side_effect = delete_file

    return manager


@pytest.fixture
def mock_bucket():
    """Mock bucket object"""
    bucket = Mock(spec=Bucket)
    bucket.name = "test-bucket"
    bucket.region = "us-east-1"
    bucket.endpoint = "https://test.r2.dev"
    return bucket


@pytest.fixture
def mock_model():
    """Mock torch.nn.Module with controllable state_dict"""
    model = Mock(spec=torch.nn.Module)

    # Default state dict with various tensor types
    default_state = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(1, 10),
        "layer2.bias": torch.randn(1),
    }

    model.state_dict.return_value = default_state
    model.load_state_dict = Mock()
    model.to = Mock(return_value=model)

    # Method to customize state dict
    def set_state_dict(state_dict):
        model.state_dict.return_value = state_dict

    model.set_state_dict = set_state_dict
    return model


@pytest.fixture
def mock_optimizer():
    """Mock torch.optim.Optimizer with nested tensor states"""
    optimizer = Mock(spec=torch.optim.Optimizer)

    # Create realistic optimizer state with nested tensors
    param_state = {
        0: {
            "step": torch.tensor(100),
            "exp_avg": torch.randn(10, 5),
            "exp_avg_sq": torch.randn(10, 5),
        },
        1: {
            "step": torch.tensor(100),
            "exp_avg": torch.randn(10),
            "exp_avg_sq": torch.randn(10),
        },
    }

    optimizer_state = {
        "state": param_state,
        "param_groups": [
            {
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0,
                "amsgrad": False,
            }
        ],
    }

    optimizer.state_dict.return_value = optimizer_state
    optimizer.load_state_dict = Mock()
    optimizer.state = {}  # Will be populated during load

    # Method to customize optimizer state
    def set_state_dict(state_dict):
        optimizer.state_dict.return_value = state_dict

    optimizer.set_state_dict = set_state_dict
    return optimizer


@pytest.fixture
def mock_scheduler():
    """Mock torch.optim.lr_scheduler.LRScheduler"""
    scheduler = Mock(spec=torch.optim.lr_scheduler.LRScheduler)

    scheduler_state = {
        "last_epoch": 99,
        "_step_count": 100,
        "_get_lr_called_within_step": False,
        "_last_lr": [0.001],
    }

    scheduler.state_dict.return_value = scheduler_state
    scheduler.load_state_dict = Mock()

    # Method to customize scheduler state
    def set_state_dict(state_dict):
        scheduler.state_dict.return_value = state_dict

    scheduler.set_state_dict = set_state_dict
    return scheduler


@pytest.fixture
def mock_metagraph():
    """Mock metagraph with configurable stake values"""
    metagraph = Mock()

    # Default stakes - validator 0 has highest stake
    stakes = torch.tensor([100.0, 50.0, 75.0, 25.0])
    metagraph.S = stakes

    # Method to configure stakes
    def set_stakes(new_stakes):
        metagraph.S = torch.tensor(new_stakes)

    metagraph.set_stakes = set_stakes
    return metagraph


@pytest.fixture
def mock_commitments(mock_bucket):
    """Mock commitments dict with validator buckets"""
    # Default commitments - validator 0 and 2 have buckets
    commitments = {
        0: mock_bucket,
        2: Mock(spec=Bucket, name="validator-2-bucket"),
    }
    return commitments


@pytest.fixture
def sample_checkpoint_data():
    """Fixture for sample checkpoint data with various formats"""
    base_data = {
        "model_state_dict": {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
        },
        "optimizer_state_dict": {
            "state": {
                0: {
                    "step": torch.tensor(50),
                    "exp_avg": torch.randn(10, 5),
                }
            },
            "param_groups": [{"lr": 0.001}],
        },
        "scheduler_state_dict": {"last_epoch": 49},
        "start_window": 0,
        "current_window": 10,
        "global_step": 500,
        "sync_window": 8,
    }

    # Variants for testing different scenarios
    variants = {
        "complete": base_data,
        "missing_start_window": {
            k: v for k, v in base_data.items() if k != "start_window"
        },
        "missing_current_window": {
            k: v for k, v in base_data.items() if k != "current_window"
        },
        "missing_sync_window": {
            k: v for k, v in base_data.items() if k != "sync_window"
        },
        "empty_model_state": {**base_data, "model_state_dict": {}},
        "empty_optimizer_state": {
            **base_data,
            "optimizer_state_dict": {"state": {}, "param_groups": []},
        },
    }

    return variants


@pytest.fixture
def temp_directory():
    """Fixture for temporary directories and cleanup"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def device_fixture():
    """Fixture for device testing (CPU/CUDA simulation)"""
    devices = {
        "cpu": "cpu",
        "cuda": "cuda:0" if torch.cuda.is_available() else "cpu",
    }
    return devices


# === Specialized Mock Fixtures ===


@pytest.fixture
def failing_storage_client():
    """Storage client configured for failure scenarios"""
    client = AsyncMock()
    client.put_object.return_value = False
    client.get_object.return_value = None
    client.list_objects.side_effect = Exception("Storage unavailable")
    client.delete_object.return_value = False
    return client


@pytest.fixture
def corrupted_checkpoint_storage():
    """Storage client that returns corrupted checkpoint data"""
    client = AsyncMock()
    client.put_object.return_value = True
    client.get_object.return_value = b"corrupted_data_not_valid_torch"
    client.list_objects.return_value = ["checkpoint-5-123-v1.0.0.pt"]
    client.delete_object.return_value = True
    return client


@pytest.fixture
def multiple_checkpoints_storage():
    """Storage client with multiple checkpoint files"""
    client = AsyncMock()
    client.put_object.return_value = True
    client.list_objects.return_value = [
        "checkpoint-10-123-v1.0.0.pt",
        "checkpoint-5-123-v1.0.0.pt",
        "checkpoint-15-123-v1.0.0.pt",
        "checkpoint-8-456-v1.0.0.pt",  # Different UID
        "other-file.txt",  # Non-checkpoint file
    ]

    # Mock get_object to return valid data for the highest checkpoint
    async def mock_get_object(key, bucket):
        if "checkpoint-15-123" in key:
            # Return valid checkpoint data
            temp_data = {
                "model_state_dict": {"test": torch.tensor([1, 2, 3])},
                "optimizer_state_dict": {"state": {}, "param_groups": []},
                "scheduler_state_dict": {},
                "current_window": 15,
                "start_window": 0,
                "sync_window": 12,
            }
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            torch.save(temp_data, temp_file.name)
            with open(temp_file.name, "rb") as f:
                data = f.read()
            Path(temp_file.name).unlink()
            return data
        return None

    client.get_object.side_effect = mock_get_object
    client.delete_object.return_value = True
    return client


@pytest.fixture
def mismatched_model():
    """Model with incompatible state dict for testing load failures"""
    model = Mock(spec=torch.nn.Module)
    model.state_dict.return_value = {"different_layer.weight": torch.randn(5, 3)}

    # Simulate RuntimeError on load_state_dict with wrong keys
    def failing_load(state_dict):
        raise RuntimeError(
            "Error(s) in loading state_dict: Missing key(s) in state_dict"
        )

    model.load_state_dict.side_effect = failing_load
    model.to = Mock(return_value=model)
    return model


@pytest.fixture
def checkpoint_manager_factory(
    mock_storage_client,
    mock_file_manager,
    mock_bucket,
    mock_metagraph,
    mock_commitments,
):
    """Factory to create CheckpointManager with different configurations"""

    def create_manager(
        uid=123,
        storage_client=None,
        file_manager=None,
        bucket=None,
        metagraph=None,
        commitments=None,
    ):
        return CheckpointManager(
            storage_client=storage_client or mock_storage_client,
            file_manager=file_manager or mock_file_manager,
            bucket=bucket or mock_bucket,
            uid=uid,
            metagraph=metagraph if metagraph is not None else mock_metagraph,
            commitments=commitments if commitments is not None else mock_commitments,
        )

    return create_manager


# === Torch Mock Patches ===


@pytest.fixture
def mock_torch_save():
    """Mock torch.save with controllable success/failure"""
    with patch("torch.save") as mock_save:
        # Default success
        mock_save.return_value = None

        # Method to make it fail
        def make_fail():
            mock_save.side_effect = Exception("Disk full")

        def make_succeed():
            mock_save.side_effect = None
            mock_save.return_value = None

        mock_save.make_fail = make_fail
        mock_save.make_succeed = make_succeed
        yield mock_save


@pytest.fixture
def mock_torch_load():
    """Mock torch.load with controllable success/failure"""
    with patch("torch.load") as mock_load:
        # Default success - returns valid checkpoint data
        default_data = {
            "model_state_dict": {"layer.weight": torch.randn(2, 2)},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "scheduler_state_dict": {},
            "current_window": 5,
            "start_window": 0,
            "sync_window": 3,
        }
        mock_load.return_value = default_data

        # Method to customize return data
        def set_return_data(data):
            mock_load.return_value = data

        # Method to make it fail
        def make_fail():
            mock_load.side_effect = Exception("Corrupted file")

        def make_succeed():
            mock_load.side_effect = None
            mock_load.return_value = default_data

        mock_load.set_return_data = set_return_data
        mock_load.make_fail = make_fail
        mock_load.make_succeed = make_succeed
        yield mock_load


# === Cleanup Fixture ===


@pytest.fixture(autouse=True)
def cleanup_temp_files(mock_file_manager):
    """Auto cleanup temp files created during tests"""
    yield
    # Clean up any temp files that weren't properly deleted
    for filepath in mock_file_manager.created_files:
        try:
            Path(filepath).unlink(missing_ok=True)
        except Exception:
            pass


class TestCheckpointManagerInit:
    def test_initialization_with_all_required_parameters(
        self, mock_storage_client, mock_file_manager, mock_bucket
    ):
        """Test initialization with all required parameters"""
        manager = CheckpointManager(
            storage_client=mock_storage_client,
            file_manager=mock_file_manager,
            bucket=mock_bucket,
            uid=123,
        )

        assert manager.storage_client == mock_storage_client
        assert manager.file_manager == mock_file_manager
        assert manager.bucket == mock_bucket
        assert manager.uid == 123
        assert manager.metagraph is None
        assert manager.commitments == {}
        assert manager.last_checkpoint_data is None

    def test_initialization_with_optional_metagraph_and_commitments(
        self,
        mock_storage_client,
        mock_file_manager,
        mock_bucket,
        mock_metagraph,
        mock_commitments,
    ):
        """Test initialization with optional metagraph and commitments"""
        manager = CheckpointManager(
            storage_client=mock_storage_client,
            file_manager=mock_file_manager,
            bucket=mock_bucket,
            uid=456,
            metagraph=mock_metagraph,
            commitments=mock_commitments,
        )

        assert manager.metagraph == mock_metagraph
        assert manager.commitments == mock_commitments
        assert manager.uid == 456

    def test_initialization_with_empty_commitments_dict(
        self, mock_storage_client, mock_file_manager, mock_bucket
    ):
        """Test initialization with empty commitments dict"""
        empty_commitments = {}
        manager = CheckpointManager(
            storage_client=mock_storage_client,
            file_manager=mock_file_manager,
            bucket=mock_bucket,
            uid=789,
            commitments=empty_commitments,
        )

        assert manager.commitments == {}

    def test_last_checkpoint_data_initializes_to_none(self, checkpoint_manager_factory):
        """Test last_checkpoint_data initializes to None"""
        manager = checkpoint_manager_factory()
        assert manager.last_checkpoint_data is None


class TestMoveToCpu:
    def test_moving_single_tensor_to_cpu(self, checkpoint_manager_factory):
        """Test moving single tensor to CPU"""
        manager = checkpoint_manager_factory()
        tensor = (
            torch.randn(3, 4).cuda() if torch.cuda.is_available() else torch.randn(3, 4)
        )

        result = manager._move_to_cpu(tensor)

        assert torch.is_tensor(result)
        assert result.device.type == "cpu"
        assert torch.equal(tensor.cpu(), result)

    def test_moving_nested_dict_with_tensors_to_cpu(self, checkpoint_manager_factory):
        """Test moving nested dict with tensors to CPU"""
        manager = checkpoint_manager_factory()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        nested_dict = {
            "layer1": {
                "weight": torch.randn(2, 3).to(device),
                "bias": torch.randn(2).to(device),
            },
            "layer2": {"weight": torch.randn(1, 2).to(device)},
        }

        result = manager._move_to_cpu(nested_dict)

        assert result["layer1"]["weight"].device.type == "cpu"
        assert result["layer1"]["bias"].device.type == "cpu"
        assert result["layer2"]["weight"].device.type == "cpu"

    def test_moving_list_of_tensors_to_cpu(self, checkpoint_manager_factory):
        """Test moving list of tensors to CPU"""
        manager = checkpoint_manager_factory()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor_list = [torch.randn(2, 2).to(device), torch.randn(3).to(device)]

        result = manager._move_to_cpu(tensor_list)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(t.device.type == "cpu" for t in result)

    def test_moving_tuple_of_tensors_to_cpu(self, checkpoint_manager_factory):
        """Test moving tuple of tensors to CPU"""
        manager = checkpoint_manager_factory()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor_tuple = (torch.randn(2, 2).to(device), torch.randn(3).to(device))

        result = manager._move_to_cpu(tensor_tuple)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(t.device.type == "cpu" for t in result)

    def test_moving_mixed_nested_structure(self, checkpoint_manager_factory):
        """Test moving mixed nested structure (dict with lists of tensors)"""
        manager = checkpoint_manager_factory()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mixed_structure = {
            "tensors": [torch.randn(2, 2).to(device), torch.randn(3).to(device)],
            "nested": {"more_tensors": (torch.randn(1).to(device),), "scalar": 42},
        }

        result = manager._move_to_cpu(mixed_structure)

        assert all(t.device.type == "cpu" for t in result["tensors"])
        assert result["nested"]["more_tensors"][0].device.type == "cpu"
        assert result["nested"]["scalar"] == 42

    def test_moving_non_tensor_objects(self, checkpoint_manager_factory):
        """Test moving non-tensor objects (strings, ints, None) - should return unchanged"""
        manager = checkpoint_manager_factory()

        assert manager._move_to_cpu("string") == "string"
        assert manager._move_to_cpu(42) == 42
        assert manager._move_to_cpu(None) is None
        assert manager._move_to_cpu(3.14) == 3.14

    def test_moving_empty_containers(self, checkpoint_manager_factory):
        """Test moving empty dict/list/tuple"""
        manager = checkpoint_manager_factory()

        assert manager._move_to_cpu({}) == {}
        assert manager._move_to_cpu([]) == []
        assert manager._move_to_cpu(()) == ()

    def test_moving_deeply_nested_structures(self, checkpoint_manager_factory):
        """Test moving deeply nested structures (dict of dict of tensors)"""
        manager = checkpoint_manager_factory()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        deep_structure = {
            "level1": {"level2": {"level3": {"tensor": torch.randn(2, 2).to(device)}}}
        }

        result = manager._move_to_cpu(deep_structure)

        assert result["level1"]["level2"]["level3"]["tensor"].device.type == "cpu"


class TestSaveCheckpoint:
    @pytest.mark.asyncio
    async def test_successful_checkpoint_save(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        mock_torch_save,
    ):
        """Test successful checkpoint save with all parameters"""
        manager = checkpoint_manager_factory()

        result = await manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            global_step=1000,
            current_window=5,
            start_window=0,
            sync_window=3,
        )

        assert result is True
        manager.storage_client.put_object.assert_called_once()
        mock_torch_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_cpu_tensor_conversion_for_model_state(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        mock_torch_save,
    ):
        """Test CPU tensor conversion for model state dict"""
        manager = checkpoint_manager_factory()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set model state with GPU tensors
        gpu_state = {
            "layer.weight": torch.randn(2, 3).to(device),
            "layer.bias": torch.randn(2).to(device),
        }
        mock_model.set_state_dict(gpu_state)

        await manager.save_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 1000, 5, 0, 3
        )

        # Verify torch.save was called with CPU tensors
        save_call_args = mock_torch_save.call_args[0][0]
        for tensor in save_call_args["model_state_dict"].values():
            assert tensor.device.type == "cpu"

    @pytest.mark.asyncio
    async def test_cpu_tensor_conversion_for_optimizer_state(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        mock_torch_save,
    ):
        """Test CPU tensor conversion for optimizer state dict with nested tensors"""
        manager = checkpoint_manager_factory()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set optimizer state with GPU tensors
        gpu_optimizer_state = {
            "state": {
                0: {
                    "step": torch.tensor(100).to(device),
                    "exp_avg": torch.randn(2, 3).to(device),
                }
            },
            "param_groups": [{"lr": 0.001}],
        }
        mock_optimizer.set_state_dict(gpu_optimizer_state)

        await manager.save_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 1000, 5, 0, 3
        )

        # Verify optimizer state tensors moved to CPU
        save_call_args = mock_torch_save.call_args[0][0]
        optimizer_state = save_call_args["optimizer_state_dict"]["state"]
        for param_state in optimizer_state.values():
            for value in param_state.values():
                if torch.is_tensor(value):
                    assert value.device.type == "cpu"

    @pytest.mark.asyncio
    async def test_temp_file_creation_and_cleanup_on_success(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        mock_file_manager,
    ):
        """Test temp file creation and cleanup on success"""
        manager = checkpoint_manager_factory()

        await manager.save_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 1000, 5, 0, 3
        )

        # Verify temp file was created and deleted
        assert len(mock_file_manager.created_files) == 1
        assert len(mock_file_manager.deleted_files) == 1
        assert mock_file_manager.created_files[0] == mock_file_manager.deleted_files[0]

    @pytest.mark.asyncio
    async def test_temp_file_cleanup_on_failure(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        mock_torch_save,
        mock_file_manager,
    ):
        """Test temp file cleanup on failure"""
        manager = checkpoint_manager_factory()
        mock_torch_save.make_fail()

        result = await manager.save_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 1000, 5, 0, 3
        )

        assert result is False
        # Verify temp file was still cleaned up despite failure
        assert len(mock_file_manager.created_files) == 1
        assert len(mock_file_manager.deleted_files) == 1

    @pytest.mark.asyncio
    async def test_torch_save_with_highest_protocol(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        mock_torch_save,
    ):
        """Test torch.save with pickle.HIGHEST_PROTOCOL"""
        manager = checkpoint_manager_factory()

        await manager.save_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 1000, 5, 0, 3
        )

        # Verify pickle protocol used
        call_kwargs = mock_torch_save.call_args[1]
        assert call_kwargs["pickle_protocol"] == pickle.HIGHEST_PROTOCOL

    @pytest.mark.asyncio
    async def test_r2_upload_success(
        self, checkpoint_manager_factory, mock_model, mock_optimizer, mock_scheduler
    ):
        """Test R2 upload success"""
        manager = checkpoint_manager_factory()
        manager.storage_client.put_object.return_value = True

        result = await manager.save_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 1000, 5, 0, 3
        )

        assert result is True
        manager.storage_client.put_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_r2_upload_failure(
        self, checkpoint_manager_factory, mock_model, mock_optimizer, mock_scheduler
    ):
        """Test R2 upload failure"""
        manager = checkpoint_manager_factory()
        manager.storage_client.put_object.return_value = False

        result = await manager.save_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 1000, 5, 0, 3
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_filename_generation_with_special_characters(
        self,
        mock_storage_client,
        mock_file_manager,
        mock_bucket,
        mock_model,
        mock_optimizer,
        mock_scheduler,
    ):
        """Test filename generation with special characters in uid"""
        manager = CheckpointManager(
            storage_client=mock_storage_client,
            file_manager=mock_file_manager,
            bucket=mock_bucket,
            uid="test/uid\\with_special",
        )

        await manager.save_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 1000, 5, 0, 3
        )

        # Verify filename has special characters replaced
        call_args = mock_storage_client.put_object.call_args[0]
        filename = call_args[0]
        assert "test_uid_with_special" in filename
        assert "/" not in filename
        assert "\\" not in filename

    @pytest.mark.asyncio
    async def test_exception_during_checkpoint_preparation(
        self, checkpoint_manager_factory, mock_model, mock_optimizer, mock_scheduler
    ):
        """Test exception during checkpoint data preparation"""
        manager = checkpoint_manager_factory()
        mock_model.state_dict.side_effect = Exception("Model error")

        result = await manager.save_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 1000, 5, 0, 3
        )

        assert result is False


class TestLoadCheckpoint:
    @pytest.mark.asyncio
    async def test_successful_checkpoint_load_with_device_movement(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        sample_checkpoint_data,
        mock_torch_load,
        device_fixture,
    ):
        """Test successful checkpoint load with device movement"""
        manager = checkpoint_manager_factory()
        mock_torch_load.set_return_data(sample_checkpoint_data["complete"])

        # Mock get_latest_checkpoint to return data
        manager.get_latest_checkpoint = AsyncMock(
            return_value=(sample_checkpoint_data["complete"], 10)
        )

        success, sync_window, opt, sched = await manager.load_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 5, device_fixture["cpu"]
        )

        assert success is True
        assert sync_window == 8  # From sample data
        mock_model.load_state_dict.assert_called_once()
        mock_optimizer.load_state_dict.assert_called_once()
        mock_scheduler.load_state_dict.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_checkpoints_found_scenario(
        self, checkpoint_manager_factory, mock_model, mock_optimizer, mock_scheduler
    ):
        """Test no checkpoints found scenario"""
        manager = checkpoint_manager_factory()
        manager.get_latest_checkpoint = AsyncMock(return_value=None)

        success, sync_window, opt, sched = await manager.load_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 5, "cpu"
        )

        assert success is False
        assert sync_window == 0
        assert opt == mock_optimizer
        assert sched == mock_scheduler

    @pytest.mark.asyncio
    async def test_invalid_checkpoint_format_missing_keys(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        sample_checkpoint_data,
    ):
        """Test invalid checkpoint format (missing keys)"""
        manager = checkpoint_manager_factory()
        incomplete_data = {"model_state_dict": {}}  # Missing required keys
        manager.get_latest_checkpoint = AsyncMock(return_value=(incomplete_data, 10))

        success, sync_window, opt, sched = await manager.load_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 5, "cpu"
        )

        assert success is False
        assert sync_window == 0

    @pytest.mark.asyncio
    async def test_model_load_state_dict_failure(
        self,
        checkpoint_manager_factory,
        mismatched_model,
        mock_optimizer,
        mock_scheduler,
        sample_checkpoint_data,
    ):
        """Test model.load_state_dict failure (key mismatch)"""
        manager = checkpoint_manager_factory()
        manager.get_latest_checkpoint = AsyncMock(
            return_value=(sample_checkpoint_data["complete"], 10)
        )

        success, sync_window, opt, sched = await manager.load_checkpoint(
            mismatched_model, mock_optimizer, mock_scheduler, 5, "cpu"
        )

        assert success is False
        assert sync_window == 0

    @pytest.mark.asyncio
    async def test_optimizer_state_loading_and_device_movement(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        sample_checkpoint_data,
        device_fixture,
    ):
        """Test optimizer state loading and device movement"""
        manager = checkpoint_manager_factory()
        manager.get_latest_checkpoint = AsyncMock(
            return_value=(sample_checkpoint_data["complete"], 10)
        )

        # Add state to optimizer to test device movement
        mock_optimizer.state = {0: {"exp_avg": torch.randn(2, 3)}}

        success, sync_window, opt, sched = await manager.load_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 5, device_fixture["cpu"]
        )

        assert success is True
        mock_optimizer.load_state_dict.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_window_info_in_checkpoint(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        sample_checkpoint_data,
    ):
        """Test missing start_window in checkpoint data - should use default value and succeed"""
        manager = checkpoint_manager_factory()
        manager.get_latest_checkpoint = AsyncMock(
            return_value=(sample_checkpoint_data["missing_start_window"], 10)
        )

        success, sync_window, opt, sched = await manager.load_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 5, "cpu"
        )

        # Should succeed with default start_window=0, sync_window should be 8 (from test data)
        assert success is True
        assert sync_window == 8  # This checkpoint still has sync_window=8
        assert manager.last_checkpoint_data is not None

    @pytest.mark.asyncio
    async def test_missing_sync_window_in_checkpoint(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        sample_checkpoint_data,
    ):
        """Test missing sync_window in checkpoint data - should use default value and succeed"""
        manager = checkpoint_manager_factory()
        manager.get_latest_checkpoint = AsyncMock(
            return_value=(sample_checkpoint_data["missing_sync_window"], 10)
        )

        success, sync_window, opt, sched = await manager.load_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 5, "cpu"
        )

        # Should succeed with default sync_window=0
        assert success is True
        assert sync_window == 0  # Missing sync_window should default to 0
        assert manager.last_checkpoint_data is not None

    @pytest.mark.asyncio
    async def test_missing_both_windows_in_checkpoint(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        sample_checkpoint_data,
    ):
        """Test missing both start_window and sync_window - should use defaults and succeed"""
        manager = checkpoint_manager_factory()

        # Create checkpoint data missing both windows
        checkpoint_missing_both = {
            k: v
            for k, v in sample_checkpoint_data["complete"].items()
            if k not in ["start_window", "sync_window"]
        }

        manager.get_latest_checkpoint = AsyncMock(
            return_value=(checkpoint_missing_both, 10)
        )

        success, sync_window, opt, sched = await manager.load_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 5, "cpu"
        )

        # Should succeed with both defaults: start_window=0, sync_window=0
        assert success is True
        assert sync_window == 0
        assert manager.last_checkpoint_data is not None

    @pytest.mark.asyncio
    async def test_checkpoint_with_none_window_values(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        sample_checkpoint_data,
    ):
        """Test checkpoint with explicit None values for windows - should fail"""
        manager = checkpoint_manager_factory()

        # Create checkpoint data with explicit None values
        checkpoint_with_none = sample_checkpoint_data["complete"].copy()
        checkpoint_with_none["start_window"] = None
        checkpoint_with_none["current_window"] = None

        manager.get_latest_checkpoint = AsyncMock(
            return_value=(checkpoint_with_none, 10)
        )

        success, sync_window, opt, sched = await manager.load_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 5, "cpu"
        )

        assert success is False
        assert sync_window == 0

    @pytest.mark.asyncio
    async def test_last_checkpoint_data_is_set(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        sample_checkpoint_data,
    ):
        """Test last_checkpoint_data is set after successful load"""
        manager = checkpoint_manager_factory()
        checkpoint_data = sample_checkpoint_data["complete"]
        manager.get_latest_checkpoint = AsyncMock(return_value=(checkpoint_data, 10))

        success, _, _, _ = await manager.load_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 5, "cpu"
        )

        assert success is True
        assert manager.last_checkpoint_data == checkpoint_data

    @pytest.mark.asyncio
    async def test_custom_init_version_parameter(
        self, checkpoint_manager_factory, mock_model, mock_optimizer, mock_scheduler
    ):
        """Test with custom init_version parameter"""
        manager = checkpoint_manager_factory()
        manager.get_latest_checkpoint = AsyncMock(return_value=None)

        await manager.load_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 5, "cpu", init_version="2.0.0"
        )

        manager.get_latest_checkpoint.assert_called_once_with("2.0.0")


class TestGetLatestCheckpoint:
    @pytest.mark.asyncio
    async def test_validator_checkpoint_found_and_returned(
        self, checkpoint_manager_factory, sample_checkpoint_data, mock_bucket
    ):
        """Test validator checkpoint found and returned"""
        manager = checkpoint_manager_factory()

        # Mock validator bucket lookup
        manager._get_highest_stake_validator_bucket = AsyncMock(
            return_value=(mock_bucket, 0)
        )
        manager._get_bucket_checkpoint = AsyncMock(
            return_value=(sample_checkpoint_data["complete"], 10)
        )

        result = await manager.get_latest_checkpoint("1.0.0")

        assert result is not None
        checkpoint_data, window = result
        assert window == 10
        # Should call validator bucket first
        manager._get_bucket_checkpoint.assert_called_with(mock_bucket, 0, "1.0.0")

    @pytest.mark.asyncio
    async def test_fallback_to_self_r2_checkpoint(
        self, checkpoint_manager_factory, sample_checkpoint_data
    ):
        """Test validator not found, self R2 checkpoint found"""
        manager = checkpoint_manager_factory()

        # Mock validator not found
        manager._get_highest_stake_validator_bucket = AsyncMock(
            return_value=(None, None)
        )
        # Mock self bucket has checkpoint
        manager._get_bucket_checkpoint = AsyncMock(
            return_value=(sample_checkpoint_data["complete"], 5)
        )

        result = await manager.get_latest_checkpoint("1.0.0")

        assert result is not None
        checkpoint_data, window = result
        assert window == 5

    @pytest.mark.asyncio
    async def test_no_checkpoints_found_anywhere(self, checkpoint_manager_factory):
        """Test validator not found, self R2 not found, returns None"""
        manager = checkpoint_manager_factory()

        manager._get_highest_stake_validator_bucket = AsyncMock(
            return_value=(None, None)
        )
        manager._get_bucket_checkpoint = AsyncMock(return_value=None)

        result = await manager.get_latest_checkpoint("1.0.0")

        assert result is None

    @pytest.mark.asyncio
    async def test_exception_during_validator_lookup(self, checkpoint_manager_factory):
        """Test exception handling during validator lookup"""
        manager = checkpoint_manager_factory()

        manager._get_highest_stake_validator_bucket = AsyncMock(
            side_effect=Exception("Validator lookup failed")
        )

        result = await manager.get_latest_checkpoint("1.0.0")

        assert result is None


class TestGetBucketCheckpoint:
    @pytest.mark.asyncio
    async def test_successful_checkpoint_retrieval(
        self,
        checkpoint_manager_factory,
        mock_bucket,
        sample_checkpoint_data,
        mock_torch_load,
    ):
        """Test successful checkpoint retrieval with highest window number"""
        manager = checkpoint_manager_factory()
        manager.storage_client.list_objects.return_value = [
            "checkpoint-10-123-v1.0.0.pt",
            "checkpoint-5-123-v1.0.0.pt",
        ]

        # Mock torch.load to return valid data
        mock_torch_load.set_return_data(sample_checkpoint_data["complete"])

        result = await manager._get_bucket_checkpoint(mock_bucket, 123, "1.0.0")

        assert result is not None
        checkpoint_data, window = result
        assert window == 10  # Should pick highest window

    @pytest.mark.asyncio
    async def test_no_matching_checkpoint_files(
        self, checkpoint_manager_factory, mock_bucket
    ):
        """Test no matching checkpoint files found"""
        manager = checkpoint_manager_factory()
        manager.storage_client.list_objects.return_value = [
            "other-file.txt",
            "checkpoint-5-456-v1.0.0.pt",  # Different UID
        ]

        result = await manager._get_bucket_checkpoint(mock_bucket, 123, "1.0.0")

        assert result is None

    @pytest.mark.asyncio
    async def test_corrupted_checkpoint_fallback(
        self,
        checkpoint_manager_factory,
        mock_bucket,
        sample_checkpoint_data,
        mock_torch_load,
    ):
        """Test corrupted checkpoint file, falls back to next valid one"""
        manager = checkpoint_manager_factory()
        manager.storage_client.list_objects.return_value = [
            "checkpoint-10-123-v1.0.0.pt",  # This will be corrupted
            "checkpoint-5-123-v1.0.0.pt",  # This will be valid
        ]

        # First load fails, second succeeds
        load_call_count = 0

        def mock_load_side_effect(*args, **kwargs):
            nonlocal load_call_count
            load_call_count += 1
            if load_call_count == 1:
                raise Exception("Corrupted file")
            return sample_checkpoint_data["complete"]

        mock_torch_load.side_effect = mock_load_side_effect

        result = await manager._get_bucket_checkpoint(mock_bucket, 123, "1.0.0")

        assert result is not None
        checkpoint_data, window = result
        assert window == 5  # Should fallback to window 5

    @pytest.mark.asyncio
    async def test_storage_client_list_objects_failure(
        self, checkpoint_manager_factory, mock_bucket
    ):
        """Test storage client list_objects failure"""
        manager = checkpoint_manager_factory()
        manager.storage_client.list_objects.side_effect = Exception("Storage error")

        result = await manager._get_bucket_checkpoint(mock_bucket, 123, "1.0.0")

        assert result is None

    @pytest.mark.asyncio
    async def test_regex_pattern_with_special_characters(
        self,
        checkpoint_manager_factory,
        mock_bucket,
        sample_checkpoint_data,
        mock_torch_load,
    ):
        """Test regex pattern with special characters in uid"""
        manager = checkpoint_manager_factory()
        manager.storage_client.list_objects.return_value = [
            "checkpoint-10-test.uid-v1.0.0.pt"
        ]
        mock_torch_load.set_return_data(sample_checkpoint_data["complete"])

        result = await manager._get_bucket_checkpoint(mock_bucket, "test.uid", "1.0.0")

        assert result is not None


class TestCleanupOldCheckpoints:
    @pytest.mark.asyncio
    async def test_successful_cleanup_keeping_specified_checkpoints(
        self, checkpoint_manager_factory
    ):
        """Test successful cleanup keeping specified number of checkpoints"""
        manager = checkpoint_manager_factory()
        manager.storage_client.list_objects.return_value = [
            "checkpoint-10-123-v1.0.0.pt",
            "checkpoint-5-123-v1.0.0.pt",
            "checkpoint-15-123-v1.0.0.pt",
            "checkpoint-8-123-v1.0.0.pt",
            "checkpoint-12-123-v1.0.0.pt",
        ]
        manager.storage_client.delete_object.return_value = True

        await manager.cleanup_old_checkpoints(keep_last=3)

        # Should delete 2 oldest checkpoints (5 and 8)
        assert manager.storage_client.delete_object.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_with_fewer_checkpoints_than_keep_last(
        self, checkpoint_manager_factory
    ):
        """Test cleanup with fewer checkpoints than keep_last (no deletion)"""
        manager = checkpoint_manager_factory()
        manager.storage_client.list_objects.return_value = [
            "checkpoint-10-123-v1.0.0.pt"
        ]

        await manager.cleanup_old_checkpoints(keep_last=3)

        # Should not delete anything
        manager.storage_client.delete_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_pattern_matching_current_uid_only(
        self, checkpoint_manager_factory
    ):
        """Test cleanup with pattern matching for current uid only"""
        manager = checkpoint_manager_factory(uid=123)
        manager.storage_client.list_objects.return_value = [
            "checkpoint-10-123-v1.0.0.pt",
            "checkpoint-5-123-v1.0.0.pt",
            "checkpoint-8-456-v1.0.0.pt",  # Different UID - should not be deleted
            "other-file.txt",
        ]
        manager.storage_client.delete_object.return_value = True

        await manager.cleanup_old_checkpoints(keep_last=1)

        # Should only delete one checkpoint (window 5) for uid 123
        assert manager.storage_client.delete_object.call_count == 1
        deleted_file = manager.storage_client.delete_object.call_args[0][0]
        assert "checkpoint-5-123" in deleted_file

    @pytest.mark.asyncio
    async def test_partial_deletion_failures(self, checkpoint_manager_factory):
        """Test partial deletion failures (some succeed, some fail)"""
        manager = checkpoint_manager_factory()
        manager.storage_client.list_objects.return_value = [
            "checkpoint-10-123-v1.0.0.pt",
            "checkpoint-5-123-v1.0.0.pt",
            "checkpoint-8-123-v1.0.0.pt",
        ]

        # First deletion succeeds, second fails
        delete_call_count = 0

        def delete_side_effect(*args):
            nonlocal delete_call_count
            delete_call_count += 1
            return delete_call_count == 1

        manager.storage_client.delete_object.side_effect = delete_side_effect

        await manager.cleanup_old_checkpoints(keep_last=1)

        assert manager.storage_client.delete_object.call_count == 2

    @pytest.mark.asyncio
    async def test_storage_list_objects_failure(self, checkpoint_manager_factory):
        """Test storage client list_objects failure"""
        manager = checkpoint_manager_factory()
        manager.storage_client.list_objects.side_effect = Exception("Storage error")

        # Should not raise exception
        await manager.cleanup_old_checkpoints()


class TestGetHighestStakeValidatorBucket:
    @pytest.mark.asyncio
    async def test_successful_validator_bucket_retrieval(
        self, checkpoint_manager_factory, mock_metagraph, mock_commitments
    ):
        """Test successful validator bucket retrieval"""
        manager = checkpoint_manager_factory(
            metagraph=mock_metagraph, commitments=mock_commitments
        )

        bucket, uid = await manager._get_highest_stake_validator_bucket()

        assert bucket is not None
        assert uid == 0  # Highest stake validator from mock

    @pytest.mark.asyncio
    async def test_no_metagraph_available(self, checkpoint_manager_factory):
        """Test no metagraph available"""
        manager = checkpoint_manager_factory(metagraph=None, commitments={})

        bucket, uid = await manager._get_highest_stake_validator_bucket()

        assert bucket is None
        assert uid is None

    @pytest.mark.asyncio
    async def test_validator_not_in_commitments(
        self, checkpoint_manager_factory, mock_metagraph
    ):
        """Test validator uid not in commitments dict"""
        empty_commitments = {}
        manager = checkpoint_manager_factory(
            metagraph=mock_metagraph, commitments=empty_commitments
        )

        bucket, uid = await manager._get_highest_stake_validator_bucket()

        assert bucket is None
        assert uid is None

    @pytest.mark.asyncio
    async def test_exception_during_metagraph_access(self, checkpoint_manager_factory):
        """Test exception during metagraph.S.argmax()"""
        bad_metagraph = Mock()
        bad_metagraph.S.argmax.side_effect = Exception("Metagraph error")

        manager = checkpoint_manager_factory(metagraph=bad_metagraph)

        bucket, uid = await manager._get_highest_stake_validator_bucket()

        assert bucket is None
        assert uid is None


class TestIntegration:
    @pytest.mark.asyncio
    async def test_complete_save_load_cycle(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        mock_torch_save,
        mock_torch_load,
        sample_checkpoint_data,
    ):
        """Test complete save -> load cycle"""
        manager = checkpoint_manager_factory()

        # Save checkpoint
        save_result = await manager.save_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 1000, 5, 0, 3
        )
        assert save_result is True

        # For load, directly mock get_latest_checkpoint to return saved data
        # This simulates finding the checkpoint we just saved
        manager.get_latest_checkpoint = AsyncMock(
            return_value=(sample_checkpoint_data["complete"], 5)
        )

        # Load checkpoint
        success, sync_window, opt, sched = await manager.load_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, 5, "cpu"
        )

        assert success is True
        assert sync_window == 8  # From sample data
        assert opt == mock_optimizer
        assert sched == mock_scheduler
        assert manager.last_checkpoint_data == sample_checkpoint_data["complete"]

    @pytest.mark.asyncio
    async def test_concurrent_checkpoint_operations(
        self, checkpoint_manager_factory, mock_model, mock_optimizer, mock_scheduler
    ):
        """Test multiple concurrent checkpoint operations"""
        manager = checkpoint_manager_factory()

        # Run multiple save operations concurrently
        tasks = [
            manager.save_checkpoint(
                mock_model, mock_optimizer, mock_scheduler, i * 100, i, 0, i - 1
            )
            for i in range(1, 4)
        ]

        results = await asyncio.gather(*tasks)

        assert all(results)
        assert manager.storage_client.put_object.call_count == 3

    @pytest.mark.asyncio
    async def test_checkpoint_fallback_chain(
        self,
        checkpoint_manager_factory,
        mock_model,
        mock_optimizer,
        mock_scheduler,
        sample_checkpoint_data,
    ):
        """Test load from validator -> fallback to self -> fallback to None"""
        manager = checkpoint_manager_factory()

        # Mock validator bucket exists but no checkpoint
        manager._get_highest_stake_validator_bucket = AsyncMock(
            return_value=(manager.bucket, 0)
        )

        call_count = 0

        async def mock_get_bucket_checkpoint(bucket, uid, version):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # Validator has no checkpoint
            elif call_count == 2:
                return (sample_checkpoint_data["complete"], 10)  # Self has checkpoint
            return None

        manager._get_bucket_checkpoint = mock_get_bucket_checkpoint

        result = await manager.get_latest_checkpoint("1.0.0")

        assert result is not None
        checkpoint_data, window = result
        assert window == 10
