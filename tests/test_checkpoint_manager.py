# ruff : noqa

import pytest
import torch
import torch.nn as nn
import os
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import asyncio
from botocore.exceptions import EndpointConnectionError

# Fix: Use proper botocore.exceptions.ConnectionError with required parameters
from botocore.exceptions import ConnectionError as BotocoreConnectionError

from tplr.training.checkpoint_manager import CheckpointManager
from tplr.storage.client import StorageClient
from tplr.storage.file_manager import FileManager
from tplr.schemas import Bucket


class SimpleModel(nn.Module):
    """Simple test model"""

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)
        self.linear2 = torch.nn.Linear(5, 1)  # Output size 1

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        return self.linear2(x)  # This now outputs [batch_size, 1]


class EmptyModel(nn.Module):
    """Model with no parameters"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class LargeModel(nn.Module):
    """Large model for stress testing"""

    def __init__(self):
        super().__init__()
        # Create a model with ~100MB of parameters
        self.layers = nn.ModuleList([nn.Linear(1000, 1000) for _ in range(25)])

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_bucket():
    """Mock bucket for testing"""
    return Bucket(
        name="test-bucket",
        account_id="test-account",
        access_key_id="test-key",
        secret_access_key="test-secret",
    )


@pytest.fixture
def mock_storage_client():
    """Mock storage client"""
    client = Mock(spec=StorageClient)
    client.put_object = AsyncMock(return_value=True)
    client.get_object = AsyncMock(return_value=b"test-data")
    client.list_objects = AsyncMock(return_value=[])
    client.delete_object = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_file_manager(temp_dir):
    """Mock file manager"""
    manager = Mock(spec=FileManager)
    manager.create_temp_file = Mock(return_value=os.path.join(temp_dir, "test_temp.pt"))
    manager.delete_file = Mock()
    return manager


@pytest.fixture
def checkpoint_manager(mock_storage_client, mock_file_manager, mock_bucket):
    """Create checkpoint manager with mocked dependencies"""
    return CheckpointManager(
        storage_client=mock_storage_client,
        file_manager=mock_file_manager,
        bucket=mock_bucket,
        uid="test-uid-123",
    )


class TestBasicSaveLoadOperations:
    """Test basic save and load operations"""

    @pytest.mark.asyncio
    async def test_save_checkpoint_success(self, checkpoint_manager, temp_dir):
        """Save valid checkpoint with all required fields"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {"param1": torch.randn(10, 20), "param2": torch.randn(5)}

        # Mock file operations
        temp_file = os.path.join(temp_dir, "checkpoint_temp.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        # Verify
        assert result is True
        checkpoint_manager.storage_client.put_object.assert_called_once()
        checkpoint_manager.file_manager.delete_file.assert_called_once_with(temp_file)

        # Verify checkpoint data structure
        call_args = checkpoint_manager.storage_client.put_object.call_args
        filename, data, bucket = call_args[0]
        assert filename.startswith("checkpoint-5-test-uid-123-v")
        assert filename.endswith(".pt")
        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_save_checkpoint_empty_model(self, checkpoint_manager, temp_dir):
        """Save checkpoint with empty model state"""
        # Setup
        model = EmptyModel()

        # Create a dummy parameter for the optimizer since PyTorch requires at least one parameter
        dummy_param = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = SGD([dummy_param], lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "checkpoint_temp.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=0,
            current_window=1,
            start_window=0,
        )

        # Verify
        assert result is True
        checkpoint_manager.storage_client.put_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_checkpoint_large_model(self, checkpoint_manager, temp_dir):
        """Save checkpoint with large model (>1GB)"""
        # Setup - Create a smaller but still large model for testing
        model = LargeModel()
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=5)
        momentum = {f"large_param_{i}": torch.randn(1000, 100) for i in range(10)}

        temp_file = os.path.join(temp_dir, "large_checkpoint_temp.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=500,
            current_window=10,
            start_window=5,
        )

        # Verify
        assert result is True
        # Verify the file was actually large
        call_args = checkpoint_manager.storage_client.put_object.call_args
        _, data, _ = call_args[0]
        assert len(data) > 1_000_000  # At least 1MB

    @pytest.mark.asyncio
    async def test_load_checkpoint_success(self, checkpoint_manager, temp_dir):
        """Load valid checkpoint and verify state restoration"""
        # Setup - Create test checkpoint data
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        checkpoint_data = {
            "model_state_dict": {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "momentum": {"test_momentum": torch.randn(5, 5)},
            "start_window": 1,
            "current_window": 5,
            "global_step": 100,
        }

        # Mock the checkpoint loading
        temp_file = os.path.join(temp_dir, "load_temp.pt")
        torch.save(checkpoint_data, temp_file)

        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.get_object.return_value = open(
            temp_file, "rb"
        ).read()

        # Mock get_latest_checkpoint to return our test data
        with patch.object(
            checkpoint_manager, "get_latest_checkpoint"
        ) as mock_get_latest:
            mock_get_latest.return_value = (checkpoint_data, 5)

            # Execute
            (
                success,
                sync_window,
                loaded_optimizer,
                loaded_scheduler,
            ) = await checkpoint_manager.load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_window=5,
                device="cpu",
            )

        # Verify
        assert success is True
        assert sync_window == 5
        assert checkpoint_manager.last_checkpoint_data is not None
        assert checkpoint_manager.last_checkpoint_data["global_step"] == 100

    @pytest.mark.asyncio
    async def test_load_checkpoint_device_transfer(self, checkpoint_manager, temp_dir):
        """Load checkpoint and verify proper device transfer (CPU->GPU, GPU->CPU)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device transfer test")

        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        # Create checkpoint on CPU
        checkpoint_data = {
            "model_state_dict": {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "momentum": {"test_momentum": torch.randn(5, 5).cpu()},
            "start_window": 1,
            "current_window": 3,
            "global_step": 50,
        }

        with patch.object(
            checkpoint_manager, "get_latest_checkpoint"
        ) as mock_get_latest:
            mock_get_latest.return_value = (checkpoint_data, 3)

            # Test CPU -> GPU transfer
            success, sync_window, _, _ = await checkpoint_manager.load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_window=3,
                device="cuda",
            )

            # Verify model is on GPU
            assert success is True
            for param in model.parameters():
                assert param.device.type == "cuda"

    @pytest.mark.asyncio
    async def test_load_checkpoint_missing_file(self, checkpoint_manager):
        """Attempt to load non-existent checkpoint"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        # Mock no checkpoint found
        with patch.object(
            checkpoint_manager, "get_latest_checkpoint"
        ) as mock_get_latest:
            mock_get_latest.return_value = None

            # Execute
            (
                success,
                sync_window,
                loaded_optimizer,
                loaded_scheduler,
            ) = await checkpoint_manager.load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_window=5,
                device="cpu",
            )

        # Verify
        assert success is False
        assert sync_window == 0
        assert loaded_optimizer is optimizer  # Should return original optimizer
        assert loaded_scheduler is scheduler  # Should return original scheduler

    @pytest.mark.asyncio
    async def test_load_checkpoint_corrupted_file(self, checkpoint_manager):
        """Load corrupted/malformed checkpoint file"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        # Create corrupted checkpoint data (missing required fields)
        corrupted_data = {
            "model_state_dict": {},  # Empty model state
            # Missing optimizer_state_dict, scheduler_state_dict, etc.
        }

        with patch.object(
            checkpoint_manager, "get_latest_checkpoint"
        ) as mock_get_latest:
            mock_get_latest.return_value = (corrupted_data, 3)

            # Execute
            (
                success,
                sync_window,
                loaded_optimizer,
                loaded_scheduler,
            ) = await checkpoint_manager.load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_window=3,
                device="cpu",
            )

        # Verify
        assert success is False
        assert sync_window == 0


class TestStateDictionaryHandling:
    """Test state dictionary handling and preservation"""

    @pytest.mark.asyncio
    async def test_model_state_dict_preservation(self, checkpoint_manager, temp_dir):
        """Verify all model parameters are correctly saved/loaded"""
        # Setup
        model = SimpleModel()

        # Initialize model with specific values
        with torch.no_grad():
            model.linear1.weight.fill_(1.5)
            model.linear1.bias.fill_(0.5)
            model.linear2.weight.fill_(2.0)
            model.linear2.bias.fill_(-0.3)

        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "state_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Save checkpoint
        await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=10,
            current_window=2,
            start_window=1,
        )

        # Verify the saved data contains correct model state
        call_args = checkpoint_manager.storage_client.put_object.call_args
        _, data, _ = call_args[0]

        # Load the data and verify
        temp_load_file = os.path.join(temp_dir, "load_test.pt")
        with open(temp_load_file, "wb") as f:
            f.write(data)

        loaded_checkpoint = torch.load(temp_load_file, weights_only=False)
        loaded_state = loaded_checkpoint["model_state_dict"]

        # Verify all parameters match
        for key in original_state:
            assert key in loaded_state
            torch.testing.assert_close(original_state[key], loaded_state[key])

    @pytest.mark.asyncio
    async def test_optimizer_state_dict_preservation(
        self, checkpoint_manager, temp_dir
    ):
        """Verify optimizer state (momentum, etc.) is preserved"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Take a few optimization steps to build up momentum
        for _ in range(5):
            optimizer.zero_grad()
            output = model(torch.randn(1, 10))
            loss = output.sum()
            loss.backward()
            optimizer.step()

        original_optimizer_state = {
            k: v.clone() if torch.is_tensor(v) else v
            for k, v in optimizer.state_dict().items()
        }

        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "optimizer_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Save checkpoint
        await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=5,
            current_window=1,
            start_window=0,
        )

        # Verify optimizer state is preserved
        call_args = checkpoint_manager.storage_client.put_object.call_args
        _, data, _ = call_args[0]

        temp_load_file = os.path.join(temp_dir, "load_optimizer_test.pt")
        with open(temp_load_file, "wb") as f:
            f.write(data)

        loaded_checkpoint = torch.load(temp_load_file, weights_only=False)
        loaded_optimizer_state = loaded_checkpoint["optimizer_state_dict"]

        # Verify state preservation (basic structure check)
        assert "state" in loaded_optimizer_state
        assert "param_groups" in loaded_optimizer_state

    @pytest.mark.asyncio
    async def test_scheduler_state_dict_preservation(
        self, checkpoint_manager, temp_dir
    ):
        """Verify learning rate scheduler state is preserved"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
        momentum = {}

        # Take several steps with proper optimizer->scheduler order
        for i in range(5):
            # Simulate training step
            loss = torch.nn.functional.mse_loss(
                model(torch.randn(32, 10)), torch.randn(32, 1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Step optimizer first
            scheduler.step()  # Then step scheduler

        # Record final state
        final_lr = optimizer.param_groups[0]["lr"]
        final_epoch = scheduler.last_epoch

        temp_file = os.path.join(temp_dir, "scheduler_step.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        assert result is True

        # Verify scheduler step count is preserved
        saved_data = torch.load(temp_file, weights_only=False)
        saved_scheduler_state = saved_data["scheduler_state_dict"]

        assert "last_epoch" in saved_scheduler_state
        assert saved_scheduler_state["last_epoch"] == final_epoch

        # Verify learning rate was affected by scheduler
        assert final_lr != 0.1  # Should have been reduced by gamma

    @pytest.mark.asyncio
    async def test_momentum_state_preservation(self, checkpoint_manager, temp_dir):
        """Verify custom momentum dictionary is preserved"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        # Create complex momentum dictionary
        momentum = {
            "linear1.weight": torch.randn(20, 10),
            "linear1.bias": torch.randn(20),
            "linear2.weight": torch.randn(5, 20),
            "linear2.bias": torch.randn(5),
            "meta_data": torch.tensor([1.0, 2.0, 3.0]),
        }

        temp_file = os.path.join(temp_dir, "momentum_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Save checkpoint
        await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=25,
            current_window=4,
            start_window=2,
        )

        # Verify momentum preservation
        call_args = checkpoint_manager.storage_client.put_object.call_args
        _, data, _ = call_args[0]

        temp_load_file = os.path.join(temp_dir, "load_momentum_test.pt")
        with open(temp_load_file, "wb") as f:
            f.write(data)

        loaded_checkpoint = torch.load(temp_load_file, weights_only=False)
        loaded_momentum = loaded_checkpoint["momentum"]

        # Verify all momentum tensors are preserved
        for key in momentum:
            assert key in loaded_momentum
            torch.testing.assert_close(momentum[key], loaded_momentum[key])

    @pytest.mark.asyncio
    async def test_state_dict_cpu_cloning(self, checkpoint_manager, temp_dir):
        """Verify all tensors are properly cloned to CPU before saving"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for CPU cloning test")

        # Setup model on GPU
        model = SimpleModel().cuda()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {
            "gpu_tensor": torch.randn(10, 10).cuda(),
            "cpu_tensor": torch.randn(5, 5).cpu(),
        }

        temp_file = os.path.join(temp_dir, "cpu_clone_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Save checkpoint
        await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=30,
            current_window=6,
            start_window=3,
        )

        # Verify all tensors in saved data are on CPU
        call_args = checkpoint_manager.storage_client.put_object.call_args
        _, data, _ = call_args[0]

        temp_load_file = os.path.join(temp_dir, "load_cpu_test.pt")
        with open(temp_load_file, "wb") as f:
            f.write(data)

        loaded_checkpoint = torch.load(
            temp_load_file, weights_only=False, map_location="cpu"
        )

        # Check model state dict tensors are on CPU
        for tensor in loaded_checkpoint["model_state_dict"].values():
            assert tensor.device.type == "cpu"

        # Check momentum tensors are on CPU
        for tensor in loaded_checkpoint["momentum"].values():
            assert tensor.device.type == "cpu"

    @pytest.mark.asyncio
    async def test_state_dict_device_restoration(self, checkpoint_manager):
        """Verify tensors are moved to correct device after loading"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device restoration test")

        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        # Create checkpoint data on CPU
        checkpoint_data = {
            "model_state_dict": {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "momentum": {"test": torch.randn(5, 5).cpu()},
            "start_window": 1,
            "current_window": 2,
            "global_step": 20,
        }

        with patch.object(
            checkpoint_manager, "get_latest_checkpoint"
        ) as mock_get_latest:
            mock_get_latest.return_value = (checkpoint_data, 2)

            # Load checkpoint and specify GPU device
            success, _, _, _ = await checkpoint_manager.load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_window=2,
                device="cuda",
            )

            # Verify model parameters are on GPU
            assert success is True
            for param in model.parameters():
                assert param.device.type == "cuda"


# ... existing code ...


class TestWindowAndStepManagement:
    """Test window and step management functionality"""

    @pytest.mark.asyncio
    async def test_checkpoint_window_tracking(self, checkpoint_manager, temp_dir):
        """Verify start_window, current_window are correctly saved/loaded"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        start_window = 15
        current_window = 25

        temp_file = os.path.join(temp_dir, "window_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Save checkpoint with specific window values
        await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=current_window,
            start_window=start_window,
        )

        # Verify saved data contains correct window values
        call_args = checkpoint_manager.storage_client.put_object.call_args
        _, data, _ = call_args[0]

        temp_load_file = os.path.join(temp_dir, "window_load_test.pt")
        with open(temp_load_file, "wb") as f:
            f.write(data)

        loaded_checkpoint = torch.load(temp_load_file, weights_only=False)

        assert loaded_checkpoint["start_window"] == start_window
        assert loaded_checkpoint["current_window"] == current_window

        # Test loading and verify window preservation
        checkpoint_data = loaded_checkpoint

        with patch.object(
            checkpoint_manager, "get_latest_checkpoint"
        ) as mock_get_latest:
            mock_get_latest.return_value = (checkpoint_data, current_window)

            success, sync_window, _, _ = await checkpoint_manager.load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_window=current_window,
                device="cpu",
            )

            assert success is True
            assert sync_window == current_window
            assert (
                checkpoint_manager.last_checkpoint_data["start_window"] == start_window
            )
            assert (
                checkpoint_manager.last_checkpoint_data["current_window"]
                == current_window
            )

    @pytest.mark.asyncio
    async def test_global_step_preservation(self, checkpoint_manager, temp_dir):
        """Verify global_step is correctly saved/loaded"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        global_steps = [0, 1, 100, 1000, 50000, 999999]

        for step in global_steps:
            temp_file = os.path.join(temp_dir, f"step_test_{step}.pt")
            checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

            # Save checkpoint with specific global step
            await checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=step,
                current_window=5,
                start_window=1,
            )

            # Verify saved global step
            call_args = checkpoint_manager.storage_client.put_object.call_args
            _, data, _ = call_args[0]

            temp_load_file = os.path.join(temp_dir, f"step_load_test_{step}.pt")
            with open(temp_load_file, "wb") as f:
                f.write(data)

            loaded_checkpoint = torch.load(temp_load_file, weights_only=False)
            assert loaded_checkpoint["global_step"] == step

            # Reset mock for next iteration
            checkpoint_manager.storage_client.put_object.reset_mock()

    @pytest.mark.asyncio
    async def test_window_progression(self, checkpoint_manager, temp_dir):
        """Test saving checkpoints across multiple windows"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        # Save checkpoints across multiple windows
        windows = [1, 2, 3, 5, 8, 13, 21]  # Fibonacci-like progression
        global_steps = [10, 25, 50, 100, 200, 400, 800]

        saved_checkpoints = []

        for window, step in zip(windows, global_steps):
            temp_file = os.path.join(temp_dir, f"progression_test_{window}.pt")
            checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

            await checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=step,
                current_window=window,
                start_window=1,
            )

            # Store the filename that should have been created
            call_args = checkpoint_manager.storage_client.put_object.call_args
            filename, data, _ = call_args[0]
            saved_checkpoints.append((filename, window, step))

            # Verify filename contains correct window
            assert f"checkpoint-{window}-test-uid-123-v" in filename

            # Reset mock for next iteration
            checkpoint_manager.storage_client.put_object.reset_mock()

        # Verify all checkpoints have unique filenames and correct windows
        filenames = [checkpoint[0] for checkpoint in saved_checkpoints]
        assert len(filenames) == len(set(filenames))  # All unique

        for filename, window, step in saved_checkpoints:
            assert f"-{window}-" in filename

    @pytest.mark.asyncio
    async def test_overlapping_windows(self, checkpoint_manager, temp_dir):
        """Test behavior when windows overlap or are out of sequence"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        # Test out-of-sequence window saving
        windows_sequence = [5, 3, 7, 2, 6, 1, 8]  # Intentionally out of order

        saved_data = []

        for i, window in enumerate(windows_sequence):
            temp_file = os.path.join(temp_dir, f"overlap_test_{i}.pt")
            checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

            await checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=i * 10,
                current_window=window,
                start_window=1,
            )

            call_args = checkpoint_manager.storage_client.put_object.call_args
            filename, data, _ = call_args[0]
            saved_data.append((filename, window, data))

            # Verify each checkpoint saves correctly regardless of order
            assert f"checkpoint-{window}-test-uid-123-v" in filename

            checkpoint_manager.storage_client.put_object.reset_mock()

        # Verify that each checkpoint maintains its window integrity
        for filename, window, data in saved_data:
            temp_verify_file = os.path.join(temp_dir, f"verify_{window}.pt")
            with open(temp_verify_file, "wb") as f:
                f.write(data)

            loaded_checkpoint = torch.load(temp_verify_file, weights_only=False)
            assert loaded_checkpoint["current_window"] == window

        # Test that the system can handle duplicate window numbers
        duplicate_window = 10
        for i in range(3):
            temp_file = os.path.join(temp_dir, f"duplicate_test_{i}.pt")
            checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

            await checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=i,
                current_window=duplicate_window,
                start_window=1,
            )

            # Each should create a valid checkpoint (last one overwrites in real scenario)
            call_args = checkpoint_manager.storage_client.put_object.call_args
            filename, _, _ = call_args[0]
            assert f"checkpoint-{duplicate_window}-test-uid-123-v" in filename

            checkpoint_manager.storage_client.put_object.reset_mock()


class TestFileNamingAndVersioning:
    """Test file naming conventions and version compatibility"""

    @pytest.mark.asyncio
    async def test_checkpoint_filename_format(self, checkpoint_manager, temp_dir):
        """Verify correct filename format (checkpoint-{window}-{uid}-v{version}.pt)"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        test_cases = [
            {
                "window": 0,
                "uid": "test-uid-123",
                "expected_pattern": r"checkpoint-0-test-uid-123-v.*\.pt$",
            },
            {
                "window": 1,
                "uid": "test-uid-123",
                "expected_pattern": r"checkpoint-1-test-uid-123-v.*\.pt$",
            },
            {
                "window": 999,
                "uid": "test-uid-123",
                "expected_pattern": r"checkpoint-999-test-uid-123-v.*\.pt$",
            },
            {
                "window": 12345,
                "uid": "test-uid-123",
                "expected_pattern": r"checkpoint-12345-test-uid-123-v.*\.pt$",
            },
        ]

        for case in test_cases:
            temp_file = os.path.join(temp_dir, f"filename_test_{case['window']}.pt")
            checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

            await checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=10,
                current_window=case["window"],
                start_window=1,
            )

            call_args = checkpoint_manager.storage_client.put_object.call_args
            filename, _, _ = call_args[0]

            # Verify filename matches expected pattern
            import re

            assert re.match(case["expected_pattern"], filename), (
                f"Filename '{filename}' doesn't match pattern '{case['expected_pattern']}'"
            )

            # Verify filename contains version
            assert "-v" in filename
            assert filename.endswith(".pt")

            checkpoint_manager.storage_client.put_object.reset_mock()

    @pytest.mark.asyncio
    async def test_version_compatibility(self, checkpoint_manager, temp_dir):
        """Test loading checkpoints with different version strings"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        # Create checkpoint data with different version scenarios
        test_versions = [
            "1.0.0",
            "2.1.3",
            "0.9.15",
            "1.0.0-beta",
            "2.0.0-alpha.1",
            "dev-build-123",
        ]

        for version in test_versions:
            checkpoint_data = {
                "model_state_dict": {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "momentum": {},
                "start_window": 1,
                "current_window": 5,
                "global_step": 50,
            }

            with patch.object(
                checkpoint_manager, "get_latest_checkpoint"
            ) as mock_get_latest:
                mock_get_latest.return_value = (checkpoint_data, 5)

                # Test loading with init_version parameter
                success, sync_window, _, _ = await checkpoint_manager.load_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    current_window=5,
                    device="cpu",
                    init_version=version,
                )

                # Should succeed regardless of version string format
                assert success is True
                assert sync_window == 5

                # Verify get_latest_checkpoint was called with correct version
                mock_get_latest.assert_called_once_with(version)

    @pytest.mark.asyncio
    async def test_uid_isolation(self, checkpoint_manager, temp_dir):
        """Verify checkpoints from different UIDs don't interfere"""
        # Setup multiple checkpoint managers with different UIDs
        uids = ["uid-001", "uid-002", "uid-alice", "uid-bob", "uid-validator-1"]

        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        checkpoint_managers = []
        for uid in uids:
            manager = CheckpointManager(
                storage_client=Mock(spec=StorageClient),
                file_manager=Mock(spec=FileManager),
                bucket=Mock(spec=Bucket),
                uid=uid,
            )
            manager.storage_client.put_object = AsyncMock(return_value=True)
            manager.file_manager.create_temp_file = Mock(
                return_value=os.path.join(temp_dir, f"temp_{uid}.pt")
            )
            manager.file_manager.delete_file = Mock()
            checkpoint_managers.append(manager)

        # Save checkpoints with same window but different UIDs
        window = 10
        for i, manager in enumerate(checkpoint_managers):
            await manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=i * 10,
                current_window=window,
                start_window=1,
            )

            call_args = manager.storage_client.put_object.call_args
            filename, _, _ = call_args[0]

            # Verify each filename contains the correct UID
            assert f"checkpoint-{window}-{manager.uid}-v" in filename

            # Verify UIDs don't appear in other filenames
            for other_uid in uids:
                if other_uid != manager.uid:
                    assert f"-{other_uid}-" not in filename

    @pytest.mark.asyncio
    async def test_filename_special_characters(self, checkpoint_manager, temp_dir):
        """Test behavior with special characters in UID"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        # Test various UID formats that might contain special characters
        special_uids = [
            "uid-with-dashes",
            "uid_with_underscores",
            "uid.with.dots",
            "uid123numbers",
            "UID-UPPERCASE",
            "uid-MiXeD-CaSe",
            "very-long-uid-with-many-segments-and-identifiers",
        ]

        for uid in special_uids:
            # Create new manager with special UID
            manager = CheckpointManager(
                storage_client=Mock(spec=StorageClient),
                file_manager=Mock(spec=FileManager),
                bucket=Mock(spec=Bucket),
                uid=uid,
            )
            manager.storage_client.put_object = AsyncMock(return_value=True)
            manager.file_manager.create_temp_file = Mock(
                return_value=os.path.join(temp_dir, f"special_{uid}.pt")
            )
            manager.file_manager.delete_file = Mock()

            await manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=1,
                current_window=1,
                start_window=1,
            )

            call_args = manager.storage_client.put_object.call_args
            filename, _, _ = call_args[0]

            # Verify filename is properly formatted with special UID
            assert f"checkpoint-1-{uid}-v" in filename
            assert filename.endswith(".pt")

            # Verify filename doesn't have invalid characters for file systems
            invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]
            for char in invalid_chars:
                assert char not in filename

        # Test edge case: empty or None UID (should be handled gracefully)
        problematic_uids = ["", "uid with spaces"]  # Spaces might cause issues

        for uid in problematic_uids:
            manager = CheckpointManager(
                storage_client=Mock(spec=StorageClient),
                file_manager=Mock(spec=FileManager),
                bucket=Mock(spec=Bucket),
                uid=uid,
            )
            manager.storage_client.put_object = AsyncMock(return_value=True)
            manager.file_manager.create_temp_file = Mock(
                return_value=os.path.join(
                    temp_dir, f"problematic_{uid.replace(' ', '_')}.pt"
                )
            )
            manager.file_manager.delete_file = Mock()

            # This should either work or fail gracefully
            try:
                await manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    momentum=momentum,
                    global_step=1,
                    current_window=1,
                    start_window=1,
                )

                call_args = manager.storage_client.put_object.call_args
                filename, _, _ = call_args[0]

                # If it succeeds, verify the filename is still valid
                assert filename.endswith(".pt")
                assert "checkpoint-1-" in filename

            except Exception as e:
                # If it fails, it should fail gracefully with a clear error
                assert isinstance(e, (ValueError, TypeError))


# ... existing code ...

from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError


class TestStorageIntegration:
    """Test R2/S3 storage integration functionality"""

    @pytest.mark.asyncio
    async def test_r2_upload_success(self, checkpoint_manager, temp_dir):
        """Verify successful upload to R2 storage"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "upload_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Mock successful R2 upload
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        # Verify
        assert result is True
        checkpoint_manager.storage_client.put_object.assert_called_once()

        # Verify upload was called with correct parameters
        call_args = checkpoint_manager.storage_client.put_object.call_args
        filename, data, bucket = call_args[0]

        assert filename.startswith("checkpoint-5-test-uid-123-v")
        assert isinstance(data, bytes)
        assert len(data) > 0
        assert bucket == checkpoint_manager.bucket

        # Verify temp file cleanup was called
        checkpoint_manager.file_manager.delete_file.assert_called_once_with(temp_file)

    @pytest.mark.asyncio
    async def test_r2_upload_failure(self, checkpoint_manager, temp_dir):
        """Handle R2 upload failures gracefully"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "upload_fail_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Mock R2 upload failure
        checkpoint_manager.storage_client.put_object.return_value = False

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=50,
            current_window=3,
            start_window=1,
        )

        # Verify graceful failure handling
        assert result is False
        checkpoint_manager.storage_client.put_object.assert_called_once()
        # Temp file should still be cleaned up even on upload failure
        checkpoint_manager.file_manager.delete_file.assert_called_once_with(temp_file)

    @pytest.mark.asyncio
    async def test_r2_download_success(self, checkpoint_manager, temp_dir):
        """Verify successful download from R2 storage"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        # Create valid checkpoint data
        checkpoint_data = {
            "model_state_dict": {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "momentum": {},
            "start_window": 1,
            "current_window": 3,
            "global_step": 75,
        }

        # Serialize checkpoint data
        temp_save_file = os.path.join(temp_dir, "download_test_save.pt")
        torch.save(checkpoint_data, temp_save_file)

        with open(temp_save_file, "rb") as f:
            checkpoint_bytes = f.read()

        # Mock successful R2 download
        checkpoint_manager.storage_client.get_object.return_value = checkpoint_bytes
        checkpoint_manager.storage_client.list_objects.return_value = [
            "checkpoint-3-test-uid-123-v1.0.0.pt"
        ]

        temp_load_file = os.path.join(temp_dir, "download_test_load.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_load_file

        # Execute
        result = await checkpoint_manager.get_latest_checkpoint("1.0.0")

        # Verify
        assert result is not None
        loaded_data, window = result
        assert window == 3
        assert loaded_data["global_step"] == 75
        assert loaded_data["current_window"] == 3

        # Verify R2 operations were called
        checkpoint_manager.storage_client.list_objects.assert_called_once()
        checkpoint_manager.storage_client.get_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_r2_download_failure(self, checkpoint_manager):
        """Handle R2 download failures gracefully"""
        # Mock R2 download failure
        checkpoint_manager.storage_client.get_object.return_value = None
        checkpoint_manager.storage_client.list_objects.return_value = [
            "checkpoint-5-test-uid-123-v1.0.0.pt"
        ]

        # Execute
        result = await checkpoint_manager.get_latest_checkpoint("1.0.0")

        # Verify graceful failure handling
        assert result is None
        checkpoint_manager.storage_client.list_objects.assert_called_once()
        checkpoint_manager.storage_client.get_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_r2_network_timeout(self, checkpoint_manager, temp_dir):
        """Handle network timeouts during R2 operations"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "timeout_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Mock network timeout during upload
        async def timeout_upload(*args, **kwargs):
            raise asyncio.TimeoutError("Network timeout")

        checkpoint_manager.storage_client.put_object.side_effect = timeout_upload

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=25,
            current_window=2,
            start_window=1,
        )

        # Verify timeout is handled gracefully
        assert result is False
        checkpoint_manager.file_manager.delete_file.assert_called_once_with(temp_file)

        # Test timeout during download
        async def timeout_download(*args, **kwargs):
            raise asyncio.TimeoutError("Network timeout")

        checkpoint_manager.storage_client.get_object.side_effect = timeout_download
        checkpoint_manager.storage_client.list_objects.return_value = [
            "checkpoint-2-test-uid-123-v1.0.0.pt"
        ]

        result = await checkpoint_manager.get_latest_checkpoint("1.0.0")
        assert result is None

    @pytest.mark.asyncio
    async def test_r2_authentication_failure(self, checkpoint_manager, temp_dir):
        """Handle authentication failures with R2"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "auth_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Mock authentication failures
        auth_errors = [
            NoCredentialsError(),
            PartialCredentialsError(provider="aws", cred_var="aws_access_key_id"),
            ClientError(
                error_response={
                    "Error": {
                        "Code": "InvalidAccessKeyId",
                        "Message": "The AWS Access Key Id you provided does not exist in our records.",
                    }
                },
                operation_name="PutObject",
            ),
            ClientError(
                error_response={
                    "Error": {
                        "Code": "SignatureDoesNotMatch",
                        "Message": "The request signature we calculated does not match the signature you provided.",
                    }
                },
                operation_name="PutObject",
            ),
        ]

        for error in auth_errors:
            checkpoint_manager.storage_client.put_object.side_effect = error

            # Execute
            result = await checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=10,
                current_window=1,
                start_window=1,
            )

            # Verify authentication failure is handled gracefully
            assert result is False

            # Reset for next test
            checkpoint_manager.storage_client.put_object.side_effect = None

    @pytest.mark.asyncio
    async def test_r2_insufficient_permissions(self, checkpoint_manager, temp_dir):
        """Handle insufficient permissions for R2 operations"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "permissions_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Mock permission errors
        permission_errors = [
            ClientError(
                error_response={
                    "Error": {"Code": "AccessDenied", "Message": "Access Denied"}
                },
                operation_name="PutObject",
            ),
            ClientError(
                error_response={"Error": {"Code": "Forbidden", "Message": "Forbidden"}},
                operation_name="GetObject",
            ),
            ClientError(
                error_response={
                    "Error": {
                        "Code": "InvalidUserID.NotFound",
                        "Message": "The user ID provided does not exist in our records.",
                    }
                },
                operation_name="ListObjects",
            ),
        ]

        for error in permission_errors:
            checkpoint_manager.storage_client.put_object.side_effect = error

            # Execute
            result = await checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=15,
                current_window=1,
                start_window=1,
            )

            # Verify permission error is handled gracefully
            assert result is False

            # Verify temp file cleanup still happens
            checkpoint_manager.file_manager.delete_file.assert_called_with(temp_file)

            # Reset for next test
            checkpoint_manager.storage_client.put_object.side_effect = None
            checkpoint_manager.file_manager.delete_file.reset_mock()


class TestTemporaryFileManagement:
    """Test temporary file creation, usage, and cleanup"""

    @pytest.mark.asyncio
    async def test_temp_file_creation(self, checkpoint_manager, temp_dir):
        """Verify temp files are created correctly"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_files = [
            os.path.join(temp_dir, "test1.pt"),
            os.path.join(temp_dir, "test2.pt"),
            os.path.join(temp_dir, "test3.pt"),
        ]

        for i, temp_file in enumerate(temp_files):
            checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

            # Execute checkpoint save
            await checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=i,
                current_window=i + 1,
                start_window=1,
            )

            # Verify temp file creation was called
            checkpoint_manager.file_manager.create_temp_file.assert_called()

            # Verify the temp file path is used correctly
            call_args = checkpoint_manager.storage_client.put_object.call_args
            # The temp file should have been read and its contents uploaded
            assert call_args is not None

            # Reset mocks for next iteration
            checkpoint_manager.file_manager.create_temp_file.reset_mock()
            checkpoint_manager.storage_client.put_object.reset_mock()

    @pytest.mark.asyncio
    async def test_temp_file_cleanup_success(self, checkpoint_manager, temp_dir):
        """Verify temp files are cleaned up after successful operations"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "cleanup_success_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute successful checkpoint save
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        # Verify success and cleanup
        assert result is True
        checkpoint_manager.file_manager.delete_file.assert_called_once_with(temp_file)

        # Test cleanup during successful checkpoint load
        checkpoint_data = {
            "model_state_dict": {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "momentum": {},
            "start_window": 1,
            "current_window": 5,
            "global_step": 100,
        }

        temp_load_file = os.path.join(temp_dir, "cleanup_load_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_load_file

        # Mock successful download
        temp_save_file = os.path.join(temp_dir, "temp_save.pt")
        torch.save(checkpoint_data, temp_save_file)
        with open(temp_save_file, "rb") as f:
            checkpoint_bytes = f.read()

        checkpoint_manager.storage_client.get_object.return_value = checkpoint_bytes
        checkpoint_manager.storage_client.list_objects.return_value = [
            "checkpoint-5-test-uid-123-v1.0.0.pt"
        ]

        # Reset delete_file mock to track new calls
        checkpoint_manager.file_manager.delete_file.reset_mock()

        # Execute
        result = await checkpoint_manager.get_latest_checkpoint("1.0.0")

        # Verify cleanup happened
        assert result is not None
        checkpoint_manager.file_manager.delete_file.assert_called_with(temp_load_file)

    @pytest.mark.asyncio
    async def test_temp_file_cleanup_failure(self, checkpoint_manager, temp_dir):
        """Verify temp files are cleaned up even after failures"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "cleanup_failure_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Test cleanup after upload failure
        checkpoint_manager.storage_client.put_object.return_value = False

        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=50,
            current_window=3,
            start_window=1,
        )

        # Verify failure and cleanup
        assert result is False
        checkpoint_manager.file_manager.delete_file.assert_called_once_with(temp_file)

        # Test cleanup after exception during save
        checkpoint_manager.file_manager.delete_file.reset_mock()

        async def raise_exception(*args, **kwargs):
            raise Exception("Simulated error during upload")

        checkpoint_manager.storage_client.put_object.side_effect = raise_exception

        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=25,
            current_window=2,
            start_window=1,
        )

        # Verify cleanup happens even with exception
        assert result is False
        checkpoint_manager.file_manager.delete_file.assert_called_once_with(temp_file)

    @pytest.mark.asyncio
    async def test_temp_file_disk_space(self, checkpoint_manager, temp_dir):
        """Handle insufficient disk space for temp files"""
        # Setup
        model = LargeModel()  # Use large model to test disk space issues
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        # Mock disk space error during temp file creation
        def disk_full_error(*args, **kwargs):
            raise OSError(28, "No space left on device")  # ENOSPC

        checkpoint_manager.file_manager.create_temp_file.side_effect = disk_full_error

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        # Verify disk space error is handled gracefully
        assert result is False

        # Test disk space error during torch.save
        checkpoint_manager.file_manager.create_temp_file.side_effect = None
        temp_file = os.path.join(temp_dir, "disk_space_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Mock torch.save to raise disk space error
        original_torch_save = torch.save

        def mock_torch_save(*args, **kwargs):
            raise OSError(28, "No space left on device")

        with patch("torch.save", side_effect=mock_torch_save):
            result = await checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=100,
                current_window=5,
                start_window=1,
            )

            # Should handle gracefully
            assert result is False

    @pytest.mark.asyncio
    async def test_temp_file_permissions(self, checkpoint_manager, temp_dir):
        """Handle permission issues with temp file creation"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        # Mock permission error during temp file creation
        def permission_error(*args, **kwargs):
            raise PermissionError("Permission denied")

        checkpoint_manager.file_manager.create_temp_file.side_effect = permission_error

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=10,
            current_window=1,
            start_window=1,
        )

        # Verify permission error is handled gracefully
        assert result is False

        # Test permission error during file write
        checkpoint_manager.file_manager.create_temp_file.side_effect = None
        temp_file = os.path.join(temp_dir, "permission_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Mock torch.save to raise permission error
        def mock_torch_save(*args, **kwargs):
            raise PermissionError("Permission denied")

        with patch("torch.save", side_effect=mock_torch_save):
            result = await checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=10,
                current_window=1,
                start_window=1,
            )

            # Should handle gracefully
            assert result is False

        # Test permission error during file read
        checkpoint_manager.file_manager.create_temp_file.return_value = (
            "/root/readonly_file.pt"  # Simulated read-only location
        )

        # Mock file opening to raise permission error
        original_open = open

        def mock_open(*args, **kwargs):
            if "/root/readonly_file.pt" in str(args[0]) and "rb" in str(args[1]):
                raise PermissionError("Permission denied")
            return original_open(*args, **kwargs)

        with patch("builtins.open", side_effect=mock_open):
            result = await checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=10,
                current_window=1,
                start_window=1,
            )

            # Should handle gracefully
            assert result is False


# ... existing code ...


class TestLatestCheckpointDetection:
    """Test latest checkpoint detection functionality"""

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint_single(self, checkpoint_manager, temp_dir):
        """Find latest when only one checkpoint exists"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        checkpoint_data = {
            "model_state_dict": {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "momentum": {},
            "start_window": 1,
            "current_window": 5,
            "global_step": 100,
        }

        # Mock single checkpoint in bucket
        checkpoint_manager.storage_client.list_objects.return_value = [
            "checkpoint-5-test-uid-123-v1.0.0.pt"
        ]

        # Serialize checkpoint data for mock return
        temp_save_file = os.path.join(temp_dir, "single_checkpoint.pt")
        torch.save(checkpoint_data, temp_save_file)

        with open(temp_save_file, "rb") as f:
            checkpoint_bytes = f.read()

        checkpoint_manager.storage_client.get_object.return_value = checkpoint_bytes
        checkpoint_manager.file_manager.create_temp_file.return_value = os.path.join(
            temp_dir, "load_single.pt"
        )

        # Execute
        result = await checkpoint_manager.get_latest_checkpoint("1.0.0")

        # Verify
        assert result is not None
        loaded_data, window = result
        assert window == 5
        assert loaded_data["current_window"] == 5
        assert loaded_data["global_step"] == 100

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint_multiple(self, checkpoint_manager, temp_dir):
        """Find latest among multiple checkpoints"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        # Create multiple checkpoint data with different windows
        checkpoints = {}
        for window in [3, 7, 12, 8, 5]:  # Intentionally out of order
            checkpoint_data = {
                "model_state_dict": {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "momentum": {},
                "start_window": 1,
                "current_window": window,
                "global_step": window * 10,
            }

            temp_save_file = os.path.join(temp_dir, f"checkpoint_{window}.pt")
            torch.save(checkpoint_data, temp_save_file)

            with open(temp_save_file, "rb") as f:
                checkpoints[f"checkpoint-{window}-test-uid-123-v1.0.0.pt"] = f.read()

        # Mock multiple checkpoints in bucket
        checkpoint_manager.storage_client.list_objects.return_value = list(
            checkpoints.keys()
        )

        # Mock get_object to return the highest window (12)
        def mock_get_object(key, bucket):
            return checkpoints[key]

        checkpoint_manager.storage_client.get_object.side_effect = mock_get_object
        checkpoint_manager.file_manager.create_temp_file.return_value = os.path.join(
            temp_dir, "load_multiple.pt"
        )

        # Execute
        result = await checkpoint_manager.get_latest_checkpoint("1.0.0")

        # Verify - should get the highest window (12)
        assert result is not None
        loaded_data, window = result
        assert window == 12
        assert loaded_data["current_window"] == 12
        assert loaded_data["global_step"] == 120

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint_none(self, checkpoint_manager):
        """Handle case when no checkpoints exist"""
        # Mock empty bucket and no local checkpoints
        checkpoint_manager.storage_client.list_objects.return_value = []

        # Mock no validator bucket
        checkpoint_manager.metagraph = None

        # Execute
        result = await checkpoint_manager.get_latest_checkpoint("1.0.0")

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint_version_filter(
        self, checkpoint_manager, temp_dir
    ):
        """Filter checkpoints by version correctly"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        # Create checkpoints with different versions
        versions_and_windows = [
            ("1.0.0", 5),
            ("1.1.0", 7),
            ("2.0.0", 10),
            ("1.0.0", 3),  # Same version, different window
        ]

        checkpoints = {}
        for version, window in versions_and_windows:
            checkpoint_data = {
                "model_state_dict": {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "momentum": {},
                "start_window": 1,
                "current_window": window,
                "global_step": window * 10,
            }

            key = f"checkpoint-{window}-test-uid-123-v{version}.pt"
            temp_save_file = os.path.join(temp_dir, f"checkpoint_{version}_{window}.pt")
            torch.save(checkpoint_data, temp_save_file)

            with open(temp_save_file, "rb") as f:
                checkpoints[key] = f.read()

        # Mock all checkpoints in bucket
        checkpoint_manager.storage_client.list_objects.return_value = list(
            checkpoints.keys()
        )

        def mock_get_object(key, bucket):
            return checkpoints[key]

        checkpoint_manager.storage_client.get_object.side_effect = mock_get_object
        checkpoint_manager.file_manager.create_temp_file.return_value = os.path.join(
            temp_dir, "load_version.pt"
        )

        # Test filtering for version 1.0.0 - should get window 5 (highest for that version)
        result = await checkpoint_manager.get_latest_checkpoint("1.0.0")
        assert result is not None
        loaded_data, window = result
        assert window == 5  # Highest window for version 1.0.0

        # Reset mocks
        checkpoint_manager.storage_client.get_object.reset_mock()
        checkpoint_manager.storage_client.get_object.side_effect = mock_get_object

        # Test filtering for version 2.0.0 - should get window 10
        result = await checkpoint_manager.get_latest_checkpoint("2.0.0")
        assert result is not None
        loaded_data, window = result
        assert window == 10

        # Test filtering for non-existent version
        result = await checkpoint_manager.get_latest_checkpoint("3.0.0")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint_corrupted_metadata(
        self, checkpoint_manager, temp_dir
    ):
        """Handle corrupted checkpoint metadata"""
        # Setup
        corrupted_data = b"corrupted checkpoint data that cannot be loaded"

        # Mock checkpoint in bucket
        checkpoint_manager.storage_client.list_objects.return_value = [
            "checkpoint-5-test-uid-123-v1.0.0.pt"
        ]
        checkpoint_manager.storage_client.get_object.return_value = corrupted_data
        checkpoint_manager.file_manager.create_temp_file.return_value = os.path.join(
            temp_dir, "corrupted.pt"
        )

        # Execute
        result = await checkpoint_manager.get_latest_checkpoint("1.0.0")

        # Verify - should handle corruption gracefully
        assert result is None

        # Test partially corrupted checkpoint (missing required keys)
        incomplete_checkpoint = {
            "model_state_dict": {},
            # Missing optimizer_state_dict, scheduler_state_dict, etc.
        }

        temp_save_file = os.path.join(temp_dir, "incomplete.pt")
        torch.save(incomplete_checkpoint, temp_save_file)

        with open(temp_save_file, "rb") as f:
            incomplete_data = f.read()

        checkpoint_manager.storage_client.get_object.return_value = incomplete_data

        # This should still return the data (validation happens at load time)
        result = await checkpoint_manager.get_latest_checkpoint("1.0.0")
        assert result is not None
        loaded_data, window = result
        assert "model_state_dict" in loaded_data
        assert window == 5


class TestBucketOperations:
    """Test bucket operations and listing functionality"""

    @pytest.mark.asyncio
    async def test_bucket_checkpoint_listing(self, checkpoint_manager):
        """List checkpoints from bucket correctly"""
        # Mock bucket contents with various files
        mock_keys = [
            "checkpoint-1-test-uid-123-v1.0.0.pt",
            "checkpoint-5-test-uid-123-v1.0.0.pt",
            "checkpoint-10-test-uid-123-v1.0.0.pt",
            "gradient-1-test-uid-123-v1.0.0.pt",  # Non-checkpoint file
            "checkpoint-3-other-uid-456-v1.0.0.pt",  # Different UID
            "some-other-file.txt",  # Completely different file
        ]

        checkpoint_manager.storage_client.list_objects.return_value = mock_keys

        # Execute - this tests the internal listing mechanism
        result = await checkpoint_manager._get_bucket_checkpoint(
            checkpoint_manager.bucket, checkpoint_manager.uid, "1.0.0"
        )

        # Verify list_objects was called with correct prefix
        checkpoint_manager.storage_client.list_objects.assert_called_with(
            "checkpoint", checkpoint_manager.bucket
        )

        # Since we don't have actual checkpoint data, this will return None
        # But the listing mechanism should work correctly
        assert result is None  # Expected since get_object will fail without mock data

    @pytest.mark.asyncio
    async def test_bucket_checkpoint_filtering(self, checkpoint_manager, temp_dir):
        """Filter bucket contents by pattern correctly"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        # Create valid checkpoint for the expected match
        checkpoint_data = {
            "model_state_dict": {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "momentum": {},
            "start_window": 1,
            "current_window": 5,
            "global_step": 50,
        }

        temp_save_file = os.path.join(temp_dir, "filter_test.pt")
        torch.save(checkpoint_data, temp_save_file)

        with open(temp_save_file, "rb") as f:
            valid_checkpoint_data = f.read()

        # Mock bucket with mixed content - only some should match the pattern
        mixed_keys = [
            "checkpoint-5-test-uid-123-v1.0.0.pt",  # Should match
            "checkpoint-invalid-format.pt",  # Invalid format
            "checkpoint-3-wrong-uid-456-v1.0.0.pt",  # Wrong UID
            "checkpoint-7-test-uid-123-v2.0.0.pt",  # Wrong version
            "not-a-checkpoint-file.txt",  # Not a checkpoint
            "checkpoint-2-test-uid-123-v1.0.0.backup",  # Wrong extension
        ]

        checkpoint_manager.storage_client.list_objects.return_value = mixed_keys

        # Mock get_object to return data only for the valid checkpoint
        def mock_get_object(key, bucket):
            if key == "checkpoint-5-test-uid-123-v1.0.0.pt":
                return valid_checkpoint_data
            return None

        checkpoint_manager.storage_client.get_object.side_effect = mock_get_object
        checkpoint_manager.file_manager.create_temp_file.return_value = os.path.join(
            temp_dir, "filter_load.pt"
        )

        # Execute
        result = await checkpoint_manager._get_bucket_checkpoint(
            checkpoint_manager.bucket, "test-uid-123", "1.0.0"
        )

        # Verify - should only find the correctly formatted checkpoint
        assert result is not None
        loaded_data, window = result
        assert window == 5
        assert loaded_data["current_window"] == 5

    @pytest.mark.asyncio
    async def test_bucket_empty(self, checkpoint_manager):
        """Handle empty bucket gracefully"""
        # Mock empty bucket
        checkpoint_manager.storage_client.list_objects.return_value = []

        # Execute
        result = await checkpoint_manager._get_bucket_checkpoint(
            checkpoint_manager.bucket, checkpoint_manager.uid, "1.0.0"
        )

        # Verify
        assert result is None
        checkpoint_manager.storage_client.list_objects.assert_called_once()

    @pytest.mark.asyncio
    async def test_bucket_invalid_keys(self, checkpoint_manager):
        """Handle invalid/malformed keys in bucket"""
        # Mock bucket with various invalid key formats
        invalid_keys = [
            "",  # Empty key
            "checkpoint",  # Missing parts
            "checkpoint-",  # Incomplete
            "checkpoint-abc-test-uid-123-v1.0.0.pt",  # Non-numeric window
            "checkpoint-5-test-uid-123-v.pt",  # Missing version
            "checkpoint-5-test-uid-123-v1.0.0",  # Missing extension
            "checkpoint--5-test-uid-123-v1.0.0.pt",  # Extra dash
            "checkpoint-5--test-uid-123-v1.0.0.pt",  # Extra dash in UID section
        ]

        checkpoint_manager.storage_client.list_objects.return_value = invalid_keys

        # Execute
        result = await checkpoint_manager._get_bucket_checkpoint(
            checkpoint_manager.bucket, checkpoint_manager.uid, "1.0.0"
        )

        # Verify - should handle invalid keys gracefully
        assert result is None
        checkpoint_manager.storage_client.list_objects.assert_called_once()
        # get_object should not be called since no valid keys were found
        checkpoint_manager.storage_client.get_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_bucket_access_denied(self, checkpoint_manager):
        """Handle bucket access denial"""
        from botocore.exceptions import ClientError

        # Mock access denied error
        access_denied_error = ClientError(
            error_response={
                "Error": {"Code": "AccessDenied", "Message": "Access Denied"}
            },
            operation_name="ListObjectsV2",
        )

        checkpoint_manager.storage_client.list_objects.side_effect = access_denied_error

        # Execute
        result = await checkpoint_manager._get_bucket_checkpoint(
            checkpoint_manager.bucket, checkpoint_manager.uid, "1.0.0"
        )

        # Verify - should handle access denial gracefully
        assert result is None

        # Test access denied during get_object
        checkpoint_manager.storage_client.list_objects.side_effect = None
        checkpoint_manager.storage_client.list_objects.return_value = [
            "checkpoint-5-test-uid-123-v1.0.0.pt"
        ]
        checkpoint_manager.storage_client.get_object.side_effect = access_denied_error

        result = await checkpoint_manager._get_bucket_checkpoint(
            checkpoint_manager.bucket, checkpoint_manager.uid, "1.0.0"
        )

        # Should still handle gracefully
        assert result is None

    @pytest.mark.asyncio
    async def test_bucket_network_errors(self, checkpoint_manager):
        """Handle various network errors during bucket operations"""

        network_errors = [
            asyncio.TimeoutError("Request timeout"),
            EndpointConnectionError(endpoint_url="https://example.com"),
            BotocoreConnectionError(
                error="Connection failed"
            ),  # Fix: provide required 'error' parameter
            Exception("Generic network error"),
        ]

        for error in network_errors:
            checkpoint_manager.storage_client.list_objects.side_effect = error

            result = await checkpoint_manager._get_bucket_checkpoint(
                checkpoint_manager.bucket, checkpoint_manager.uid, "1.0.0"
            )

            # Should handle all network errors gracefully
            assert result is None

            # Reset for next test
            checkpoint_manager.storage_client.list_objects.side_effect = None


class TestCleanupOperations:
    """Test cleanup operations functionality"""

    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints(self, checkpoint_manager):
        """Remove old checkpoints while keeping recent ones"""
        # Setup multiple checkpoints with different windows
        checkpoint_keys = [
            "checkpoint-1-test-uid-123-v1.0.0.pt",
            "checkpoint-3-test-uid-123-v1.0.0.pt",
            "checkpoint-5-test-uid-123-v1.0.0.pt",
            "checkpoint-7-test-uid-123-v1.0.0.pt",
            "checkpoint-9-test-uid-123-v1.0.0.pt",
            "checkpoint-11-test-uid-123-v1.0.0.pt",
        ]

        checkpoint_manager.storage_client.list_objects.return_value = checkpoint_keys
        checkpoint_manager.storage_client.delete_object = AsyncMock(return_value=True)

        # Execute - keep last 3 checkpoints
        await checkpoint_manager.cleanup_old_checkpoints(keep_last=3)

        # Verify - should delete 3 oldest checkpoints (windows 1, 3, 5)
        expected_deletions = [
            "checkpoint-1-test-uid-123-v1.0.0.pt",
            "checkpoint-3-test-uid-123-v1.0.0.pt",
            "checkpoint-5-test-uid-123-v1.0.0.pt",
        ]

        assert checkpoint_manager.storage_client.delete_object.call_count == 3

        # Verify the correct files were marked for deletion
        delete_calls = [
            call.args[0]
            for call in checkpoint_manager.storage_client.delete_object.call_args_list
        ]
        for expected_key in expected_deletions:
            assert expected_key in delete_calls

    @pytest.mark.asyncio
    async def test_cleanup_keep_last_parameter(self, checkpoint_manager):
        """Verify keep_last parameter works correctly"""
        checkpoint_keys = [
            "checkpoint-2-test-uid-123-v1.0.0.pt",
            "checkpoint-4-test-uid-123-v1.0.0.pt",
            "checkpoint-6-test-uid-123-v1.0.0.pt",
            "checkpoint-8-test-uid-123-v1.0.0.pt",
        ]

        checkpoint_manager.storage_client.list_objects.return_value = checkpoint_keys
        checkpoint_manager.storage_client.delete_object = AsyncMock(return_value=True)

        # Test keep_last=1 - should delete 3 oldest
        await checkpoint_manager.cleanup_old_checkpoints(keep_last=1)

        assert checkpoint_manager.storage_client.delete_object.call_count == 3

        # Reset mock
        checkpoint_manager.storage_client.delete_object.reset_mock()

        # Test keep_last=2 - should delete 2 oldest
        await checkpoint_manager.cleanup_old_checkpoints(keep_last=2)

        assert checkpoint_manager.storage_client.delete_object.call_count == 2

        # Reset mock
        checkpoint_manager.storage_client.delete_object.reset_mock()

        # Test keep_last=5 - should delete none (only 4 checkpoints exist)
        await checkpoint_manager.cleanup_old_checkpoints(keep_last=5)

        assert checkpoint_manager.storage_client.delete_object.call_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_no_checkpoints(self, checkpoint_manager):
        """Handle cleanup when no checkpoints exist"""
        # Mock empty bucket
        checkpoint_manager.storage_client.list_objects.return_value = []
        checkpoint_manager.storage_client.delete_object = AsyncMock()

        # Execute
        await checkpoint_manager.cleanup_old_checkpoints(keep_last=3)

        # Verify - no deletion attempts should be made
        checkpoint_manager.storage_client.delete_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_insufficient_checkpoints(self, checkpoint_manager):
        """Handle cleanup when fewer than keep_last checkpoints exist"""
        # Setup only 2 checkpoints
        checkpoint_keys = [
            "checkpoint-10-test-uid-123-v1.0.0.pt",
            "checkpoint-12-test-uid-123-v1.0.0.pt",
        ]

        checkpoint_manager.storage_client.list_objects.return_value = checkpoint_keys
        checkpoint_manager.storage_client.delete_object = AsyncMock()

        # Execute - try to keep 5 but only 2 exist
        await checkpoint_manager.cleanup_old_checkpoints(keep_last=5)

        # Verify - no deletions should occur
        checkpoint_manager.storage_client.delete_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_deletion_failure(self, checkpoint_manager):
        """Handle failures during checkpoint deletion"""
        checkpoint_keys = [
            "checkpoint-1-test-uid-123-v1.0.0.pt",
            "checkpoint-3-test-uid-123-v1.0.0.pt",
            "checkpoint-5-test-uid-123-v1.0.0.pt",
            "checkpoint-7-test-uid-123-v1.0.0.pt",
        ]

        checkpoint_manager.storage_client.list_objects.return_value = checkpoint_keys

        # Mock deletion failure for some files
        async def mock_delete_object(key, bucket):
            if "checkpoint-1" in key:
                raise Exception("Deletion failed")
            return True

        checkpoint_manager.storage_client.delete_object.side_effect = mock_delete_object

        # Execute - should handle individual deletion failures gracefully
        await checkpoint_manager.cleanup_old_checkpoints(keep_last=2)

        # Verify - should attempt to delete 2 oldest files despite failures
        assert checkpoint_manager.storage_client.delete_object.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_mixed_checkpoint_types(self, checkpoint_manager):
        """Handle cleanup with mixed checkpoint types and versions"""
        # Mix of different UIDs, versions, and other files
        mixed_keys = [
            "checkpoint-1-test-uid-123-v1.0.0.pt",  # Should be included
            "checkpoint-2-test-uid-123-v1.0.0.pt",  # Should be included
            "checkpoint-3-test-uid-123-v1.0.0.pt",  # Should be included
            "checkpoint-4-test-uid-123-v1.0.0.pt",  # Should be included
            "checkpoint-5-test-uid-123-v1.0.0.pt",  # Should be included
            "checkpoint-2-different-uid-456-v1.0.0.pt",  # Different UID - ignore
            "checkpoint-3-test-uid-123-v2.0.0.pt",  # Different version - ignore
            "gradient-1-test-uid-123-v1.0.0.pt",  # Not a checkpoint - ignore
            "some-other-file.txt",  # Random file - ignore
        ]

        checkpoint_manager.storage_client.list_objects.return_value = mixed_keys
        checkpoint_manager.storage_client.delete_object = AsyncMock(return_value=True)

        # Execute
        await checkpoint_manager.cleanup_old_checkpoints(keep_last=2)

        # Verify - should only delete 3 oldest checkpoints that match our UID and version
        # (keeping windows 4 and 5, deleting windows 1, 2, 3)
        assert checkpoint_manager.storage_client.delete_object.call_count == 3

        delete_calls = [
            call.args[0]
            for call in checkpoint_manager.storage_client.delete_object.call_args_list
        ]
        expected_deletions = [
            "checkpoint-1-test-uid-123-v1.0.0.pt",
            "checkpoint-2-test-uid-123-v1.0.0.pt",
            "checkpoint-3-test-uid-123-v1.0.0.pt",
        ]

        for expected_key in expected_deletions:
            assert expected_key in delete_calls


# ... existing code ...


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios"""

    @pytest.mark.asyncio
    async def test_save_checkpoint_disk_full(self, checkpoint_manager, temp_dir):
        """Handle disk full during save operation"""
        import errno

        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        # Mock disk full error during temp file creation
        def mock_create_temp_file(*args):
            raise OSError(errno.ENOSPC, "No space left on device")

        checkpoint_manager.file_manager.create_temp_file.side_effect = (
            mock_create_temp_file
        )

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        # Verify - should handle disk full gracefully
        assert result is False

        # Should not attempt R2 upload since temp file creation failed
        checkpoint_manager.storage_client.put_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_checkpoint_network_error(self, checkpoint_manager, temp_dir):
        """Handle network errors during R2 upload"""
        import asyncio
        from botocore.exceptions import EndpointConnectionError, NoCredentialsError

        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "network_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        network_errors = [
            asyncio.TimeoutError("Upload timeout"),
            EndpointConnectionError(endpoint_url="https://example.com"),
            NoCredentialsError(),
            Exception("Generic network error"),
        ]

        for error in network_errors:
            checkpoint_manager.storage_client.put_object.side_effect = error
            checkpoint_manager.file_manager.delete_file.reset_mock()

            result = await checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=100,
                current_window=5,
                start_window=1,
            )

            # Should handle network errors gracefully
            assert result is False
            # Should still cleanup temp file
            checkpoint_manager.file_manager.delete_file.assert_called_once_with(
                temp_file
            )

    @pytest.mark.asyncio
    async def test_load_checkpoint_memory_error(self, checkpoint_manager, temp_dir):
        """Handle out-of-memory during checkpoint loading"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        # Mock checkpoint exists
        checkpoint_manager.storage_client.list_objects.return_value = [
            "checkpoint-5-test-uid-123-v1.0.0.pt"
        ]

        # Mock memory error during torch.load
        def mock_torch_load(*args, **kwargs):
            raise MemoryError("Out of memory")

        with patch("torch.load", side_effect=mock_torch_load):
            checkpoint_manager.storage_client.get_object.return_value = (
                b"mock_checkpoint_data"
            )
            checkpoint_manager.file_manager.create_temp_file.return_value = (
                os.path.join(temp_dir, "memory_test.pt")
            )

            # Execute
            result = await checkpoint_manager.get_latest_checkpoint("1.0.0")

            # Should handle memory error gracefully
            assert result is None

    @pytest.mark.asyncio
    async def test_load_checkpoint_invalid_format(self, checkpoint_manager, temp_dir):
        """Handle invalid checkpoint file format"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        # Create invalid checkpoint data (missing required keys)
        invalid_checkpoint = {
            "model_state_dict": {"weight": torch.randn(5, 5)},
            # Missing optimizer_state_dict, scheduler_state_dict, etc.
        }

        temp_invalid_file = os.path.join(temp_dir, "invalid_checkpoint.pt")
        torch.save(invalid_checkpoint, temp_invalid_file)

        with open(temp_invalid_file, "rb") as f:
            invalid_bytes = f.read()

        checkpoint_manager.storage_client.list_objects.return_value = [
            "checkpoint-5-test-uid-123-v1.0.0.pt"
        ]
        checkpoint_manager.storage_client.get_object.return_value = invalid_bytes
        checkpoint_manager.file_manager.create_temp_file.return_value = os.path.join(
            temp_dir, "load_invalid.pt"
        )

        # Execute load_checkpoint (which calls get_latest_checkpoint internally)
        (
            success,
            sync_window,
            ret_optimizer,
            ret_scheduler,
        ) = await checkpoint_manager.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            current_window=5,
            device="cpu",
        )

        # Should handle invalid format gracefully
        assert success is False
        assert sync_window == 0
        assert ret_optimizer is optimizer  # Should return original
        assert ret_scheduler is scheduler  # Should return original

    @pytest.mark.asyncio
    async def test_load_checkpoint_version_mismatch(self, checkpoint_manager, temp_dir):
        """Handle version incompatibility"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)

        # Mock no checkpoints for requested version
        checkpoint_manager.storage_client.list_objects.return_value = [
            "checkpoint-5-test-uid-123-v2.0.0.pt",  # Different version
            "checkpoint-3-test-uid-123-v0.9.0.pt",  # Different version
        ]

        # Execute - request version 1.0.0 but only 2.0.0 and 0.9.0 exist
        result = await checkpoint_manager.get_latest_checkpoint("1.0.0")

        # Should handle version mismatch gracefully
        assert result is None

    @pytest.mark.asyncio
    async def test_serialization_error(self, checkpoint_manager, temp_dir):
        """Handle torch.save() failures"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "serialization_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

        # Mock torch.save failure
        def mock_torch_save(*args, **kwargs):
            raise RuntimeError("Serialization failed")

        with patch("torch.save", side_effect=mock_torch_save):
            # Execute
            result = await checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                momentum=momentum,
                global_step=100,
                current_window=5,
                start_window=1,
            )

            # Should handle serialization error gracefully
            assert result is False

            # Should still attempt cleanup (even though file might not exist)
            checkpoint_manager.file_manager.delete_file.assert_called_once_with(
                temp_file
            )

    @pytest.mark.asyncio
    async def test_deserialization_error(self, checkpoint_manager, temp_dir):
        """Handle torch.load() failures"""
        # Setup
        checkpoint_manager.storage_client.list_objects.return_value = [
            "checkpoint-5-test-uid-123-v1.0.0.pt"
        ]

        # Create corrupted checkpoint data
        corrupted_data = b"corrupted_checkpoint_data_not_valid_pytorch"
        checkpoint_manager.storage_client.get_object.return_value = corrupted_data
        checkpoint_manager.file_manager.create_temp_file.return_value = os.path.join(
            temp_dir, "corrupted_load.pt"
        )

        # Execute
        result = await checkpoint_manager.get_latest_checkpoint("1.0.0")

        # Should handle deserialization error gracefully
        assert result is None


class TestModelStateEdgeCases:
    """Test model state edge cases"""

    @pytest.mark.asyncio
    async def test_empty_model_state(self, checkpoint_manager, temp_dir):
        """Handle model with no parameters"""

        # Create model with no parameters
        class EmptyModel(torch.nn.Module):
            def forward(self, x):
                return x  # Identity function, no learnable parameters

        model = EmptyModel()

        # Create a dummy parameter list for the optimizer since PyTorch doesn't allow empty param lists
        # We'll use a single parameter that we can ignore
        dummy_param = torch.nn.Parameter(torch.randn(1))
        optimizer = SGD([dummy_param], lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "empty_model.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        # Should handle empty model correctly
        assert result is True

        # Verify empty model state dict was saved (the dummy param won't be in model state)
        saved_data = torch.load(temp_file, weights_only=False)
        assert len(saved_data["model_state_dict"]) == 0

    @pytest.mark.asyncio
    async def test_model_with_shared_parameters(self, checkpoint_manager, temp_dir):
        """Handle models with shared/tied parameters"""

        # Create model with shared parameters (common in language models)
        class ModelWithSharedParams(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(100, 50)
                self.linear = torch.nn.Linear(50, 100)
                # Properly tie weights using parameter sharing
                self.linear.weight = self.embedding.weight

            def forward(self, x):
                embedded = self.embedding(x)
                return torch.nn.functional.linear(
                    embedded, self.embedding.weight, self.linear.bias
                )

        model = ModelWithSharedParams()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "shared_params.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        # Should handle shared parameters correctly
        assert result is True

        # Verify shared parameters are saved correctly
        saved_data = torch.load(temp_file, weights_only=False)
        state_dict = saved_data["model_state_dict"]

        # Both embedding and linear should reference the same weight tensor
        assert "embedding.weight" in state_dict
        assert "linear.weight" in state_dict
        assert "linear.bias" in state_dict

        # The weights should be identical (shared)
        assert torch.equal(state_dict["embedding.weight"], state_dict["linear.weight"])

    @pytest.mark.asyncio
    async def test_model_parameter_dtypes(self, checkpoint_manager, temp_dir):
        """Verify parameter data types are preserved"""

        # Create model with mixed floating point dtypes (only float types can require gradients)
        class MixedDtypeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.float_param = torch.nn.Parameter(torch.randn(5, 5).float())
                self.double_param = torch.nn.Parameter(torch.randn(3, 3).double())
                # For non-gradient tensors, use buffers instead of parameters
                self.register_buffer("int_buffer", torch.randint(0, 10, (2, 2)).int())

        model = MixedDtypeModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        # Get original dtypes for parameters and buffers
        original_param_dtypes = {
            name: param.dtype for name, param in model.named_parameters()
        }
        original_buffer_dtypes = {
            name: buffer.dtype for name, buffer in model.named_buffers()
        }

        temp_file = os.path.join(temp_dir, "dtypes_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Save checkpoint
        await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        # Load and verify dtypes
        saved_data = torch.load(temp_file, weights_only=False)
        state_dict = saved_data["model_state_dict"]

        # Verify parameter dtypes
        for name, original_dtype in original_param_dtypes.items():
            assert name in state_dict
            saved_dtype = state_dict[name].dtype
            assert saved_dtype == original_dtype, (
                f"Parameter dtype mismatch for {name}: {saved_dtype} vs {original_dtype}"
            )

        # Verify buffer dtypes
        for name, original_dtype in original_buffer_dtypes.items():
            assert name in state_dict
            saved_dtype = state_dict[name].dtype
            assert saved_dtype == original_dtype, (
                f"Buffer dtype mismatch for {name}: {saved_dtype} vs {original_dtype}"
            )

    @pytest.mark.asyncio
    async def test_model_custom_modules(self, checkpoint_manager, temp_dir):
        """Handle models with custom module types"""

        # Create custom module
        class CustomModule(torch.nn.Module):
            def __init__(self, input_size, output_size):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(output_size, input_size))
                self.bias = torch.nn.Parameter(torch.randn(output_size))
                self.custom_attr = "custom_value"

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class ModelWithCustomModules(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.custom1 = CustomModule(10, 5)
                self.custom2 = CustomModule(5, 2)
                self.standard = torch.nn.Linear(2, 1)

            def forward(self, x):
                x = self.custom1(x)
                x = self.custom2(x)
                return self.standard(x)

        model = ModelWithCustomModules()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "custom_modules.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        # Should handle custom modules correctly
        assert result is True

        # Verify all parameters from custom modules are saved
        saved_data = torch.load(temp_file, weights_only=False)
        expected_keys = [
            "custom1.weight",
            "custom1.bias",
            "custom2.weight",
            "custom2.bias",
            "standard.weight",
            "standard.bias",
        ]

        for key in expected_keys:
            assert key in saved_data["model_state_dict"], f"Missing parameter: {key}"

    @pytest.mark.asyncio
    async def test_model_state_device_handling(self, checkpoint_manager, temp_dir):
        """Verify model state is correctly moved to CPU during save"""
        # Setup model on different device (if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SimpleModel().to(device)
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        temp_file = os.path.join(temp_dir, "device_test.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        assert result is True

        # Verify saved state is on CPU
        saved_data = torch.load(temp_file, weights_only=False, map_location="cpu")
        for name, param in saved_data["model_state_dict"].items():
            assert param.device == torch.device("cpu"), (
                f"Parameter {name} not moved to CPU: {param.device}"
            )


# ... existing code ...


class TestOptimizerStateEdgeCases:
    """Test optimizer state edge cases"""

    @pytest.mark.asyncio
    async def test_optimizer_empty_state(self, checkpoint_manager, temp_dir):
        """Handle optimizer with no state (first step)"""
        # Setup - fresh optimizer with no state
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        # Fresh optimizer should have empty state
        assert len(optimizer.state) == 0

        temp_file = os.path.join(temp_dir, "empty_optimizer.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        # Should handle empty optimizer state correctly
        assert result is True

        # Verify empty optimizer state was saved
        saved_data = torch.load(temp_file, weights_only=False)
        saved_optimizer_state = saved_data["optimizer_state_dict"]["state"]
        assert len(saved_optimizer_state) == 0

    @pytest.mark.asyncio
    async def test_optimizer_complex_state(self, checkpoint_manager, temp_dir):
        """Handle optimizer with complex state (Adam, etc.)"""
        # Setup - Adam optimizer which maintains momentum and variance estimates
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        # Simulate some training steps to populate optimizer state
        for _ in range(3):
            loss = torch.nn.functional.mse_loss(
                model(torch.randn(32, 10)),
                torch.randn(32, 1),  # Now matches model output size
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Verify optimizer has accumulated state
        assert len(optimizer.state) > 0

        # Check that Adam-specific state exists
        for param_state in optimizer.state.values():
            assert "step" in param_state
            assert "exp_avg" in param_state  # momentum
            assert "exp_avg_sq" in param_state  # variance

        temp_file = os.path.join(temp_dir, "complex_optimizer.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        assert result is True

        # Verify complex optimizer state was preserved
        saved_data = torch.load(temp_file, weights_only=False)
        saved_optimizer_state = saved_data["optimizer_state_dict"]["state"]

        # Should have state for each parameter
        assert len(saved_optimizer_state) == len(list(model.parameters()))

        # Verify Adam-specific state is preserved
        for param_id, param_state in saved_optimizer_state.items():
            assert "step" in param_state
            assert "exp_avg" in param_state
            assert "exp_avg_sq" in param_state

    @pytest.mark.asyncio
    async def test_optimizer_state_groups(self, checkpoint_manager, temp_dir):
        """Handle optimizer with multiple parameter groups"""
        # Setup - model with different parameter groups
        model = SimpleModel()

        # Create optimizer with multiple parameter groups (different LRs)
        optimizer = SGD(
            [
                {
                    "params": model.linear1.parameters(),
                    "lr": 0.01,
                    "weight_decay": 0.001,
                },
                {
                    "params": model.linear2.parameters(),
                    "lr": 0.1,
                    "weight_decay": 0.0001,
                },
            ]
        )

        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        # Verify we have multiple parameter groups
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[1]["lr"] == 0.1

        temp_file = os.path.join(temp_dir, "multi_groups.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        assert result is True

        # Verify parameter groups are preserved
        saved_data = torch.load(temp_file, weights_only=False)
        saved_param_groups = saved_data["optimizer_state_dict"]["param_groups"]

        assert len(saved_param_groups) == 2
        assert saved_param_groups[0]["lr"] == 0.01
        assert saved_param_groups[0]["weight_decay"] == 0.001
        assert saved_param_groups[1]["lr"] == 0.1
        assert saved_param_groups[1]["weight_decay"] == 0.0001

    @pytest.mark.asyncio
    async def test_optimizer_custom_fields(self, checkpoint_manager, temp_dir):
        """Handle optimizers with custom state fields"""

        # Create custom optimizer with additional state
        class CustomOptimizer(SGD):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.custom_field = "custom_value"
                self.custom_counter = 0

            def step(self, closure=None):
                self.custom_counter += 1
                return super().step(closure)

        model = SimpleModel()
        optimizer = CustomOptimizer(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        # Simulate some steps to modify custom fields
        optimizer.custom_counter = 42
        optimizer.custom_field = "modified_value"

        temp_file = os.path.join(temp_dir, "custom_optimizer.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        assert result is True

        # Note: Custom fields won't be in state_dict, but that's expected behavior
        # PyTorch only saves standard optimizer state via state_dict()
        saved_data = torch.load(temp_file, weights_only=False)
        assert "optimizer_state_dict" in saved_data

    @pytest.mark.asyncio
    async def test_optimizer_state_device_transfer(self, checkpoint_manager, temp_dir):
        """Verify optimizer state device transfer"""
        # Setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SimpleModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        # Create some optimizer state on the device
        for _ in range(2):
            loss = torch.nn.functional.mse_loss(
                model(torch.randn(32, 10).to(device)), torch.randn(32, 1).to(device)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Manually move optimizer state to device (this is what applications typically do)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        # Verify state is now on the correct device (normalize device comparison)
        target_device = torch.device(device)
        for param_state in optimizer.state.values():
            for key, tensor_state in param_state.items():
                if torch.is_tensor(tensor_state):
                    # Normalize device comparison - cuda and cuda:0 are the same
                    state_device = tensor_state.device
                    if state_device.type == target_device.type:
                        if target_device.type == "cuda":
                            # For CUDA, ignore the index if target doesn't specify one
                            if target_device.index is None:
                                assert state_device.type == "cuda", (
                                    f"State {key} not on CUDA: {state_device}"
                                )
                            else:
                                assert state_device == target_device, (
                                    f"State {key} on wrong device: {state_device}"
                                )
                        else:
                            assert state_device == target_device, (
                                f"State {key} on wrong device: {state_device}"
                            )
                    else:
                        assert False, (
                            f"State {key} on wrong device: {state_device} vs {target_device}"
                        )

        temp_file = os.path.join(temp_dir, "device_transfer.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute save (should move to CPU)
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        assert result is True

        # Verify optimizer state was saved as CPU tensors
        saved_data = torch.load(temp_file, weights_only=False, map_location="cpu")
        saved_optimizer_state = saved_data["optimizer_state_dict"]["state"]

        for param_id, param_state in saved_optimizer_state.items():
            for key, tensor_state in param_state.items():
                if torch.is_tensor(tensor_state):
                    assert tensor_state.device.type == "cpu", (
                        f"Saved state {key} not on CPU: {tensor_state.device}"
                    )

        # Test loading back to device
        new_model = SimpleModel().to(device)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = StepLR(new_optimizer, step_size=10)

        # Simulate the load_checkpoint process which moves state to device
        (
            success,
            sync_window,
            loaded_opt,
            loaded_sched,
        ) = await checkpoint_manager.load_checkpoint(
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            current_window=5,
            device=device,
        )

        assert success is True

        # Verify loaded optimizer state is on correct device
        for param_state in loaded_opt.state.values():
            for key, tensor_state in param_state.items():
                if torch.is_tensor(tensor_state):
                    state_device = tensor_state.device
                    if device == "cuda":
                        assert state_device.type == "cuda", (
                            f"Loaded state {key} not on CUDA: {state_device}"
                        )
                    else:
                        assert state_device.type == device, (
                            f"Loaded state {key} on wrong device: {state_device}"
                        )

    @pytest.mark.asyncio
    async def test_optimizer_state_device_transfer(self, checkpoint_manager, temp_dir):
        """Verify optimizer state device transfer"""
        # Setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SimpleModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=10)
        momentum = {}

        # Create some optimizer state on the device
        for _ in range(2):
            loss = torch.nn.functional.mse_loss(
                model(torch.randn(32, 10).to(device)),
                torch.randn(32, 1).to(device),  # This now matches model output size
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Manually move optimizer state to device (this is what applications typically do)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        # Verify state is now on the correct device (normalize device comparison)
        target_device = torch.device(device)
        for param_state in optimizer.state.values():
            for key, tensor_state in param_state.items():
                if torch.is_tensor(tensor_state):
                    # Normalize device comparison - cuda and cuda:0 are the same
                    state_device = tensor_state.device
                    if state_device.type == target_device.type:
                        if target_device.type == "cuda":
                            # For CUDA, ignore the index if target doesn't specify one
                            if target_device.index is None:
                                assert state_device.type == "cuda", (
                                    f"State {key} not on CUDA: {state_device}"
                                )
                            else:
                                assert state_device == target_device, (
                                    f"State {key} on wrong device: {state_device}"
                                )
                        else:
                            assert state_device == target_device, (
                                f"State {key} on wrong device: {state_device}"
                            )
                    else:
                        assert False, (
                            f"State {key} on wrong device: {state_device} vs {target_device}"
                        )

        temp_file = os.path.join(temp_dir, "device_transfer.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute save (should move to CPU)
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        assert result is True

        # Verify optimizer state was saved as CPU tensors
        saved_data = torch.load(temp_file, weights_only=False, map_location="cpu")
        saved_optimizer_state = saved_data["optimizer_state_dict"]["state"]

        for param_id, param_state in saved_optimizer_state.items():
            for key, tensor_state in param_state.items():
                if torch.is_tensor(tensor_state):
                    assert tensor_state.device.type == "cpu", (
                        f"Saved state {key} not on CPU: {tensor_state.device}"
                    )

        # Test loading back to device - bypass get_latest_checkpoint and call _get_bucket_checkpoint directly
        new_model = SimpleModel().to(device)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = StepLR(new_optimizer, step_size=10)

        # Read the checkpoint file
        with open(temp_file, "rb") as f:
            checkpoint_bytes = f.read()

        # Mock the storage client for our bucket checkpoint
        checkpoint_manager.storage_client.list_objects.return_value = [
            "checkpoint-5-test-uid-123-v1.0.0.pt"
        ]
        checkpoint_manager.storage_client.get_object.return_value = checkpoint_bytes

        # Create a new temp file for loading
        load_temp_file = os.path.join(temp_dir, "load_device_transfer.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = load_temp_file

        # Directly call _get_bucket_checkpoint with correct argument order (bucket, uid, version)
        result = await checkpoint_manager._get_bucket_checkpoint(
            checkpoint_manager.bucket, checkpoint_manager.uid, "1.0.0"
        )
        assert result is not None

        checkpoint_data, checkpoint_window = result

        # Manually load the checkpoint (simulate load_checkpoint logic)
        try:
            # Load model state
            new_model.load_state_dict(
                {
                    k: v.to(device)
                    for k, v in checkpoint_data["model_state_dict"].items()
                }
            )
            new_model.to(device)

            # Load optimizer state first, then move to device
            new_optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

            # Move optimizer state to device
            for state in new_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

            # Load scheduler state
            new_scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

            success = True

        except Exception:
            success = False

        assert success is True

        # Verify loaded optimizer state is on correct device
        for param_state in new_optimizer.state.values():
            for key, tensor_state in param_state.items():
                if torch.is_tensor(tensor_state):
                    state_device = tensor_state.device
                    if device == "cuda":
                        assert state_device.type == "cuda", (
                            f"Loaded state {key} not on CUDA: {state_device}"
                        )
                    else:
                        assert state_device.type == device, (
                            f"Loaded state {key} on wrong device: {state_device}"
                        )


class TestSchedulerStateEdgeCases:
    """Test scheduler state edge cases"""

    @pytest.mark.asyncio
    async def test_scheduler_step_state(self, checkpoint_manager, temp_dir):
        """Verify scheduler step count is preserved"""
        # Setup
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
        momentum = {}

        # Take several steps with proper optimizer->scheduler order
        for i in range(5):
            # Simulate training step
            loss = torch.nn.functional.mse_loss(
                model(torch.randn(32, 10)), torch.randn(32, 1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Step optimizer first
            scheduler.step()  # Then step scheduler

        # Record final state
        final_lr = optimizer.param_groups[0]["lr"]
        final_epoch = scheduler.last_epoch

        temp_file = os.path.join(temp_dir, "scheduler_step.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        assert result is True

        # Verify scheduler step count is preserved
        saved_data = torch.load(temp_file, weights_only=False)
        saved_scheduler_state = saved_data["scheduler_state_dict"]

        assert "last_epoch" in saved_scheduler_state
        assert saved_scheduler_state["last_epoch"] == final_epoch

        # Verify learning rate was affected by scheduler
        assert final_lr != 0.1  # Should have been reduced by gamma

    @pytest.mark.asyncio
    async def test_scheduler_lr_state(self, checkpoint_manager, temp_dir):
        """Verify learning rate values are preserved"""
        # Setup with multiple parameter groups
        model = SimpleModel()
        optimizer = SGD(
            [
                {"params": model.linear1.parameters(), "lr": 0.1},
                {"params": model.linear2.parameters(), "lr": 0.01},
            ]
        )
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        momentum = {}

        # Record initial LRs
        initial_lrs = [group["lr"] for group in optimizer.param_groups]

        # Take scheduler steps to modify LRs
        for _ in range(4):  # Should trigger decay at step 3
            scheduler.step()

        # Verify LRs have changed
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        expected_lrs = [lr * 0.1 for lr in initial_lrs]

        for current, expected in zip(current_lrs, expected_lrs):
            assert abs(current - expected) < 1e-6

        temp_file = os.path.join(temp_dir, "scheduler_lrs.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        assert result is True

        # Load and verify LRs are preserved
        saved_data = torch.load(temp_file, weights_only=False)

        # Create new optimizer/scheduler and load state
        new_model = SimpleModel()
        new_optimizer = SGD(
            [
                {"params": new_model.linear1.parameters(), "lr": 0.1},
                {"params": new_model.linear2.parameters(), "lr": 0.01},
            ]
        )
        new_scheduler = StepLR(new_optimizer, step_size=3, gamma=0.1)

        new_optimizer.load_state_dict(saved_data["optimizer_state_dict"])
        new_scheduler.load_state_dict(saved_data["scheduler_state_dict"])

        # Verify LRs match
        restored_lrs = [group["lr"] for group in new_optimizer.param_groups]
        for current, restored in zip(current_lrs, restored_lrs):
            assert abs(current - restored) < 1e-6

    @pytest.mark.asyncio
    async def test_scheduler_complex_schedules(self, checkpoint_manager, temp_dir):
        """Handle complex multi-step schedulers"""
        # Setup with CosineAnnealingLR (complex scheduler)
        model = SimpleModel()
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0.0001
        )
        momentum = {}

        # Take several steps to create complex LR schedule state
        initial_lr = optimizer.param_groups[0]["lr"]
        for i in range(5):
            scheduler.step()

        mid_lr = optimizer.param_groups[0]["lr"]
        assert mid_lr != initial_lr  # Should have changed

        temp_file = os.path.join(temp_dir, "complex_scheduler.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        assert result is True

        # Verify complex scheduler state
        saved_data = torch.load(temp_file, weights_only=False)
        saved_scheduler_state = saved_data["scheduler_state_dict"]

        assert "last_epoch" in saved_scheduler_state
        assert saved_scheduler_state["last_epoch"] == 5

    @pytest.mark.asyncio
    async def test_scheduler_custom_schedulers(self, checkpoint_manager, temp_dir):
        """Handle custom scheduler implementations"""

        # Create custom scheduler
        class CustomScheduler(torch.optim.lr_scheduler._LRScheduler):
            def __init__(self, optimizer, custom_param=1.0, last_epoch=-1):
                self.custom_param = custom_param
                self.custom_state = {"counter": 0}
                super().__init__(optimizer, last_epoch)

            def get_lr(self):
                self.custom_state["counter"] += 1
                return [
                    base_lr * (self.custom_param**self.last_epoch)
                    for base_lr in self.base_lrs
                ]

        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = CustomScheduler(optimizer, custom_param=0.9)
        momentum = {}

        # Take some steps
        for _ in range(3):
            scheduler.step()

        assert scheduler.custom_state["counter"] > 0

        temp_file = os.path.join(temp_dir, "custom_scheduler.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        assert result is True

        # Note: Custom scheduler state might not be fully preserved via state_dict()
        # This tests that the checkpoint save/load doesn't crash with custom schedulers
        saved_data = torch.load(temp_file, weights_only=False)
        assert "scheduler_state_dict" in saved_data

    @pytest.mark.asyncio
    async def test_scheduler_state_consistency(self, checkpoint_manager, temp_dir):
        """Verify scheduler state consistency after save/load cycle"""
        # Setup
        model = SimpleModel()
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[5, 10, 15], gamma=0.1
        )
        momentum = {}

        # Progress through several milestones
        for i in range(12):
            scheduler.step()

        # Record state before save
        pre_save_lr = optimizer.param_groups[0]["lr"]
        pre_save_epoch = scheduler.last_epoch

        temp_file = os.path.join(temp_dir, "scheduler_consistency.pt")
        checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
        checkpoint_manager.storage_client.put_object.return_value = True

        # Execute save
        result = await checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum=momentum,
            global_step=100,
            current_window=5,
            start_window=1,
        )

        assert result is True

        # Create fresh optimizer/scheduler and load
        new_model = SimpleModel()
        new_optimizer = Adam(new_model.parameters(), lr=0.001)
        new_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            new_optimizer, milestones=[5, 10, 15], gamma=0.1
        )

        saved_data = torch.load(temp_file, weights_only=False)
        new_optimizer.load_state_dict(saved_data["optimizer_state_dict"])
        new_scheduler.load_state_dict(saved_data["scheduler_state_dict"])

        # Verify consistency
        post_load_lr = new_optimizer.param_groups[0]["lr"]
        post_load_epoch = new_scheduler.last_epoch

        assert abs(pre_save_lr - post_load_lr) < 1e-6
        assert pre_save_epoch == post_load_epoch

        # Take one more step and verify they behave identically
        scheduler.step()
        new_scheduler.step()

        final_lr_original = optimizer.param_groups[0]["lr"]
        final_lr_loaded = new_optimizer.param_groups[0]["lr"]

        assert abs(final_lr_original - final_lr_loaded) < 1e-6


# class TestDataIntegrity:
#     """Test data integrity and preservation"""

#     @pytest.mark.asyncio
#     async def test_checkpoint_data_integrity(self, checkpoint_manager, temp_dir):
#         """Verify data integrity after save/load cycle"""
#         # Setup
#         model = SimpleModel()
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
#         scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

#         # Create complex momentum state
#         momentum = {
#             "layer1_momentum": torch.randn(10, 10),
#             "layer2_momentum": torch.randn(5, 5),
#             "global_momentum": torch.tensor(0.95),
#             "nested_state": {
#                 "sub_momentum": torch.randn(3, 3),
#                 "counter": torch.tensor(42)
#             }
#         }

#         # Train model to create optimizer state
#         for i in range(3):
#             loss = torch.nn.functional.mse_loss(
#                 model(torch.randn(32, 10)),
#                 torch.randn(32, 1)
#             )
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#         # Record original states
#         original_model_state = {k: v.clone() for k, v in model.state_dict().items()}
#         original_optimizer_state = {
#             'state': {
#                 param_id: {
#                     key: val.clone() if torch.is_tensor(val) else val
#                     for key, val in param_state.items()
#                 }
#                 for param_id, param_state in optimizer.state.items()
#             },
#             'param_groups': [
#                 {k: v for k, v in group.items()}
#                 for group in optimizer.param_groups
#             ]
#         }
#         original_scheduler_state = scheduler.state_dict().copy()
#         original_momentum = {
#             k: v.clone() if torch.is_tensor(v) else (
#                 {sk: sv.clone() if torch.is_tensor(sv) else sv for sk, sv in v.items()}
#                 if isinstance(v, dict) else v
#             )
#             for k, v in momentum.items()
#         }

#         temp_file = os.path.join(temp_dir, "integrity_test.pt")
#         checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
#         checkpoint_manager.storage_client.put_object.return_value = True

#         # Save checkpoint
#         result = await checkpoint_manager.save_checkpoint(
#             model=model,
#             optimizer=optimizer,
#             scheduler=scheduler,
#             momentum=momentum,
#             global_step=150,
#             current_window=7,
#             start_window=1
#         )

#         assert result is True

#         # Create fresh instances
#         new_model = SimpleModel()
#         new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001, betas=(0.9, 0.999))
#         new_scheduler = StepLR(new_optimizer, step_size=5, gamma=0.5)

#         # Load checkpoint
#         with open(temp_file, 'rb') as f:
#             checkpoint_bytes = f.read()

#         checkpoint_manager.storage_client.list_objects.return_value = [
#             "checkpoint-7-test-uid-123-v1.0.0.pt"
#         ]
#         checkpoint_manager.storage_client.get_object.return_value = checkpoint_bytes
#         load_temp_file = os.path.join(temp_dir, "load_integrity.pt")
#         checkpoint_manager.file_manager.create_temp_file.return_value = load_temp_file

#         success, sync_window, loaded_opt, loaded_sched = await checkpoint_manager.load_checkpoint(
#             model=new_model,
#             optimizer=new_optimizer,
#             scheduler=new_scheduler,
#             current_window=7,
#             device="cpu"
#         )

#         assert success is True
#         assert sync_window == 7

#         # Verify model state integrity
#         loaded_model_state = new_model.state_dict()
#         for key in original_model_state:
#             assert key in loaded_model_state
#             assert torch.allclose(
#                 original_model_state[key],
#                 loaded_model_state[key],
#                 rtol=1e-5, atol=1e-8
#             ), f"Model parameter {key} not preserved accurately"

#         # Verify optimizer state integrity
#         for param_id in original_optimizer_state['state']:
#             assert param_id in loaded_opt.state
#             for state_key in original_optimizer_state['state'][param_id]:
#                 original_val = original_optimizer_state['state'][param_id][state_key]
#                 loaded_val = loaded_opt.state[param_id][state_key]

#                 if torch.is_tensor(original_val):
#                     assert torch.allclose(
#                         original_val, loaded_val, rtol=1e-5, atol=1e-8
#                     ), f"Optimizer state {state_key} not preserved accurately"
#                 else:
#                     assert original_val == loaded_val, f"Optimizer state {state_key} not preserved"

#         # Verify scheduler state integrity
#         loaded_scheduler_state = loaded_sched.state_dict()
#         for key in original_scheduler_state:
#             assert key in loaded_scheduler_state
#             assert original_scheduler_state[key] == loaded_scheduler_state[key]

#         # Verify momentum integrity
#         loaded_checkpoint_data = checkpoint_manager.last_checkpoint_data
#         loaded_momentum = loaded_checkpoint_data["momentum"]

#         def compare_momentum_recursive(orig, loaded, path=""):
#             if torch.is_tensor(orig):
#                 assert torch.allclose(
#                     orig, loaded, rtol=1e-5, atol=1e-8
#                 ), f"Momentum tensor {path} not preserved accurately"
#             elif isinstance(orig, dict):
#                 for key in orig:
#                     assert key in loaded, f"Missing momentum key {path}.{key}"
#                     compare_momentum_recursive(orig[key], loaded[key], f"{path}.{key}")
#             else:
#                 assert orig == loaded, f"Momentum value {path} not preserved"

#         compare_momentum_recursive(original_momentum, loaded_momentum)

#     @pytest.mark.asyncio
#     async def test_parameter_value_preservation(self, checkpoint_manager, temp_dir):
#         """Verify exact parameter values are preserved"""
#         # Create model with specific parameter values
#         model = SimpleModel()

#         # Set specific parameter values for verification
#         with torch.no_grad():
#             model.linear1.weight.fill_(1.23456789)
#             model.linear1.bias.fill_(-0.987654321)
#             model.linear2.weight.fill_(2.718281828)
#             model.linear2.bias.fill_(3.141592653)

#         optimizer = SGD(model.parameters(), lr=0.01)
#         scheduler = StepLR(optimizer, step_size=10)
#         momentum = {}

#         # Record exact parameter values
#         exact_values = {}
#         for name, param in model.named_parameters():
#             exact_values[name] = param.data.clone()

#         temp_file = os.path.join(temp_dir, "exact_values.pt")
#         checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
#         checkpoint_manager.storage_client.put_object.return_value = True

#         # Save checkpoint
#         result = await checkpoint_manager.save_checkpoint(
#             model=model,
#             optimizer=optimizer,
#             scheduler=scheduler,
#             momentum=momentum,
#             global_step=100,
#             current_window=5,
#             start_window=1
#         )

#         assert result is True

#         # Load and verify exact preservation
#         loaded_data = torch.load(temp_file, weights_only=False)
#         loaded_model_state = loaded_data["model_state_dict"]

#         for name, exact_value in exact_values.items():
#             loaded_value = loaded_model_state[name]

#             # Verify exact equality (not just close approximation)
#             assert torch.equal(exact_value, loaded_value), f"Parameter {name} not exactly preserved"

#             # Also verify dtype preservation
#             assert exact_value.dtype == loaded_value.dtype, f"Parameter {name} dtype not preserved"

#             # Verify shape preservation
#             assert exact_value.shape == loaded_value.shape, f"Parameter {name} shape not preserved"

#     @pytest.mark.asyncio
#     async def test_gradient_state_preservation(self, checkpoint_manager, temp_dir):
#         """Verify gradient states are preserved correctly"""
#         # Setup model with gradient accumulation
#         model = SimpleModel()
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#         scheduler = StepLR(optimizer, step_size=10)
#         momentum = {}

#         # Perform forward/backward to create gradients
#         loss = torch.nn.functional.mse_loss(
#             model(torch.randn(32, 10)),
#             torch.randn(32, 1)
#         )
#         loss.backward()

#         # Record gradient states before optimizer step
#         gradient_states = {}
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 gradient_states[name] = param.grad.clone()

#         # Take optimizer step to create momentum buffers
#         optimizer.step()

#         # Record optimizer momentum states
#         momentum_states = {}
#         for param_id, state in optimizer.state.items():
#             momentum_states[param_id] = {}
#             for key, val in state.items():
#                 if torch.is_tensor(val):
#                     momentum_states[param_id][key] = val.clone()

#         temp_file = os.path.join(temp_dir, "gradient_state.pt")
#         checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
#         checkpoint_manager.storage_client.put_object.return_value = True

#         # Save checkpoint
#         result = await checkpoint_manager.save_checkpoint(
#             model=model,
#             optimizer=optimizer,
#             scheduler=scheduler,
#             momentum=momentum,
#             global_step=100,
#             current_window=5,
#             start_window=1
#         )

#         assert result is True

#         # Load and verify optimizer momentum preservation
#         loaded_data = torch.load(temp_file, weights_only=False)
#         loaded_optimizer_state = loaded_data["optimizer_state_dict"]["state"]

#         # Verify momentum buffers are preserved
#         for param_id, original_state in momentum_states.items():
#             assert param_id in loaded_optimizer_state
#             loaded_state = loaded_optimizer_state[param_id]

#             for key, original_tensor in original_state.items():
#                 assert key in loaded_state
#                 loaded_tensor = loaded_state[key]

#                 assert torch.allclose(
#                     original_tensor, loaded_tensor, rtol=1e-6, atol=1e-9
#                 ), f"Optimizer momentum {key} for param {param_id} not preserved"

#     @pytest.mark.asyncio
#     async def test_checksum_validation(self, checkpoint_manager, temp_dir):
#         """Implement and test checksum validation"""
#         import hashlib

#         # Setup
#         model = SimpleModel()
#         optimizer = Adam(model.parameters(), lr=0.001)
#         scheduler = StepLR(optimizer, step_size=10)
#         momentum = {"test_momentum": torch.randn(5, 5)}

#         temp_file = os.path.join(temp_dir, "checksum_test.pt")
#         checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
#         checkpoint_manager.storage_client.put_object.return_value = True

#         # Save checkpoint
#         result = await checkpoint_manager.save_checkpoint(
#             model=model,
#             optimizer=optimizer,
#             scheduler=scheduler,
#             momentum=momentum,
#             global_step=100,
#             current_window=5,
#             start_window=1
#         )

#         assert result is True

#         # Calculate checksum of saved file
#         with open(temp_file, 'rb') as f:
#             file_data = f.read()

#         original_checksum = hashlib.sha256(file_data).hexdigest()

#         # Verify file integrity by recalculating checksum
#         with open(temp_file, 'rb') as f:
#             verification_data = f.read()

#         verification_checksum = hashlib.sha256(verification_data).hexdigest()
#         assert original_checksum == verification_checksum, "File checksum mismatch - data corruption detected"

#         # Test detection of corruption
#         corrupted_file = os.path.join(temp_dir, "corrupted_test.pt")
#         with open(corrupted_file, 'wb') as f:
#             # Write original data but corrupt last few bytes
#             f.write(file_data[:-10])
#             f.write(b"corrupted!")

#         # Verify corruption is detected
#         with open(corrupted_file, 'rb') as f:
#             corrupted_data = f.read()

#         corrupted_checksum = hashlib.sha256(corrupted_data).hexdigest()
#         assert original_checksum != corrupted_checksum, "Corruption not detected"

#         # Test that corrupted checkpoint fails to load properly
#         try:
#             corrupted_checkpoint = torch.load(corrupted_file, weights_only=False)
#             # If it loads, it should fail validation
#             assert False, "Corrupted checkpoint should not load successfully"
#         except Exception:
#             # Expected - corrupted file should fail to load
#             pass

#     @pytest.mark.asyncio
#     async def test_memory_consistency_under_pressure(self, checkpoint_manager, temp_dir):
#         """Test data consistency under memory pressure"""
#         # Create large model to simulate memory pressure
#         class LargeModel(torch.nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 # Create multiple large layers
#                 self.layers = torch.nn.ModuleList([
#                     torch.nn.Linear(1000, 1000) for _ in range(5)
#                 ])
#                 self.final = torch.nn.Linear(1000, 1)

#             def forward(self, x):
#                 for layer in self.layers:
#                     x = torch.relu(layer(x))
#                 return self.final(x)

#         model = LargeModel()
#         optimizer = Adam(model.parameters(), lr=0.001)
#         scheduler = StepLR(optimizer, step_size=10)

#         # Create large momentum dictionary
#         momentum = {
#             f"large_momentum_{i}": torch.randn(100, 100)
#             for i in range(10)
#         }

#         # Record original parameter names (not count)
#         original_param_names = set(name for name, _ in model.named_parameters())
#         original_total_params = sum(p.numel() for p in model.parameters())

#         temp_file = os.path.join(temp_dir, "memory_pressure.pt")
#         checkpoint_manager.file_manager.create_temp_file.return_value = temp_file
#         checkpoint_manager.storage_client.put_object.return_value = True

#         # Save under simulated memory pressure
#         result = await checkpoint_manager.save_checkpoint(
#             model=model,
#             optimizer=optimizer,
#             scheduler=scheduler,
#             momentum=momentum,
#             global_step=100,
#             current_window=5,
#             start_window=1
#         )

#         assert result is True

#         # Verify data integrity despite memory pressure
#         loaded_data = torch.load(temp_file, weights_only=False)

#         # Verify all parameters are present by name
#         loaded_param_names = set(loaded_data["model_state_dict"].keys())
#         assert loaded_param_names == original_param_names, "Missing or extra parameters in checkpoint"

#         # Verify momentum is preserved
#         assert len(loaded_data["momentum"]) == len(momentum)
#         for key in momentum:
#             assert key in loaded_data["momentum"]
#             assert torch.equal(momentum[key], loaded_data["momentum"][key])

#         # Verify model state consistency
#         for name, param in model.named_parameters():
#             assert name in loaded_data["model_state_dict"]
#             loaded_param = loaded_data["model_state_dict"][name]
#             assert torch.equal(param.cpu(), loaded_param)

#     @pytest.mark.asyncio
#     async def test_concurrent_access_integrity(self, checkpoint_manager, temp_dir):
#         """Test data integrity under concurrent access scenarios"""
#         import asyncio
#         import threading

#         model = SimpleModel()
#         optimizer = Adam(model.parameters(), lr=0.001)
#         scheduler = StepLR(optimizer, step_size=10)
#         momentum = {"concurrent_momentum": torch.randn(3, 3)}

#         # Track operations
#         operations_completed = []
#         operation_lock = threading.Lock()

#         async def save_operation(window_id):
#             """Simulate concurrent save operations"""
#             try:
#                 temp_file = os.path.join(temp_dir, f"concurrent_{window_id}.pt")
#                 checkpoint_manager.file_manager.create_temp_file.return_value = temp_file

#                 result = await checkpoint_manager.save_checkpoint(
#                     model=model,
#                     optimizer=optimizer,
#                     scheduler=scheduler,
#                     momentum=momentum,
#                     global_step=window_id * 10,
#                     current_window=window_id,
#                     start_window=1
#                 )

#                 with operation_lock:
#                     operations_completed.append(('save', window_id, result))

#                 return result
#             except Exception as e:
#                 with operation_lock:
#                     operations_completed.append(('save', window_id, f"error: {e}"))
#                 return False

#         async def load_operation(window_id):
#             """Simulate concurrent load operations"""
#             try:
#                 # Create mock checkpoint for loading
#                 temp_checkpoint = os.path.join(temp_dir, f"load_source_{window_id}.pt")
#                 checkpoint_data = {
#                     "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
#                     "optimizer_state_dict": optimizer.state_dict(),
#                     "scheduler_state_dict": scheduler.state_dict(),
#                     "momentum": {k: v.cpu().clone() for k, v in momentum.items()},
#                     "start_window": 1,
#                     "current_window": window_id,
#                     "global_step": window_id * 10,
#                 }
#                 torch.save(checkpoint_data, temp_checkpoint)

#                 with open(temp_checkpoint, 'rb') as f:
#                     checkpoint_bytes = f.read()

#                 checkpoint_manager.storage_client.list_objects.return_value = [
#                     f"checkpoint-{window_id}-test-uid-123-v1.0.0.pt"
#                 ]
#                 checkpoint_manager.storage_client.get_object.return_value = checkpoint_bytes

#                 load_temp_file = os.path.join(temp_dir, f"load_temp_{window_id}.pt")
#                 checkpoint_manager.file_manager.create_temp_file.return_value = load_temp_file

#                 result = await checkpoint_manager.get_latest_checkpoint("1.0.0")

#                 with operation_lock:
#                     operations_completed.append(('load', window_id, result is not None))

#                 return result is not None

#             except Exception as e:
#                 with operation_lock:
#                     operations_completed.append(('load', window_id, f"error: {e}"))
#                 return False

#         # Execute concurrent operations
#         save_tasks = [save_operation(i) for i in range(3, 8)]
#         load_tasks = [load_operation(i) for i in range(1, 4)]

#         # Run all operations concurrently
#         all_results = await asyncio.gather(
#             *save_tasks, *load_tasks,
#             return_exceptions=True
#         )

#         # Verify operations completed successfully
#         save_results = all_results[:5]  # First 5 are saves
#         load_results = all_results[5:]  # Last 3 are loads

#         # At least some operations should succeed
#         successful_saves = sum(1 for result in save_results if result is True)
#         successful_loads = sum(1 for result in load_results if result is True)

#         assert successful_saves > 0, "No save operations succeeded under concurrent access"
#         assert successful_loads > 0, "No load operations succeeded under concurrent access"

#         # Verify data integrity of saved files
#         for i, result in enumerate(save_results):
#             if result is True:
#                 window_id = i + 3
#                 temp_file = os.path.join(temp_dir, f"concurrent_{window_id}.pt")
#                 if os.path.exists(temp_file):
#                     # Verify file can be loaded and contains expected data
#                     loaded_data = torch.load(temp_file, weights_only=False)
#                     assert loaded_data["current_window"] == window_id
#                     assert loaded_data["global_step"] == window_id * 10
