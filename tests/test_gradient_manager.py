"""
Comprehensive test suite for GradientManager class
Covers normal cases, edge cases, error conditions, and boundary scenarios
"""

import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, patch

from src.tplr.training.gradient_manager import GradientManager
from src.tplr.storage.client import StorageClient
from src.tplr.storage.file_manager import FileManager


@pytest.fixture
def mock_storage_client():
    return Mock(spec=StorageClient)


@pytest.fixture
def mock_file_manager():
    manager = Mock(spec=FileManager)
    manager.create_temp_file.return_value = "/tmp/test_gradient_file.pt"
    return manager


@pytest.fixture
def mock_hparams():
    hparams = Mock()
    hparams.topk_compression = 3
    return hparams


@pytest.fixture
def gradient_manager(mock_storage_client, mock_file_manager, mock_hparams):
    return GradientManager(
        storage_client=mock_storage_client,
        file_manager=mock_file_manager,
        device="cpu",
        hparams=mock_hparams,
    )


class TestGradientManagerInit:
    """Test GradientManager initialization"""

    def test_init_success(self, mock_storage_client, mock_file_manager, mock_hparams):
        device = "cuda"

        gm = GradientManager(
            mock_storage_client, mock_file_manager, device, mock_hparams
        )

        assert gm.storage_client == mock_storage_client
        assert gm.file_manager == mock_file_manager
        assert gm.device == device
        assert gm.hparams == mock_hparams

    def test_init_with_none_storage_client(self, mock_file_manager, mock_hparams):
        # Should handle None gracefully - assignment happens regardless
        gm = GradientManager(None, mock_file_manager, "cpu", mock_hparams)
        assert gm.storage_client is None
        assert gm.file_manager == mock_file_manager
        assert gm.device == "cpu"
        assert gm.hparams == mock_hparams

    def test_init_with_invalid_device(
        self, mock_storage_client, mock_file_manager, mock_hparams
    ):
        # Test with empty string, invalid device names
        invalid_devices = ["", "invalid_device", "gpu", None]
        for device in invalid_devices:
            gm = GradientManager(
                mock_storage_client, mock_file_manager, device, mock_hparams
            )
            assert gm.device == device  # Assignment happens regardless of validity

    def test_init_missing_topk_compression_hparam(
        self, mock_storage_client, mock_file_manager
    ):
        hparams = Mock()
        # Don't set topk_compression attribute
        gm = GradientManager(mock_storage_client, mock_file_manager, "cpu", hparams)
        assert gm.hparams == hparams


class TestSerializeGradient:
    """Test gradient serialization functionality"""

    @pytest.mark.asyncio
    async def test_serialize_gradient_success(self, gradient_manager):
        state_dict = {
            "layer1.weight.vals": torch.tensor([1.0, 2.0, 3.0]),
            "layer1.weight.idxs": torch.tensor([0, 1, 2]),
        }
        global_step = 42

        with patch("torch.save") as mock_save:
            result = await gradient_manager.serialize_gradient(state_dict, global_step)

            assert result == "/tmp/test_gradient_file.pt"
            gradient_manager.file_manager.create_temp_file.assert_called_once_with(
                "gradient_serialize"
            )
            mock_save.assert_called_once()
            save_data = mock_save.call_args[0][0]
            assert save_data["state_dict"] == state_dict
            assert save_data["global_step"] == global_step

    @pytest.mark.asyncio
    async def test_serialize_gradient_empty_state_dict(self, gradient_manager):
        state_dict = {}
        global_step = 0

        with patch("torch.save") as mock_save:
            result = await gradient_manager.serialize_gradient(state_dict, global_step)

            assert result == "/tmp/test_gradient_file.pt"
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_serialize_gradient_large_state_dict(self, gradient_manager):
        # Test memory handling with large tensors
        state_dict = {f"layer{i}.weight.vals": torch.randn(10000) for i in range(100)}
        global_step = 1000

        with patch("torch.save"):
            result = await gradient_manager.serialize_gradient(state_dict, global_step)
            assert result == "/tmp/test_gradient_file.pt"

    @pytest.mark.asyncio
    async def test_serialize_gradient_negative_global_step(self, gradient_manager):
        state_dict = {"param": torch.tensor([1.0])}
        global_step = -5

        with patch("torch.save") as mock_save:
            await gradient_manager.serialize_gradient(state_dict, global_step)

            save_data = mock_save.call_args[0][0]
            assert save_data["global_step"] == -5

    @pytest.mark.asyncio
    async def test_serialize_gradient_none_values(self, gradient_manager):
        state_dict = {"param": None}
        global_step = 0

        with patch("torch.save") as mock_save:
            await gradient_manager.serialize_gradient(state_dict, global_step)

            save_data = mock_save.call_args[0][0]
            assert save_data["state_dict"]["param"] is None

    @pytest.mark.asyncio
    async def test_serialize_gradient_temp_file_creation_fails(self, gradient_manager):
        gradient_manager.file_manager.create_temp_file.side_effect = Exception(
            "File creation failed"
        )

        with pytest.raises(Exception, match="File creation failed"):
            await gradient_manager.serialize_gradient({}, 0)

    @pytest.mark.asyncio
    async def test_serialize_gradient_torch_save_fails(self, gradient_manager):
        state_dict = {"param": torch.tensor([1.0])}

        with patch("torch.save", side_effect=RuntimeError("Disk full")):
            with pytest.raises(RuntimeError, match="Disk full"):
                await gradient_manager.serialize_gradient(state_dict, 0)

    @pytest.mark.asyncio
    async def test_serialize_gradient_complex_tensors(self, gradient_manager):
        state_dict = {
            "float16": torch.tensor([1.0], dtype=torch.float16),
            "bfloat16": torch.tensor([1.0], dtype=torch.bfloat16),
            "complex64": torch.tensor([1.0 + 2.0j], dtype=torch.complex64),
            "int64": torch.tensor([1], dtype=torch.int64),
        }

        with patch("torch.save") as mock_save:
            await gradient_manager.serialize_gradient(state_dict, 0)
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_serialize_gradient_mixed_device_tensors(self, gradient_manager):
        state_dict = {
            "cpu_tensor": torch.tensor([1.0]),
            "other_tensor": torch.tensor([2.0]),  # Assume on same device
        }

        with patch("torch.save") as mock_save:
            await gradient_manager.serialize_gradient(state_dict, 0)
            mock_save.assert_called_once()


class TestDeserializeGradient:
    """Test gradient deserialization functionality"""

    @pytest.mark.asyncio
    async def test_deserialize_gradient_success(self, gradient_manager):
        expected_state_dict = {"param": torch.tensor([1.0, 2.0])}
        expected_global_step = 42
        mock_data = {
            "state_dict": expected_state_dict,
            "global_step": expected_global_step,
        }

        with patch("torch.load", return_value=mock_data) as mock_load:
            state_dict, global_step = await gradient_manager.deserialize_gradient(
                "/path/to/file.pt"
            )

            mock_load.assert_called_once_with(
                "/path/to/file.pt", map_location="cpu", weights_only=False
            )
            assert state_dict == expected_state_dict
            assert global_step == expected_global_step

    @pytest.mark.asyncio
    async def test_deserialize_gradient_file_not_found(self, gradient_manager):
        with patch("torch.load", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                await gradient_manager.deserialize_gradient("/nonexistent/file.pt")

    @pytest.mark.asyncio
    async def test_deserialize_gradient_corrupted_file(self, gradient_manager):
        with patch("torch.load", side_effect=RuntimeError("Invalid file format")):
            with pytest.raises(RuntimeError):
                await gradient_manager.deserialize_gradient("/corrupted/file.pt")

    @pytest.mark.asyncio
    async def test_deserialize_gradient_missing_state_dict(self, gradient_manager):
        mock_data = {"global_step": 42}  # Missing state_dict

        with patch("torch.load", return_value=mock_data):
            state_dict, global_step = await gradient_manager.deserialize_gradient(
                "/path/to/file.pt"
            )

            assert state_dict == {}
            assert global_step == 42

    @pytest.mark.asyncio
    async def test_deserialize_gradient_missing_global_step(self, gradient_manager):
        mock_data = {
            "state_dict": {"param": torch.tensor([1.0])}
        }  # Missing global_step

        with patch("torch.load", return_value=mock_data):
            state_dict, global_step = await gradient_manager.deserialize_gradient(
                "/path/to/file.pt"
            )

            assert global_step == 0
            assert "param" in state_dict

    @pytest.mark.asyncio
    async def test_deserialize_gradient_device_mapping(self, gradient_manager):
        gradient_manager.device = "cuda"
        mock_data = {"state_dict": {"param": torch.tensor([1.0])}, "global_step": 0}

        with patch("torch.load", return_value=mock_data) as mock_load:
            await gradient_manager.deserialize_gradient("/path/to/file.pt")

            mock_load.assert_called_once_with(
                "/path/to/file.pt", map_location="cuda", weights_only=False
            )

    @pytest.mark.asyncio
    async def test_deserialize_gradient_permission_denied(self, gradient_manager):
        with patch("torch.load", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                await gradient_manager.deserialize_gradient("/restricted/file.pt")

    @pytest.mark.asyncio
    async def test_deserialize_gradient_empty_file(self, gradient_manager):
        with patch("torch.load", side_effect=EOFError("Empty file")):
            with pytest.raises(EOFError):
                await gradient_manager.deserialize_gradient("/empty/file.pt")

    @pytest.mark.asyncio
    async def test_deserialize_gradient_partial_data(self, gradient_manager):
        mock_data = {}  # Empty dict - no state_dict or global_step

        with patch("torch.load", return_value=mock_data):
            state_dict, global_step = await gradient_manager.deserialize_gradient(
                "/path/to/file.pt"
            )

            assert state_dict == {}
            assert global_step == 0


class TestValidateGradient:
    """Test gradient validation functionality"""

    def test_validate_gradient_success(self, gradient_manager):
        #  Use topk_compression=3 to match the actual indices count
        gradient_manager.hparams.topk_compression = 3

        state_dict = {
            "layer1.weightidxs": torch.tensor([0, 1, 2]),  # 3 indices
            "layer1.weightvals": torch.tensor([1.0, 2.0, 3.0]),
            "layer2.biasidxs": torch.tensor([0]),  # 1 index
            "layer2.biasvals": torch.tensor([0.5]),
        }

        # Temporarily change topk_compression for layer2.bias check
        original_topk = gradient_manager.hparams.topk_compression

        # Test layer1.weight with 3 indices
        result = gradient_manager.validate_gradient(
            {
                "layer1.weightidxs": state_dict["layer1.weightidxs"],
                "layer1.weightvals": state_dict["layer1.weightvals"],
            },
            {"layer1.weight": 1000},
        )
        assert result is True

        # Test layer2.bias with 1 index
        gradient_manager.hparams.topk_compression = 1
        result = gradient_manager.validate_gradient(
            {
                "layer2.biasidxs": state_dict["layer2.biasidxs"],
                "layer2.biasvals": state_dict["layer2.biasvals"],
            },
            {"layer2.bias": 100},
        )
        assert result is True

        # Restore original
        gradient_manager.hparams.topk_compression = original_topk

    def test_validate_gradient_empty_state_dict(self, gradient_manager):
        result = gradient_manager.validate_gradient({}, {})
        assert result is True

    @patch("tplr.logger")
    def test_validate_gradient_missing_totalk(self, mock_logger, gradient_manager):
        state_dict = {"layer1.weight.idxs": torch.tensor([0, 1])}
        totalks = {}  # Missing layer1.weight

        result = gradient_manager.validate_gradient(state_dict, totalks)

        assert result is False
        mock_logger.warning.assert_called()

    @patch("tplr.logger")
    def test_validate_gradient_nan_values(self, mock_logger, gradient_manager):
        state_dict = {"layer1.weight.vals": torch.tensor([1.0, float("nan"), 3.0])}
        totalks = {}

        result = gradient_manager.validate_gradient(state_dict, totalks)

        assert result is False
        mock_logger.warning.assert_called()

    @patch("tplr.logger")
    def test_validate_gradient_inf_values(self, mock_logger, gradient_manager):
        state_dict = {"layer1.weight.vals": torch.tensor([1.0, float("inf"), 3.0])}
        totalks = {}

        result = gradient_manager.validate_gradient(state_dict, totalks)

        assert result is False
        mock_logger.warning.assert_called()

    @patch("tplr.logger")
    def test_validate_gradient_invalid_indices(self, mock_logger, gradient_manager):
        state_dict = {
            "layer1.weight.idxs": torch.tensor([0, 1000, 2])  # 1000 out of bounds
        }
        totalks = {"layer1.weight": 100}  # max index should be 99

        result = gradient_manager.validate_gradient(state_dict, totalks)

        assert result is False
        mock_logger.warning.assert_called()

    def test_validate_gradient_other_parameters(self, gradient_manager):
        state_dict = {
            "layer1.weight": torch.tensor([1.0, 2.0]),  # Doesn't end with idxs/vals
            "layer1.bias.custom": torch.tensor([3.0]),  # Doesn't end with idxs/vals
        }
        totalks = {}

        result = gradient_manager.validate_gradient(state_dict, totalks)
        assert result is True  # Should ignore non-idxs/vals parameters

    def test_validate_gradient_device_transfer_fails(self, gradient_manager):
        # Create tensor that will fail to transfer to device
        mock_tensor = Mock()
        mock_tensor.to.side_effect = RuntimeError("Device transfer failed")

        state_dict = {"layer1.weight.idxs": mock_tensor}
        totalks = {"layer1.weight": 100}

        with patch("tplr.logger"):
            result = gradient_manager.validate_gradient(state_dict, totalks)
            assert result is False

    def test_validate_gradient_mixed_validity(self, gradient_manager):
        state_dict = {
            "layer1.weight.idxs": torch.tensor([0, 1]),  # Valid
            "layer1.weight.vals": torch.tensor([1.0, 2.0]),  # Valid
            "layer2.bias.vals": torch.tensor([float("nan")]),  # Invalid
        }
        totalks = {"layer1.weight": 100}

        with patch("tplr.logger"):
            result = gradient_manager.validate_gradient(state_dict, totalks)
            assert result is False  # Should fail due to NaN

    def test_validate_gradient_none_totalks(self, gradient_manager):
        state_dict = {"layer1.weight.idxs": torch.tensor([0])}
        totalks = {"layer1.weight": None}

        with patch("tplr.logger"):
            result = gradient_manager.validate_gradient(state_dict, totalks)
            assert result is False


class TestCheckCompressedIndices:
    """Test compressed indices validation"""

    def test_check_compressed_indices_valid_flat(self, gradient_manager):
        idxs = torch.tensor([0, 1, 2, 3, 4])
        gradient_manager.hparams.topk_compression = 5

        # Should not raise
        gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=100)

    def test_check_compressed_indices_valid_multidim(self, gradient_manager):
        idxs = torch.tensor([[0, 1, 2], [3, 4, 5]])  # 2x3, last dim = 3
        gradient_manager.hparams.topk_compression = 3

        gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=100)

    def test_check_compressed_indices_valid_scalar(self, gradient_manager):
        idxs = 42  # Single scalar

        gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=100)

    def test_check_compressed_indices_valid_nested_list(self, gradient_manager):
        idxs = [[0, 1, 2], [3, 4, 5]]  # Nested list structure
        gradient_manager.hparams.topk_compression = 3

        gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=100)

    def test_check_compressed_indices_negative_index(self, gradient_manager):
        idxs = torch.tensor([0, -1, 2])
        gradient_manager.hparams.topk_compression = 3

        with pytest.raises(ValueError, match="Index -1 out of bounds"):
            gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=100)

    def test_check_compressed_indices_index_too_large(self, gradient_manager):
        idxs = torch.tensor([0, 1, 100])  # 100 >= totalk=50
        gradient_manager.hparams.topk_compression = 3

        with pytest.raises(ValueError, match="Index 100 out of bounds"):
            gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=50)

    def test_check_compressed_indices_empty_tensor(self, gradient_manager):
        idxs = torch.tensor([])

        # The actual error message is about invalid number of indices, not empty index list
        with pytest.raises(ValueError, match="Invalid number of indices"):
            gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=100)

    def test_check_compressed_indices_wrong_count_flat(self, gradient_manager):
        idxs = torch.tensor([0, 1])  # 2 elements
        gradient_manager.hparams.topk_compression = 5  # Expected 5

        with pytest.raises(ValueError, match="Invalid number of indices"):
            gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=100)

    def test_check_compressed_indices_wrong_last_dim(self, gradient_manager):
        idxs = torch.tensor([[0, 1], [2, 3]])  # Last dim = 2
        gradient_manager.hparams.topk_compression = 3  # Expected 3

        with pytest.raises(ValueError, match="Last dimension size invalid"):
            gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=100)

    def test_check_compressed_indices_unconvertible(self, gradient_manager):
        idxs = object()  # Cannot convert to tensor

        with pytest.raises(ValueError, match="Failed to convert indices to tensor"):
            gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=100)

    def test_check_compressed_indices_wrong_sublist_length(self, gradient_manager):
        idxs = [[0, 1], [2, 3, 4]]  # Different lengths
        gradient_manager.hparams.topk_compression = 3

        with pytest.raises(ValueError, match="Invalid number of indices"):
            gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=100)

    def test_check_compressed_indices_scalar_float(self, gradient_manager):
        idxs = 42.7  # Float scalar

        # Should convert to int and validate
        gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=100)

    def test_check_compressed_indices_none_allowed_topk(self, gradient_manager):
        idxs = torch.tensor([0, 1, 2])
        gradient_manager.hparams.topk_compression = 5

        # Should use min(5, 100) = 5, but tensor has 3 elements
        with pytest.raises(ValueError, match="Invalid number of indices"):
            gradient_manager.check_compressed_indices(
                "param.idxs", idxs, totalk=100, allowed_topk=None
            )

    def test_check_compressed_indices_zero_totalk(self, gradient_manager):
        idxs = 0

        with pytest.raises(ValueError, match="out of bounds"):
            gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=0)

    def test_check_compressed_indices_topk_exceeds_totalk(self, gradient_manager):
        idxs = torch.tensor([0, 1])  # 2 elements
        gradient_manager.hparams.topk_compression = 100
        totalk = 50

        # allowed_topk becomes min(100, 50) = 50, but we have 2 elements
        with pytest.raises(ValueError, match="Invalid number of indices"):
            gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=totalk)


class TestApplyGradientsToModel:
    """Test gradient application to model"""

    @pytest.mark.asyncio
    async def test_apply_gradients_success(self, gradient_manager):
        # topk_compression to match test data
        gradient_manager.hparams.topk_compression = 2

        # Mock model with named parameters
        model = Mock()
        param1 = Mock()
        param1.grad = None
        param2 = Mock()
        param2.grad = torch.tensor([1.0])
        model.named_parameters.return_value = [
            ("layer1.weight", param1),
            ("layer2.bias", param2),
        ]
        model.train = Mock()
        model.zero_grad = Mock()

        # Mock optimizer and scheduler
        optimizer = Mock()
        scheduler = Mock()

        # Mock transformer and compressor - accept all arguments with *args, **kwargs
        def transformer_decode_side_effect(*args, **kwargs):
            # Extract shape from args - usually the last argument
            if len(args) >= 2:
                shape = args[1]
                if shape == (2,):  # layer1.weight
                    return torch.tensor([0.5, 0.6])
                elif shape == (1,):  # layer2.bias
                    return torch.tensor([0.3])
            return torch.tensor([0.1])

        def compressor_decompress_side_effect(*args, **kwargs):
            # Accept any number of arguments, determine output based on context
            # Check if this is for a parameter with shape (2,) or (1,)
            # We can look at the tensor values or use a simpler approach
            if len(args) >= 3:
                # Usually: vals, idxs, shape, ... other args
                shape = args[2] if len(args) > 2 else (1,)
                if isinstance(shape, tuple) and len(shape) > 0 and shape[0] == 2:
                    return torch.tensor([0.1, 0.2])
                else:
                    return torch.tensor([0.3])
            return torch.tensor([0.1])

        transformer = Mock()
        transformer.decode.side_effect = transformer_decode_side_effect
        compressor = Mock()
        compressor.batch_decompress.side_effect = compressor_decompress_side_effect

        class MockStateDict:
            def __getattr__(self, name):
                return {
                    "layer1.weightidxs": [0, 1],  # 2 indices
                    "layer1.weightvals": [0.5, 0.6],  # 2 values
                    "layer2.biasidxs": [0],  # 1 index
                    "layer2.biasvals": [0.3],  # 1 value
                }.get(name)

        gather_result = Mock()
        gather_result.state_dict = MockStateDict()

        shapes = {"layer1.weight": (2,), "layer2.bias": (1,)}
        totalks = {"layer1.weight": 100, "layer2.bias": 50}

        success, new_step = await gradient_manager.apply_gradients_to_model(
            gather_result,
            model,
            optimizer,
            scheduler,
            transformer,
            compressor,
            "cpu",
            window=1,
            global_step=10,
            shapes=shapes,
            totalks=totalks,
        )

        assert success is True
        assert new_step == 11
        model.train.assert_called_once()
        optimizer.zero_grad.assert_called_once()
        model.zero_grad.assert_called_once()
        optimizer.step.assert_called_once()
        scheduler.step.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_gradients_empty_gather_result(self, gradient_manager):
        success, step = await gradient_manager.apply_gradients_to_model(
            None, Mock(), Mock(), Mock(), Mock(), Mock(), "cpu", 1, 10
        )
        assert success is False
        assert step == 10

        # Test with empty state_dict
        gather_result = Mock()
        gather_result.state_dict = None
        success, step = await gradient_manager.apply_gradients_to_model(
            gather_result, Mock(), Mock(), Mock(), Mock(), Mock(), "cpu", 1, 10
        )
        assert success is False
        assert step == 10

    @pytest.mark.asyncio
    @patch("tplr.logger")
    async def test_apply_gradients_missing_shapes(self, mock_logger, gradient_manager):
        gather_result = Mock()
        gather_result.state_dict = Mock()

        success, step = await gradient_manager.apply_gradients_to_model(
            gather_result,
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            "cpu",
            1,
            10,
            shapes=None,
            totalks={},
        )

        assert success is False
        assert step == 10
        mock_logger.error.assert_called_with(
            "shapes and totalks parameters are required"
        )

    @pytest.mark.asyncio
    @patch("tplr.logger")
    async def test_apply_gradients_missing_totalks(self, mock_logger, gradient_manager):
        gather_result = Mock()
        gather_result.state_dict = Mock()

        success, step = await gradient_manager.apply_gradients_to_model(
            gather_result,
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            "cpu",
            1,
            10,
            shapes={},
            totalks=None,
        )

        assert success is False
        assert step == 10
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    @patch("tplr.logger")
    async def test_apply_gradients_missing_param_shape(
        self, mock_logger, gradient_manager
    ):
        model = Mock()
        param = Mock()
        model.named_parameters.return_value = [("missing_param", param)]

        class MockStateDict:
            def __getattr__(self, name):
                return [0]

        gather_result = Mock()
        gather_result.state_dict = MockStateDict()

        shapes = {}  # missing_param not in shapes
        totalks = {"missing_param": 100}

        success, step = await gradient_manager.apply_gradients_to_model(
            gather_result,
            model,
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            "cpu",
            1,
            10,
            shapes=shapes,
            totalks=totalks,
        )

        assert success is True  # Still succeeds but skips param
        mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    @patch("tplr.logger")
    async def test_apply_gradients_decode_fails(self, mock_logger, gradient_manager):
        model = Mock()
        param = Mock()
        param.to.return_value = param
        model.named_parameters.return_value = [("layer1.weight", param)]

        transformer = Mock()
        transformer.decode.side_effect = RuntimeError("Decode failed")
        compressor = Mock()
        compressor.batch_decompress.return_value = torch.tensor([0.1])

        class MockStateDict:
            def __getattr__(self, name):
                return [0]

        gather_result = Mock()
        gather_result.state_dict = MockStateDict()

        success, step = await gradient_manager.apply_gradients_to_model(
            gather_result,
            model,
            Mock(),
            Mock(),
            transformer,
            compressor,
            "cpu",
            1,
            10,
            shapes={"layer1.weight": (1,)},
            totalks={"layer1.weight": 100},
        )

        assert success is False
        assert step == 10
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_apply_gradients_existing_param_grads(self, gradient_manager):
        model = Mock()
        param = Mock()
        existing_grad = torch.tensor([1.0, 2.0])
        param.grad = existing_grad
        param.grad.copy_ = Mock()
        param.grad.sign_ = Mock()
        param.to.return_value = param
        model.named_parameters.return_value = [("layer1.weight", param)]
        model.train = Mock()
        model.zero_grad = Mock()

        new_grad = torch.tensor([0.5, 0.6])
        transformer = Mock()
        transformer.decode.return_value = new_grad
        compressor = Mock()
        compressor.batch_decompress.return_value = torch.tensor([0.1, 0.2])

        class MockStateDict:
            def __getattr__(self, name):
                return [0, 1]

        gather_result = Mock()
        gather_result.state_dict = MockStateDict()

        success, step = await gradient_manager.apply_gradients_to_model(
            gather_result,
            model,
            Mock(),
            Mock(),
            transformer,
            compressor,
            "cpu",
            1,
            10,
            shapes={"layer1.weight": (2,)},
            totalks={"layer1.weight": 100},
        )

        assert success is True
        param.grad.copy_.assert_called_once_with(new_grad)
        param.grad.sign_.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_gradients_no_existing_param_grads(self, gradient_manager):
        gradient_manager.hparams.topk_compression = 2

        model = Mock()
        param = Mock()
        param.grad = None
        param.to.return_value = param
        model.named_parameters.return_value = [("layer1.weight", param)]
        model.train = Mock()
        model.zero_grad = Mock()

        new_grad = torch.tensor([0.5, 0.6])
        new_grad.sign_ = Mock()
        transformer = Mock()
        transformer.decode.return_value = new_grad
        compressor = Mock()
        compressor.batch_decompress.return_value = torch.tensor([0.1, 0.2])

        class MockStateDict:
            def __getattr__(self, name):
                return [0, 1]

        gather_result = Mock()
        gather_result.state_dict = MockStateDict()

        success, step = await gradient_manager.apply_gradients_to_model(
            gather_result,
            model,
            Mock(),
            Mock(),
            transformer,
            compressor,
            "cpu",
            1,
            10,
            shapes={"layer1.weight": (2,)},
            totalks={"layer1.weight": 100},
        )

        assert success is True
        #  Use torch.equal for tensor comparison
        assert torch.equal(param.grad, new_grad)
        new_grad.sign_.assert_called_once()


class TestValidateGradientTensor:
    """Test individual tensor validation"""

    @patch("tplr.logger")
    def test_validate_tensor_valid(self, mock_logger, gradient_manager):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = gradient_manager.validate_gradient_tensor(tensor, "test_param")
        assert result is True
        mock_logger.warning.assert_not_called()

    @patch("tplr.logger")
    def test_validate_tensor_nan(self, mock_logger, gradient_manager):
        tensor = torch.tensor([1.0, float("nan"), 3.0])
        result = gradient_manager.validate_gradient_tensor(tensor, "test_param")
        assert result is False
        mock_logger.warning.assert_called_with("NaN detected in test_param")

    @patch("tplr.logger")
    def test_validate_tensor_inf(self, mock_logger, gradient_manager):
        tensor = torch.tensor([1.0, float("inf"), 3.0])
        result = gradient_manager.validate_gradient_tensor(tensor, "test_param")
        assert result is False
        mock_logger.warning.assert_called_with("Inf detected in test_param")

    @patch("tplr.logger")
    def test_validate_tensor_neg_inf(self, mock_logger, gradient_manager):
        tensor = torch.tensor([1.0, float("-inf"), 3.0])
        result = gradient_manager.validate_gradient_tensor(tensor, "test_param")
        assert result is False
        mock_logger.warning.assert_called_with("Inf detected in test_param")

    def test_validate_tensor_empty(self, gradient_manager):
        tensor = torch.tensor([])
        result = gradient_manager.validate_gradient_tensor(tensor, "test_param")
        assert result is True  # Empty tensor is valid

    def test_validate_tensor_scalar(self, gradient_manager):
        tensor = torch.tensor(5.0)
        result = gradient_manager.validate_gradient_tensor(tensor, "test_param")
        assert result is True

    def test_validate_tensor_large(self, gradient_manager):
        tensor = torch.randn(10000, 10000)  # Large tensor
        result = gradient_manager.validate_gradient_tensor(tensor, "test_param")
        assert result is True

    def test_validate_tensor_different_dtypes(self, gradient_manager):
        dtypes = [torch.float16, torch.float32, torch.float64, torch.int32, torch.int64]
        for dtype in dtypes:
            tensor = torch.tensor([1, 2, 3], dtype=dtype)
            result = gradient_manager.validate_gradient_tensor(
                tensor, f"test_param_{dtype}"
            )
            assert result is True

    @patch("tplr.logger")
    def test_validate_tensor_operation_exception(self, mock_logger, gradient_manager):
        tensor = Mock()
        tensor.__class__ = torch.Tensor  # Make it look like a tensor
        # Mock torch.isnan to raise exception
        with patch("torch.isnan", side_effect=RuntimeError("Operation failed")):
            result = gradient_manager.validate_gradient_tensor(tensor, "test_param")
            assert result is False
            mock_logger.error.assert_called()


class TestIntegrationScenarios:
    """Test complex integration scenarios"""

    @pytest.mark.asyncio
    async def test_serialize_deserialize_cycle(self, gradient_manager):
        original_state_dict = {
            "layer1.weight.vals": torch.tensor([1.0, 2.0, 3.0]),
            "layer1.weight.idxs": torch.tensor([0, 1, 2]),
            "layer2.bias.vals": torch.tensor([0.5]),
        }
        original_global_step = 42

        # Create a temporary file for real serialization
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            gradient_manager.file_manager.create_temp_file.return_value = tmp_path

            # Serialize
            file_path = await gradient_manager.serialize_gradient(
                original_state_dict, original_global_step
            )
            assert file_path == tmp_path

            # Deserialize
            (
                loaded_state_dict,
                loaded_global_step,
            ) = await gradient_manager.deserialize_gradient(file_path)

            # Verify data integrity
            assert loaded_global_step == original_global_step
            assert len(loaded_state_dict) == len(original_state_dict)
            for key in original_state_dict:
                torch.testing.assert_close(
                    loaded_state_dict[key], original_state_dict[key]
                )

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_validate_after_deserialize(self, gradient_manager):
        #  Set correct topk_compression and use matching test data
        gradient_manager.hparams.topk_compression = 2

        state_dict = {
            "layer1.weightvals": torch.tensor([1.0, 2.0]),
            "layer1.weightidxs": torch.tensor(
                [0, 1]
            ),  # 2 indices to match topk_compression
        }
        totalks = {"layer1.weight": 100}

        # Serialize
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            gradient_manager.file_manager.create_temp_file.return_value = tmp_path
            await gradient_manager.serialize_gradient(state_dict, 0)

            # Deserialize
            loaded_state_dict, _ = await gradient_manager.deserialize_gradient(tmp_path)

            # Validate
            result = gradient_manager.validate_gradient(loaded_state_dict, totalks)
            assert result is True

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestEdgeCasesAndBoundaryConditions:
    """Test boundary conditions and edge cases"""

    def test_max_tensor_size(self, gradient_manager):
        # Test with reasonably large tensor (avoid OOM)
        large_tensor = torch.randn(1000, 1000)
        result = gradient_manager.validate_gradient_tensor(large_tensor, "large_param")
        assert result is True

    def test_min_tensor_size(self, gradient_manager):
        tensor = torch.tensor([42.0])  # Single element
        result = gradient_manager.validate_gradient_tensor(tensor, "min_param")
        assert result is True

    def test_exact_topk_elements(self, gradient_manager):
        gradient_manager.hparams.topk_compression = 3
        idxs = torch.tensor([0, 1, 2])  # Exactly 3 elements
        gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=100)

    def test_topk_one(self, gradient_manager):
        gradient_manager.hparams.topk_compression = 1
        idxs = torch.tensor([42])  # Single element
        gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=100)

    def test_large_topk(self, gradient_manager):
        gradient_manager.hparams.topk_compression = 1000000
        idxs = torch.tensor([0, 1, 2])  # Much smaller than topk
        with pytest.raises(ValueError, match="Invalid number of indices"):
            gradient_manager.check_compressed_indices("param.idxs", idxs, totalk=100)

    @pytest.mark.asyncio
    async def test_global_step_boundaries(self, gradient_manager):
        boundary_values = [0, -1, 2**31 - 1, 2**63 - 1]
        state_dict = {"param": torch.tensor([1.0])}

        for global_step in boundary_values:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                gradient_manager.file_manager.create_temp_file.return_value = tmp_path
                await gradient_manager.serialize_gradient(state_dict, global_step)
                _, loaded_step = await gradient_manager.deserialize_gradient(tmp_path)
                assert loaded_step == global_step
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    def test_special_char_param_names(self, gradient_manager):
        special_names = [
            "layer/with/slashes.idxs",
            "layer-with-dashes.vals",
            "layer.with.dots.idxs",
            "layer@with@symbols.vals",
        ]

        for name in special_names:
            tensor = torch.tensor([1.0, 2.0])
            result = gradient_manager.validate_gradient_tensor(tensor, name)
            assert result is True

    def test_long_param_names(self, gradient_manager):
        long_name = "very_" * 100 + "long_parameter_name.vals"
        tensor = torch.tensor([1.0])
        result = gradient_manager.validate_gradient_tensor(tensor, long_name)
        assert result is True
