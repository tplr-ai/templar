"""
Tests for trainer.init_model with meta parameter
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from neurons.trainer import Trainer


class TestTrainerMetaInit:
    """Test suite for Trainer.init_model with meta parameter"""

    @pytest.fixture
    def trainer(self):
        """Create a trainer instance for testing"""
        trainer = Trainer()

        # Mock required attributes
        trainer.hparams = MagicMock()
        trainer.hparams.tokenizer = MagicMock()
        trainer.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainer.world_size = 1

        return trainer

    def test_init_model_without_meta_calls_initialize(self, trainer):
        """Test init_model(meta=False) calls initialize_torchtitan_model"""
        # Mock the model factory
        with patch("neurons.trainer.model_factory") as mock_factory:
            mock_model = MagicMock()
            mock_factory.initialize_torchtitan_model.return_value = mock_model
            mock_factory.create_meta_model.return_value = MagicMock()

            # Call init_model without meta
            trainer.init_model(validator=False, meta=False)

            # Verify initialize_torchtitan_model was called
            mock_factory.initialize_torchtitan_model.assert_called_once_with(
                hparams=trainer.hparams,
                role="miner",
                device=str(trainer.device),
                world_size=trainer.world_size,
            )

            # Verify create_meta_model was NOT called
            mock_factory.create_meta_model.assert_not_called()

            # Verify model is assigned
            assert trainer.model == mock_model

    def test_init_model_with_meta_calls_create_meta(self, trainer):
        """Test init_model(meta=True) calls create_meta_model"""
        # Mock the model factory
        with patch("neurons.trainer.model_factory") as mock_factory:
            mock_meta_model = MagicMock()
            mock_factory.create_meta_model.return_value = mock_meta_model
            mock_factory.initialize_torchtitan_model.return_value = MagicMock()

            # Call init_model with meta=True
            trainer.init_model(validator=False, meta=True)

            # Verify create_meta_model was called
            mock_factory.create_meta_model.assert_called_once_with(
                hparams=trainer.hparams, role="miner", world_size=trainer.world_size
            )

            # Verify initialize_torchtitan_model was NOT called
            mock_factory.initialize_torchtitan_model.assert_not_called()

            # Verify model is assigned
            assert trainer.model == mock_meta_model

    def test_init_model_validator_role(self, trainer):
        """Test init_model passes correct role for validator"""
        # Mock the model factory
        with patch("neurons.trainer.model_factory") as mock_factory:
            mock_model = MagicMock()
            mock_factory.initialize_torchtitan_model.return_value = mock_model

            # Call init_model as validator
            trainer.init_model(validator=True, meta=False)

            # Verify role is "validator"
            mock_factory.initialize_torchtitan_model.assert_called_once_with(
                hparams=trainer.hparams,
                role="validator",
                device=str(trainer.device),
                world_size=trainer.world_size,
            )

    def test_init_model_validator_with_meta(self, trainer):
        """Test init_model as validator with meta=True"""
        # Mock the model factory
        with patch("neurons.trainer.model_factory") as mock_factory:
            mock_meta_model = MagicMock()
            mock_factory.create_meta_model.return_value = mock_meta_model

            # Call init_model as validator with meta
            trainer.init_model(validator=True, meta=True)

            # Verify role is "validator" and meta model is created
            mock_factory.create_meta_model.assert_called_once_with(
                hparams=trainer.hparams, role="validator", world_size=trainer.world_size
            )

    def test_init_model_sets_expected_compressed_params(self, trainer):
        """Test init_model sets expected_compressed_params"""
        # Mock the model factory
        with patch("neurons.trainer.model_factory") as mock_factory:
            # Create a mock model with named parameters
            mock_model = MagicMock()
            mock_model.named_parameters.return_value = [
                ("layer1.weight", MagicMock()),
                ("layer1.bias", MagicMock()),
                ("layer2.weight", MagicMock()),
            ]
            mock_factory.initialize_torchtitan_model.return_value = mock_model

            # Call init_model
            trainer.init_model(validator=False, meta=False)

            # Verify expected_compressed_params is set correctly
            expected_params = {
                "layer1.weightidxs",
                "layer1.weightvals",
                "layer1.weightquant_params",
                "layer1.biasidxs",
                "layer1.biasvals",
                "layer1.biasquant_params",
                "layer2.weightidxs",
                "layer2.weightvals",
                "layer2.weightquant_params",
            }
            assert trainer.expected_compressed_params == expected_params

    def test_init_model_sets_tokenizer(self, trainer):
        """Test init_model sets tokenizer from hparams"""
        # Mock the model factory
        with patch("neurons.trainer.model_factory") as mock_factory:
            mock_model = MagicMock()
            mock_factory.initialize_torchtitan_model.return_value = mock_model

            # Set a specific tokenizer
            trainer.hparams.tokenizer = "test_tokenizer"

            # Call init_model
            trainer.init_model(validator=False, meta=False)

            # Verify tokenizer is set
            assert trainer.tokenizer == "test_tokenizer"

    def test_init_model_meta_no_device_param(self, trainer):
        """Test meta model creation doesn't pass device parameter"""
        # Mock the model factory
        with patch("neurons.trainer.model_factory") as mock_factory:
            mock_meta_model = MagicMock()
            mock_factory.create_meta_model.return_value = mock_meta_model

            # Call init_model with meta=True
            trainer.init_model(validator=False, meta=True)

            # Verify create_meta_model was called WITHOUT device parameter
            call_kwargs = mock_factory.create_meta_model.call_args[1]
            assert "device" not in call_kwargs
            assert "hparams" in call_kwargs
            assert "role" in call_kwargs
            assert "world_size" in call_kwargs

    def test_bare_model_removed(self, trainer):
        """Test that bare_model is no longer set"""
        # Mock the model factory
        with patch("neurons.trainer.model_factory") as mock_factory:
            mock_model = MagicMock()
            mock_factory.initialize_torchtitan_model.return_value = mock_model

            # Call init_model
            trainer.init_model(validator=False, meta=False)

            # Verify bare_model is NOT set (attribute should not exist)
            assert not hasattr(trainer, "bare_model")

            # Verify model is set
            assert trainer.model == mock_model
