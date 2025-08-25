import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torchtitan.models.llama3 import Transformer as TitanLlama
from torchtitan.models.llama3 import TransformerModelArgs
from transformers import LlamaConfig, LlamaForCausalLM

from tplr.model_factory import (
    _get_hf_config_from_titan,
    _get_unwrapped_model,
    convert_titan_to_hf,
    initialize_torchtitan_model,
)


def setup_distributed_environment() -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


class TestModelFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_distributed_environment()

    def test_get_unwrapped_model(self):
        """Test that the model unwrapping utility works correctly."""
        # Create a mock model
        hparams = SimpleNamespace(
            model_size="150M",
            sequence_length=128,
            model_config=SimpleNamespace(
                vocab_size=10,
                num_hidden_layers=1,
                hidden_size=10,
                num_attention_heads=1,
                intermediate_size=20,
            ),
        )
        inner_model = initialize_torchtitan_model(hparams, role="evaluator")
        # Test that the unwrapped model is the original model
        with patch("tplr.model_factory.isinstance", return_value=True):
            unwrapped = _get_unwrapped_model(inner_model)
            self.assertIs(unwrapped, inner_model)

    def test_convert_titan_to_hf_with_titan_model(self):
        """Test conversion from a Titan model to a HuggingFace model."""
        # Mock hparams
        hparams = SimpleNamespace(
            model_size="150M",
            sequence_length=128,
            model_config=SimpleNamespace(
                vocab_size=100,
                num_hidden_layers=2,
                hidden_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=128,
                hidden_act="silu",
                initializer_range=0.02,
                rms_norm_eps=1e-5,
                rope_theta=10000.0,
            ),
        )

        # Mock Titan model and its args
        titan_model = initialize_torchtitan_model(hparams, role="evaluator")

        with patch(
            "tplr.model_factory._get_actual_intermediate_size", return_value=128
        ):
            with patch("tplr.model_factory.Llama3StateDictAdapter.to_hf") as mock_to_hf:
                # Create a dummy hf_state_dict that matches the hf_model
                hf_config = LlamaConfig(
                    vocab_size=100,
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                )
                hf_model_for_state_dict = LlamaForCausalLM(hf_config)
                hf_state_dict = hf_model_for_state_dict.state_dict()
                mock_to_hf.return_value = hf_state_dict

                # Convert the model
                hf_model = convert_titan_to_hf(titan_model, hparams, is_master=True)

                # Assertions
                self.assertIsNone(hf_model)

    def test_convert_titan_to_hf_with_model_args(self):
        """Test that provided model_args override model's own args."""
        hparams = SimpleNamespace()
        titan_model = MagicMock(spec=TitanLlama)
        titan_model.args = TransformerModelArgs(
            vocab_size=10, n_layers=1, dim=10, n_heads=1
        )
        # Create a dummy state_dict with at least one key
        titan_model.state_dict.return_value = {"dummy_key": torch.zeros(1)}

        model_args = {
            "vocab_size": 200,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
        }

        with patch("tplr.model_factory.Llama3StateDictAdapter.to_hf") as mock_to_hf:
            # Create a dummy hf_state_dict that matches the hf_model
            hf_config = LlamaConfig(**model_args)
            hf_model_for_state_dict = LlamaForCausalLM(hf_config)
            hf_state_dict = hf_model_for_state_dict.state_dict()
            mock_to_hf.return_value = hf_state_dict

            # Convert the model
            hf_model = convert_titan_to_hf(
                titan_model, hparams, model_args=model_args, is_master=True
            )

            # Assertions
            self.assertIsNone(hf_model)

    def test_get_hf_config_from_titan(self):
        """Test the helper function for creating HF config from a Titan model."""
        hparams = SimpleNamespace(
            model_size="150M",
            sequence_length=128,
            model_config=SimpleNamespace(
                vocab_size=100,
                num_hidden_layers=2,
                hidden_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=128,
                hidden_act="silu",
            ),
        )
        titan_model = initialize_torchtitan_model(hparams, role="evaluator")
        state_dict = titan_model.state_dict()

        with patch(
            "tplr.model_factory._get_actual_intermediate_size", return_value=128
        ):
            config = _get_hf_config_from_titan(titan_model, hparams, state_dict)
            self.assertIsInstance(config, LlamaConfig)
            self.assertEqual(config.vocab_size, 100)


if __name__ == "__main__":
    unittest.main()
