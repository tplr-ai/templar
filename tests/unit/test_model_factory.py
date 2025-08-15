import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tplr.model_factory import (
    create_job_config,
    create_parallel_dims,
    create_titan_model_args,
    initialize_torchtitan_model,
)


class TestModelFactory(unittest.TestCase):
    def setUp(self):
        self.hparams = SimpleNamespace(
            sequence_length=1024,
            model_config=SimpleNamespace(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                vocab_size=50257,
            ),
            torchtitan=SimpleNamespace(),
        )

    def test_create_titan_model_args(self):
        args = create_titan_model_args(
            self.hparams.model_config, self.hparams.sequence_length
        )
        self.assertEqual(args.dim, 768)
        self.assertEqual(args.n_layers, 12)

    def test_create_job_config(self):
        config = create_job_config(self.hparams)
        self.assertIsNotNone(config)

    def test_create_parallel_dims(self):
        dims = create_parallel_dims(1, self.hparams)
        self.assertEqual(dims.world_size, 1)

    @patch("tplr.model_factory.TitanLlama")
    @patch("tplr.model_factory.parallelize_llama")
    def test_initialize_torchtitan_model(self, mock_parallelize, mock_llama):
        mock_model = MagicMock()
        mock_parallelize.return_value = mock_model
        model = initialize_torchtitan_model(self.hparams)
        self.assertIsNotNone(model)
        mock_parallelize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
