import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tplr.hparams import create_namespace, load_hparams


class TestHparams(unittest.TestCase):
    def test_create_namespace(self):
        with (
            patch("tplr.hparams.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch("tplr.hparams.LlamaConfig") as mock_config,
        ):
            hparams = {"tokenizer_name": "test/tokenizer"}
            namespace = create_namespace(hparams)
            self.assertIsInstance(namespace, SimpleNamespace)
            mock_tokenizer.assert_called_once_with(
                "test/tokenizer",
                verbose=False,
                clean_up_tokenization_spaces=True,
            )
            mock_config.assert_called_once()

    def test_load_hparams(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hparams_dir = Path(tmpdir)
            with open(hparams_dir / "hparams.json", "w") as f:
                json.dump({"model_size": "small"}, f)
            with open(hparams_dir / "small.json", "w") as f:
                json.dump({"hidden_size": 128}, f)

            with (
                patch("tplr.hparams.AutoTokenizer.from_pretrained"),
                patch("tplr.hparams.LlamaConfig"),
            ):
                namespace = load_hparams(hparams_dir=str(hparams_dir))
                self.assertEqual(namespace.model_size, "small")
                self.assertEqual(namespace.hidden_size, 128)


if __name__ == "__main__":
    unittest.main()
