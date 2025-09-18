import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

from tplr.wandb_ops import initialize_wandb


class TestWandb(unittest.TestCase):
    @patch("tplr.wandb_ops.wandb")
    @patch("tplr.wandb_ops.os.path.exists", return_value=False)
    @patch("tplr.wandb_ops.open", new_callable=mock_open)
    def test_initialize_wandb_new_run(self, mock_open, mock_exists, mock_wandb):
        mock_run = MagicMock()
        mock_run.config = {}
        mock_wandb.init.return_value = mock_run

        config = MagicMock()
        config.project = "test_project"
        config.log_to_private_wandb = False

        run = initialize_wandb(
            "test_prefix", "test_uid", config, "test_group", "test_job"
        )

        self.assertIsNotNone(run)
        mock_wandb.init.assert_called_once()
        self.assertIn("version_history", run.config)


if __name__ == "__main__":
    unittest.main()
