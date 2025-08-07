import asyncio
import unittest
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from tplr.sharded_dataset import ShardedDatasetManager, SharedShardedDataset


@contextmanager
def mock_dataset_dependencies(token_array, id_array):
    with (
        patch(
            "tplr.sharded_dataset.SharedShardedDataset.locate_shards",
            return_value=("dummy_token_path", "dummy_id_path"),
        ),
        patch(
            "tplr.sharded_dataset.SharedShardedDataset.check_paths", return_value=None
        ),
        patch("tplr.sharded_dataset.np.memmap") as mock_memmap,
    ):

        def memmap_side_effect(path, dtype, mode):
            if path == "dummy_token_path":
                return token_array
            elif path == "dummy_id_path":
                return id_array
            return MagicMock()

        mock_memmap.side_effect = memmap_side_effect
        yield


class TestSharedShardedDataset(unittest.TestCase):
    def test_init(self):
        mock_tokens = np.arange(100, dtype=np.uint16)
        mock_ids = np.arange(10, dtype=np.uint64)

        with mock_dataset_dependencies(mock_tokens, mock_ids):
            dataset = SharedShardedDataset(0, 10, 0, 1)
            self.assertEqual(len(dataset), 10)

    def test_getitem(self):
        seq_len = 10
        total_samples = 10
        mock_tokens = np.arange(total_samples * seq_len, dtype=np.uint16)
        mock_ids = np.arange(total_samples, dtype=np.uint64)

        with mock_dataset_dependencies(mock_tokens, mock_ids):
            dataset = SharedShardedDataset(
                shard_index=0, sequence_length=seq_len, rank=0, world_size=1
            )

            idx = 5
            item = dataset[idx]
            expected_item = torch.from_numpy(
                mock_tokens[idx * seq_len : (idx + 1) * seq_len]
            )

            self.assertTrue(torch.equal(item, expected_item))
            self.assertEqual(item.shape, (seq_len,))


class TestShardedDatasetManager:
    @pytest.mark.asyncio
    async def test_prepare_shard_exists(self):
        with (
            patch(
                "tplr.sharded_dataset.SharedShardedDataset.locate_shards"
            ) as mock_locate,
            patch(
                "tplr.sharded_dataset.os.path.exists", return_value=True
            ) as mock_exists,
        ):
            mock_locate.return_value = ("a", "b")
            manager = ShardedDatasetManager(10, 0, 1, MagicMock())
            task = manager.prepare_shard(0)
            await task
            assert isinstance(task, asyncio.Task)

    @pytest.mark.asyncio
    async def test_prepare_shard_download(self):
        with (
            patch(
                "tplr.sharded_dataset.SharedShardedDataset.locate_shards"
            ) as mock_locate,
            patch(
                "tplr.sharded_dataset.os.path.exists", return_value=False
            ) as mock_exists,
            patch(
                "tplr.sharded_dataset.ShardedDatasetManager.download_files",
                new_callable=AsyncMock,
            ) as mock_download,
        ):
            mock_locate.return_value = ("a", "b")
            manager = ShardedDatasetManager(10, 0, 1, MagicMock())
            task = manager.prepare_shard(0)
            await task
            assert isinstance(task, asyncio.Task)


if __name__ == "__main__":
    unittest.main()
