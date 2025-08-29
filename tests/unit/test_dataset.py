import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from tplr.dataset import DatasetLoader, SubsetLoader


class TestSubsetLoader(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.eos_token_id = 0
        self.loader = SubsetLoader(
            batch_size=2,
            sequence_length=10,
            num_pages=1,
            tokenizer=self.mock_tokenizer,
        )

    def test_init(self):
        self.assertEqual(self.loader.batch_size, 2)
        self.assertEqual(self.loader.sequence_length, 10)
        self.assertEqual(self.loader.num_pages, 1)
        self.assertEqual(self.loader.tokenizer, self.mock_tokenizer)

    def test_get_pad_size(self):
        self.assertEqual(self.loader._get_pad_size([1, 2, 3]), 7)
        self.loader.pack_samples = True
        self.assertEqual(self.loader._get_pad_size([1, 2, 3]), 1)

    def test_refill_padded_buffer(self):
        self.loader.buffer = [1, 2, 3, 0, 4, 5, 6, 0]
        self.loader._refill_padded_buffer()
        self.assertEqual(len(self.loader.padded_buffer), 10)

    def test_iter_next(self):
        self.loader.buffer = [1, 2, 3, 0, 4, 5, 6, 0] * 5
        self.loader.padded_buffer = []
        iterator = iter(self.loader)
        batch = next(iterator)
        self.assertEqual(batch.shape, (2, 10))


class TestDatasetLoader:
    @pytest.mark.asyncio
    async def test_create(self):
        with (
            patch(
                "tplr.dataset.DatasetLoader.fetch_dataset_configs",
                new_callable=AsyncMock,
            ) as mock_fetch_configs,
            patch(
                "tplr.dataset.DatasetLoader._fetch_data_to_buffer",
                new_callable=AsyncMock,
            ) as mock_fetch_data,
        ):
            mock_fetch_configs.return_value = {
                "config1": {"num_rows": 200, "split": "train"}
            }
            mock_tokenizer = MagicMock()
            mock_tokenizer.eos_token_id = 0
            loader = await DatasetLoader.create(
                batch_size=2,
                sequence_length=10,
                num_pages=1,
                tokenizer=mock_tokenizer,
            )
            assert loader is not None

    @pytest.mark.asyncio
    async def test_fetch_data_for_pages(self):
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "rows": [{"row": {"text": "hello world"}}]
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value.__aenter__.return_value = mock_response

            mock_tokenizer = MagicMock()
            mock_tokenizer.eos_token_id = 0
            mock_tokenizer.encode.return_value = [1, 2, 3]
            loader = SubsetLoader(
                batch_size=2,
                sequence_length=10,
                num_pages=1,
                tokenizer=mock_tokenizer,
            )
            loader.rows_base_url = "http://test.com"
            with patch.object(loader, "_fetch_data_for_page", new_callable=AsyncMock):
                await loader.fetch_data_for_pages([("c1", 1, "train")])
                assert loader._fetch_data_for_page.call_count == 1


if __name__ == "__main__":
    unittest.main()
