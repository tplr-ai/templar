#ruff : noqa
"""Unit tests for dataset functionality"""
import pytest
import torch
import numpy as np
from ..utils.assertions import assert_tensor_equal
from ..utils.env_setup import setup_test_environment
from tplr.logging import logger, debug, T  # already imported if not, add it here
from tplr.r2_dataset import R2DatasetLoader
from tplr.hparams import load_hparams
import os

# Setup environment before imports
setup_test_environment()

from tplr.dataset import DatasetLoader

# Mark all tests as async
pytestmark = pytest.mark.asyncio


class TestR2LoaderIntegration:
    """Integration tests for R2DatasetLoader functionality (moved from tests/test_r2_loader.py)"""
    
    @pytest.mark.asyncio
    async def test_local_parquet_loader(self):
        """
        Integration test to ensure R2DatasetLoader can fetch pages from your R2 parquet data.
        Adjust environment variables & parameters as needed.
        """
        start_time = T()
        logger.info("Starting test_local_parquet_loader")
    
        # Check for required R2 dataset environment variables.
        required_vars = [
            "R2_DATASET_ACCOUNT_ID",
            "R2_DATASET_BUCKET_NAME",
            "R2_DATASET_READ_ACCESS_KEY_ID",
            "R2_DATASET_READ_SECRET_ACCESS_KEY",
            "R2_DATASET_WRITE_ACCESS_KEY_ID",
            "R2_DATASET_WRITE_SECRET_ACCESS_KEY",
        ]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        if missing_vars:
            pytest.skip(f"Missing environment variables: {', '.join(missing_vars)}")
    
        hparams = load_hparams()
        tokenizer = hparams.tokenizer
        logger.info(f"Tokenizer loaded ({T() - start_time:.2f}s)")
    
        # Prepare test parameters
        offset = 0
        n_pages = 2  # Number of pages to fetch
        seed = "my-test-seed"
        batch_size = 2
        sequence_length = 128
    
        # 1. Generate random pages
        pages = await R2DatasetLoader.next_pages(offset=offset, n_pages=n_pages, seed=seed)
        logger.info(f"Random pages selected: {pages} ({T() - start_time:.2f}s)")
    
        # 2. Create loader
        loader = await R2DatasetLoader.create(
            batch_size=batch_size,
            sequence_length=sequence_length,
            pages_info=pages,
            tokenizer=tokenizer,
            pack_samples=False,
        )
        logger.info(f"Loader created ({T() - start_time:.2f}s)")
    
        # 3. Iterate over a few batches
        batch_count = 0
        for batch in loader:
            logger.info(f"Batch {batch_count} shape: {batch.shape}")
            for i, sequence in enumerate(batch):
                tokens = sequence[sequence != tokenizer.pad_token_id].tolist()
                text = tokenizer.decode(tokens)
                logger.info(f"Sequence {i}: {text[:200]}...")
            batch_count += 1
            if batch_count >= 2:
                break
    
        assert batch_count > 0, "No batches were produced by the R2DatasetLoader"
        logger.info(f"Test completed successfully. Processed {batch_count} batches ({T() - start_time:.2f}s)")
        
    @pytest.mark.asyncio
    async def test_large_page_offset_handling(self):
        """
        Test that the loader correctly handles large page offsets which might exceed row group bounds.
        """
        start_time = T()
        logger.info("Starting test_large_page_offset_handling")
    
        hparams = load_hparams()
        tokenizer = hparams.tokenizer
    
        # Get dataset configs to choose the one with most rows
        configs_data = await R2DatasetLoader.fetch_dataset_configs()
        max_rows_config = max(configs_data.items(), key=lambda x: x[1]["num_rows"])
        config_name = max_rows_config[0]
        num_rows = max_rows_config[1]["num_rows"]
    
        test_cases = [
            (0, "start of dataset"),
            (num_rows // 2, "middle of dataset"),
            (num_rows - 200, "near end of dataset"),
        ]
    
        for offset, description in test_cases:
            logger.info(f"Testing {description} (offset: {offset})")
            pages = [(config_name, offset, "train")]
    
            loader = await R2DatasetLoader.create(
                batch_size=2,
                sequence_length=128,
                pages_info=pages,
                tokenizer=tokenizer,
                pack_samples=False,
            )
            batch = next(iter(loader))
            assert batch is not None, f"Failed to get batch for offset {offset}"
            assert batch.shape == (2, 128), f"Unexpected batch shape: {batch.shape}"
            for sequence in batch:
                valid_tokens = sequence[sequence != tokenizer.pad_token_id]
                assert len(valid_tokens) > 0, "Sequence contains no valid tokens"
                text = tokenizer.decode(valid_tokens)
                assert len(text.strip()) > 0, "Decoded text is empty"
            logger.info(f"Successfully processed batch for offset {offset}")
    
        logger.info(f"All offset tests completed successfully ({T() - start_time:.2f}s)")
    
    @pytest.mark.asyncio
    async def test_seed_consistency(self):
        """
        Test that R2DatasetLoader consistently returns the same pages for the same seed
        and different pages for different seeds.
        """
        start_time = T()
        logger.info("Starting test_seed_consistency")
    
        hparams = load_hparams()
        tokenizer = hparams.tokenizer
    
        offset = 1000
        n_pages = 2
        batch_size = 2
        sequence_length = 128
    
        seed1 = 42
        seed2 = 42
        seed3 = 43  # Different seed
    
        pages1 = await R2DatasetLoader.next_pages(offset=offset, n_pages=n_pages, seed=seed1)
        pages2 = await R2DatasetLoader.next_pages(offset=offset, n_pages=n_pages, seed=seed2)
        pages3 = await R2DatasetLoader.next_pages(offset=offset, n_pages=n_pages, seed=seed3)
    
        assert pages1 == pages2, "Same seed should produce identical pages"
        assert pages1 != pages3, "Different seeds should produce different pages"
    
        loader1 = await R2DatasetLoader.create(
            batch_size=batch_size,
            sequence_length=sequence_length,
            pages_info=pages1,
            tokenizer=tokenizer,
            pack_samples=False,
        )
        loader2 = await R2DatasetLoader.create(
            batch_size=batch_size,
            sequence_length=sequence_length,
            pages_info=pages2,
            tokenizer=tokenizer,
            pack_samples=False,
        )
        batch1 = torch.tensor(next(iter(loader1)))
        batch2 = torch.tensor(next(iter(loader2)))
    
        assert torch.equal(batch1, batch2), "Same seed should produce identical batch content"
        logger.info(f"Seed consistency test completed successfully ({T() - start_time:.2f}s)")
        
# TODO: After confirming these tests run correctly, consider removing the old tests/test_r2_loader.py file. 