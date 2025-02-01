"""Unit tests for R2 dataset loader functionality"""
import pytest
import os
from pathlib import Path
from ..utils.env_setup import setup_test_environment
from ..utils.assertions import assert_tensor_equal

# Setup environment before imports
setup_test_environment()

from tplr.r2_dataset import R2DatasetLoader
from tplr.hparams import load_hparams

# Mark all tests as async
pytestmark = pytest.mark.asyncio

class TestR2LoaderBasics:
    """Test basic R2 loader functionality"""
    
    @pytest.fixture
    async def dataset_config(self):
        """Get dataset configuration"""
        return await R2DatasetLoader.fetch_dataset_configs()

    @pytest.fixture
    def tokenizer(self):
        """Get tokenizer from hparams"""
        hparams = load_hparams()
        return hparams.tokenizer

    async def test_config_loading(self, dataset_config):
        """Test dataset configuration loading"""
        assert dataset_config is not None
        assert len(dataset_config) > 0
        
        # Verify config structure
        for config_name, config in dataset_config.items():
            assert "num_rows" in config
            assert "num_row_groups" in config
            assert config["num_rows"] > 0
            assert config["num_row_groups"] > 0

    async def test_basic_loader_creation(self, dataset_config, tokenizer):
        """Test basic loader instantiation"""
        # Setup simple test page
        config_name = next(iter(dataset_config))
        pages = [(config_name, 0, "train")]
        
        loader = await R2DatasetLoader.create(
            batch_size=2,
            sequence_length=128,
            pages_info=pages,
            tokenizer=tokenizer,
            pack_samples=False
        )
        
        # Verify loader properties
        assert loader is not None
        assert hasattr(loader, "__iter__")
        
        # Get first batch
        batch = next(iter(loader))
        assert batch is not None
        assert batch.shape == (2, 128)

class TestR2LoaderPaging:
    """Test page handling and navigation"""
    
    @pytest.fixture
    async def dataset_config(self):
        """Get dataset configuration"""
        return await R2DatasetLoader.fetch_dataset_configs()

    @pytest.fixture
    def tokenizer(self):
        """Get tokenizer from hparams"""
        hparams = load_hparams()
        return hparams.tokenizer

    async def test_page_generation(self, dataset_config):
        """Test page generation with different parameters"""
        test_cases = [
            (0, 1, "Basic single page"),
            (100, 2, "Multiple pages with offset"),
            (1000, 5, "Multiple pages with large offset")
        ]
        
        for offset, n_pages, description in test_cases:
            pages = await R2DatasetLoader.next_pages(
                offset=offset,
                n_pages=n_pages,
                seed="test-seed"
            )
            
            assert len(pages) == n_pages
            for page in pages:
                assert len(page) == 3  # (config_name, offset, split)
                assert page[1] >= offset
                assert page[2] == "train"

    async def test_large_page_offset(self, dataset_config, tokenizer):
        """Test handling of large page offsets"""
        # Find config with most rows
        max_rows_config = max(dataset_config.items(), key=lambda x: x[1]["num_rows"])
        config_name = max_rows_config[0]
        num_rows = max_rows_config[1]["num_rows"]
        
        test_offsets = [
            0,                  # Start
            num_rows // 2,      # Middle
            num_rows - 200      # Near end
        ]
        
        for offset in test_offsets:
            pages = [(config_name, offset, "train")]
            
            loader = await R2DatasetLoader.create(
                batch_size=2,
                sequence_length=128,
                pages_info=pages,
                tokenizer=tokenizer,
                pack_samples=False
            )
            
            # Verify data loading
            batch = next(iter(loader))
            assert batch is not None
            assert batch.shape == (2, 128)
            
            # Verify token validity
            for sequence in batch:
                valid_tokens = sequence[sequence != tokenizer.pad_token_id]
                assert len(valid_tokens) > 0
                text = tokenizer.decode(valid_tokens)
                assert len(text.strip()) > 0

class TestR2LoaderEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    async def dataset_config(self):
        """Get dataset configuration"""
        return await R2DatasetLoader.fetch_dataset_configs()

    @pytest.fixture
    def tokenizer(self):
        """Get tokenizer from hparams"""
        hparams = load_hparams()
        return hparams.tokenizer

    async def test_row_group_boundaries(self, dataset_config, tokenizer):
        """Test handling of row group boundaries"""
        config_name = next(iter(dataset_config))
        config = dataset_config[config_name]
        row_group_size = config["num_rows"] // config["num_row_groups"]
        
        test_offsets = [
            row_group_size - 1,  # End of first group
            row_group_size,      # Start of second group
            row_group_size + 1   # Just into second group
        ]
        
        for offset in test_offsets:
            pages = [(config_name, offset, "train")]
            
            loader = await R2DatasetLoader.create(
                batch_size=2,
                sequence_length=128,
                pages_info=pages,
                tokenizer=tokenizer,
                pack_samples=False
            )
            
            batch = next(iter(loader))
            assert batch is not None
            assert batch.shape == (2, 128)

    async def test_invalid_configs(self, dataset_config, tokenizer):
        """Test handling of invalid configurations"""
        with pytest.raises(Exception):
            await R2DatasetLoader.create(
                batch_size=0,  # Invalid batch size
                sequence_length=128,
                pages_info=[("invalid_config", 0, "train")],
                tokenizer=tokenizer,
                pack_samples=False
            ) 

class TestR2LoaderPerformance:
    """Test performance characteristics and optimizations"""
    
    @pytest.fixture
    async def dataset_config(self):
        """Get dataset configuration"""
        return await R2DatasetLoader.fetch_dataset_configs()

    @pytest.fixture
    def tokenizer(self):
        """Get tokenizer from hparams"""
        hparams = load_hparams()
        return hparams.tokenizer

    async def test_batch_iteration_speed(self, dataset_config, tokenizer):
        """Test speed of batch iteration"""
        import time
        
        # Setup test parameters
        config_name = next(iter(dataset_config))
        pages = [(config_name, 0, "train")]
        batch_size = 8
        sequence_length = 512
        num_batches = 10
        
        # Create loader
        loader = await R2DatasetLoader.create(
            batch_size=batch_size,
            sequence_length=sequence_length,
            pages_info=pages,
            tokenizer=tokenizer,
            pack_samples=False
        )
        
        # Time batch iteration
        start_time = time.perf_counter()
        batches = []
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            batches.append(batch)
        end_time = time.perf_counter()
        
        # Verify performance
        duration = end_time - start_time
        batches_per_second = num_batches / duration
        
        # Log performance metrics
        print(f"\nProcessed {num_batches} batches in {duration:.2f}s")
        print(f"Batches per second: {batches_per_second:.2f}")
        
        # Basic performance assertion - adjust threshold as needed
        assert batches_per_second > 1.0, "Batch processing too slow"

    async def test_memory_usage(self, dataset_config, tokenizer):
        """Test memory usage during loading"""
        import psutil
        import gc
        
        process = psutil.Process()
        gc.collect()  # Initial cleanup
        
        # Record starting memory
        start_mem = process.memory_info().rss
        
        # Create and use loader
        config_name = next(iter(dataset_config))
        pages = [(config_name, 0, "train")]
        loader = await R2DatasetLoader.create(
            batch_size=16,
            sequence_length=1024,  # Larger sequences
            pages_info=pages,
            tokenizer=tokenizer,
            pack_samples=False
        )
        
        # Process some batches
        batches = []
        for i, batch in enumerate(loader):
            if i >= 5:
                break
            batches.append(batch)
        
        # Force cleanup
        del loader
        gc.collect()
        end_mem = process.memory_info().rss
        
        # Check memory growth
        mem_growth = end_mem - start_mem
        print(f"\nMemory growth: {mem_growth / 1024 / 1024:.2f}MB")
        
        # Allow some memory growth but not excessive
        assert mem_growth < 500 * 1024 * 1024  # 500MB limit

class TestR2LoaderFeatures:
    """Test specific loader features and options"""
    
    @pytest.fixture
    async def dataset_config(self):
        """Get dataset configuration"""
        return await R2DatasetLoader.fetch_dataset_configs()

    @pytest.fixture
    def tokenizer(self):
        """Get tokenizer from hparams"""
        hparams = load_hparams()
        return hparams.tokenizer

    async def test_packed_samples(self, dataset_config, tokenizer):
        """Test sample packing functionality"""
        config_name = next(iter(dataset_config))
        pages = [(config_name, 0, "train")]
        
        # Create loader with packed samples
        loader = await R2DatasetLoader.create(
            batch_size=2,
            sequence_length=128,
            pages_info=pages,
            tokenizer=tokenizer,
            pack_samples=True
        )
        
        batch = next(iter(loader))
        
        # Verify packed batch properties
        assert batch is not None
        assert batch.shape == (2, 128)
        
        # Check for efficient packing
        for sequence in batch:
            pad_tokens = (sequence == tokenizer.pad_token_id).sum()
            assert pad_tokens < sequence.shape[0] * 0.1  # Less than 10% padding

    async def test_different_sequence_lengths(self, dataset_config, tokenizer):
        """Test handling of different sequence lengths"""
        config_name = next(iter(dataset_config))
        pages = [(config_name, 0, "train")]
        
        test_lengths = [32, 64, 128, 256, 512]
        
        for seq_length in test_lengths:
            loader = await R2DatasetLoader.create(
                batch_size=2,
                sequence_length=seq_length,
                pages_info=pages,
                tokenizer=tokenizer,
                pack_samples=False
            )
            
            batch = next(iter(loader))
            assert batch.shape == (2, seq_length)
            
            # Verify content
            for sequence in batch:
                valid_tokens = sequence[sequence != tokenizer.pad_token_id]
                assert len(valid_tokens) > 0
                text = tokenizer.decode(valid_tokens)
                assert len(text.strip()) > 0

    async def test_multi_page_iteration(self, dataset_config, tokenizer):
        """Test iteration across multiple pages"""
        # Get multiple pages
        pages = await R2DatasetLoader.next_pages(
            offset=0,
            n_pages=3,
            seed="test-seed"
        )
        
        loader = await R2DatasetLoader.create(
            batch_size=2,
            sequence_length=128,
            pages_info=pages,
            tokenizer=tokenizer,
            pack_samples=False
        )
        
        # Track unique texts to verify we're getting different content
        seen_texts = set()
        batch_count = 0
        
        for batch in loader:
            for sequence in batch:
                valid_tokens = sequence[sequence != tokenizer.pad_token_id]
                text = tokenizer.decode(valid_tokens)[:50]  # First 50 chars
                seen_texts.add(text)
            
            batch_count += 1
            if batch_count >= 10:
                break
        
        # Verify we got diverse content
        assert len(seen_texts) > 5, "Not enough unique content across pages" 