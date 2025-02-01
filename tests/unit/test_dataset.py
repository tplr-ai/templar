"""Unit tests for dataset functionality"""
import pytest
import torch
import numpy as np
from types import SimpleNamespace
from ..utils.assertions import assert_tensor_equal
from ..utils.env_setup import setup_test_environment

# Setup environment before imports
setup_test_environment()

from tplr.dataset import DatasetLoader
from tplr.r2_dataset import R2DatasetLoader
from tplr.hparams import load_hparams

# Mark all tests as async
pytestmark = pytest.mark.asyncio

class TestDatasetBasics:
    """Test basic dataset functionality"""
    
    @pytest.fixture
    def tokenizer(self):
        """Get tokenizer from hparams"""
        hparams = load_hparams()
        return hparams.tokenizer

    def set_random_seeds(self, seed=42):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    async def test_basic_loading(self, tokenizer):
        """Test basic dataset loading"""
        # Get some pages
        pages = await DatasetLoader.next_pages(
            offset=0,
            n_pages=1,
            seed=42
        )
        
        # Create loader
        loader = await DatasetLoader.create(
            batch_size=2,
            sequence_length=128,
            pages_info=pages,
            tokenizer=tokenizer,
            pack_samples=False
        )
        
        # Get first batch
        batch = next(iter(loader))
        
        # Verify basic properties
        assert batch is not None
        assert batch.shape == (2, 128)
        assert torch.is_tensor(batch)

class TestDatasetEquivalence:
    """Test equivalence between DatasetLoader and R2DatasetLoader"""
    
    @pytest.fixture
    def tokenizer(self):
        """Get tokenizer from hparams"""
        hparams = load_hparams()
        return hparams.tokenizer

    def set_random_seeds(self, seed=42):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    async def test_page_generation(self):
        """Test that both loaders generate identical pages"""
        # Test parameters
        offset = 0
        n_pages = 2
        seed = 255

        # Generate pages using both methods
        self.set_random_seeds()
        r2_pages = await R2DatasetLoader.next_pages(
            offset=offset,
            n_pages=n_pages,
            seed=seed
        )

        self.set_random_seeds()
        hf_pages = await DatasetLoader.next_pages(
            offset=offset,
            n_pages=n_pages,
            seed=seed
        )

        # Verify pages are identical
        assert r2_pages == hf_pages, (
            f"Page generation differs:\nR2: {r2_pages}\nHF: {hf_pages}"
        )

    async def test_batch_equivalence(self, tokenizer):
        """Test that both loaders generate equivalent batches"""
        # Test parameters
        batch_size = 2
        sequence_length = 128
        n_pages = 2
        seed = 255
        offset = 0

        # Generate pages
        self.set_random_seeds()
        pages = await R2DatasetLoader.next_pages(
            offset=offset,
            n_pages=n_pages,
            seed=seed
        )

        # Create both loaders
        self.set_random_seeds()
        r2_loader = await R2DatasetLoader.create(
            batch_size=batch_size,
            sequence_length=sequence_length,
            pages_info=pages,
            tokenizer=tokenizer,
            pack_samples=False
        )

        self.set_random_seeds()
        hf_loader = await DatasetLoader.create(
            batch_size=batch_size,
            sequence_length=sequence_length,
            pages_info=pages,
            tokenizer=tokenizer,
            pack_samples=False
        )

        # Compare batches
        r2_batches = list(r2_loader)
        hf_batches = list(hf_loader)

        assert len(r2_batches) == len(hf_batches), (
            f"Different batch counts: R2={len(r2_batches)}, HF={len(hf_batches)}"
        )

        for batch_idx, (r2_batch, hf_batch) in enumerate(zip(r2_batches, hf_batches)):
            # Convert to tensors if needed
            r2_tensor = torch.tensor(r2_batch) if not isinstance(r2_batch, torch.Tensor) else r2_batch
            hf_tensor = torch.tensor(hf_batch) if not isinstance(hf_batch, torch.Tensor) else hf_batch

            # Verify shapes match
            assert r2_tensor.shape == hf_tensor.shape, (
                f"Batch {batch_idx} shapes differ: R2={r2_tensor.shape}, HF={hf_tensor.shape}"
            )

            # Verify content matches
            assert_tensor_equal(r2_tensor, hf_tensor, f"Batch {batch_idx} contents differ")

    async def test_tokenization_equivalence(self, tokenizer):
        """Test that both loaders tokenize text identically"""
        # Get some pages
        pages = await DatasetLoader.next_pages(
            offset=0,
            n_pages=1,
            seed=42
        )
        
        # Create both loaders
        r2_loader = await R2DatasetLoader.create(
            batch_size=1,
            sequence_length=64,  # Shorter for easier comparison
            pages_info=pages,
            tokenizer=tokenizer,
            pack_samples=False
        )
        
        hf_loader = await DatasetLoader.create(
            batch_size=1,
            sequence_length=64,
            pages_info=pages,
            tokenizer=tokenizer,
            pack_samples=False
        )
        
        # Get first batch from each
        r2_batch = next(iter(r2_loader))
        hf_batch = next(iter(hf_loader))
        
        # Decode and compare texts
        r2_text = tokenizer.decode(r2_batch[0])
        hf_text = tokenizer.decode(hf_batch[0])
        
        # Compare first 100 chars to keep output manageable
        assert r2_text[:100] == hf_text[:100], (
            f"Tokenization differs:\nR2: {r2_text[:100]}\nHF: {hf_text[:100]}"
        )

class TestR2DatasetLoader:
    """Test R2DatasetLoader specific functionality"""

    @pytest.fixture
    async def dataset_config(self):
        """Get dataset configuration"""
        return await R2DatasetLoader.fetch_dataset_configs()

    @pytest.fixture
    def tokenizer(self):
        """Get tokenizer from hparams"""
        hparams = load_hparams()
        return hparams.tokenizer

    async def test_large_page_offset(self, dataset_config, tokenizer):
        """Test handling of large page offsets"""
        # Find config with most rows
        max_rows_config = max(dataset_config.items(), key=lambda x: x[1]["num_rows"])
        config_name = max_rows_config[0]
        num_rows = max_rows_config[1]["num_rows"]

        # Test different offsets
        test_cases = [
            (0, "start of dataset"),
            (num_rows // 2, "middle of dataset"),
            (num_rows - 200, "near end of dataset")
        ]

        for offset, description in test_cases:
            # Create single-page test
            pages = [(config_name, offset, "train")]

            # Create loader
            loader = await R2DatasetLoader.create(
                batch_size=2,
                sequence_length=128,
                pages_info=pages,
                tokenizer=tokenizer,
                pack_samples=False
            )

            # Get first batch
            batch = next(iter(loader))

            # Verify batch
            assert batch is not None
            assert batch.shape == (2, 128)

            # Verify tokens are valid
            for sequence in batch:
                valid_tokens = sequence[sequence != tokenizer.pad_token_id]
                assert len(valid_tokens) > 0
                
                # Verify decoded text
                text = tokenizer.decode(valid_tokens)
                assert len(text.strip()) > 0

    async def test_row_group_boundaries(self, dataset_config, tokenizer):
        """Test handling of row group boundaries"""
        # Get first config
        config_name = next(iter(dataset_config))
        config = dataset_config[config_name]
        
        # Calculate row group size
        row_group_size = config["num_rows"] // config["num_row_groups"]
        
        # Test offsets around row group boundaries
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

            # Verify we can get data
            batch = next(iter(loader))
            assert batch is not None
            assert batch.shape == (2, 128) 