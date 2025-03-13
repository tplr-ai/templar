# ruff: noqa
import os
from pathlib import Path
from dotenv import load_dotenv
import io
import asyncio
import pyarrow as pa
import pyarrow.parquet as pq

# Find and load the correct .env file
env_path = Path(__file__).parent.parent / ".env"
if not env_path.exists():
    raise FileNotFoundError(f"Required .env file not found at {env_path}")

# Load environment variables before any other imports
load_dotenv(env_path, override=True)

# Verify environment variables are loaded
required_vars = [
    "R2_GRADIENTS_ACCOUNT_ID",
    "R2_GRADIENTS_BUCKET_NAME",
    "R2_GRADIENTS_READ_ACCESS_KEY_ID",
    "R2_GRADIENTS_READ_SECRET_ACCESS_KEY",
    "R2_GRADIENTS_WRITE_ACCESS_KEY_ID",
    "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY",
    "R2_DATASET_ACCOUNT_ID",
    "R2_DATASET_BUCKET_NAME",
    "R2_DATASET_READ_ACCESS_KEY_ID",
    "R2_DATASET_READ_SECRET_ACCESS_KEY",
    "R2_DATASET_WRITE_ACCESS_KEY_ID",
    "R2_DATASET_WRITE_SECRET_ACCESS_KEY",
]

missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables in .env file: {', '.join(missing_vars)}"
    )

# Only import after environment variables are loaded and verified
import pytest
from transformers import AutoTokenizer
from tplr.logging import logger, debug, T
from tplr.r2_dataset import R2DatasetLoader
from tplr.hparams import load_hparams
import torch
import random


# Enable debug logging for tests
debug()


@pytest.mark.asyncio
async def test_local_parquet_loader():
    """
    Simple integration test to ensure R2DatasetLoader can fetch pages from your R2 parquet data.
    Adjust environment variables & the code below to point to your actual dataset, then run:
        pytest tests/test_local_parquet_loader.py
    """

    start_time = T()
    logger.info("Starting test_local_parquet_loader")

    # Make sure the required R2 environment variables are set
    missing_vars = []
    for var in [
        "R2_DATASET_ACCOUNT_ID",
        "R2_DATASET_BUCKET_NAME",
        "R2_DATASET_READ_ACCESS_KEY_ID",
        "R2_DATASET_READ_SECRET_ACCESS_KEY",
        "R2_DATASET_WRITE_ACCESS_KEY_ID",
        "R2_DATASET_WRITE_SECRET_ACCESS_KEY",
    ]:
        if not os.environ.get(var):
            missing_vars.append(var)

    if missing_vars:
        pytest.skip(f"Missing environment variables: {', '.join(missing_vars)}")

    # Instantiate a tokenizer
    hparams = load_hparams()
    tokenizer = hparams.tokenizer
    logger.info(f"Tokenizer loaded ({T() - start_time:.2f}s)")

    # Prepare test parameters
    offset = 0
    n_pages = 2  # The number of random pages to fetch
    seed = "my-test-seed"  # Arbitrary seed for reproducibility
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

    # 3. Iterate over the loader a few batches
    batch_count = 0
    try:
        for batch in loader:
            logger.info(f"[cyan]Batch {batch_count} shape: {batch.shape}[/cyan]")

            # Decode each sequence in the batch
            for i, sequence in enumerate(batch):
                # Convert to tokens, skip padding tokens
                tokens = sequence[sequence != tokenizer.pad_token_id].tolist()
                text = tokenizer.decode(tokens)
                logger.info(f"Sequence {i}:")
                logger.info(f"First 50 tokens: {tokens[:50]}...")
                logger.info(f"Text: {text[:200]}...")
                logger.info("[dim]" + "-" * 80 + "[/dim]")

            batch_count += 1
            if batch_count >= 2:  # Look at first 2 batches
                break
    except Exception as e:
        logger.error(f"[red]Error during iteration: {str(e)}[/red]", exc_info=True)

    # Basic assertion: We expect at least 1 batch if pages > 0
    assert batch_count > 0, "No batches were produced by the R2DatasetLoader"
    logger.info(
        f"[green]Test completed successfully. Processed {batch_count} batches ({T() - start_time:.2f}s)[/green]"
    )


@pytest.mark.asyncio
async def test_large_page_offset_handling():
    """
    Test that the loader correctly handles large page offsets that might exceed row group bounds.
    This specifically tests the fix for the row group index calculation.
    """
    start_time = T()
    logger.info("Starting test_large_page_offset_handling")

    # Load tokenizer
    hparams = load_hparams()
    tokenizer = hparams.tokenizer

    # Get dataset configs to find maximum rows
    configs_data = await R2DatasetLoader.fetch_dataset_configs()

    # Find a config with the most rows to test boundary conditions
    max_rows_config = max(configs_data.items(), key=lambda x: x[1]["num_rows"])
    config_name = max_rows_config[0]
    num_rows = max_rows_config[1]["num_rows"]

    # Test cases with different offsets
    test_cases = [
        (0, "start of dataset"),
        (num_rows // 2, "middle of dataset"),
        (num_rows - 200, "near end of dataset"),  # Leave room for page size
    ]

    for offset, description in test_cases:
        logger.info(f"\nTesting {description} (offset: {offset})")

        # Create a single-page test with specific offset
        pages = [(config_name, offset, "train")]

        try:
            # Create loader with test page
            loader = await R2DatasetLoader.create(
                batch_size=2,
                sequence_length=128,
                pages_info=pages,
                tokenizer=tokenizer,
                pack_samples=False,
            )

            # Verify we can get at least one batch
            batch = next(iter(loader))

            # Basic validation
            assert batch is not None, f"Failed to get batch for offset {offset}"
            assert batch.shape == (2, 128), f"Unexpected batch shape: {batch.shape}"

            # Verify the batch contains valid token IDs
            for sequence in batch:
                valid_tokens = sequence[sequence != tokenizer.pad_token_id]
                assert len(valid_tokens) > 0, "Sequence contains no valid tokens"

                # Decode to verify we got meaningful text
                text = tokenizer.decode(valid_tokens)
                assert len(text.strip()) > 0, "Decoded text is empty"

            logger.info(
                f"[green]Successfully processed batch for offset {offset}[/green]"
            )

        except Exception as e:
            logger.error(
                f"[red]Error processing offset {offset}: {str(e)}[/red]", exc_info=True
            )
            raise

    logger.info(
        f"[green]All offset tests completed successfully ({T() - start_time:.2f}s)[/green]"
    )


@pytest.mark.asyncio
async def test_seed_consistency():
    """
    Test that R2DatasetLoader consistently returns the same pages for the same seed
    and different pages for different seeds.
    """
    start_time = T()
    logger.info("Starting test_seed_consistency")

    # Load tokenizer
    hparams = load_hparams()
    tokenizer = hparams.tokenizer

    # Test parameters
    offset = 1000  # Arbitrary offset
    n_pages = 2
    batch_size = 2
    sequence_length = 128

    # Test same seed returns same pages
    seed1 = 42
    seed2 = 42
    seed3 = 43  # Different seed

    # Get pages with same seed
    pages1 = await R2DatasetLoader.next_pages(
        offset=offset, n_pages=n_pages, seed=seed1
    )
    pages2 = await R2DatasetLoader.next_pages(
        offset=offset, n_pages=n_pages, seed=seed2
    )

    # Get pages with different seed
    pages3 = await R2DatasetLoader.next_pages(
        offset=offset, n_pages=n_pages, seed=seed3
    )

    # Test same seed produces same pages
    assert pages1 == pages2, "Same seed should produce identical pages"

    # Test different seeds produce different pages
    assert pages1 != pages3, "Different seeds should produce different pages"

    # Test page content consistency
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

    # Get first batch from each loader and convert to tensors
    batch1 = torch.tensor(next(iter(loader1)))
    batch2 = torch.tensor(next(iter(loader2)))

    # Test content consistency
    assert torch.equal(batch1, batch2), (
        "Same seed should produce identical batch content"
    )

    # Test seed range
    seeds = [random.randint(0, 10000) for _ in range(10)]
    unique_pages = set()

    for seed in seeds:
        pages = await R2DatasetLoader.next_pages(
            offset=offset, n_pages=n_pages, seed=seed
        )
        page_tuple = tuple(
            [(p[0], p[1]) for p in pages]
        )  # Convert to tuple for hashing
        unique_pages.add(page_tuple)

    # Check if we got different pages for different seeds
    assert len(unique_pages) > 1, "Random seeds should produce variety of pages"

    logger.success(
        f"[green]Seed consistency test completed successfully ({T() - start_time:.2f}s)[/green]"
    )


# --- Test: Transient failures recover (retry succeeds) ---
@pytest.mark.asyncio
async def test_retry_mechanism_success(monkeypatch):
    """
    Ensure that transient errors during fs.open are retried and eventually succeed.
    Edge case: The dummy filesystem fails a fixed number of times (2), then returns valid data.
    """
    # Create a minimal parquet file in memory
    table = pa.table({"text": ["Testing retry", "More testing"]})
    parquet_buffer = io.BytesIO()
    pq.write_table(table, parquet_buffer)
    parquet_bytes = parquet_buffer.getvalue()

    # Define a buffer factory that returns a new BytesIO each time
    def buffer_factory():
        return io.BytesIO(parquet_bytes)

    # DummyFS simulates transient failures before finally succeeding.
    class DummyFS:
        def __init__(self, real_fs, fail_times=2, buffer_factory=None):
            self.real_fs = real_fs
            self.fail_times = fail_times
            self.call_count = 0
            self.buffer_factory = buffer_factory

        def open(self, *args, **kwargs):
            if self.call_count < self.fail_times:
                self.call_count += 1
                raise Exception("Simulated transient error")
            return self.buffer_factory()

        def __getattr__(self, attr):
            return getattr(self.real_fs, attr)

    # Patch R2DatasetLoader._get_fs to return our DummyFS; note the lambda accepts self
    real_fs = R2DatasetLoader._get_fs()
    dummy_fs = DummyFS(real_fs, fail_times=2, buffer_factory=buffer_factory)
    monkeypatch.setattr(R2DatasetLoader, "_get_fs", lambda self: dummy_fs)

    # Setup dummy metadata to bypass actual R2 calls.
    dummy_config = "dummy_config"
    dummy_shard = {"path": "dummy/path", "num_rows": 2}

    async def dummy_load_r2_metadata(self):
        return (
            {
                dummy_config: {
                    "total_rows": 2,
                    "split": "train",
                    "shards": [dummy_shard],
                }
            },
            {"configs": [{"config_name": dummy_config}]},
        )

    monkeypatch.setattr(R2DatasetLoader, "_load_r2_metadata", dummy_load_r2_metadata)

    # Initialize a basic tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a loader instance.
    loader = R2DatasetLoader(
        batch_size=1, sequence_length=10, tokenizer=tokenizer, pack_samples=False
    )
    loader.num_rows_per_page = 2

    # Trigger processing a page.
    tokens = await loader._process_page(
        (dummy_config, 0, "train"), asyncio.Semaphore(1)
    )
    assert isinstance(tokens, list), "Tokens should be a list"
    assert len(tokens) > 0, "Tokens list should not be empty"
    # The DummyFS should have failed exactly 2 times before succeeding.
    assert dummy_fs.call_count == 2, (
        "DummyFS did not simulate the expected number of transient errors"
    )


# --- Test: Persistent failure raises exception ---
@pytest.mark.asyncio
async def test_retry_mechanism_failure(monkeypatch):
    """
    Ensure that persistent errors in fs.open eventually raise an exception after max retries.
    Edge case: The dummy filesystem always fails.
    """

    # AlwaysFailFS simulates persistent errors by always raising an Exception.
    class AlwaysFailFS:
        def open(self, *args, **kwargs):
            raise Exception("Persistent transient error")

        def __getattr__(self, attr):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(R2DatasetLoader, "_get_fs", lambda self: AlwaysFailFS())

    # Setup dummy metadata (with "self" parameter).
    dummy_config = "dummy_config"
    dummy_shard = {"path": "dummy/path", "num_rows": 2}

    async def dummy_load_r2_metadata(self):
        return (
            {
                dummy_config: {
                    "total_rows": 2,
                    "split": "train",
                    "shards": [dummy_shard],
                }
            },
            {"configs": [{"config_name": dummy_config}]},
        )

    monkeypatch.setattr(R2DatasetLoader, "_load_r2_metadata", dummy_load_r2_metadata)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = R2DatasetLoader(
        batch_size=1, sequence_length=10, tokenizer=tokenizer, pack_samples=False
    )
    loader.num_rows_per_page = 2

    with pytest.raises(Exception, match="Persistent transient error"):
        await loader._process_page((dummy_config, 0, "train"), asyncio.Semaphore(1))
