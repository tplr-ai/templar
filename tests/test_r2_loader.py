# ruff: noqa
import os
from pathlib import Path
from dotenv import load_dotenv

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
    max_rows_config = max(configs_data.items(), key=lambda x: x[1]['num_rows'])
    config_name = max_rows_config[0]
    num_rows = max_rows_config[1]['num_rows']
    
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
                
            logger.info(f"[green]Successfully processed batch for offset {offset}[/green]")
            
        except Exception as e:
            logger.error(f"[red]Error processing offset {offset}: {str(e)}[/red]", exc_info=True)
            raise

    logger.info(
        f"[green]All offset tests completed successfully ({T() - start_time:.2f}s)[/green]"
    )
