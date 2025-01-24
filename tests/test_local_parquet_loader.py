import os
from dotenv import load_dotenv

# Load .env file before any other imports
load_dotenv(override=True)

import pytest
from transformers import AutoTokenizer
from tplr.local_parquet_dataset import LocalParquetDatasetLoader
from tplr.logging import logger, debug, T

# Enable debug logging for tests
debug()


@pytest.mark.asyncio
async def test_local_parquet_loader():
    """
    Simple integration test to ensure LocalParquetDatasetLoader can fetch pages from your R2 parquet data.
    Adjust environment variables & the code below to point to your actual dataset, then run:
        pytest tests/test_local_parquet_loader.py
    """

    start_time = T()
    logger.info("Starting test_local_parquet_loader")

    # Make sure the required R2 environment variables are set
    missing_vars = []
    for var in [
        "R2_ACCOUNT_ID",
        "R2_READ_ACCESS_KEY_ID",
        "R2_READ_SECRET_ACCESS_KEY",
        "R2_WRITE_ACCESS_KEY_ID",
        "R2_WRITE_SECRET_ACCESS_KEY",
    ]:
        if not os.environ.get(var):
            missing_vars.append(var)

    if missing_vars:
        pytest.skip(f"Missing environment variables: {', '.join(missing_vars)}")

    # Instantiate a tokenizer (replace with any HF checkpoint you like)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    logger.info(f"Tokenizer loaded ({T() - start_time:.2f}s)")

    # Prepare test parameters
    offset = 0
    n_pages = 2  # The number of random pages to fetch
    seed = "my-test-seed"  # Arbitrary seed for reproducibility
    batch_size = 2
    sequence_length = 128

    # 1. Generate random pages
    pages = await LocalParquetDatasetLoader.next_pages(
        offset=offset, n_pages=n_pages, seed=seed
    )
    logger.info(f"Random pages selected: {pages} ({T() - start_time:.2f}s)")

    # 2. Create loader
    loader = await LocalParquetDatasetLoader.create(
        batch_size=batch_size,
        sequence_length=sequence_length,
        pages_info=pages,
        tokenizer=tokenizer,
        pack_samples=False,  # Or True if you want to test sample packing
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
                logger.info(f"[green]Sequence {i}:[/green]")
                logger.info(f"[dim]First 50 tokens: {tokens[:50]}...[/dim]")
                logger.info(f"[yellow]Text: {text[:200]}...[/yellow]")
                logger.info("[dim]" + "-" * 80 + "[/dim]")

            batch_count += 1
            if batch_count >= 2:  # Look at first 2 batches
                break
    except Exception as e:
        logger.error(f"[red]Error during iteration: {str(e)}[/red]", exc_info=True)

    # Basic assertion: We expect at least 1 batch if pages > 0
    assert batch_count > 0, "No batches were produced by the LocalParquetDatasetLoader"
    logger.info(
        f"[green]Test completed successfully. Processed {batch_count} batches ({T() - start_time:.2f}s)[/green]"
    )
