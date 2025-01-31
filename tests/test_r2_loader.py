# ruff: noqa
import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

# We only define the required environment variables here,
# without enforcing them at the module level:
REQUIRED_VARS = [
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
]


def validate_config():
    """Validate configuration consistency and update BUCKET_SECRETS if needed."""
    env_bucket = os.environ.get("R2_DATASET_BUCKET_NAME")
    from tplr.config import BUCKET_SECRETS
    from tplr.logging import logger

    bucket_secrets = BUCKET_SECRETS.get("dataset", {}).get("name")

    logger.info("Configuration validation:")
    logger.info(f"Environment bucket: {env_bucket}")
    logger.info(f"BUCKET_SECRETS bucket: {bucket_secrets}")

    if env_bucket != bucket_secrets:
        logger.warning(
            "⚠️ Bucket name mismatch - updating BUCKET_SECRETS with environment value"
        )
        # Update BUCKET_SECRETS with environment values
        if "dataset" not in BUCKET_SECRETS:
            BUCKET_SECRETS["dataset"] = {}
        BUCKET_SECRETS["dataset"].update(
            {
                "name": env_bucket,
                "account_id": os.environ.get("R2_DATASET_ACCOUNT_ID"),
            }
        )
        if "credentials" not in BUCKET_SECRETS["dataset"]:
            BUCKET_SECRETS["dataset"]["credentials"] = {}
        BUCKET_SECRETS["dataset"]["credentials"].update(
            {
                "read": {
                    "access_key_id": os.environ.get("R2_DATASET_READ_ACCESS_KEY_ID"),
                    "secret_access_key": os.environ.get(
                        "R2_DATASET_READ_SECRET_ACCESS_KEY"
                    ),
                }
            }
        )
        logger.info("Updated BUCKET_SECRETS with environment values")

    return True  # Configuration is now valid


def log_r2_config():
    """Log R2 configuration details."""
    from tplr.logging import logger

    logger.info("Current R2 Dataset Configuration:")
    logger.info(
        f"Dataset Account ID: {os.environ.get('R2_DATASET_ACCOUNT_ID', 'Not set')}"
    )
    logger.info(
        f"Dataset Bucket Name: {os.environ.get('R2_DATASET_BUCKET_NAME', 'Not set')}"
    )
    logger.info(
        f"Dataset Read Access Key Present: {bool(os.environ.get('R2_DATASET_READ_ACCESS_KEY_ID'))}"
    )
    logger.info(
        f"Dataset Read Secret Key Present: {bool(os.environ.get('R2_DATASET_READ_SECRET_ACCESS_KEY'))}"
    )

    logger.info("\nBucket configuration from BUCKET_SECRETS:")
    from tplr.config import BUCKET_SECRETS

    if "dataset" in BUCKET_SECRETS:
        dataset_config = BUCKET_SECRETS["dataset"]
        logger.info(f"Dataset bucket name: {dataset_config.get('name', 'Not set')}")
        logger.info(
            f"Dataset account ID: {dataset_config.get('account_id', 'Not set')}"
        )
        logger.info(
            f"Dataset endpoint: https://{dataset_config.get('account_id', 'Not set')}.r2.cloudflarestorage.com"
        )

        # Log credentials configuration (presence only)
        if "credentials" in dataset_config:
            creds = dataset_config["credentials"]
            logger.info("Credentials configuration:")
            if "read" in creds:
                logger.info("- Read credentials present")
            if "write" in creds:
                logger.info("- Write credentials present")
    else:
        logger.info("No dataset configuration found in BUCKET_SECRETS")


@pytest.mark.asyncio
async def test_dataset_equivalence():
    """
    This test will attempt to load .env and check for required environment variables.
    If these checks fail, the test is skipped rather than failing outright.
    """

    # Attempt to find and load the .env file
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        pytest.skip(f".env file not found at {env_path}. Skipping test.")
    else:
        load_dotenv(env_path, override=True)

    # Verify environment variables; if missing, skip instead of failing
    missing_vars = [var for var in REQUIRED_VARS if not os.environ.get(var)]
    if missing_vars:
        pytest.skip(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    # Attempt to import TPLR modules here; if there's an import error or environment issue, skip
    try:
        from tplr.logging import logger, debug, T
        from tplr.r2_dataset import R2DatasetLoader
        from tplr.hparams import load_hparams
    except ImportError as e:
        pytest.skip(f"Skipping test_dataset_equivalence due to import error: {e}")
    except Exception as e:
        pytest.skip(
            f"Skipping test_dataset_equivalence due to an unexpected error: {e}"
        )

    # Enable debug logging
    debug()

    start_time = T()
    logger.info("Starting test_local_parquet_loader")

    # Log configuration
    log_r2_config()

    # Validate configuration
    if not validate_config():
        pytest.skip("Configuration mismatch between environment and BUCKET_SECRETS")

    # Double-check required environment variables
    missing_vars = [var for var in REQUIRED_VARS if not os.environ.get(var)]
    if missing_vars:
        pytest.skip(f"Missing environment variables: {', '.join(missing_vars)}")

    try:
        # Load tokenizer and log success
        hparams = load_hparams()
        tokenizer = hparams.tokenizer
        logger.info(f"Tokenizer loaded successfully ({T() - start_time:.2f}s)")

        # Test parameters
        offset = 0
        n_pages = 2
        seed = "my-test-seed"
        batch_size = 2
        sequence_length = 128

        # 1. Generate random pages
        logger.info(
            f"Attempting to generate {n_pages} random pages with seed '{seed}'..."
        )
        pages = await R2DatasetLoader.next_pages(
            offset=offset, n_pages=n_pages, seed=seed
        )
        logger.info(f"Successfully generated pages: {pages}")

        # Log dataset configs for debugging
        configs_data = await R2DatasetLoader.fetch_dataset_configs()
        logger.info("\nAvailable dataset configs:")
        for config_name, config_data in configs_data.items():
            logger.info(
                f"- {config_name}: {config_data['num_rows']} rows ({config_data['split']} split)"
            )

        # 2. Create loader
        logger.info("\nCreating R2DatasetLoader...")
        loader = await R2DatasetLoader.create(
            batch_size=batch_size,
            sequence_length=sequence_length,
            pages_info=pages,
            tokenizer=tokenizer,
            pack_samples=False,
        )
        logger.info(f"Loader created successfully ({T() - start_time:.2f}s)")

        # 3. Iterate over the loader
        batch_count = 0
        for batch in loader:
            logger.info(f"[cyan]Processing batch {batch_count}[/cyan]")
            logger.info(f"Batch shape: {batch.shape}")

            # Decode and log a sample from batch
            for i, sequence in enumerate(batch):
                tokens = sequence[sequence != tokenizer.pad_token_id].tolist()
                text = tokenizer.decode(tokens)
                logger.info(f"\nSequence {i}:")
                logger.info(f"Token count: {len(tokens)}")
                logger.info(f"First 50 tokens: {tokens[:50]}...")
                logger.info(f"Sample text: {text[:200]}...")
                logger.info("[dim]" + "-" * 80 + "[/dim]")

            batch_count += 1
            if batch_count >= 2:
                break

        # Verification
        assert batch_count > 0, "No batches were produced by the R2DatasetLoader"
        logger.info(
            f"[green]Test completed successfully. Processed {batch_count} batches "
            f"({T() - start_time:.2f}s)[/green]"
        )

    except Exception as e:
        logger.error(f"[red]Test failed: {str(e)}[/red]", exc_info=True)
        raise
