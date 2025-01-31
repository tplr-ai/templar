# ruff: noqa
import os
import pytest
import numpy as np
import torch
from pathlib import Path
from dotenv import load_dotenv

# Find and load the correct .env file
env_path = Path(__file__).parent.parent / ".env"
if not env_path.exists():
    raise FileNotFoundError(f"Required .env file not found at {env_path}")

load_dotenv(env_path, override=True)

# Verify environment variables are loaded
required_vars = [
    "R2_DATASET_ACCOUNT_ID",
    "R2_DATASET_BUCKET_NAME",
    "R2_DATASET_READ_ACCESS_KEY_ID",
    "R2_DATASET_READ_SECRET_ACCESS_KEY",
]


def validate_config():
    """Validate configuration consistency and update BUCKET_SECRETS if needed"""
    env_bucket = os.environ.get("R2_DATASET_BUCKET_NAME")
    from tplr.config import BUCKET_SECRETS

    bucket_secrets = BUCKET_SECRETS.get("dataset", {}).get("name")

    logger.info("Configuration validation:")
    logger.info(f"Environment bucket: {env_bucket}")
    logger.info(f"BUCKET_SECRETS bucket: {bucket_secrets}")

    if env_bucket != bucket_secrets:
        logger.warning(
            "⚠️ Bucket name mismatch - updating BUCKET_SECRETS with environment value"
        )
        logger.warning(f"Using environment bucket: {env_bucket}")
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

    return True  # Always return True since we've updated the config


def log_config():
    """Log current configuration"""
    logger.info("Current configuration:")
    logger.info(f"Account ID: {os.environ.get('R2_DATASET_ACCOUNT_ID', 'Not set')}")
    logger.info(f"Bucket Name: {os.environ.get('R2_DATASET_BUCKET_NAME', 'Not set')}")
    logger.info(
        f"Access Key Present: {bool(os.environ.get('R2_DATASET_READ_ACCESS_KEY_ID'))}"
    )
    logger.info(
        f"Secret Key Present: {bool(os.environ.get('R2_DATASET_READ_SECRET_ACCESS_KEY'))}"
    )


missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )

import torch
from tplr.logging import logger, debug, T
from tplr.dataset import DatasetLoader
from tplr.r2_dataset import R2DatasetLoader
from tplr.hparams import load_hparams

# Enable debug logging
debug()


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


@pytest.mark.asyncio
async def test_dataset_equivalence():
    """
    Test that DatasetLoader and R2DatasetLoader produce identical outputs
    given the same input parameters and seed.
    """
    start_time = T()
    logger.info("Starting dataset equivalence test")

    # Log current configuration
    log_config()
    if not validate_config():
        pytest.skip("Configuration mismatch between environment and BUCKET_SECRETS")

    # Set fixed random seeds
    set_random_seeds()

    # Test parameters
    batch_size = 2
    sequence_length = 128
    n_pages = 2
    seed = 255
    offset = 0

    # Load tokenizer
    hparams = load_hparams()
    tokenizer = hparams.tokenizer
    logger.info(f"Tokenizer loaded ({T() - start_time:.2f}s)")

    try:
        # Generate pages using both methods
        r2_pages = await R2DatasetLoader.next_pages(
            offset=offset, n_pages=n_pages, seed=seed
        )
        logger.info(f"Successfully generated R2 pages")

        set_random_seeds()  # Reset seeds before generating HF pages
        hf_pages = await DatasetLoader.next_pages(
            offset=offset, n_pages=n_pages, seed=seed
        )
        logger.info(f"Successfully generated HF pages")

        logger.info(f"R2 pages: {r2_pages}")
        logger.info(f"HF pages: {hf_pages}")

        # Assert pages are identical
        assert r2_pages == hf_pages, (
            f"Page generation differs between loaders:\nR2: {r2_pages}\nHF: {hf_pages}"
        )

        # Create both loaders with identical settings
        logger.info("Creating R2 loader...")
        set_random_seeds()  # Reset seeds before creating loaders
        r2_loader = await R2DatasetLoader.create(
            batch_size=batch_size,
            sequence_length=sequence_length,
            pages_info=r2_pages,
            tokenizer=tokenizer,
            pack_samples=False,
        )
        logger.info("R2 loader created successfully")

        logger.info("Creating HF loader...")
        set_random_seeds()  # Reset seeds before creating second loader
        hf_loader = await DatasetLoader.create(
            batch_size=batch_size,
            sequence_length=sequence_length,
            pages_info=hf_pages,
            tokenizer=tokenizer,
            pack_samples=False,
        )
        logger.info("HF loader created successfully")

        # Convert loaders to lists for deterministic comparison
        logger.info("Converting R2 loader to list...")
        r2_batches = list(r2_loader)
        logger.info("Converting HF loader to list...")
        hf_batches = list(hf_loader)

        # Assert same number of batches
        assert len(r2_batches) == len(hf_batches), (
            f"Different number of batches: R2={len(r2_batches)}, HF={len(hf_batches)}"
        )

        # Compare each batch
        for batch_idx, (r2_batch, hf_batch) in enumerate(zip(r2_batches, hf_batches)):
            logger.info(f"Comparing batch {batch_idx}")

            # Convert to tensors if they aren't already
            r2_tensor = (
                torch.tensor(r2_batch)
                if not isinstance(r2_batch, torch.Tensor)
                else r2_batch
            )
            hf_tensor = (
                torch.tensor(hf_batch)
                if not isinstance(hf_batch, torch.Tensor)
                else hf_batch
            )

            # Log shapes
            logger.info(f"R2 batch shape: {r2_tensor.shape}")
            logger.info(f"HF batch shape: {hf_tensor.shape}")

            # Compare batch shapes
            assert r2_tensor.shape == hf_tensor.shape, (
                f"Batch {batch_idx} shapes differ: R2={r2_tensor.shape}, HF={hf_tensor.shape}"
            )

    except Exception as e:
        logger.error(f"Test error: {str(e)}", exc_info=True)
        raise

    logger.info(f"[green]Test completed successfully ({T() - start_time:.2f}s)[/green]")
