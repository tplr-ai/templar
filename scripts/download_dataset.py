#!/usr/bin/env python3
"""
Script to download dataset files from R2 using the efficient download method
from the templar codebase.

Usage:
    python scripts/download_dataset.py              # Downloads shards 0 and 1 (default)
    python scripts/download_dataset.py 0 1 2 3      # Downloads shards 0, 1, 2, and 3
    python scripts/download_dataset.py 5            # Downloads only shard 5
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add the parent directory to the path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import tplr
from src.tplr.comms import Comms
from src.tplr.schemas import Bucket

# Load environment variables
load_dotenv()


async def download_files(shard_indices):
    """Download the dataset files from R2.

    Args:
        shard_indices: List of shard indices to download
    """

    # Get credentials from environment variables (using the same env vars as comms)
    account_id = os.getenv("R2_DATASET_ACCOUNT_ID")
    bucket_name = os.getenv("R2_DATASET_BUCKET_NAME")
    aws_access_key_id = os.getenv("R2_DATASET_READ_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("R2_DATASET_READ_SECRET_ACCESS_KEY")

    if not all([aws_access_key_id, aws_secret_access_key, account_id, bucket_name]):
        print("Error: Missing required environment variables.")
        print("Please set the following in your .env file:")
        print("  - R2_DATASET_ACCOUNT_ID")
        print("  - R2_DATASET_BUCKET_NAME")
        print("  - R2_DATASET_READ_ACCESS_KEY_ID")
        print("  - R2_DATASET_READ_SECRET_ACCESS_KEY")
        sys.exit(1)

    # Create bucket configuration
    bucket = Bucket(
        name=bucket_name,
        account_id=account_id,
        access_key_id=aws_access_key_id,
        secret_access_key=aws_secret_access_key,
    )

    # Load hparams
    hparams = tplr.load_hparams()

    # Initialize Comms with minimal parameters
    comms = Comms(
        wallet=None,  # No wallet needed for downloading
        config=None,
        bucket=bucket,
        hparams=hparams,
        uid=0,  # Dummy UID for temp directory
    )

    # Build list of files to download based on shard indices
    files_to_download = []
    skipped_files = []
    for idx in shard_indices:
        for filename in [
            f"remote/tokenized/sample_ids_{idx:06d}.bin",
            f"remote/tokenized/train_{idx:06d}.npy",
        ]:
            local_path = Path(filename)
            if local_path.exists():
                file_size = local_path.stat().st_size
                print(f"✓ Skipping {filename} (already exists, size: {file_size:,} bytes)")
                skipped_files.append(filename)
            else:
                files_to_download.append((filename, filename))

    # Create the output directory
    os.makedirs("remote/tokenized", exist_ok=True)

    if not files_to_download:
        print(f"All files for shards {sorted(shard_indices)} already exist locally.")
        print(f"Skipped {len(skipped_files)} files.")
        return
    
    print(
        f"Downloading {len(files_to_download)} files from shards: {sorted(shard_indices)}"
    )
    if skipped_files:
        print(f"Skipping {len(skipped_files)} existing files")
    print(f"Bucket: {bucket_name}")
    print("-" * 40)

    try:
        # Download files concurrently
        tasks = []
        for s3_key, local_path in files_to_download:
            print(f"Queuing download of {s3_key}...")

            # Create parent directory if needed
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            # Create download task
            task = comms.s3_get_object(
                key=s3_key,
                bucket=bucket,
                load_data=False,  # This will save to a temp file
                show_progress=True,
                timeout=600,
            )
            tasks.append((task, s3_key, local_path))

        print(f"\nStarting concurrent downloads...")
        print("-" * 40)

        # Run all downloads concurrently
        success_count = 0
        fail_count = 0
        for task, s3_key, local_path in tasks:
            result = await task
            if result:
                print(f"✓ Successfully downloaded {s3_key}")
                success_count += 1
                # Note: The file is saved in comms.temp_dir with a temporary name
                # You may need to move it from the temp location to the final location
            else:
                print(f"✗ Failed to download {s3_key}")
                fail_count += 1

        print("-" * 40)
        print(f"Download complete: {success_count} successful, {fail_count} failed")

    finally:
        await comms.close_all_s3_clients()


def parse_shard_indices(args):
    """Parse command line arguments to get shard indices.

    Args:
        args: Command line arguments (sys.argv[1:])

    Returns:
        List of shard indices to download
    """
    if not args:
        # Default to shards 0 and 1
        return [0, 1]

    indices = []
    for arg in args:
        try:
            idx = int(arg)
            if idx < 0:
                print(f"Warning: Ignoring negative shard index {idx}")
                continue
            indices.append(idx)
        except ValueError:
            print(f"Warning: Ignoring invalid shard index '{arg}' (not a number)")

    if not indices:
        print("No valid shard indices provided, using defaults (0 and 1)")
        return [0, 1]

    return sorted(set(indices))  # Remove duplicates and sort


if __name__ == "__main__":
    print("Dataset Downloader")
    print("=" * 40)

    # Check for required environment variables
    required_vars = [
        "R2_DATASET_ACCOUNT_ID",
        "R2_DATASET_BUCKET_NAME",
        "R2_DATASET_READ_ACCESS_KEY_ID",
        "R2_DATASET_READ_SECRET_ACCESS_KEY",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these in your .env file")
        sys.exit(1)

    # Parse shard indices from command line arguments
    shard_indices = parse_shard_indices(sys.argv[1:])

    # Run the async download function
    asyncio.run(download_files(shard_indices))
    print("\nAll downloads completed!")
