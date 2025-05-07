#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "boto3",
#   "tqdm",
#   "uvloop",
#   "python-dotenv",
#   "aiofiles",
# ]
# ///
"""
R2StorageValidator - Validates files in Cloudflare R2 storage against a manifest file

Efficiently validates file existence and sizes of files in R2 storage against a manifest JSON file,
using concurrent processing for optimal performance.
"""

import asyncio
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple

import boto3
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()

MAX_CONCURRENT_TASKS = 50
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for download
MAX_CONNECTIONS = 100
MAX_THREAD_WORKERS = 20
BATCH_SIZE = 100


@dataclass
class ManifestFile:
    """Represents a file entry from the manifest file."""

    path: str
    num_rows: int
    byte_size: int
    global_shard: str = ""


@dataclass
class ValidationResult:
    """Stores the validation result for a single file."""

    path: str
    exists: bool
    size_match: bool
    expected_size: int
    actual_size: Optional[int]
    error: Optional[str] = None


@dataclass
class R2Config:
    """Configuration for Cloudflare R2 storage."""

    account_id: str
    access_key_id: str
    access_key_secret: str
    bucket_name: str
    region: str = "auto"


def create_s3_client(r2_config: R2Config) -> Any:
    """
    Create an S3 client optimized for Cloudflare R2 access.

    Args:
        r2_config: Configuration containing R2 credentials and settings

    Returns:
        Configured boto3 S3 client
    """
    session = boto3.Session(
        aws_access_key_id=r2_config.access_key_id,
        aws_secret_access_key=r2_config.access_key_secret,
        region_name=r2_config.region,
    )

    endpoint_url = f"https://{r2_config.account_id}.r2.cloudflarestorage.com"

    return session.client(
        "s3",
        endpoint_url=endpoint_url,
        config=boto3.session.Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            max_pool_connections=MAX_CONNECTIONS,
            tcp_keepalive=True,
        ),
    )


def parse_manifest(manifest_path: str) -> List[ManifestFile]:
    """
    Parse the manifest JSON file containing file information.

    Args:
        manifest_path: Path to the manifest JSON file

    Returns:
        List of ManifestFile objects containing file metadata
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    files = []
    # Check if the manifest has the expected structure with global_shard keys
    if any(key != "default" and isinstance(manifest[key], dict) for key in manifest):
        # Standard format with global_shard keys
        for global_shard, data in manifest.items():
            if global_shard == "default":
                continue

            for shard in data.get("shards", []):
                files.append(
                    ManifestFile(
                        path=shard.get("path", ""),
                        num_rows=shard.get("num_rows", 0),
                        byte_size=shard.get("byte_size", 0),
                        global_shard=global_shard,
                    )
                )
    else:
        # Alternative format: each key is a file path and value contains metadata
        for path, metadata in manifest.items():
            if isinstance(metadata, dict):
                files.append(
                    ManifestFile(
                        path=path,
                        num_rows=metadata.get("num_rows", 0), 
                        byte_size=metadata.get("byte_size", 0)
                    )
                )

    if not files:
        print("Warning: No valid files found in manifest. Check the manifest format.")
    
    return files


def file_batches(
    files: List[ManifestFile], batch_size: int
) -> Iterator[List[ManifestFile]]:
    """
    Split files into batches for concurrent processing.

    Args:
        files: List of files to process
        batch_size: Maximum files per batch

    Returns:
        Iterator of file batches
    """
    for i in range(0, len(files), batch_size):
        yield files[i : i + batch_size]


def normalize_r2_path(path: str) -> str:
    """
    Normalize a path from the manifest for R2 storage access.
    Explicitly handles the 'dataset/' prefix case.

    Args:
        path: Original path from manifest

    Returns:
        Normalized path suitable for R2 access
    """
    # Directly handle the specific prefix case you mentioned
    if path.startswith("dataset/"):
        return path[len("dataset/") :]

    return path


async def check_file_exists_and_size(
    s3_client: Any, bucket: str, key: str, debug: bool = False
) -> Tuple[bool, Optional[int]]:
    """
    Check if a file exists in R2 storage and get its size.

    Args:
        s3_client: Boto3 S3 client
        bucket: Bucket name
        key: Object key (path)
        debug: Whether to print debug information

    Returns:
        Tuple of (exists, size) where size is None if file doesn't exist
    """
    if debug:
        print(f"R2 API call: HEAD bucket={bucket}, key={key}")

    loop = asyncio.get_event_loop()

    try:
        with ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS) as executor:
            response = await loop.run_in_executor(
                executor, lambda: s3_client.head_object(Bucket=bucket, Key=key)
            )

            if debug:
                print(f"File exists: {key}, size: {response['ContentLength']}")

            return True, response["ContentLength"]
    except Exception as e:
        if debug:
            print(f"File not found: {key}, error: {str(e)}")
        return False, None


async def validate_file(
    s3_client: Any,
    bucket: str,
    file: ManifestFile,
    debug: bool = False,
) -> ValidationResult:
    """
    Validate a single file against the manifest.

    Args:
        s3_client: Boto3 S3 client
        bucket: Bucket name
        file: ManifestFile object with expected metadata
        debug: Whether to print debug information

    Returns:
        ValidationResult containing comparison results
    """
    try:
        # Extract the actual path for R2 access
        manifest_path = file.path
        r2_path = normalize_r2_path(manifest_path)

        if debug:
            print("\nProcessing file:")
            print(f"  Manifest path: {manifest_path}")
            print(f"  Normalized R2 path: {r2_path}")

        # Check if file exists and get size
        exists, actual_size = await check_file_exists_and_size(
            s3_client, bucket, r2_path, debug
        )

        if not exists:
            return ValidationResult(
                path=manifest_path,
                exists=False,
                size_match=False,
                expected_size=file.byte_size,
                actual_size=None,
                error=f"File does not exist (R2 path: {r2_path})",
            )

        # Check size
        size_match = actual_size == file.byte_size

        error = None
        if not size_match:
            error = f"Size mismatch: expected {file.byte_size}, got {actual_size}"

        return ValidationResult(
            path=manifest_path,
            exists=True,
            size_match=size_match,
            expected_size=file.byte_size,
            actual_size=actual_size,
            error=error,
        )
    except Exception as e:
        if debug:
            print(f"Error validating file {file.path}: {str(e)}")

        return ValidationResult(
            path=file.path,
            exists=False,
            size_match=False,
            expected_size=file.byte_size,
            actual_size=None,
            error=f"Validation error: {str(e)}",
        )


async def validate_files_batch(
    s3_client: Any,
    bucket: str,
    files: List[ManifestFile],
    progress_bar: Optional[tqdm] = None,
    debug: bool = False,
) -> List[ValidationResult]:
    """
    Validate a batch of files concurrently.

    Args:
        s3_client: Boto3 S3 client
        bucket: Bucket name
        files: List of files to validate
        progress_bar: Optional progress bar
        debug: Whether to print debug information

    Returns:
        List of ValidationResult objects
    """
    tasks = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    async def validate_with_semaphore(file: ManifestFile) -> ValidationResult:
        async with semaphore:
            result = await validate_file(
                s3_client,
                bucket,
                file,
                debug,
            )
            if progress_bar:
                progress_bar.update(1)
            return result

    for file in files:
        tasks.append(validate_with_semaphore(file))

    return await asyncio.gather(*tasks)


async def save_results_to_csv(
    results: List[ValidationResult], output_path: str
) -> None:
    """
    Save validation results to a CSV file.

    Args:
        results: List of validation results
        output_path: Path to output CSV file
    """
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write results to CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "path",
                "exists",
                "size_match",
                "expected_size",
                "actual_size",
                "error",
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result.path,
                    result.exists,
                    result.size_match,
                    result.expected_size,
                    result.actual_size if result.actual_size is not None else "",
                    result.error if result.error is not None else "",
                ]
            )


def print_summary(results: List[ValidationResult]) -> None:
    """
    Print a summary of validation results to the console.

    Args:
        results: List of validation results
    """
    total = len(results)
    errors = [r for r in results if r.error is not None]
    error_count = len(errors)

    print("\nValidation Summary:")
    print(f"  Total files: {total}")
    print(f"  Successful: {total - error_count}")
    print(f"  Failed: {error_count}")

    if error_count > 0:
        # Categorize errors
        missing_files = [r for r in errors if not r.exists]
        size_mismatches = [r for r in errors if r.exists and not r.size_match]
        other_errors = [r for r in errors if r.exists and r.size_match and r.error]

        print("\nError breakdown:")
        print(f"  Missing files: {len(missing_files)}")
        print(f"  Size mismatches: {len(size_mismatches)}")
        print(f"  Other errors: {len(other_errors)}")

        # Print sample errors from each category
        def print_sample(
            category_name: str,
            errors_list: List[ValidationResult],
            max_samples: int = 5,
        ) -> None:
            if not errors_list:
                return

            print(
                f"\n{category_name} (showing {min(max_samples, len(errors_list))} of {len(errors_list)}):"
            )
            for i, err in enumerate(errors_list[:max_samples]):
                if category_name == "Missing files":
                    r2_path = normalize_r2_path(err.path)
                    print(f"  {i + 1}. {err.path} → {r2_path}")
                elif category_name == "Size mismatches":
                    print(
                        f"  {i + 1}. {err.path}: expected {err.expected_size} bytes, got {err.actual_size} bytes"
                    )
                else:
                    print(f"  {i + 1}. {err.path}: {err.error}")

        print_sample("Missing files", missing_files)
        print_sample("Size mismatches", size_mismatches)
        print_sample("Other errors", other_errors)


async def validate_r2_storage(
    manifest_path: str,
    r2_config: R2Config,
    output_path: str,
    batch_size: int = BATCH_SIZE,
    silent_mode: bool = False,
    debug: bool = False,
) -> None:
    """
    Validate R2 storage against a manifest file.

    Args:
        manifest_path: Path to the manifest JSON file
        r2_config: R2 configuration
        output_path: Path to save the CSV results
        batch_size: Number of files to validate in each batch
        silent_mode: Whether to suppress progress display
        debug: Whether to print debug information
    """
    # Parse manifest
    print(f"Parsing manifest file: {manifest_path}")
    files = parse_manifest(manifest_path)
    total_files = len(files)
    print(f"Found {total_files} files in manifest")

    if total_files == 0:
        print("No files to validate. Exiting.")
        return

    # Create S3 client
    print("Initializing R2 client...")
    s3_client = create_s3_client(r2_config)

    # Print R2 configuration for debugging
    if debug:
        print("R2 Configuration:")
        print(f"  Account ID: {r2_config.account_id}")
        print(f"  Bucket: {r2_config.bucket_name}")
        print(f"  Region: {r2_config.region}")
        print(f"  Endpoint: https://{r2_config.account_id}.r2.cloudflarestorage.com")

    # Create progress bar
    progress_bar = (
        None
        if silent_mode
        else tqdm(total=total_files, desc="Validating files", unit="file")
    )

    # Process files in batches
    print(
        f"Validating files in batches of {batch_size}, checking only existence and file size"
    )
    all_results = []

    for i, batch in enumerate(file_batches(files, batch_size)):
        if not silent_mode:
            batch_num = i + 1
            batch_total = (total_files + batch_size - 1) // batch_size
            print(f"Processing batch {batch_num}/{batch_total} ({len(batch)} files)")

        results = await validate_files_batch(
            s3_client,
            r2_config.bucket_name,
            batch,
            progress_bar,
            debug,
        )
        all_results.extend(results)

    if progress_bar:
        progress_bar.close()

    # Save results to CSV
    print(f"\nSaving results to {output_path}")
    await save_results_to_csv(all_results, output_path)

    # Print summary
    print_summary(all_results)


async def main() -> None:
    """
    Main entry point for the script that validates file existence and sizes.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate R2 storage against a manifest file"
    )
    parser.add_argument("manifest_path", help="Path to the manifest JSON file")
    parser.add_argument(
        "--output", default="validation_results.csv", help="Output CSV file path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of files to validate in a batch",
    )
    parser.add_argument("--silent", action="store_true", help="Suppress progress bar")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    # R2 configuration
    parser.add_argument(
        "--r2-account-id", required=True, help="Cloudflare R2 account ID"
    )
    parser.add_argument(
        "--r2-access-key-id", required=True, help="Cloudflare R2 access key ID"
    )
    parser.add_argument(
        "--r2-access-key-secret", required=True, help="Cloudflare R2 access key secret"
    )
    parser.add_argument("--r2-bucket", required=True, help="Cloudflare R2 bucket name")
    parser.add_argument("--r2-region", default="auto", help="Cloudflare R2 region")

    args = parser.parse_args()

    r2_config = R2Config(
        account_id=args.r2_account_id,
        access_key_id=args.r2_access_key_id,
        access_key_secret=args.r2_access_key_secret,
        bucket_name=args.r2_bucket,
        region=args.r2_region,
    )

    print(f"Validating R2 storage against manifest: {args.manifest_path}")
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")

    await validate_r2_storage(
        manifest_path=args.manifest_path,
        r2_config=r2_config,
        output_path=args.output,
        batch_size=args.batch_size,
        silent_mode=args.silent,
        debug=args.debug,
    )


if __name__ == "__main__":
    import uvloop

    uvloop.install()
    asyncio.run(main())
