#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb",
#     "asyncio",
#     "uvloop",
#     "boto3",
#     "aiohttp",
#     "python-dateutil",
#     "pandas",
#     "zstandard",
# ]
# [tool.uv]
# prerelease = "allow"
# ///

"""
WandB Historical Data Backup to R2

This script performs a complete historical backup of WandB run data to
Cloudflare R2 storage. It queries all historical run data in configurable time chunks
and uploads them in a time-partitioned format.

Data is stored with the path pattern:
wandb_data/{entity}/{project}/{run_name}/year=%Y/month=%m/{run_id}/{timestamp}_{auto_increment_id}.csv.zst

Usage:
./backup_wandb.py \
  --debug \
  --max-workers=8 \
  --wandb-entity="tplr" \
  --wandb-project="templar" \
  --start-date="2024-06-01" \
  --end-date="2025-06-10" \
  --chunk-days=7 \
  --timeout=600 \
  --r2-bucket="wandb-logs" \
  --r2-endpoint="https://<account_id>.r2.cloudflarestorage.com" \
  --r2-access-key-id="..." \
  --r2-secret-access-key="..."
"""

import argparse
import asyncio
import concurrent.futures
import datetime
import hashlib
import logging
import os
import random
import signal
import sys
import time
import traceback
from typing import List, Optional, Tuple

import boto3
import pandas as pd
import uvloop
import wandb
import zstandard as zstd
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("wandb-backup")

CHECKPOINT_FILE = ".wandb_backup_checkpoint"
DEFAULT_CHUNK_DAYS = 7


class R2Storage:
    """Handles uploading data to Cloudflare R2 storage."""

    def __init__(
        self,
        bucket_name: str,
        endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str = "auto",
    ):
        """
        Initialize R2 storage connection.

        Args:
            bucket_name: The R2 bucket name
            endpoint_url: The R2 endpoint URL
            aws_access_key_id: R2 access key ID
            aws_secret_access_key: R2 secret access key
            region_name: Region name (default: auto)
        """
        self.bucket_name = bucket_name

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

        self.s3_client = session.client(
            "s3",
            endpoint_url=endpoint_url,
            config=boto3.session.Config(  # type: ignore
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                retries={"max_attempts": 3},
                tcp_keepalive=True,
                max_pool_connections=50,
            ),
        )
        logger.info(f"Initialized R2 storage with bucket: {bucket_name}")

    async def check_run_exists(self, entity: str, project: str, run_id: str) -> bool:
        """
        Check if a run has already been backed up to R2.

        Args:
            entity: WandB entity name
            project: WandB project name
            run_id: Run ID to check

        Returns:
            bool: True if run already exists in R2, False otherwise
        """
        try:
            # Check for any files in the run's directory
            prefix = f"wandb_data/{entity}/{project}/"

            # Use list_objects_v2 with prefix to find any files for this run_id
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=1000,  # Should be enough to scan for existing runs
            )

            if "Contents" not in response:
                return False

            # Check if any object path contains this run_id
            for obj in response["Contents"]:
                key = obj["Key"]
                # Pattern: wandb_data/{entity}/{project}/{run_name}/year=YYYY/month=MM/{run_id}/...
                if f"/{run_id}/" in key:
                    logger.debug(f"Found existing backup for run {run_id}: {key}")
                    return True

            return False

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchBucket":
                logger.warning(f"Bucket {self.bucket_name} does not exist")
                return False
            else:
                logger.error(f"Error checking if run {run_id} exists: {e}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error checking run {run_id}: {e}")
            return False

    async def get_uploaded_runs(self, entity: str, project: str) -> set:
        """
        Get a set of all run IDs that have already been uploaded to R2.

        Args:
            entity: WandB entity name
            project: WandB project name

        Returns:
            set: Set of run IDs that already exist in R2
        """
        uploaded_runs = set()

        try:
            prefix = f"wandb_data/{entity}/{project}/"

            # Use paginator to handle large numbers of objects
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            for page in page_iterator:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]
                    # Extract run_id from path pattern
                    # wandb_data/{entity}/{project}/{run_name}/year=YYYY/month=MM/{run_id}/filename
                    parts = key.split("/")
                    if len(parts) >= 7:  # Minimum path depth
                        # Find the run_id part (after month= part)
                        for i, part in enumerate(parts):
                            if part.startswith("month=") and i + 1 < len(parts):
                                potential_run_id = parts[i + 1]
                                # Run IDs are typically alphanumeric, 8 chars
                                if (
                                    len(potential_run_id) >= 6
                                    and potential_run_id.isalnum()
                                ):
                                    uploaded_runs.add(potential_run_id)
                                break

            logger.info(
                f"Found {len(uploaded_runs)} existing runs in R2 for {entity}/{project}"
            )
            if logger.level <= logging.DEBUG and uploaded_runs:
                logger.debug(f"Existing runs: {sorted(list(uploaded_runs))[:10]}...")

            return uploaded_runs

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchBucket":
                logger.warning(
                    f"Bucket {self.bucket_name} does not exist, no existing runs"
                )
                return set()
            else:
                logger.error(f"Error getting uploaded runs: {e}")
                return set()
        except Exception as e:
            logger.error(f"Unexpected error getting uploaded runs: {e}")
            return set()

    async def upload_run_data(
        self,
        df: pd.DataFrame,
        entity: str,
        project: str,
        run_id: str,
        run_name: str,
        run_created_at: datetime.datetime,
    ) -> bool:
        """
        Upload WandB run data to R2 storage.

        Args:
            df: DataFrame containing run history data
            entity: WandB entity name for path construction
            project: WandB project name for path construction
            run_id: Run ID for path construction
            run_name: Run name for path construction
            run_created_at: Run creation timestamp for path construction

        Returns:
            bool: True if upload successful, False otherwise
        """
        if df.empty:
            return True

        # Use the run creation time consistently for both path and filename
        timestamp = run_created_at

        # Sanitize run_name for filesystem safety
        safe_run_name = "".join(
            c for c in run_name if c.isalnum() or c in ("-", "_", ".")
        )[:100]
        if not safe_run_name:
            safe_run_name = "unnamed_run"

        path_template = (
            f"wandb_data/{entity}/{project}/{safe_run_name}/"
            f"year={timestamp.year}/month={timestamp.month:02d}/{run_id}"
        )

        random_hex = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
        # Use run creation time for filename, not backup time
        timestamp_str = timestamp.strftime("%Y-%m-%d-%H-%M")
        filename = f"{timestamp_str}_{random_hex}.csv.zst"

        full_path = f"{path_template}/{filename}"

        # Convert DataFrame to CSV and compress with zstd
        csv_content = df.to_csv(index=False)
        compressor = zstd.ZstdCompressor()
        compressed_content = compressor.compress(csv_content.encode("utf-8"))

        logger.info(f"Uploading {len(df)} records to R2 as {full_path}")

        if not df.empty and logger.level <= logging.DEBUG:
            logger.debug(f"Sample record keys: {list(df.columns)[:10]}")

        # Retry mechanism for critical uploads
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Uploading to bucket: {self.bucket_name}, key: {full_path} (attempt {attempt + 1}/{max_retries})"
                )
                start_time = time.time()

                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=full_path,
                    Body=compressed_content,
                    ContentType="application/zstd",
                    ContentEncoding="zstd",
                    Metadata={
                        "run_id": run_id,
                        "entity": entity,
                        "project": project,
                        "backup_timestamp": timestamp.isoformat(),
                        "record_count": str(len(df)),
                    },
                )

                duration = time.time() - start_time
                logger.info(
                    f"Successfully uploaded {len(df)} records to {full_path} in {duration:.2f}s"
                )
                return True

            except ClientError as e:
                logger.error(
                    f"Upload attempt {attempt + 1} failed with ClientError: {e}"
                )
                if hasattr(e, "response") and "Error" in e.response:
                    logger.error(f"Error details: {e.response['Error']}")

                if attempt == max_retries - 1:
                    # Final attempt - save to local file as backup
                    local_backup_path = f"failed_upload_{run_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv.zst"
                    try:
                        with open(local_backup_path, "wb") as f:
                            f.write(compressed_content)
                        logger.error(
                            f"Saved failed upload to local file: {local_backup_path}"
                        )
                    except Exception as local_e:
                        logger.error(f"Could not save local backup: {local_e}")
                    return False
                else:
                    # Wait before retry
                    wait_time = 2**attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

            except Exception as e:
                logger.error(
                    f"Upload attempt {attempt + 1} failed with unexpected error: {type(e).__name__}: {e}"
                )
                logger.error(traceback.format_exc())

                if attempt == max_retries - 1:
                    # Final attempt - save to local file as backup
                    local_backup_path = f"failed_upload_{run_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv.zst"
                    try:
                        with open(local_backup_path, "wb") as f:
                            f.write(compressed_content)
                        logger.error(
                            f"Saved failed upload to local file: {local_backup_path}"
                        )
                    except Exception as local_e:
                        logger.error(f"Could not save local backup: {local_e}")
                    return False
                else:
                    # Wait before retry
                    wait_time = 2**attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        return False


def load_checkpoint() -> Optional[datetime.datetime]:
    """Load checkpoint file to track progress"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as cp:
                timestamp_str = cp.read().strip()
                if timestamp_str:
                    return datetime.datetime.fromisoformat(timestamp_str)
        except (ValueError, IOError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    return None


def save_checkpoint(timestamp: datetime.datetime) -> None:
    """Save processed timestamp to checkpoint"""
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as cp:
            cp.write(timestamp.isoformat())
        logger.debug(f"Saved checkpoint: {timestamp.isoformat()}")
    except IOError as e:
        logger.error(f"Failed to save checkpoint: {e}")


def get_runs_in_date_range(
    api: wandb.Api,
    entity: str,
    project: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> List:
    """
    Get WandB runs within a specific date range.

    Args:
        api: WandB API instance
        entity: WandB entity name
        project: WandB project name
        start_date: Start date for filtering runs
        end_date: End date for filtering runs

    Returns:
        List of WandB runs within the date range
    """
    # Format dates for MongoDB query
    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%S")
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S")

    # Build MongoDB-style query for date filtering
    date_filter = {
        "$and": [
            {"createdAt": {"$gte": start_str}},
            {"createdAt": {"$lte": end_str}},
        ]
    }

    try:
        logger.info(f"Querying runs from {start_date.date()} to {end_date.date()}")
        runs = api.runs(f"{entity}/{project}", filters=date_filter)
        run_list = list(runs)
        logger.info(f"Found {len(run_list)} runs in date range")
        return run_list
    except Exception as e:
        logger.error(f"Error fetching runs: {e}")
        return []


def process_run_data(run) -> pd.DataFrame:
    """
    Process a single WandB run and extract its history data.
    This function NEVER returns None - it always returns a DataFrame, even if empty.

    Args:
        run: WandB run object

    Returns:
        DataFrame containing run history data (may be empty but never None)
    """
    run_id = getattr(run, "id", "unknown_id")
    run_name = getattr(run, "name", "unknown_name")

    try:
        logger.info(f"Processing run: {run_id} ({run_name})")

        # Initialize empty DataFrame with basic structure
        df = pd.DataFrame()

        # Try to get history data and optimize DataFrame creation
        try:
            history_generator = run.scan_history()
            history_data = list(history_generator)

            if history_data:
                # Create DataFrame once with all history data to avoid fragmentation
                df = pd.DataFrame(history_data, copy=False)  # Avoid unnecessary copy
                logger.info(
                    f"Retrieved {len(history_data)} history records for run {run_id}"
                )
            else:
                logger.warning(
                    f"No history data found for run {run_id}, creating minimal record"
                )
                # Create minimal record even if no history
                df = pd.DataFrame([{"_timestamp": None, "_step": None}])

        except Exception as e:
            logger.error(f"Failed to retrieve history for run {run_id}: {e}")
            # Create minimal record even on history retrieval failure
            df = pd.DataFrame(
                [{"_timestamp": None, "_step": None, "_history_error": str(e)}]
            )

        # Collect all additional columns to add at once (performance optimization)
        additional_columns = {}

        # ALWAYS add run metadata, regardless of history success/failure
        try:
            additional_columns["run_id"] = run_id
            additional_columns["run_name"] = run_name
            additional_columns["run_state"] = getattr(run, "state", "unknown")

            # Handle datetime attributes safely
            created_at = getattr(run, "created_at", None)
            if created_at is not None:
                if hasattr(created_at, "isoformat"):
                    additional_columns["run_created_at"] = created_at.isoformat()
                else:
                    additional_columns["run_created_at"] = str(created_at)
            else:
                additional_columns["run_created_at"] = None

            updated_at = getattr(run, "updated_at", None)
            if updated_at is not None:
                if hasattr(updated_at, "isoformat"):
                    additional_columns["run_updated_at"] = updated_at.isoformat()
                else:
                    additional_columns["run_updated_at"] = str(updated_at)
            else:
                additional_columns["run_updated_at"] = None

            # Handle URL safely
            additional_columns["run_url"] = str(getattr(run, "url", "")) or None

            # Handle path safely - it might be a tuple/list
            path_attr = getattr(run, "path", None)
            if path_attr is not None:
                if isinstance(path_attr, (list, tuple)):
                    additional_columns["run_path"] = "/".join(str(p) for p in path_attr)
                else:
                    additional_columns["run_path"] = str(path_attr)
            else:
                additional_columns["run_path"] = None
        except Exception as e:
            logger.error(f"Failed to add run metadata for run {run_id}: {e}")
            # Try to add minimal metadata
            additional_columns["run_id"] = run_id
            additional_columns["run_name"] = run_name
            additional_columns["_metadata_error"] = str(e)

        # Try to add config data
        config_count = 0
        try:
            if hasattr(run, "config"):
                for key, value in run.config.items():
                    if not key.startswith("_"):
                        # Convert complex values to strings to avoid pandas length mismatch
                        if isinstance(value, (list, tuple, dict)):
                            additional_columns[f"config_{key}"] = str(value)
                        elif value is None:
                            additional_columns[f"config_{key}"] = None
                        else:
                            additional_columns[f"config_{key}"] = value
                        config_count += 1
        except Exception as e:
            logger.warning(f"Could not access config for run {run_id}: {e}")
            additional_columns["_config_error"] = str(e)

        # Try to add summary data
        summary_count = 0
        try:
            if hasattr(run, "summary") and hasattr(run.summary, "_json_dict"):
                for key, value in run.summary._json_dict.items():
                    if not key.startswith("_"):
                        # Convert complex values to strings to avoid pandas length mismatch
                        if isinstance(value, (list, tuple, dict)):
                            additional_columns[f"summary_{key}"] = str(value)
                        elif value is None:
                            additional_columns[f"summary_{key}"] = None
                        else:
                            additional_columns[f"summary_{key}"] = value
                        summary_count += 1
        except Exception as e:
            logger.warning(f"Could not access summary for run {run_id}: {e}")
            additional_columns["_summary_error"] = str(e)

        # Add metadata about data completeness
        try:
            additional_columns["_backup_timestamp"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            additional_columns["_config_fields_count"] = config_count
            additional_columns["_summary_fields_count"] = summary_count
            additional_columns["_data_complete"] = len(df) > 1 or any(
                col
                for col in df.columns
                if not col.startswith("_")
                and not col.startswith("run_")
                and not col.startswith("config_")
                and not col.startswith("summary_")
            )
        except Exception as e:
            logger.warning(f"Could not add metadata for run {run_id}: {e}")
            # Try to add just the essentials
            additional_columns["_backup_timestamp"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            additional_columns["_data_complete"] = False

        # Add all columns at once using pd.concat for optimal performance
        try:
            # Create a DataFrame from the additional columns with the same index as df
            additional_df = pd.DataFrame(additional_columns, index=df.index)

            # Concatenate horizontally to avoid fragmentation
            df = pd.concat([df, additional_df], axis=1)
        except Exception as e:
            logger.error(
                f"Failed to concatenate additional columns for run {run_id}: {e}"
            )
            # Fallback to assign method
            try:
                df = df.assign(**additional_columns)
            except Exception as assign_e:
                logger.error(
                    f"Failed to assign additional columns for run {run_id}: {assign_e}"
                )
                # Final fallback to individual assignment
                for col_name, col_value in additional_columns.items():
                    try:
                        df[col_name] = col_value
                    except Exception as col_e:
                        logger.warning(
                            f"Failed to add column {col_name} for run {run_id}: {col_e}"
                        )

        logger.info(
            f"Successfully processed run {run_id}: {len(df)} records, {config_count} config fields, {summary_count} summary fields"
        )
        return df

    except Exception as e:
        logger.error(f"Critical error processing run {run_id}: {e}")
        logger.error(traceback.format_exc())

        # Even on complete failure, return a minimal DataFrame with error info
        error_df = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "run_name": run_name,
                    "run_state": "error",
                    "run_created_at": None,
                    "run_updated_at": None,
                    "_critical_error": str(e),
                    "_backup_timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "_data_complete": False,
                }
            ]
        )

        logger.info(f"Created error record for run {run_id}")
        return error_df


async def process_and_upload_runs(
    executor: concurrent.futures.ThreadPoolExecutor,
    api: wandb.Api,
    r2_storage: R2Storage,
    entity: str,
    project: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    skip_existing: bool = True,
) -> bool:
    """
    Process WandB runs and upload results to R2.

    Args:
        executor: ThreadPoolExecutor for running queries
        api: WandB API instance
        r2_storage: R2 storage instance
        entity: WandB entity name
        project: WandB project name
        start_time: Start time for the query
        end_time: End time for the query

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get runs in the date range
        loop = asyncio.get_running_loop()
        runs = await loop.run_in_executor(
            executor,
            lambda: get_runs_in_date_range(api, entity, project, start_time, end_time),
        )

        if not runs:
            logger.info(f"No runs found for {start_time.date()} to {end_time.date()}")
            return True

        # Get list of already uploaded runs to avoid duplicates (if enabled)
        new_runs = runs
        skipped_count = 0

        if skip_existing:
            logger.info("Checking for existing uploads in R2...")
            uploaded_runs = await r2_storage.get_uploaded_runs(entity, project)

            # Filter out runs that have already been uploaded
            new_runs = []

            for run in runs:
                run_id = getattr(run, "id", "unknown_id")
                if run_id in uploaded_runs:
                    skipped_count += 1
                    logger.debug(f"Skipping already uploaded run: {run_id}")
                else:
                    new_runs.append(run)

            logger.info(
                f"Found {len(runs)} total runs, {skipped_count} already uploaded, {len(new_runs)} new runs to process"
            )

            if not new_runs:
                logger.info("All runs have already been uploaded, nothing to process")
                return True
        else:
            logger.info(f"Processing all {len(runs)} runs (skip-existing disabled)")

        # Process and upload in parallel using pipeline pattern
        async def process_and_upload_single_run(run):
            """Process a single run and upload immediately for optimal parallelism"""
            run_id = getattr(run, "id", "unknown_id")
            try:
                # Process run data in executor
                df = await loop.run_in_executor(
                    executor, lambda r=run: process_run_data(r)
                )

                # Get created_at for upload - handle both datetime objects and strings
                created_at_attr = getattr(run, "created_at", None)
                if created_at_attr is not None:
                    if hasattr(created_at_attr, "isoformat"):
                        # It's a datetime object
                        created_at = created_at_attr
                    else:
                        # It's likely a string, try to parse it
                        try:
                            created_at = datetime.datetime.fromisoformat(
                                str(created_at_attr).replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Could not parse created_at '{created_at_attr}' for run {run_id}, using current time"
                            )
                            created_at = datetime.datetime.now(datetime.timezone.utc)
                else:
                    logger.warning(
                        f"No created_at found for run {run_id}, using current time"
                    )
                    created_at = datetime.datetime.now(datetime.timezone.utc)

                # Upload immediately after processing
                run_name = getattr(run, "name", "unknown_name")
                upload_result = await r2_storage.upload_run_data(
                    df, entity, project, run_id, run_name, created_at
                )
                return {"run_id": run_id, "success": upload_result, "error": None}

            except Exception as e:
                # If processing fails, create a critical error record
                logger.error(f"Critical failure processing run {run_id}: {e}")

                try:
                    critical_error_df = pd.DataFrame(
                        [
                            {
                                "run_id": run_id,
                                "run_name": getattr(run, "name", "unknown_name"),
                                "run_state": "critical_failure",
                                "_executor_error": str(e),
                                "_backup_timestamp": datetime.datetime.now(
                                    datetime.timezone.utc
                                ).isoformat(),
                                "_data_complete": False,
                            }
                        ]
                    )

                    created_at_attr = getattr(run, "created_at", None)
                    if created_at_attr is not None:
                        if hasattr(created_at_attr, "isoformat"):
                            # It's a datetime object
                            created_at = created_at_attr
                        else:
                            # It's likely a string, try to parse it
                            try:
                                created_at = datetime.datetime.fromisoformat(
                                    str(created_at_attr).replace("Z", "+00:00")
                                )
                            except (ValueError, TypeError):
                                created_at = datetime.datetime.now(
                                    datetime.timezone.utc
                                )
                    else:
                        created_at = datetime.datetime.now(datetime.timezone.utc)

                    # Upload error record
                    run_name = getattr(run, "name", "unknown_name")
                    upload_result = await r2_storage.upload_run_data(
                        critical_error_df, entity, project, run_id, run_name, created_at
                    )
                    return {"run_id": run_id, "success": upload_result, "error": str(e)}

                except Exception as upload_e:
                    logger.error(
                        f"Failed to upload error record for run {run_id}: {upload_e}"
                    )
                    return {
                        "run_id": run_id,
                        "success": False,
                        "error": f"Processing: {e}, Upload: {upload_e}",
                    }

        # Create tasks for new runs to process and upload in parallel
        logger.info(
            f"Starting parallel processing and upload for {len(new_runs)} new runs"
        )
        pipeline_tasks = [process_and_upload_single_run(run) for run in new_runs]

        # Execute all tasks concurrently with progress tracking
        results = []
        completed = 0

        # Process in batches to avoid overwhelming the system
        batch_size = min(10, len(pipeline_tasks))  # Process up to 10 runs concurrently

        for i in range(0, len(pipeline_tasks), batch_size):
            batch = pipeline_tasks[i : i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(pipeline_tasks) + batch_size - 1) // batch_size} ({len(batch)} runs)"
            )

            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)
            completed += len(batch)

            logger.info(f"Completed {completed}/{len(new_runs)} new runs")

        # Analyze results
        successful_uploads = 0
        failed_uploads = 0
        processing_errors = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Pipeline task {i} failed with exception: {result}")
                failed_uploads += 1
            elif isinstance(result, dict):
                if result.get("success", False):
                    successful_uploads += 1
                else:
                    failed_uploads += 1
                    if result.get("error"):
                        processing_errors += 1
                        logger.error(
                            f"Run {result.get('run_id', 'unknown')} failed: {result['error']}"
                        )
            else:
                logger.warning(f"Unexpected result type for task {i}: {type(result)}")
                failed_uploads += 1

        logger.info(
            f"Pipeline results: {successful_uploads} successful, {failed_uploads} failed "
            f"({processing_errors} processing errors) out of {len(new_runs)} new runs processed "
            f"({skipped_count} runs skipped as already uploaded)"
        )

        # Consider the chunk successful if at least some uploads succeeded
        # For mission-critical backup, we log failures but don't fail the entire chunk
        if successful_uploads > 0:
            return True
        else:
            logger.error("All pipeline operations in this chunk failed")
            return False

    except Exception as e:
        logger.error(f"Error processing and uploading runs: {e}")
        logger.error(traceback.format_exc())
        return False


def generate_time_chunks(
    start_date: datetime.datetime, end_date: datetime.datetime, chunk_days: int
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """
    Generate time chunks for processing data in manageable pieces.

    Args:
        start_date: Start date for backup
        end_date: End date for backup
        chunk_days: Number of days per chunk

    Returns:
        List of (start_time, end_time) tuples
    """
    chunks = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + datetime.timedelta(days=chunk_days), end_date)
        chunks.append((current_start, current_end))
        current_start = current_end

    return chunks


async def main():
    """Main entry point for the WandB historical backup to R2."""
    parser = argparse.ArgumentParser(description="WandB Historical Data Backup to R2")

    parser.add_argument(
        "--wandb-entity",
        required=True,
        help="WandB entity name",
    )
    parser.add_argument(
        "--wandb-project",
        required=True,
        help="WandB project name",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date for backup (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for backup (YYYY-MM-DD format). Defaults to now.",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=DEFAULT_CHUNK_DAYS,
        help=f"Number of days to process per chunk (default: {DEFAULT_CHUNK_DAYS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Connection timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )

    parser.add_argument("--r2-bucket", required=True, help="R2 bucket name")
    parser.add_argument("--r2-endpoint", required=True, help="R2 endpoint URL")
    parser.add_argument("--r2-region", default="auto", help="R2 region")
    parser.add_argument(
        "--r2-access-key-id",
        default=os.environ.get("R2_ACCESS_KEY_ID"),
        help="R2 access key ID (defaults to R2_ACCESS_KEY_ID env var)",
    )
    parser.add_argument(
        "--r2-secret-access-key",
        default=os.environ.get("R2_SECRET_ACCESS_KEY"),
        help="R2 secret access key (defaults to R2_SECRET_ACCESS_KEY env var)",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of worker threads (default: 5)",
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test connection with a simple query first",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip runs that already exist in R2 bucket (default: True)",
    )
    parser.add_argument(
        "--force-reupload",
        action="store_true",
        help="Force re-upload of all runs, even if they exist in R2",
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if not args.r2_access_key_id or not args.r2_secret_access_key:
        logger.error(
            "R2 credentials must be provided via --r2-access-key-id and --r2-secret-access-key arguments or R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY environment variables"
        )
        sys.exit(1)

    try:
        start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").replace(
            tzinfo=datetime.timezone.utc
        )
    except ValueError:
        logger.error("Invalid start date format. Use YYYY-MM-DD.")
        sys.exit(1)

    end_date = datetime.datetime.now(datetime.timezone.utc)
    if args.end_date:
        try:
            end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").replace(
                tzinfo=datetime.timezone.utc
            )
        except ValueError:
            logger.error("Invalid end date format. Use YYYY-MM-DD.")
            sys.exit(1)

    if args.resume:
        checkpoint_date = load_checkpoint()
        if checkpoint_date:
            start_date = checkpoint_date
            logger.info(f"Resuming from checkpoint: {start_date.date()}")

    if start_date >= end_date:
        logger.error("Start date must be before end date")
        sys.exit(1)

    r2_storage = R2Storage(
        bucket_name=args.r2_bucket,
        endpoint_url=args.r2_endpoint,
        aws_access_key_id=args.r2_access_key_id,
        aws_secret_access_key=args.r2_secret_access_key,
        region_name=args.r2_region,
    )

    api = wandb.Api()

    running = True
    loop = asyncio.get_running_loop()

    def signal_handler():
        nonlocal running
        logger.info("Received shutdown signal, stopping...")
        running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        logger.info(
            f"Starting WandB historical backup from {start_date.date()} to {end_date.date()}"
        )
        logger.info(f"Entity: {args.wandb_entity}, Project: {args.wandb_project}")

        if args.test_connection:
            logger.info("Testing WandB connection...")
            try:
                test_runs = api.runs(f"{args.wandb_entity}/{args.wandb_project}")
                list(test_runs)[:1]  # Just get one run to test connection
                logger.info("Connection test successful! Found runs in project.")
                return
            except Exception as e:
                logger.error(f"Connection test failed: {e}")
                return

        time_chunks = generate_time_chunks(start_date, end_date, args.chunk_days)
        logger.info(
            f"Processing {len(time_chunks)} time chunks of {args.chunk_days} days each"
        )

        total_chunks = len(time_chunks)
        successful_chunks = 0

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            for i, (chunk_start, chunk_end) in enumerate(time_chunks):
                if not running:
                    logger.info("Stopping due to shutdown signal")
                    break

                try:
                    logger.info(
                        f"Processing chunk {i + 1}/{total_chunks}: {chunk_start.date()} to {chunk_end.date()}"
                    )

                    start_time = time.time()
                    success = await process_and_upload_runs(
                        executor,
                        api,
                        r2_storage,
                        args.wandb_entity,
                        args.wandb_project,
                        chunk_start,
                        chunk_end,
                        skip_existing=args.skip_existing and not args.force_reupload,
                    )

                    duration = time.time() - start_time
                    if success:
                        logger.info(
                            f"Completed chunk {i + 1}/{total_chunks} in {duration:.2f} seconds"
                        )
                        successful_chunks += 1
                        save_checkpoint(chunk_end)
                    else:
                        logger.warning(
                            f"Chunk {i + 1}/{total_chunks} completed with errors in {duration:.2f} seconds"
                        )

                except Exception as e:
                    logger.error(f"Error processing chunk {i + 1}/{total_chunks}: {e}")
                    logger.error(traceback.format_exc())

        logger.info(
            f"Backup completed: {successful_chunks}/{total_chunks} chunks processed successfully"
        )

        if successful_chunks == total_chunks:
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
                logger.info("Backup completed successfully, checkpoint file removed")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        logger.info("WandB Historical Backup shutting down")


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())
