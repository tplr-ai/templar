#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "influxdb-client",
#     "asyncio",
#     "uvloop",
#     "boto3",
#     "aiohttp",
#     "python-dateutil",
# ]
# [tool.uv]
# prerelease = "allow"
# ///

"""
InfluxDB Historical Data Backup to R2

This script performs a complete historical backup of InfluxDB metrics data to
Cloudflare R2 storage. It queries all historical data in configurable time chunks
and uploads them in the same time-partitioned format as the metrics collector.

Data is stored with the path pattern:
influxdb_data/role={role}/uid={uid}/_measurement={_measurement}/_field={_field}/year=%Y/month=%m/day=%d/hour=%H/{timestamp}_{auto_increment_id}.jsonl

./backup_metrics.py \
  --debug \
  --max-workers=8 \
  --influxdb-url="..." \
  --influxdb-token="..." \
  --influxdb-org="tplr" \
  --influxdb-bucket="tplr" \
  --start-date="2025-04-01" \
  --chunk-days=7 \
  --timeout=600 \
  --r2-bucket="influxdb-logs" \
  --r2-endpoint="https://<account_id>.r2.cloudflarestorage.com" \
  --r2-access-key-id="..." \
  --r2-secret-access-key="..."
"""

import argparse
import asyncio
import concurrent.futures
import datetime
import hashlib
import json
import logging
import os
import random
import signal
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import boto3
import uvloop
from botocore.exceptions import ClientError
from dateutil.relativedelta import relativedelta
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.influxdb_client import InfluxDBClient
from urllib3.util.timeout import Timeout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("influx-backup")

CHECKPOINT_FILE = ".backup_checkpoint"
DEFAULT_CHUNK_DAYS = 7  # Process data in 7-day chunks


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

    async def upload_data(
        self, records: List[Dict], role: str, uid: str, measurement: str, field: str
    ) -> bool:
        """
        Upload InfluxDB data to R2 storage.

        Args:
            records: List of data records to upload
            role: Role value for path construction
            uid: UID value for path construction
            measurement: Measurement name for path construction
            field: Field name for path construction

        Returns:
            bool: True if upload successful, False otherwise
        """
        if not records:
            return True

        timestamp = datetime.datetime.now(datetime.timezone.utc)
        if records and "_time" in records[0]:
            try:
                time_str = records[0]["_time"]
                if isinstance(time_str, str):
                    timestamp = datetime.datetime.fromisoformat(
                        time_str.replace("Z", "+00:00")
                    )
                elif isinstance(time_str, datetime.datetime):
                    timestamp = time_str
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse timestamp from record: {e}")

        path_template = (
            f"influxdb_data/role={role}/uid={uid}/_measurement={measurement}/_field={field}/"
            f"year={timestamp.year}/month={timestamp.month:02d}/day={timestamp.day:02d}/hour={timestamp.hour:02d}"
        )

        random_hex = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
        timestamp_str = timestamp.strftime("%y-%m-%d-%H-%M")
        filename = f"{timestamp_str}_{random_hex}.jsonl"

        full_path = f"{path_template}/{filename}"

        jsonl_content = "\n".join(json.dumps(record) for record in records)

        logger.info(f"Uploading {len(records)} records to R2 as {full_path}")

        if records and logger.level <= logging.DEBUG:
            sample_record = json.dumps(records[0])
            logger.debug(f"Sample record: {sample_record[:200]}...")

        try:
            logger.info(
                f"Putting object to bucket: {self.bucket_name}, key: {full_path}"
            )
            start_time = time.time()
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=full_path,
                Body=jsonl_content.encode("utf-8"),
                ContentType="application/x-ndjson",
            )
            duration = time.time() - start_time
            logger.info(
                f"Successfully uploaded {len(records)} records to {full_path} in {duration:.2f}s"
            )
            return True
        except ClientError as e:
            logger.error(f"Failed to upload data to R2: {e}")
            if hasattr(e, "response") and "Error" in e.response:
                logger.error(f"Error details: {e.response['Error']}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading to R2: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
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


def build_flux_query(
    bucket: str, start_time: datetime.datetime, end_time: datetime.datetime
) -> str:
    """
    Build a Flux query to fetch historical data for validator and evaluator metrics only.

    Args:
        bucket: InfluxDB bucket name
        start_time: Start time for the query
        end_time: End time for the query

    Returns:
        str: Formatted Flux query
    """
    start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Most common/important measurements first for better query optimization
    validator_measurements = [
        "Vvalidator_window_v2",
        "Vvalidator_openskill",
        # "Vvalidator_sync_score",
        "Vvalidator_inactivity",
        "Vvalidator_slash",
        # "Vvalidator_scores",
        "Vtiming",
    ]

    evaluator_measurements = [
        "Ebenchmark_task",  # Most frequent evaluator metric
        "Ebenchmark_metrics",
        "Ebenchmark_summary",
    ]

    # Create efficient OR chain with most frequent measurements first
    all_measurements = validator_measurements + evaluator_measurements
    measurement_conditions = [f'r["_measurement"] == "{m}"' for m in all_measurements]
    measurement_filter = " or ".join(measurement_conditions)

    return f"""
from(bucket: "{bucket}")
  |> range(start: {start_str}, stop: {end_str})
  |> filter(fn: (r) => r["config_netuid"] == "3")
  |> filter(fn: (r) => r["config_project"] == "templar")
  |> filter(fn: (r) => r["role"] == "validator" or r["role"] == "evaluator")
  |> filter(fn: (r) => {measurement_filter})
"""


def query_influxdb(
    client: InfluxDBClient,
    query: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> Dict[Tuple[str, str, str, str], List[Dict[str, Any]]]:
    """
    Query InfluxDB for data.

    This is a synchronous function intended to be run in a ThreadPoolExecutor.

    Args:
        client: InfluxDB client
        query: Flux query string
        start_time: Start time for the query (for logging)
        end_time: End time for the query (for logging)

    Returns:
        Dictionary of record groups, keyed by (role, uid, measurement, field)
    """
    try:
        logger.info(
            f"Executing InfluxDB query for {start_time.date()} to {end_time.date()}"
        )
        logger.debug(f"Query execution started at {datetime.datetime.now()}")

        query_api = client.query_api()
        logger.debug("Query API initialized, sending query...")

        query_start = time.time()
        tables = query_api.query(query)
        query_duration = time.time() - query_start

        logger.info(f"Query completed in {query_duration:.2f}s, processing results...")
        logger.debug(
            f"Number of tables returned: {len(tables) if hasattr(tables, '__len__') else 'unknown'}"
        )

        record_groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}

        reserved = {
            "_start",
            "_stop",
            "result",
            "table",
        }

        total_records = 0
        for table in tables:
            for record in table.records:
                values = record.values

                role = values.get("role", "unknown")
                uid = values.get("uid", "unknown")
                measurement = values.get("_measurement", "unknown")
                field = record.get_field() or "unknown"

                group_key = (role, uid, measurement, field)
                if group_key not in record_groups:
                    record_groups[group_key] = []

                record_dict = {k: v for k, v in values.items() if k not in reserved}

                if isinstance(record_dict.get("_time"), datetime.datetime):
                    record_dict["_time"] = record_dict["_time"].isoformat()

                record_dict["_field"] = field
                record_dict["_value"] = record.get_value()

                record_groups[group_key].append(record_dict)
                total_records += 1

        logger.info(
            f"Retrieved {total_records} records across {len(record_groups)} groups for {start_time.date()} to {end_time.date()}"
        )
        return record_groups

    except InfluxDBError as e:
        logger.error(f"InfluxDB error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching data: {e}")
        logger.error(traceback.format_exc())
        raise


async def process_and_upload(
    executor: concurrent.futures.ThreadPoolExecutor,
    client: InfluxDBClient,
    r2_storage: R2Storage,
    query: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> bool:
    """
    Process InfluxDB query and upload results to R2.

    Args:
        executor: ThreadPoolExecutor for running the query
        client: InfluxDB client
        r2_storage: R2 storage instance
        query: Flux query string
        start_time: Start time for the query
        end_time: End time for the query

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        loop = asyncio.get_running_loop()
        record_groups = await loop.run_in_executor(
            executor, lambda: query_influxdb(client, query, start_time, end_time)
        )

        if not record_groups:
            logger.info(f"No data found for {start_time.date()} to {end_time.date()}")
            return True

        tasks = []
        for (role, uid, measurement, field), records in record_groups.items():
            if records:
                logger.info(
                    f"Uploading {len(records)} records for role={role}, uid={uid}, measurement={measurement}, field={field}"
                )
                task = r2_storage.upload_data(records, role, uid, measurement, field)
                tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in upload task {i}: {result}")
                    return False

            return all(
                result for result in results if not isinstance(result, Exception)
            )

        return True

    except Exception as e:
        logger.error(f"Error processing and uploading data: {e}")
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
    """Main entry point for the InfluxDB historical backup to R2."""
    parser = argparse.ArgumentParser(
        description="InfluxDB Historical Data Backup to R2"
    )

    parser.add_argument(
        "--influxdb-url",
        default=os.getenv("SOURCE_INFLUXDB_URL", ""),
        help="InfluxDB URL",
    )
    parser.add_argument(
        "--influxdb-token",
        default=os.getenv("SOURCE_INFLUXDB_TOKEN", ""),
        help="InfluxDB token",
    )
    parser.add_argument(
        "--influxdb-org",
        default=os.getenv("SOURCE_INFLUXDB_ORG", "tplr"),
        help="InfluxDB organization",
    )
    parser.add_argument(
        "--influxdb-bucket",
        default=os.getenv("SOURCE_INFLUXDB_BUCKET", "tplr"),
        help="InfluxDB bucket",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for backup (YYYY-MM-DD format). If not provided, will start from earliest data or checkpoint.",
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

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if not args.r2_access_key_id or not args.r2_secret_access_key:
        logger.error(
            "R2 credentials must be provided via --r2-access-key-id and --r2-secret-access-key arguments or R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY environment variables"
        )
        sys.exit(1)

    # Determine date range
    end_date = datetime.datetime.now(datetime.timezone.utc)
    if args.end_date:
        try:
            end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").replace(
                tzinfo=datetime.timezone.utc
            )
        except ValueError:
            logger.error("Invalid end date format. Use YYYY-MM-DD.")
            sys.exit(1)

    start_date = None
    if args.resume:
        start_date = load_checkpoint()
        if start_date:
            logger.info(f"Resuming from checkpoint: {start_date.date()}")

    if not start_date and args.start_date:
        try:
            start_date = datetime.datetime.strptime(
                args.start_date, "%Y-%m-%d"
            ).replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            logger.error("Invalid start date format. Use YYYY-MM-DD.")
            sys.exit(1)

    if not start_date:
        # Default to 30 days ago if no start date provided
        start_date = end_date - relativedelta(days=30)
        logger.info(
            f"No start date provided, defaulting to 30 days ago: {start_date.date()}"
        )

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
            f"Starting InfluxDB historical backup from {start_date.date()} to {end_date.date()}"
        )

        # Configure HTTP timeout for large queries
        http_timeout = Timeout(connect=30.0, read=args.timeout, total=args.timeout + 30)

        client = InfluxDBClient(
            url=args.influxdb_url,
            token=args.influxdb_token,
            org=args.influxdb_org,
            timeout=args.timeout * 1000,  # Convert to milliseconds
        )

        # Override the client's HTTP configuration
        client.api_client.rest_client.pool_manager.clear()
        import urllib3

        client.api_client.rest_client.pool_manager = urllib3.PoolManager(
            timeout=http_timeout,
            retries=urllib3.Retry(
                total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504]
            ),
        )

        # Test connection with a simple query first
        if args.test_connection:
            logger.info("Testing InfluxDB connection with a simple query...")
            try:
                test_query = f'from(bucket: "{args.influxdb_bucket}") |> range(start: -1h) |> limit(n: 1)'
                test_api = client.query_api()
                test_result = test_api.query(test_query)
                logger.info(
                    f"Connection test successful! Found {len(test_result)} tables."
                )
                return
            except Exception as e:
                logger.error(f"Connection test failed: {e}")
                return

        # Generate time chunks
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

                    flux_query = build_flux_query(
                        args.influxdb_bucket, chunk_start, chunk_end
                    )
                    logger.debug(f"Using Flux query:\n{flux_query}")

                    start_time = time.time()
                    success = await process_and_upload(
                        executor, client, r2_storage, flux_query, chunk_start, chunk_end
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
            # Clean up checkpoint file on successful completion
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
                logger.info("Backup completed successfully, checkpoint file removed")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if "client" in locals():
            client.close()
            logger.info("Closed InfluxDB connection")

        logger.info("InfluxDB Historical Backup shutting down")


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())
