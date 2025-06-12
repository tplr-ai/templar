#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "influxdb-client",
#     "asyncio",
#     "uvloop",
#     "boto3",
#     "aiohttp",
# ]
# [tool.uv]
# prerelease = "allow"
# ///

"""
InfluxDB to R2 Data Archiver

This script connects to InfluxDB, queries the most recent data (default: 30 minutes),
and uploads the results to Cloudflare R2 storage in a time-partitioned manner.

Data is stored with the path pattern:
influxdb_data/role={role}/uid={uid}/_measurement={_measurement}/_field={_field}/year=%Y/month=%m/day=%d/hour=%H/{timestamp}_{auto_increment_id}.jsonl
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
from typing import Any, Dict, List, Tuple

import boto3
import uvloop
from botocore.exceptions import ClientError
from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("influx-archiver")


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
            config=boto3.session.Config(
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


def build_flux_query(bucket: str, minutes: int = 30) -> str:
    """
    Build a Flux query to fetch the latest data.

    Args:
        bucket: InfluxDB bucket name
        minutes: Number of minutes to look back

    Returns:
        str: Formatted Flux query
    """
    end_time = datetime.datetime.now(datetime.timezone.utc)
    start_time = end_time - datetime.timedelta(minutes=minutes)

    start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    return f"""
from(bucket: "{bucket}")
  |> range(start: {start_str}, stop: {end_str})
  |> filter(fn: (r) => r["config_netuid"] == "3")
  |> filter(fn: (r) => r["config_project"] == "templar")
"""


def query_influxdb(
    client: InfluxDBClient, query: str
) -> Dict[Tuple[str, str, str, str], List[Dict[str, Any]]]:
    """
    Query InfluxDB for data.

    This is a synchronous function intended to be run in a ThreadPoolExecutor.

    Args:
        client: InfluxDB client
        query: Flux query string

    Returns:
        Dictionary of record groups, keyed by (role, uid, measurement, field)
    """
    try:
        logger.info("Executing InfluxDB query")
        query_api = client.query_api()
        tables = query_api.query(query)

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
                field = record.get_field() or "unknown"  # Get field from record

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
            f"Retrieved {total_records} records across {len(record_groups)} groups"
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
) -> bool:
    """
    Process InfluxDB query and upload results to R2.

    Args:
        executor: ThreadPoolExecutor for running the query
        client: InfluxDB client
        r2_storage: R2 storage instance
        query: Flux query string

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        loop = asyncio.get_running_loop()
        record_groups = await loop.run_in_executor(
            executor, lambda: query_influxdb(client, query)
        )

        if not record_groups:
            logger.info("No data found for the specified time range")
            return True

        tasks = []
        for (role, uid, measurement, field), records in record_groups.items():
            if records:
                logger.info(
                    f"Uploading {len(records)} records for role={role}, uid={uid}, measurement={measurement}, field={field}"
                )
                # Updated to pass field to upload_data
                task = r2_storage.upload_data(records, role, uid, measurement, field)
                tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in upload task {i}: {result}")
                    return False

            return all(results)

        return True

    except Exception as e:
        logger.error(f"Error processing and uploading data: {e}")
        logger.error(traceback.format_exc())
        return False


async def main():
    """Main entry point for the InfluxDB to R2 archiver."""
    parser = argparse.ArgumentParser(description="InfluxDB to R2 Data Archiver")

    parser.add_argument(
        "--influxdb-url",
        default=os.getenv(
            "SOURCE_INFLUXDB_URL",
            "",
        ),
        help="InfluxDB URL",
    )
    parser.add_argument(
        "--influxdb-token",
        default=os.getenv(
            "SOURCE_INFLUXDB_TOKEN",
            "",
        ),
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
        "--minutes", type=int, default=30, help="Minutes of data to fetch (default: 30)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,  # 10 minutes
        help="Connection timeout in seconds (default: 600)",
    )

    parser.add_argument("--r2-bucket", required=True, help="R2 bucket name")
    parser.add_argument("--r2-endpoint", required=True, help="R2 endpoint URL")
    parser.add_argument("--r2-region", default="auto", help="R2 region")

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--interval",
        type=int,
        default=1800,  # 30 minutes in seconds
        help="Interval between runs in seconds (default: 1800)",
    )
    parser.add_argument("--run-once", action="store_true", help="Run once and exit")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of worker threads (default: 5)",
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    aws_access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")

    if not aws_access_key_id or not aws_secret_access_key:
        logger.error(
            "R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY environment variables must be set"
        )
        sys.exit(1)

    r2_storage = R2Storage(
        bucket_name=args.r2_bucket,
        endpoint_url=args.r2_endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=args.r2_region,
    )

    flux_query = build_flux_query(args.influxdb_bucket, args.minutes)
    logger.info(f"Using Flux query:\n{flux_query}")

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
            f"Starting InfluxDB to R2 Data Archiver, fetching last {args.minutes} minutes of data"
        )

        client = InfluxDBClient(
            url=args.influxdb_url,
            token=args.influxdb_token,
            org=args.influxdb_org,
            timeout=args.timeout,
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            while running:
                try:
                    start_time = time.time()
                    logger.info(
                        f"Beginning data fetch at {datetime.datetime.now().isoformat()}"
                    )

                    success = await process_and_upload(
                        executor, client, r2_storage, flux_query
                    )

                    duration = time.time() - start_time
                    if success:
                        logger.info(f"Completed archive run in {duration:.2f} seconds")
                    else:
                        logger.warning(
                            f"Archive run completed with errors in {duration:.2f} seconds"
                        )

                    if args.run_once:
                        logger.info("Run-once mode enabled, exiting")
                        break

                    next_run = args.interval - duration
                    if next_run > 0:
                        logger.info(f"Waiting {next_run:.2f} seconds until next run")
                        wait_chunk = 5.0  # 5 seconds
                        for _ in range(int(next_run / wait_chunk)):
                            if not running:
                                break
                            await asyncio.sleep(wait_chunk)
                        if running and next_run % wait_chunk > 0:
                            await asyncio.sleep(next_run % wait_chunk)

                except Exception as e:
                    logger.error(f"Error during archive run: {e}")
                    logger.error(traceback.format_exc())

                    if args.run_once:
                        sys.exit(1)

                    logger.info("Waiting 60 seconds before retrying")
                    await asyncio.sleep(60)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if "client" in locals():
            client.close()
            logger.info("Closed InfluxDB connection")

        logger.info("InfluxDB to R2 Data Archiver shutting down")


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())
