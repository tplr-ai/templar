#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "influxdb-client",
#     "asyncio",
#     "uvloop",
# ]
# [tool.uv]
# prerelease = "allow"
# ///


import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import List, Set

import uvloop
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import WriteOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Destination (new) InfluxDB
DEST_URL = os.getenv(
    "DEST_INFLUXDB_URL",
    "",
)
DEST_TOKEN = os.getenv(
    "DEST_INFLUXDB_TOKEN",
    "",
)
DEST_ORG = os.getenv("DEST_INFLUXDB_ORG", "")
DEST_BUCKET = os.getenv("DEST_INFLUXDB_BUCKET", "")

BACKUP_DIR = os.getenv("BACKUP_DIR", "backup_influx_data")
CHECKPOINT_FILE = os.getenv("UPLOAD_CHECKPOINT_FILE", ".upload_checkpoint")
BATCH_SIZE = 5_000


def load_checkpoint() -> Set[str]:
    """Load checkpoint file to track processed files"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as cp:
            return {line.strip() for line in cp if line.strip()}
    return set()


def save_checkpoint(file_path: str) -> None:
    """Save processed file to checkpoint"""
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as cp:
        cp.write(file_path + "\n")


def extract_date_from_path(file_path: str) -> datetime:
    """
    Extract date components from file path and filename
    Returns a datetime object for sorting
    """
    path_parts = file_path.split(os.sep)
    year = None
    month = None
    day = None
    hour = None

    for part in path_parts:
        if part.startswith("year="):
            year = int(part[5:])
        elif part.startswith("month="):
            month = int(part[6:])
        elif part.startswith("day="):
            day = int(part[4:])
        elif part.startswith("hour="):
            hour = int(part[5:])

    filename = os.path.basename(file_path)
    minute = 0

    if "_" in filename:
        timestamp_part = filename.split("_")[0]
        parts = timestamp_part.split("-")
        if len(parts) >= 5:
            try:
                minute = int(parts[4])
            except ValueError:
                pass

    if year is not None and month is not None and day is not None and hour is not None:
        try:
            return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
        except ValueError:
            pass

    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def find_jsonl_files() -> List[str]:
    """
    Find all JSONL files in the backup directory and sort them
    from most recent to oldest based on directory structure
    """
    jsonl_files = []

    for root, _, files in os.walk(BACKUP_DIR):
        for file in files:
            if file.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, file))

    return sorted(jsonl_files, key=extract_date_from_path, reverse=True)


def main():
    processed = load_checkpoint()

    jsonl_files = find_jsonl_files()
    logger.info("Found %d JSONL files to process", len(jsonl_files))

    dest_client = InfluxDBClient(
        url=DEST_URL,
        token=DEST_TOKEN,
        org=DEST_ORG,
    )

    write_opts = WriteOptions(
        batch_size=BATCH_SIZE,
        flush_interval=10_000,
        jitter_interval=2_000,
        retry_interval=5_000,
        max_retries=3,
    )
    write_api = dest_client.write_api(write_options=write_opts)

    batch = []
    total_records = 0

    reserved = {
        "_measurement",
        "_field",
        "_value",
        "_time",
        "result",
        "table",
        "_start",
        "_stop",
        "measurement",
        "field",
        "value",
        "time",
    }

    for file_path in jsonl_files:
        if file_path in processed:
            logger.debug("Skipping already processed file: %s", file_path)
            continue

        try:
            logger.info("Processing file: %s", file_path)

            records = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.error(
                                "Error parsing JSON line in %s: %s", file_path, str(e)
                            )

            if not records:
                logger.warning("No records found in file: %s", file_path)
                save_checkpoint(file_path)
                continue

            file_record_count = 0
            for record in records:
                try:
                    measurement = record.get("measurement")
                    if not measurement:
                        logger.warning("Record missing measurement in %s", file_path)
                        continue

                    pt = Point(measurement)

                    for k, v in record.items():
                        if k not in reserved and v is not None:
                            pt.tag(k, str(v))

                    field = record.get("field")
                    value = record.get("value")
                    if field is None or value is None:
                        logger.warning("Record missing field or value in %s", file_path)
                        continue

                    try:
                        if isinstance(value, (int, float)):
                            pt.field(field, value)
                        elif isinstance(value, bool):
                            pt.field(field, value)
                        elif isinstance(value, str):
                            if value.lower() == "true":
                                pt.field(field, True)
                            elif value.lower() == "false":
                                pt.field(field, False)
                            else:
                                pt.field(field, value)
                        else:
                            pt.field(field, str(value))
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            "Invalid value for field %s in %s: %s (Error: %s)",
                            field,
                            file_path,
                            value,
                            str(e),
                        )
                        continue

                    timestamp = record.get("time")
                    if not timestamp:
                        logger.warning("Record missing timestamp in %s", file_path)
                        continue

                    pt.time(timestamp, WritePrecision.NS)

                    batch.append(pt)
                    total_records += 1
                    file_record_count += 1

                    if len(batch) >= BATCH_SIZE:
                        write_api.write(bucket=DEST_BUCKET, org=DEST_ORG, record=batch)
                        logger.info("Wrote %d points...", len(batch))
                        batch.clear()

                except Exception as e:
                    logger.error("Error processing record in %s: %s", file_path, str(e))

            logger.info(
                "Processed %d records from file %s", file_record_count, file_path
            )
            save_checkpoint(file_path)

        except Exception as e:
            logger.exception("Error processing file %s: %s", file_path, str(e))

    if batch:
        write_api.write(bucket=DEST_BUCKET, org=DEST_ORG, record=batch)
        logger.info("Wrote final %d points.", len(batch))

    write_api.flush()
    logger.info("Upload complete: %d points transferred.", total_records)

    logger.info("Pausing for 30 seconds before closing connections...")
    time.sleep(30)
    logger.info("Resuming after pause.")

    dest_client.close()


if __name__ == "__main__":
    uvloop.install()
    main()
