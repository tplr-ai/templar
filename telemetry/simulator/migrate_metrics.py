#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "influxdb-client",
#     "asyncio",
# ]
# [tool.uv]
# prerelease = "allow"
# ///


import logging
import os
import time
from datetime import timezone

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import WriteOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Source (old) InfluxDB
SOURCE_URL = os.getenv(
    "SOURCE_INFLUXDB_URL",
    "",
)
SOURCE_TOKEN = os.getenv(
    "SOURCE_INFLUXDB_TOKEN",
    "",
)
SOURCE_ORG = os.getenv("SOURCE_INFLUXDB_ORG", "tplr")
SOURCE_BUCKET = os.getenv("SOURCE_INFLUXDB_BUCKET", "tplr")

# Destination (new) InfluxDB
DEST_URL = os.getenv(
    "DEST_INFLUXDB_URL",
    "",
)
DEST_TOKEN = os.getenv(
    "DEST_INFLUXDB_TOKEN",
    "",
)
DEST_ORG = os.getenv("DEST_INFLUXDB_ORG", "tplr")
DEST_BUCKET = os.getenv("DEST_INFLUXDB_BUCKET", "test")

TIME_RANGE = os.getenv("TIME_RANGE", "-60d")
VERSIONS_RAW = os.getenv("VERSIONS", "0.2.63,0.2.67,0.2.69,0.2.71,0.2.73,0.2.74,0.2.75")
VERSIONS = [v.strip() for v in VERSIONS_RAW.split(",") if v.strip()]
CONFIG_NETUID = os.getenv("CONFIG_NETUID", "3")


def build_flux_query():
    version_filters = " or ".join([f'r["version"] == "{v}"' for v in VERSIONS])

    return f"""
from(bucket: "{SOURCE_BUCKET}")
  |> range(start: {TIME_RANGE})
  |> filter(fn: (r) => r["_measurement"] == "Ebenchmark_metrics" or r["_measurement"] == "Ebenchmark_task")
  |> filter(fn: (r) => r["role"] == "evaluator")
  |> filter(fn: (r) => {version_filters})
  |> filter(fn: (r) => r["config_netuid"] == "{CONFIG_NETUID}")
  |> filter(fn: (r) => r["config_project"] == "templar")
  |> sort(columns: ["_time"], desc: true)
"""


def main():
    flux = build_flux_query()
    logger.info("Running Flux query:\n%s", flux.strip())

    src_client = InfluxDBClient(
        url=SOURCE_URL,
        token=SOURCE_TOKEN,
        org=SOURCE_ORG,
    )
    query_api = src_client.query_api()

    dest_client = InfluxDBClient(
        url=DEST_URL,
        token=DEST_TOKEN,
        org=DEST_ORG,
    )
    write_opts = WriteOptions(
        batch_size=5_000,
        flush_interval=10_000,
        jitter_interval=2_000,
        retry_interval=5_000,
        max_retries=3,
    )
    write_api = dest_client.write_api(write_options=write_opts)

    tables = query_api.query(flux)

    batch = []
    BATCH_SIZE = 5_000

    reserved = {
        "_measurement",
        "_field",
        "_value",
        "_time",
        "result",
        "table",
        "_start",
        "_stop",
    }

    total = 0
    for table in tables:
        for record in table.records:
            pt = Point(record.get_measurement())

            for k, v in record.values.items():
                if k in reserved:
                    continue
                pt.tag(k, str(v))

            pt.field(record.get_field(), float(record.get_value()))
            pt.time(
                record.get_time().replace(tzinfo=timezone.utc).isoformat(),
                WritePrecision.NS,
            )

            batch.append(pt)
            total += 1

            if len(batch) >= BATCH_SIZE:
                write_api.write(bucket=DEST_BUCKET, org=DEST_ORG, record=batch)
                logger.info("Wrote %d points...", len(batch))
                batch.clear()

    if batch:
        write_api.write(bucket=DEST_BUCKET, org=DEST_ORG, record=batch)
        logger.info("Wrote final %d points.", len(batch))

    write_api.flush()
    logger.info("Migration complete: %d points transferred.", total)

    logger.info("Pausing for 30 seconds before closing connections...")
    time.sleep(30)
    logger.info("Resuming after pause.")

    src_client.close()
    dest_client.close()


if __name__ == "__main__":
    main()
