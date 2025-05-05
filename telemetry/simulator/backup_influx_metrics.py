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


import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Set, Tuple

from influxdb_client import InfluxDBClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SOURCE_URL: str = os.getenv(
    "SOURCE_INFLUXDB_URL",
    "",
)
SOURCE_TOKEN: str = os.getenv(
    "SOURCE_INFLUXDB_TOKEN",
    "",
)
SOURCE_ORG: str = os.getenv("SOURCE_INFLUXDB_ORG", "tplr")
SOURCE_BUCKET: str = os.getenv("SOURCE_INFLUXDB_BUCKET", "tplr")

timeout_ms: int = int(os.getenv("INFLUX_TIMEOUT_MS", "900000"))
PARALLEL_QUERIES: int = int(os.getenv("PARALLEL_QUERIES", "5"))
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "backup_influx_data")
CHECKPOINT_FILE: str = os.getenv("CHECKPOINT_FILE", ".checkpoint")

ROLES: List[str] = ["validator", "aggregator"]
NETUID: str = os.getenv("CONFIG_NETUID", "3")
PROJECT: str = os.getenv("CONFIG_PROJECT", "templar")
MEASUREMENTS: List[str] = [
    "Vvalidator_window_v2",
    "Aaggregation_step",
    "Vvalidator_openskill",
    "Vvalidator_sync_score",
]

_reserved_keys = {
    "_measurement",
    "_field",
    "_value",
    "_time",
    "result",
    "table",
    "_start",
    "_stop",
}


def get_intervals() -> List[Tuple[datetime, datetime]]:
    """
    Generate 30-min intervals starting from 3 days ago, going back to 30 days ago.
    Returns a list of (start, stop) tuples in descending order (most recent first).
    """
    now = datetime.now(timezone.utc)
    newest = now - timedelta(days=3)
    oldest = now - timedelta(days=30)
    intervals: List[Tuple[datetime, datetime]] = []
    end_time = newest

    while end_time > oldest:
        start_time = max(oldest, end_time - timedelta(minutes=30))
        intervals.append((start_time, end_time))
        end_time = start_time

    return intervals


def build_flux_query(start: datetime, stop: datetime) -> str:
    """
    Build the Flux query for a given time window.
    """
    start_iso = start.isoformat()
    stop_iso = stop.isoformat()

    meas_filter = " or ".join([f'r["_measurement"] == "{m}"' for m in MEASUREMENTS])
    role_filter = " or ".join([f'r["role"] == "{r}"' for r in ROLES])

    return f"""
from(bucket: \"{SOURCE_BUCKET}\")
  |> range(start: {start_iso}, stop: {stop_iso})
  |> filter(fn: (r) => {meas_filter})
  |> filter(fn: (r) => {role_filter})
  |> filter(fn: (r) => r["config_netuid"] == \"{NETUID}\")
  |> filter(fn: (r) => r["config_project"] == \"{PROJECT}\")
  |> sort(columns: ["_time"], desc: true)
"""


async def process_interval(
    start: datetime,
    stop: datetime,
    client: InfluxDBClient,
    processed: Set[str],
    sem: asyncio.Semaphore,
) -> None:
    """
    Fetch data for one interval and write JSONL files grouped by (role, uid, measurement, field).
    Uses a semaphore to limit concurrent queries.
    """
    interval_key = start.isoformat()
    if interval_key in processed:
        logger.debug("Skipping already processed interval %s", interval_key)
        return

    async with sem:
        try:
            flux_query = build_flux_query(start, stop)
            logger.info("Running interval %s to %s", start, stop)
            tables = await asyncio.to_thread(client.query_api().query, flux_query)

            groups = {}
            for table in tables:
                for rec in table.records:
                    role = rec.values.get("role")
                    uid = rec.values.get("config_netuid")
                    measurement = rec.get_measurement()
                    field = rec.get_field()
                    key = (role, uid, measurement, field)

                    obj = {
                        "measurement": measurement,
                        "time": rec.get_time().replace(tzinfo=timezone.utc).isoformat(),
                        "field": field,
                        "value": rec.get_value(),
                    }

                    for k, v in rec.values.items():
                        if k not in _reserved_keys and k not in {
                            "role",
                            "config_netuid",
                        }:
                            obj[k] = v

                    groups.setdefault(key, []).append(obj)

            for idx, ((role, uid, meas, field), records) in enumerate(groups.items()):
                year = start.year
                month = f"{start.month:02d}"
                day = f"{start.day:02d}"
                hour = f"{start.hour:02d}"

                dir_path = os.path.join(
                    OUTPUT_DIR,
                    f"role={role}",
                    f"uid={uid}",
                    f"_measurement={meas}",
                    f"_field={field}",
                    f"year={year}",
                    f"month={month}",
                    f"day={day}",
                    f"hour={hour}",
                )
                os.makedirs(dir_path, exist_ok=True)

                timestamp = start.strftime("%y-%m-%d-%H-%M")
                file_name = f"{timestamp}_{idx}.jsonl"
                file_path = os.path.join(dir_path, file_name)

                with open(file_path, "w", encoding="utf-8") as fp:
                    for record in records:
                        fp.write(json.dumps(record) + "\n")

                logger.info("Wrote %d records to %s", len(records), file_path)

            with open(CHECKPOINT_FILE, "a", encoding="utf-8") as cp:
                cp.write(interval_key + "\n")

        except Exception:
            logger.exception("Failed to process interval %s", interval_key)
            return


def load_checkpoint() -> Set[str]:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as cp:
            return {line.strip() for line in cp if line.strip()}
    return set()


async def main() -> None:
    intervals = get_intervals()
    processed = load_checkpoint()

    client = InfluxDBClient(
        url=SOURCE_URL,
        token=SOURCE_TOKEN,
        org=SOURCE_ORG,
        timeout=timeout_ms,
    )

    sem = asyncio.Semaphore(PARALLEL_QUERIES)
    tasks = [process_interval(s, e, client, processed, sem) for s, e in intervals]

    await asyncio.gather(*tasks)
    client.close()
    logger.info("All intervals processed.")


if __name__ == "__main__":
    asyncio.run(main())
