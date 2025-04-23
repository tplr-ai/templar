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
import os
import random
from datetime import datetime

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# InfluxDB Configuration
INFLUXDB_URL = os.getenv(
    "INFLUXDB_URL",
    "https://cddnkvuk6l-tzxeth774u3fvf.timestream-influxdb.us-east-2.on.aws:8086",
)
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "")
INFLUXDB_ORG = "tplr"
INFLUXDB_BUCKET = "tplr"

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)


async def simulate_metrics():
    while True:
        timestamp = datetime.utcnow()

        value = random.randint(0, 42)

        point = (
            Point("test").tag("env", "local").field("test_field", value).time(timestamp)
        )

        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)

        print(f"Logged metrics at {timestamp.isoformat()}: test_field {value}")

        await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(simulate_metrics())
