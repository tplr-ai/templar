#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "influxdb-client",
#     "asyncio",
#     "numpy",
#     "psutil",
#     "torch"
# ]
# [tool.uv]
# prerelease = "allow"
# ///

import asyncio
import os
import random
from datetime import datetime

import psutil
import torch
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# InfluxDB Configuration
INFLUXDB_URL = os.getenv(
    "INFLUXDB_URL",
    "https://pliftu8n85-tzxeth774u3fvf.timestream-influxdb.us-east-2.on.aws:8086",
)
INFLUXDB_TOKEN = os.getenv(
    "INFLUXDB_TOKEN",
    "lTRclLtRXOJWGOB-vr1mhtp5SholImgBH705pMgK1_0sCzTzAXivhd4gPwJhRoK6HLRvG8cxjhOTEy1hlm4D3Q==",
)
INFLUXDB_ORG = "tplr"
INFLUXDB_BUCKET = "tplr"

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)


async def simulate_metrics():
    while True:
        timestamp = datetime.utcnow()

        metrics = {
            "active_peers": random.randint(15, 40),  # int
            "gather_success_rate": random.uniform(80.0, 99.0),  # float
            "loss": random.uniform(0.1, 2.0),  # float
            "tokens_per_sec": random.uniform(1500.0, 2000.0),  # float
            "batch_time": random.uniform(9.0, 11.0),  # float
            "grad_step_time": random.uniform(0.5, 2.0),  # float
            "optimizer_state_size": random.randint(100, 200),  # int
            "batch_size": random.randint(45, 55),  # int
            "gpu_utilization": random.uniform(50.0, 90.0),  # float
            "sys_cpu_usage": float(psutil.cpu_percent()),  # float
            "sys_mem_used": float(psutil.virtual_memory().used / (1024**2)),  # float
            "sys_mem_total": int(
                psutil.virtual_memory().total / (1024**2)
            ),  # <-- fixed to int
            "gpu_mem_allocated_mb": float(torch.cuda.memory_allocated() / (1024**2))
            if torch.cuda.is_available()
            else 0.0,  # float
            "gpu_mem_cached_mb": float(torch.cuda.memory_reserved() / (1024**2))
            if torch.cuda.is_available()
            else 0.0,  # float
            "gpu_mem_total_mb": int(
                torch.cuda.get_device_properties(0).total_memory / (1024**2)
            )
            if torch.cuda.is_available()
            else 0,  # int
            "gpu_mem_segments": random.randint(500, 1000),  # int
        }

        point = (
            Point("Mtraining_step")
            .tag("uid", "miner-001")
            .tag("role", "miner")
            .tag("version", "0.1.0")
            .time(timestamp)
        )

        for field, value in metrics.items():
            if isinstance(value, int):
                point.field(field, value)
            else:
                point.field(field, float(value))

        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)

        print(f"Logged metrics at {timestamp.isoformat()}: {metrics}")

        await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(simulate_metrics())
