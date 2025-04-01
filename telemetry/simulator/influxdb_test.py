#!/usr/bin/env python3
# Test script for InfluxDB integration

import os
import argparse
import tplr


def test_influxdb_integration():
    """Test InfluxDB integration by logging simple metrics"""
    parser = argparse.ArgumentParser(description="Test InfluxDB integration")
    parser.add_argument(
        "--role",
        type=str,
        choices=["miner", "validator"],
        default="miner",
        help="Role to test",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        tplr.debug()

    # Set environment variable
    os.environ["ENABLE_INFLUXDB"] = "true"
    print(
        f"ENABLE_INFLUXDB environment variable set to: {os.environ.get('ENABLE_INFLUXDB')}"
    )

    # Initialize metrics logger
    metrics_logger = tplr.metrics.MetricsLogger(
        prefix="T" if args.role == "miner" else "T",
        uid="1234",
        role=args.role,
        group=args.role,
        job_type=f"{args.role}_test",
    )

    print(f"Initialized MetricsLogger for {args.role}")

    # Log a simple metric
    metrics_logger.log(
        measurement="test_influxdb",
        tags={
            "window": 100,
            "global_step": 50,
            "test": "true",
        },
        fields={
            "value": 42.0,
            "count": 1,
            "test_string": "test",
        },
    )

    print(
        "Logged test metric to InfluxDB. If no errors occurred, the integration is working."
    )


if __name__ == "__main__":
    test_influxdb_integration()
