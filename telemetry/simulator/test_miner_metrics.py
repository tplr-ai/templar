#!/usr/bin/env python3
# Test script for miner metrics with proper types

import os
import argparse
import tplr
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_miner_metrics():
    """Test miner metrics logging with proper field types"""
    parser = argparse.ArgumentParser(description="Test miner metrics logging")
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
        prefix="M",
        uid="1234",
        role="miner",
        group="miner",
        job_type="mining_test",
    )

    print("Initialized MetricsLogger for miner")

    # Log a test metric with proper types
    metrics_logger.log(
        measurement="training_step_v2",
        tags={
            "window": 100,
            "global_step": 50,
            "test": "true",
        },
        fields={
            "loss": 0.5,
            "tokens_per_sec": 1000.5,
            "batch_tokens": 32,
            "grad_norm_std": 0.01,
            "mean_weight_norm": 0.5,
            "mean_momentum_norm": 0.3,
            "batch_duration": 5.2,
            "total_tokens": 1000,
            "active_peers": 10,
            "effective_batch_size": 320,
            "learning_rate": 0.001,
            "mean_grad_norm": 0.2,
            "gather_success_rate": 95.5,
            "max_grad_norm": 0.3,
            "min_grad_norm": 0.1,
            "gather_peers": json.dumps([1, 2, 3, 4, 5]),
            "skipped_peers": json.dumps([6, 7]),
            "window_total_time": 120.5,
            "peer_update_time": 0.5,
            "data_loading_time": 2.0,
            "training_time": 90.0,
            "compression_time": 5.0,
            "gather_time": 10.0,
            "model_update_time": 3.0,
        },
    )

    print("Logged test metrics with proper types to InfluxDB.")
    print("If no errors occurred, the integration is working correctly.")


if __name__ == "__main__":
    test_miner_metrics()
