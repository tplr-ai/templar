#!/usr/bin/env python3
# Test script for validator metrics with proper types

import os
import argparse
import tplr
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_validator_metrics():
    """Test validator metrics logging with proper field types"""
    parser = argparse.ArgumentParser(description="Test validator metrics logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.debug:
        tplr.debug()

    # Set environment variable
    os.environ["ENABLE_INFLUXDB"] = "true"
    print(f"ENABLE_INFLUXDB environment variable set to: {os.environ.get('ENABLE_INFLUXDB')}")
    
    # Initialize metrics logger
    metrics_logger = tplr.metrics.MetricsLogger(
        prefix="V",
        uid="1234",
        role="validator",
        group="validator",
        job_type="validation_test",
    )
    
    print(f"Initialized MetricsLogger for validator")
    
    # Log a test metric with proper types
    metrics_logger.log(
        measurement="validator_window_v2",
        tags={
            "window": int(100),
            "global_step": int(50),
            "test": "true",
        },
        fields={
            "loss_own_before": float(2.5),
            "loss_own_after": float(2.1),
            "loss_random_before": float(2.6),
            "loss_random_after": float(2.2),
            "loss_own_improvement": float(0.1),
            "loss_random_improvement": float(0.15),
            "current_block": int(12345),
            "evaluated_uids_count": int(5),
            "learning_rate": float(0.001),
            "active_miners_count": int(10),
            "gather_success_rate": float(95.5),
            "window_total_time": float(120.5),
            "peer_update_time": float(0.5),
            "gather_time": float(10.0),
            "evaluation_time": float(60.0),
            "model_update_time": float(3.0),
            "total_peers": int(15),
            "total_skipped": int(2),
        },
    )
    
    print("Logged test validator metrics with proper types to InfluxDB.")
    print("If no errors occurred, the integration is working correctly.")

if __name__ == "__main__":
    test_validator_metrics()