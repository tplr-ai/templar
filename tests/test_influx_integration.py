"""
Integration test for InfluxDB metrics from miner.
This test simulates the miner flow and sends realistic metrics to InfluxDB.
NOTE: This test is meant to be run manually and should be skipped in CI.
"""

import argparse
import asyncio
import json
import os
import random
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

import tplr

# Skip this test in CI environments
pytestmark = pytest.mark.skipif(
    os.environ.get("E2E_TESTS") != "true",
    reason="Integration test, skipped in CI environment",
)


class MinerSimulator:
    """
    Simulates the Miner class to test metrics reporting to InfluxDB
    without performing actual training or network operations.
    """

    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description="Miner metrics simulator")
        parser.add_argument(
            "--netuid", type=int, default=999, help="Bittensor network UID."
        )
        parser.add_argument(
            "--device", type=str, default="cpu", help="Device to use for mock test"
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument("--trace", action="store_true", help="Enable trace logging")
        parser.add_argument(
            "--test-duration",
            type=int,
            default=60,
            help="Duration of the test in seconds",
        )
        parser.add_argument(
            "--window-interval",
            type=float,
            default=5.0,
            help="Interval between windows in seconds",
        )
        parser.add_argument(
            "--num-peers", type=int, default=10, help="Number of simulated peers"
        )
        parser.add_argument(
            "--enable-loki",
            action="store_true",
            help="Enable Loki logging (disabled by default)",
        )
        parser.add_argument(
            "--uid",
            type=int,
            help="Specific UID to use for testing (random if not specified)",
        )

        # Always parse args from command line when running the script
        config = parser.parse_args()

        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()

        return config

    def __init__(self):
        tplr.logger.info("Starting miner simulator initialization...")

        self.config = MinerSimulator.config()  # type: ignore

        # Mock hparams similar to real miner
        self.hparams = MagicMock()
        self.hparams.batch_size = 32
        self.hparams.sequence_length = 2048
        self.hparams.blocks_per_window = 100
        self.hparams.pages_per_window = 5
        self.hparams.target_chunk = 65536
        self.hparams.topk_compression = 0.1
        self.hparams.learning_rate = 0.001
        self.hparams.time_window_delta_seconds = 60

        self.uid = (
            self.config.uid if self.config.uid is not None else random.randint(0, 100)
        )

        self.current_block = 1000000
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        self.start_window = self.current_window
        self.global_step = 0
        self.window_step = 0
        self.total_tokens_processed = 0
        self.batch_times = []

        self.peers = list(range(self.config.num_peers))
        if self.uid in self.peers:
            self.peers.remove(self.uid)

        self.skipped_uids = random.sample(self.peers, k=min(3, len(self.peers)))
        self.successful_peers = [p for p in self.peers if p not in self.skipped_uids]

        self.metrics_logger = tplr.metrics.MetricsLogger(
            prefix="M",
            uid=self.uid,  # type: ignore
            config=self.config,  # type: ignore
            role="miner",
            group="miner",
            job_type="training",
        )

        self.log_connection_info()

        self.verify_influxdb_connection()

    def log_connection_info(self):
        """Log InfluxDB connection details"""
        tplr.logger.info(f"Initialized miner simulator with UID: {self.uid}")
        tplr.logger.info(f"InfluxDB URL: {self.metrics_logger.client.url}")
        tplr.logger.info(f"InfluxDB bucket: {self.metrics_logger.database}")
        tplr.logger.info(f"InfluxDB org: {self.metrics_logger.org}")

        if os.environ.get("INFLUXDB_HOST"):
            tplr.logger.info(
                f"Using custom INFLUXDB_HOST: {os.environ.get('INFLUXDB_HOST')}"
            )
        if os.environ.get("INFLUXDB_PORT"):
            tplr.logger.info(
                f"Using custom INFLUXDB_PORT: {os.environ.get('INFLUXDB_PORT')}"
            )
        if os.environ.get("INFLUXDB_DATABASE"):
            tplr.logger.info(
                f"Using custom INFLUXDB_DATABASE: {os.environ.get('INFLUXDB_DATABASE')}"
            )
        if os.environ.get("INFLUXDB_ORG"):
            tplr.logger.info(
                f"Using custom INFLUXDB_ORG: {os.environ.get('INFLUXDB_ORG')}"
            )
        if os.environ.get("INFLUXDB_TOKEN"):
            tplr.logger.info("Using custom INFLUXDB_TOKEN from environment")
        else:
            tplr.logger.warning("Using fallback INFLUXDB_TOKEN")

    def verify_influxdb_connection(self):
        """Verify connectivity to InfluxDB"""
        try:
            ping_result = self.metrics_logger.client.ping()
            tplr.logger.info(f"InfluxDB ping successful: {ping_result}")
        except Exception as e:
            tplr.logger.error(f"InfluxDB connection failed: {e}")
            tplr.logger.error("Please check your connection settings and credentials")
            tplr.logger.error("Test will continue but metrics may not be recorded")

    def simulate_gradients_and_weights(self):
        """Simulate gradient and weight values"""
        decay_factor = 0.99**self.global_step
        base_value = random.uniform(0.05, 0.2)

        return {
            "grad_norms": [
                base_value * decay_factor * random.uniform(0.5, 1.5) for _ in range(10)
            ],
            "weight_norms": [random.uniform(0.5, 2.0) for _ in range(10)],
            "momentum_norms": [
                base_value * 0.1 * decay_factor * random.uniform(0.5, 1.5)
                for _ in range(10)
            ],
        }

    def simulate_batch_processing(self):
        """Simulate batch processing and return metrics"""
        n_batches = random.randint(15, 25)

        duration = random.uniform(8.0, 15.0)
        self.batch_times.append(duration)

        tokens_this_window = (
            n_batches * self.hparams.batch_size * self.hparams.sequence_length
        )
        self.total_tokens_processed += tokens_this_window

        tokens_per_second = tokens_this_window / duration

        base_loss = 2.5 * (0.98**self.global_step)  # Exponential decay
        loss = max(0.5, base_loss + random.uniform(-0.2, 0.2))  # Add noise

        return {
            "n_batches": n_batches,
            "duration": duration,
            "loss": loss,
            "tokens_per_second": tokens_per_second,
        }

    def simulate_gather_results(self):
        """Simulate gather results with peer statistics"""
        self.skipped_uids = random.sample(self.peers, k=min(3, len(self.peers)))
        self.successful_peers = [p for p in self.peers if p not in self.skipped_uids]

        success_rate = (
            len(self.successful_peers) / len(self.peers) * 100 if self.peers else 0
        )

        success_rate = min(100, max(70, success_rate + random.uniform(-5, 5)))

        return {
            "skipped_uids": self.skipped_uids,
            "successful_peers": self.successful_peers,
            "success_rate": success_rate,
        }

    def get_learning_rate(self):
        """Simulate learning rate schedule like the real miner"""
        if self.global_step < 10:
            return self.hparams.learning_rate * (0.1 + 0.09 * self.global_step)
        else:
            return self.hparams.learning_rate * (
                0.1 + 0.9 * (1 + np.cos(np.pi * (self.global_step - 10) / 50)) / 2
            )

    async def run(self):
        """Simulate the miner's run method"""
        tplr.logger.info(
            f"Starting miner simulation for {self.config.test_duration} seconds"
        )
        tplr.logger.info(f"Window interval: {self.config.window_interval} seconds")

        start_time = time.time()

        while time.time() - start_time < self.config.test_duration:
            window_start = time.time()
            step_window = self.current_window

            tplr.logger.info(
                f"\n{'-' * 40} Window: {step_window} (Global Step: {self.global_step}) {'-' * 40}"
            )

            # 1. Simulate peer update
            peer_start = time.time()
            # In a real miner this would update peers from the network
            time.sleep(random.uniform(0.1, 0.3))
            peer_time = time.time() - peer_start

            # 2. Simulate training data loading
            data_start = time.time()
            # In a real miner this would load data from R2
            time.sleep(random.uniform(0.3, 0.8))
            data_time = time.time() - data_start

            # 3. Simulate gradient accumulation
            train_start = time.time()
            batch_results = self.simulate_batch_processing()
            time.sleep(random.uniform(1.0, 2.0))
            train_time = time.time() - train_start

            # 4. Simulate gradient compression
            compress_start = time.time()
            gradient_results = self.simulate_gradients_and_weights()
            time.sleep(random.uniform(0.2, 0.5))
            compress_time = time.time() - compress_start

            # 5. Simulate gather operation
            gather_start = time.time()
            gather_results = self.simulate_gather_results()
            time.sleep(random.uniform(0.5, 1.5))
            gather_time = time.time() - gather_start

            # 6. Simulate model update
            update_start = time.time()
            learning_rate = self.get_learning_rate()
            time.sleep(random.uniform(0.2, 0.6))
            update_time = time.time() - update_start

            # Calculate window total time
            window_total_time = (
                peer_time
                + data_time
                + train_time
                + compress_time
                + gather_time
                + update_time
            )

            # Prepare all metrics fields similar to the real miner
            fields = {
                "loss": batch_results["loss"],
                "tokens_per_sec": batch_results["tokens_per_second"],
                "batch_tokens": batch_results["n_batches"],
                "grad_norm_std": np.std(gradient_results["grad_norms"]).item(),
                "mean_weight_norm": sum(gradient_results["weight_norms"])
                / len(gradient_results["weight_norms"]),
                "mean_momentum_norm": sum(gradient_results["momentum_norms"])
                / len(gradient_results["momentum_norms"]),
                "batch_duration": batch_results["duration"],
                "total_tokens": self.total_tokens_processed,
                "active_peers": len(self.peers),
                "effective_batch_size": len(self.peers) * self.hparams.batch_size,
                "learning_rate": learning_rate,
                "mean_grad_norm": sum(gradient_results["grad_norms"])
                / len(gradient_results["grad_norms"]),
                "gather_success_rate": gather_results["success_rate"],
                "max_grad_norm": max(gradient_results["grad_norms"]),
                "min_grad_norm": min(gradient_results["grad_norms"]),
                "gather_peers": json.dumps(self.peers),
                "skipped_peers": json.dumps(gather_results["skipped_uids"]),
                "window_total_time": window_total_time,
                "peer_update_time": peer_time,
                "data_loading_time": data_time,
                "training_time": train_time,
                "compression_time": compress_time,
                "gather_time": gather_time,
                "model_update_time": update_time,
            }

            self.metrics_logger.log(
                measurement="training_step",
                tags={
                    "window": self.current_window,
                    "global_step": self.global_step,
                },
                fields=fields,
            )

            tplr.logger.info(f"Logged metrics for window {self.current_window}")
            tplr.logger.info(
                f"Loss: {fields['loss']:.4f}, Learning rate: {fields['learning_rate']:.6f}"
            )
            tplr.logger.info(
                f"Tokens/sec: {fields['tokens_per_sec']:.1f}, Batch tokens: {fields['batch_tokens']}"
            )
            tplr.logger.info(
                f"Gather success rate: {fields['gather_success_rate']:.1f}%"
            )

            self.global_step += 1
            self.window_step += 1
            self.current_window += 1

            # Wait until next window interval
            elapsed = time.time() - window_start
            wait_time = max(0, self.config.window_interval - elapsed)
            if wait_time > 0:
                tplr.logger.info(f"Waiting {wait_time:.2f}s for next window")
                await asyncio.sleep(wait_time)

        tplr.logger.info(f"Test completed after {self.global_step} windows")

        await self.verify_data_written()

    async def verify_data_written(self):
        """Verify that data was successfully written to InfluxDB"""
        tplr.logger.info("Verifying data was written to InfluxDB...")

        query_api = self.metrics_logger.client.query_api()

        query = f"""
        from(bucket: "{self.metrics_logger.database}")
          |> range(start: -10m)
          |> filter(fn: (r) => r["runtime_id"] == "{tplr.metrics.RUNTIME_ID}")
          |> filter(fn: (r) => r["uid"] == "{self.uid}")
          |> filter(fn: (r) => r["_measurement"] == "Mtraining_step")
          |> count()
        """

        try:
            result = query_api.query(query=query, org=self.metrics_logger.org)

            print(result)

            if not result or len(result) == 0:
                tplr.logger.error("No data found in InfluxDB!")
                return

            count = 0
            for table in result:
                for record in table.records:
                    count = record.get_value()

            if count > 0:
                tplr.logger.info(
                    f"Successfully verified data in InfluxDB: {count} points found"
                )
            else:
                tplr.logger.warning("No data points found in InfluxDB query results")

        except Exception as e:
            tplr.logger.error(f"Error verifying InfluxDB data: {e}")
            tplr.logger.error(
                "This may be due to permission issues or incorrect credentials"
            )


@pytest.mark.manual_test
def test_influxdb_integration():
    """
    Manual integration test to verify InfluxDB metrics from miner.

    This test:
    1. Creates a simulator that mimics the real miner's metrics flow
    2. Generates realistic metrics similar to production
    3. Sends metrics to the configured InfluxDB instance using the same MetricsLogger
    4. Verifies data was written successfully

    Run with: pytest -xvs tests/test_influx_integration.py

    Optional arguments (when running directly):
    - --uid: Specify a particular UID to use for the test (otherwise a random one is selected)
    - --test-duration: Duration of the test in seconds (default: 60)
    - --window-interval: Interval between windows in seconds (default: 5.0)
    - --num-peers: Number of simulated peers (default: 10)

    Environment variables:
    - INFLUXDB_HOST: Override the default InfluxDB host
    - INFLUXDB_PORT: Override the default InfluxDB port
    - INFLUXDB_DATABASE: Override the default InfluxDB database/bucket
    - INFLUXDB_ORG: Override the default InfluxDB organization
    - INFLUXDB_TOKEN: Override the default InfluxDB token
    """
    tplr.logger.info("Starting InfluxDB integration test")

    simulator = MinerSimulator()
    asyncio.run(simulator.run())


if __name__ == "__main__":
    test_influxdb_integration()
