#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "aiohttp>=3.9.1",
#   "websockets>=12.0",
#   "boto3>=1.34.7",
# ]
# ///

"""
Loki to R2 Log Collector

This script connects to Loki's streaming API and forwards logs to Cloudflare R2 storage
in a time-partitioned manner. It maintains a persistent WebSocket connection to Loki
and processes incoming log streams, storing them as JSONL files in the R2 bucket.

Logs are stored with the path pattern:
logs/version=${version}/year=%Y/month=%m/day=%d/hour=%H/${service}_${uid}_%{hex_random}.jsonl
"""

import argparse
import asyncio
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
from urllib.parse import urlencode

import aiohttp
import boto3
from botocore.exceptions import ClientError
from websockets.exceptions import ConnectionClosed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("log-collector")


class R2Storage:
    """Handles uploading logs to Cloudflare R2 storage."""

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

        # Create a session with the credentials
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

        # Create a client with the session and proper config
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

    async def upload_logs(
        self, logs: List[Dict], service: str, uid: str, version: str
    ) -> bool:
        """
        Upload logs to R2 storage.

        Args:
            logs: List of log entries to upload
            service: Service name for path construction
            uid: UID for path construction
            version: Version for path construction

        Returns:
            bool: True if upload successful, False otherwise
        """
        if not logs:
            return True

        timestamp = datetime.datetime.now()
        if logs and "timestamp" in logs[0]:
            try:
                timestamp = datetime.datetime.fromisoformat(logs[0]["timestamp"])
            except (ValueError, TypeError):
                pass

        path_template = f"logs/version={version}/year={timestamp.year}/month={timestamp.month:02d}/day={timestamp.day:02d}/hour={timestamp.hour:02d}"

        random_hex = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
        filename = f"{service}_{uid}_{random_hex}.jsonl"

        full_path = f"{path_template}/{filename}"

        jsonl_content = "\n".join(json.dumps(log) for log in logs)

        logger.info(f"Uploading {len(logs)} logs to R2 as {full_path}")

        # Log a sample of the first log entry for debugging
        if logs and logger.level <= logging.DEBUG:
            sample_log = json.dumps(logs[0])
            logger.debug(f"Sample log entry: {sample_log[:200]}...")

        try:
            # Upload to R2
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
                f"Successfully uploaded {len(logs)} logs to {full_path} in {duration:.2f}s"
            )
            return True
        except ClientError as e:
            logger.error(f"Failed to upload logs to R2: {e}")
            if hasattr(e, "response") and "Error" in e.response:
                logger.error(f"Error details: {e.response['Error']}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading to R2: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            return False


class LokiClient:
    """Client for connecting to Loki and streaming logs."""

    def __init__(
        self,
        loki_url: str,
        query: str = '{job=~".+"}',
        delay_for: int = 0,
        limit: int = 1000,
        buffer_size: int = 100,
        buffer_timeout: int = 60,
    ):
        """
        Initialize Loki client.

        Args:
            loki_url: Base URL for Loki API
            query: LogQL query to filter logs
            delay_for: Delay in seconds to let slow loggers catch up
            limit: Max number of entries to return
            buffer_size: Max number of logs to buffer before uploading
            buffer_timeout: Max time to buffer logs before uploading (seconds)
        """
        self.loki_url = loki_url.rstrip("/")
        self.query = query
        self.delay_for = min(delay_for, 5)  # Max allowed is 5 seconds
        self.limit = limit
        self.buffer_size = buffer_size
        self.buffer_timeout = buffer_timeout
        self.running = False
        self.log_buffer: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        self.last_flush_time = time.time()
        logger.info(f"Initialized Loki client with URL: {loki_url}")

    async def check_loki_health(self) -> bool:
        """
        Check if Loki is healthy by calling the /ready endpoint.

        Returns:
            bool: True if Loki is healthy, False otherwise
        """
        try:
            ready_url = f"{self.loki_url}/ready"
            logger.debug(f"Checking Loki health at: {ready_url}")

            async with aiohttp.ClientSession() as session:
                async with session.get(ready_url) as response:
                    is_healthy = response.status == 200

                    if is_healthy:
                        logger.debug("Loki is healthy")
                    else:
                        content = await response.text()
                        logger.error(
                            f"Loki health check failed with status {response.status}: {content}"
                        )

                    return is_healthy
        except Exception as e:
            logger.error(f"Failed to check Loki health: {e}")
            return False

    def _build_tail_url(self) -> str:
        """
        Build URL for the Loki tail endpoint with query parameters.

        Returns:
            str: Formatted URL for WebSocket connection
        """
        # Start time defaulting to 1 hour ago
        start_time = int(time.time() - 3600) * 1_000_000_000  # Convert to nanoseconds

        # Use a very simple query to match all logs
        query = "{}"
        if self.query and self.query != "{}":
            logger.warning(
                f"Using simplified query format: {query} instead of {self.query}"
            )

        params = {
            "query": query,
            "delay_for": str(self.delay_for),
            "limit": str(self.limit),
            "start": str(start_time),
        }

        ws_url = self.loki_url.replace("http://", "ws://").replace("https://", "wss://")

        return f"{ws_url}/loki/api/v1/tail?{urlencode(params)}"

    async def process_log_entry(self, log_entry: Dict, r2_storage: R2Storage) -> None:
        """
        Process a log entry from Loki, extract metadata, and add to buffer.

        Args:
            log_entry: Log entry from Loki
            r2_storage: R2 storage instance for uploading logs
        """
        for stream in log_entry.get("streams", []):
            stream_labels = stream.get("stream", {})

            for value in stream.get("values", []):
                if len(value) != 2:
                    continue

                _, log_line = value

                try:
                    log_data = json.loads(log_line)

                    service = log_data.get("service", "unknown")
                    uid = log_data.get("uid", "unknown")
                    version = log_data.get("version", "unknown")

                    for key, value in stream_labels.items():
                        if key not in log_data:
                            log_data[f"label_{key}"] = value

                    buffer_key = (service, uid, version)
                    if buffer_key not in self.log_buffer:
                        self.log_buffer[buffer_key] = []

                    self.log_buffer[buffer_key].append(log_data)

                    await self._check_flush_buffer(r2_storage)

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse log line as JSON: {log_line}")
                except Exception as e:
                    logger.error(f"Error processing log entry: {e}")

    async def _check_flush_buffer(self, r2_storage: R2Storage) -> None:
        """
        Check if any buffers should be flushed and upload to R2 if needed.

        Args:
            r2_storage: R2 storage instance for uploading logs
        """
        current_time = time.time()
        time_since_last_flush = current_time - self.last_flush_time

        keys_to_flush = []
        for buffer_key, logs in self.log_buffer.items():
            if len(logs) >= self.buffer_size:
                keys_to_flush.append(buffer_key)

        if time_since_last_flush >= self.buffer_timeout:
            keys_to_flush = list(self.log_buffer.keys())

        for key in keys_to_flush:
            service, uid, version = key
            logs = self.log_buffer.pop(key, [])
            if logs:
                await r2_storage.upload_logs(logs, service, uid, version)

        if keys_to_flush:
            self.last_flush_time = current_time

    async def stream_logs(self, r2_storage: R2Storage) -> None:
        """
        Start streaming logs from Loki.

        Args:
            r2_storage: R2 storage instance for uploading logs
        """
        self.running = True
        reconnect_delay = 1
        max_reconnect_delay = 60

        while self.running:
            try:
                if not await self.check_loki_health():
                    logger.error("Loki is not healthy, waiting before retry...")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                    continue

                reconnect_delay = 1

                tail_url = self._build_tail_url()
                logger.info(f"Connecting to Loki WebSocket: {tail_url}")

                # Try using HTTP API instead of WebSocket for now
                http_url = f"{self.loki_url}/loki/api/v1/query_range"
                logger.info(f"Using HTTP API instead of WebSocket: {http_url}")

                # Query every 10 seconds in a loop
                while self.running:
                    try:
                        # Looking back further to catch more logs - 5 minutes
                        lookback_period = 300  # 5 minutes

                        # Use the query_range endpoint with a valid matcher
                        # Loki requires at least one matcher that is not empty-compatible
                        params = {
                            "query": '{service=~".+"}',  # Match any service (not empty)
                            "start": str(
                                int(time.time() - lookback_period) * 1_000_000_000
                            ),  # 5 minutes ago
                            "end": str(int(time.time()) * 1_000_000_000),  # now
                            "limit": str(self.limit),
                        }

                        logger.info(
                            f"Querying Loki with {lookback_period}s lookback period, query: {params['query']}"
                        )
                        async with aiohttp.ClientSession() as session:
                            async with session.get(http_url, params=params) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    logger.info(
                                        f"Received response from Loki: {str(data)[:100]}..."
                                    )

                                    # Process the response
                                    if "data" in data and "result" in data["data"]:
                                        results = data["data"]["result"]
                                        if results:
                                            logger.info(
                                                f"Found {len(results)} log streams to process"
                                            )
                                            total_values = sum(
                                                len(stream.get("values", []))
                                                for stream in results
                                            )
                                            logger.info(
                                                f"Total log entries: {total_values}"
                                            )

                                            # Debug log the first stream for inspection
                                            if (
                                                logger.level <= logging.DEBUG
                                                and results
                                            ):
                                                first_stream = results[0]
                                                logger.debug(
                                                    f"First stream: {first_stream}"
                                                )

                                            for stream in results:
                                                logger.info(
                                                    f"Processing stream with {len(stream.get('values', []))} log values"
                                                )
                                                stream_data = {
                                                    "streams": [
                                                        {
                                                            "stream": stream.get(
                                                                "stream", {}
                                                            ),
                                                            "values": stream.get(
                                                                "values", []
                                                            ),
                                                        }
                                                    ]
                                                }
                                                await self.process_log_entry(
                                                    stream_data, r2_storage
                                                )

                                            # Log buffer status after processing
                                            buffer_size = sum(
                                                len(logs)
                                                for logs in self.log_buffer.values()
                                            )
                                            logger.info(
                                                f"Buffer status after processing: {buffer_size} logs"
                                            )
                                        else:
                                            logger.info(
                                                "No logs found in the last 15 seconds"
                                            )
                                    else:
                                        logger.warning(
                                            f"Unexpected response format from Loki: {data}"
                                        )
                                else:
                                    content = await response.text()
                                    logger.error(
                                        f"Failed to query Loki: {response.status} - {content}"
                                    )
                    except Exception as e:
                        logger.error(f"Error querying Loki HTTP API: {e}")

                    # Wait 10 seconds before next query
                    await asyncio.sleep(10)

            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}, reconnecting...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

            except Exception as e:
                logger.error(f"Error streaming logs: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

        # Flush any remaining logs before exiting
        for key, logs in self.log_buffer.items():
            service, uid, version = key
            if logs:
                await r2_storage.upload_logs(logs, service, uid, version)

    def stop(self) -> None:
        """Stop streaming logs."""
        self.running = False
        logger.info("Stopping Loki client")


async def main():
    """Main entry point for the log collector."""
    parser = argparse.ArgumentParser(description="Loki to R2 Log Collector")

    parser.add_argument(
        "--loki-url", default="http://localhost:3100", help="Loki base URL"
    )
    parser.add_argument(
        "--query", default='{service="validator"}', help="LogQL query to filter logs"
    )
    parser.add_argument(
        "--delay-for", type=int, default=0, help="Delay for slow loggers (max 5s)"
    )
    parser.add_argument(
        "--limit", type=int, default=1000, help="Max entries to fetch per request"
    )

    parser.add_argument("--r2-bucket", required=True, help="R2 bucket name")
    parser.add_argument("--r2-endpoint", required=True, help="R2 endpoint URL")
    parser.add_argument("--r2-region", default="auto", help="R2 region")

    parser.add_argument(
        "--buffer-size", type=int, default=100, help="Max logs to buffer before upload"
    )
    parser.add_argument(
        "--buffer-timeout", type=int, default=60, help="Max seconds to buffer logs"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

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

    loki_client = LokiClient(
        loki_url=args.loki_url,
        query=args.query,
        delay_for=args.delay_for,
        limit=args.limit,
        buffer_size=args.buffer_size,
        buffer_timeout=args.buffer_timeout,
    )

    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Received shutdown signal, stopping...")
        loki_client.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        logger.info("Starting Loki to R2 Log Collector")
        await loki_client.stream_logs(r2_storage)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        logger.info("Log collector shutting down")


if __name__ == "__main__":
    asyncio.run(main())
