# ruff: noqa

"""
Benchmark for comparing the old and new implementations of
s3_get_object (disk‑based vs in‑memory streaming) on an already
uploaded 120 MB file in a real R2 bucket.

This script:
  - Assumes the object (key: "gradient-120mb-test.pt") is already on the bucket.
  - Retrieves (GET) the file using both implementations.
  - Prints object metadata and benchmark timings for verification.
"""

import os
import io
import time
import asyncio
from aiobotocore import session
from dotenv import load_dotenv
from pathlib import Path

from tplr import config
from tplr.logging import logger

# Load .env similar to benchmark_parquet_loader.py
env_path = Path(__file__).parent.parent.parent / ".env"
if not env_path.exists():
    raise FileNotFoundError(f"Required .env file not found at {env_path}")
load_dotenv(env_path, override=True)

# Cloudflare region, defaulting to 'auto'
CF_REGION_NAME = os.getenv("CF_REGION_NAME", "auto")

# Credentials for the R2 bucket.
GRADIENTS_ACCOUNT_ID = os.getenv("R2_GRADIENTS_ACCOUNT_ID")
GRADIENTS_BUCKET_NAME = os.getenv("R2_GRADIENTS_BUCKET_NAME")
GRADIENTS_READ_ACCESS_KEY_ID = os.getenv("R2_GRADIENTS_READ_ACCESS_KEY_ID")
GRADIENTS_READ_SECRET_ACCESS_KEY = os.getenv("R2_GRADIENTS_READ_SECRET_ACCESS_KEY")

# Test key for our already uploaded 120 MB file.
TEST_KEY = "benchmark-120mb-test.pt"

# --- Helper Classes for R2 Bucket Operations ------------------------------


class R2Bucket:
    """
    Represents a real R2 bucket using credentials from .env.
    """

    def __init__(self, name, account_id, access_key_id, secret_access_key):
        self.name = name
        self.account_id = account_id
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key


class DummyConfig:
    def __init__(self, device="cpu"):
        self.device = device


class R2Comms:
    """
    Implements S3 GET operations for a real R2 bucket.
    Contains both the old (disk‑based) and the new (in‑memory) GET implementations.
    Instead of deserializing the object (via torch.load), both methods now simply download
    the raw bytes.
    """

    def __init__(self, config, temp_dir="./tmp_test"):
        self.session = session.get_session()
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        self.config = config

    # Use account_id (not bucket name) to build the endpoint URL.
    def get_base_url(self, account_id):
        return f"https://{account_id}.r2.cloudflarestorage.com"

    async def s3_head_object(self, key: str, bucket: R2Bucket, timeout: int = 5):
        """Retrieve metadata (HEAD) for the object."""
        async with self.session.create_client(
            "s3",
            endpoint_url=self.get_base_url(bucket.account_id),
            region_name=CF_REGION_NAME,
            config=config.client_config,
            aws_access_key_id=bucket.access_key_id,
            aws_secret_access_key=bucket.secret_access_key,
        ) as s3_client:
            try:
                response = await asyncio.wait_for(
                    s3_client.head_object(Bucket=bucket.name, Key=key),
                    timeout=timeout,
                )
                return response
            except Exception as e:
                logger.error(f"Error in s3_head_object for {key}: {e}")
                return None

    async def wait_for_object(
        self, key: str, bucket: R2Bucket, retries: int = 5, delay: int = 2
    ):
        """Retries HEAD until the object is available."""
        for i in range(retries):
            metadata = await self.s3_head_object(key, bucket)
            if metadata is not None:
                return metadata
            print(
                f"Object {key} not found, retrying in {delay} seconds... (Attempt {i + 1}/{retries})"
            )
            await asyncio.sleep(delay)
        return None

    async def s3_get_object_old(self, key: str, bucket: R2Bucket, timeout: int = 5):
        """
        Old implementation: downloads the file to a temporary in‑memory buffer.
        Returns the raw downloaded bytes.
        """
        temp_file_path = os.path.join(self.temp_dir, f"temp_{key}")
        data_buffer = io.BytesIO()
        try:
            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(bucket.account_id),
                region_name=CF_REGION_NAME,
                config=config.client_config,
                aws_access_key_id=bucket.access_key_id,
                aws_secret_access_key=bucket.secret_access_key,
            ) as s3_client:
                try:
                    await asyncio.wait_for(
                        s3_client.head_object(Bucket=bucket.name, Key=key),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    logger.debug(f"Timeout checking for {key}")
                    return None
                except Exception as e:
                    if "404" in str(e):
                        logger.debug(
                            f"Object {key} not found in bucket {bucket.name}"
                        )
                        return None
                    raise

                response = await asyncio.wait_for(
                    s3_client.get_object(Bucket=bucket.name, Key=key),
                    timeout=timeout,
                )
                async with response["Body"] as stream:
                    chunk_size = 1 * 1024 * 1024  # 1 MB chunks
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                stream.content.read(chunk_size), timeout=timeout
                            )
                        except asyncio.TimeoutError:
                            logger.debug(f"Timeout reading chunk from {key}")
                            return None
                        if not chunk:
                            break
                        data_buffer.write(chunk)
            data_buffer.seek(0)
            # Instead of deserializing, simply return the raw bytes.
            return data_buffer.read()
        except Exception as e:
            logger.error(f"Error in s3_get_object_old for {key}: {e}")
            return None

    async def s3_get_object_new(self, key: str, bucket: R2Bucket, timeout: int = 5):
        """
        New implementation: streams data into an in‑memory buffer.
        Returns the raw downloaded bytes.
        """
        data_buffer = io.BytesIO()
        try:
            async with self.session.create_client(
                "s3",
                endpoint_url=self.get_base_url(bucket.account_id),
                region_name=CF_REGION_NAME,
                config=config.client_config,
                aws_access_key_id=bucket.access_key_id,
                aws_secret_access_key=bucket.secret_access_key,
            ) as s3_client:
                try:
                    await asyncio.wait_for(
                        s3_client.head_object(Bucket=bucket.name, Key=key),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    logger.debug(f"Timeout checking for {key}")
                    return None
                except Exception as e:
                    if "404" in str(e):
                        logger.debug(
                            f"Object {key} not found in bucket {bucket.name}"
                        )
                        return None
                    raise

                response = await asyncio.wait_for(
                    s3_client.get_object(Bucket=bucket.name, Key=key),
                    timeout=timeout,
                )
                async with response["Body"] as stream:
                    chunk_size = 1 * 1024 * 1024  # 1 MB chunks
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                stream.content.read(chunk_size), timeout=timeout
                            )
                        except asyncio.TimeoutError:
                            logger.debug(f"Timeout reading chunk from {key}")
                            return None
                        if not chunk:
                            break
                        data_buffer.write(chunk)
            data_buffer.seek(0)
            # Return the raw bytes without deserializing.
            return data_buffer.read()
        except Exception as e:
            logger.error(f"Error in s3_get_object_new for {key}: {e}")
            return None


# --- Benchmark Function ----------------------------------------------------------


async def benchmark_s3_get_object():
    """
    Benchmarks the old and new GET implementations for an already uploaded object.
    """
    config = DummyConfig(device="cpu")
    comms = R2Comms(config)

    bucket = R2Bucket(
        name=GRADIENTS_BUCKET_NAME,
        account_id=GRADIENTS_ACCOUNT_ID,
        access_key_id=GRADIENTS_READ_ACCESS_KEY_ID,
        secret_access_key=GRADIENTS_READ_SECRET_ACCESS_KEY,
    )

    # Ensure that the object is available
    metadata = await comms.wait_for_object(TEST_KEY, bucket)
    if metadata:
        file_size = metadata["ContentLength"]
        print(f"Object {TEST_KEY} is {file_size / (1024 * 1024):.2f} MB in size")
    else:
        print(f"Object {TEST_KEY} not found on bucket {bucket.name}")
        return

    iterations = 3
    old_times = []
    new_times = []

    # Warm-up: call each method once to prime caches/connections.
    _ = await comms.s3_get_object_old(TEST_KEY, bucket)
    _ = await comms.s3_get_object_new(TEST_KEY, bucket)

    for i in range(iterations):
        start = time.perf_counter()
        _ = await comms.s3_get_object_old(TEST_KEY, bucket)
        duration_old = time.perf_counter() - start
        old_times.append(duration_old)
        logger.info(
            f"Old implementation iteration {i + 1}: {duration_old:.2f} seconds"
        )

        start = time.perf_counter()
        _ = await comms.s3_get_object_new(TEST_KEY, bucket)
        duration_new = time.perf_counter() - start
        new_times.append(duration_new)
        logger.info(
            f"New implementation iteration {i + 1}: {duration_new:.2f} seconds"
        )

    print("\nBenchmark Results for GET:")
    avg_old = sum(old_times) / len(old_times)
    avg_new = sum(new_times) / len(new_times)
    print(
        f"Old implementation average time: {avg_old:.2f} seconds over {iterations} iterations"
    )
    print(
        f"New implementation average time: {avg_new:.2f} seconds over {iterations} iterations"
    )

    return {
        "old_times": old_times,
        "new_times": new_times,
        "avg_old": avg_old,
        "avg_new": avg_new,
    }


async def main():
    print("Starting GET benchmark:")
    await benchmark_s3_get_object()


if __name__ == "__main__":
    asyncio.run(main())
