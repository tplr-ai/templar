#!/usr/bin/env python3
"""
Benchmark S3 Propagation Delay Using Comms.s3_put_object

This script benchmarks:
  - The upload (PUT) times for a 110 MB file using Comms.s3_put_object.
  - The propagation delay until the object becomes visible (via head_object).
  
It performs multiple iterations, saves the results in CSV format under 
`scripts/benchmarks/benchmark_results/`, generates plots, and finally deletes the
temporary test file.

Usage:
    python scripts/benchmarks/benchmark_s3_propagation.py
"""

import os
import time
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import aiofiles
from aiobotocore.config import AioConfig

# Load environment variables from .env file.
load_dotenv(override=True)

###############################################
# Cloudflare R2 Credentials & settings.
###############################################
R2_ACCOUNT_ID = os.getenv("R2_GRADIENTS_ACCOUNT_ID")
R2_BUCKET_NAME = os.getenv("R2_GRADIENTS_BUCKET_NAME")
R2_WRITE_ACCESS_KEY_ID = os.getenv("R2_GRADIENTS_WRITE_ACCESS_KEY_ID")
R2_WRITE_SECRET_ACCESS_KEY = os.getenv("R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY")
if not all([R2_ACCOUNT_ID, R2_BUCKET_NAME, R2_WRITE_ACCESS_KEY_ID, R2_WRITE_SECRET_ACCESS_KEY]):
    raise EnvironmentError("Missing one or more R2 credentials.")

ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
REGION_NAME = "enam"
client_config = AioConfig(max_pool_connections=50)

###############################################
# Dummy objects for Comms Initialization.
###############################################
class DummyHotkey:
    ss58_address = "dummy_hotkey_address"

class DummyWallet:
    hotkey = DummyHotkey()

class DummyConfig:
    device = os.getenv("CUDA_DEVICE", "cpu")

class DummyHparams:
    active_check_interval = 10  # seconds
    recent_windows = 3
    blocks_per_window = 10      # added to satisfy Comms requirements

dummy_wallet = DummyWallet()
dummy_config = DummyConfig()
dummy_hparams = DummyHparams()
dummy_metagraph = None
dummy_uid = "9999"

###############################################
# Build BUCKET_SECRETS from environment.
###############################################
BUCKET_SECRETS = {
    "gradients": {
        "name": R2_BUCKET_NAME,
        "account_id": R2_ACCOUNT_ID,
        "credentials": {
            "read": {
                "access_key_id": os.getenv("R2_GRADIENTS_READ_ACCESS_KEY_ID"),
                "secret_access_key": os.getenv("R2_GRADIENTS_READ_SECRET_ACCESS_KEY")
            },
            "write": {
                "access_key_id": R2_WRITE_ACCESS_KEY_ID,
                "secret_access_key": R2_WRITE_SECRET_ACCESS_KEY
            }
        }
    },
    "dataset": {
        "name": os.getenv("R2_DATASET_BUCKET_NAME"),
        "account_id": os.getenv("R2_DATASET_ACCOUNT_ID"),
        "credentials": {
            "read": {
                "access_key_id": os.getenv("R2_DATASET_READ_ACCESS_KEY_ID"),
                "secret_access_key": os.getenv("R2_DATASET_READ_SECRET_ACCESS_KEY")
            },
            "write": {
                "access_key_id": os.getenv("R2_DATASET_WRITE_ACCESS_KEY_ID"),
                "secret_access_key": os.getenv("R2_DATASET_WRITE_SECRET_ACCESS_KEY")
            }
        }
    }
}

###############################################
# Monkey-patch BUCKET_SECRETS into Comms.
###############################################
import tplr.comms as comms_module
comms_module.__dict__['BUCKET_SECRETS'] = BUCKET_SECRETS
from tplr.comms import Comms

###############################################
# File and Benchmark Parameters.
###############################################
TEST_FILE_NAME = "temp_110mb_file.bin"
TARGET_SIZE = 110 * 1024 * 1024  # 110 MB in bytes
N_ITERATIONS = 10                # Number of benchmark iterations.
POLL_INTERVAL = 1.0              # seconds between head_object polls.
MAX_POLL_TIME = 180              # Maximum seconds to wait for object availability.

RESULTS_DIR = "scripts/benchmarks/benchmark_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "s3_propagation_benchmark_results.csv")

async def ensure_test_file():
    """
    Ensure a 110 MB test file exists locally; otherwise, generate it.
    """
    if os.path.exists(TEST_FILE_NAME) and os.path.getsize(TEST_FILE_NAME) == TARGET_SIZE:
        print(f"Test file '{TEST_FILE_NAME}' exists.")
        return

    print(f"Generating {TARGET_SIZE} bytes test file '{TEST_FILE_NAME}' ...")
    block_size = 1024 * 1024  # 1 MB
    num_blocks = TARGET_SIZE // block_size
    remainder = TARGET_SIZE % block_size
    async with aiofiles.open(TEST_FILE_NAME, "wb") as f:
        for _ in range(num_blocks):
            await f.write(b'\0' * block_size)
        if remainder:
            await f.write(b'\0' * remainder)
    print("Test file generated.")

async def benchmark_iteration(comms_instance, iteration):
    """
    For one iteration:
      - Upload the 110 MB file using comms.s3_put_object.
      - Record the upload (PUT) duration.
      - Poll the S3 bucket with head_object until the file is visible, and record propagation delay.
      - Delete the object from S3.
      - Return a dictionary of metrics.
    """
    key = f"benchmark_propagation_{iteration}.bin"
    print(f"Iteration {iteration}: Uploading file with key '{key}' ...")

    iteration_start = time.time()
    upload_start = time.time()
    await comms_instance.s3_put_object(key=key, file_path=TEST_FILE_NAME)
    upload_end = time.time()
    upload_duration = upload_end - upload_start
    print(f"Iteration {iteration}: Upload complete in {upload_duration:.2f} seconds.")

    # Poll S3 for availability.
    poll_start = time.time()
    propagation_delay = None
    async with comms_instance.session.create_client(
        "s3",
        endpoint_url=comms_instance.get_base_url(BUCKET_SECRETS["gradients"]["account_id"]),
        region_name=REGION_NAME,
        config=client_config,
        aws_access_key_id=BUCKET_SECRETS["gradients"]["credentials"]["write"]["access_key_id"],
        aws_secret_access_key=BUCKET_SECRETS["gradients"]["credentials"]["write"]["secret_access_key"]
    ) as client:
        while time.time() - poll_start < MAX_POLL_TIME:
            try:
                await client.head_object(Bucket=BUCKET_SECRETS["gradients"]["name"], Key=key)
                propagation_delay = time.time() - upload_end
                print(f"Iteration {iteration}: Object available after {propagation_delay:.2f} seconds.")
                break
            except Exception as e:
                await asyncio.sleep(POLL_INTERVAL)
    if propagation_delay is None:
        print(f"Iteration {iteration}: Timeout; object not available after {MAX_POLL_TIME} seconds.")
        propagation_delay = np.nan
    iteration_end = time.time()
    total_duration = iteration_end - iteration_start

    # Clean up: delete the uploaded object.
    try:
        async with comms_instance.session.create_client(
            "s3",
            endpoint_url=comms_instance.get_base_url(BUCKET_SECRETS["gradients"]["account_id"]),
            region_name=REGION_NAME,
            config=client_config,
            aws_access_key_id=BUCKET_SECRETS["gradients"]["credentials"]["write"]["access_key_id"],
            aws_secret_access_key=BUCKET_SECRETS["gradients"]["credentials"]["write"]["secret_access_key"]
        ) as client:
            await client.delete_object(Bucket=BUCKET_SECRETS["gradients"]["name"], Key=key)
    except Exception as e:
        print(f"Iteration {iteration}: Failed to delete object '{key}': {e}")

    return {
        "iteration": iteration,
        "upload_duration": upload_duration,
        "propagation_delay": propagation_delay,
        "total_duration": total_duration
    }

async def run_benchmark():
    await ensure_test_file()

    comms_instance = Comms(
        wallet=dummy_wallet,
        key_prefix="model",
        config=dummy_config,
        netuid=int(os.getenv("NETUID", 3)),
        metagraph=dummy_metagraph,
        hparams=dummy_hparams,
        uid=dummy_uid
    )
    results = []
    for i in range(N_ITERATIONS):
        res = await benchmark_iteration(comms_instance, i)
        results.append(res)
        await asyncio.sleep(2)  # Pause between iterations.
    return results

def save_results(results):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Results saved to '{RESULTS_CSV}'")
    return df

def plot_results(df):
    # Plot both upload durations and propagation delays.
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, y="upload_duration", color="lightgreen")
    sns.stripplot(data=df, y="upload_duration", color="black", jitter=0.1, size=8)
    plt.title("Upload (PUT) Duration")
    plt.ylabel("Duration (seconds)")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, y="propagation_delay", color="lightblue")
    sns.stripplot(data=df, y="propagation_delay", color="black", jitter=0.1, size=8)
    plt.title("Propagation Delay")
    plt.ylabel("Delay (seconds)")
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "s3_propagation_benchmark_plots.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Plots saved to '{plot_path}'")
    
    print("\nBenchmark Summary Statistics:")
    print(df.describe())

async def main():
    results = await run_benchmark()
    df = save_results(results)
    plot_results(df)
    # Cleanup: remove local temporary test file.
    if os.path.exists(TEST_FILE_NAME):
        os.remove(TEST_FILE_NAME)
        print(f"Temporary test file '{TEST_FILE_NAME}' removed.")

if __name__ == "__main__":
    asyncio.run(main())