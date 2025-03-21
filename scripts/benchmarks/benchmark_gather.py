#!/usr/bin/env python3

# ruff: noqa
"""
Benchmark Gather Operation Using a Compressed Model State

This benchmark simulates a gather operation where 20 fake peers have uploaded a
compressed state file created from an actual model's state dict (using the same
model and compression as the miner).

For each iteration:
  1. If not already present, a state file is created by instantiating the model with
     hparams loaded via `tplr.load_hparams()`, running the compression pipeline, and
     saving the result.
  2. It pre-uploads 20 fake peer state files to R2 using keys of the form:
         "gradient-{window}-{peer_uid}-v{__version__}.pt"
  3. It then calls Comms.gather with our own state dict, which triggers a put using
     the key:
         "gradient-{window}-{dummy_uid}-v{__version__}.pt"
  4. The benchmark measures the peer upload time and overall gather duration.
  5. Finally, all remote files from peers and our own state file are deleted, and the
     local state file is removed.

Results are stored as CSV in the benchmark_results directory and graphs are plotted.

Usage:
    python scripts/benchmarks/benchmark_gather_state.py
"""

import os
import time
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from dotenv import load_dotenv
from aiobotocore.config import AioConfig
from transformers import LlamaForCausalLM
import tplr

# Load environment variables from .env
load_dotenv(override=True)

###############################################
# Cloudflare R2 Credentials & Config.
###############################################
R2_ACCOUNT_ID = os.getenv("R2_GRADIENTS_ACCOUNT_ID")
R2_BUCKET_NAME = os.getenv("R2_GRADIENTS_BUCKET_NAME")
R2_WRITE_ACCESS_KEY_ID = os.getenv("R2_GRADIENTS_WRITE_ACCESS_KEY_ID")
R2_WRITE_SECRET_ACCESS_KEY = os.getenv("R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY")
if not all(
    [R2_ACCOUNT_ID, R2_BUCKET_NAME, R2_WRITE_ACCESS_KEY_ID, R2_WRITE_SECRET_ACCESS_KEY]
):
    raise EnvironmentError("Missing one or more R2 credentials.")

ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
REGION_NAME = "enam"
client_config = AioConfig(max_pool_connections=50)

###############################################
# Load hparams the same way as miner does.
###############################################
dummy_hparams = tplr.load_hparams()


###############################################
# Dummy Objects for Comms.
###############################################
class DummyHotkey:
    ss58_address = "dummy_hotkey_address"


class DummyWallet:
    hotkey = DummyHotkey()


class DummyConfig:
    device = os.getenv("CUDA_DEVICE", "cpu")


dummy_wallet = DummyWallet()
dummy_config = DummyConfig()
dummy_metagraph = None
dummy_uid = "9999"  # Our own UID

###############################################
# Build BUCKET_SECRETS from Environment.
###############################################
BUCKET_SECRETS = {
    "gradients": {
        "name": R2_BUCKET_NAME,
        "account_id": R2_ACCOUNT_ID,
        "credentials": {
            "read": {
                "access_key_id": os.getenv("R2_GRADIENTS_READ_ACCESS_KEY_ID"),
                "secret_access_key": os.getenv("R2_GRADIENTS_READ_SECRET_ACCESS_KEY"),
            },
            "write": {
                "access_key_id": R2_WRITE_ACCESS_KEY_ID,
                "secret_access_key": R2_WRITE_SECRET_ACCESS_KEY,
            },
        },
    },
    "dataset": {
        "name": os.getenv("R2_DATASET_BUCKET_NAME"),
        "account_id": os.getenv("R2_DATASET_ACCOUNT_ID"),
        "credentials": {
            "read": {
                "access_key_id": os.getenv("R2_DATASET_READ_ACCESS_KEY_ID"),
                "secret_access_key": os.getenv("R2_DATASET_READ_SECRET_ACCESS_KEY"),
            },
            "write": {
                "access_key_id": os.getenv("R2_DATASET_WRITE_ACCESS_KEY_ID"),
                "secret_access_key": os.getenv("R2_DATASET_WRITE_SECRET_ACCESS_KEY"),
            },
        },
    },
}

###############################################
# Monkey-patch BUCKET_SECRETS into Comms.
###############################################
import tplr.comms as comms_module

comms_module.__dict__["BUCKET_SECRETS"] = BUCKET_SECRETS
from tplr.comms import Comms, __version__
from tplr.compress import TransformDCT, CompressDCT

###############################################
# Benchmark Parameters & File Names.
###############################################
STATE_FILE_NAME = "model_state.pt"  # local temporary model state file
N_ITERATIONS = 5  # Number of benchmark iterations.
N_FAKE_PEERS = 20  # Number of fake peers to simulate.
WINDOW = 1
GLOBAL_STEP = 1
RESULTS_DIR = "scripts/benchmarks/benchmark_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "gather_state_benchmark_results.csv")


###############################################
# Helper to delete an object from S3.
###############################################
async def delete_object(comms_instance, key: str):
    async with comms_instance.session.create_client(
        "s3",
        region_name=REGION_NAME,
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=BUCKET_SECRETS["gradients"]["credentials"]["write"][
            "access_key_id"
        ],
        aws_secret_access_key=BUCKET_SECRETS["gradients"]["credentials"]["write"][
            "secret_access_key"
        ],
        config=client_config,
    ) as client:
        try:
            await client.delete_object(
                Bucket=BUCKET_SECRETS["gradients"]["name"], Key=key
            )
            print(f"Deleted remote object '{key}'")
        except Exception as e:
            print(f"Failed to delete object '{key}': {e}")


###############################################
# Create a compressed state file using the model and compression.
###############################################
def ensure_model_state_file():
    """
    Creates the compressed model state file if it doesn't exist.
    The state is created by instantiating the model using the hparams loaded via tplr.load_hparams,
    running the compression pipeline using TransformDCT and CompressDCT, and saving the result.

    The file is saved with the local filename STATE_FILE_NAME.
    """
    if os.path.exists(STATE_FILE_NAME):
        print(f"State file '{STATE_FILE_NAME}' already exists.")
        return

    print("Creating compressed model state file...")
    # Instantiate the model exactly as done in miner.py.
    model = LlamaForCausalLM(dummy_hparams.model_config)
    model.eval()

    # Choose a target_chunk size (for example, 128)
    target_chunk = 128

    # Instantiate the transformer with the required arguments.
    transformer = TransformDCT(model, target_chunk)
    compressor = CompressDCT()

    state_compressed = {}
    # Set topk value. Optionally, use a value from hparams if available.
    topk = getattr(dummy_hparams, "compress_topk", 128)

    # Compress each parameter from the model's state dict.
    for name, param in model.state_dict().items():
        # Ensure the param is a float tensor.
        param = param.float()
        # Pass the parameter with its original shape.
        encoded = transformer.encode(param)
        # Compress the encoded tensor using the provided topk value.
        indices, values, orig_shape, totalk = compressor.compress(encoded, topk)
        # Save the compressed representations with keys appended with "idxs" and "vals"
        state_compressed[name + "idxs"] = indices
        state_compressed[name + "vals"] = values

    # Save the compressed state dict to file.
    torch.save(state_compressed, STATE_FILE_NAME)
    print("Compressed model state file created.")


###############################################
# S3 File Upload & Delete Helpers.
# Files use the naming convention:
#    filename = f"gradient-{window}-{uid}-v{__version__}.pt"
###############################################
def get_s3_key(uid: str, window: int) -> str:
    return f"gradient-{window}-{uid}-v{__version__}.pt"


async def upload_fake_peer(comms_instance, peer_uid: str, window: int):
    # Always load the local compressed state with weights_only=True.
    state = torch.load(STATE_FILE_NAME, weights_only=True)
    # Build the S3 key. Uid is now a plain number (as a string).
    key = get_s3_key(peer_uid, window)
    # Use the monkey-patched s3_put_object from the comms instance.
    await comms_instance.s3_put_object(key, state)
    print(f"Uploaded fake peer state file: {key}")


async def delete_fake_peer(peer_uid: str, window: int):
    key = get_s3_key(peer_uid, window)
    await delete_object(None, key)


async def delete_own_state(uid: str, window: int):
    key = get_s3_key(uid, window)
    await delete_object(None, key)


###############################################
# Benchmark a Single Gather Iteration.
###############################################
async def benchmark_gather_iteration(comms_instance, iteration: int):
    print(f"\n--- Gather State Benchmark Iteration {iteration} ---")
    global WINDOW  # your global window number
    # Generate 20 fake peer UIDs as strings (plain numbers, e.g., "0", "1", "2", ...)
    fake_peer_uids = [str(i) for i in range(N_FAKE_PEERS)]

    # Pre-upload fake peer state files concurrently.
    peer_upload_start = time.time()
    upload_tasks = [
        upload_fake_peer(comms_instance, uid, WINDOW) for uid in fake_peer_uids
    ]
    await asyncio.gather(*upload_tasks)
    peer_total_upload_time = time.time() - peer_upload_start
    print(f"Fake peer uploads completed in {peer_total_upload_time:.2f} seconds.")

    # Load our own compressed state with weights_only=True.
    state = torch.load(STATE_FILE_NAME, weights_only=True)

    # Start the gather call. Using key "gradient" creates our file:
    # "gradient-{WINDOW}-{dummy_uid}-v{__version__}.pt"
    print("Starting gather call ...")
    gather_start = time.time()
    gather_result = await comms_instance.gather(
        state_dict=state,  # non-empty state triggers own put call
        my_uid=dummy_uid,
        uids=fake_peer_uids,  # array of plain number UIDs
        window=WINDOW,
        key="gradient",  # key set so our file is named per protocol
        timeout=60,
        device=dummy_config.device,
        global_step=GLOBAL_STEP,
        local=False,
        stale_retention=100,
    )
    gather_duration = time.time() - gather_start
    print(f"Gather call completed in {gather_duration:.2f} seconds.")

    # Cleanup: remove fake peer objects and our own state from S3.
    delete_tasks = [delete_fake_peer(uid, WINDOW) for uid in fake_peer_uids]
    await asyncio.gather(*delete_tasks)
    await delete_own_state(dummy_uid, WINDOW)

    total_iteration_time = peer_total_upload_time + gather_duration
    metrics = {
        "iteration": iteration,
        "peer_total_upload_time": peer_total_upload_time,
        "gather_duration": gather_duration,
        "total_iteration_time": total_iteration_time,
    }
    return metrics


###############################################
# Run the Benchmark.
###############################################
async def run_benchmark():
    ensure_model_state_file()
    comms_instance = Comms(
        wallet=dummy_wallet,
        key_prefix="gradient",
        config=dummy_config,
        netuid=int(os.getenv("NETUID", 3)),
        metagraph=dummy_metagraph,
        hparams=dummy_hparams,
        uid=dummy_uid,
    )

    # Monkey patch: always return our local bucket regardless of uid.
    comms_instance.get_peer_bucket = lambda uid: comms_instance.bucket

    # Monkey-patch s3_put_object to handle dict input.
    original_s3_put_object = comms_instance.s3_put_object

    async def s3_put_object_wrapper(filename, data):
        if isinstance(data, dict):
            temp_path = os.path.join("/tmp", filename.replace("-", "_"))
            torch.save(data, temp_path)
            await original_s3_put_object(filename, temp_path)
            os.remove(temp_path)
        else:
            await original_s3_put_object(filename, data)

    comms_instance.s3_put_object = s3_put_object_wrapper

    results = []
    for i in range(N_ITERATIONS):
        res = await benchmark_gather_iteration(comms_instance, i)
        results.append(res)
        await asyncio.sleep(2)
    return results


def save_results(results):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Results saved to '{RESULTS_CSV}'")
    return df


def plot_results(df):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, y="peer_total_upload_time", color="lightgreen")
    sns.stripplot(
        data=df, y="peer_total_upload_time", color="black", jitter=0.2, size=8
    )
    plt.title("Total Peer Upload Time")
    plt.ylabel("Time (seconds)")

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, y="gather_duration", color="lightblue")
    sns.stripplot(data=df, y="gather_duration", color="black", jitter=0.2, size=8)
    plt.title("Gather Duration")
    plt.ylabel("Time (seconds)")

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "gather_state_benchmark_plots.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Plots saved to '{plot_path}'")

    print("\nBenchmark Summary Statistics:")
    print(df.describe())


###############################################
# Main execution.
###############################################
async def main():
    results = await run_benchmark()
    df = save_results(results)
    plot_results(df)
    # Cleanup local state file.
    if os.path.exists(STATE_FILE_NAME):
        os.remove(STATE_FILE_NAME)
        print(f"Temporary model state file '{STATE_FILE_NAME}' removed.")


if __name__ == "__main__":
    asyncio.run(main())
