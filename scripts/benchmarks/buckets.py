# ruff: noqa
# pylint: disable=all
# mypy: ignore-errors
# type: ignore
"""
Benchmark module for uploading, listing, downloading, and cleaning files on S3
compatible platforms, with detailed logging of execution times.
"""

import asyncio
import io
import json
import os
import tempfile
import time
from pathlib import Path
from pprint import pprint
from typing import Any, Literal

import aioboto3
import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from templar import get_indices_for_window, load_hparams
from tqdm.auto import trange
from transformers import LlamaForCausalLM

SEED = 42
DEVICE = torch.device("cuda:6")
PLATFORM = Literal["CF", "AWS"]
FILE_NAME = "slice.pt"


async def get_bucket_and_client(platform: PLATFORM) -> tuple[str, Any]:
    # Get secrets
    load_dotenv()
    access_key_id = os.getenv(f"{platform}_ACCESS_KEY_ID")
    secret_access_key = os.getenv(f"{platform}_SECRET_ACCESS_KEY")
    bucket = os.getenv(f"{platform}_BUCKET_NAME")
    account_id = os.getenv(f"{platform}_ACCOUNT_ID")
    base_url = f"https://{account_id}.r2.cloudflarestorage.com"
    session = aioboto3.Session()
    client = await session.client(
        service_name="s3",
        endpoint_url=base_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    ).__aenter__()
    return bucket, client


def test_operations_with_json_file():
    file_path = Path(__file__).parents[1] / "hparams.json"
    with open(file_path) as f:
        json_data = json.load(f)

    # Convert JSON to bytes and create a file-like object
    file_content = json.dumps(json_data).encode("utf-8")  # Convert JSON to bytes
    client.upload_fileobj(io.BytesIO(file_content), bucket, file_path.name)

    # Get object information
    object_information = client.head_object(Bucket=bucket, Key=file_path.name)

    # Delete object
    client.delete_object(Bucket=bucket, Key=file_path.name)


def get_model():
    hparams = load_hparams()
    model = LlamaForCausalLM(config=hparams.model_config)
    return model.to(DEVICE)


async def get_slice(model, compression: int = 1e2):
    model_state_dict = model.state_dict()
    indices = await get_indices_for_window(model, str(SEED), compression)

    # Apply the slice to the model parameters
    for name, param in model.named_parameters():
        model_state_dict[name] = param.data.view(-1)[
            indices[name].to(model.device)
        ].cpu()

    # Create a temporary file and write the sliced model state dictionary to it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_file:
        torch.save(model_state_dict, temp_file)
        temp_file_name = temp_file.name  # Store the temporary file name

    return temp_file_name  # Return the path to the temporary file


async def upload_slice(temp_file_name, client, bucket):
    with open(temp_file_name, "rb") as f:
        client.upload_fileobj(f, bucket, FILE_NAME)
    logger.success(f"Uploaded slice to {bucket} as {FILE_NAME}")

    # Optional cleanup of the temporary file
    os.remove(temp_file_name)
    logger.info(f"Temporary file {temp_file_name} deleted after upload.")
    # Optional verification
    try:
        response = await client.head_object(Bucket=bucket, Key=FILE_NAME)
        print("File upload verified:", response)
    except client.exceptions.NoSuchKey:
        print("Upload verification failed. File not found on S3.")


async def list_objects_all_at_once(client, bucket):
    l = client.list_objects_v2(Bucket=bucket)
    pprint(l)


async def list_objects_page_by_page(client, bucket):
    paginator = client.get_paginator("list_objects_v2")
    num_objs = 0
    async for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            filename = obj["Key"]
            num_objs += 1
    logger.success(f"Went through all pages and found {num_objs} objs")


async def get_object(client, bucket):
    obj = await client.get_object(Bucket=bucket, Key=FILE_NAME)
    logger.success(f"Downloaded object of type {type(obj)}")


async def clean_bucket(client, bucket):
    response = await client.list_objects_v2(Bucket=bucket)
    file_names = [content["Key"] for content in response.get("Contents", [])]
    for file_name in file_names:
        client.delete_object(Bucket=bucket, Key=file_name)
        print(f"Deleted {file_name}")


async def benchmark(client, bucket) -> dict[str, float]:
    model = get_model()
    temp_file_name = await get_slice(model)
    durations = {}

    # Upload
    start = time.time()
    await upload_slice(temp_file_name, client, bucket)
    end = time.time()
    durations["upload"] = end - start

    # List
    start = time.time()
    await list_objects_page_by_page(client, bucket)
    end = time.time()
    durations["list"] = end - start

    # Download
    start = time.time()
    await get_object(client, bucket)
    end = time.time()
    durations["download"] = end - start

    # Clean
    start = time.time()
    await clean_bucket(client, bucket)
    end = time.time()
    durations["clean"] = end - start

    return durations


def log_statistics(results, platform):
    tasks = ["upload", "list", "download", "clean"]
    metrics = []
    for task in tasks:
        task_durations = [run[task] for run in results[platform]]
        mean_duration = np.mean(task_durations)
        std_duration = np.std(task_durations, ddof=1)  # Sample standard deviation
        metrics.append(f"{task}: {mean_duration:.2f}±{std_duration:.2f}")

    # Join all metrics for this platform into a single line
    logger.info(f"{platform}:\t" + " | ".join(metrics))


async def main():
    results = {"AWS": [], "CF": []}
    for _ in trange(10):
        for platform in ("AWS", "CF"):
            logger.info(f"{'-' * 10} Starting benchmarks for {platform} {'-' * 10}")
            bucket, client = await get_bucket_and_client(platform)
            durations = await benchmark(client, bucket)
            results[platform].append(durations)
            logger.info(
                ", ".join(f"{key}: {value:.2f}" for key, value in durations.items())
            )
    for platform in results:
        log_statistics(results, platform)


if __name__ == "__main__":
    asyncio.run(main())
