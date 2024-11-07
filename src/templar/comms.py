# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import torch
import uvloop
import hashlib
import asyncio
import tempfile
import aiofiles
import numpy as np
import bittensor as bt
from typing import List, Dict
from types import SimpleNamespace
from filelock import FileLock, Timeout
from aiobotocore.session import get_session
import re
import sys
from templar.logging import logger

from . import *

# Set uvloop as the event loop policy
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Define a semaphore to limit concurrent downloads (adjust as needed)
semaphore = asyncio.Semaphore(1000)

async def get_slices( filename:str, device:str ) -> Dict[str, torch.Tensor]:
    # Attempt to acquire the lock with a timeout of 1 second.
    lock: FileLock = FileLock(f"{filename}.lock")
    with lock.acquire(timeout=5):
        pass
    return torch.load(
        filename,
        map_location=torch.device(device),
        weights_only = True,
    )

async def apply_slices_to_model(model: torch.nn.Module, window: int, seed: str, compression: int, key:str = 'slice') -> List[str]:
    """
    Applies slices from a specific window to the given model.

    Args:
        model (torch.nn.Module): The PyTorch model to which the slices will be applied.
        window (int): The window identifier.
        seed (str): The seed used for generating indices.
        compression (int): The compression factor.

    Returns:
        List[str]: A list of all the slice files that were applied.
    """
    # First get the indices associated with the window given the model.
    indices_dict = await get_indices_for_window(model, seed, compression)
    
    # Load all the slices associated with this window.
    slice_files = await load_files_for_window(window=window, key = key)

    # Dictionary to keep track of the number of slices applied per parameter.
    slices_per_param = {name: 0 for name, _ in model.named_parameters()}

    # Dictionary to accumulate the sum of values for each parameter.
    param_sums = {name: torch.zeros_like(param.data) for name, param in model.named_parameters()}

    # Iterate over each slice file and compute the sum of values.
    for file_i in slice_files:
        # Create a file lock to ensure exclusive access to the slice file.
        try:
            slice_i = await get_slices(file_i, model.device)
            for name, param in model.named_parameters():
                if name not in indices_dict or name not in slice_i:
                    continue
                values = slice_i[name].to(param.data.device)
                param_indices = indices_dict[name].to(param.data.device)
                param_sums[name].view(-1)[param_indices] += values
                slices_per_param[name] += 1
                del values
            del slice_i
        except Timeout:
            # The lock could not be acquired within the timeout.
            logger.error(f"Timeout occurred while trying to acquire lock on {file_i}")
            continue  
        except Exception as e:
            logger.exception(f"Error applying slice from {file_i}: {e}")

    # Apply the average to the parameters.
    for name, param in model.named_parameters():
        if name not in slices_per_param or name not in indices_dict or slices_per_param[name] == 0:
            continue
        param_indices = indices_dict[name].to(param.data.device)
        avg_param = param_sums[name].view(-1)[param_indices] / slices_per_param[name]
        avg_param = avg_param.to(param.data.dtype)
        avg_param = avg_param.to(param.data.device)
        param.data.view(-1)[param_indices] = avg_param.clone()

    # Return the list of the files applied.
    return slice_files

async def upload_slice_for_window(bucket: str, model: torch.nn.Module, window: int, seed: str, wallet: 'bt.wallet', compression: int, key:str = 'slice'):
    """
    Uploads a compressed slice of a PyTorch model to an S3 bucket.

    Args:
        bucket (str): Name of the S3 bucket.
        model (torch.nn.Module): The PyTorch model to be sliceed and uploaded.
        window (int): The window identifier.
        wallet (bt.wallet): The wallet object containing the hotkey.
        compression (int): The compression factor.
    """
    filename = f'{key}-{window}-{wallet.hotkey.ss58_address}.pt'
    logger.debug(f"Uploading slice to S3: {filename}")

    model_state_dict = model.state_dict()
    indices = await get_indices_for_window(model, seed, compression)

    # Apply the slice to the model parameters
    for name, param in model.named_parameters():
        model_state_dict[name] = param.data.view(-1)[indices[name].to(model.device)].cpu()

    # Create a temporary file and write the sliceed model state dictionary to it
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        torch.save(model_state_dict, temp_file)
        temp_file_name = temp_file.name  # Store the temporary file name

    # Upload the file to S3
    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:
        try:
            with open(temp_file_name, 'rb') as f:
                await s3_client.put_object(Bucket=bucket, Key=filename, Body=f)
            # Set the object ACL to public-read
            await s3_client.put_object_acl(
                Bucket=bucket,
                Key=filename,
                ACL='public-read'
            )
            logger.debug(f"Successfully uploaded slice to S3: {filename}")
        except Exception:
            logger.exception(f"Failed to upload slice {filename} to S3")
        finally:
            # Clean up the temporary file
            os.remove(temp_file_name)
            logger.debug(f"Temporary file {temp_file_name} removed")

async def upload_master(bucket: str, model: torch.nn.Module, wallet: 'bt.wallet'):
    """
    Uploads the master PyTorch model to an S3 bucket.

    Args:
        bucket (str): Name of the S3 bucket.
        model (torch.nn.Module): The PyTorch model to be uploaded.
        wallet (bt.wallet): The wallet object containing the hotkey.
    """
    upload_filename = f'master-{wallet.hotkey.ss58_address}.pt'
    logger.debug(f"Uploading master model to S3: {upload_filename}")

    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:
        try:
            # Create a temporary file and write the model state dictionary to it
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                torch.save(model.state_dict(), temp_file)
                temp_file_name = temp_file.name

            # Upload the file to S3
            with open(temp_file_name, 'rb') as f:
                await s3_client.put_object(Bucket=bucket, Key=upload_filename, Body=f)
            # Set the object ACL to public-read
            await s3_client.put_object_acl(
                Bucket=bucket,
                Key=upload_filename,
                ACL='public-read'
            )
            logger.debug(f"Successfully uploaded master model to S3: {upload_filename}")
        except Exception:
            logger.exception(f"Failed to upload master model {upload_filename} to S3")
        finally:
            # Clean up the temporary file
            os.remove(temp_file_name)
            logger.debug(f"Temporary file {temp_file_name} removed")

async def get_indices_for_window(model: torch.nn.Module, seed: str, compression: int) -> Dict[str, torch.LongTensor]:
    """
    Computes the indices for the given window and compression factor.

    Args:
        model (torch.nn.Module): The PyTorch model.
        seed (str): The window seed identifier.
        compression (int): The compression factor.

    Returns:
        Dict[str, torch.LongTensor]: A dictionary mapping parameter names to index tensors.
    """
    logger.debug(f"Computing indices for window seed {seed} with compression {compression}")
    result = {}
    # Seed the random number generator with the seed
    seed = int(hashlib.md5(str(seed).encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    for name, param in model.named_parameters():
        # Randomly select indices based on the compression factor
        num_indices = max(1, int(param.numel() // compression))
        indices = rng.choice(param.numel(), size=num_indices, replace=False)
        result[name] = torch.from_numpy(indices).long().cpu()
    return result

async def download_file(s3_client, bucket: str, filename: str) -> str:
    """
    Downloads a file from S3, using parallel downloads for large files.

    Args:
        s3_client: The S3 client.
        bucket (str): Name of the S3 bucket.
        filename (str): The S3 object key (filename).

    Returns:
        str: The path to the downloaded file in the temporary directory.
    """
    async with semaphore:
        temp_file = os.path.join(tempfile.gettempdir(), filename)
        # Check if the file exists.
        if os.path.exists(temp_file):
            logger.debug(f"File {temp_file} already exists, skipping download.")
            return temp_file
        lock_file = f"{temp_file}.lock"
        lock = FileLock(lock_file)
        try:
            # Try to acquire both locks with a timeout
            with lock.acquire(timeout=1):
                # Proceed to download the file
                logger.debug(f"Downloading file {filename} to {temp_file}")
                head_response = await s3_client.head_object(Bucket=bucket, Key=filename)
                object_size = head_response['ContentLength']
                CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB

                response = await s3_client.get_object(Bucket=bucket, Key=filename)
                async with aiofiles.open(temp_file, 'wb') as outfile:
                    while True:
                        chunk = await response['Body'].read(CHUNK_SIZE)
                        if not chunk:
                            break
                        await outfile.write(chunk)

                logger.debug(f"Successfully downloaded file {filename} to {temp_file}")
                return temp_file

        except Timeout:
            logger.error(f"Timeout occurred while trying to acquire lock on {lock_file}")
            return None
        except Exception as e:
            logger.exception(f"Failed to download file {filename} from bucket {bucket}: {e}")
            return None
        finally:
            # The lock is automatically released when exiting the 'with' block
            pass

async def handle_file(s3_client, bucket: str, filename: str, hotkey: str, window: int):
    """
    Handles downloading a single file from S3.

    Args:
        s3_client: The S3 client.
        bucket (str): Name of the S3 bucket.
        filename (str): The S3 object key (filename).
        hotkey (str): The hotkey identifier.
        window (int): The window identifier.

    Returns:
        SimpleNamespace: An object containing file metadata and the path to the downloaded file.
    """
    logger.debug(f"Handling file {filename} for window {window} and hotkey {hotkey}")
    temp_file = await download_file(s3_client, bucket, filename)
    if temp_file:
        return SimpleNamespace(bucket=bucket, hotkey=hotkey, filename=filename, window=window, temp_file=temp_file)
    return None

async def process_bucket(s3_client, bucket: str, windows: List[int], key: str = 'slice'):
    """
    Processes an S3 bucket to download files matching the given windows.

    Args:
        s3_client: The S3 client.
        bucket (str): Name of the S3 bucket.
        windows (List[int]): A list of window identifiers.

    Returns:
        List[SimpleNamespace]: A list of file metadata and paths for downloaded files.
    """
    # Validate the bucket name before processing
    if not is_valid_bucket(bucket):
        logger.error(f"Skipping invalid bucket: '{bucket}'")
        return []

    logger.debug(f"Processing bucket '{bucket}' for windows {windows}")
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')

    for window in windows:
        prefix = f'{key}-{window}'
        logger.debug(f"Listing objects with prefix '{prefix}' in bucket '{bucket}'")
        try:
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                logger.trace(f"Processing page for prefix '{prefix}' in bucket '{bucket}'")
                if 'Contents' not in page:
                    logger.trace(f"No contents found for prefix '{prefix}' in bucket '{bucket}'")
                    continue
                download_tasks = []
                for obj in page.get('Contents', []):
                    filename = obj['Key']
                    logger.trace(f"Processing object with key '{filename}' in bucket '{bucket}'")
                    try:
                        parts = filename.split('-')
                        if len(parts) < 3:
                            logger.error(f"Filename '{filename}' does not conform to the expected format.")
                            continue
                        slice_window = int(parts[1])
                        slice_hotkey = parts[2].split('.')[0]
                        logger.trace(f"Parsed filename '{filename}' into window '{slice_window}' and hotkey '{slice_hotkey}'")
                        if slice_window == window:
                            download_tasks.append(handle_file(s3_client, bucket, filename, slice_hotkey, slice_window))
                    except ValueError:
                        logger.exception(f"Error parsing window ID in filename '{filename}'")
                        continue
                    except Exception as e:
                        logger.exception(f"Unexpected error processing filename '{filename}': {e}")
                        continue
                # Download the files concurrently
                try:
                    results = await asyncio.gather(*download_tasks, return_exceptions=True)
                    for res in results:
                        if isinstance(res, Exception):
                            logger.error(f"Download task failed: {res}")
                        elif res:
                            files.append(res)
                except Exception as e:
                    logger.exception(f"Error during asyncio.gather for prefix '{prefix}': {e}")
                logger.trace(f"Completed processing page for prefix '{prefix}' in bucket '{bucket}'")
        except Exception as e:
            logger.error(f"Error listing objects in bucket '{bucket}' with prefix '{prefix}': {e}")
    logger.trace(f"Completed processing bucket '{bucket}' for windows {windows}")
    return files

async def download_slices_for_buckets_and_windows(buckets: List[str], windows: List[int], key:str = 'slice') -> Dict[int, List[SimpleNamespace]]:
    """
    Downloads files from multiple S3 buckets for the given windows.

    Args:
        buckets (List[str]): A list of S3 bucket names.
        windows (List[int]): A list of window identifiers.

    Returns:
        Dict[int, List[SimpleNamespace]]: A dictionary mapping windows to lists of file metadata and paths.
    """
    logger.debug(f"Downloading files for buckets {set(buckets)} and windows {windows}")
    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:
        tasks = []
        for bucket in set(buckets):
            logger.debug(f'bucket: {bucket}')
            if not bucket:
                continue
            tasks.append(process_bucket(s3_client, bucket, windows, key))
        results = await asyncio.gather(*tasks)
        # Flatten the list of lists
        files = [item for sublist in results for item in sublist]

        # Create a dictionary with windows as keys and list of files as values
        windows_dict = {}
        for file in files:
            window = file.window
            if window not in windows_dict:
                windows_dict[window] = []
            windows_dict[window].append(file)

        logger.debug(f"Downloaded all files grouped by windows: {windows}")
        return windows_dict

async def load_files_for_window(window: int, key: str = 'slice') -> List[str]:
    """
    Retrieves the paths to downloaded window files from the temporary directory.

    Args:
        window (int): The window identifier.

    Returns:
        List[str]: A list of file paths corresponding to the window.
    """
    logger.debug(f"Retrieving files for window {window} from temporary directory")
    temp_dir = tempfile.gettempdir()
    window_files = []
    for filename in os.listdir(temp_dir):
        if filename.startswith(f"{key}-{window}-") and filename.endswith(".pt"):
            window_files.append(os.path.join(temp_dir, filename))
            logger.debug(f"Found file {filename} for window {window}")
    return window_files


async def delete_files_before_window(window_max: int, key: str = 'slice'):
    """
    Deletes all files on the local machine which have a window id before a specific value window_max.

    Args:
        window_max (int): The maximum window id. Files with window ids less than this value will be deleted.
        key (str): The prefix of the files to consider (default is 'slice').
    """
    logger.debug(f"Deleting files with window id before {window_max}")
    temp_dir = tempfile.gettempdir()
    pattern = re.compile(rf"^{re.escape(key)}-(\d+)-.*\.(pt|pt\.lock)$")
    for filename in os.listdir(temp_dir):
        match = pattern.match(filename)
        if match:
            try:
                window_id = int(match.group(1))
                if window_id < window_max:
                    file_path = os.path.join(temp_dir, filename)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.debug(f"Deleted file {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {filename}: {e}")


async def delete_files_from_bucket_before_window(bucket: str, window_max: int, key: str = 'slice'):
    """
    Deletes all files in the specified S3 bucket which have a window id before a specific value window_max.

    Args:
        bucket (str): The name of the S3 bucket.
        window_max (int): The maximum window id. Files with window ids less than this value will be deleted.
        key (str): The prefix of the files to consider (default is 'slice').
    """
    logger.debug(f"Deleting files in bucket {bucket} with window id before {window_max}")
    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:
        try:
            response = await s3_client.list_objects_v2(Bucket=bucket)
            if 'Contents' in response:
                for obj in response['Contents']:
                    filename = obj['Key']
                    match = re.match(rf"^{re.escape(key)}-(\d+)-.*\.(pt|pt\.lock)$", filename)
                    if match:
                        try:
                            window_id = int(match.group(1))
                            if window_id < window_max:
                                await s3_client.delete_object(Bucket=bucket, Key=filename)
                                logger.debug(f"Deleted file {filename} from bucket {bucket}")
                        except Exception as e:
                            logger.error(f"Error deleting file {filename} from bucket {bucket}: {e}")
        except Exception as e:
            logger.error(f"Error listing objects in bucket {bucket}: {e}")

BUCKET_REGEX = re.compile(
    r'^(?=.{3,63}$)(?!.*\.\.)(?!\-)(?!\.)(?!.*\.$)[a-z0-9]+(?:[\.-][a-z0-9]+)*$'
)

ARN_REGEX = re.compile(
    r'^arn:(aws|aws-cn|aws-us-gov):s3-object-lambda:[a-z0-9\-]+:\d{12}:accesspoint[/:][a-zA-Z0-9.\-_]{1,63}$'
    r'|^arn:(aws|aws-cn|aws-us-gov):s3-outposts:[a-z0-9\-]+:\d{12}:outpost[/:][a-zA-Z0-9.\-_]{1,63}[/:]accesspoint[/:][a-zA-Z0-9\-]{1,63}$'
)

def is_valid_bucket(bucket_name: str) -> bool:
    """
    Validates the bucket name against AWS S3 naming conventions and ARN patterns.

    Args:
        bucket_name (str): The name of the S3 bucket.

    Returns:
        bool: True if valid, False otherwise.
    """
    if BUCKET_REGEX.match(bucket_name) or ARN_REGEX.match(bucket_name):
        return True
    logger.error(f"Invalid bucket name: {bucket_name}")
    return False

def validate_bucket_or_exit(bucket_name: str):
    """
    Validates the bucket name and exits the program if invalid.

    Args:
        bucket_name (str): The name of the S3 bucket.
    """
    logger.debug("Validating Bucket name")
    if not is_valid_bucket(bucket_name):
        logger.info(f"Bucket name {bucket_name} is invalid. Please refer to the AWS documentation on naming conventions ")
        sys.exit(1)

__all__ = [
    'get_slices',
    'apply_slices_to_model',
    'upload_slice_for_window',
    'upload_master',
    'process_bucket',
    'download_slices_for_buckets_and_windows',
    'load_files_for_window',
    'delete_files_before_window',
    'delete_files_from_bucket_before_window',
    'is_valid_bucket',
    'validate_bucket_or_exit',
]