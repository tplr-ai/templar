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

# Global imports
import aiofiles
import asyncio
import hashlib
import numpy as np
import os
import re
import tempfile
import torch
import uvloop
from aiobotocore.session import get_session
import bittensor as bt
from collections import defaultdict
from filelock import FileLock, Timeout
from types import SimpleNamespace
from typing import List, Dict

# Local imports
from . import __version__
from .config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    BUCKET_SECRETS,
    client_config,
)
from templar.constants import CF_REGION_NAME
from templar.logging import logger
from templar.schemas import Bucket


def get_base_url(account_id: str) -> str:
    """Gets the base URL for Cloudflare R2 storage.

    Args:
        account_id (str): The Cloudflare account ID

    Returns:
        str: The base URL for R2 storage in the format https://{account_id}.r2.cloudflarestorage.com
    """
    return f"https://{account_id}.r2.cloudflarestorage.com"


def get_bucket(bucket_secrets: dict[str, str | dict[str, str]]) -> Bucket:
    """Creates a Bucket object from bucket secrets configuration.

    Args:
        bucket_secrets (dict[str, str | dict[str, str]]): Dictionary containing bucket configuration with:
            - bucket_name: Name of the bucket
            - account_id: Cloudflare account ID
            - read: Dict containing read access credentials:
                - access_key_id: Access key ID for read operations
                - secret_access_key: Secret access key for read operations

    Returns:
        Bucket: A Bucket object initialized with the provided configuration

    Example:
        >>> secrets = {
        ...     "bucket_name": "my-bucket",
        ...     "account_id": "abc123",
        ...     "read": {
        ...         "access_key_id": "KEY123",
        ...         "secret_access_key": "SECRET123"
        ...     }
        ... }
        >>> bucket = get_bucket(secrets)
    """
    return Bucket(
        name=bucket_secrets["bucket_name"],
        account_id=bucket_secrets["account_id"],
        access_key_id=bucket_secrets["read"]["access_key_id"],
        secret_access_key=bucket_secrets["read"]["secret_access_key"],
    )


# Set uvloop as the event loop policy
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Define a semaphore to limit concurrent downloads (adjust as needed)
semaphore = asyncio.Semaphore(1000)


async def get_slices(filename: str, device: str) -> Dict[str, torch.Tensor]:
    """
    Loads model parameter slices from a file with thread-safe locking.
    Handles missing files gracefully.
    """
    lock_path = f"{filename}.lock"
    try:
        # Check if file exists before trying to acquire lock
        if not os.path.exists(filename):
            logger.warning(f"Slice file not found: {filename}")
            return {}

        lock = FileLock(lock_path)
        with lock.acquire(timeout=1):
            # Check again if file exists after acquiring lock
            if not os.path.exists(filename):
                logger.warning(f"Slice file not found after acquiring lock: {filename}")
                return {}
            try:
                return torch.load(
                    filename,
                    map_location=torch.device(device),
                    weights_only=True,
                )
            except (
                torch.serialization.pickle.UnpicklingError,
                RuntimeError,
                EOFError,
                FileNotFoundError,
            ) as e:
                logger.warning(f"Failed to load slice file {filename}: {e}")
                return {}
            except Exception as e:
                logger.warning(f"Error loading slice file {filename}: {e}")
                return {}
    except Timeout:
        logger.warning(f"Timeout acquiring lock for {filename}")
        return {}
    except Exception as e:
        logger.warning(f"Error during slice loading for {filename}: {e}")
        return {}
    finally:
        # Cleanup lock file if it exists
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception as e:
            logger.warning(f"Failed to remove lock file {lock_path}: {e}")


async def apply_slices_to_model(
    model: torch.nn.Module,
    window: int,
    seed: str,
    compression: int,
    save_location: str,
    key: str = "slice",
) -> int:
    """
    Applies downloaded model parameter slices to a model for a specific window,
    weighting each contribution equally based on the norm of each miner's update
    and preserving the overall parameter scale.

    Args:
        model (torch.nn.Module): The PyTorch model to apply slices to
        window (int): The window number to load slices for
        seed (str): Seed used to determine which parameters to select
        compression (int): Compression factor for parameter selection
        save_location (str): Directory where slices are saved
        key (str, optional): Prefix for the slice files. Defaults to 'slice'.

    Returns:
        int: The maximum global step seen across all applied slices.
    """
    max_global_step = 0
    indices_dict = await get_indices_for_window(model, seed, compression)
    slice_files = await load_files_for_window(
        window=window, save_location=save_location, key=key
    )

    param_sums = {
        name: torch.zeros(
            len(indices_dict[name]), dtype=param.data.dtype, device=model.device
        )
        for name, param in model.named_parameters()
        if name in indices_dict
    }
    slice_norms = []  # Collect norms for computing median
    num_files = 0  # Track the number of valid files

    for file_i in slice_files:
        try:
            filename = os.path.basename(file_i)
            match = re.match(
                rf"^{key}-{window}-.+-v{re.escape(__version__)}\.pt$",
                filename,
            )
            if not match:
                logger.warning(
                    f"Skipping file {file_i} due to version mismatch in filename."
                )
                continue

            slice_i = await get_slices(file_i, model.device)
            slice_global_step = slice_i.get("global_step")

            if slice_global_step is None:
                logger.warning(
                    f"Skipping slice {file_i} because it has no global_step."
                )
                continue

            max_global_step = max(max_global_step, slice_global_step)

            # Compute norm of the slice
            slice_norm = 0.0
            slice_values = {}

            for name, param in model.named_parameters():
                if name not in indices_dict or name not in slice_i:
                    continue
                values = slice_i[name].to(model.device)
                slice_norm += torch.norm(values, p=2).item() ** 2  # Square of L2 norm
                slice_values[name] = values

            slice_norm = (
                np.sqrt(slice_norm) + 1e-8
            )  # Add epsilon to avoid division by zero
            slice_norms.append(slice_norm)  # Collect norm for computing median
            num_files += 1  # Increment valid file count

            # Normalize and accumulate
            for name, values in slice_values.items():
                normalized_values = values / slice_norm
                param_sums[name] += normalized_values

            del slice_i, slice_values

        except Timeout:
            logger.error(f"Timeout occurred while trying to acquire lock on {file_i}")
            continue
        except Exception as e:
            logger.exception(f"Error applying slice from {file_i}: {e}")
            continue

    if not num_files or not slice_norms:
        logger.warning(f"No valid slices found for window {window}")
        return max_global_step

    # Compute median norm
    median_norm = torch.median(torch.tensor(slice_norms))

    # Apply the average of normalized slices to the parameters and scale by median_norm
    for name, param in model.named_parameters():
        if name not in indices_dict:
            continue
        param_indices = indices_dict[name].to(model.device)
        avg_param = param_sums[name] / num_files  # Average normalized slices
        avg_param = avg_param * median_norm  # Scale by median norm
        avg_param = avg_param.to(param.data.dtype)
        param.data.view(-1)[param_indices] = avg_param.clone()

    return max_global_step


async def upload_slice_for_window(
    bucket: str,
    model: torch.nn.Module,
    window: int,
    seed: str,
    wallet: "bt.wallet",
    compression: int,
    save_location: str,
    key: str = "slice",
    global_step: int = 0,
):
    """
    Uploads a slice of model parameters to S3 for a specific window.
    Handles concurrent file operations gracefully.
    """
    filename = f"{key}-{window}-{wallet.hotkey.ss58_address}-v{__version__}.pt"
    logger.debug(f"Uploading slice to S3: {filename}")

    # Prepare the slice data
    indices = await get_indices_for_window(model, seed, compression)

    # Create the slice dictionary with global_step
    slice_data = {"global_step": global_step}
    for name, param in model.named_parameters():
        slice_data[name] = param.data.view(-1)[indices[name].to(model.device)].cpu()

    # Use save_location for temporary file
    temp_file_name = os.path.join(save_location, filename)

    try:
        # Save the file
        torch.save(slice_data, temp_file_name)

        # Upload the file to S3
        session = get_session()
        async with session.create_client(
            "s3",
            endpoint_url=get_base_url(BUCKET_SECRETS["account_id"]),
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
            aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
        ) as s3_client:
            try:
                with open(temp_file_name, "rb") as f:
                    await s3_client.put_object(Bucket=bucket, Key=filename, Body=f)
                logger.debug(f"Successfully uploaded slice to S3: {filename}")
            except Exception as e:
                logger.warning(f"Failed to upload slice {filename} to S3: {str(e)}")
                # Don't raise, allow process to continue
    except Exception as e:
        logger.warning(
            f"Error during slice preparation/upload for {filename}: {str(e)}"
        )
        # Don't raise, allow process to continue
    finally:
        # Clean up the temporary file if it exists
        try:
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)
                logger.debug(f"Temporary file {temp_file_name} removed")
        except Exception as e:
            logger.warning(
                f"Failed to remove temporary file {temp_file_name}: {str(e)}"
            )
            # Don't raise, allow process to continue


async def upload_master(bucket: str, model: torch.nn.Module, wallet: "bt.wallet"):
    """
    Uploads the master PyTorch model to an S3 bucket.

    Args:
        bucket (str): Name of the S3 bucket.
        model (torch.nn.Module): The PyTorch model to be uploaded.
        wallet (bt.wallet): The wallet object containing the hotkey.
    """
    upload_filename = f"master-{wallet.hotkey.ss58_address}.pt"
    logger.debug(f"Uploading master model to S3: {upload_filename}")

    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=get_base_url(BUCKET_SECRETS["account_id"]),
        region_name=CF_REGION_NAME,
        config=client_config,
        aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
        aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
    ) as s3_client:
        try:
            # Create a temporary file and write the model state dictionary to it
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                torch.save(model.state_dict(), temp_file)
                temp_file_name = temp_file.name

            # Upload the file to S3
            with open(temp_file_name, "rb") as f:
                await s3_client.put_object(Bucket=bucket, Key=upload_filename, Body=f)
            logger.debug(f"Successfully uploaded master model to S3: {upload_filename}")
        except Exception:
            logger.exception(f"Failed to upload master model {upload_filename} to S3")
        finally:
            # Clean up the temporary file
            os.remove(temp_file_name)
            logger.debug(f"Temporary file {temp_file_name} removed")


async def get_indices_for_window(
    model: torch.nn.Module, seed: str, compression: int
) -> Dict[str, torch.LongTensor]:
    """
    Computes the indices for the given window and compression factor.

    Args:
        model (torch.nn.Module): The PyTorch model.
        seed (str): The window seed identifier.
        compression (int): The compression factor.

    Returns:
        Dict[str, torch.LongTensor]: A dictionary mapping parameter names to index tensors.
    """
    logger.debug(
        f"Computing indices for window seed {seed} with compression {compression}"
    )
    result = {}
    # Seed the random number generator with the seed
    seed = int(hashlib.md5(str(seed).encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    for name, param in model.named_parameters():
        # Randomly select indices based on the compression factor
        num_indices = max(1, int(param.numel() // compression))
        indices = rng.choice(param.numel(), size=num_indices, replace=False)
        result[name] = torch.from_numpy(indices).long().cpu()
    return result


async def download_file(
    s3_client, bucket: str, filename: str, save_location: str
) -> str:
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
        temp_file = os.path.join(save_location, filename)
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
                CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB

                response = await s3_client.get_object(Bucket=bucket, Key=filename)
                async with aiofiles.open(temp_file, "wb") as outfile:
                    while True:
                        chunk = await response["Body"].read(CHUNK_SIZE)
                        if not chunk:
                            break
                        await outfile.write(chunk)

                logger.debug(f"Successfully downloaded file {filename} to {temp_file}")
                return temp_file

        except Timeout:
            logger.error(
                f"Timeout occurred while trying to acquire lock on {lock_file}"
            )
            return None
        except Exception as e:
            logger.exception(
                f"Failed to download file {filename} from bucket {bucket}: {e}"
            )
            return None
        finally:
            # The lock is automatically released when exiting the 'with' block
            pass


async def handle_file(
    s3_client,
    bucket: str,
    filename: str,
    hotkey: str,
    window: int,
    version: str,
    save_location: str,
):
    """
    Handles downloading a single file from S3.

    Args:
        s3_client: The S3 client.
        bucket (str): Name of the S3 bucket.
        filename (str): The S3 object key (filename).
        hotkey (str): The hotkey identifier.
        window (int): The window identifier.
        version (str): The version extracted from the filename.

    Returns:
        SimpleNamespace: An object containing file metadata and the path to the downloaded file,
                         including the version.
    """
    logger.debug(
        f"Handling file '{filename}' for window {window} and hotkey '{hotkey}'"
    )
    temp_file = await download_file(s3_client, bucket, filename, save_location)
    if temp_file:
        return SimpleNamespace(
            bucket=bucket,
            hotkey=hotkey,
            filename=filename,
            window=window,
            temp_file=temp_file,
            version=version,
        )
    return None


async def process_bucket(
    s3_client, bucket: str, windows: List[int], key: str, save_location: str
):
    """
    Processes a single S3 bucket to download files for specified windows.

    Args:
        s3_client: The S3 client to use for operations.
        bucket (str): Name of the S3 bucket to process.
        windows (List[int]): List of window IDs to download files for.
        key (str, optional): Prefix to filter files by. Defaults to 'slice'.

    Returns:
        List[SimpleNamespace]: List of downloaded file metadata objects containing:
            - bucket: The S3 bucket name.
            - hotkey: The hotkey identifier.
            - filename: The original S3 object key.
            - window: The window ID.
            - temp_file: Path to the downloaded file.
            - version: Extracted version from the filename.

    The function:
    1. Validates the bucket name.
    2. For each window:
        - Lists objects with matching prefix.
        - Parses filenames to extract metadata.
        - Downloads matching files concurrently.
        - Handles version checking and error cases.
    3. Returns list of successfully downloaded files.
    """
    # Import the required modules
    import re
    from templar import __version__  # Ensure __version__ is imported

    logger.debug(f"Processing bucket '{bucket}' for windows {windows}")
    files = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for window in windows:
        prefix = f"{key}-{window}"
        logger.debug(f"Listing objects with prefix '{prefix}' in bucket '{bucket}'")
        try:
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                logger.trace(
                    f"Processing page for prefix '{prefix}' in bucket '{bucket}'"
                )
                if "Contents" not in page:
                    logger.trace(
                        f"No contents found for prefix '{prefix}' in bucket '{bucket}'"
                    )
                    continue
                download_tasks = []
                for obj in page.get("Contents", []):
                    filename = obj["Key"]
                    logger.trace(
                        f"Processing object with key '{filename}' in bucket '{bucket}'"
                    )
                    try:
                        # Extract hotkey and version from the filename using non-greedy matching
                        match = re.match(rf"^{key}-{window}-(.+?)-v(.+)\.pt$", filename)
                        if not match:
                            logger.error(
                                f"Filename '{filename}' does not conform to the expected format."
                            )
                            continue
                        slice_hotkey = match.group(1)
                        slice_version = match.group(2)

                        # Compare version with the expected version
                        if slice_version != __version__:
                            logger.warning(
                                f"Skipping file '{filename}' due to version mismatch "
                                f"(expected {__version__}, got {slice_version})."
                            )
                            continue
                        logger.trace(
                            f"Parsed filename '{filename}' into window '{window}', "
                            f"hotkey '{slice_hotkey}', and version '{slice_version}'"
                        )
                        # Add the download task, passing the version
                        download_tasks.append(
                            handle_file(
                                s3_client,
                                bucket,
                                filename,
                                slice_hotkey,
                                window,
                                slice_version,
                                save_location,
                            )
                        )
                    except ValueError:
                        logger.exception(f"Error parsing filename '{filename}'")
                        continue
                    except Exception as e:
                        logger.exception(
                            f"Unexpected error processing filename '{filename}': {e}"
                        )
                        continue
                # Download the files concurrently
                try:
                    results = await asyncio.gather(
                        *download_tasks, return_exceptions=True
                    )
                    for res in results:
                        if isinstance(res, Exception):
                            logger.error(f"Download task failed: {res}")
                        elif res:
                            files.append(res)
                    logger.trace(
                        f"Completed processing page for prefix '{prefix}' in bucket '{bucket}'"
                    )
                except Exception as e:
                    logger.exception(
                        f"Error during asyncio.gather for prefix '{prefix}': {e}"
                    )
        except Exception as e:
            logger.error(
                f"Error listing objects in bucket '{bucket}' with prefix '{prefix}': {e}"
            )
    logger.trace(f"Completed processing bucket '{bucket}' for windows {windows}")
    return files


async def download_slices_for_buckets_and_windows(
    buckets: List[Bucket], windows: List[int], key: str, save_location: str
) -> Dict[int, List[SimpleNamespace]]:
    """Downloads model slices from multiple S3 buckets for specified windows.

    This function downloads model slice files from a list of S3 buckets for the given window IDs.
    It processes the buckets concurrently and combines the results into a dictionary mapping
    window IDs to lists of downloaded slices.

    Args:
        buckets (List[Bucket]): List of Bucket objects containing S3 credentials and configuration
        windows (List[int]): List of window IDs to download slices for
        key (str, optional): Prefix to filter files by. Defaults to "slice"

    Returns:
        Dict[int, List[SimpleNamespace]]: Dictionary mapping window IDs to lists of downloaded slices.
            Each slice is represented as a SimpleNamespace object containing metadata and file path.

    Example:
        >>> buckets = [Bucket(...), Bucket(...)]  # List of bucket configs
        >>> windows = [1, 2, 3]  # Window IDs to download
        >>> slices = await download_slices_for_buckets_and_windows(buckets, windows)
        >>> print(slices[1])  # Get all slices for window 1
        [Slice(path='/tmp/slice-1-abc.pt'), Slice(path='/tmp/slice-1-def.pt')]

    Note:
        - Filters out None buckets from input list
        - Downloads files concurrently across buckets
        - Uses CloudFront for downloads if configured
        - Handles S3 authentication using bucket credentials
        - Returns empty dict if no valid buckets provided
    """
    # Filter out None buckets
    valid_buckets = []
    for b in buckets:
        if b is None:
            continue
        if isinstance(b, str):
            logger.warning(f"Received string instead of Bucket object: {b}")
            continue
        if not isinstance(b, Bucket):
            logger.warning(f"Invalid bucket type: {type(b)}")
            continue
        valid_buckets.append(b)

    if not valid_buckets:
        logger.warning("No valid buckets provided")
        return {}

    try:
        logger.debug(
            f"Downloading files for buckets {[b.name for b in valid_buckets]} and windows {windows}"
        )
    except Exception as e:
        logger.error(f"Error logging bucket names: {e}")
        return {}

    session = get_session()
    tasks = []
    for bucket in set(valid_buckets):
        async with session.create_client(
            "s3",
            endpoint_url=get_base_url(bucket.account_id),
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=bucket.access_key_id,
            aws_secret_access_key=bucket.secret_access_key,
        ) as s3_client:
            logger.debug(f"Processing bucket: {bucket.name}")
            tasks.append(
                process_bucket(s3_client, bucket.name, windows, key, save_location)
            )

    results = await asyncio.gather(*tasks)
    # Combine results into a dictionary mapping window IDs to lists of slices
    slices = defaultdict(list)
    for result in results:
        for item in result:
            slices[item.window].append(item)
    return slices


async def load_files_for_window(
    window: int, save_location: str, key: str = "slice"
) -> List[str]:
    """
    Loads files for a specific window from the temporary directory.

    Args:
        window (int): The window identifier to load files for
        key (str, optional): The prefix to filter files by. Defaults to 'slice'.

    Returns:
        List[str]: A list of full file paths matching the window and key pattern

    Example:
        >>> files = await load_files_for_window(123, 'state')
        >>> print(files)
        ['/tmp/state-123-abc-v1.0.0.pt', '/tmp/state-123-def-v1.0.0.pt']

    Note:
        - Only returns files matching pattern: {key}-{window}-*-v{version}.pt
        - Files must be in the system temp directory
        - Version number is pulled from templar.__version__
    """

    logger.debug(f"Retrieving files for window {window} from temporary directory")
    temp_dir = save_location
    window_files = []
    pattern = re.compile(rf"^{key}-{window}-.+-v{__version__}\.pt$")
    for filename in os.listdir(temp_dir):
        if pattern.match(filename):
            window_files.append(os.path.join(temp_dir, filename))
            logger.debug(f"Found file {filename} for window {window}")
    return window_files


async def delete_files_before_window(
    window_max: int, save_location: str, key: str = "slice"
):
    """
    Deletes temporary files with window IDs less than the specified maximum.

    Args:
        window_max (int): Maximum window ID to keep. Files with window IDs less than this will be deleted
        key (str, optional): The prefix to filter files by. Defaults to 'slice'

    Example:
        >>> await delete_files_before_window(100, 'state')
        # Deletes all state-*.pt files with window < 100

    Note:
        - Deletes both .pt and .pt.lock files
        - Only deletes files matching pattern: {key}-{window}-*-v{version}.pt
        - Files must be in system temp directory
        - Version number is pulled from templar.__version__
    """

    logger.debug(f"Deleting files with window id before {window_max}")
    temp_dir = save_location
    pattern = re.compile(rf"^{re.escape(key)}-(\d+)-.+-v{__version__}\.(pt|pt\.lock)$")
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


async def delete_files_from_bucket_before_window(
    bucket: str, window_max: int, key: str = "slice"
):
    """
    Deletes files from an S3 bucket with window IDs less than the specified maximum.

    Args:
        bucket (str): Name of the S3 bucket to delete files from
        window_max (int): Maximum window ID to keep. Files with window IDs less than this will be deleted
        key (str, optional): The prefix to filter files by. Defaults to 'slice'

    Example:
        >>> await delete_files_from_bucket_before_window('my-bucket', 100, 'state')
        # Deletes all state-*.pt files with window < 100 from my-bucket

    Note:
        - Deletes both .pt and .pt.lock files
        - Only deletes files matching pattern: {key}-{window}-*-v{version}.pt
        - Version number is pulled from templar.__version__
        - Requires valid AWS credentials and bucket permissions
    """

    logger.debug(
        f"Deleting files in bucket {bucket} with window id before {window_max}"
    )
    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=get_base_url(BUCKET_SECRETS["account_id"]),
        region_name=CF_REGION_NAME,
        config=client_config,
        aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
        aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
    ) as s3_client:
        try:
            response = await s3_client.list_objects_v2(Bucket=bucket)
            if "Contents" in response:
                for obj in response["Contents"]:
                    filename = obj["Key"]
                    match = re.match(
                        rf"^{re.escape(key)}-(\d+)-.+-v{__version__}\.(pt|pt\.lock)$",
                        filename,
                    )
                    if match:
                        try:
                            window_id = int(match.group(1))
                            if window_id < window_max:
                                await s3_client.delete_object(
                                    Bucket=bucket, Key=filename
                                )
                                logger.debug(
                                    f"Deleted file {filename} from bucket {bucket}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error deleting file {filename} from bucket {bucket}: {e}"
                            )
        except Exception as e:
            logger.error(f"Error listing objects in bucket {bucket}: {e}")


BUCKET_REGEX = re.compile(
    r"^(?=.{3,63}$)(?!.*\.\.)(?!\-)(?!\.)(?!.*\.$)[a-z0-9]+(?:[\.-][a-z0-9]+)*$"
)

ARN_REGEX = re.compile(
    r"^arn:(aws|aws-cn|aws-us-gov):s3-object-lambda:[a-z0-9\-]+:\d{12}:accesspoint[/:][a-zA-Z0-9.\-_]{1,63}$"
    r"|^arn:(aws|aws-cn|aws-us-gov):s3-outposts:[a-z0-9\-]+:\d{12}:outpost[/:][a-zA-Z0-9.\-_]{1,63}[/:]accesspoint[/:][a-zA-Z0-9\-]{1,63}$"
)


async def delete_old_version_files(bucket_name: str, current_version: str):
    """
    Deletes files from the S3 bucket that do not match the current version.

    Args:
        bucket_name (str): The name of the S3 bucket.
        current_version (str): The current version string.
    """
    session = get_session()
    async with session.create_client(
        "s3",
        region_name="us-east-1",
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    ) as s3_client:
        paginator = s3_client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket_name):
            to_delete = []
            for obj in page.get("Contents", []):
                filename = obj["Key"]
                # Check if the file version matches the current version
                match = re.match(r".+-v(.+)\.pt$", filename)
                if match:
                    file_version = match.group(1)
                    if file_version != current_version:
                        to_delete.append({"Key": filename})
                        logger.debug(f"Scheduled for deletion: {filename}")
            # Delete old versions in batches of 1000 (S3 limit for delete_objects)
            if to_delete:
                response = await s3_client.delete_objects(
                    Bucket=bucket_name, Delete={"Objects": to_delete}
                )
                deleted = response.get("Deleted", [])
                logger.info(
                    f"Deleted {len(deleted)} old version files from bucket {bucket_name}"
                )


# def is_valid_bucket(bucket_name: str) -> bool:
#     """
#     Validates if the bucket name matches AWS S3 bucket naming rules
#     and checks if the bucket exists and is accessible.

#     Args:
#         bucket_name (str): The bucket name to validate.

#     Returns:
#         bool: True if valid and accessible, False otherwise.
#     """
#     # Ensure bucket_name is a string
#     if isinstance(bucket_name, bytes):
#         bucket_name = bucket_name.decode('utf-8')

#     # # Check if the bucket name matches the regex
#     # if not (BUCKET_REGEX.match(bucket_name) or ARN_REGEX.match(bucket_name)):
#     #     logger.debug(f"Invalid bucket name format: {bucket_name}")
#     #     return False

#     # Create S3 client
#     s3_client = boto3.client(
#         's3',
#         region_name=CF_REGION_NAME,
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         config=client_config
#     )

#     # Check if the bucket exists and is accessible
#     try:
#         # Try to list objects in the bucket
#         s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
#         logger.debug(f"Bucket '{bucket_name}' exists and is accessible.")
#         return True  # Bucket exists and is accessible
#     except ClientError as e:
#         error_code = e.response['Error']['Code']
#         if error_code in ['NoSuchBucket', '404']:
#             logger.debug(f"Bucket '{bucket_name}' does not exist.")
#         elif error_code in ['AccessDenied', '403']:
#             logger.debug(f"Access denied for bucket '{bucket_name}'.")
#         elif error_code == 'AllAccessDisabled':
#             logger.debug(f"All access disabled for bucket '{bucket_name}'.")
#         else:
#             logger.debug(f"Error accessing bucket '{bucket_name}': {e}")
#         return False
#     except Exception as e:
#         logger.debug(f"Unexpected error when accessing bucket '{bucket_name}': {e}")
#         return False

# def validate_bucket_or_exit(bucket_name: str):
#     """
#     Validates the bucket name and exits the program if invalid.

#     Args:
#         bucket_name (str): The name of the S3 bucket.
#     """
#     logger.debug("Validating Bucket name")
#     if not is_valid_bucket(bucket_name):
#         logger.error(f"Bucket name {bucket_name} is invalid. Please refer to the AWS documentation on naming conventions ")
#         sys.exit(1)


async def save_checkpoint(
    filename, model, optimizer=None, scheduler=None, global_step=0, **kwargs
):
    """
    Saves the checkpoint to the specified filename asynchronously.

    Args:
        filename (str): Path to save the checkpoint.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer, optional): Optimizer to save.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to save.
        global_step (int): The current global step.
        **kwargs: Additional state variables to save.
    """
    # Gather the checkpoint data
    checkpoint = {
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
    }
    if optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    # Include additional state variables
    for key, value in kwargs.items():
        checkpoint[key] = value

    # Save the checkpoint asynchronously to avoid blocking the main thread
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, torch.save, checkpoint, filename)
    torch.save(checkpoint, filename)


async def load_checkpoint(
    filename, model, optimizer=None, scheduler=None, device="cpu"
):
    """
    Loads the checkpoint from the specified filename.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load the state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load the state into.
        device (str): Device to map the checkpoint.

    Returns:
        global_step (int): The global step at which the checkpoint was saved.
        additional_state (dict): Dictionary of additional state variables.
    """
    try:
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        global_step = checkpoint.get("global_step", 0)
        additional_state = {
            k: checkpoint[k]
            for k in checkpoint
            if k
            not in [
                "global_step",
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
            ]
        }
        return global_step, additional_state
    except (torch.serialization.pickle.UnpicklingError, RuntimeError, EOFError) as e:
        logger.error(f"Checkpoint at {filename} is corrupt: {e}")
        # Return global_step as 0 and an empty additional_state
        return 0, {}
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {filename}: {e}")
        return 0, {}


def get_neuron_temp_dir(wallet) -> str:
    """
    Returns a unique temporary directory for the neuron based on its wallet hotkey.
    """

    temp_dir = os.path.join(
        tempfile.gettempdir(), f"neuron_{wallet.hotkey.ss58_address}"
    )
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir
