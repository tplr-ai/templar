import os
import asyncio
import aiofiles
import torch
import numpy as np
from .schemas import Bucket
from typing import Dict, Optional, List
from aiobotocore.session import get_session

from . import __version__
from .config import BUCKET_SECRETS, client_config
from .constants import CF_REGION_NAME
from .commitment import get_all_commitments
from .logging import logger
import bittensor as bt

def get_base_url(account_id: str) -> str:
    return f"https://{account_id}.r2.cloudflarestorage.com"

def get_all_buckets(
    subtensor: bt.Subtensor,
    netuid: int,
    metagraph,
) -> Dict[str, Optional[Bucket]]:
    """
    Retrieves and parses all bucket commitments from the network.

    Returns:
        Dict[str, Optional[Bucket]]: Mapping of neuron hotkeys to their Bucket objects.
    """
    # Fetch all commitments
    commitments = get_all_commitments(
        substrate=subtensor.substrate,
        netuid=netuid,
        metagraph=metagraph,
    )

    # Map UIDs to hotkeys
    uid_to_hotkey = dict(zip(metagraph.uids.tolist(), metagraph.hotkeys))
    hotkey_to_bucket = {}
    for uid, bucket in commitments.items():
        hotkey = uid_to_hotkey.get(uid)
        if hotkey:
            hotkey_to_bucket[hotkey] = bucket
        else:
            hotkey_to_bucket[hotkey] = None
    return hotkey_to_bucket

async def save_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    global_step: int = 0,
    **kwargs
):
    """
    Saves the checkpoint to the specified filename asynchronously.
    """
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

    # Save the checkpoint asynchronously to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, torch.save, checkpoint, filename)
    logger.info(f"Checkpoint saved at {filename}")

async def load_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    device: str = "cpu"
):
    """
    Loads the checkpoint from the specified filename.
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
            if k not in ["global_step", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict"]
        }
        return global_step, additional_state
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {filename}: {e}")
        return 0, {}

async def upload_checkpoint(
    bucket: str,
    checkpoint_path: str,
    wallet: bt.wallet,
    key_prefix: str = "neuron_checkpoints",
):
    """
    Uploads the checkpoint file to S3 storage.
    """
    filename = f"{key_prefix}/{wallet.hotkey.ss58_address}-v{__version__}.pth"
    logger.debug(f"Uploading checkpoint to S3: {filename}")

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
            with open(checkpoint_path, "rb") as f:
                await s3_client.put_object(Bucket=bucket, Key=filename, Body=f)
            logger.debug(f"Successfully uploaded checkpoint to S3: {filename}")
        except Exception as e:
            logger.exception(f"Failed to upload checkpoint to S3: {filename}. Error: {e}")

async def download_checkpoint_from_neuron(
    bucket_info: Bucket,
    neuron_hotkey: str,
    checkpoint_dir: str,
    key_prefix: str = "neuron_checkpoints",
) -> Optional[str]:
    """
    Downloads the checkpoint file from the neuron's S3 storage.

    Args:
        bucket_info (Bucket): The neuron's bucket credentials.
        neuron_hotkey (str): Hotkey of the neuron.
        checkpoint_dir (str): Directory to save the checkpoint.
        key_prefix (str, optional): S3 key prefix. Defaults to 'neuron_checkpoints'.

    Returns:
        Optional[str]: Path to the downloaded checkpoint file, or None if failed.
    """
    filename = f"{key_prefix}/{neuron_hotkey}-v{__version__}.pth"
    local_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{neuron_hotkey}.pth")
    logger.debug(f"Downloading checkpoint from S3: {filename}")

    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=get_base_url(bucket_info.account_id),
        region_name=CF_REGION_NAME,
        config=client_config,
        aws_access_key_id=bucket_info.access_key_id,
        aws_secret_access_key=bucket_info.secret_access_key,
    ) as s3_client:
        try:
            response = await s3_client.get_object(Bucket=bucket_info.name, Key=filename)
            async with aiofiles.open(local_checkpoint_path, 'wb') as f:
                await f.write(await response['Body'].read())
            logger.debug(f"Successfully downloaded checkpoint: {local_checkpoint_path}")
            return local_checkpoint_path
        except Exception as e:
            logger.exception(f"Failed to download checkpoint from S3: {filename}. Error: {e}")
            return None

def get_neuron_with_highest_stake(
    metagraph,
    exclude_hotkeys: Optional[List[str]] = None
) -> Optional[str]:
    """
    Retrieves the hotkey of the neuron with the highest stake.
    """
    try:
        stakes = metagraph.S  # Stake values for all neurons
        if not stakes.any():
            logger.warning("Stake values are empty.")
            return None

        exclude_indices = []
        if exclude_hotkeys:
            hotkey_to_index = {hotkey: idx for idx, hotkey in enumerate(metagraph.hotkeys)}
            exclude_indices = [hotkey_to_index[hotkey] for hotkey in exclude_hotkeys if hotkey in hotkey_to_index]
            stakes[exclude_indices] = -np.inf

        highest_stake_uid = stakes.argmax()
        highest_stake_hotkey = metagraph.hotkeys[highest_stake_uid]
        return highest_stake_hotkey
    except Exception as e:
        logger.exception(f"Error fetching neuron with highest stake: {e}")
        return None