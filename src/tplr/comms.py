# The MIT License (MIT)
# © 2024 templar.tech

# Global imports
import time
import aiofiles
import asyncio
import os
import torch
from aiobotocore.session import get_session
import bittensor as bt
from typing import List, Dict, Optional

# Local imports
from . import __version__
from .config import (
    client_config,
    BUCKET_SECRETS
)
from .chain import ChainManager
from .logging import logger
from .schemas import Bucket

CF_REGION_NAME: str = "enam"

class Comms(ChainManager):
    def __init__(
        self,
        bucket: Bucket,
        save_location: str = '/tmp',
        key_prefix: str = 'slice',
        subtensor: Optional["bt.Subtensor"] = None,
        netuid: Optional[int] = None,
        metagraph = None,
    ):
        # Initialize parent class
        ChainManager.__init__(self, subtensor, netuid, metagraph)
        """
        Initializes the R2Communicator for interacting with Cloudflare R2 buckets.

        Args:
            bucket (Bucket): Contains the configuration for the R2 bucket.
            save_location (str): Temporary directory for file operations.
            key_prefix (str): Prefix for the filenames stored in the bucket.
            subtensor (bt.Subtensor, optional): Subtensor instance for chain operations
            netuid (int, optional): Network UID for chain operations
            metagraph: Metagraph instance containing network state
        """
        self.bucket = bucket
        self.save_location = save_location
        self.key_prefix = key_prefix
        self.session = get_session()
        self.lock = asyncio.Lock()

        # Update neuron's bucket commitment
        self.try_commit()
        # Start the background task to fetch commitments
        self.start_commitment_fetcher()
        logger.debug("Started commitment fetcher background task.")

    def get_base_url(self) -> str:
        """Constructs the base URL for the R2 storage endpoint."""
        return f"https://{self.bucket.account_id}.r2.cloudflarestorage.com"

    async def put(
        self,
        state_dict: dict,
        uid: str,
        window: int,
        key: Optional[str] = None,
        global_step: int = 0,
    ):
        """
        Uploads a slice of the model parameters to the R2 bucket.

        Args:
            state_dict (dict): The state dictionary to upload.
            uid (str): Unique identifier for the upload (e.g., hotkey or user ID).
            window (int): The window number for synchronization.
            key (str, optional): Custom key for the filename. Defaults to self.key_prefix.
            global_step (int): Global training step.
        """
        key = key or self.key_prefix
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        temp_file_path = os.path.join(self.save_location, filename)
        # Include global_step in state_dict
        state_dict["global_step"] = global_step
        # Save the state_dict to a temporary file
        torch.save(state_dict, temp_file_path)

        # Upload the file to R2 bucket
        async with self.session.create_client(
            "s3",
            endpoint_url=self.get_base_url(BUCKET_SECRETS["account_id"]),
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
            aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
        ) as s3_client:
            try:
                async with aiofiles.open(temp_file_path, "rb") as f:
                    data = await f.read()
                    await s3_client.put_object(Bucket=self.bucket.name, Key=filename, Body=data)
                logger.debug(f"Successfully uploaded {filename} to R2 bucket.")
            except Exception as e:
                logger.error(f"Failed to upload {filename} to R2 bucket: {e}")
            finally:
                # Clean up the temporary file
                os.remove(temp_file_path)

    async def get(
        self,
        uid: str,
        window: int,
        key: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Downloads a slice of the model parameters from the R2 bucket.

        Args:
            uid (str): Unique identifier for the download.
            window (int): The window number for synchronization.
            key (str, optional): Custom key for the filename. Defaults to self.key_prefix.
            timeout (int): Timeout in seconds for the download operation.

        Returns:
            dict: The state dictionary downloaded from the bucket.
        """
        key = key or self.key_prefix
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        temp_file_path = os.path.join(self.save_location, filename)

        # Get bucket credentials for this uid
        bucket = self.chain.get_bucket(uid)

        async with self.session.create_client(
            "s3",
            endpoint_url=self.get_base_url(bucket.account_id),
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=bucket.access_key_id,
            aws_secret_access_key=bucket.secret_access_key,
        ) as s3_client:
            try:
                # Use asyncio timeout
                async with asyncio.timeout(timeout):
                    response = await s3_client.get_object(Bucket=self.bucket.name, Key=filename)
                    async with aiofiles.open(temp_file_path, "wb") as f:
                        while True:
                            chunk = await response["Body"].read(1024 * 1024)  # Read 1 MB chunks
                            if not chunk:
                                break
                            await f.write(chunk)
                # Load the state_dict
                state_dict = torch.load(temp_file_path, map_location="cpu")
                logger.debug(f"Successfully downloaded {filename} from R2 bucket.")
                return state_dict
            except asyncio.TimeoutError:
                logger.error(f"Timeout while downloading {filename} from R2 bucket.")
                return None
            except Exception as e:
                logger.error(f"Failed to download {filename} from R2 bucket: {e}")
                return None
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    async def get_with_retry(
        self,
        uid: str,
        window: int,
        key: Optional[str] = None,
        timeout: int = 30,
        retry_interval: float = 0.1,
    ):
        """
        Attempts to download data from the R2 bucket, retrying until success or timeout.

        Args:
            uid (str): Unique identifier for the download.
            window (int): The window number for synchronization.
            key (str, optional): Custom key for the filename.
            timeout (int): Total timeout duration for retries.
            retry_interval (float): Time to wait between retries.

        Returns:
            dict: The state dictionary downloaded from the bucket.
        """
        start_time = time.time()
        while True:
            state_dict = await self.get(uid, window, key, timeout)
            if state_dict is not None:
                return state_dict
            if time.time() - start_time > timeout:
                logger.error(f"Exceeded timeout while downloading data for UID {uid}.")
                return None
            await asyncio.sleep(retry_interval)

    async def gather(
        self,
        state_dict: Dict[str, torch.Tensor],
        my_uid: str,
        uids: List[str],
        window: int,
        key: Optional[str] = None,
        timeout: int = 30,
        device: str = "cpu",
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Gathers slices from multiple peers and assembles them for aggregation.

        Args:
            state_dict (Dict[str, torch.Tensor]): Local state dictionary.
            my_uid (str): This node's unique identifier.
            uids (List[str]): List of peer UIDs to gather data from.
            window (int): The window number for synchronization.
            key (str, optional): Custom key for filenames.
            timeout (int): Timeout for gathering data from each peer.
            device (str): Device to map tensors onto.

        Returns:
            Dict[str, List[torch.Tensor]]: Aggregated state dictionaries from all peers.
        """
        key = key or self.key_prefix
        # Put own state_dict to the bucket
        await self.put(state_dict, my_uid, window, key)

        # Gather state_dicts from peers
        gather_tasks = [
            self.get_with_retry(uid=uid, window=window, key=key, timeout=timeout)
            for uid in uids
        ]

        responses = await asyncio.gather(*gather_tasks)
        # Initialize the gather_result dictionary
        gather_result = {
            param_name: [] for param_name in state_dict.keys()
        }
        # Assemble the results
        for idx, peer_state in enumerate(responses):
            if peer_state is None:
                # Handle missing peer data, e.g., fill with zeros or skip
                for param_name in state_dict.keys():
                    gather_result[param_name].append(torch.zeros_like(state_dict[param_name]).to(device))
            else:
                for param_name in state_dict.keys():
                    gather_result[param_name].append(peer_state[param_name].to(device))

        return gather_result
