# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# type: ignore

# Global imports
import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

import bittensor as bt
import numpy as np
import torch
import websockets.exceptions
from bittensor import Wallet
from bittensor.core.chain_data import decode_account_id
from pydantic import ValidationError

# Local imports
from .logging import logger
from .schemas import Bucket


class ChainManager:
    """Base class for handling chain interactions."""

    def __init__(
        self,
        config,
        netuid: Optional[int] = None,
        metagraph=None,
        hparams=None,
        fetch_interval: int = 600,  # Fetch interval in seconds
        wallet: Optional["bt.wallet"] = None,
        bucket: Optional[Bucket] = None,
    ):
        """
        Initialize chain commitment handler.

        Args:
            subtensor (bt.Subtensor): Subtensor instance for chain operations
            netuid (int): Network UID for chain operations
            metagraph: Metagraph instance containing network state
            hparams: Hyperparameters namespace containing model configuration
            fetch_interval (int): Interval in seconds between fetching commitments
            wallet (bt.wallet, optional): Wallet to sign commitments
            bucket (Bucket, optional): Bucket configuration to commit
        """
        self.config = config
        self.netuid = netuid
        self.metagraph = metagraph
        self.hparams = hparams or {}

        # Block and window tracking - now centralized
        self.current_block = 0
        self.current_window = 0
        self.window_duration = self.hparams.blocks_per_window
        self.window_time = 0
        self.window_seeds = {}

        # Events for block/window updates
        self.block_event = asyncio.Event()
        self.new_window_event = asyncio.Event()
        self.stop_event = asyncio.Event()

        # Block listener task management
        self._block_listener_task: Optional[asyncio.Task] = None

        # Initialize bucket storage
        self.commitments = {}
        self.peers = []
        self.eval_peers = defaultdict(int)
        self.fetch_interval = fetch_interval
        self._fetch_task = None

        # Store wallet and bucket
        self.wallet = wallet
        self.bucket = bucket

    def start_block_listener(self) -> None:
        """Start the centralized block listener if not already running."""
        if self._block_listener_task is None or self._block_listener_task.done():
            self._block_listener_task = asyncio.create_task(self._block_listener_loop())
            logger.info("Started block listener")

    async def stop_block_listener(self) -> None:
        """Stop the centralized block listener."""
        self.stop_event.set()
        if self._block_listener_task and not self._block_listener_task.done():
            self._block_listener_task.cancel()
            try:
                await self._block_listener_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped  block listener")

    async def _block_listener_loop(self) -> None:
        """
        Centralized block listener that maintains current_block and current_window.
        Handles reconnection and errors with exponential backoff.
        """
        backoff = 1
        max_backoff = 60

        def handler(event):
            try:
                old_block = self.current_block
                old_window = self.current_window
                
                self.current_block = int(event["header"]["number"])
                new_window = int(self.current_block / self.hparams.blocks_per_window)
                
                # Set block event for any block update
                self.block_event.set()
                
                # Check for window transition
                if new_window != self.current_window:
                    self.current_window = new_window
                    self.new_window_event.set()
                    logger.info(
                        f"Window transition: {old_window} -> {self.current_window} "
                        f"(block: {old_block} -> {self.current_block})"
                    )
                
                # Clear events after setting them so they can be waited on again
                self.block_event.clear()
                if new_window != old_window:
                    self.new_window_event.clear()
                    
            except Exception as e:
                logger.error(f"Error processing block event: {e}")

        while not self.stop_event.is_set():
            try:
                subtensor = bt.subtensor(config=self.config)
                await asyncio.to_thread(
                    subtensor.substrate.subscribe_block_headers, handler
                )
                backoff = 1  # Reset backoff on successful connection
                break
            except websockets.exceptions.ConnectionClosedError as e:
                logger.warning(
                    f"Websocket ConnectionClosedError: {e}. Retrying in {backoff}s"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            except Exception as e:
                logger.error(f"Block subscription error: {e}. Retrying in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def get_block_timestamp(self, block_number: int) -> Optional[datetime]:
        """
        Query blockchain timestamp for a specific block number.
        
        Args:
            block_number: The block number to query timestamp for
            
        Returns:
            Timezone-aware datetime object or None on failure
        """
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                subtensor = bt.subtensor(config=self.config)
                
                # Query the timestamp module
                timestamp_result = await asyncio.to_thread(
                    subtensor.query_module,
                    "Timestamp", 
                    "Now",
                    block=block_number
                )
                
                if timestamp_result is not None:
                    # Handle ScaleObj parsing
                    if hasattr(timestamp_result, 'value'):
                        timestamp_ms = timestamp_result.value
                    else:
                        timestamp_ms = timestamp_result
                    
                    # Convert from milliseconds to seconds and create datetime
                    timestamp_seconds = timestamp_ms / 1000.0
                    return datetime.fromtimestamp(timestamp_seconds)
                    
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed to get block timestamp "
                    f"for block {block_number}: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
        
        logger.error(f"Failed to get block timestamp for block {block_number} after {max_retries} attempts")
        return None

    def start_commitment_fetcher(self):
        """Attach to the already-running event loop."""
        if self._fetch_task is None:
            self._fetch_task = asyncio.create_task(
                self._fetch_commitments_periodically()
            )

    async def _fetch_commitments_periodically(self):
        """Background task to periodically fetch commitments."""
        while True:
            try:
                # Create new subtensor instance for metagraph sync
                subtensor_sync = bt.subtensor(config=self.config)
                await asyncio.to_thread(
                    lambda: self.metagraph.sync(subtensor=subtensor_sync)
                )

                # Create new subtensor instance for commitments
                commitments = await self.get_commitments()
                if commitments:
                    self.commitments = commitments
                    # update_peers_with_buckets is now handled by PeerManager
                    logger.debug(f"Updated commitments: {self.commitments}")
            except Exception as e:
                logger.error(f"Error fetching commitments: {e}")
            await asyncio.sleep(self.fetch_interval)

    def get_bucket(self, uid: int) -> Optional[Bucket]:
        """Helper function to get the bucket for a given UID.

        Args:
            uid (int): The UID to retrieve the bucket for.

        Returns:
            Optional[Bucket]: The bucket corresponding to the UID, or None if not found.
        """
        return self.commitments.get(uid)

    def get_all_buckets(self) -> Dict[int, Optional[Bucket]]:
        """Helper function to get all buckets for all UIDs in the metagraph.

        Returns:
            Dict[int, Optional[Bucket]]: Mapping of UIDs to their bucket configurations
        """
        return {uid: self.get_bucket(uid) for uid in self.metagraph.uids}

    def block_to_window(self, block: int) -> int:
        """Returns the slice window based on a block."""
        return int(block / self.hparams.blocks_per_window)

    def window_to_seed(self, window: int) -> str:
        """Returns the slice window based on a block."""
        return str(self.subtensor.get_block_hash(window * self.hparams.blocks_per_window))

    def commit(self, wallet: "bt.wallet", bucket: Bucket) -> None:
        """Commits bucket configuration to the chain.

        Args:
            wallet (bt.wallet): Wallet to sign the commitment
            bucket (Bucket): Bucket configuration to commit
        """
        subtensor = bt.subtensor(config=self.config)
        concatenated = (
            bucket.account_id + bucket.access_key_id + bucket.secret_access_key
        )
        subtensor.commit(wallet, self.netuid, concatenated)
        logger.info(
            f"Committed bucket configuration to chain for hotkey {wallet.hotkey.ss58_address}"
        )

    def try_commit(self, wallet: Wallet, bucket: Bucket) -> None:
        """Attempts to verify existing commitment matches current bucket config and commits if not.

        Args:
            wallet (bt.wallet): Wallet to sign the commitment
            bucket (Bucket): Current bucket configuration to verify/commit
        """
        try:
            # Get existing commitment
            commitment = self.get_commitment(
                self.metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            )

            # Convert Bucket objects to concatenated strings for comparison
            commitment_str = (
                commitment.name
                + commitment.access_key_id
                + commitment.secret_access_key
            )
            bucket_details_from_env = (
                bucket.name + bucket.access_key_id + bucket.secret_access_key
            )

            logger.debug(
                "Comparing current commitment to bucket details from the environment:\n"
                f"Commitment: {commitment_str}\n"
                f"Current: {bucket_details_from_env}"
            )

            if bucket_details_from_env != commitment_str:
                if commitment_str == "":
                    log_msg_base = "Commitment is empty, likely because you're running your miner for the first time"
                else:
                    log_msg_base = "Bucket details have changed"
                logger.info(f"{log_msg_base}. Committing the new details.")
                self.commit(wallet, bucket)

        except Exception as e:
            logger.error(
                f"Error while verifying commitment: {str(e)}\n"
                "Committing the bucket details from the environment."
            )
            self.commit(wallet, bucket)

    def get_commitment(self, uid: int) -> Bucket:
        """Retrieves and parses committed bucket configuration data for a given
        UID.

        This method fetches commitment data for a specific UID from the
        subtensor network and decodes it into a structured format. The
        retrieved data is split into the following fields:
        - Account ID: A string of fixed length 32 characters.
        - Access key ID: A string of fixed length 32 characters.
        - Secret access key: A string of variable length (up to 64 characters).

        The parsed fields are then mapped to an instance of the `Bucket` class.
        When initializing the Bucket object, the account ID is also used as the
        bucket name.

        The retrieval process involves:
        - Fetching the commitment data for the specified UID using the
          configured `netuid` from the subtensor network.
        - Splitting the concatenated string into individual fields based on
          their expected lengths and order.
        - Mapping the parsed fields to a `Bucket` instance.

        **Note:** The order of fields (bucket name, account ID, access key ID,
        secret access key) in the concatenated string is critical for accurate
        parsing.

        Args:
            uid: The UID of the neuron whose commitment data is being
                retrieved.

        Returns:
            Bucket: An instance of the `Bucket` class containing the parsed
                bucket configuration details.

        Raises:
            ValueError: If the parsed data does not conform to the expected
                format for the `Bucket` class.
            Exception: If an error occurs while retrieving the commitment data
                from the subtensor network.
        """

        subtensor = bt.subtensor(config=self.config)
        try:
            concatenated = subtensor.get_commitment(self.netuid, uid)
            logger.success(f"Commitment fetched: {concatenated}")
        except Exception as e:
            raise Exception(f"Couldn't get commitment from uid {uid} because {e}")
        if len(concatenated) != 128:
            raise ValueError(
                f"Commitment '{concatenated}' is of length {len(concatenated)} but should be of length 128."
            )

        try:
            return Bucket(
                name=concatenated[:32],
                account_id=concatenated[:32],
                access_key_id=concatenated[32:64],
                secret_access_key=concatenated[64:],
            )
        except ValidationError as e:
            raise ValueError(f"Invalid data in commitment: {e}")

    def decode_metadata(self, encoded_ss58: tuple, metadata: dict) -> tuple[str, str]:
        # Decode the key into an SS58 address.
        decoded_key = decode_account_id(encoded_ss58[0])
        # Get the commitment from the metadata.
        commitment = metadata["info"]["fields"][0][0]
        bytes_tuple = commitment[next(iter(commitment.keys()))][0]
        return decoded_key, bytes(bytes_tuple).decode()

    async def get_commitments(self, block: Optional[int] = None) -> Dict[int, Bucket]:
        """Retrieves all bucket commitments from the chain.

        Args:
            block (int, optional): Block number to query at

        Returns:
            Dict[int, Bucket]: Mapping of UIDs to their bucket configurations
        """
        subtensor = bt.subtensor(config=self.config)
        substrate = subtensor.substrate
        # Query commitments via substrate.query_map
        query_result = substrate.query_map(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[self.netuid],
            block_hash=None if block is None else substrate.get_block_hash(block),
        )

        hotkey_to_uid = dict(zip(self.metagraph.hotkeys, self.metagraph.uids))
        commitments = {}

        for key, value in query_result:
            try:
                decoded_ss58, commitment_str = self.decode_metadata(key, value.value)
            except Exception as e:
                logger.error(f"Failed to decode metadata for key {key.value}: {e}")
                continue

            if decoded_ss58 not in hotkey_to_uid:
                continue

            uid = hotkey_to_uid[decoded_ss58]
            if len(commitment_str) != 128:
                logger.error(
                    f"Invalid commitment length for UID {uid}: {len(commitment_str)}"
                )
                continue

            try:
                bucket = Bucket(
                    name=commitment_str[:32],
                    account_id=commitment_str[:32],
                    access_key_id=commitment_str[32:64],
                    secret_access_key=commitment_str[64:],
                )
                commitments[uid] = bucket
                logger.debug(f"Retrieved bucket commitment for UID {uid}")
            except Exception as e:
                logger.error(f"Failed to build bucket for UID {uid}: {e}")
                continue

        return commitments

    def get_commitments_sync(self, block: Optional[int] = None) -> Dict[int, Bucket]:
        """
        Retrieves all bucket commitments from the chain.

        Args:
            block (int, optional): Block number to query at

        Returns:
            Dict[int, Bucket]: Mapping of UIDs to their bucket configurations
        """
        subtensor = bt.subtensor(config=self.config)
        # Use the new API. It returns a dict mapping hotkeys to the decoded commitment string.
        all_commitments = subtensor.get_all_commitments(self.netuid, block)
        hotkey_to_uid = dict(zip(self.metagraph.hotkeys, self.metagraph.uids))
        commitments = {}

        for hotkey, commitment in all_commitments.items():
            if hotkey not in hotkey_to_uid:
                continue
            uid = hotkey_to_uid[hotkey]
            if len(commitment) != 128:
                logger.error(
                    f"Invalid commitment length for UID {uid}: {len(commitment)}"
                )
                continue

            try:
                bucket = Bucket(
                    name=commitment[:32],
                    account_id=commitment[:32],
                    access_key_id=commitment[32:64],
                    secret_access_key=commitment[64:],
                )
                commitments[uid] = bucket
                logger.debug(f"Retrieved bucket commitment for UID {uid}")
            except Exception as e:
                logger.error(f"Failed to decode commitment for UID {uid}: {e}")
                continue

        return commitments

    async def get_bucket_for_neuron(self, wallet: "bt.wallet") -> Optional[Bucket]:
        """Get bucket configuration for a specific neuron's wallet

        Args:
            wallet (bt.wallet): The wallet to get bucket for

        Returns:
            Optional[Bucket]: The bucket assigned to this neuron, or None if not found
        """
        try:
            # Get UID by finding hotkey's index in metagraph
            uid = self.metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            return await self.get_bucket(uid)
        except ValueError:
            logger.warning(
                f"Hotkey {wallet.hotkey.ss58_address} not found in metagraph"
            )
            return None

    async def fetch_commitments(self):
        """Fetches commitments and updates self.commitments."""
        commitments = await self.get_commitments()
        if commitments:
            self.commitments = commitments
            # update_peers_with_buckets is now handled by PeerManager
            logger.debug(f"Fetched commitments: {self.commitments}")
        else:
            logger.warning("No commitments fetched.")

    def get_hotkey(self, uid: int) -> Optional[str]:
        """Returns the hotkey for a given UID."""
        # Handle different data types for uids
        if isinstance(self.metagraph.uids, (np.ndarray, torch.Tensor)):
            uids_list = self.metagraph.uids.tolist()
        else:
            uids_list = self.metagraph.uids

        # Handle different data types for hotkeys
        if isinstance(self.metagraph.hotkeys, (np.ndarray, torch.Tensor)):
            hotkeys_list = self.metagraph.hotkeys.tolist()
        else:
            hotkeys_list = self.metagraph.hotkeys

        if uid in uids_list:
            index = uids_list.index(uid)
            return hotkeys_list[index]
        else:
            return None

    # NOTE: update_peers_with_buckets has been moved to PeerManager
    # This method was removed to fix the 'active_peers' attribute error
