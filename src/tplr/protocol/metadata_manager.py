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

import asyncio
import json
import re
from typing import Optional, Tuple, List

import aiofiles
import torch

import tplr
from ..storage.client import StorageClient
from ..storage.file_manager import FileManager
from ..chain import ChainManager
from ..schemas import Bucket

# Constants
PEERS_FILE_PREFIX = "peers_"


class MetadataManager:
    """Manages protocol metadata"""

    def __init__(
        self,
        storage_client: StorageClient,
        file_manager: FileManager,
        chain_manager: ChainManager,
        version: str,
        bucket: Bucket,
    ):
        """Initialize with dependencies"""
        self.storage_client = storage_client
        self.file_manager = file_manager
        self.chain_manager = chain_manager
        self.version = version
        self.bucket = bucket

    async def post_start_window(self, start_window: int) -> None:
        """Upload the start window as a JSON object to the node's R2 bucket."""
        try:
            key = f"start_window_v{self.version}.json"
            start_window_data = {"start_window": start_window}

            # Create temporary JSON file
            temp_file = self.file_manager.create_temp_file("start_window", ".json")

            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(start_window_data))

            # Read the file and upload
            with open(temp_file, "rb") as f:
                data = f.read()

            success = await self.storage_client.put_object(key, data, self.bucket)

            # Cleanup temp file
            self.file_manager.delete_file(temp_file)

            if success:
                tplr.logger.info(f"Successfully posted start_window: {start_window}")
            else:
                tplr.logger.error(f"Failed to post start_window: {start_window}")

        except Exception as e:
            tplr.logger.error(f"Error posting start_window: {e}")

    async def get_start_window(self, retries: int = -1) -> Optional[int]:
        """Get start window from validator bucket with retry logic"""
        attempt = 0
        while retries == -1 or attempt < retries:
            try:
                (
                    validator_bucket,
                    validator_uid,
                ) = await self._get_highest_stake_validator_bucket()
                if validator_bucket is None:
                    tplr.logger.warning(
                        "No highest staked validator bucket found. Retrying in 10 seconds"
                    )
                    attempt += 1
                    await asyncio.sleep(10)
                    continue

                tplr.logger.info(
                    f"Attempting to fetch start_window from UID {validator_uid} bucket {validator_bucket.name}"
                )

                key = f"start_window_v{self.version}.json"
                data = await self.storage_client.get_object(
                    key, validator_bucket, timeout=15
                )

                if data is not None:
                    # Parse JSON data
                    json_str = data.decode("utf-8")
                    start_window_json = json.loads(json_str)
                    start_window = start_window_json["start_window"]

                    tplr.logger.info(f"Fetched start_window: {start_window}")
                    return start_window

                tplr.logger.warning(
                    "start_window.json not found or empty. Retrying in 10 seconds"
                )
                attempt += 1
                await asyncio.sleep(10)

            except Exception as e:
                tplr.logger.error(f"Error fetching start_window: {e}")
                attempt += 1
                await asyncio.sleep(10)

        tplr.logger.warning("Max retries exceeded while trying to fetch start_window")
        return None

    async def post_peer_list(
        self,
        peers: list[int],
        first_effective_window: int,
        sync_window: int,
        weights: torch.Tensor,
        initial_selection: bool,
    ) -> None:
        """Upload peer list and debug data as JSON to the node's R2 bucket."""
        try:
            key = f"{PEERS_FILE_PREFIX}{first_effective_window}_v{self.version}.json"
            peers_and_weights = {
                "peers": peers,
                "initial_selection": initial_selection,
                "sync_window": sync_window,
                "first_effective_window": first_effective_window,
                "weights": weights.tolist() if torch.is_tensor(weights) else weights,
            }

            # Create temporary JSON file
            temp_file = self.file_manager.create_temp_file("peer_list", ".json")

            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(peers_and_weights))

            # Read and upload
            with open(temp_file, "rb") as f:
                data = f.read()

            success = await self.storage_client.put_object(key, data, self.bucket)

            # Cleanup temp file
            self.file_manager.delete_file(temp_file)

            if success:
                tplr.logger.info(
                    f"Successfully posted peer list for window {first_effective_window}"
                )
            else:
                tplr.logger.error(
                    f"Failed to post peer list for window {first_effective_window}"
                )

        except Exception as e:
            tplr.logger.error(f"Failed to upload peer list: {e}")

    async def get_peer_list(
        self, fetch_previous: bool = False
    ) -> tuple[list[int], int]] | None:
        """Get peer list from validator bucket"""
        tplr.logger.info(
            f"Looking for a {'previous' if fetch_previous else 'current'} peer list on a validator bucket"
        )

        while True:
            try:
                (
                    validator_bucket,
                    validator_uid,
                ) = await self._get_highest_stake_validator_bucket()

                if validator_bucket is None:
                    tplr.logger.warning(
                        "No highest staked validator bucket found. Retrying in 10 seconds."
                    )
                    await asyncio.sleep(10)
                    continue

                tplr.logger.info(
                    f"Attempting to fetch peer list from UID {validator_uid} bucket {validator_bucket.name}"
                )

                # List all peer files
                keys = await self.storage_client.list_objects(
                    PEERS_FILE_PREFIX, validator_bucket
                )
                pattern = rf"^{PEERS_FILE_PREFIX}(?P<window>\d+)_v{re.escape(self.version)}\.json$"

                valid_keys = []
                for key in keys:
                    if re.match(pattern, key):
                        valid_keys.append(key)

                if len(valid_keys) == 0:
                    tplr.logger.info("No peer list files found")
                    return None

                # Parse windows from all keys
                window_to_key = {}
                for key in valid_keys:
                    match = re.match(pattern, key)
                    if match:
                        window = int(match.group("window"))
                        window_to_key[window] = key

                if not window_to_key:
                    tplr.logger.error(
                        f"Failed to parse windows from peer list files. First "
                        f"{len(valid_keys[:5])} peer list files are {valid_keys[:5]}"
                    )
                    return None

                # Sort windows to find the most recent or the previous one
                sorted_windows = sorted(window_to_key.keys(), reverse=True)

                if len(sorted_windows) == 0:
                    return None

                # Select window based on fetch_previous flag
                selected_window = None
                if fetch_previous and len(sorted_windows) > 1:
                    selected_window = sorted_windows[1]  # Second most recent
                    tplr.logger.info(f"Selected previous window {selected_window}")
                elif fetch_previous and len(sorted_windows) <= 1:
                    tplr.logger.info("Found no previous window")
                    return None
                else:
                    selected_window = sorted_windows[0]  # Most recent
                    tplr.logger.info(f"Selected most recent window {selected_window}")

                selected_key = window_to_key[selected_window]

                # Download and parse peer data
                data = await self.storage_client.get_object(
                    selected_key, validator_bucket
                )
                if data is None:
                    tplr.logger.error(f"Failed to download peer list: {selected_key}")
                    await asyncio.sleep(10)
                    continue

                # Parse JSON
                json_str = data.decode("utf-8")
                peers_dict = json.loads(json_str)

                return peers_dict["peers"], peers_dict["first_effective_window"]

            except Exception as e:
                tplr.logger.error(f"Error fetching peer list: {e}")
                await asyncio.sleep(10)

    async def get_debug_dict(self, window: int) -> dict | None:
        """
        Get debug dictionary from validator bucket for a specific window.

        Args:
            window: Specific window to retrieve debug data for

        Returns:
            Debug dictionary or None if not found
        """
        try:
            (
                validator_bucket,
                validator_uid,
            ) = await self._get_highest_stake_validator_bucket()
            if not validator_bucket or validator_uid is None:
                tplr.logger.warning(
                    "No validator bucket - cannot proceed with debug fetch"
                )
                return None

            key = f"debug-{window}-{validator_uid}-v{self.version}.pt"
            tplr.logger.info(
                f"Attempting to retrieve debug dictionary for window {window} from validator {validator_uid}"
            )

            data = await self.storage_client.get_object(
                key, validator_bucket, timeout=20
            )

            if data is None:
                tplr.logger.warning(f"No debug dictionary found for window {window}")
                return None

            # Save to temp file and load with torch
            temp_file = self.file_manager.create_temp_file("debug_dict")
            with open(temp_file, "wb") as f:
                f.write(data)

            result = torch.load(temp_file, weights_only=False)
            self.file_manager.delete_file(temp_file)

            tplr.logger.info(
                f"Successfully retrieved debug dictionary for window {window}"
            )
            return result

        except Exception as e:
            tplr.logger.error(
                f"Error getting debug dictionary for window {window}: {e}"
            )
            return None

    async def _get_highest_stake_validator_bucket(
        self,
    ) -> tuple[Bucket, int] | tuple[None, None]:
        """Get the bucket for the validator with highest stake."""
        try:
            # Get validator with highest stake
            validator_uid = self.chain_manager.metagraph.S.argmax().item()
            tplr.logger.debug(f"Found validator with highest stake: {validator_uid}")

            if validator_uid is None:
                tplr.logger.info("No active validators found")
                return None, None

            # Get validator's bucket from commitments
            validator_bucket = self.chain_manager.commitments.get(int(validator_uid))
            if not validator_bucket:
                tplr.logger.warning(f"No bucket found for validator {validator_uid}")
                return None, None

            tplr.logger.debug(f"Validator Bucket: {validator_bucket}")
            return validator_bucket, validator_uid

        except Exception as e:
            tplr.logger.error(f"Error getting highest stake validator bucket: {e}")
            return None, None

    async def post_debug_dict(self, debug_data: dict, window: int, uid: str) -> bool:
        """Post debug dictionary to storage"""
        try:
            key = f"debug-{window}-{uid}-v{self.version}.pt"

            # Serialize to temp file
            temp_file = self.file_manager.create_temp_file("debug_post")
            torch.save(debug_data, temp_file)

            # Upload
            with open(temp_file, "rb") as f:
                data = f.read()

            success = await self.storage_client.put_object(key, data, self.bucket)

            # Cleanup
            self.file_manager.delete_file(temp_file)

            if success:
                tplr.logger.info(f"Successfully posted debug dict for window {window}")
            else:
                tplr.logger.error(f"Failed to post debug dict for window {window}")

            return success

        except Exception as e:
            tplr.logger.error(f"Error posting debug dict: {e}")
            return False

    # TODO: Add metadata versioning and migration
    # TODO: Add metadata integrity verification
    # TODO: Add metadata caching mechanisms
    # TODO: Add metadata compression for large datasets
    # TODO: Add metadata backup and recovery
