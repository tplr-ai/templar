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
from collections import defaultdict
import re
import json
import numpy as np
import torch

from tplr import logger
from tplr import __version__
from tplr.storage import StorageManager

import tempfile


class PeerManager:
    """Manages peer discovery and tracking"""

    def __init__(self, chain, hparams, metagraph):
        self.chain = chain
        self.hparams = hparams
        self.metagraph = metagraph
        self.active_peers = set()
        self.inactive_peers = set()
        self.eval_peers = defaultdict(int)

    async def track_active_peers(self):
        """Background task to keep track of active peers"""
        while True:
            active_peers = set()
            tasks = []
            semaphore = asyncio.Semaphore(10)  # Limit concurrent S3 requests

            logger.debug(f"Commitments: {self.chain.commitments}")

            async def check_peer(uid):
                async with semaphore:
                    is_active = await self.is_miner_active(
                        uid, recent_windows=self.hparams.recent_windows
                    )
                    if is_active:
                        active_peers.add(uid)

            for uid in self.chain.commitments.keys():
                tasks.append(check_peer(uid))

            await asyncio.gather(*tasks)
            self.active_peers = active_peers

            logger.info(
                f"Updated active peers: {[int(uid) for uid in self.active_peers]}"
            )

            # Update chain's active peers list
            self.chain.active_peers = self.active_peers

            # Update peer lists based on active peers
            self.chain.update_peers_with_buckets()

            await asyncio.sleep(self.hparams.active_check_interval)

    async def is_miner_active(self, uid: int, recent_windows: int = 3) -> bool:
        """Check if a miner is active by looking for recent gradient files"""
        # Get the bucket for this UID
        bucket = self.chain.get_bucket(uid)
        if not bucket:
            return False

        # Check the most recent windows
        current_window = self.chain.current_window
        for window in range(current_window, current_window - recent_windows, -1):
            if window <= 0:
                continue

            try:
                temp_dir = tempfile.TemporaryDirectory().name
                storage = StorageManager(temp_dir=temp_dir, save_location=temp_dir)

                gradient_key = f"gradient-{window}-{uid}-v{__version__}.pt"
                exists = await storage.s3_head_object(key=gradient_key, bucket=bucket)

                if exists:
                    return True
            except Exception as e:
                logger.error(f"Error checking activity for UID {uid}: {e}")

        return False

    async def get_peer_list(self) -> tuple:
        """
        Retrieve the most recent peer list from the highest-staked validator bucket.
        Returns a tuple of (peer_array, first_effective_window) or None if not found.
        """
        PEERS_FILE_PREFIX = "peers_"
        logger.info("Starting to look for a peer list on a validator bucket")
        while True:
            try:
                # Use the chain to get the highest stake validator bucket.
                validator_bucket, validator_uid = await self.chain._get_highest_stake_validator_bucket()
                if validator_bucket is None:
                    logger.warning("No highest staked validator bucket found. Retrying in 10 seconds.")
                    await asyncio.sleep(10)
                    continue

                logger.info(f"Attempting to fetch peer list from UID {validator_uid} bucket {validator_bucket.name}")

                s3_client = await self.chain._get_s3_client(validator_bucket)
                list_args = {"Bucket": validator_bucket.name, "Prefix": "peers_"}
                response = await s3_client.list_objects_v2(**list_args)

                pattern = rf"^{PEERS_FILE_PREFIX}(?P<window>\d+)_v{__version__}\.json$"
                keys = [obj["Key"] for obj in response.get("Contents", []) if re.match(pattern, obj["Key"])]
                if not keys:
                    logger.info("No peer list files found")
                    return None

                max_window = -1
                selected_key = None
                for key in keys:
                    match = re.match(pattern, key)
                    if match:
                        window = int(match.group("window"))
                        if window > max_window:
                            max_window = window
                            selected_key = key

                if selected_key is None:
                    logger.error(f"Failed to select most recent peers file on bucket. First few: {keys[:5]}")
                    return None

                peers_data = await self.chain.s3_get_object(key=selected_key, bucket=validator_bucket)
                if isinstance(peers_data, dict):
                    peers_dict = peers_data
                else:
                    peers_dict = json.loads(peers_data.decode("utf-8"))
                return np.array(peers_dict["peers"]), peers_dict["first_effective_window"]
            except Exception as e:
                logger.error(f"Error fetching peer list: {e}")
                await asyncio.sleep(10)

    def select_initial_peers(self) -> np.ndarray | None:
        """
        Select initial peers based on incentive and activity.

        Steps:
          1. Pick peers with positive incentive (from metagraph) that are active.
          2. If there aren't enough, fill up with additional active peers.
          3. Return None if we can't satisfy minimum peers.
        """
        try:
            logger.info("Starting selection of initial gather peers")
            uid_to_non_zero_incentive = {
                uid: incentive
                for uid, incentive in zip(self.metagraph.uids, self.metagraph.I)
                if incentive > 0 and uid in self.active_peers
            }
            top_incentive_peers = sorted(
                uid_to_non_zero_incentive,
                key=uid_to_non_zero_incentive.get,
                reverse=True,
            )[: self.hparams.max_topk_peers]

            top_incentive_peers = np.array(top_incentive_peers, dtype=np.int64)

            if top_incentive_peers.size >= self.hparams.minimum_peers:
                logger.info(
                    f"Selected {top_incentive_peers.size} initial peers purely based on incentive: {top_incentive_peers.tolist()}"
                )
                return top_incentive_peers

            remaining_active_peers = np.array(list(self.active_peers - set(top_incentive_peers)))
            top_incentive_and_active_peers = np.concatenate(
                [top_incentive_peers, remaining_active_peers]
            )[: self.hparams.max_topk_peers]

            if top_incentive_and_active_peers.size >= self.hparams.minimum_peers:
                logger.info(
                    f"Selected {top_incentive_and_active_peers.size} initial peers: {top_incentive_and_active_peers.tolist()}"
                )
                return top_incentive_and_active_peers

            logger.info(
                f"Failed to find at least {self.hparams.minimum_peers} initial peers. Found only {top_incentive_and_active_peers.size}"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to create new peer list: {e}")
            return None

    def select_next_peers(
        self, current_peers: np.ndarray, weights: torch.Tensor
    ) -> np.ndarray | None:
        """
        Update the peer list based on current evaluation weights.

        Steps:
          1. Drop any peers that are either inactive or have zero weight.
          2. If fewer than max_topk_peers remain, add new candidate peers.
          3. If exactly max_topk_peers and enough candidates exist, replace a subset.
          4. Return the final peer array.
        """
        try:
            old_peers = current_peers
            non_zero_weight_uids = torch.nonzero(weights).flatten().cpu().numpy()

            still_active = np.intersect1d(old_peers, np.array(list(self.active_peers)))
            still_non_zero_weight = np.intersect1d(old_peers, non_zero_weight_uids)
            active_gather_peers = np.intersect1d(still_active, still_non_zero_weight)

            dropped_peers = np.setdiff1d(old_peers, active_gather_peers)
            if dropped_peers.size > 0:
                logger.info(
                    f"Dropping peers (inactive or zero-weight): {dropped_peers.tolist()}. Remaining: {active_gather_peers.tolist()}"
                )

            candidates = np.setdiff1d(np.array(list(self.active_peers)), active_gather_peers)
            current_len = len(active_gather_peers)
            selected_peers = active_gather_peers.copy()

            if current_len < self.hparams.max_topk_peers:
                needed = self.hparams.max_topk_peers - current_len
                if candidates.size > 0:
                    to_add = np.random.choice(
                        candidates, size=min(needed, len(candidates)), replace=False
                    )
                    selected_peers = np.concatenate([selected_peers, to_add])
                    logger.info(
                        f"Added {to_add.tolist()} to increase peers from {current_len} to {len(selected_peers)}."
                    )
                else:
                    logger.info("Fewer than max_topk_peers remain, but no new candidates available.")
            elif current_len == self.hparams.max_topk_peers:
                if len(candidates) >= self.hparams.peers_to_replace:
                    outgoing = np.random.choice(
                        selected_peers, size=self.hparams.peers_to_replace, replace=False
                    )
                    ingoing = np.random.choice(
                        candidates, size=self.hparams.peers_to_replace, replace=False
                    )
                    selected_peers = np.setdiff1d(selected_peers, outgoing)
                    selected_peers = np.concatenate([selected_peers, ingoing])
                    logger.info(
                        f"Replaced {outgoing.tolist()} with {ingoing.tolist()} to maintain total of {self.hparams.max_topk_peers} peers."
                    )
                else:
                    logger.info("Max peers reached but insufficient candidates for replacement.")

            selected_peers = selected_peers.astype(np.int64)
            logger.info(f"Final peers after selection: {selected_peers.tolist()}")
            return selected_peers
        except Exception as e:
            logger.error(f"Error during peer selection: {e}")
            return None
