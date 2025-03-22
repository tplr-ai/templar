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

import asyncio
from typing import Dict, Optional, Tuple
from collections import defaultdict

import bittensor as bt

from tplr import logger
from tplr.schemas import Bucket


class ChainSync:
    """Handles blockchain synchronization and peer tracking"""

    def __init__(
        self,
        config,
        netuid=None,
        metagraph=None,
        hparams=None,
        fetch_interval=600,
        wallet=None,
    ):
        self.config = config
        self.netuid = netuid
        self.metagraph = metagraph
        self.hparams = hparams or {}

        # Block and window tracking
        self.current_block = 0
        self.current_window = 0
        self.window_duration = (
            self.hparams.blocks_per_window
            if hasattr(self.hparams, "blocks_per_window")
            else 100
        )

        # Peer data
        self.commitments = {}
        self.peers = []
        self.eval_peers = defaultdict(int)
        self.active_peers = set()
        self.inactive_peers = set()

        # Fetch control
        self.fetch_interval = fetch_interval
        self._fetch_task = None

        # Store wallet
        self.wallet = wallet

    def start_commitment_fetcher(self):
        """Start background task to fetch commitments periodically"""
        if self._fetch_task is None:
            self._fetch_task = asyncio.create_task(
                self._fetch_commitments_periodically()
            )

    async def _fetch_commitments_periodically(self):
        """Background task to periodically fetch commitments"""
        while True:
            try:
                # Create new subtensor instance for metagraph sync
                subtensor_sync = bt.subtensor(config=self.config)
                await asyncio.to_thread(
                    lambda: self.metagraph.sync(subtensor=subtensor_sync)
                )

                # Fetch commitments and update
                commitments = await self.get_commitments()
                if commitments:
                    self.commitments = commitments
                    self.update_peers_with_buckets()
                    logger.debug(f"Updated commitments: {self.commitments}")
            except Exception as e:
                logger.error(f"Error fetching commitments: {e}")
            await asyncio.sleep(self.fetch_interval)

    async def get_commitments(self) -> Dict[int, Bucket]:
        """Fetch commitments from the blockchain"""
        try:
            subtensor = bt.subtensor(config=self.config)
            commitments = {}

            # Get all hotkeys with commitments
            for uid in self.metagraph.uids.tolist():
                try:
                    commitment = subtensor.get_commitment(netuid=self.netuid, uid=uid)

                    if not commitment or len(commitment) == 0:
                        continue

                    if len(commitment) != 128:
                        logger.error(
                            f"Invalid commitment length for UID {uid}: {len(commitment)}"
                        )
                        continue

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

            return commitments
        except Exception as e:
            logger.error(f"Error fetching commitments: {e}")
            return {}

    def get_bucket(self, uid: int) -> Optional[Bucket]:
        """Get bucket for a specific UID"""
        return self.commitments.get(uid)

    def update_peers_with_buckets(self):
        """Updates peers for gradient gathering, evaluation peers, and tracks inactive peers"""
        # Create mappings
        uid_to_stake = dict(
            zip(self.metagraph.uids.tolist(), self.metagraph.S.tolist())
        )

        # Get currently active peers
        active_peers = set(int(uid) for uid in self.active_peers)

        # Track inactive peers (previously active peers that are no longer active)
        previously_active = set(self.eval_peers.keys())
        newly_inactive = previously_active - active_peers
        self.inactive_peers = newly_inactive

        logger.debug(f"Active peers: {active_peers}")
        logger.info(f"Newly inactive peers: {newly_inactive}")

        if not active_peers:
            logger.warning("No active peers found. Skipping update.")
            return

        # Convert self.eval_peers into a dict while retaining old counts
        # for peers still active with stake <= threshold
        stake_threshold = getattr(self.hparams, "eval_stake_threshold", 20000)
        self.eval_peers = {
            int(uid): self.eval_peers.get(int(uid), 1)
            for uid in active_peers
            if uid in uid_to_stake and uid_to_stake[uid] <= stake_threshold
        }

        logger.debug(f"Filtered eval peers: {list(self.eval_peers.keys())}")

        self.set_gather_peers()

        logger.info(
            f"Updated gather peers (top {self.hparams.topk_peers}% or "
            f"minimum {self.hparams.minimum_peers}): {self.peers}"
        )
        logger.info(f"Total evaluation peers: {len(self.eval_peers)}")
        logger.info(f"Total inactive peers: {len(self.inactive_peers)}")

    def set_gather_peers(self) -> None:
        """Determines the list of peers for gradient gathering based on incentive scores"""
        # Get active peers
        active_peers = set(int(uid) for uid in getattr(self, "active_peers", []))
        if not active_peers:
            logger.warning("No active peers available for gathering")
            self.peers = []
            return

        # Get incentive scores
        uid_to_incentive = dict(
            zip(self.metagraph.uids.tolist(), self.metagraph.I.tolist())
        )

        # Use only active peers for incentive calculation
        miner_incentives = [(uid, uid_to_incentive.get(uid, 0)) for uid in active_peers]
        miner_incentives.sort(key=lambda x: x[1], reverse=True)

        # Determine the number of top-k peers based on percentage
        topk_peers_pct = getattr(self.hparams, "topk_peers", 10)
        max_topk_peers = getattr(self.hparams, "max_topk_peers", 10)
        minimum_peers = getattr(self.hparams, "minimum_peers", 3)

        n_topk_peers = int(len(miner_incentives) * (topk_peers_pct / 100))
        n_topk_peers = min(max(n_topk_peers, 1), max_topk_peers)

        n_peers = max(minimum_peers, n_topk_peers)

        # Take top n_peers by incentive from only the active peers
        self.peers = [uid for uid, _ in miner_incentives[:n_peers]]
        logger.info(f"Updated gather peers (active only): {self.peers}")

    async def _get_highest_stake_validator_bucket(
        self,
    ) -> Tuple[Optional[Bucket], Optional[int]]:
        """Get the bucket of the highest staked validator"""
        try:
            # Get validator UIDs and their stakes
            if self.metagraph is None:
                return None, None

            validators = []
            for i, uid in enumerate(self.metagraph.uids.tolist()):
                # Check if the hotkey is a validator (trust > 0)
                if self.metagraph.T[i] > 0:
                    validators.append((uid, self.metagraph.S[i].item()))

            if not validators:
                logger.warning("No validators found in metagraph")
                return None, None

            # Sort by stake and get highest
            validators.sort(key=lambda x: x[1], reverse=True)
            highest_validator_uid = validators[0][0]

            # Get bucket for this validator
            validator_bucket = self.get_bucket(highest_validator_uid)

            return validator_bucket, highest_validator_uid
        except Exception as e:
            logger.error(f"Error getting highest stake validator: {e}")
            return None, None
