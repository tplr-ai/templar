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
from typing import Optional, Tuple, List
from collections import defaultdict
import re

import bittensor as bt

from tplr import logger
from tplr.schemas import Bucket
from tplr import config  # For bucket secrets access


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
        self.hparams = hparams if hparams is not None else {}

        # Block and window tracking
        self.current_block = 0
        self.current_window = 0
        self.window_duration = (
            self.hparams.blocks_per_window
            if hasattr(self.hparams, "blocks_per_window")
            else 7  # current default value for blocks_per_window in config.yaml
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

                    # Check and commit our own bucket if necessary.
                    await self.check_and_commit_bucket()
            except Exception as e:
                logger.error(f"Error fetching commitments: {e}")
            await asyncio.sleep(self.fetch_interval)

    async def check_and_commit_bucket(self):
        """
        Check if our own gradient-read bucket (determined by get_own_bucket)
        matches the on-chain commitment. If not, commit it using try_commit.
        """
        if not self.wallet:
            logger.warning("Wallet not provided. Skipping bucket commitment check.")
            return

        own_uid = getattr(self.wallet, "uid", None)
        if own_uid is None:
            # Attempt to retrieve uid using wallet.hotkey.ss58_address and metagraph.hotkeys.
            if hasattr(self.wallet, "hotkey") and hasattr(self.metagraph, "hotkeys"):
                try:
                    own_uid = self.metagraph.hotkeys.index(
                        self.wallet.hotkey.ss58_address
                    )
                    logger.info(
                        f"Determined wallet uid as {own_uid} using hotkey.ss58_address."
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not determine wallet uid using hotkey: {e}. Skipping bucket commitment check."
                    )
                    return
            else:
                logger.warning(
                    "Wallet uid missing and no hotkey available. Skipping bucket commitment check."
                )
                return

        # Get the on-chain bucket (if any) for our uid.
        chain_bucket = self.get_bucket(own_uid)
        # Compute our intended bucket (for gradients read).
        own_bucket = self.get_own_bucket("gradients", "read")

        if not self._buckets_match(chain_bucket, own_bucket):
            logger.info(
                "Bucket commitment mismatch detected. Attempting to commit new bucket."
            )
            try:
                # Use try_commit instead of non-existent commit_bucket method
                result = await asyncio.to_thread(
                    self.subtensor.set_commitment,
                    wallet=self.wallet,
                    netuid=self.netuid,
                    commitment=own_bucket.to_commitment(),
                )
                if result and result.success:
                    logger.info(f"Successfully committed bucket: {own_bucket}")
                    # Update local commitment cache
                    if own_uid is not None:
                        self.commitments[own_uid] = own_bucket.to_commitment()
                else:
                    logger.warning(
                        f"Failed to commit bucket: {result.message if result else 'Unknown error'}"
                    )
            except Exception as e:
                logger.error(f"Failed to commit own bucket: {e}")

    def get_own_bucket(self, bucket_type: str, rw: str = "read") -> Bucket:
        """
        Create and return the bucket object for the given type and access mode.
        """
        bucket_conf = config.BUCKET_SECRETS.get(bucket_type)
        logger.debug("Bucket config for %s: %s", bucket_type, bucket_conf)
        if not bucket_conf:
            raise ValueError(f"No bucket configuration found for '{bucket_type}'.")

        name = bucket_conf.get("name", "").strip()
        if not name:
            raise ValueError(f"Bucket name for '{bucket_type}' must not be empty.")

        account_id = bucket_conf.get("account_id", "").strip()
        if not account_id:
            raise ValueError(
                f"Bucket account_id for '{bucket_type}' must not be empty."
            )

        creds = bucket_conf.get("credentials", {}).get(rw, {})
        access_key = creds.get("access_key_id", "").strip()
        secret_key = creds.get("secret_access_key", "").strip()
        if not access_key or not secret_key:
            raise ValueError(
                f"Bucket credentials for '{bucket_type}' in '{rw}' mode must not be empty."
            )

        bucket = Bucket(
            name=name,
            account_id=account_id,
            access_key_id=access_key,
            secret_access_key=secret_key,
        )
        logger.debug("Created Bucket: %s", bucket)
        return bucket

    def try_commit(self, wallet, bucket: Bucket) -> None:
        """
        Attempt to commit the provided bucket on-chain.
        Adjust the subtensor API call as required.
        """
        subtensor = bt.subtensor(config=self.config)
        try:
            subtensor.commit_bucket(wallet=wallet, bucket=bucket)
            logger.info("Bucket committed successfully.")
        except Exception as e:
            logger.error(f"Bucket commit failed: {e}")

    @staticmethod
    def _buckets_match(bucket1, bucket2) -> bool:
        """
        Compare two Bucket objects. Assumes Bucket has fields:
        name, account_id, access_key_id, and secret_access_key.
        """
        if bucket1 is None or bucket2 is None:
            return False
        return (
            bucket1.name == bucket2.name
            and bucket1.account_id == bucket2.account_id
            and bucket1.access_key_id == bucket2.access_key_id
            and bucket1.secret_access_key == bucket2.secret_access_key
        )

    async def get_commitments(self) -> dict[int, Bucket]:
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
                    logger.debug(f"Retrieved bucket commitment for UID {uid}: {bucket}")
                except Exception as e:
                    logger.error(f"Failed to decode commitment for UID {uid}: {e}")

            return commitments
        except Exception as e:
            logger.error(f"Error fetching commitments: {e}")
            return {}

    def get_bucket(self, uid: int) -> Optional[Bucket]:
        """Get bucket for a specific UID from commitments"""
        if uid is None:
            logger.warning("Cannot get bucket: UID is None")
            return None

        # Ensure uid is an integer
        uid = int(uid)

        # Log all available uids in commitments for debugging
        logger.debug(f"Available commitment UIDs: {list(self.commitments.keys())}")

        # Check if this uid exists in our commitments
        if uid in self.commitments:
            logger.debug(
                f"Retrieved bucket from commitment for UID {uid}: {self.commitments[uid]}"
            )
            return self.commitments[uid]

        # If it's a string or bytes, convert to Bucket
        if isinstance(self.commitments.get(uid), (str, bytes)):
            try:
                # Ensure we have a properly sized commitment
                if len(self.commitments.get(uid)) != 128:
                    logger.error(
                        f"Invalid commitment length for UID {uid}: {len(self.commitments.get(uid))}"
                    )
                    return None

                # Parse the commitment into a Bucket object
                bucket = Bucket(
                    name=self.commitments.get(uid)[:32].decode("utf-8").rstrip("\0")
                    if isinstance(self.commitments.get(uid), bytes)
                    else self.commitments.get(uid)[:32].rstrip("\0"),
                    account_id=self.commitments.get(uid)[32:64]
                    .decode("utf-8")
                    .rstrip("\0")
                    if isinstance(self.commitments.get(uid), bytes)
                    else self.commitments.get(uid)[32:64].rstrip("\0"),
                    access_key_id=self.commitments.get(uid)[64:96]
                    .decode("utf-8")
                    .rstrip("\0")
                    if isinstance(self.commitments.get(uid), bytes)
                    else self.commitments.get(uid)[64:96].rstrip("\0"),
                    secret_access_key=self.commitments.get(uid)[96:]
                    .decode("utf-8")
                    .rstrip("\0")
                    if isinstance(self.commitments.get(uid), bytes)
                    else self.commitments.get(uid)[96:].rstrip("\0"),
                )
                logger.debug(f"Created Bucket from commitment for UID {uid}: {bucket}")
                return bucket
            except Exception as e:
                logger.error(f"Failed to decode commitment for UID {uid}: {e}")
                return None

        logger.error(
            f"Unexpected commitment format for UID {uid}: {type(self.commitments.get(uid))}"
        )
        return None

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

    def set_gather_peers(self) -> list:
        """Determines the list of peers for gradient gathering based solely on incentive scores."""
        # Get incentive scores from the metagraph.
        uid_to_incentive = dict(
            zip(self.metagraph.uids.tolist(), self.metagraph.I.tolist())
        )

        # Use all peers from the metagraph.
        all_peers = self.metagraph.uids.tolist()
        miner_incentives = [(uid, uid_to_incentive.get(uid, 0)) for uid in all_peers]
        miner_incentives.sort(key=lambda x: x[1], reverse=True)

        # Determine the number of top-k peers based on percentage and hard limits.
        topk_peers_pct = getattr(self.hparams, "topk_peers", 10)
        max_topk_peers = getattr(self.hparams, "max_topk_peers", 10)
        minimum_peers = getattr(self.hparams, "minimum_peers", 3)

        n_topk_peers = int(len(miner_incentives) * (topk_peers_pct / 100))
        n_topk_peers = min(max(n_topk_peers, 1), max_topk_peers)

        n_peers = max(minimum_peers, n_topk_peers)

        # Select the top n_peers by incentive.
        self.peers = [uid for uid, _ in miner_incentives[:n_peers]]
        logger.info(f"Updated gather peers (by incentive): {self.peers}")
        return self.peers

    async def _get_highest_stake_validator_bucket(
        self, refresh_commitments: bool = False
    ) -> Tuple[Optional[Bucket], Optional[int]]:
        """Get the bucket of the highest staked validator"""
        try:
            # If no commitments are cached yet, fetch them
            if refresh_commitments or not self.commitments:
                logger.info("No commitments available, fetching now...")
                self.commitments = await self.get_commitments()

            if (
                self.metagraph is None
                or not hasattr(self.metagraph, "S")
                or self.metagraph.S is None
            ):
                logger.warning("Metagraph not available or has no stake information")
                return None, None

            validator_uid = self.metagraph.S.argmax().item()
            logger.info(f"Found validator with highest stake: {validator_uid}")

            if validator_uid is None:
                logger.info("No active validators found")
                return None, None

            validator_bucket = self.get_bucket(validator_uid)
            if not validator_bucket:
                logger.warning(f"No bucket found for validator {validator_uid}")
                return None, None

            logger.info(f"Using validator bucket: {validator_bucket}")
            return validator_bucket, validator_uid

        except Exception as e:
            logger.error(f"Error getting highest stake validator: {e}")
            return None, None

    async def _get_highest_staked_checkpoint(
        self, checkpoint_files: List[str]
    ) -> Optional[str]:
        """
        Given a list of checkpoint filenames, filter and return the latest checkpoint
        for the highest staked validator.

        Expected filename structure:
            checkpoint-{window}-{uid}-v{version}.pt

        Uses regex to robustly parse filenames.
        Only returns a checkpoint whose uid matches the highest staked validator.
        """
        # Force refresh to ensure up-to-date highest validator info
        _, highest_uid = await self._get_highest_stake_validator_bucket(
            refresh_commitments=True
        )
        if highest_uid is None:
            logger.error("Cannot determine highest staked validator UID.")
            return None

        valid_checkpoints = []
        # Regex: capture window, UID, and version from the filename.
        pattern = re.compile(r"^checkpoint-(\d+)-(\d+)-v(.+)\.pt$")

        for filename in checkpoint_files:
            m = pattern.match(filename)
            if not m:
                logger.info(f"Skipping invalid checkpoint format: {filename}")
                continue

            try:
                window = int(m.group(1))
                uid = int(m.group(2))
                # version = m.group(3)  # could be used for version validation if needed
            except Exception as e:
                logger.warning(f"Error parsing checkpoint filename {filename}: {e}")
                continue

            if uid != highest_uid:
                logger.info(
                    f"Skipping checkpoint not from highest staked validator (expected uid {highest_uid}): {filename}"
                )
                continue

            valid_checkpoints.append((window, filename))

        if not valid_checkpoints:
            logger.warning(
                "No valid checkpoints found for the highest staked validator."
            )
            return None

        # Select the checkpoint with the highest window number (most recent)
        latest_checkpoint = max(valid_checkpoints, key=lambda x: x[0])[1]
        logger.info(
            f"Selected checkpoint for highest staked validator (uid {highest_uid}): {latest_checkpoint}"
        )
        return latest_checkpoint
