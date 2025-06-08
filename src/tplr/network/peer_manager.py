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
import random
from typing import Set, List, Optional

import tplr
from ..chain import ChainManager
from ..storage.client import StorageClient


class PeerManager:
    """Manages peer discovery and activity tracking"""

    def __init__(
        self, chain_manager: ChainManager, storage_client: StorageClient, hparams
    ):
        """Initialize with chain manager and configuration"""
        self.chain_manager = chain_manager
        self.storage_client = storage_client
        self.hparams = hparams

        self.active_peers: Set[int] = set()
        self.active_check_interval = hparams.active_check_interval
        self.recent_windows = hparams.recent_windows

        # Gather target management as per SPEC.md
        self.gather_target_peers: List[int] = []
        self.next_gather_target_peers: Optional[List[int]] = None
        self.gather_targets_update_window: int = -1

        # Background task management
        self._peer_tracking_task = None
        self._stop_tracking = False

    async def refresh_gather_targets(self, current_window: int, metadata_manager) -> None:
        """
        Refresh gather target peers using the logic from neuron_utils.update_peers.
        
        Args:
            current_window: Current blockchain window
            metadata_manager: MetadataManager instance for fetching peer lists
        """
        # Check if peers list is empty and fetch previous list if needed
        if len(self.gather_target_peers) == 0:
            tplr.logger.info(
                "Current gather target peers list is empty, attempting to fetch previous peer list"
            )
            result = await metadata_manager.get_peer_list(fetch_previous=True)
            if result is not None:
                prev_peers, prev_update_window = result
                tplr.logger.info(
                    f"Got previous peer list with {len(prev_peers)} peers "
                    f"and update window {prev_update_window}"
                )
                self.gather_target_peers = prev_peers
                # Don't set next_gather_target_peers here, as we want the normal update process to continue
            else:
                tplr.logger.warning(
                    "Failed to fetch previous peer list, continuing with empty peers"
                )

        # Get next peers
        if (
            self.next_gather_target_peers is None  # next peers are not fetched yet
            and self.gather_targets_update_window  # they should be on bucket by now
            + self.hparams.peer_replacement_frequency
            - current_window
            < self.hparams.peer_list_window_margin
        ):
            result = await metadata_manager.get_peer_list()
            if result is None:
                tplr.logger.info("Unable to get peer list from bucket")
            else:
                next_peers, peers_update_window = result
                tplr.logger.info(
                    f"Got peer list {next_peers} and update window "
                    f"{peers_update_window} from bucket"
                )
                if (
                    self.gather_targets_update_window is None
                    or peers_update_window > self.gather_targets_update_window
                ):
                    self.next_gather_target_peers = next_peers
                    self.gather_targets_update_window = peers_update_window
                    tplr.logger.info("This list is new, updating next_gather_target_peers")

        # Update peers, if it's time
        if self.next_gather_target_peers is not None and current_window >= self.gather_targets_update_window:
            self.gather_target_peers = self.next_gather_target_peers
            late_text = (
                f"{current_window - self.gather_targets_update_window} windows late"
                if current_window - self.gather_targets_update_window > 0
                else "on time"
            )
            tplr.logger.info(
                f"Updated gather target peers {late_text} - gather:{len(self.gather_target_peers)}. "
                f"Next update expected on window "
                f"{self.gather_targets_update_window + self.hparams.peer_list_window_margin}"
            )
            self.next_gather_target_peers = None
        else:
            reason = (
                "next gather target peers are not defined yet"
                if self.next_gather_target_peers is None
                else f"current window is {current_window} and peers update window "
                f"is {self.gather_targets_update_window}"
            )
            tplr.logger.info(f"Not time to replace gather target peers: {reason}")

    async def track_active_peers(self) -> None:
        """Background task to track active peers"""
        while not self._stop_tracking:
            try:
                await self._update_active_peers()
                await asyncio.sleep(self.active_check_interval)
            except Exception as e:
                tplr.logger.error(f"Error in peer tracking: {e}")
                await asyncio.sleep(self.active_check_interval)

    async def _update_active_peers(self) -> None:
        """Update the set of active peers"""
        try:
            # Get current window from the chain manager
            try:
                current_window = self.chain_manager.current_window
                if current_window is None:
                    # If no current_window attribute, we can't determine activity
                    tplr.logger.debug(
                        "No current_window available, skipping peer activity check"
                    )
                    return
            except Exception as e:
                tplr.logger.debug(f"Could not get current window: {e}")
                return

            # Check recent windows for activity
            check_windows = [current_window - i for i in range(self.recent_windows)]
            check_windows = [
                w for w in check_windows if w >= 0
            ]  # Filter out negative windows

            active_peers = set()

            # Check each UID for activity in recent windows
            if (
                hasattr(self.chain_manager, "commitments")
                and self.chain_manager.commitments
            ):
                for uid in self.chain_manager.commitments.keys():
                    if await self._is_peer_active(uid, check_windows):
                        active_peers.add(uid)
            else:
                # Fallback to metagraph UIDs if commitments not available
                if (
                    self.chain_manager.metagraph
                    and self.chain_manager.metagraph.uids is not None
                ):
                    for uid in range(len(self.chain_manager.metagraph.uids)):
                        if await self._is_peer_active(uid, check_windows):
                            active_peers.add(uid)

            # Update active peers and filter by bucket availability
            self.active_peers = active_peers
            self.update_peers_with_buckets()

            if (
                self.chain_manager.metagraph
                and self.chain_manager.metagraph.uids is not None
            ):
                total_peers = len(self.chain_manager.metagraph.uids)
            else:
                total_peers = 0

            tplr.logger.debug(
                f"Updated active peers: {len(self.active_peers)} active out of {total_peers} total"
            )

        except Exception as e:
            tplr.logger.error(f"Error updating active peers: {e}")

    async def is_peer_active(self, uid: int, recent_windows: int = 3) -> bool:
        """Check if a peer is active by looking for recent gradient uploads"""
        try:
            # Get current window from the chain manager or comms object
            current_window = getattr(self.chain_manager, "current_window", None)
            if current_window is None:
                return False

            # Check recent windows for gradient uploads
            for window_offset in range(recent_windows):
                check_window = current_window - window_offset
                if check_window < 0:
                    continue

                # Try to get peer's bucket and check for gradients
                try:
                    peer_bucket = self.chain_manager.get_bucket(uid)
                    if peer_bucket is None:
                        continue

                    gradient_key = (
                        f"gradient-{check_window}-{uid}-v{tplr.__version__}.pt"
                    )

                    # Check if gradient exists (just HEAD request)
                    size = await self.storage_client.get_object_size(
                        gradient_key, peer_bucket
                    )
                    if size is not None:
                        return True

                except Exception as e:
                    tplr.logger.debug(
                        f"Error checking activity for uid {uid}, window {check_window}: {e}"
                    )
                    continue

            return False

        except Exception as e:
            tplr.logger.error(f"Error checking if peer {uid} is active: {e}")
            return False

    def update_peers_with_buckets(self) -> None:
        """Update peers that have valid bucket configurations"""
        try:
            peers_with_buckets = set()

            # Use commitments if available, otherwise fall back to metagraph
            if (
                hasattr(self.chain_manager, "commitments")
                and self.chain_manager.commitments
            ):
                for uid in self.active_peers:
                    if uid in self.chain_manager.commitments:
                        peers_with_buckets.add(uid)
            else:
                # Fallback method using get_bucket_for_uid if available
                for uid in self.active_peers:
                    try:
                        if hasattr(self.chain_manager, "get_bucket"):
                            bucket = self.chain_manager.get_bucket(uid)
                            if bucket is not None:
                                peers_with_buckets.add(uid)
                    except Exception:
                        continue

            # Update active peers to only include those with buckets
            self.active_peers = self.active_peers.intersection(peers_with_buckets)

            tplr.logger.debug(f"Peers with buckets: {len(peers_with_buckets)}")

        except Exception as e:
            tplr.logger.error(f"Error updating peers with buckets: {e}")

    def get_active_peers(self) -> Set[int]:
        """Get the current set of active peers"""
        return self.active_peers.copy()

    def get_inactive_peers(self) -> Set[int]:
        """Get the set of inactive peers"""
        try:
            if (
                self.chain_manager.metagraph
                and self.chain_manager.metagraph.uids is not None
            ):
                all_peers = set(range(len(self.chain_manager.metagraph.uids)))
                return all_peers - self.active_peers
            return set()
        except Exception:
            return set()

    def weighted_random_sample_no_replacement(
        self, candidates: list[int], weights: list[int], k: int
    ) -> list[int]:
        """
        Perform a weighted random sample (without replacement) of size k.
        candidates: list of items (uids).
        weights:    list of corresponding weights (integers or floats).
        k:          number of items to sample.
        Returns a list of selected items.
        """
        tplr.logger.debug("Starting weighted random sampling")
        tplr.logger.debug(f"Candidates: {candidates}")
        tplr.logger.debug(f"Weights: {weights}")
        tplr.logger.debug(f"Sample size (k): {k}")

        # Safety checks
        if not candidates or not weights or k <= 0:
            tplr.logger.warning("Invalid input detected. Returning empty list.")
            return []

        # Pair up each candidate with its weight
        pool = list(zip(candidates, weights))
        total_w = float(sum(weights))
        selected = []

        # If total weight is 0, return empty
        if total_w <= 0:
            tplr.logger.warning("Total weight is zero. Returning empty list")
            return []

        tplr.logger.debug(f"Initial total weight: {total_w}")

        for _ in range(min(k, len(candidates))):
            if total_w <= 0 or len(pool) == 0:
                tplr.logger.info("No more items to sample. Stopping early.")
                break

            # Draw a uniform sample in [0, total_w]
            r = random.uniform(0.0, total_w)
            tplr.logger.debug(f"Random threshold: {r}")
            cumulative = 0.0
            for idx, (uid, w) in enumerate(pool):
                cumulative += w
                if cumulative >= r:
                    # Found our pick
                    selected.append(uid)
                    tplr.logger.info(f"Selected item: {uid} with weight {w}")
                    # Remove from pool and subtract from total_w
                    total_w -= w
                    pool.pop(idx)
                    tplr.logger.debug(f"Updated total weight: {total_w}")
                    break

        tplr.logger.debug(f"Final selected items: {selected}")
        return selected

    async def start_peer_tracking(self) -> None:
        """Start the background peer tracking task"""
        if self._peer_tracking_task is None or self._peer_tracking_task.done():
            self._stop_tracking = False
            self._peer_tracking_task = asyncio.create_task(self.track_active_peers())
            tplr.logger.info("Started peer tracking task")

    async def stop_peer_tracking(self) -> None:
        """Stop the background peer tracking task"""
        self._stop_tracking = True
        if self._peer_tracking_task and not self._peer_tracking_task.done():
            self._peer_tracking_task.cancel()
            try:
                await self._peer_tracking_task
            except asyncio.CancelledError:
                pass
        tplr.logger.info("Stopped peer tracking task")

    def get_peer_count(self) -> int:
        """Get total number of peers in metagraph"""
        try:
            if (
                self.chain_manager.metagraph
                and self.chain_manager.metagraph.uids is not None
            ):
                return len(self.chain_manager.metagraph.uids)
            return 0
        except Exception:
            return 0

    def get_active_peer_count(self) -> int:
        """Get number of currently active peers"""
        return len(self.active_peers)

    def is_peer_in_active_set(self, uid: int) -> bool:
        """Check if a specific peer is in the active set"""
        return uid in self.active_peers

    async def _is_peer_active(self, uid: int, check_windows: list[int]) -> bool:
        """Check if peer is active in any of the given windows"""
        try:
            if (
                not hasattr(self.chain_manager, "commitments")
                or not self.chain_manager.commitments
            ):
                return False

            peer_bucket = self.chain_manager.commitments.get(uid)
            if peer_bucket is None:
                return False

            for window in check_windows:
                gradient_key = f"gradient-{window}-{uid}-v{tplr.__version__}.pt"
                try:
                    size = await self.storage_client.get_object_size(
                        gradient_key, peer_bucket
                    )
                    if size is not None:
                        return True
                except Exception as e:
                    tplr.logger.debug(
                        f"Error checking gradient for uid {uid}, window {window}: {e}"
                    )
                    continue
            return False
        except Exception as e:
            tplr.logger.error(f"Error checking if peer {uid} is active: {e}")
            return False

    async def _filter_peers_with_buckets(self) -> None:
        """Filter active peers to only include those with valid buckets"""
        try:
            peers_with_buckets = set()

            if (
                hasattr(self.chain_manager, "commitments")
                and self.chain_manager.commitments
            ):
                for uid in self.active_peers:
                    if uid in self.chain_manager.commitments:
                        peers_with_buckets.add(uid)

            # Update active peers to only include those with buckets
            self.active_peers = self.active_peers.intersection(peers_with_buckets)

            tplr.logger.debug(
                f"Filtered to peers with buckets: {len(self.active_peers)}"
            )

        except Exception as e:
            tplr.logger.error(f"Error filtering peers with buckets: {e}")
