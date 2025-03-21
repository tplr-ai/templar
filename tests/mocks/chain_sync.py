"""Mock chain synchronization utilities"""

from unittest.mock import AsyncMock, MagicMock
from collections import defaultdict
from typing import Dict, Optional, Tuple
from .base import BaseMock
from .subtensor import MockSubtensor
from tplr.schemas import Bucket


class MockChainSync(BaseMock):
    """Mock chain synchronization that simulates blockchain operations."""

    def __init__(
        self,
        config=None,
        netuid=1,
        metagraph=None,
        hparams=None,
        fetch_interval=600,
        wallet=None,
        subtensor=None,
    ):
        super().__init__()
        self.config = config or MagicMock()
        self.netuid = netuid
        self.metagraph = metagraph or MagicMock()
        self.hparams = hparams or MagicMock()
        self.subtensor = subtensor or MockSubtensor()

        # Set default hparams if not provided
        if not hasattr(self.hparams, "blocks_per_window"):
            self.hparams.blocks_per_window = 100
        if not hasattr(self.hparams, "topk_peers"):
            self.hparams.topk_peers = 10
        if not hasattr(self.hparams, "minimum_peers"):
            self.hparams.minimum_peers = 3
        if not hasattr(self.hparams, "max_topk_peers"):
            self.hparams.max_topk_peers = 10

        # Block and window tracking
        self.current_block = self.subtensor.get_current_block()
        self.current_window = 10
        self.window_duration = self.hparams.blocks_per_window
        self.start_window = 0

        # Peer data
        self.commitments = {}
        self.peers = [1, 2, 3]  # Some default peers
        self.eval_peers = defaultdict(int, {1: 1, 2: 1, 3: 1})
        self.active_peers = set([1, 2, 3])  # Some default active peers
        self.inactive_peers = set()

        # Fetch control
        self.fetch_interval = fetch_interval
        self._fetch_task = None

        # Store wallet
        self.wallet = wallet

        # Mock bucket
        self.mock_bucket = Bucket(
            name="test-bucket",
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret",
        )

        # Mock highest stake validator
        self.highest_stake_validator_uid = 1

        # Mock methods as AsyncMock instead of regular functions
        self.start_commitment_fetcher = AsyncMock()
        self.update_peers_with_buckets = MagicMock()
        self.get_bucket = MagicMock(return_value=self.mock_bucket)
        self._get_highest_stake_validator_bucket = AsyncMock(
            return_value=(self.mock_bucket, 1)
        )
        self.get_miner_bucket = MagicMock(return_value=self.mock_bucket)
        self.get_validator_bucket = MagicMock(return_value=self.mock_bucket)

    def start_commitment_fetcher(self):
        """Mock starting the background task to fetch commitments."""
        self._fetch_task = AsyncMock()

    async def get_commitments(self) -> Dict[int, object]:
        """Mock fetching commitments from the blockchain."""
        # Use the subtensor to get commitments
        mock_commitments = {
            uid: self.mock_bucket
            for uid in self.metagraph.uids.tolist()
            if uid in [1, 2, 3]
        }
        return mock_commitments

    def get_bucket(self, uid: int) -> Optional[object]:
        """Mock getting bucket for a specific UID."""
        return self.mock_bucket if uid in [1, 2, 3] else None

    def update_peers_with_buckets(self):
        """Mock updating peers for gradient gathering and tracking inactive peers."""
        # Simple implementation that maintains active/inactive peer sets
        previously_active = set(self.eval_peers.keys())
        self.inactive_peers = previously_active - self.active_peers

    def set_gather_peers(self) -> None:
        """Mock determination of peers for gradient gathering."""
        # Simply use first few active peers
        active_list = list(self.active_peers)
        count = min(len(active_list), self.hparams.minimum_peers)
        self.peers = active_list[:count]

    async def _get_highest_stake_validator_bucket(
        self,
    ) -> Tuple[Optional[object], Optional[int]]:
        """Mock getting the bucket of the highest staked validator."""
        return self.mock_bucket, self.highest_stake_validator_uid

    async def get_start_window(self) -> int:
        """Mock getting the network start window."""
        return self.start_window

    async def check_and_perform_catch_up(
        self,
        sync_window,
        model,
        optimizer,
        scheduler,
        momentum,
        totalks,
        transformer,
        compressor,
    ) -> bool:
        """Mock catch-up mechanism."""
        return True

    async def _fetch_commitments_periodically(self):
        """Mock background task to periodically fetch commitments."""
        # This would use the subtensor in the real implementation
        self.commitments = await self.get_commitments()
        self.update_peers_with_buckets()
