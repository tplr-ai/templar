"""Mock chain synchronization utilities"""

from unittest.mock import AsyncMock, MagicMock
from collections import defaultdict
from typing import Dict, Optional, Tuple
from .base import BaseMock


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
    ):
        super().__init__()
        self.config = config or MagicMock()
        self.netuid = netuid
        self.metagraph = metagraph or MagicMock()
        self.hparams = hparams or MagicMock()

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
        self.current_block = 1000
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
        self.mock_bucket = MagicMock()
        self.mock_bucket.name = "test-bucket"
        self.mock_bucket.account_id = "test-account"
        self.mock_bucket.access_key_id = "test-key"
        self.mock_bucket.secret_access_key = "test-secret"

        # Mock highest stake validator
        self.highest_stake_validator_uid = 1

    def start_commitment_fetcher(self):
        """Mock starting the background task to fetch commitments."""
        self._fetch_task = AsyncMock()

    async def get_commitments(self) -> Dict[int, object]:
        """Mock fetching commitments from the blockchain."""
        return {uid: self.mock_bucket for uid in [1, 2, 3]}

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
