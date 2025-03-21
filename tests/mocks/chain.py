"""Mock chain manager integrating metagraph and peer tracking"""

from .metagraph import MockMetagraph  # Inherit from the unified mock metagraph


class MockChainManager(MockMetagraph):
    """Mock chain manager that extends the mock metagraph with active/inactive peer tracking."""

    def __init__(self, n_validators: int = 10):
        super().__init__(n_validators)  # Initialize metagraph attributes
        self.active_peers = set()
        self.eval_peers = []
        self.inactive_peers = set()

    def update_peers_with_buckets(self):
        """Update inactive peers based on active peers.

        Inactive peers are computed as those in the evaluation list that are no longer active.
        """
        self.inactive_peers = set(self.eval_peers) - self.active_peers

        # TODO: Consider raising a warning if eval_peers is empty.
