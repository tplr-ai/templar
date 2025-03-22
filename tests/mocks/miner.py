from types import SimpleNamespace
from tests.mocks.wallet import MockWallet
from tests.mocks.subtensor import MockSubtensor
from tests.mocks.metagraph import MockMetagraph
from tests.mocks.model import (
    MockModel,
    MockOptimizer,
    MockScheduler,
    MockTransformer,
    MockCompressor,
)
from tests.mocks.comms import MockComms


class MockMiner:
    """
    A minimal mock implementation of a Miner that simulates key behaviors without
    actual training, blockchain connectivity, or heavy dependencies.
    """

    def __init__(self):
        # Basic configuration to mimic the command-line config.
        self.config = SimpleNamespace(netuid=1, device="cpu", test=True, peers=[])
        self.config.stop_event = SimpleNamespace(
            is_set=lambda: False
        )  # Dummy stop_event

        # Hyperparameters for simulation.
        self.hparams = SimpleNamespace(
            checkpoint_frequency=100,
            blocks_per_window=100,
            pages_per_window=1,
            batch_size=1,
            sequence_length=10,
            time_window_delta_seconds=10,
            validator_offset=1,
        )

        # Simulate initial window and step values.
        self.current_window = 1000
        self.start_window = 1000
        self.global_step = 0
        self.uid = 0
        self.peers = [1, 2]  # Dummy evaluation peers

        # Use mock implementations for components.
        self.wallet = MockWallet()
        self.subtensor = MockSubtensor()
        self.metagraph = MockMetagraph()

        # Create a dummy model and associated artifacts.
        self.model = MockModel()
        self.optimizer = MockOptimizer()
        self.scheduler = MockScheduler()
        self.transformer = MockTransformer()
        self.compressor = MockCompressor()
        self.totalks = {"param": 1}
        self.comms = MockComms()

        # Extra attributes that might be used.
        self.tokenizer = None
        self.block_listener = lambda loop: None  # No-op block listener.
        self.wandb = None  # Dummy WandB logger.
        self.batch_times = []
        self.total_tokens_processed = 0

    async def run(self):
        """
        A minimal async run implementation.
        For testing, you can simulate a single iteration and immediately exit.
        """
        # Simulate some operations (e.g., logging, checkpointing) if needed.
        self.global_step += 1
        return "run completed"
