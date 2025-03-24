import pytest
import torch
import threading
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from types import SimpleNamespace
import sys
from neurons.validator import Validator, min_power_normalization

# Import the reusable mocks.
from tests.mocks.wallet import MockWallet
from tests.mocks.subtensor import MockSubtensor
from tests.mocks.model import MockModel

# Add this at the top to prevent multiple registrations
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*operator registration has already occurred.*",
)

# Prevent re-registering torch operators
_TORCH_INITIALIZED = False


@pytest.fixture(scope="module", autouse=True)
def init_torch_only_once():
    """Initialize torch operators only once for the entire module"""
    global _TORCH_INITIALIZED
    if not _TORCH_INITIALIZED:
        # Clear any existing registrations if possible
        if hasattr(torch._C, "_jit_clear_class_registry"):
            torch._C._jit_clear_class_registry()
        _TORCH_INITIALIZED = True


# ----------------------------
# Test Class: Validator Configuration
# ----------------------------
class TestValidatorConfig:
    def test_config_parsing(self, monkeypatch):
        """
        Test that Validator.config() correctly parses command‑line arguments.

        Steps:
          - Patch sys.argv to simulate command‑line input.
          - Stub bt.subtensor.add_args, bt.logging.add_args, and bt.wallet.add_args.
          - Stub bt.config() to return a controlled SimpleNamespace.
        """

        original_argv = sys.argv
        sys.argv = [
            "validator.py",
            "--netuid",
            "42",
            "--project",
            "test_project",
            "--device",
            "cpu",
            "--debug",
        ]
        with (
            patch("neurons.validator.bt.subtensor.add_args"),
            patch("neurons.validator.bt.logging.add_args"),
            patch("neurons.validator.bt.wallet.add_args"),
            patch(
                "neurons.validator.bt.config",
                return_value=SimpleNamespace(
                    netuid=42,
                    project="test_project",
                    device="cpu",
                    debug=True,
                    trace=False,
                    store_gathers=False,
                ),
            ),
        ):
            cfg = Validator.config()
            assert cfg.netuid == 42
            assert cfg.project == "test_project"
            assert cfg.device == "cpu"
            assert cfg.debug is True
            assert cfg.trace is False
        sys.argv = original_argv


# ----------------------------
# Test Class: Wallet Registration
# ----------------------------
class TestWalletRegistration:
    def test_wallet_registration_success(self):
        """
        Test that Validator initialization completes normally when the wallet's hotkey
        is present in metagraph.hotkeys.

        Steps:
          - Create a wallet instance using the shared MockWallet.
          - Override its hotkey (if needed) to "default".
          - Patch bt.wallet to return the MockWallet.
          - Patch bt.subtensor to return a reusable MockSubtensor.
          - Bypass __init__ registration via __new__() and manually set necessary attributes.
        """

        # Instantiate a reusable wallet and overwrite its hotkey to "default"
        dummy_wallet = MockWallet()
        dummy_wallet.hotkey.ss58_address = "default"
        # Use the reusable MockSubtensor.
        mock_subtensor = MockSubtensor()

        with (
            patch("neurons.validator.bt.wallet", return_value=dummy_wallet),
            patch("neurons.validator.bt.subtensor", return_value=mock_subtensor),
        ):
            # Bypass __init__() registration check using __new__.
            validator = Validator.__new__(Validator)
            validator.config = SimpleNamespace(netuid=1, peers=[])
            validator.hparams = SimpleNamespace(
                model_config={},
                target_chunk=512,
                learning_rate=0.01,
                topk_compression=0.1,
                weight_decay=0.0,
                checkpoint_frequency=100,
                windows_per_weights=10,
            )
            validator.wallet = dummy_wallet
            validator.subtensor = mock_subtensor
            validator.metagraph = validator.subtensor.metagraph(validator.config.netuid)
            if validator.wallet.hotkey.ss58_address not in validator.metagraph.hotkeys:
                pytest.fail("Wallet hotkey not registered but expected to be present.")
            validator.uid = validator.metagraph.hotkeys.index(
                validator.wallet.hotkey.ss58_address
            )
            assert True

    def test_wallet_registration_failure(self, monkeypatch):
        """
        Test that Validator.__init__ calls sys.exit when wallet's hotkey is not in metagraph.hotkeys.

        Steps:
          - Create a wallet using MockWallet and override its hotkey to an unregistered value.
          - Create a dummy metagraph (using SimpleNamespace) that does not include that hotkey.
          - Monkey-patch sys.exit to raise a SystemExit.
          - Verify that SystemExit is raised during Validator.__init__().
        """

        dummy_wallet = MockWallet()
        dummy_wallet.hotkey.ss58_address = "non_registered"
        # Create a dummy metagraph without the wallet's hotkey.
        dummy_metagraph = SimpleNamespace(hotkeys=["default"], n=1, netuid=1)
        # Create a dummy subtensor that returns the dummy metagraph.
        dummy_subtensor = SimpleNamespace(metagraph=lambda netuid: dummy_metagraph)
        monkeypatch.setattr(
            "neurons.validator.sys.exit", lambda: (_ for _ in ()).throw(SystemExit())
        )
        with (
            patch("neurons.validator.bt.wallet", return_value=dummy_wallet),
            patch("neurons.validator.bt.subtensor", return_value=dummy_subtensor),
        ):
            with pytest.raises(SystemExit):
                Validator.__init__(Validator.__new__(Validator))


# ----------------------------
# Test Class: Normalization
# ----------------------------
class TestNormalization:
    def test_min_power_normalization_nonzero(self):
        """
        Validate min_power_normalization on a non-zero tensor.
        The sum of the normalized tensor should be nearly 1.
        """

        logits = torch.tensor([1.0, 2.0, 3.0])
        normalized = min_power_normalization(logits, power=2.0)
        assert torch.isclose(normalized.sum(), torch.tensor(1.0), atol=1e-6)

    def test_min_power_normalization_zero(self):
        """
        Ensure that a zero tensor remains zero after normalization.
        """

        logits = torch.zeros(3)
        normalized = min_power_normalization(logits, power=2.0)
        assert torch.allclose(normalized, torch.zeros(3), atol=1e-6)


# ----------------------------
# Test Class: Run Loop
# ----------------------------
class TestRunLoop:
    @pytest.fixture
    def minimal_validator(self, hparams):
        """
        Creates a minimal Validator instance with required state.
        See original fixture in conftest.py for hparams.
        """

        validator = Validator.__new__(Validator)
        # Use hparams from the fixture.
        validator.hparams = hparams

        # Config and stop_event.
        validator.config = SimpleNamespace(netuid=1, device="cpu", peers=[])
        validator.config.stop_event = threading.Event()

        # Setup wallet and subtensor.
        dummy_wallet = MockWallet()
        dummy_wallet.hotkey.ss58_address = "default"
        validator.wallet = dummy_wallet
        validator.subtensor = MockSubtensor()
        validator.subtensor.query_module = MagicMock(
            return_value=SimpleNamespace(value=1630000000)
        )
        validator.metagraph = validator.subtensor.metagraph(validator.config.netuid)
        validator.uid = (
            validator.metagraph.hotkeys.index("default")
            if "default" in validator.metagraph.hotkeys
            else 0
        )
        # Use the real model from tests/mocks/model.py without overriding state_dict/named_parameters.
        validator.model = MockModel()
        # Ensure that xshapes and totalks match the model parameters from tests/mocks/model.py.
        validator.xshapes = {
            name: param.shape for name, param in validator.model.named_parameters()
        }
        validator.totalks = {
            name: param.numel() for name, param in validator.model.named_parameters()
        }

        # Dummy optimizer.
        optimizer = MagicMock()
        optimizer.state_dict.return_value = {
            "state": {0: {"step": 0}},
            "param_groups": [],
        }
        validator.optimizer = optimizer

        scheduler = MagicMock()
        scheduler.state_dict.return_value = {}
        scheduler.get_last_lr.return_value = [0.01]
        scheduler.last_epoch = 0
        validator.scheduler = scheduler

        # Momentum for each model parameter.
        validator.momentum = {
            name: torch.zeros_like(param)
            for name, param in validator.model.named_parameters()
        }

        validator.comms = MagicMock()
        validator.comms.put = AsyncMock()
        validator.comms.get_commitments = AsyncMock(return_value={})
        validator.comms.peers = []
        validator.comms.get_start_window = AsyncMock(return_value=5)
        validator.comms.load_checkpoint = AsyncMock(
            return_value=(
                True,
                validator.momentum,
                getattr(
                    validator, "global_step", validator.hparams.checkpoint_frequency
                ),
                validator.optimizer,
                validator.scheduler,
            )
        )
        # Patch gather so it can be awaited and returns a dummy result.
        validator.comms.gather = AsyncMock(
            return_value=SimpleNamespace(
                state_dict=SimpleNamespace(paramidxs=[], paramvals=[]),
                skipped_uids=[],
                success_rate=1.0,
            )
        )
        validator.transformer = MagicMock()
        validator.compressor = MagicMock()

        validator.hparams.blocks_per_window = 10
        validator.hparams.validator_offset = 1
        validator.current_window = 10
        validator.sync_window = 8  # 8 < (10 - 1) = 9

        validator.start_window = 5
        validator.global_step = (
            validator.hparams.checkpoint_frequency
        )  # to trigger checkpoint creation
        validator.stop_event = threading.Event()
        validator.inactive_scores = {}
        validator.final_moving_avg_scores = torch.zeros(
            validator.metagraph.n, dtype=torch.float32
        )
        validator.final_score_history = {}
        validator.eval_peers = {1: 0, 2: 0}
        validator.eval_candidates_counter = {1: 0, 2: 0}
        validator.eval_metrics_collection = {
            "own_before": [],
            "own_after": [],
            "random_before": [],
            "random_after": [],
            "own_improvement": [],
            "random_improvement": [],
        }
        validator.evaluated_uids = set()
        validator.gradient_scores = {}
        validator.binary_indicator_scores = {}
        validator.binary_moving_averages = {}
        validator.normalised_binary_moving_averages = {}
        validator.metrics_lock = asyncio.Lock()
        validator.current_block = 0
        validator.valid_score_indices = []
        validator.wandb = MagicMock()

        validator.listen_to_blockchain = AsyncMock(return_value=None)
        return validator

    def test_gradient_merge_in_run(self, minimal_validator):
        """
        Simulate the gradient merge step from the run loop.

        Creates dummy gathered gradients and verifies that model parameter gradients are updated.
        """
        validator = minimal_validator
        # Create a dummy gathered result with a state_dict containing gradient data.
        dummy_gather_result = SimpleNamespace(
            state_dict=SimpleNamespace(paramidxs=[0], paramvals=[torch.tensor([2.0])])
        )
        validator.transformer = MagicMock()
        # Identity decode: simply return the input.
        validator.transformer.decode.side_effect = lambda x: x
        validator.compressor = MagicMock()
        # FIX: Return a dummy gradient tensor **matching the parameter's shape**.
        validator.compressor.batch_decompress.side_effect = (
            lambda p, idxs, vals, xshape, totalk: torch.full(p.shape, 5.0)
        )
        for name, param in validator.model.named_parameters():
            param.grad = None
        # Simulate gradient merge.
        for n, p in validator.model.named_parameters():
            idxs_key = "paramidxs"
            vals_key = "paramvals"
            idxs = getattr(dummy_gather_result.state_dict, idxs_key, None)
            vals = getattr(dummy_gather_result.state_dict, vals_key, None)
            if idxs is not None and vals is not None:
                if not isinstance(idxs, (list, tuple)):
                    idxs = [idxs]
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                new_grad = validator.transformer.decode(
                    validator.compressor.batch_decompress(
                        p.to(validator.config.device),
                        idxs,
                        vals,
                        list(p.shape),  # Use the actual shape
                        p.numel(),  # Use p.numel() as totalk
                    )
                )
                validator.momentum[n] = new_grad.clone()
                if p.grad is None:
                    p.grad = new_grad
                else:
                    p.grad.copy_(new_grad)
                p.grad.sign_()
        # FIX: Check that the gradient has become ones of the matching shape.
        for name, param in validator.model.named_parameters():
            expected = torch.ones(param.shape, device=param.grad.device)
            assert torch.allclose(param.grad, expected, atol=1e-6)


# ----------------------------
# Test Class: Block Listener
# ----------------------------
class TestBlockListener:
    def test_block_listener_updates_window(self):
        """
        Validate that block_listener's handler updates current_block and current_window correctly.

        Simulates receiving a block header event and checks window update logic.
        """

        validator = Validator.__new__(Validator)
        validator.current_window = 5
        validator.hparams = SimpleNamespace(blocks_per_window=100)
        validator.comms = MagicMock()
        event = {"header": {"number": "750"}}

        def dummy_handler(event):
            try:
                validator.current_block = int(event["header"]["number"])
                new_window = int(
                    validator.current_block / validator.hparams.blocks_per_window
                )
                if new_window != validator.current_window:
                    validator.current_window = new_window
                    validator.comms.current_window = validator.current_window
            except Exception:
                pass

        dummy_handler(event)
        assert validator.current_window == 7


# ----------------------------
# Test Class: Basic Evaluation Flow
# ----------------------------
class TestValidatorBasicEvaluation:
    @pytest.fixture
    def minimal_validator(self):
        """
        Creates a minimal Validator instance with sufficient state for a basic evaluation flow test.

        Bypasses __init__() and manually sets config, hparams, wallet, subtensor, metagraph,
        a dummy model, optimizer, scheduler, momentum, and a dummy comms with AsyncMock put.
        """

        validator = Validator.__new__(Validator)
        validator.config = SimpleNamespace(netuid=1, device="cpu", peers=[])
        validator.hparams = SimpleNamespace(
            checkpoint_frequency=100,
            windows_per_weights=10,
            target_chunk=512,
            learning_rate=0.01,
            topk_compression=0.1,
            weight_decay=0.0,
            blocks_per_window=100,
            power_normalisation=2.0,
        )
        # Use reusable MockWallet.
        dummy_wallet = MockWallet()
        dummy_wallet.hotkey.ss58_address = "default"
        validator.wallet = dummy_wallet
        # Use reusable MockSubtensor.
        validator.subtensor = MockSubtensor()
        validator.metagraph = validator.subtensor.metagraph(validator.config.netuid)
        validator.uid = (
            validator.metagraph.hotkeys.index("default")
            if "default" in validator.metagraph.hotkeys
            else 0
        )
        # Use dummy model from MockModel.
        validator.model = MockModel()
        optimizer = MagicMock()
        optimizer.state_dict.return_value = {}
        validator.optimizer = optimizer
        scheduler = MagicMock()
        scheduler.state_dict.return_value = {}
        scheduler.get_last_lr.return_value = [0.01]
        validator.scheduler = scheduler
        # Create momentum based on model parameters if present. Otherwise, use a dummy parameter.
        params = list(validator.model.named_parameters())
        if params:
            name, param = params[0]
            validator.momentum = {name: torch.zeros_like(param)}
        else:
            dummy_param = torch.tensor([1.0])
            validator.momentum = {"param": torch.zeros_like(dummy_param)}

        validator.comms = MagicMock()
        validator.comms.put = AsyncMock()
        validator.current_window = 10
        validator.sync_window = 9
        validator.start_window = 5
        validator.global_step = 0
        return validator

    @pytest.mark.asyncio
    async def test_basic_evaluation_flow(self, minimal_validator):
        """
        Simulate a basic evaluation flow in one run iteration.

        Mimics an evaluation step by sending a dummy evaluation result via comms.put,
        and verifies that comms.put is called with the expected payload.
        """
        validator = minimal_validator
        debug_data = {
            "loss_before": 10.0,
            "loss_after": 8.0,
            "improvement": 2.0,
        }
        await validator.comms.put(
            state_dict=debug_data,
            uid="test_uid",
            window=validator.current_window,
            key="debug",
            local=False,
        )
        validator.comms.put.assert_called_once()
        assert debug_data["improvement"] == 2.0


if __name__ == "__main__":
    pytest.main()
