# ruff: noqa

# Register the asyncio marker
def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as requiring async")


import pytest
import torch
import tplr.comms as comms_module
import tplr.compress as compress
from types import SimpleNamespace
import tplr
import os
import sys
import numpy as np


# Get the project root directory (one level up from tests/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to Python path so we can import from neurons/
sys.path.insert(0, project_root)

from neurons.validator import Validator


hparams = tplr.load_hparams()


# Dummy wallet and other objects (adjust based on your actual code structure)
class DummyWallet:
    def __init__(self):
        # Simulate a wallet with the required property.
        self.hotkey = SimpleNamespace(ss58_address="dummy_address")


class DummyConfig:
    def __init__(self):
        self.netuid = 1
        self.device = "cpu"  # Add device attribute if needed by your tests


class DummyHParams:
    active_check_interval = 60
    recent_windows = 5
    catch_up_threshold = 5
    catch_up_min_peers = 1
    catch_up_batch_size = 10
    catch_up_timeout = 300
    target_chunk = 512
    topk_compression = 3  # Expected number of indices will be 3 (min(3, totalk))


class DummyMetagraph:
    pass


@pytest.fixture
def model():
    # Create a simple dummy model for testing.
    return torch.nn.Sequential(torch.nn.Linear(10, 10))


# New fixture to supply totalks information for gradient compression.
@pytest.fixture
def totalks():
    # For a Linear layer: weight shape is (10, 10) so totalk = 10*10 = 100,
    # and bias shape is (10,) so totalk = 10.
    return {"0.weight": 100, "0.bias": 10}


@pytest.fixture
async def comms_instance():
    wallet = DummyWallet()
    config = DummyConfig()
    hparams = DummyHParams()
    metagraph = DummyMetagraph()

    # Initialize Comms as per production (see miner.py)
    comms = comms_module.Comms(
        wallet=wallet,
        save_location="/tmp",
        key_prefix="model",
        config=config,
        netuid=config.netuid,
        metagraph=metagraph,
        hparams=hparams,
        uid=0,
    )

    # Manually add transformer and compressor as production code expects them to be available later.
    transformer = compress.TransformDCT(None, target_chunk=hparams.target_chunk)
    compressor = compress.CompressDCT()

    # Set expected parameter shapes and totalks.
    # For example, assume a model with a Linear layer having weight shape (10, 10) and bias (10,)
    transformer.shapes = {"0.weight": (10, 10), "0.bias": (10,)}
    # When p.shape[0]==10, we want the value 10 to be returned (so totalk for weight = 10*10 = 100).
    transformer.shape_dict = {10: 10}
    transformer.totalks = {"0.weight": 100, "0.bias": 10}

    # Attach transformer/compressor to the comms instance.
    comms.transformer = transformer
    comms.compressor = compressor
    # Also attach totalks attribute (used in gather and catch-up) matching the base parameter names.
    comms.totalks = {"0.weight": 100, "0.bias": 10}

    return comms


@pytest.fixture(autouse=True)
def enable_tplr_logger_propagation():
    tplr.logger.setLevel("INFO")
    tplr.logger.propagate = True


@pytest.fixture
def num_non_zero_incentive():
    """Default fixture for number of non-zero incentive miners."""
    return 100  # Default value if not parameterized


@pytest.fixture
def num_active_miners(request):
    """Fixture for number of active miners.
    Returns parameterized value if available, otherwise returns default."""
    try:
        return request.param
    except AttributeError:
        return 250  # Default value if not parameterized


@pytest.fixture
def mock_metagraph(mocker, num_non_zero_incentive, num_miners=250):
    """Fixture that creates a mock metagraph with a specified number of miners and incentive distribution."""
    metagraph = mocker.Mock()

    metagraph.uids = np.arange(num_miners)

    # Create incentive distribution
    non_zero_incentives = np.random.rand(num_non_zero_incentive)
    non_zero_incentives /= non_zero_incentives.sum()  # Normalize to sum to 1
    zero_incentives = np.zeros(num_miners - num_non_zero_incentive)

    # Combine and shuffle incentives
    incentives = np.concatenate([non_zero_incentives, zero_incentives])
    np.random.shuffle(incentives)
    metagraph.I = incentives

    return metagraph


@pytest.fixture
def mock_validator(mocker, mock_metagraph, num_active_miners):
    # Initialize validator without calling the constructor
    validator = object.__new__(Validator)

    # Define necessary attributes
    validator.metagraph = mock_metagraph
    validator.hparams = hparams

    validator.comms = mocker.Mock(spec=["peers", "active_peers"])
    validator.comms.peers = None

    # Use the num_active_miners parameter from the fixture
    if num_active_miners > 0:
        validator.comms.active_peers = np.random.choice(
            a=validator.metagraph.uids,
            size=num_active_miners,
            replace=False,
        )
    else:
        validator.comms.active_peers = np.array([])
    tplr.logger.info(f"Created {len(validator.comms.active_peers)} active peers.")

    return validator
