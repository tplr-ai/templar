# ruff: noqa

# Register the asyncio marker
def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as requiring async")


import pytest
import torch
import tplr.comms as comms_module
import tplr.compress as compress
from types import SimpleNamespace


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
