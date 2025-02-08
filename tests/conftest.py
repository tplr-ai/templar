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


class DummyHParams:
    active_check_interval = 60
    recent_windows = 5
    catch_up_threshold = 5
    catch_up_min_peers = 1
    catch_up_batch_size = 10
    catch_up_timeout = 300
    target_chunk = 512


class DummyMetagraph:
    pass


@pytest.fixture
def model():
    # Create a simple dummy model for testing.
    return torch.nn.Sequential(torch.nn.Linear(10, 10))


@pytest.fixture
async def comms_instance():
    wallet = DummyWallet()
    config = DummyConfig()
    hparams = DummyHParams()
    metagraph = DummyMetagraph()
    # Instantiate the transformer and compressor
    transformer = compress.TransformDCT(None, target_chunk=hparams.target_chunk)
    compressor = compress.CompressDCT()

    comms = comms_module.Comms(
        wallet=wallet,
        save_location="/tmp",
        key_prefix="model",
        config=config,
        netuid=config.netuid,
        metagraph=metagraph,
        hparams=hparams,
        uid=0,
        transformer=transformer,
        compressor=compressor,
    )
    return comms
