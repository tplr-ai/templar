"""Global pytest fixtures"""
import pytest
import os
from pathlib import Path
from unittest.mock import patch
from tests.mocks import (
    MockWallet,
    MockSubtensor,
    MockMetagraph,
    MockModel,
    MockTransformer,
    MockCompressor,
    MockOptimizer,
    MockScheduler
)
from tests.utils.env_setup import setup_test_environment
from tests.mocks.bittensor import mock_bt

@pytest.fixture(autouse=True)
def mock_config():
    """Mock the config module"""
    with patch('tplr.config.BUCKET_SECRETS', {
        "gradients": {
            "account_id": "test_account",
            "bucket_name": "test-bucket",
            "read": {
                "access_key_id": "test_read_key",
                "secret_access_key": "test_read_secret",
            },
            "write": {
                "access_key_id": "test_write_key",
                "secret_access_key": "test_write_secret",
            },
        },
        "dataset": {
            "account_id": "test_dataset_account",
            "bucket_name": "test-dataset-bucket",
            "read": {
                "access_key_id": "test_dataset_read_key",
                "secret_access_key": "test_dataset_read_secret",
            }
        }
    }), patch('tplr.config.client_config', {}):
        yield

@pytest.fixture(autouse=True)
def setup_mocks():
    """Setup global mocks"""
    with patch.dict('sys.modules', {'bittensor': mock_bt, 'bt': mock_bt}):
        yield

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "asyncio: mark test as requiring async")
    
    # Setup test environment
    setup_test_environment()

@pytest.fixture
def mock_wallet():
    """Provide a standard mock wallet"""
    return MockWallet()

@pytest.fixture
def mock_subtensor():
    """Provide a standard mock subtensor"""
    return MockSubtensor()

@pytest.fixture
def mock_metagraph():
    """Provide a standard mock metagraph"""
    return MockMetagraph()

@pytest.fixture
def mock_model():
    """Provide a standard mock model"""
    return MockModel()

@pytest.fixture
def mock_transformer():
    """Provide a standard mock transformer"""
    return MockTransformer()

@pytest.fixture
def mock_compressor():
    """Provide a standard mock compressor"""
    return MockCompressor()

@pytest.fixture
async def mock_comms():
    """Provide a standard mock comms"""
    return MockComms()

@pytest.fixture
def mock_chain():
    """Provide a standard mock chain"""
    return MockChain()

@pytest.fixture
def test_data_dir(tmp_path):
    """Provide a temporary directory for test data"""
    return tmp_path / "test_data"
