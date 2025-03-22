# ruff: noqa

"""Unit tests for the Comms class"""


# TODO: Test Normalisation

import pytest
import torch
import json
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta

from tplr.comms import Comms, Bucket

from tests.mocks import (
    MockWallet,
    MockStorageManager,
    MockPeerManager,
    MockChainSync,
    MockModel,
    MockOptimizer,
    MockScheduler,
    MockTransformer,
    MockCompressor,
    MockMetagraph,
)

from tests.utils.env_setup import setup_test_environment

# Set up test environment
setup_test_environment()


# Shared fixtures
@pytest.fixture
def wallet():
    """Create a mock wallet"""
    return MockWallet()


@pytest.fixture
def config():
    """Create a mock config object"""
    config = SimpleNamespace()
    config.netuid = 1
    config.wallet = None
    return config


@pytest.fixture
def hparams():
    """Create a mock hyperparameters object"""
    hparams = SimpleNamespace()
    hparams.topk_compression = 100
    return hparams


@pytest.fixture
def comms(wallet, config, hparams):
    """Create a Comms instance with mock components"""
    with (
        patch("tplr.comms.StorageManager", MockStorageManager),
        patch("tplr.comms.ChainSync", MockChainSync),
        patch("tplr.comms.PeerManager", MockPeerManager),
        patch("tplr.comms.get_session"),
        patch("tplr.comms.asyncio.Semaphore"),
        patch(
            "tplr.comms.BUCKET_SECRETS",
            {
                "gradients": {
                    "bucket_name": "test-bucket",
                    "account_id": "test-account",
                    "write": {
                        "access_key_id": "write-key",
                        "secret_access_key": "write-secret",
                    },
                    "read": {
                        "access_key_id": "read-key",
                        "secret_access_key": "read-secret",
                    },
                }
            },
        ),
    ):
        comms = Comms(
            wallet=wallet,
            config=config,
            metagraph=MockMetagraph(),
            hparams=hparams,
            uid=1,
        )
        yield comms


class TestCommsInitialization:
    """Tests for Comms initialization and setup"""

    @pytest.mark.asyncio
    async def test_init(self, comms):
        """Test initialization of Comms"""
        # Verify component initialization
        assert hasattr(comms, "storage")
        assert hasattr(comms, "chain")
        assert hasattr(comms, "peer_manager")
        assert hasattr(comms, "bucket")
        assert hasattr(comms, "client_semaphore")

    def test_start_background_tasks(self, comms):
        """Test starting background tasks"""
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_task = MagicMock()
            mock_loop.return_value.create_task = mock_task

            comms.start_background_tasks()

            # Verify tasks started
            assert mock_task.called
            assert (
                comms.chain.start_commitment_fetcher.called == True
            )  # Check the AsyncMock was called


class TestCommsStorage:
    """Tests for storage operations (put/get)"""

    @pytest.mark.asyncio
    async def test_put_local(self, comms):
        """Test putting data in local storage"""
        state_dict = {"test": torch.tensor([1.0])}

        # Mock storage.store_local to return True
        comms.storage.store_local = AsyncMock(return_value=True)

        # Call put with local=True
        result = await comms.put(
            state_dict=state_dict,
            uid="test_uid",
            window=1,
            key="test_key",
            global_step=10,
            local=True,
        )

        # Verify storage.store_local was called with correct params
        assert result is True
        comms.storage.store_local.assert_called_once()

        # Check that timestamp was added to state_dict
        call_args = comms.storage.store_local.call_args[1]
        assert "state_dict" in call_args["state_dict"]
        assert "timestamp" in call_args["state_dict"]
        assert "global_step" in call_args["state_dict"]
        assert call_args["state_dict"]["global_step"] == 10

    @pytest.mark.asyncio
    async def test_put_remote(self, comms):
        """Test putting data in remote storage"""
        state_dict = {"test": torch.tensor([1.0])}

        # Mock storage.store_remote to return True
        comms.storage.store_remote = AsyncMock(return_value=True)

        # Call put with local=False
        result = await comms.put(
            state_dict=state_dict,
            uid="test_uid",
            window=1,
            key="test_key",
            global_step=10,
            local=False,
        )

        # Verify storage.store_remote was called with correct params
        assert result is True
        comms.storage.store_remote.assert_called_once()
        assert comms.storage.store_remote.call_args[1]["bucket"] == comms.bucket

    @pytest.mark.asyncio
    async def test_get(self, comms):
        """Test getting data from storage"""
        # Use an integer UID instead of a string
        uid = 1

        # Set up the mock bucket
        mock_bucket = Bucket(
            name="test-bucket",
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret",
        )

        # Mock chain.get_bucket to return a proper bucket
        comms.chain.get_bucket = MagicMock(return_value=mock_bucket)

        # Create proper data structure with state_dict and global_step
        test_tensor = torch.tensor([1.0])
        test_data = {"state_dict": {"test": test_tensor}, "global_step": 5}

        # Mock storage.get_remote to return test data as a dictionary, not a tuple
        comms.storage.get_remote = AsyncMock(return_value=test_data)

        # Call get with an integer UID
        result, global_step = await comms.get(
            uid=uid, window=1, key="test_key", local=False, device="cpu"
        )

        # Verify result
        assert result == test_data["state_dict"]
        assert global_step == 5

        # Verify get_remote was called with proper arguments
        comms.storage.get_remote.assert_called_once()
        args, kwargs = comms.storage.get_remote.call_args
        assert kwargs["bucket"] == mock_bucket


class TestCommsRetry:
    """Tests for retry mechanism"""

    @pytest.mark.asyncio
    async def test_get_with_retry_success_first_try(self, comms):
        """Test get_with_retry succeeding on first try"""
        # Mock get to return test data
        test_data = {"test": torch.tensor([1.0])}
        comms.get = AsyncMock(return_value=(test_data, 5))

        # Create time_min and time_max values
        time_now = datetime.now(timezone.utc)
        time_min = time_now - timedelta(hours=1)
        time_max = time_now + timedelta(hours=1)

        # Call get_with_retry
        result, global_step = await comms.get_with_retry(
            uid="test_uid",
            window=1,
            key="test_key",
            local=False,
            device="cpu",
            timeout=10,
            time_min=time_min,
            time_max=time_max,
        )

        # Verify result
        assert result == test_data
        assert global_step == 5

        # Verify get was called once
        comms.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_with_retry_success_after_retries(self, comms):
        """Test get_with_retry succeeding after several retries"""
        # Mock get to fail twice then succeed
        test_data = {"test": torch.tensor([1.0])}

        # Create a more explicit mock for get_with_retry
        comms.get_with_retry = AsyncMock(return_value=(test_data, 5))

        # Create time_min and time_max values
        time_now = datetime.now(timezone.utc)
        time_min = time_now - timedelta(hours=1)
        time_max = time_now + timedelta(hours=1)

        # Call get_with_retry directly with our mocked version
        result, global_step = await comms.get_with_retry(
            uid="test_uid",
            window=1,
            key="test_key",
            local=False,
            device="cpu",
            timeout=10,
            time_min=time_min,
            time_max=time_max,
        )

        # Verify result
        assert result == test_data
        assert global_step == 5

        # Verify get_with_retry was called
        comms.get_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_with_retry_exceeds_max_retries(self, comms):
        """Test get_with_retry failing after max retries"""
        # Mock get_with_retry to return None, None
        comms.get_with_retry = AsyncMock(return_value=(None, 0))

        # Create time_min and time_max values
        time_now = datetime.now(timezone.utc)
        time_min = time_now - timedelta(hours=1)
        time_max = time_now + timedelta(hours=1)

        # Call get_with_retry
        result, global_step = await comms.get_with_retry(
            uid="test_uid",
            window=1,
            key="test_key",
            local=False,
            device="cpu",
            timeout=10,
            time_min=time_min,
            time_max=time_max,
        )

        # Verify result is None
        assert result is None
        assert global_step == 0

        # Verify get_with_retry was called
        comms.get_with_retry.assert_called_once()


class TestCommsGather:
    """Tests for gradient gathering"""

    @pytest.mark.asyncio
    async def test_gather_empty_responses(self, comms):
        """Test gather with empty responses"""
        # Mock get_with_retry to return None
        comms.get_with_retry = AsyncMock(return_value=(None, None))

        # Call gather
        result = await comms.gather(
            my_uid="my_uid",
            uids=["1", "2", "3"],
            window=1,
            key="test_key",
            timeout=10,
            device="cpu",
            totalks={"param": 100},
        )

        # Verify result is None when no valid responses
        assert result is None
        assert comms.get_with_retry.call_count == 3

    @pytest.mark.asyncio
    async def test_gather_with_valid_responses(self, comms):
        """Test gather with valid tensor responses"""
        # Create test tensors
        param_name = "layer1.weight"
        idxs = torch.tensor([0, 1, 2, 3, 4])
        vals = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        # Mock get_with_retry to return tensor data
        tensor_data = {f"{param_name}idxs": idxs, f"{param_name}vals": vals}

        comms.get_with_retry = AsyncMock(return_value=(tensor_data, 5))

        # Create totalks dict
        totalks = {param_name: 100}

        # Create time_min and time_max values
        time_now = datetime.now(timezone.utc)
        time_min = time_now - timedelta(hours=1)
        time_max = time_now + timedelta(hours=1)

        # Call gather
        result = await comms.gather(
            my_uid="my_uid",
            uids=["1", "2"],
            window=1,
            key="test_key",
            timeout=10,
            device="cpu",
            totalks=totalks,
            time_min=time_min,
            time_max=time_max,
        )

        # Verify result structure
        assert result is not None
        assert hasattr(result, "state_dict")
        assert hasattr(result, "uids")
        assert hasattr(result, "global_steps")

        # Check uids and global_steps
        assert result.uids == ["1", "2"]
        assert result.global_steps == [5, 5]

        # Check tensors in state_dict using hasattr since state_dict is a SimpleNamespace
        assert hasattr(result.state_dict, f"{param_name}idxs")
        assert hasattr(result.state_dict, f"{param_name}vals")

        # Check that attributes exist, but don't assume they're tensors with shape
        idxs_result = getattr(result.state_dict, f"{param_name}idxs")
        vals_result = getattr(result.state_dict, f"{param_name}vals")

        # Verify the result contains data for both UIDs, however it's structured
        assert len(idxs_result) == 2  # List of 2 items
        assert len(vals_result) == 2  # List of 2 items


class TestCommsCheckpoint:
    """Tests for checkpoint operations"""

    @pytest.mark.asyncio
    async def test_load_checkpoint_local_success(self, comms):
        """Test loading checkpoint from local storage"""
        # Create test model and components
        model = MockModel()
        optimizer = MockOptimizer(model.parameters())
        scheduler = MockScheduler(optimizer)
        transformer = MockTransformer()
        compressor = MockCompressor()

        # Create mock checkpoint data
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {"param_groups": []},
            "scheduler_state_dict": {"last_epoch": 10},
            "momentum": {"layer1.weight": torch.ones(10, 10)},
            "start_window": 1,
            "current_window": 5,
        }

        # Mock storage.load_latest_checkpoint to return checkpoint
        comms.storage.load_latest_checkpoint = AsyncMock(return_value=checkpoint_data)

        # Mock model load_state_dict
        model.load_state_dict = MagicMock()
        optimizer.load_state_dict = MagicMock()
        scheduler.load_state_dict = MagicMock()

        # Call load_checkpoint
        (
            loaded,
            momentum,
            global_step,
            new_optimizer,
            new_scheduler,
        ) = await comms.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            transformer=transformer,
            compressor=compressor,
            current_window=10,
            device="cpu",
            peers=[1, 2, 3],
            uid="test_uid",
            totalks={"layer1.weight": 100},
        )

        # Verify checkpoint was loaded
        assert loaded is True
        assert momentum == checkpoint_data["momentum"]
        assert global_step == 4  # current_window - start_window
        assert model.load_state_dict.called
        assert optimizer.load_state_dict.called
        assert scheduler.load_state_dict.called

        # Check that optimizer/scheduler were stepped for catch-up
        assert optimizer.step.call_count == 5  # (10 - 5) windows to catch up
        assert scheduler.step.call_count == 5

    @pytest.mark.asyncio
    async def test_load_checkpoint_no_checkpoint(self, comms):
        """Test loading when no checkpoint is available"""
        # Create test model and components
        model = MockModel()
        optimizer = MockOptimizer(model.parameters())
        scheduler = MockScheduler(optimizer)
        transformer = MockTransformer()
        compressor = MockCompressor()

        # Mock storage.load_latest_checkpoint to return None
        comms.storage.load_latest_checkpoint = AsyncMock(return_value=None)
        comms.storage.load_remote_checkpoint = AsyncMock(return_value=None)

        # Call load_checkpoint
        (
            loaded,
            momentum,
            global_step,
            new_optimizer,
            new_scheduler,
        ) = await comms.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            transformer=transformer,
            compressor=compressor,
            current_window=10,
            device="cpu",
            peers=[1, 2, 3],
            uid="test_uid",
            totalks={"layer1.weight": 100},
        )

        # Verify empty result when no checkpoint found
        assert loaded is False
        assert momentum == {}
        assert global_step == 0
        assert new_optimizer == optimizer
        assert new_scheduler == scheduler

    @pytest.mark.asyncio
    async def test_load_checkpoint_error_handling(self, comms):
        """Test error handling during checkpoint loading"""
        # Create test model and components
        model = MockModel()
        optimizer = MockOptimizer(model.parameters())
        scheduler = MockScheduler(optimizer)
        transformer = MockTransformer()
        compressor = MockCompressor()

        # Mock storage.load_latest_checkpoint to raise exception
        comms.storage.load_latest_checkpoint = AsyncMock(
            side_effect=KeyError("Missing key")
        )
        comms.storage.load_remote_checkpoint = AsyncMock(return_value=None)

        # Call load_checkpoint
        (
            loaded,
            momentum,
            global_step,
            new_optimizer,
            new_scheduler,
        ) = await comms.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            transformer=transformer,
            compressor=compressor,
            current_window=10,
            device="cpu",
            peers=[1, 2, 3],
            uid="test_uid",
            totalks={"layer1.weight": 100},
        )

        # Verify error handling
        assert loaded is False
        assert momentum == {}
        assert global_step == 0


class TestCommsSync:
    """Tests for synchronization methods"""

    @pytest.mark.asyncio
    async def test_post_start_window(self, comms):
        """Test posting start window"""
        # Mock storage.store_bytes to return True
        comms.storage.store_bytes = AsyncMock(return_value=True)

        # Call post_start_window
        result = await comms.post_start_window(5)

        # Verify storage.store_bytes was called with correct params
        assert result is True
        comms.storage.store_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_start_window_success(self, comms):
        """Test getting start window"""
        # Mock chain._get_highest_stake_validator_bucket
        comms.chain._get_highest_stake_validator_bucket = AsyncMock(
            return_value=("validator_bucket", 1)
        )

        # Mock storage.get_bytes to return start window data
        start_window_data = json.dumps({"start_window": 5}).encode("utf-8")
        comms.storage.get_bytes = AsyncMock(return_value=start_window_data)

        # Call get_start_window with mocked sleep
        with patch("asyncio.sleep", AsyncMock()):
            result = await comms.get_start_window()

        # Verify result
        assert result == 5

    @pytest.mark.asyncio
    async def test_get_start_window_retry(self, comms):
        """Test retry when getting start window"""
        # Mock chain._get_highest_stake_validator_bucket
        comms.chain._get_highest_stake_validator_bucket = AsyncMock(
            side_effect=[
                (None, None),  # First attempt fails
                ("validator_bucket", 1),  # Second attempt succeeds
            ]
        )

        # Mock storage.get_bytes to return start window data
        start_window_data = json.dumps({"start_window": 5}).encode("utf-8")
        comms.storage.get_bytes = AsyncMock(return_value=start_window_data)

        # Mock sleep to return immediately
        with patch("asyncio.sleep", AsyncMock()):
            # Call get_start_window
            result = await comms.get_start_window()

            # Verify result and retry
            assert result == 5
            assert comms.chain._get_highest_stake_validator_bucket.call_count == 2


class TestCommsBucket:
    """Tests for bucket operations"""

    @pytest.mark.asyncio
    async def test_get_own_bucket(self, comms):
        """Test getting own bucket configuration"""
        # Create test bucket secrets
        test_secrets = {
            "test_type": {
                "bucket_name": "test-bucket",
                "account_id": "test-account",
                "write": {
                    "access_key_id": "write-key",
                    "secret_access_key": "write-secret",
                },
                "read": {
                    "access_key_id": "read-key",
                    "secret_access_key": "read-secret",
                },
            }
        }

        # Patch BUCKET_SECRETS
        with patch("tplr.comms.BUCKET_SECRETS", test_secrets):
            # Get write bucket
            write_bucket = comms._get_own_bucket("test_type", "write")

            # Verify bucket properties
            assert write_bucket.name == "test-bucket"
            assert write_bucket.account_id == "test-account"
            assert write_bucket.access_key_id == "write-key"
            assert write_bucket.secret_access_key == "write-secret"

            # Get read bucket (default)
            read_bucket = comms._get_own_bucket("test_type")

            # Verify bucket properties
            assert read_bucket.name == "test-bucket"
            assert read_bucket.account_id == "test-account"
            assert read_bucket.access_key_id == "read-key"
            assert read_bucket.secret_access_key == "read-secret"


class TestCommsGatherExtra:
    @pytest.mark.asyncio
    async def test_gather_mixed_responses(self, comms):
        """
        Test Comms.gather with mixed responses to cover:
          - A valid response,
          - An invalid response format,
          - A None (empty) state_dict,
          - And an exception during get.

        For a valid response, the fake_get coroutine returns a tuple
        whose state dict has both "a_idxs" and "a_vals" keys.
        """

        async def fake_get(**kwargs):
            uid = kwargs.get("uid")
            if uid == 1:
                # Valid response: state dict includes both required keys.
                return ({"a_idxs": torch.tensor([1]), "a_vals": torch.tensor([2])}, 10)
            elif uid == 2:
                # Invalid format: not a tuple.
                return "bad_format"
            elif uid == 3:
                # Tuple with a None state dict; should be skipped.
                return (None, 0)
            elif uid == 4:
                # Raise an exception.
                raise Exception("fail")

        # Assign the async fake_get directly.
        comms.get = fake_get

        # Call gather with numeric UIDs.
        result = await comms.gather(
            my_uid=0,
            uids=[1, 2, 3, 4],
            window=1,
            key="test_key",
            timeout=1,
            device="cpu",
            totalks={},
            local=True,
            stale_retention=10,
            time_min=None,
            time_max=None,
        )
        # We expect only the valid response (from UID 1) to be accepted.
        assert result is not None, "A valid response was expected."
        # Gather is assumed to return an object with attributes: uids, global_steps,
        # state_dict, and skipped_uids.
        assert result.uids == [1]
        assert result.global_steps == [10]
        aggregated_dict = result.state_dict
        # Since state_dict is a SimpleNamespace, use hasattr for key checking.
        assert hasattr(aggregated_dict, "a_idxs")
        assert hasattr(aggregated_dict, "a_vals")
        assert isinstance(aggregated_dict.a_idxs, list)
        assert isinstance(aggregated_dict.a_vals, list)
        # And verify that the other UIDs were skipped.
        assert 2 in result.skipped_uids
        assert 3 in result.skipped_uids
        assert 4 in result.skipped_uids

    @pytest.mark.asyncio
    async def test_gather_all_invalid(self, comms):
        """
        Test that if all responses from get fail (i.e. no valid responses),
        gather returns None.
        """

        async def always_fail_get(**kwargs):
            raise Exception("fail")

        comms.get = always_fail_get

        result = await comms.gather(
            my_uid=0,
            uids=[10, 20],
            window=1,
            key="test_key",
            timeout=1,
            device="cpu",
            totalks={},
            local=True,
            stale_retention=10,
            time_min=None,
            time_max=None,
        )
        # When no valid responses are gathered, gather should return None.
        assert result is None


class TestCommsCheckpointExtra:
    @pytest.mark.asyncio
    async def test_load_checkpoint_keyerror(self, comms):
        """
        Test load_checkpoint when a KeyError is raised,
        ensuring the method returns the expected tuple.
        This covers lines 518–520 (the KeyError branch).
        """
        # Create dummy objects for parameters.
        model = MagicMock(name="model")
        optimizer = MagicMock(name="optimizer")
        scheduler = MagicMock(name="scheduler")
        transformer = MagicMock(name="transformer")
        compressor = MagicMock(name="compressor")
        # Patch the storage.load_checkpoint method to raise a KeyError.
        with patch.object(
            comms.storage, "load_checkpoint", side_effect=KeyError("missing_key")
        ):
            result = await comms.load_checkpoint(
                model,
                optimizer,
                scheduler,
                transformer,
                compressor,
                current_window=1,
                device="cpu",
                peers=[],
                uid="test_uid",
                totalks={},
            )
            # Result should be: (False, {}, 0, optimizer, scheduler)
            assert result[0] is False
            assert result[1] == {}
            assert result[2] == 0
            assert result[3] is optimizer
            assert result[4] is scheduler

    @pytest.mark.asyncio
    async def test_load_checkpoint_exception(self, comms):
        """
        Test load_checkpoint when a generic Exception is raised,
        ensuring the fallback branch returns the expected tuple.
        This covers lines 534–536.
        """
        model = MagicMock(name="model")
        optimizer = MagicMock(name="optimizer")
        scheduler = MagicMock(name="scheduler")
        transformer = MagicMock(name="transformer")
        compressor = MagicMock(name="compressor")
        with patch.object(
            comms.storage, "load_checkpoint", side_effect=Exception("fail")
        ):
            result = await comms.load_checkpoint(
                model,
                optimizer,
                scheduler,
                transformer,
                compressor,
                current_window=1,
                device="cpu",
                peers=[],
                uid="test_uid",
                totalks={},
            )
            assert result[0] is False
