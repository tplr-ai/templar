"""
Unit tests for communications functionality.
Uses mocks from tests/mocks and common assertions from tests/utils/assertions.
"""

import os
import asyncio
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
from tests.utils.assertions import assert_tensor_equal

pytestmark = pytest.mark.asyncio


# --- Basic Operations Tests ---
class TestCommsBasicOperations:
    @pytest.fixture
    async def comms_instance(self, mock_wallet, mock_metagraph):
        from tplr.comms import Comms

        hparams = SimpleNamespace(
            active_check_interval=60, recent_windows=3, blocks_per_window=10, topk_compression=5
        )
        with patch("tplr.comms.Comms.get_own_bucket") as mock_get_bucket:
            mock_get_bucket.return_value = SimpleNamespace(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret",
            )
            comms = Comms(
                wallet=mock_wallet,
                save_location="/tmp",
                key_prefix="test",
                config=SimpleNamespace(netuid=1),
                metagraph=mock_metagraph,
                hparams=hparams,
                uid=1,
                totalks={},  # Provided totalks (even if empty) as required by gather.
            )
            yield comms

            # Cleanup temporary directories if they exist.
            if os.path.exists(comms.temp_dir):
                import shutil

                shutil.rmtree(comms.temp_dir)
            if os.path.exists(comms.save_location):
                import shutil

                shutil.rmtree(comms.save_location)

    @pytest.fixture
    def test_state_dict(self):
        return {
            "state_dict": {"param": torch.tensor([1, 2, 3])},
            "global_step": 42,
        }

    @pytest.fixture
    def uid(self):
        return "123"

    @pytest.fixture
    def window(self):
        return 1

    @pytest.fixture
    def key(self):
        return "testcheckpoint"

    async def test_put_local(self, comms_instance, uid, window, key, test_state_dict):
        # Ensure the local directory is clean.
        local_dir = os.path.join("/tmp/local_store", str(uid), str(window))
        if os.path.exists(local_dir):
            for filename in os.listdir(local_dir):
                os.remove(os.path.join(local_dir, filename))
            os.rmdir(local_dir)

        with patch.object(comms_instance, "cleanup_local_data") as mock_cleanup:
            await comms_instance.put(
                state_dict=test_state_dict,
                uid=uid,
                window=window,
                key=key,
                global_step=test_state_dict["global_step"],
                local=True,
                stale_retention=10,
            )
            mock_cleanup.assert_called_once()

        await asyncio.sleep(0.05)
        files = os.listdir(local_dir)
        assert len(files) == 1
        assert files[0].startswith(key)

    async def test_put_then_get_local(
        self, comms_instance, uid, window, key, test_state_dict
    ):
        # Clean local directory.
        local_dir = os.path.join("/tmp/local_store", str(uid), str(window))
        if os.path.exists(local_dir):
            for filename in os.listdir(local_dir):
                os.remove(os.path.join(local_dir, filename))
            os.rmdir(local_dir)
    
        # Store checkpoint locally.
        await comms_instance.put(
            state_dict=test_state_dict,
            uid=uid,
            window=window,
            key=key,
            global_step=test_state_dict["global_step"],
            local=True,
            stale_retention=10,
        )
        await asyncio.sleep(0.05)
    
        # Patch cleanup so it does not remove the file while retrieving.
        with pytest.MonkeyPatch.context() as mp:
            async def dummy_cleanup(uid, current_window, stale_retention):
                return None

            mp.setattr(comms_instance, "cleanup_local_data", dummy_cleanup)
            checkpoint, global_step = await comms_instance.get(
                uid=uid, window=window, key=key, local=True
            )
    
        assert checkpoint is not None, "Checkpoint returned is None"
        # Expecting the checkpoint to be stored as a dict with a "state_dict" key.
        assert "state_dict" in checkpoint, "Missing 'state_dict' key in checkpoint"
        assert torch.equal(
            checkpoint["state_dict"]["param"], test_state_dict["state_dict"]["param"]
        )
        assert global_step == test_state_dict["global_step"]

    async def test_gather_timeout(self, comms_instance):
        async def slow_get(*args, **kwargs):
            await asyncio.sleep(0.2)
            return None

        comms_instance.get_with_retry = AsyncMock(side_effect=slow_get)
        result = await comms_instance.gather(
            my_uid="0",
            uids=["1"],
            window=1,
            key="gradient",
            timeout=0.1,
            device="cpu",
            totalks={},
        )
        assert result is None


# --- Storage Operations Tests ---
class TestCommsStorageOperations:
    @pytest.fixture
    async def comms_instance(self, mock_wallet, mock_metagraph):
        from tplr.comms import Comms
        import tempfile, shutil

        temp_dir = tempfile.mkdtemp()
        save_dir = tempfile.mkdtemp()
        hparams = SimpleNamespace(
            active_check_interval=60, recent_windows=3, blocks_per_window=10
        )
        with patch("tplr.comms.Comms.get_own_bucket") as mock_get_bucket:
            mock_get_bucket.return_value = SimpleNamespace(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret",
            )
            comms = Comms(
                wallet=mock_wallet,
                save_location=save_dir,
                key_prefix="test",
                config=SimpleNamespace(netuid=1),
                metagraph=mock_metagraph,
                hparams=hparams,
                uid=1,
            )
            comms.temp_dir = temp_dir
            yield comms
            shutil.rmtree(temp_dir)
            shutil.rmtree(save_dir)

    async def test_cleanup_temp_file(self, comms_instance):
        test_file = os.path.join(comms_instance.temp_dir, "test_temp_file.npz")
        with open(test_file, "w") as f:
            f.write("test")
        await comms_instance._cleanup_temp_file(test_file)
        await asyncio.sleep(1.1)  # Allow cleanup time
        assert not os.path.exists(test_file)


# --- Gradient Batching Tests ---
class TestCommsGradientBatching:
    @pytest.fixture
    async def comms_instance(self, mock_wallet, mock_metagraph):
        from tplr.comms import Comms

        hparams = SimpleNamespace(
            active_check_interval=60, recent_windows=3, blocks_per_window=10, topk_compression=5
        )
        with patch("tplr.comms.Comms.get_own_bucket") as mock_get_bucket:
            mock_get_bucket.return_value = SimpleNamespace(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret",
            )
            comms = Comms(
                wallet=mock_wallet,
                save_location="/tmp",
                key_prefix="test",
                config=SimpleNamespace(netuid=1),
                metagraph=mock_metagraph,
                hparams=hparams,
                uid=1,
                totalks={},
            )
            yield comms

    async def test_gradient_batching(self, comms_instance):
        # Simulate a gather call to test gradient batching result structure.
        result = await comms_instance.gather(
            my_uid="0",
            uids=["1"],
            window=1,
            key="gradient",
            timeout=5,
            device="cpu",
            totalks={},
            local=True,
        )
        assert hasattr(result, "state_dict")


# --- Gather Operations Tests ---
class TestCommsGatherOperations:
    @pytest.fixture
    async def comms_instance(self, mock_wallet, mock_metagraph):
        from tplr.comms import Comms

        hparams = SimpleNamespace(
            active_check_interval=60, recent_windows=3, blocks_per_window=10, topk_compression=5
        )
        with patch("tplr.comms.Comms.get_own_bucket") as mock_get_bucket:
            mock_get_bucket.return_value = SimpleNamespace(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret",
            )
            comms = Comms(
                wallet=mock_wallet,
                save_location="/tmp",
                key_prefix="test",
                config=SimpleNamespace(netuid=1),
                metagraph=mock_metagraph,
                hparams=hparams,
                uid=1,
                totalks={},
            )
            yield comms

    async def test_gather_basic_functionality(self, comms_instance):
        result = await comms_instance.gather(
            my_uid="0",
            uids=["1"],
            window=1,
            key="gradient",
            timeout=5,
            device="cpu",
            totalks={},
        )
        # Verify that the resulting namespace includes a state_dict and expected keys.
        assert hasattr(result, "state_dict")
        state = result.state_dict
        assert "layer1.weightidxs" in state
        assert "layer1.weightvals" in state

    async def test_gather_normalization(self, comms_instance):
        result = await comms_instance.gather(
            my_uid="0",
            uids=["1"],
            window=1,
            key="gradient",
            timeout=5,
            device="cpu",
            totalks={},
        )
        # Dummy normalization check: verify tensor values.
        weight_vals = result.state_dict.get("layer1.weightvals")[0]
        total = weight_vals.sum().item()
        # This is a placeholder check; adjust the condition based on real normalization behavior.
        expected_total = weight_vals.numel() * 0.1
        assert abs(total - expected_total) < 1e-6


# --- Error Handling Tests ---
class TestCommsErrorHandling:
    @pytest.fixture
    async def comms_instance(self, mock_wallet, mock_metagraph):
        from tplr.comms import Comms

        hparams = SimpleNamespace(
            active_check_interval=60, recent_windows=3, blocks_per_window=10, topk_compression=5
        )
        with patch("tplr.comms.Comms.get_own_bucket") as mock_get_bucket:
            mock_get_bucket.return_value = SimpleNamespace(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret",
            )
            comms = Comms(
                wallet=mock_wallet,
                save_location="/tmp",
                key_prefix="test",
                config=SimpleNamespace(netuid=1),
                metagraph=mock_metagraph,
                hparams=hparams,
                uid=1,
                totalks={},
            )
            yield comms

    async def test_gather_empty_responses(self, comms_instance):
        comms_instance.get_with_retry = AsyncMock(return_value=(None, None))
        result = await comms_instance.gather(
            my_uid="0",
            uids=["1"],
            window=1,
            key="gradient",
            timeout=1,
            device="cpu",
            totalks={},
        )
        assert result is None

    async def test_gather_timeout(self, comms_instance):
        async def slow_get(*args, **kwargs):
            await asyncio.sleep(0.2)
            return None

        comms_instance.get_with_retry = AsyncMock(side_effect=slow_get)
        result = await comms_instance.gather(
            my_uid="0",
            uids=["1"],
            window=1,
            key="gradient",
            timeout=0.1,
            device="cpu",
            totalks={},
        )
        assert result is None