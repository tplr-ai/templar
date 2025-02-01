"""Unit tests for communications functionality"""
import os
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
from ..utils.assertions import assert_tensor_equal
import asyncio

# Mark all tests as async
pytestmark = pytest.mark.asyncio

class TestCommsBasicOperations:
    """Test basic communication operations"""
    
    @pytest.fixture
    async def comms_instance(self, mock_wallet, mock_metagraph):
        """Create comms instance with standard mocks"""
        from tplr.comms import Comms
        
        hparams = SimpleNamespace(
            active_check_interval=60,
            recent_windows=3
        )
        
        with patch("tplr.comms.Comms.get_own_bucket") as mock_get_bucket:
            mock_get_bucket.return_value = SimpleNamespace(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret"
            )
            
            comms = Comms(
                wallet=mock_wallet,
                save_location="/tmp",
                key_prefix="test",
                config=SimpleNamespace(netuid=1),
                metagraph=mock_metagraph,
                hparams=hparams,
                uid=1
            )
            
            yield comms
            
            # Cleanup
            if os.path.exists(comms.temp_dir):
                import shutil
                shutil.rmtree(comms.temp_dir)
            if os.path.exists(comms.save_location):
                shutil.rmtree(comms.save_location)

    async def test_put_local(self, comms_instance):
        """Test putting data to local storage"""
        test_state_dict = {"param": torch.tensor([1, 2, 3])}
        uid = "0"
        window = 1
        key = "gradient"

        # Clean up test directory first
        expected_dir = os.path.join("/tmp/local_store", uid, str(window))
        base_dir = os.path.dirname(expected_dir)

        if os.path.exists(base_dir):
            import shutil
            shutil.rmtree(base_dir)

        # Test put operation
        with patch.object(comms_instance, "cleanup_local_data") as mock_cleanup:
            await comms_instance.put(
                state_dict=test_state_dict,
                uid=uid,
                window=window,
                key=key,
                local=True
            )
            mock_cleanup.assert_called_once()

        # Verify file was saved
        files = os.listdir(expected_dir)
        assert len(files) == 1
        assert files[0].startswith(key)

    async def test_get_local(self, comms_instance):
        """Test getting data from local storage"""
        test_state_dict = {
            "state_dict": {"param": torch.tensor([1, 2, 3])},
            "global_step": 10
        }
        uid = "0"
        window = 1
        key = "gradient"
        
        # Prepare local file
        local_dir = os.path.join("/tmp/local_store", uid, str(window))
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, f"{key}-{window}-{uid}-v1.0.0.pt")
        torch.save(test_state_dict, local_path)

        # Test get operation
        with patch.object(comms_instance, "cleanup_local_data") as mock_cleanup:
            state_dict, global_step = await comms_instance.get(
                uid=uid,
                window=window,
                key=key,
                local=True
            )
            mock_cleanup.assert_called_once()

        # Verify retrieved data
        assert torch.equal(state_dict["param"], test_state_dict["state_dict"]["param"])
        assert global_step == test_state_dict["global_step"]

class TestCommsGatherOperations:
    """Test gather functionality"""
    
    @pytest.fixture
    async def comms_instance(self, mock_wallet, mock_metagraph):
        """Create comms instance for gather tests"""
        from tplr.comms import Comms
        
        hparams = SimpleNamespace(
            active_check_interval=60,
            recent_windows=3
        )
        
        with patch("tplr.comms.Comms.get_own_bucket") as mock_get_bucket:
            mock_get_bucket.return_value = SimpleNamespace(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret"
            )
            
            return Comms(
                wallet=mock_wallet,
                save_location="/tmp",
                key_prefix="test",
                config=SimpleNamespace(netuid=1),
                metagraph=mock_metagraph,
                hparams=hparams,
                uid=1
            )

    async def test_gather_basic_functionality(self, comms_instance):
        """Test basic gather operation"""
        state_dict = {
            "layer1.weightsidxs": torch.tensor([0, 1, 2]),
            "layer1.weightsvals": torch.tensor([0.1, 0.2, 0.3])
        }

        # Mock peer responses
        peer1_response = (
            {
                "layer1.weightsidxs": torch.tensor([0, 1, 2]),
                "layer1.weightsvals": torch.tensor([0.4, 0.5, 0.6])
            },
            1
        )
        peer2_response = (
            {
                "layer1.weightsidxs": torch.tensor([0, 1, 2]),
                "layer1.weightsvals": torch.tensor([0.7, 0.8, 0.9])
            },
            2
        )

        comms_instance.get_with_retry = AsyncMock(side_effect=[peer1_response, peer2_response])

        result = await comms_instance.gather(
            state_dict=state_dict,
            my_uid="0",
            uids=["1", "2"],
            window=1,
            key="gradient",
            timeout=5,
            device="cpu",
            global_step=0,
            local=True
        )

        # Verify result structure
        assert result is not None
        assert hasattr(result, "state_dict")
        assert hasattr(result, "uids")
        assert hasattr(result, "global_steps")
        assert len(result.uids) == 2
        assert len(result.global_steps) == 2

    async def test_gather_normalization(self, comms_instance):
        """Test gradient normalization in gather"""
        vals = torch.tensor([3.0, 4.0])  # norm should be 5
        state_dict = {
            "layer.idxs": torch.tensor([0, 1]),
            "layer.vals": vals
        }

        comms_instance.get_with_retry = AsyncMock(return_value=(state_dict, 1))

        result = await comms_instance.gather(
            state_dict=None,
            my_uid="0",
            uids=["1"],
            window=1,
            key="gradient",
            timeout=5,
            device="cpu",
            global_step=0
        )

        # Verify normalization
        normalized_vals = getattr(result.state_dict, "layer.vals")[0]
        expected_norm = torch.tensor([0.6, 0.8])  # [3/5, 4/5]
        assert_tensor_equal(normalized_vals, expected_norm)

class TestCommsErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.fixture
    async def comms_instance(self, mock_wallet, mock_metagraph):
        """Create comms instance for error tests"""
        from tplr.comms import Comms
        
        hparams = SimpleNamespace(
            active_check_interval=60,
            recent_windows=3
        )
        
        with patch("tplr.comms.Comms.get_own_bucket") as mock_get_bucket:
            mock_get_bucket.return_value = SimpleNamespace(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret"
            )
            
            return Comms(
                wallet=mock_wallet,
                save_location="/tmp",
                key_prefix="test",
                config=SimpleNamespace(netuid=1),
                metagraph=mock_metagraph,
                hparams=hparams,
                uid=1
            )

    async def test_gather_empty_responses(self, comms_instance):
        """Test handling of empty gather responses"""
        comms_instance.get_with_retry = AsyncMock(side_effect=[None, (None, 0)])

        result = await comms_instance.gather(
            state_dict=None,
            my_uid="0",
            uids=["1", "2"],
            window=1,
            key="gradient",
            timeout=5,
            device="cpu",
            global_step=0,
            local=True
        )

        assert result is None

    async def test_gather_timeout(self, comms_instance):
        """Test gather operation with timeout"""
        async def slow_get(*args, **kwargs):
            await asyncio.sleep(0.2)  # Sleep longer than timeout
            return None

        comms_instance.get_with_retry = AsyncMock(side_effect=slow_get)

        result = await comms_instance.gather(
            state_dict=None,
            my_uid="0",
            uids=["1"],
            window=1,
            key="gradient",
            timeout=0.1,  # Short timeout
            device="cpu",
            global_step=0
        )

        assert result is None

class TestCommsStorageOperations:
    """Test storage operations and cleanup"""
    
    @pytest.fixture
    async def comms_instance(self, mock_wallet, mock_metagraph):
        """Create comms instance for storage tests"""
        from tplr.comms import Comms
        import tempfile
        
        # Create temp directories
        temp_dir = tempfile.mkdtemp()
        save_dir = tempfile.mkdtemp()
        
        hparams = SimpleNamespace(
            active_check_interval=60,
            recent_windows=3
        )
        
        with patch("tplr.comms.Comms.get_own_bucket") as mock_get_bucket:
            mock_get_bucket.return_value = SimpleNamespace(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret"
            )
            
            comms = Comms(
                wallet=mock_wallet,
                save_location=save_dir,
                key_prefix="test",
                config=SimpleNamespace(netuid=1),
                metagraph=mock_metagraph,
                hparams=hparams,
                uid=1
            )
            
            # Override temp dir
            comms.temp_dir = temp_dir
            
            yield comms
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
            shutil.rmtree(save_dir)

    async def test_store_gradient_data_success(self, comms_instance):
        """Test successful gradient data storage"""
        uid = "1"
        window = 10
        global_step = 5
        state_dict_resp = {
            "layer1.weight": torch.tensor([1.0, 2.0, 3.0]),
            "layer1.bias": torch.tensor([0.1, 0.2])
        }
        global_step_resp = 5

        # Mock s3_put_object
        comms_instance.s3_put_object = AsyncMock()

        await comms_instance._store_gradient_data(
            uid=uid,
            window=window,
            global_step=global_step,
            state_dict_resp=state_dict_resp,
            global_step_resp=global_step_resp
        )

        # Wait for tasks
        await asyncio.sleep(0.1)

        # Verify s3_put_object was called correctly
        assert comms_instance.s3_put_object.called
        call_args = comms_instance.s3_put_object.call_args
        assert call_args is not None
        assert call_args.kwargs["key"].startswith(f"gathers/v1.0.0/{uid}/{window}/")

    async def test_cleanup_temp_file(self, comms_instance):
        """Test temporary file cleanup"""
        # Create test file
        test_file = os.path.join(comms_instance.temp_dir, "test_temp_file.npz")
        with open(test_file, "w") as f:
            f.write("test")

        await comms_instance._cleanup_temp_file(test_file)
        await asyncio.sleep(1.1)  # Wait for cleanup

        assert not os.path.exists(test_file)

class TestCommsGradientBatching:
    """Test gradient batching functionality"""
    
    @pytest.fixture
    async def comms_instance(self, mock_wallet, mock_metagraph):
        """Create comms instance for batch tests"""
        from tplr.comms import Comms
        
        hparams = SimpleNamespace(
            active_check_interval=60,
            recent_windows=3
        )
        
        with patch("tplr.comms.Comms.get_own_bucket") as mock_get_bucket:
            mock_get_bucket.return_value = SimpleNamespace(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret"
            )
            
            return Comms(
                wallet=mock_wallet,
                save_location="/tmp",
                key_prefix="test",
                config=SimpleNamespace(netuid=1),
                metagraph=mock_metagraph,
                hparams=hparams,
                uid=1
            )

    async def test_gather_with_batching(self, comms_instance):
        """Test gather operation with batched processing"""
        state_dict = {
            "layer1.weightsidxs": torch.tensor([0, 1, 2]),
            "layer1.weightsvals": torch.tensor([0.1, 0.2, 0.3])
        }

        # Create multiple peer responses
        peer_responses = [
            (
                {
                    "layer1.weightsidxs": torch.tensor([i, i+1]),
                    "layer1.weightsvals": torch.tensor([0.1*i, 0.2*i])
                },
                i
            ) for i in range(1, 8)
        ]

        call_count = 0
        async def mock_get_with_retry(*args, **kwargs):
            nonlocal call_count
            if call_count < len(peer_responses):
                response = peer_responses[call_count]
                call_count += 1
                return response
            return None

        comms_instance.get_with_retry = AsyncMock(side_effect=mock_get_with_retry)

        result = await comms_instance.gather(
            state_dict=state_dict,
            my_uid="0",
            uids=[str(i) for i in range(1, 8)],
            window=1,
            key="gradient",
            timeout=5,
            device="cpu",
            global_step=0,
            local=True
        )

        # Verify batched results
        assert result is not None
        assert len(result.uids) == 7
        assert len(result.global_steps) == 7
        
        # Verify tensor batching
        vals = getattr(result.state_dict, "layer1.weightsvals")
        assert isinstance(vals, list)
        assert len(vals) == 7
        assert all(isinstance(v, torch.Tensor) for v in vals)

# Continue with more test classes... 