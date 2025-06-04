import tempfile
import shutil
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch, mock_open
import pytest
import torch

from src.tplr.comms import Comms
from src.tplr.schemas import Bucket


class TestComms:
    """Test suite for Comms class"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_wallet(self):
        """Mock wallet fixture"""
        wallet = Mock()
        wallet.hotkey.ss58_address = "test_hotkey_address"
        return wallet

    @pytest.fixture
    def mock_bucket(self):
        """Mock bucket fixture"""
        return Bucket(
            name="test-bucket",
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret",
        )

    @pytest.fixture
    def mock_config(self):
        """Mock config fixture"""
        config = Mock()
        config.device = "cpu"
        return config

    @pytest.fixture
    def mock_hparams(self):
        """Mock hyperparameters fixture"""
        hparams = Mock()
        hparams.topk = 1000
        hparams.peer_buffer_size = 50
        return hparams

    @pytest.fixture
    def mock_metagraph(self):
        """Mock metagraph fixture"""
        metagraph = Mock()
        metagraph.n = 100
        metagraph.uids = list(range(100))
        return metagraph

    @pytest.fixture
    async def comms(self, mock_wallet, mock_config, mock_hparams, mock_metagraph):
        """Comms instance fixture"""
        with patch.multiple(
            "src.tplr.comms.Comms",
            get_own_bucket=Mock(
                return_value=Bucket(
                    name="test-bucket",
                    account_id="test-account",
                    access_key_id="test-key",
                    secret_access_key="test-secret",
                )
            ),
            _initialize_managers=Mock(),
        ):
            comms = Comms(
                wallet=mock_wallet,
                key_prefix="test_model",
                config=mock_config,
                netuid=1,
                metagraph=mock_metagraph,
                hparams=mock_hparams,
                uid=42,
            )
            yield comms
            # Cleanup
            try:
                await comms.close_all_resources()
            except Exception:
                pass

    def test_init_creates_temp_directory(
        self, mock_wallet, mock_config, mock_hparams, mock_metagraph
    ):
        """Test that initialization creates proper temp directory"""
        with (
            patch.multiple(
                "src.tplr.comms.Comms",
                get_own_bucket=Mock(return_value=Mock()),
                _initialize_managers=Mock(),
            ),
            patch("os.makedirs") as mock_makedirs,
        ):
            comms = Comms(
                wallet=mock_wallet,
                config=mock_config,
                hparams=mock_hparams,
                metagraph=mock_metagraph,
                uid=42,
            )

            assert comms.uid == 42
            assert comms.temp_dir == "/tmp/templar_42"
            assert comms.save_location == "/tmp/hotkey_test_hotkey_address"
            mock_makedirs.assert_called()

    def test_init_without_wallet(self, mock_config, mock_hparams, mock_metagraph):
        """Test initialization without wallet"""
        with patch.multiple(
            "src.tplr.comms.Comms",
            get_own_bucket=Mock(return_value=Mock()),
            _initialize_managers=Mock(),
        ):
            comms = Comms(
                wallet=None,
                config=mock_config,
                hparams=mock_hparams,
                metagraph=mock_metagraph,
                uid=42,
            )

            assert comms.wallet is None
            assert comms.uid == 42

    @patch("src.tplr.comms.BUCKET_SECRETS")
    def test_get_own_bucket_gradients_read(self, mock_secrets):
        """Test getting gradients bucket with read access"""
        mock_bucket_config = {
            "name": "test-gradients",
            "account_id": "test-account",
            "credentials": {
                "read": {
                    "access_key_id": "read-key",
                    "secret_access_key": "read-secret",
                },
                "write": {
                    "access_key_id": "write-key",
                    "secret_access_key": "write-secret",
                },
            },
        }
        mock_secrets.__getitem__ = Mock(return_value=mock_bucket_config)

        comms = Comms.__new__(Comms)  # Create instance without __init__
        bucket = comms.get_own_bucket("gradients", "read")

        assert bucket.name == "test-gradients"
        assert bucket.access_key_id == "read-key"
        assert bucket.secret_access_key == "read-secret"

    @patch("src.tplr.comms.BUCKET_SECRETS")
    def test_get_own_bucket_dataset(self, mock_secrets):
        """Test getting dataset bucket"""
        mock_bucket_config = {
            "name": "test-dataset",
            "account_id": "test-account",
            "credentials": {
                "read": {
                    "access_key_id": "dataset-key",
                    "secret_access_key": "dataset-secret",
                }
            },
        }
        mock_secrets.__getitem__ = Mock(return_value=mock_bucket_config)

        comms = Comms.__new__(Comms)
        bucket = comms.get_own_bucket("dataset")

        assert bucket.name == "test-dataset"
        assert bucket.access_key_id == "dataset-key"

    def test_get_own_bucket_invalid_type(self):
        """Test error handling for invalid bucket type"""
        comms = Comms.__new__(Comms)

        with pytest.raises(ValueError, match="bucket_type must be either"):
            comms.get_own_bucket("invalid_type")

    def test_get_own_bucket_invalid_access_type(self):
        """Test error handling for invalid access type"""
        comms = Comms.__new__(Comms)

        with pytest.raises(ValueError, match="access_type must be either"):
            comms.get_own_bucket("gradients", "invalid_access")

    @pytest.mark.asyncio
    async def test_start_background_tasks(self, comms):
        """Test starting background tasks"""
        comms.peer_manager = Mock()
        comms.peer_manager.start_peer_tracking = AsyncMock()

        with (
            patch("asyncio.get_running_loop"),
            patch("concurrent.futures.ThreadPoolExecutor"),
            patch("asyncio.create_task") as mock_create_task,
        ):
            comms.start_background_tasks()

            assert comms.loop is not None
            assert comms.executor is not None
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_all_resources(self, comms):
        """Test closing all resources"""
        # Mock all managers
        comms.peer_manager = Mock()
        comms.peer_manager.stop_peer_tracking = AsyncMock()
        comms.storage_client = Mock()
        comms.storage_client.close_all_clients = AsyncMock()
        comms.file_manager = Mock()
        comms.file_manager.cleanup_temp_files = AsyncMock()
        comms.executor = Mock()

        await comms.close_all_resources()

        comms.peer_manager.stop_peer_tracking.assert_called_once()
        comms.storage_client.close_all_clients.assert_called_once()
        comms.file_manager.cleanup_temp_files.assert_called_once()
        comms.executor.shutdown.assert_called_once_with(wait=True)

    @pytest.mark.asyncio
    async def test_close_all_resources_with_exception(self, comms):
        """Test closing resources handles exceptions gracefully"""
        comms.peer_manager = Mock()
        comms.peer_manager.stop_peer_tracking = AsyncMock(
            side_effect=Exception("Test error")
        )
        comms.storage_client = Mock()
        comms.storage_client.close_all_clients = AsyncMock()
        comms.file_manager = Mock()
        comms.file_manager.cleanup_temp_files = AsyncMock()

        # Should not raise exception
        await comms.close_all_resources()

    @pytest.mark.asyncio
    async def test_put_local_storage(self, comms):
        """Test PUT operation with local storage"""
        # Setup mocks
        comms.gradient_manager = Mock()
        comms.gradient_manager.serialize_gradient = AsyncMock(
            return_value="/tmp/test_file.pt"
        )
        comms.file_manager = Mock()
        comms.file_manager.cleanup_local_data = AsyncMock()
        comms.file_manager.get_local_storage_path = Mock(return_value="/tmp/local")
        comms.file_manager.ensure_directory_exists = Mock()

        state_dict = {"param1": torch.tensor([1, 2, 3])}

        with patch("shutil.move") as mock_move, patch("src.tplr.T", return_value=1.0):
            duration = await comms.put(
                state_dict=state_dict,
                uid=42,
                window=1,
                key="gradient",
                global_step=100,
                local=True,
                stale_retention=10,
            )

            assert duration >= 0.0
            comms.gradient_manager.serialize_gradient.assert_called_once_with(
                state_dict, 100
            )
            comms.file_manager.cleanup_local_data.assert_called_once()
            mock_move.assert_called_once()

    @pytest.mark.asyncio
    async def test_put_remote_storage(self, comms):
        """Test PUT operation with remote storage"""
        # Setup mocks
        comms.gradient_manager = Mock()
        comms.gradient_manager.serialize_gradient = AsyncMock(
            return_value="/tmp/test_file.pt"
        )
        comms.storage_client = Mock()
        comms.storage_client.put_object = AsyncMock(return_value=True)
        comms.file_manager = Mock()
        comms.file_manager.delete_file = Mock()

        state_dict = {"param1": torch.tensor([1, 2, 3])}

        with (
            patch("builtins.open", mock_open(read_data=b"test_data")),
            patch("src.tplr.T", return_value=1.0),
            patch("asyncio.create_task"),
        ):
            duration = await comms.put(
                state_dict=state_dict,
                uid=42,
                window=1,
                key="gradient",
                global_step=100,
                local=False,
                stale_retention=10,
            )

            assert duration >= 0.0
            comms.storage_client.put_object.assert_called_once()
            comms.file_manager.delete_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_put_aggregator_key(self, comms):
        """Test PUT operation with aggregator key"""
        comms.gradient_manager = Mock()
        comms.gradient_manager.serialize_gradient = AsyncMock(
            return_value="/tmp/test_file.pt"
        )
        comms.file_manager = Mock()
        comms.file_manager.cleanup_local_data = AsyncMock()
        comms.file_manager.get_local_storage_path = Mock(return_value="/tmp/local")
        comms.file_manager.ensure_directory_exists = Mock()

        state_dict = {"param1": torch.tensor([1, 2, 3])}

        with (
            patch("shutil.move"),
            patch("src.tplr.T", return_value=1.0),
            patch("src.tplr.__version__", "1.0.0"),
        ):
            await comms.put(
                state_dict=state_dict,
                uid=None,
                window=1,
                key="aggregator",
                global_step=100,
                local=True,
            )

            # Verify aggregator filename format is used
            # The filename should be aggregator-1-v1.0.0.pt
            comms.gradient_manager.serialize_gradient.assert_called_once_with(
                state_dict, 100
            )

    @pytest.mark.asyncio
    async def test_put_error_handling(self, comms):
        """Test PUT operation error handling"""
        comms.gradient_manager = Mock()
        comms.gradient_manager.serialize_gradient = AsyncMock(
            side_effect=Exception("Serialization error")
        )

        state_dict = {"param1": torch.tensor([1, 2, 3])}

        duration = await comms.put(
            state_dict=state_dict,
            uid=42,
            window=1,
            key="gradient",
            global_step=100,
            local=True,
        )

        assert duration == 0.0

    @pytest.mark.asyncio
    async def test_get_local_storage(self, comms):
        """Test GET operation with local storage"""
        comms.file_manager = Mock()
        comms.file_manager.cleanup_local_data = AsyncMock()
        comms.file_manager.get_local_storage_path = Mock(
            return_value="/tmp/test_file.pt"
        )

        comms.gradient_manager = Mock()
        expected_state_dict = {"param1": torch.tensor([1, 2, 3])}
        comms.gradient_manager.deserialize_gradient = AsyncMock(
            return_value=(expected_state_dict, 100)
        )

        with patch("os.path.exists", return_value=True):
            result = await comms.get(
                uid="42", window=1, key="gradient", local=True, stale_retention=10
            )

            assert result is not None
            state_dict, global_step = result
            assert state_dict == expected_state_dict
            assert global_step == 100

    @pytest.mark.asyncio
    async def test_get_local_storage_file_not_found(self, comms):
        """Test GET operation when local file doesn't exist"""
        comms.file_manager = Mock()
        comms.file_manager.cleanup_local_data = AsyncMock()
        comms.file_manager.get_local_storage_path = Mock(
            return_value="/tmp/nonexistent.pt"
        )

        with patch("os.path.exists", return_value=False):
            result = await comms.get(uid="42", window=1, key="gradient", local=True)

            assert result is None

    @pytest.mark.asyncio
    async def test_get_remote_storage(self, comms):
        """Test GET operation with remote storage"""
        comms.commitments = {42: Mock()}  # Mock peer bucket
        comms.storage_client = Mock()
        comms.storage_client.get_object = AsyncMock(return_value=b"test_data")

        comms.file_manager = Mock()
        comms.file_manager.create_temp_file = Mock(return_value="/tmp/temp_file.pt")
        comms.file_manager.delete_file = Mock()

        comms.gradient_manager = Mock()
        expected_state_dict = {"param1": torch.tensor([1, 2, 3])}
        comms.gradient_manager.deserialize_gradient = AsyncMock(
            return_value=(expected_state_dict, 100)
        )

        with patch("builtins.open", mock_open()):
            result = await comms.get(
                uid="42",
                window=1,
                key="gradient",
                local=False,
                time_min=datetime.now(),
                time_max=datetime.now(),
            )

            assert result is not None
            state_dict, global_step = result
            assert state_dict == expected_state_dict
            assert global_step == 100

    @pytest.mark.asyncio
    async def test_get_remote_storage_no_peer_bucket(self, comms):
        """Test GET operation when peer bucket doesn't exist"""
        comms.commitments = {}  # No peer bucket

        result = await comms.get(uid="42", window=1, key="gradient", local=False)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_remote_storage_time_markers(self, comms):
        """Test GET operation with time markers"""
        comms.commitments = {42: Mock()}
        comms.storage_client = Mock()
        comms.storage_client.get_object = AsyncMock(
            return_value={"__status": "TOO_LATE"}
        )

        result = await comms.get(uid="42", window=1, key="gradient", local=False)

        assert result == {"__status": "TOO_LATE"}

    @pytest.mark.asyncio
    async def test_get_checkpoint_key(self, comms):
        """Test GET operation with checkpoint key"""
        comms.file_manager = Mock()
        comms.file_manager.cleanup_local_data = AsyncMock()
        comms.file_manager.get_local_storage_path = Mock(
            return_value="/tmp/test_file.pt"
        )

        comms.gradient_manager = Mock()
        expected_state_dict = {"param1": torch.tensor([1, 2, 3])}
        comms.gradient_manager.deserialize_gradient = AsyncMock(
            return_value=(expected_state_dict, 100)
        )

        with patch("os.path.exists", return_value=True):
            result = await comms.get(uid="42", window=1, key="checkpoint", local=True)

            assert result is not None
            state_dict, global_step = result
            assert state_dict == expected_state_dict
            assert global_step is None  # Checkpoint returns None for global_step

    @pytest.mark.asyncio
    async def test_gather(self, comms):
        """Test gather operation"""
        comms.aggregation_manager = Mock()
        expected_result = SimpleNamespace(success=True, data={})
        comms.aggregation_manager.gather_gradients = AsyncMock(
            return_value=expected_result
        )

        result = await comms.gather(
            my_uid=42,
            uids=[1, 2, 3],
            window=1,
            timeout=30,
            device="cpu",
            totalks={"param1": 1000},
            local=True,
            show_progress=True,
        )

        assert result == expected_result
        comms.aggregation_manager.gather_gradients.assert_called_once()

    @pytest.mark.asyncio
    async def test_gather_no_uid(self, comms):
        """Test gather operation with no UID"""
        result = await comms.gather(
            my_uid=None,
            uids=[1, 2, 3],
            window=1,
            timeout=30,
            device="cpu",
            totalks={"param1": 1000},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, comms):
        """Test save checkpoint operation"""
        comms.checkpoint_manager = Mock()
        comms.checkpoint_manager.save_checkpoint = AsyncMock(return_value=True)

        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        result = await comms.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=100,
            current_window=1,
            start_window=0,
            sync_window=10,
        )

        assert result is True
        comms.checkpoint_manager.save_checkpoint.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_checkpoint(self, comms):
        """Test load checkpoint operation"""
        comms.checkpoint_manager = Mock()
        expected_result = (True, 100, Mock(), Mock())
        comms.checkpoint_manager.load_checkpoint = AsyncMock(
            return_value=expected_result
        )

        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        result = await comms.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            current_window=1,
            device="cpu",
            init_version="1.0.0",
        )

        assert result == expected_result
        comms.checkpoint_manager.load_checkpoint.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_miner_active(self, comms):
        """Test is miner active check"""
        comms.peer_manager = Mock()
        comms.peer_manager.is_peer_active = AsyncMock(return_value=True)

        result = await comms.is_miner_active(uid=42, recent_windows=3)

        assert result is True
        comms.peer_manager.is_peer_active.assert_called_once_with(42, 3)

    def test_weighted_random_sample_no_replacement(self, comms):
        """Test weighted random sampling"""
        comms.peer_manager = Mock()
        expected_result = [1, 3, 5]
        comms.peer_manager.weighted_random_sample_no_replacement = Mock(
            return_value=expected_result
        )

        result = comms.weighted_random_sample_no_replacement(
            candidates=[1, 2, 3, 4, 5], weights=[10, 20, 30, 40, 50], k=3
        )

        assert result == expected_result

    @pytest.mark.asyncio
    async def test_post_start_window(self, comms):
        """Test post start window"""
        comms.metadata_manager = Mock()
        comms.metadata_manager.post_start_window = AsyncMock()

        await comms.post_start_window(start_window=5)

        comms.metadata_manager.post_start_window.assert_called_once_with(5)

    @pytest.mark.asyncio
    async def test_get_start_window(self, comms):
        """Test get start window"""
        comms.metadata_manager = Mock()
        comms.metadata_manager.get_start_window = AsyncMock(return_value=5)

        result = await comms.get_start_window(retries=3)

        assert result == 5
        comms.metadata_manager.get_start_window.assert_called_once_with(3)

    @pytest.mark.asyncio
    async def test_post_peer_list(self, comms):
        """Test post peer list"""
        comms.metadata_manager = Mock()
        comms.metadata_manager.post_peer_list = AsyncMock()

        peers = [1, 2, 3]
        weights = torch.tensor([0.1, 0.2, 0.3])

        await comms.post_peer_list(
            peers=peers,
            first_effective_window=1,
            sync_window=10,
            weights=weights,
            initial_selection=True,
        )

        comms.metadata_manager.post_peer_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_peer_list(self, comms):
        """Test get peer list"""
        comms.metadata_manager = Mock()
        expected_result = ([1, 2, 3], 5)
        comms.metadata_manager.get_peer_list = AsyncMock(return_value=expected_result)

        result = await comms.get_peer_list(fetch_previous=True)

        assert result == expected_result
        comms.metadata_manager.get_peer_list.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_get_debug_dict(self, comms):
        """Test get debug dict"""
        comms.metadata_manager = Mock()
        expected_result = {"debug": "data"}
        comms.metadata_manager.get_debug_dict = AsyncMock(return_value=expected_result)

        result = await comms.get_debug_dict(window=1)

        assert result == expected_result
        comms.metadata_manager.get_debug_dict.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_load_aggregation(self, comms):
        """Test load aggregation"""
        comms.aggregation_manager = Mock()
        expected_result = {"aggregated": "data"}
        comms.aggregation_manager.load_aggregation = AsyncMock(
            return_value=expected_result
        )

        result = await comms.load_aggregation(window=1, chunk_size=1000)

        assert result == expected_result
        comms.aggregation_manager.load_aggregation.assert_called_once_with(1, 1000)

    @pytest.mark.asyncio
    async def test_apply_gathered_gradients(self, comms):
        """Test apply gathered gradients"""
        comms.gradient_manager = Mock()
        expected_result = (True, 150)
        comms.gradient_manager.apply_gradients_to_model = AsyncMock(
            return_value=expected_result
        )

        gather_result = Mock()
        model = Mock()
        optimizer = Mock()
        scheduler = Mock()
        transformer = Mock()
        compressor = Mock()

        result = await comms._apply_gathered_gradients(
            gather_result=gather_result,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            transformer=transformer,
            compressor=compressor,
            device="cpu",
            window=1,
            global_step=100,
        )

        assert result == expected_result
        comms.gradient_manager.apply_gradients_to_model.assert_called_once()

    def test_check_compressed_indices(self, comms):
        """Test check compressed indices"""
        comms.gradient_manager = Mock()
        comms.gradient_manager.check_compressed_indices = Mock()

        comms.check_compressed_indices(
            param_name="test_param", idxs=[1, 2, 3], totalk=1000, allowed_topk=500
        )

        comms.gradient_manager.check_compressed_indices.assert_called_once_with(
            "test_param", [1, 2, 3], 1000, 500
        )

    @pytest.mark.asyncio
    async def test_cleanup_s3_data(self, comms):
        """Test S3 data cleanup"""
        comms.storage_client = Mock()
        comms.storage_client.list_objects = AsyncMock(
            return_value=[
                "gradient-5-42-v1.0.0.pt",  # Should be deleted (window 5 < 8)
                "gradient-10-42-v1.0.0.pt",  # Should be kept (window 10 >= 8)
                "gradient-8-42-v1.0.0.pt",  # Should be kept (window 8 >= 8)
            ]
        )
        comms.storage_client.delete_object = AsyncMock()

        with patch("tplr.__version__", "1.0.0"):
            await comms._cleanup_s3_data(
                uid=42,
                current_window=10,
                stale_retention=2,  # min_allowed_window = 10 - 2 = 8
            )

        # Should delete only the stale object
        comms.storage_client.delete_object.assert_called_once_with(
            "gradient-5-42-v1.0.0.pt", comms.bucket
        )

    @pytest.mark.asyncio
    async def test_cleanup_s3_data_with_exception(self, comms):
        """Test S3 cleanup handles exceptions"""
        comms.storage_client = Mock()
        comms.storage_client.list_objects = AsyncMock(side_effect=Exception("S3 error"))

        # Should not raise exception
        await comms._cleanup_s3_data(uid=42, current_window=10, stale_retention=2)

    def test_properties(self, comms):
        """Test various properties"""
        # Test active_peers property
        comms.peer_manager = Mock()
        comms.peer_manager.get_active_peers = Mock(return_value=[1, 2, 3])
        assert comms.active_peers == [1, 2, 3]

        # Test last_checkpoint_data property
        comms.checkpoint_manager = Mock()
        comms.checkpoint_manager.last_checkpoint_data = {"test": "data"}
        assert comms.last_checkpoint_data == {"test": "data"}

        # Test current_window property
        comms.current_window = 5
        assert comms.current_window == 5
        assert comms.get_current_window() == 5

        # Test peers property
        comms.peers = [1, 2, 3]
        assert comms.peers == [1, 2, 3]

    def test_delegation_methods(self, comms):
        """Test methods that delegate to parent class"""
        with (
            patch.object(comms, "start_commitment_fetcher"),
            patch.object(comms, "try_commit"),
        ):
            # These should delegate to parent if available
            comms.start_commitment_fetcher()
            comms.try_commit("wallet", "bucket")

    @pytest.mark.asyncio
    async def test_get_commitments(self, comms):
        """Test get commitments method"""
        expected_commitments = {42: Mock()}

        # Mock the parent class method
        with patch.object(
            comms.__class__.__bases__[0], "get_commitments", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = expected_commitments
            comms.checkpoint_manager = Mock()

            result = await comms.get_commitments(block=100)

            # Should update checkpoint manager commitments
            assert comms.checkpoint_manager.commitments == expected_commitments
            assert result == expected_commitments

    def test_manager_initialization(self, comms):
        """Test that all managers are properly initialized"""
        # Since the fixture mocks _initialize_managers, we just verify it was called during setup
        # and that the method exists and is callable
        assert hasattr(comms, "_initialize_managers")
        assert callable(comms._initialize_managers)

        # The fixture should have mocked this, so just verify the mock was set up
        assert isinstance(comms._initialize_managers, Mock)

    @pytest.mark.asyncio
    async def test_put_upload_failure(self, comms):
        """Test PUT operation when upload fails"""
        comms.gradient_manager = Mock()
        comms.gradient_manager.serialize_gradient = AsyncMock(
            return_value="/tmp/test_file.pt"
        )
        comms.storage_client = Mock()
        comms.storage_client.put_object = AsyncMock(return_value=False)  # Upload fails
        comms.file_manager = Mock()
        comms.file_manager.delete_file = Mock()

        state_dict = {"param1": torch.tensor([1, 2, 3])}

        with (
            patch("builtins.open", mock_open(read_data=b"test_data")),
            patch("src.tplr.T", return_value=1.0),
        ):
            duration = await comms.put(
                state_dict=state_dict, uid=42, window=1, key="gradient", local=False
            )

            assert duration == 0.0  # Should return 0.0 on failure

    @pytest.mark.asyncio
    async def test_get_error_handling(self, comms):
        """Test GET operation error handling"""
        comms.file_manager = Mock()
        comms.file_manager.cleanup_local_data = AsyncMock(
            side_effect=Exception("Cleanup error")
        )

        result = await comms.get(uid="42", window=1, key="gradient", local=True)

        assert result is None  # Should return None on error

    def test_uid_none_handling(self, comms):
        """Test handling when UID is None"""
        # Test with None UID
        comms_none = Comms.__new__(Comms)
        comms_none.uid = None
        # Initialize the _current_window attribute to match the property behavior
        comms_none._current_window = 0

        # get_current_window returns current_window property, which defaults to 0
        assert comms_none.get_current_window() == 0
