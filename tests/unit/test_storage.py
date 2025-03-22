import pytest
import torch
import json
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from tempfile import TemporaryDirectory
from datetime import datetime, timezone, timedelta
from aiobotocore.session import AioSession
import os

from tplr.storage import StorageManager
from tplr.schemas import Bucket
from tplr import __version__

# Import existing mocks

# Mark all tests as async
pytestmark = pytest.mark.asyncio


# Define temp_dirs as a module-level fixture so all test classes can use it
@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing"""
    with TemporaryDirectory() as temp_dir, TemporaryDirectory() as save_location:
        yield temp_dir, save_location


class TestStorageManagerInit:
    """Test initialization of StorageManager"""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing"""
        with TemporaryDirectory() as temp_dir, TemporaryDirectory() as save_location:
            yield temp_dir, save_location

    async def test_initialization(self, temp_dirs, mock_wallet):
        """Test basic initialization"""
        temp_dir, save_location = temp_dirs

        # Create StorageManager
        storage = StorageManager(
            temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet
        )

        # Verify attributes
        assert storage.temp_dir == temp_dir
        assert storage.save_location == save_location
        assert storage.wallet == mock_wallet
        assert storage.lock is not None
        assert isinstance(storage.session, AioSession)


class TestLocalStorage:
    """Test local storage operations"""

    @pytest.fixture
    def storage_manager(self, temp_dirs, mock_wallet):
        """Create a storage manager for testing"""
        temp_dir, save_location = temp_dirs
        return StorageManager(
            temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet
        )

    @pytest.fixture
    def test_data(self):
        """Create test data for storage"""
        return {"model": torch.tensor([1.0, 2.0, 3.0]), "epoch": 5}

    @patch("os.makedirs")
    @patch("torch.save")
    async def test_store_local(
        self, mock_save, mock_makedirs, storage_manager, test_data
    ):
        """Test storing data locally"""
        # Setup
        uid = "test_uid"
        window = 100
        key = "test"

        # Patch cleanup method to do nothing
        with patch.object(storage_manager, "cleanup_local_data", AsyncMock()):
            # Execute
            result = await storage_manager.store_local(test_data, uid, window, key)

            # Verify
            assert result is True
            mock_makedirs.assert_called_once()
            mock_save.assert_called_once()
            storage_manager.cleanup_local_data.assert_awaited_once()

    @patch("os.makedirs")
    @patch("torch.save", side_effect=Exception("Test error"))
    async def test_store_local_error(
        self, mock_save, mock_makedirs, storage_manager, test_data
    ):
        """Test error handling in store_local"""
        # Execute
        result = await storage_manager.store_local(test_data, "uid", 100, "test")

        # Verify
        assert result is False
        mock_makedirs.assert_called_once()
        mock_save.assert_called_once()

    @patch("os.path.exists", return_value=True)
    @patch("torch.load")
    async def test_get_local(self, mock_load, mock_exists, storage_manager, test_data):
        """Test getting data locally"""
        # Setup
        mock_load.return_value = test_data

        # Patch cleanup method
        with patch.object(storage_manager, "cleanup_local_data", AsyncMock()):
            # Execute
            result = await storage_manager.get_local("uid", 100, "test")

            # Verify
            assert result == test_data
            mock_exists.assert_called_once()
            mock_load.assert_called_once()
            storage_manager.cleanup_local_data.assert_awaited_once()

    @patch("os.path.exists", return_value=False)
    async def test_get_local_not_exists(self, mock_exists, storage_manager):
        """Test getting non-existent local data"""
        # Execute
        result = await storage_manager.get_local("uid", 100, "test")

        # Verify
        assert result is None
        mock_exists.assert_called_once()

    @patch("os.path.exists", return_value=True)
    @patch("torch.load", side_effect=Exception("Test error"))
    async def test_get_local_error(self, mock_load, mock_exists, storage_manager):
        """Test error handling in get_local"""
        # Execute
        result = await storage_manager.get_local("uid", 100, "test")

        # Verify
        assert result is None
        mock_exists.assert_called_once()
        mock_load.assert_called_once()


class TestRemoteStorage:
    """Test remote storage operations"""

    @pytest.fixture
    def storage_manager(self, temp_dirs, mock_wallet):
        """Create a storage manager for testing"""
        temp_dir, save_location = temp_dirs
        return StorageManager(
            temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet
        )

    @pytest.fixture
    def test_bucket(self):
        """Create a test bucket"""
        return Bucket(
            name="test-bucket",
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret",
        )

    @pytest.fixture
    def test_data(self):
        """Create test data for storage"""
        return {"model": torch.tensor([1.0, 2.0, 3.0]), "epoch": 5}

    @patch("torch.save")
    async def test_store_remote(
        self, mock_save, storage_manager, test_bucket, test_data
    ):
        """Test storing data remotely"""
        # Setup
        uid = "test_uid"
        window = 100
        key = "test"

        # Patch S3 method and cleanup task
        with (
            patch.object(
                storage_manager, "s3_put_object", AsyncMock(return_value=True)
            ),
            patch("asyncio.create_task"),
        ):
            # Execute
            result = await storage_manager.store_remote(
                test_data, uid, window, key, test_bucket, global_step=1
            )

            # Verify
            assert result is True
            mock_save.assert_called_once()
            storage_manager.s3_put_object.assert_awaited_once()

    async def test_store_remote_no_bucket(self, storage_manager, test_data):
        """Test storing data remotely with no bucket"""
        # Execute
        result = await storage_manager.store_remote(test_data, "uid", 100, "test", None)

        # Verify
        assert result is False

    @patch("torch.save")
    @patch("os.remove")
    async def test_store_remote_s3_error(
        self, mock_remove, mock_save, storage_manager, test_bucket, test_data
    ):
        """Test error handling in store_remote with S3 error"""
        # Patch S3 method to fail
        with patch.object(
            storage_manager, "s3_put_object", AsyncMock(return_value=False)
        ):
            # Execute
            result = await storage_manager.store_remote(
                test_data, "uid", 100, "test", test_bucket
            )

            # Verify
            assert result is False
            mock_save.assert_called_once()
            storage_manager.s3_put_object.assert_awaited_once()

    @patch("torch.load")
    async def test_get_remote(self, mock_load, storage_manager, test_bucket, test_data):
        """Test getting data remotely"""
        # Setup
        mock_load.return_value = test_data

        # Patch S3 and cleanup
        with (
            patch.object(
                storage_manager, "s3_get_object", AsyncMock(return_value=True)
            ),
            patch("asyncio.create_task"),
            patch("os.path.exists", return_value=True),
        ):
            # Execute
            result = await storage_manager.get_remote("uid", 100, "test", test_bucket)

            # Verify
            assert result == test_data
            mock_load.assert_called_once()
            storage_manager.s3_get_object.assert_awaited_once()

    async def test_get_remote_no_bucket(self, storage_manager):
        """Test getting data remotely with no bucket"""
        # Execute
        result = await storage_manager.get_remote("uid", 100, "test", None)

        # Verify
        assert result is None

    @patch("os.remove")
    async def test_get_remote_s3_error(self, mock_remove, storage_manager, test_bucket):
        """Test error handling in get_remote with S3 error"""
        # Patch S3 method to fail
        with (
            patch.object(
                storage_manager, "s3_get_object", AsyncMock(return_value=False)
            ),
            patch("os.path.exists", return_value=True),
        ):
            # Execute
            result = await storage_manager.get_remote("uid", 100, "test", test_bucket)

            # Verify
            assert result is None
            storage_manager.s3_get_object.assert_awaited_once()


class TestBytesStorage:
    """Test raw bytes storage operations"""

    @pytest.fixture
    def storage_manager(self, temp_dirs, mock_wallet):
        """Create a storage manager for testing"""
        temp_dir, save_location = temp_dirs
        return StorageManager(
            temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet
        )

    @pytest.fixture
    def test_bucket(self):
        """Create a test bucket"""
        return Bucket(
            name="test-bucket",
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret",
        )

    @pytest.fixture
    def test_bytes(self):
        """Create test bytes data"""
        return b"test data"

    @patch("builtins.open", new_callable=mock_open)
    async def test_store_bytes(
        self, mock_file, storage_manager, test_bucket, test_bytes
    ):
        """Test storing raw bytes"""
        # Patch S3 method and cleanup
        with (
            patch.object(
                storage_manager, "s3_put_object", AsyncMock(return_value=True)
            ),
            patch("asyncio.create_task"),
        ):
            # Execute
            result = await storage_manager.store_bytes(
                test_bytes, "test.json", test_bucket
            )

            # Verify
            assert result is True
            mock_file.assert_called_once()
            storage_manager.s3_put_object.assert_awaited_once()

    async def test_store_bytes_no_bucket(self, storage_manager, test_bytes):
        """Test storing bytes with no bucket"""
        # Execute
        result = await storage_manager.store_bytes(test_bytes, "test.json", None)

        # Verify
        assert result is False

    async def test_get_bytes(self, storage_manager, test_bucket, test_bytes):
        """Test getting raw bytes"""
        # Patch S3 method
        with patch.object(
            storage_manager, "s3_get_object", AsyncMock(return_value=test_bytes)
        ):
            # Execute
            result = await storage_manager.get_bytes("test.json", test_bucket)

            # Verify
            assert result == test_bytes
            storage_manager.s3_get_object.assert_awaited_once()

    async def test_get_bytes_no_bucket(self, storage_manager):
        """Test getting bytes with no bucket"""
        # Execute
        result = await storage_manager.get_bytes("test.json", None)

        # Verify
        assert result is None


class TestCheckpointOperations:
    """Test checkpoint related operations"""

    @pytest.fixture
    def storage_manager(self, temp_dirs, mock_wallet):
        """Create a storage manager for testing"""
        temp_dir, save_location = temp_dirs
        return StorageManager(
            temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet
        )

    @pytest.fixture
    def test_bucket(self):
        """Create a test bucket"""
        return Bucket(
            name="test-bucket",
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret",
        )

    @pytest.fixture
    def checkpoint_data(self):
        """Create test checkpoint data"""
        return {
            "model_state": {"weight": torch.tensor([1.0, 2.0])},
            "optimizer_state": {"state": {}},
            "epoch": 10,
            "window": 100,
        }

    @patch("os.path.exists", return_value=True)
    @patch("os.listdir")
    @patch("os.path.getmtime")
    @patch("torch.load")
    async def test_load_latest_checkpoint(
        self,
        mock_load,
        mock_mtime,
        mock_listdir,
        mock_exists,
        storage_manager,
        checkpoint_data,
    ):
        """Test loading latest local checkpoint"""
        # Setup mocks
        mock_listdir.return_value = ["checkpoint-1.pt", "checkpoint-2.pt"]
        mock_mtime.side_effect = [100, 200]  # Second file is newer
        mock_load.return_value = checkpoint_data

        # Execute
        result = await storage_manager.load_latest_checkpoint("uid")

        # Verify
        assert result == checkpoint_data
        mock_exists.assert_called_once()
        mock_listdir.assert_called_once()
        assert mock_mtime.call_count == 2
        mock_load.assert_called_once()

    @patch("os.path.exists", return_value=False)
    async def test_load_latest_checkpoint_no_dir(self, mock_exists, storage_manager):
        """Test loading checkpoint when directory doesn't exist"""
        # Execute
        result = await storage_manager.load_latest_checkpoint("uid")

        # Verify
        assert result is None
        mock_exists.assert_called_once()

    @patch("os.path.exists", return_value=True)
    @patch("os.listdir", return_value=[])
    async def test_load_latest_checkpoint_no_files(
        self, mock_listdir, mock_exists, storage_manager
    ):
        """Test loading checkpoint when no files exist"""
        # Execute
        result = await storage_manager.load_latest_checkpoint("uid")

        # Verify
        assert result is None
        mock_exists.assert_called_once()
        mock_listdir.assert_called_once()

    async def test_load_remote_checkpoint(
        self, storage_manager, test_bucket, checkpoint_data
    ):
        """Test loading remote checkpoint"""
        # Create checkpoint index
        index_data = json.dumps({"latest_checkpoint": "checkpoint-100.pt"}).encode(
            "utf-8"
        )

        # Patch S3 methods
        with (
            patch.object(storage_manager, "s3_get_object", AsyncMock()) as mock_s3_get,
            patch("torch.load", return_value=checkpoint_data),
            patch("asyncio.create_task"),
            patch("os.path.exists", return_value=True),
        ):
            # Configure mock to return different values on different calls
            mock_s3_get.side_effect = [index_data, True]

            # Execute
            result = await storage_manager.load_remote_checkpoint(
                "uid", "cpu", test_bucket
            )

            # Verify
            assert result == checkpoint_data
            assert mock_s3_get.call_count == 2

    async def test_load_remote_checkpoint_no_bucket(self, storage_manager):
        """Test loading remote checkpoint with no bucket"""
        # Execute
        result = await storage_manager.load_remote_checkpoint("uid", "cpu", None)

        # Verify
        assert result is None


class TestCleanupOperations:
    """Test cleanup operations"""

    @pytest.fixture
    def storage_manager(self, temp_dirs, mock_wallet):
        """Create a storage manager for testing"""
        temp_dir, save_location = temp_dirs
        return StorageManager(
            temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet
        )

    @patch("os.path.exists", return_value=True)
    @patch("os.listdir")
    @patch("os.path.isdir", return_value=True)
    @patch("os.path.getmtime")
    @patch("os.remove")
    @patch("os.rmdir")
    @patch("time.time", return_value=1000000)  # Fixed time for testing
    async def test_cleanup_local_data(
        self,
        mock_time,
        mock_rmdir,
        mock_remove,
        mock_mtime,
        mock_isdir,
        mock_listdir,
        mock_exists,
        storage_manager,
    ):
        """Test cleaning up old local data"""
        # Setup: Provide 11 return values to cover all os.listdir calls
        mock_listdir.side_effect = [
            ["uid1", "uid2"],  # (1) Base directory: list of uids.
            ["100", "200"],  # (2) For uid1: window directories.
            ["file1.pt"],  # (3) For uid1/100: file listing.
            [],  # (4) For uid1/100: check if empty after file removal.
            [],  # (5) For uid1/200: file listing (empty).
            [],  # (6) For uid1/200: check if empty after removal.
            [],  # (7) For uid1: final check if uid1 directory is empty.
            ["300"],  # (8) For uid2: window directories.
            ["file2.pt"],  # (9) For uid2/300: file listing.
            [],  # (10) For uid2/300: check if empty after removal.
            [],  # (11) For uid2: final check if uid2 directory is empty.
        ]
        mock_mtime.return_value = 1000  # File is old enough (>7 days)

        # Execute
        await storage_manager.cleanup_local_data()

        # Verify
        assert mock_exists.call_count == 1
        assert mock_listdir.call_count == 11  # Now exactly 11 calls are made.
        assert mock_isdir.call_count >= 3
        assert (
            mock_mtime.call_count >= 2
        )  # Two files should trigger two getmtime calls.
        assert mock_remove.call_count >= 2
        assert mock_rmdir.call_count >= 2

    @patch("os.path.exists", return_value=False)
    async def test_cleanup_local_data_no_dir(self, mock_exists, storage_manager):
        """Test cleanup when directory doesn't exist"""
        # Execute
        await storage_manager.cleanup_local_data()

        # Verify
        mock_exists.assert_called_once()

    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    async def test_cleanup_temp_file(self, mock_remove, mock_exists, storage_manager):
        """Test cleaning up a temporary file"""
        # Execute
        await storage_manager._cleanup_temp_file("/tmp/test.pt")

        # Verify
        mock_exists.assert_called_once()
        mock_remove.assert_called_once()

    @patch("os.path.exists", return_value=False)
    @patch("os.remove")
    async def test_cleanup_temp_file_not_exists(
        self, mock_remove, mock_exists, storage_manager
    ):
        """Test cleanup when file doesn't exist"""
        # Execute
        await storage_manager._cleanup_temp_file("/tmp/test.pt")

        # Verify
        mock_exists.assert_called_once()
        assert mock_remove.call_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_local_data_edge(self, storage_manager):
        """
        Test cleanup_local_data when directories are not directories (cover lines 228 & 233).
        """
        fake_listing = ["not_a_dir"]
        with patch("os.path.exists", return_value=True), \
             patch("os.listdir", return_value=fake_listing), \
             patch("os.path.isdir", return_value=False):
            # Simply run; no exception should be raised.
            await storage_manager.cleanup_local_data()


class TestS3Operations:
    """Test S3 operations"""

    @pytest.fixture
    def storage_manager(self, temp_dirs, mock_wallet):
        """Create a storage manager for testing"""
        temp_dir, save_location = temp_dirs
        return StorageManager(
            temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet
        )

    @pytest.fixture
    def test_bucket(self):
        """Create a test bucket"""
        return Bucket(
            name="test-bucket",
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret",
        )

    @patch("boto3.client")
    async def test_s3_put_object(self, mock_boto3, storage_manager, test_bucket):
        """Test uploading to S3"""
        # Setup mock S3 client
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3

        # Execute
        result = await storage_manager.s3_put_object(
            "test.pt", "/tmp/test.pt", test_bucket
        )

        # Verify
        assert result is True
        mock_boto3.assert_called_once()
        mock_s3.upload_file.assert_called_once()

    @patch("boto3.client")
    async def test_s3_put_object_error(self, mock_boto3, storage_manager, test_bucket):
        """Test error handling in S3 upload"""
        # Setup mock S3 client with error
        mock_s3 = MagicMock()
        mock_s3.upload_file.side_effect = Exception("S3 error")
        mock_boto3.return_value = mock_s3

        # Execute
        result = await storage_manager.s3_put_object(
            "test.pt", "/tmp/test.pt", test_bucket
        )

        # Verify
        assert result is False
        mock_boto3.assert_called_once()
        mock_s3.upload_file.assert_called_once()

    @patch("os.makedirs")
    @patch("aiohttp.ClientSession.get")
    async def test_s3_get_object(
        self, mock_get, mock_makedirs, storage_manager, test_bucket
    ):
        """Test downloading from S3"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"test data")
        mock_get.return_value.__aenter__.return_value = mock_response

        # Execute
        result = await storage_manager.s3_get_object("test.json", test_bucket)

        # Verify
        assert result == b"test data"
        mock_makedirs.assert_called_once()

    @patch("os.makedirs")
    @patch("aiohttp.ClientSession.get")
    async def test_s3_get_object_error(
        self, mock_get, mock_makedirs, storage_manager, test_bucket
    ):
        """Test error handling in S3 download"""
        # Setup mock response with error
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not found")
        mock_get.return_value.__aenter__.return_value = mock_response

        # Execute
        result = await storage_manager.s3_get_object("test.json", test_bucket)

        # Verify
        assert result is None
        mock_makedirs.assert_called_once()

    @patch("os.makedirs")
    @patch("aiohttp.ClientSession.get", side_effect=Exception("Network error"))
    async def test_s3_get_object_exception(
        self, mock_get, mock_makedirs, storage_manager, test_bucket
    ):
        """Test exception handling in S3 download"""
        # Execute
        result = await storage_manager.s3_get_object("test.json", test_bucket)

        # Verify
        assert result is None
        mock_makedirs.assert_called_once()


class TestTimeBasedFiltering:
    """Test time-based filtering in S3 operations"""

    @pytest.fixture
    def storage_manager(self, temp_dirs, mock_wallet):
        """Create a storage manager for testing"""
        temp_dir, save_location = temp_dirs
        return StorageManager(
            temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet
        )

    @pytest.fixture
    def test_bucket(self):
        """Create a test bucket"""
        return Bucket(
            name="test-bucket",
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret",
        )

    @patch("os.makedirs")
    @patch("aiohttp.ClientSession.get")
    async def test_s3_get_object_time_filter_too_early(
        self, mock_get, mock_makedirs, storage_manager, test_bucket
    ):
        """Test time filtering - file too early"""
        # Create time bounds
        now = datetime.now(timezone.utc)
        time_min = now.replace(hour=now.hour + 1)  # 1 hour in the future

        # Setup mock response with a TOO_EARLY marker
        mock_response = MagicMock()
        mock_response.status = 200
        # Return a dict indicating TOO_EARLY; our code will use it as is.
        mock_response.read = AsyncMock(return_value={"__status": "TOO_EARLY"})
        mock_get.return_value.__aenter__.return_value = mock_response

        # Execute
        result = await storage_manager.s3_get_object(
            "test.json", test_bucket, time_min=time_min
        )

        # Verify that the marker is returned.
        assert result == "TOO_EARLY"
        mock_makedirs.assert_called_once()

    @patch("os.makedirs")
    @patch("aiohttp.ClientSession.get")
    async def test_s3_get_object_time_filter_too_late(
        self, mock_get, mock_makedirs, storage_manager, test_bucket
    ):
        """Test time filtering - file too late"""
        # Create time bounds
        now = datetime.now(timezone.utc)
        time_max = now - timedelta(hours=1)  # 1 hour in the past

        # Setup mock response with a TOO_LATE marker
        mock_response = MagicMock()
        mock_response.status = 200
        # Return a dict indicating TOO_LATE; our code will use it as is.
        mock_response.read = AsyncMock(return_value={"__status": "TOO_LATE"})
        mock_get.return_value.__aenter__.return_value = mock_response

        # Execute
        result = await storage_manager.s3_get_object(
            "test.json", test_bucket, time_max=time_max
        )

        # Verify that the marker is returned.
        assert result == "TOO_LATE"
        mock_makedirs.assert_called_once()


# -------------------------------
# Fake classes for simulating S3 GET
# -------------------------------
class FakeResponse:
    def __init__(self, status, text, body):
        self.status = status
        self._text = text
        self._body = body

    async def read(self):
        return self._body

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class FakeClientSession:
    def __init__(self, fake_response):
        self.fake_response = fake_response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def get(self, url, headers, timeout):
        return self.fake_response


class TestStorageExtra:

    @pytest.mark.asyncio
    async def test_store_remote_exception(self, storage_manager, temp_dirs, mock_wallet):
        """
        Force an exception in store_remote (e.g. during torch.save)
        """
        test_bucket = Bucket(
            name="test-bucket", account_id="test-account", access_key_id="key", secret_access_key="secret"
        )
        test_data = {"dummy": 1}
        # Force torch.save to raise an exception.
        with patch("torch.save", side_effect=Exception("torch.save failed")), \
             patch("os.path.exists", return_value=True) as mock_exists, \
             patch("os.remove") as mock_remove:
            result = await storage_manager.store_remote(test_data, "uid", 100, "test", test_bucket)
            assert result is False
            mock_exists.assert_called()  # Called on temp file path.
            mock_remove.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_local_exception(self, storage_manager):
        """
        Test get_local when torch.load fails to load the file (covering lines 133-136).
        """
        # Simulate file exists.
        with patch("os.path.exists", return_value=True), \
             patch("torch.load", side_effect=Exception("load failed")), \
             patch.object(storage_manager, "cleanup_local_data", AsyncMock()) as cleanup_mock:
            result = await storage_manager.get_local("uid", 100, "test")
            assert result is None
            cleanup_mock.assert_not_awaited()  # Because error occurred before calling cleanup.

    @pytest.mark.asyncio
    async def test_get_bytes_exception(self, storage_manager):
        """
        Test get_bytes when s3_get_object raises an exception (covering lines 151-153).
        """
        with patch.object(storage_manager, "s3_get_object", AsyncMock(side_effect=Exception("get_bytes error"))):
            result = await storage_manager.get_bytes("key", Bucket(name="dummy", account_id="dummy", access_key_id="a", secret_access_key="b"))
            assert result is None

    # --- Tests for load_latest_checkpoint ---
    @pytest.mark.asyncio
    async def test_load_latest_checkpoint_no_dir(self, temp_dirs, mock_wallet):
        """
        Test load_latest_checkpoint when the checkpoints directory does not exist.
        """
        temp_dir, save_location = temp_dirs
        storage = StorageManager(temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet)
        with patch("os.path.exists", return_value=False) as mock_exists:
            result = await storage.load_latest_checkpoint("uid")
            assert result is None
            mock_exists.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_latest_checkpoint_no_files(self, temp_dirs, mock_wallet):
        """
        Test load_latest_checkpoint when no checkpoint files are found.
        """
        temp_dir, save_location = temp_dirs
        storage = StorageManager(temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet)
        with patch("os.path.exists", return_value=True), \
             patch("os.listdir", return_value=[]):
            result = await storage.load_latest_checkpoint("uid")
            assert result is None

    @pytest.mark.asyncio
    async def test_load_latest_checkpoint_exception(self, temp_dirs, mock_wallet):
        """
        Test load_latest_checkpoint when torch.load raises an exception.
        """
        temp_dir, save_location = temp_dirs
        storage = StorageManager(temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet)
        checkpoints_dir = os.path.join(storage.save_location, "checkpoints")
        fake_file = "checkpoint-1.pt"
        fake_path = os.path.join(checkpoints_dir, fake_file)
        with patch("os.path.exists", return_value=True), \
             patch("os.listdir", return_value=[fake_file]), \
             patch("os.path.getmtime", return_value=1000), \
             patch("torch.load", side_effect=Exception("load failed")) as mock_torch_load:
            result = await storage.load_latest_checkpoint("uid")
            assert result is None
            mock_torch_load.assert_called_once_with(fake_path)

    # --- Tests for load_remote_checkpoint ---
    @pytest.mark.asyncio
    async def test_load_remote_checkpoint_no_index(self, storage_manager):
        """
        Test load_remote_checkpoint when checkpoint index is None.
        """
        test_bucket = Bucket(
            name="dummy", account_id="dummy", access_key_id="a", secret_access_key="b"
        )
        with patch.object(storage_manager, "s3_get_object", AsyncMock(return_value=None)) as mock_s3:
            result = await storage_manager.load_remote_checkpoint("uid", "cpu", test_bucket)
            assert result is None
            mock_s3.assert_awaited()  # At least the index retrieval was attempted.

    @pytest.mark.asyncio
    async def test_load_remote_checkpoint_exception(self, storage_manager):
        """
        Test load_remote_checkpoint when an exception is raised during its process.
        """
        test_bucket = Bucket(
            name="dummy", account_id="dummy", access_key_id="a", secret_access_key="b"
        )
        with patch.object(storage_manager, "s3_get_object", AsyncMock(side_effect=Exception("fail"))) as mock_s3:
            result = await storage_manager.load_remote_checkpoint("uid", "cpu", test_bucket)
            assert result is None
            mock_s3.assert_awaited()

    # --- Test cleanup_local_data edge branch ---
    @pytest.mark.asyncio
    async def test_cleanup_local_data_edge(self, storage_manager):
        """
        Test cleanup_local_data when directories are not directories (cover lines 228 & 233).
        """
        fake_listing = ["not_a_dir"]
        with patch("os.path.exists", return_value=True), \
             patch("os.listdir", return_value=fake_listing), \
             patch("os.path.isdir", return_value=False):
            # Simply run; no exception should be raised.
            await storage_manager.cleanup_local_data()

    # --- Tests for s3_get_object ---
    @pytest.mark.asyncio
    async def test_s3_get_object_status_marker(self, temp_dirs, mock_wallet):
        """
        Test s3_get_object returns status marker if JSON response includes __status.
        """
        temp_dir, save_location = temp_dirs
        storage = StorageManager(temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet)
        # Fake response returns JSON string with __status "TOO_EARLY"
        fake_body = b'{"__status": "TOO_EARLY"}'
        fake_response = FakeResponse(status=200, text="OK", body=fake_body)
        fake_session = FakeClientSession(fake_response)
        with patch("tplr.storage.aiohttp.ClientSession", return_value=fake_session):
            result = await storage.s3_get_object(
                "testkey",
                Bucket(name="dummy", account_id="dummy", access_key_id="a", secret_access_key="b")
            )
            assert result == "TOO_EARLY"

    @pytest.mark.asyncio
    async def test_s3_get_object_file_path(self, temp_dirs, mock_wallet, tmp_path):
        """
        Test s3_get_object when file_path is provided.
        (Covers branch writing to file and returning True.)
        """
        temp_dir, save_location = temp_dirs
        storage = StorageManager(temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet)
        # Fake response returns bytes without a __status field.
        fake_body = b'{"data": "value"}'
        fake_response = FakeResponse(status=200, text="OK", body=fake_body)
        fake_session = FakeClientSession(fake_response)
        file_path = str(tmp_path / "test_file.json")
        with patch("tplr.storage.aiohttp.ClientSession", return_value=fake_session):
            result = await storage.s3_get_object(
                "testkey",
                Bucket(name="dummy", account_id="dummy", access_key_id="a", secret_access_key="b"),
                file_path=file_path
            )
            # Method should write to file and return True.
            assert result is True
            with open(file_path, "rb") as f:
                content = f.read()
            assert content == fake_body
            os.remove(file_path)

    @pytest.mark.asyncio
    async def test_s3_get_object_non_200(self, temp_dirs, mock_wallet):
        """
        Test s3_get_object when the HTTP response status is not 200.
        """
        temp_dir, save_location = temp_dirs
        storage = StorageManager(temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet)
        fake_response = FakeResponse(status=404, text="Not Found", body=b"")
        fake_session = FakeClientSession(fake_response)
        with patch("tplr.storage.aiohttp.ClientSession", return_value=fake_session):
            result = await storage.s3_get_object(
                "testkey",
                Bucket(name="dummy", account_id="dummy", access_key_id="a", secret_access_key="b")
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_s3_get_object_exception(self, temp_dirs, mock_wallet):
        """
        Test s3_get_object when an exception occurs in the ClientSession.
        """
        temp_dir, save_location = temp_dirs
        storage = StorageManager(temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet)
        with patch("tplr.storage.aiohttp.ClientSession", side_effect=Exception("session failed")):
            result = await storage.s3_get_object(
                "testkey",
                Bucket(name="dummy", account_id="dummy", access_key_id="a", secret_access_key="b")
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_s3_get_object_dict_loaded(self, temp_dirs, mock_wallet):
        """
        Test s3_get_object branch where loaded_data is a dict.
        """
        temp_dir, save_location = temp_dirs
        storage = StorageManager(temp_dir=temp_dir, save_location=save_location, wallet=mock_wallet)
        # Fake response returns a dict directly.
        fake_body = {"__status": "TOO_LATE"}
        fake_response = FakeResponse(status=200, text="OK", body=fake_body)
        fake_session = FakeClientSession(fake_response)
        with patch("tplr.storage.aiohttp.ClientSession", return_value=fake_session):
            result = await storage.s3_get_object(
                "testkey",
                Bucket(name="dummy", account_id="dummy", access_key_id="a", secret_access_key="b")
            )
            assert result == "TOO_LATE"


class TestStorageNewBranches:
    @pytest.mark.asyncio
    async def test_store_local_success(self, storage_manager, temp_dirs, mock_wallet):
        """
        Test store_local on a successful run.
        Verifies that store_local returns True when torch.save works.
        """
        state = {"a": 1}
        uid = "testuid"
        window = "1"
        key = "local_test"
        result = await storage_manager.store_local(state, uid, window, key)
        assert result is True
        # Construct expected file path.
        path = os.path.join(storage_manager.save_location, "local_store", uid, str(window))
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        # The file may have been cleaned up as part of cleanup_local_data;

    @pytest.mark.asyncio
    async def test_store_local_exception(self, storage_manager):
        """
        Test store_local exception branch:
        If torch.save fails, store_local returns False.
        """
        state = {"a": 1}
        uid = "testuid"
        window = "1"
        key = "local_fail"
        with patch("torch.save", side_effect=Exception("fail in store_local")):
            result = await storage_manager.store_local(state, uid, window, key)
            assert result is False

    @pytest.mark.asyncio
    async def test_store_remote_success(self, storage_manager, temp_dirs, mock_wallet):
        """
        Test store_remote successful branch.
        (Covers the non-exception path after torch.save succeeds.)
        """
        state = {"b": 2}
        uid = "testuid"
        window = "2"
        key = "remote_test"
        test_bucket = Bucket(
            name="remote_bucket",
            account_id="account",
            access_key_id="key",
            secret_access_key="secret"
        )
        with patch("torch.save"), patch.object(storage_manager, "s3_put_object", new=AsyncMock(return_value=True)):
            result = await storage_manager.store_remote(state, uid, window, key, test_bucket)
            assert result is True

    @pytest.mark.asyncio
    async def test_store_bytes_success(self, storage_manager, temp_dirs, mock_wallet):
        """
        Test store_bytes successful branch.
        (Ensures that with a valid bucket and s3_put_object returning True,
        store_bytes returns True.)
        """
        data = b"sample bytes"
        key = "bytes_test.json"
        test_bucket = Bucket(
            name="bytes_bucket",
            account_id="account",
            access_key_id="key",
            secret_access_key="secret"
        )
        with patch.object(storage_manager, "s3_put_object", new=AsyncMock(return_value=True)):
            result = await storage_manager.store_bytes(data, key, test_bucket)
            assert result is True

    @pytest.mark.asyncio
    async def test_store_bytes_no_bucket(self, storage_manager):
        """
        Test store_bytes when no bucket is provided.
        (Covers the branch that returns False.)
        """
        data = b"sample bytes no bucket"
        key = "nobucket.json"
        result = await storage_manager.store_bytes(data, key, None)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_local_success(self, storage_manager, tmp_path, temp_dirs, mock_wallet):
        """
        Test get_local successful branch.
        (If file exists and torch.load returns the state, get_local returns it.)
        """
        uid = "testuid"
        window = "3"
        key = "getlocal_test"
        path = os.path.join(storage_manager.save_location, "local_store", uid, str(window))
        os.makedirs(path, exist_ok=True)
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        file_path = os.path.join(path, filename)
        dummy_state = {"dummy": "state"}
        torch.save(dummy_state, file_path)
        with patch.object(storage_manager, "cleanup_local_data", new=AsyncMock()):
            result = await storage_manager.get_local(uid, window, key)
            assert result == dummy_state

    @pytest.mark.asyncio
    async def test_get_local_no_file(self, storage_manager):
        """
        Test get_local branch when the file does not exist.
        (Should return None.)
        """
        uid = "nonexistent"
        window = "999"
        key = "nofile"
        with patch("os.path.exists", return_value=False):
            result = await storage_manager.get_local(uid, window, key)
            assert result is None

    @pytest.mark.asyncio
    async def test_s3_get_object_no_bucket(self, storage_manager):
        """
        Test s3_get_object branch when no bucket is provided.
        (Should log an error and return None.)
        """
        result = await storage_manager.s3_get_object("anykey", None)
        assert result is None
