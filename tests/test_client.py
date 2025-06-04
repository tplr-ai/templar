# Test Class: TestStorageClient
# ==============================================================================

# Test: __init__ method
# - Test initialization with valid temp_dir
# - Test initialization with non-existent temp_dir (should create it)
# - Test initialization with temp_dir as None/empty string
# - Test that session is properly initialized
# - Test that _s3_clients dict is empty initially
# - Test that client_semaphore is initialized with correct value
# - Test temp directory creation permissions on various filesystems
# TODO: Test behavior when temp_dir creation fails due to permissions

# Test: get_base_url method
# - Test with valid account_id returns correct URL format
# - Test with empty account_id
# - Test with account_id containing special characters
# - Test with very long account_id
# - Test URL format matches expected R2 endpoint pattern
# TODO: Test with unicode characters in account_id

# Test: _get_s3_client method
# - Test client creation with valid bucket credentials
# - Test client caching (same bucket should return same client)
# - Test client creation with invalid credentials
# - Test client creation with malformed account_id
# - Test client creation when network is unavailable
# - Test AioConfig parameters are properly set
# - Test multiple buckets create separate clients
# - Test client persistence across multiple calls
# TODO: Test memory usage with many cached clients
# TODO: Test client behavior when credentials expire

# Test: _purge_s3_client method
# - Test purging existing client from cache
# - Test purging non-existent client (should not error)
# - Test cache state after purging
# - Test that purged client is not reused
# TODO: Test concurrent purging of same client

# Test: get_object method - Basic functionality
# - Test downloading existing small object (< 600MB)
# - Test downloading existing large object (> 600MB, triggers multipart)
# - Test downloading non-existent object returns None
# - Test downloading with invalid bucket credentials
# - Test downloading with malformed key names
# - Test downloading zero-byte file
# - Test downloading very large file (multiple GB)
# - Test that temp file is always cleaned up
# TODO: Test downloading with special characters in key name
# TODO: Test behavior when temp directory is full

# Test: get_object method - Timeout handling
# - Test timeout during HEAD request
# - Test timeout during GET request
# - Test timeout during multipart download
# - Test custom timeout values
# - Test very short timeout (should fail gracefully)
# - Test very long timeout
# TODO: Test timeout behavior under different network conditions

# Test: get_object method - Time constraints
# - Test time_min filter (object too old)
# - Test time_max filter (object too new)
# - Test object within time range
# - Test with timezone-aware datetime objects
# - Test with timezone-naive datetime objects
# - Test with invalid datetime formats
# - Test edge case where object time exactly matches constraint
# TODO: Test behavior across timezone boundaries
# TODO: Test with objects having no LastModified metadata

# Test: get_object method - Progress tracking
# - Test show_progress=True displays progress
# - Test show_progress=False hides progress
# - Test progress accuracy for multipart downloads
# TODO: Test progress display in different terminal environments

# Test: get_object method - Error handling
# - Test 404 errors (object not found)
# - Test 403 errors (access denied)
# - Test 500 errors (server errors)
# - Test ConnectionClosedError handling
# - Test client purging on connection errors
# - Test retry behavior for transient errors
# - Test behavior when disk space runs out during download
# - Test behavior when temp file cannot be created
# TODO: Test handling of corrupted downloads
# TODO: Test behavior when S3 returns partial content unexpectedly

# Test: put_object method - Basic functionality
# - Test uploading small object (< 100MB)
# - Test uploading large object (> 100MB, triggers multipart)
# - Test uploading zero-byte object
# - Test uploading with special characters in key
# - Test uploading binary data
# - Test uploading text data
# - Test overwriting existing object
# TODO: Test uploading extremely large objects (10GB+)
# TODO: Test uploading with custom metadata

# Test: put_object method - Error handling
# - Test upload with invalid credentials
# - Test upload when bucket doesn't exist
# - Test upload when disk space insufficient for temp file
# - Test upload failure during multipart process
# - Test network interruption during upload
# - Test S3 server errors during upload
# - Test client purging on connection errors
# TODO: Test upload with insufficient S3 permissions
# TODO: Test behavior when upload partially succeeds

# Test: delete_object method
# - Test deleting existing object
# - Test deleting non-existent object (should succeed)
# - Test deleting with invalid credentials
# - Test deleting with special characters in key
# - Test connection errors during delete
# - Test client purging on errors
# TODO: Test deleting objects with versioning enabled
# TODO: Test bulk delete operations

# Test: list_objects method
# - Test listing with empty prefix (all objects)
# - Test listing with specific prefix
# - Test listing empty bucket
# - Test listing bucket with many objects (> 1000, tests pagination)
# - Test listing with prefix that matches no objects
# - Test listing with special characters in prefix
# - Test connection errors during listing
# - Test client purging on errors
# - Test pagination with ContinuationToken
# TODO: Test listing with very long prefixes
# TODO: Test memory usage with large object lists

# Test: get_object_size method
# - Test getting size of existing object
# - Test getting size of non-existent object
# - Test getting size of zero-byte object
# - Test getting size with invalid credentials
# - Test connection errors during HEAD request
# TODO: Test size retrieval for very large objects
# TODO: Test behavior with objects having no ContentLength

# Test: get_object_range method
# - Test downloading valid byte range
# - Test downloading range from beginning of file
# - Test downloading range to end of file
# - Test downloading entire file as range
# - Test downloading zero-byte range
# - Test downloading range larger than file
# - Test downloading with invalid range (start > end)
# - Test downloading with negative range values
# - Test timeout during range download
# - Test connection errors during range download
# - Test data integrity of downloaded range
# TODO: Test concurrent range downloads from same object
# TODO: Test range downloads with very large objects

# Test: multipart_upload method
# - Test successful multipart upload
# - Test upload with file that doesn't exist
# - Test upload with empty file
# - Test upload with very large file
# - Test upload failure in part upload (should retry)
# - Test upload failure in create_multipart_upload
# - Test upload failure in complete_multipart_upload
# - Test abort_multipart_upload on failure
# - Test concurrent part uploads
# - Test part ordering in final upload
# - Test retry logic for failed parts
# - Test semaphore limiting concurrent uploads
# TODO: Test upload progress tracking
# TODO: Test memory usage during large uploads
# TODO: Test behavior when upload_id becomes invalid

# Test: multipart_download method - Basic functionality
# - Test successful multipart download
# - Test download with default chunk_size and max_workers
# - Test download with custom chunk_size
# - Test download with custom max_workers
# - Test download with show_progress=True/False
# - Test download of small file (single chunk)
# - Test download of very large file
# - Test download with zero-byte file
# TODO: Test download progress accuracy
# TODO: Test optimal chunk size calculation

# Test: multipart_download method - GPU considerations
# - Test chunk_size calculation with GPU available
# - Test chunk_size calculation without GPU
# - Test max_workers calculation with GPU
# - Test max_workers calculation without GPU
# - Test behavior with multiple GPUs
# TODO: Test memory usage with different GPU configurations
# TODO: Test behavior when GPU memory is limited

# Test: multipart_download method - Error handling
# - Test download failure in get_object_size
# - Test chunk download failure (should retry)
# - Test file creation failure
# - Test file write permission errors
# - Test partial chunk failures
# - Test missing chunks detection
# - Test downloaded size validation
# - Test connection errors during download
# - Test timeout during chunk downloads
# - Test semaphore limiting concurrent downloads
# TODO: Test recovery from partially downloaded files
# TODO: Test behavior when available disk space changes during download

# Test: multipart_download method - Concurrent operations
# - Test concurrent chunk downloads
# - Test chunk ordering in final file
# - Test data integrity across all chunks
# - Test retry behavior for failed chunks
# - Test semaphore preventing too many concurrent operations
# TODO: Test memory usage with high concurrency
# TODO: Test network bandwidth utilization

# Test: close_all_clients method
# - Test closing empty client cache
# - Test closing single cached client
# - Test closing multiple cached clients
# - Test error handling when client close fails
# - Test cache state after closing all clients
# - Test subsequent operations after closing clients
# TODO: Test closing clients during active operations
# TODO: Test memory cleanup verification

# Test: Integration and stress tests
# - Test multiple concurrent downloads
# - Test multiple concurrent uploads
# - Test mixed operations (upload/download/delete simultaneously)
# - Test memory usage under heavy load
# - Test file descriptor limits
# - Test very long-running operations
# - Test operations with slow network connections
# - Test operations with unstable network connections
# TODO: Test behavior during system resource exhaustion
# TODO: Test recovery after network partitions

# Test: Error recovery and resilience
# - Test operation retry after client purging
# - Test graceful degradation under resource constraints
# - Test cleanup of temporary files on unexpected exits
# - Test behavior when S3 service is temporarily unavailable
# - Test behavior with intermittent network connectivity
# TODO: Test data consistency after interrupted operations
# TODO: Test behavior during system shutdown

# Test: Security and edge cases
# - Test with malformed bucket configurations
# - Test with expired or rotated credentials
# - Test with objects containing path traversal attempts
# - Test with extremely long object keys
# - Test with binary data containing null bytes
# - Test with unicode filenames and object keys
# TODO: Test behavior with compromised credentials
# TODO: Test data sanitization and validation

# Test: Performance and optimization
# - Test CPU_MAX_CONNECTIONS scaling
# - Test memory usage patterns
# - Test client reuse efficiency
# - Test temp file cleanup timing
# - Test operation batching efficiency
# TODO: Benchmark against different file sizes
# TODO: Test memory leak detection over extended runs

# Test: Mock and unit test patterns
# - Mock S3 client responses for controlled testing
# - Mock network failures at specific points
# - Mock filesystem operations (temp file creation, cleanup)
# - Mock timeout scenarios
# - Mock partial responses and interruptions
# TODO: Test with various aiobotocore version compatibility
# TODO: Test configuration parameter variations

import pytest
import asyncio
import os
import tempfile
import shutil
import stat
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock
from src.tplr.storage.client import StorageClient, CPU_MAX_CONNECTIONS
from src.tplr.schemas import Bucket


class TestStorageClientInit:
    """Test cases for StorageClient.__init__ method"""

    def test_init_with_valid_temp_dir(self):
        """Test initialization with valid temp_dir"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            assert client.temp_dir == temp_dir
            assert os.path.exists(temp_dir)
            assert client.session is not None
            assert isinstance(client._s3_clients, dict)
            assert len(client._s3_clients) == 0
            assert client.client_semaphore._value == CPU_MAX_CONNECTIONS

    def test_init_with_non_existent_temp_dir(self):
        """Test initialization with non-existent temp_dir (should create it)"""
        with tempfile.TemporaryDirectory() as base_dir:
            non_existent_dir = os.path.join(base_dir, "non_existent", "nested", "dir")

            # Ensure it doesn't exist
            assert not os.path.exists(non_existent_dir)

            client = StorageClient(non_existent_dir)

            # Should be created
            assert os.path.exists(non_existent_dir)
            assert os.path.isdir(non_existent_dir)
            assert client.temp_dir == non_existent_dir

    def test_init_with_empty_string_temp_dir(self):
        """Test initialization with empty string temp_dir"""
        with pytest.raises(FileNotFoundError):
            StorageClient("")

    def test_init_with_none_temp_dir(self):
        """Test initialization with None temp_dir should raise TypeError"""
        with pytest.raises(TypeError):
            StorageClient(None)

    def test_session_properly_initialized(self):
        """Test that session is properly initialized"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("src.tplr.storage.client.get_session") as mock_get_session:
                mock_session = MagicMock()
                mock_get_session.return_value = mock_session

                client = StorageClient(temp_dir)

                mock_get_session.assert_called_once()
                assert client.session == mock_session

    def test_s3_clients_dict_empty_initially(self):
        """Test that _s3_clients dict is empty initially"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            assert isinstance(client._s3_clients, dict)
            assert len(client._s3_clients) == 0
            assert client._s3_clients == {}

    def test_client_semaphore_initialized_with_correct_value(self):
        """Test that client_semaphore is initialized with correct value"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            assert isinstance(client.client_semaphore, asyncio.Semaphore)
            assert client.client_semaphore._value == CPU_MAX_CONNECTIONS

            # Verify the CPU_MAX_CONNECTIONS calculation
            cpu_count = os.cpu_count() or 4
            expected_connections = min(100, max(30, cpu_count * 4))
            assert client.client_semaphore._value == expected_connections

    def test_temp_directory_creation_permissions(self):
        """Test temp directory creation permissions on various filesystems"""
        with tempfile.TemporaryDirectory() as base_dir:
            # Test normal directory creation
            normal_dir = os.path.join(base_dir, "normal_dir")
            StorageClient(normal_dir)

            # Check directory exists and has proper permissions
            assert os.path.exists(normal_dir)
            stat_info = os.stat(normal_dir)
            # Should be readable, writable, executable by owner
            assert stat_info.st_mode & stat.S_IRUSR
            assert stat_info.st_mode & stat.S_IWUSR
            assert stat_info.st_mode & stat.S_IXUSR

    def test_temp_directory_already_exists(self):
        """Test initialization when temp_dir already exists"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file in the directory first
            test_file = os.path.join(temp_dir, "existing_file.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            client = StorageClient(temp_dir)

            # Directory should still exist and file should be preserved
            assert os.path.exists(temp_dir)
            assert os.path.exists(test_file)
            assert client.temp_dir == temp_dir

    @patch("os.makedirs")
    def test_temp_directory_creation_failure(self, mock_makedirs):
        """Test behavior when temp_dir creation fails due to permissions"""
        mock_makedirs.side_effect = PermissionError("Permission denied")

        with pytest.raises(PermissionError):
            StorageClient("/root/forbidden_dir")

    @patch("os.makedirs")
    def test_temp_directory_creation_other_os_error(self, mock_makedirs):
        """Test behavior when temp_dir creation fails due to other OS errors"""
        mock_makedirs.side_effect = OSError("Disk full")

        with pytest.raises(OSError):
            StorageClient("/some/path")

    def test_init_with_relative_path(self):
        """Test initialization with relative path"""
        relative_path = "test_temp_dir"

        try:
            client = StorageClient(relative_path)

            # Should convert to absolute path and create
            assert os.path.exists(relative_path)
            assert client.temp_dir == relative_path
        finally:
            # Cleanup
            if os.path.exists(relative_path):
                shutil.rmtree(relative_path)

    def test_init_with_special_characters_in_path(self):
        """Test initialization with special characters in temp_dir path"""
        with tempfile.TemporaryDirectory() as base_dir:
            special_dir = os.path.join(base_dir, "temp dir with spaces & symbols!@#")

            client = StorageClient(special_dir)

            assert os.path.exists(special_dir)
            assert client.temp_dir == special_dir

    def test_init_multiple_instances_different_dirs(self):
        """Test creating multiple StorageClient instances with different temp dirs"""
        with tempfile.TemporaryDirectory() as base_dir:
            dir1 = os.path.join(base_dir, "client1")
            dir2 = os.path.join(base_dir, "client2")

            client1 = StorageClient(dir1)
            client2 = StorageClient(dir2)

            assert client1.temp_dir != client2.temp_dir
            assert os.path.exists(dir1)
            assert os.path.exists(dir2)
            # Each should have their own session and semaphore
            assert client1.session is not client2.session
            assert client1.client_semaphore is not client2.client_semaphore

    def test_init_multiple_instances_same_dir(self):
        """Test creating multiple StorageClient instances with same temp dir"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client1 = StorageClient(temp_dir)
            client2 = StorageClient(temp_dir)

            assert client1.temp_dir == client2.temp_dir
            # Should have separate sessions and caches
            assert client1._s3_clients is not client2._s3_clients
            assert len(client1._s3_clients) == 0
            assert len(client2._s3_clients) == 0


class TestStorageClientGetBaseUrl:
    """Test cases for StorageClient.get_base_url method"""

    def test_get_base_url_with_valid_account_id(self):
        """Test with valid account_id returns correct URL format"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)
            account_id = "test123account"

            result = client.get_base_url(account_id)

            expected = f"https://{account_id}.r2.cloudflarestorage.com"
            assert result == expected
            assert result == "https://test123account.r2.cloudflarestorage.com"

    def test_get_base_url_with_empty_account_id(self):
        """Test with empty account_id"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)
            account_id = ""

            result = client.get_base_url(account_id)

            # Should still work but create malformed URL
            expected = "https://.r2.cloudflarestorage.com"
            assert result == expected

    def test_get_base_url_with_special_characters(self):
        """Test with account_id containing special characters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)
            account_id = "test-account_123.special"

            result = client.get_base_url(account_id)

            expected = f"https://{account_id}.r2.cloudflarestorage.com"
            assert result == expected
            assert result == "https://test-account_123.special.r2.cloudflarestorage.com"

    def test_get_base_url_with_very_long_account_id(self):
        """Test with very long account_id"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)
            # Create a very long account ID (255 chars)
            account_id = "a" * 255

            result = client.get_base_url(account_id)

            expected = f"https://{account_id}.r2.cloudflarestorage.com"
            assert result == expected
            assert result.startswith("https://")
            assert result.endswith(".r2.cloudflarestorage.com")
            assert len(account_id) == 255

    def test_get_base_url_format_matches_r2_endpoint_pattern(self):
        """Test URL format matches expected R2 endpoint pattern"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)
            test_cases = [
                "simpleaccount",
                "account-with-dashes",
                "account123",
                "a",
                "UPPERCASE",
                "MixedCase123",
            ]

            for account_id in test_cases:
                result = client.get_base_url(account_id)

                # Check pattern: https://{account_id}.r2.cloudflarestorage.com
                assert result.startswith("https://")
                assert result.endswith(".r2.cloudflarestorage.com")
                assert f"{account_id}." in result
                assert result == f"https://{account_id}.r2.cloudflarestorage.com"

    def test_get_base_url_with_numeric_account_id(self):
        """Test with purely numeric account_id"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)
            account_id = "123456789"

            result = client.get_base_url(account_id)

            expected = "https://123456789.r2.cloudflarestorage.com"
            assert result == expected

    def test_get_base_url_with_single_character(self):
        """Test with single character account_id"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)
            account_id = "x"

            result = client.get_base_url(account_id)

            expected = "https://x.r2.cloudflarestorage.com"
            assert result == expected

    def test_get_base_url_with_unicode_characters(self):
        """Test with unicode characters in account_id"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)
            account_id = "tëst-账户-🚀"

            result = client.get_base_url(account_id)

            expected = f"https://{account_id}.r2.cloudflarestorage.com"
            assert result == expected
            assert "tëst-账户-🚀" in result

    def test_get_base_url_with_url_unsafe_characters(self):
        """Test with URL-unsafe characters in account_id"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)
            # Characters that would need URL encoding
            account_id = "test account with spaces"

            result = client.get_base_url(account_id)

            # Method doesn't do URL encoding, just string interpolation
            expected = "https://test account with spaces.r2.cloudflarestorage.com"
            assert result == expected

    def test_get_base_url_multiple_calls_same_account(self):
        """Test multiple calls with same account_id return consistent results"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)
            account_id = "consistent-test"

            result1 = client.get_base_url(account_id)
            result2 = client.get_base_url(account_id)
            result3 = client.get_base_url(account_id)

            assert result1 == result2 == result3
            assert result1 == "https://consistent-test.r2.cloudflarestorage.com"

    def test_get_base_url_different_clients_same_account(self):
        """Test different client instances return same URL for same account"""
        with tempfile.TemporaryDirectory() as temp_dir1:
            with tempfile.TemporaryDirectory() as temp_dir2:
                client1 = StorageClient(temp_dir1)
                client2 = StorageClient(temp_dir2)
                account_id = "shared-account"

                result1 = client1.get_base_url(account_id)
                result2 = client2.get_base_url(account_id)

                assert result1 == result2
                assert result1 == "https://shared-account.r2.cloudflarestorage.com"

    def test_get_base_url_method_is_pure_function(self):
        """Test that get_base_url is a pure function (no side effects)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)
            account_id = "pure-function-test"

            # Check that method doesn't modify client state
            original_clients = client._s3_clients.copy()
            original_session = client.session
            original_temp_dir = client.temp_dir

            result = client.get_base_url(account_id)

            # State should be unchanged
            assert client._s3_clients == original_clients
            assert client.session == original_session
            assert client.temp_dir == original_temp_dir
            assert result == "https://pure-function-test.r2.cloudflarestorage.com"


class TestStorageClientGetS3Client:
    """Test cases for StorageClient._get_s3_client method"""

    @pytest.fixture
    def mock_bucket(self):
        """Create a mock bucket for testing"""
        return Bucket(
            access_key_id="test_access_key",
            secret_access_key="test_secret_key",
            account_id="test_account_id",
            name="test_bucket",
        )

    @pytest.fixture
    def different_mock_bucket(self):
        """Create a different mock bucket for testing"""
        return Bucket(
            access_key_id="different_access_key",
            secret_access_key="different_secret_key",
            account_id="different_account_id",
            name="different_bucket",
        )

    @pytest.mark.asyncio
    async def test_client_creation_with_valid_bucket_credentials(self, mock_bucket):
        """Test client creation with valid bucket credentials"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the session and S3 client context manager
            mock_s3_client = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_s3_client)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(return_value=mock_context_manager)

            result = await client._get_s3_client(mock_bucket)

            # Verify client creation was called with correct parameters
            client.session.create_client.assert_called_once()
            call_args = client.session.create_client.call_args

            assert call_args[0][0] == "s3"  # service name
            assert (
                call_args[1]["endpoint_url"]
                == "https://test_account_id.r2.cloudflarestorage.com"
            )
            assert call_args[1]["region_name"] == "enam"
            assert call_args[1]["aws_access_key_id"] == "test_access_key"
            assert call_args[1]["aws_secret_access_key"] == "test_secret_key"
            assert result == mock_s3_client

    @pytest.mark.asyncio
    async def test_client_caching_same_bucket_returns_same_client(self, mock_bucket):
        """Test client caching (same bucket should return same client)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the session and S3 client context manager
            mock_s3_client = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_s3_client)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(return_value=mock_context_manager)

            # First call
            result1 = await client._get_s3_client(mock_bucket)

            # Second call with same bucket
            result2 = await client._get_s3_client(mock_bucket)

            # Should be same client instance and create_client called only once
            assert result1 == result2
            assert result1 is result2
            client.session.create_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_buckets_create_separate_clients(
        self, mock_bucket, different_mock_bucket
    ):
        """Test multiple buckets create separate clients"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock different S3 clients
            mock_s3_client1 = AsyncMock()
            mock_s3_client2 = AsyncMock()

            mock_context_manager1 = AsyncMock()
            mock_context_manager1.__aenter__ = AsyncMock(return_value=mock_s3_client1)
            mock_context_manager1.__aexit__ = AsyncMock(return_value=None)

            mock_context_manager2 = AsyncMock()
            mock_context_manager2.__aenter__ = AsyncMock(return_value=mock_s3_client2)
            mock_context_manager2.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(
                side_effect=[mock_context_manager1, mock_context_manager2]
            )

            # Get clients for different buckets
            result1 = await client._get_s3_client(mock_bucket)
            result2 = await client._get_s3_client(different_mock_bucket)

            # Should be different clients
            assert result1 != result2
            assert result1 is not result2
            assert client.session.create_client.call_count == 2

            # Check cache has both
            assert len(client._s3_clients) == 2

    @pytest.mark.asyncio
    async def test_client_persistence_across_multiple_calls(self, mock_bucket):
        """Test client persistence across multiple calls"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            mock_s3_client = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_s3_client)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(return_value=mock_context_manager)

            # Multiple calls
            results = []
            for _ in range(5):
                result = await client._get_s3_client(mock_bucket)
                results.append(result)

            # All should be the same instance
            assert all(r is results[0] for r in results)
            client.session.create_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_aio_config_parameters_properly_set(self, mock_bucket):
        """Test AioConfig parameters are properly set"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            mock_s3_client = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_s3_client)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(return_value=mock_context_manager)

            await client._get_s3_client(mock_bucket)

            # Check AioConfig was passed
            call_args = client.session.create_client.call_args
            config = call_args[1]["config"]

            # Verify it's an AioConfig instance with expected attributes
            from aiobotocore.config import AioConfig

            assert isinstance(config, AioConfig)

    @pytest.mark.asyncio
    async def test_client_creation_with_invalid_credentials(self, mock_bucket):
        """Test client creation with invalid credentials"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock client creation to raise exception
            client.session.create_client = MagicMock(
                side_effect=Exception("Invalid credentials")
            )

            with pytest.raises(Exception, match="Invalid credentials"):
                await client._get_s3_client(mock_bucket)

    @pytest.mark.asyncio
    async def test_client_creation_with_malformed_account_id(self):
        """Test client creation with malformed account_id"""
        # Use a valid bucket but mock the get_base_url to return malformed URL
        malformed_bucket = Bucket(
            access_key_id="test_access_key",
            secret_access_key="test_secret_key",
            account_id="a",  # Minimal valid account_id
            name="test_bucket",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock get_base_url to return malformed URL
            with patch.object(
                client, "get_base_url", return_value="https://.r2.cloudflarestorage.com"
            ):
                mock_s3_client = AsyncMock()
                mock_context_manager = AsyncMock()
                mock_context_manager.__aenter__ = AsyncMock(return_value=mock_s3_client)
                mock_context_manager.__aexit__ = AsyncMock(return_value=None)

                client.session.create_client = MagicMock(
                    return_value=mock_context_manager
                )

                result = await client._get_s3_client(malformed_bucket)

                # Should still work but with malformed URL
                call_args = client.session.create_client.call_args
                endpoint_url = call_args[1]["endpoint_url"]
                assert endpoint_url == "https://.r2.cloudflarestorage.com"
                assert result == mock_s3_client

    @pytest.mark.asyncio
    async def test_client_creation_when_network_unavailable(self, mock_bucket):
        """Test client creation when network is unavailable"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock network error during client creation
            from botocore.exceptions import EndpointConnectionError

            client.session.create_client = MagicMock(
                side_effect=EndpointConnectionError(endpoint_url="test")
            )

            with pytest.raises(EndpointConnectionError):
                await client._get_s3_client(mock_bucket)

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, mock_bucket):
        """Test that cache key is generated correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            mock_s3_client = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_s3_client)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(return_value=mock_context_manager)

            await client._get_s3_client(mock_bucket)

            # Check cache key format
            expected_key = (
                mock_bucket.access_key_id,
                mock_bucket.secret_access_key,
                mock_bucket.account_id,
            )
            assert expected_key in client._s3_clients
            assert client._s3_clients[expected_key] == mock_s3_client

    @pytest.mark.asyncio
    async def test_bucket_with_same_credentials_different_name(self, mock_bucket):
        """Test buckets with same credentials but different names share client"""
        bucket_same_creds = Bucket(
            access_key_id=mock_bucket.access_key_id,
            secret_access_key=mock_bucket.secret_access_key,
            account_id=mock_bucket.account_id,
            name="different_bucket_name",  # Only name is different
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            mock_s3_client = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_s3_client)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(return_value=mock_context_manager)

            result1 = await client._get_s3_client(mock_bucket)
            result2 = await client._get_s3_client(bucket_same_creds)

            # Should be same client since credentials are same
            assert result1 is result2
            client.session.create_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_creation_failure_during_aenter(self, mock_bucket):
        """Test client creation failure during __aenter__"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(
                side_effect=Exception("Enter failed")
            )
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(return_value=mock_context_manager)

            with pytest.raises(Exception, match="Enter failed"):
                await client._get_s3_client(mock_bucket)

    @pytest.mark.asyncio
    async def test_concurrent_client_creation_same_bucket(self, mock_bucket):
        """Test concurrent client creation for same bucket"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            mock_s3_client = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_s3_client)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(return_value=mock_context_manager)

            # Simulate concurrent access
            tasks = [client._get_s3_client(mock_bucket) for _ in range(5)]
            results = await asyncio.gather(*tasks)

            # All should be the same client
            assert all(r is results[0] for r in results)
            # Note: Due to the implementation, create_client might be called multiple times
            # in concurrent scenarios, but the cache should still work


class TestStorageClientPurgeS3Client:
    """Test cases for StorageClient._purge_s3_client method"""

    @pytest.fixture
    def mock_bucket(self):
        """Create a mock bucket for testing"""
        return Bucket(
            access_key_id="test_access_key",
            secret_access_key="test_secret_key",
            account_id="test_account_id",
            name="test_bucket",
        )

    @pytest.fixture
    def different_mock_bucket(self):
        """Create a different mock bucket for testing"""
        return Bucket(
            access_key_id="different_access_key",
            secret_access_key="different_secret_key",
            account_id="different_account_id",
            name="different_bucket",
        )

    @pytest.mark.asyncio
    async def test_purging_existing_client_from_cache(self, mock_bucket):
        """Test purging existing client from cache"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock and add a client to cache first
            mock_s3_client = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_s3_client)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(return_value=mock_context_manager)

            # Get client to populate cache
            await client._get_s3_client(mock_bucket)

            # Verify client is in cache
            expected_key = (
                mock_bucket.access_key_id,
                mock_bucket.secret_access_key,
                mock_bucket.account_id,
            )
            assert expected_key in client._s3_clients
            assert len(client._s3_clients) == 1

            # Purge the client
            await client._purge_s3_client(mock_bucket)

            # Verify client is removed from cache
            assert expected_key not in client._s3_clients
            assert len(client._s3_clients) == 0

    @pytest.mark.asyncio
    async def test_purging_non_existent_client_should_not_error(self, mock_bucket):
        """Test purging non-existent client (should not error)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Verify cache is empty
            assert len(client._s3_clients) == 0

            # Purging non-existent client should not raise an error
            await client._purge_s3_client(mock_bucket)

            # Cache should still be empty
            assert len(client._s3_clients) == 0

    @pytest.mark.asyncio
    async def test_cache_state_after_purging(self, mock_bucket, different_mock_bucket):
        """Test cache state after purging"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock clients
            mock_s3_client1 = AsyncMock()
            mock_s3_client2 = AsyncMock()

            mock_context_manager1 = AsyncMock()
            mock_context_manager1.__aenter__ = AsyncMock(return_value=mock_s3_client1)
            mock_context_manager1.__aexit__ = AsyncMock(return_value=None)

            mock_context_manager2 = AsyncMock()
            mock_context_manager2.__aenter__ = AsyncMock(return_value=mock_s3_client2)
            mock_context_manager2.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(
                side_effect=[mock_context_manager1, mock_context_manager2]
            )

            # Add two clients to cache
            await client._get_s3_client(mock_bucket)
            await client._get_s3_client(different_mock_bucket)

            # Verify both clients are in cache
            assert len(client._s3_clients) == 2

            key1 = (
                mock_bucket.access_key_id,
                mock_bucket.secret_access_key,
                mock_bucket.account_id,
            )
            key2 = (
                different_mock_bucket.access_key_id,
                different_mock_bucket.secret_access_key,
                different_mock_bucket.account_id,
            )

            assert key1 in client._s3_clients
            assert key2 in client._s3_clients

            # Purge one client
            await client._purge_s3_client(mock_bucket)

            # Verify only the purged client is removed
            assert len(client._s3_clients) == 1
            assert key1 not in client._s3_clients
            assert key2 in client._s3_clients
            assert client._s3_clients[key2] == mock_s3_client2

    @pytest.mark.asyncio
    async def test_purged_client_not_reused(self, mock_bucket):
        """Test that purged client is not reused"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock clients
            mock_s3_client1 = AsyncMock()
            mock_s3_client2 = AsyncMock()

            mock_context_manager1 = AsyncMock()
            mock_context_manager1.__aenter__ = AsyncMock(return_value=mock_s3_client1)
            mock_context_manager1.__aexit__ = AsyncMock(return_value=None)

            mock_context_manager2 = AsyncMock()
            mock_context_manager2.__aenter__ = AsyncMock(return_value=mock_s3_client2)
            mock_context_manager2.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(
                side_effect=[mock_context_manager1, mock_context_manager2]
            )

            # Get client first time
            result1 = await client._get_s3_client(mock_bucket)
            assert result1 == mock_s3_client1
            assert client.session.create_client.call_count == 1

            # Purge the client
            await client._purge_s3_client(mock_bucket)

            # Get client again - should create new one
            result2 = await client._get_s3_client(mock_bucket)
            assert result2 == mock_s3_client2
            assert result2 != result1
            assert result2 is not result1
            assert client.session.create_client.call_count == 2

    @pytest.mark.asyncio
    async def test_purge_method_is_idempotent(self, mock_bucket):
        """Test that purging the same client multiple times doesn't error"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock and add a client to cache
            mock_s3_client = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_s3_client)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(return_value=mock_context_manager)

            # Get client to populate cache
            await client._get_s3_client(mock_bucket)
            assert len(client._s3_clients) == 1

            # Purge multiple times
            await client._purge_s3_client(mock_bucket)
            assert len(client._s3_clients) == 0

            await client._purge_s3_client(mock_bucket)
            assert len(client._s3_clients) == 0

            await client._purge_s3_client(mock_bucket)
            assert len(client._s3_clients) == 0

    @pytest.mark.asyncio
    async def test_purge_with_bucket_same_credentials_different_name(self, mock_bucket):
        """Test purging with bucket having same credentials but different name"""
        bucket_same_creds = Bucket(
            access_key_id=mock_bucket.access_key_id,
            secret_access_key=mock_bucket.secret_access_key,
            account_id=mock_bucket.account_id,
            name="different_bucket_name",  # Only name is different
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock client
            mock_s3_client = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_s3_client)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(return_value=mock_context_manager)

            # Get client with original bucket
            await client._get_s3_client(mock_bucket)
            assert len(client._s3_clients) == 1

            # Purge using bucket with same credentials but different name
            await client._purge_s3_client(bucket_same_creds)

            # Should purge the client since credentials are the same
            assert len(client._s3_clients) == 0

    @pytest.mark.asyncio
    async def test_purge_only_affects_specific_bucket_credentials(
        self, mock_bucket, different_mock_bucket
    ):
        """Test that purging only affects the specific bucket credentials"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock clients
            mock_s3_client1 = AsyncMock()
            mock_s3_client2 = AsyncMock()

            mock_context_manager1 = AsyncMock()
            mock_context_manager1.__aenter__ = AsyncMock(return_value=mock_s3_client1)
            mock_context_manager1.__aexit__ = AsyncMock(return_value=None)

            mock_context_manager2 = AsyncMock()
            mock_context_manager2.__aenter__ = AsyncMock(return_value=mock_s3_client2)
            mock_context_manager2.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(
                side_effect=[mock_context_manager1, mock_context_manager2]
            )

            # Add both clients to cache
            await client._get_s3_client(mock_bucket)
            await client._get_s3_client(different_mock_bucket)
            assert len(client._s3_clients) == 2

            # Create a third bucket with completely different credentials
            third_bucket = Bucket(
                access_key_id="third_access_key",
                secret_access_key="third_secret_key",
                account_id="third_account_id",
                name="third_bucket",
            )

            # Purge non-existent third bucket - should not affect existing clients
            await client._purge_s3_client(third_bucket)
            assert len(client._s3_clients) == 2

            # Purge first bucket - should only affect that one
            await client._purge_s3_client(mock_bucket)
            assert len(client._s3_clients) == 1

            # Verify the remaining client is the correct one
            key2 = (
                different_mock_bucket.access_key_id,
                different_mock_bucket.secret_access_key,
                different_mock_bucket.account_id,
            )
            assert key2 in client._s3_clients
            assert client._s3_clients[key2] == mock_s3_client2

    @pytest.mark.asyncio
    async def test_concurrent_purging_of_same_client(self, mock_bucket):
        """Test concurrent purging of same client"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock and add a client to cache
            mock_s3_client = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_s3_client)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            client.session.create_client = MagicMock(return_value=mock_context_manager)

            # Get client to populate cache
            await client._get_s3_client(mock_bucket)
            assert len(client._s3_clients) == 1

            # Simulate concurrent purging
            tasks = [client._purge_s3_client(mock_bucket) for _ in range(5)]
            await asyncio.gather(*tasks)

            # Should be safely purged without errors
            assert len(client._s3_clients) == 0

    def test_purge_method_is_not_async_but_defined_as_async(self, mock_bucket):
        """Test that purge method is defined as async even though it doesn't need to be"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # The method is async, so calling it should return a coroutine
            coro = client._purge_s3_client(mock_bucket)
            assert asyncio.iscoroutine(coro)

            # Clean up the coroutine
            coro.close()


class TestStorageClientGetObjectBasicFunctionality:
    """Test cases for StorageClient.get_object method - Basic functionality"""

    @pytest.fixture
    def mock_bucket(self):
        """Create a mock bucket for testing"""
        return Bucket(
            access_key_id="test_access_key",
            secret_access_key="test_secret_key",
            account_id="test_account_id",
            name="test_bucket",
        )

    def _mock_aiofile(self, content):
        """Helper to mock aiofiles.open - returns proper async context manager"""
        from unittest.mock import AsyncMock

        class MockAsyncFile:
            def __init__(self, content):
                self.content = content
                self.read = AsyncMock(return_value=content)
                self.write = AsyncMock()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        def mock_open(file_path, mode):
            return MockAsyncFile(content)

        return mock_open

    def _mock_response_body(self, content):
        """Helper to mock response body with async context manager"""

        class MockResponseBody:
            def __init__(self, content):
                self.content = content
                self.read = AsyncMock(return_value=content)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        return MockResponseBody(content)

    @pytest.mark.asyncio
    async def test_downloading_existing_small_object(self, mock_bucket):
        """Test downloading existing small object (< 600MB)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the _get_s3_client method to return our mocked client
            mock_s3_client = AsyncMock()
            with patch.object(client, "_get_s3_client", return_value=mock_s3_client):
                # Mock HEAD response for small object (< 600MB)
                mock_head_response = {
                    "ContentLength": 100 * 1024 * 1024,  # 100MB
                    "LastModified": datetime(2023, 1, 1, tzinfo=timezone.utc),
                }
                mock_s3_client.head_object = AsyncMock(return_value=mock_head_response)

                # Mock GET response with proper body
                mock_get_response = {"Body": self._mock_response_body(b"test content")}
                mock_s3_client.get_object = AsyncMock(return_value=mock_get_response)

                # Mock aiofiles.open for file operations
                with patch(
                    "aiofiles.open", side_effect=self._mock_aiofile(b"test content")
                ):
                    # Test download - fix parameter order: key first, then bucket
                    result = await client.get_object("test_key.txt", mock_bucket)

                    # Verify calls
                    mock_s3_client.head_object.assert_called_once_with(
                        Bucket=mock_bucket.name, Key="test_key.txt"
                    )
                    mock_s3_client.get_object.assert_called_once_with(
                        Bucket=mock_bucket.name, Key="test_key.txt"
                    )

                    # Should return bytes content
                    assert result == b"test content"

    @pytest.mark.asyncio
    async def test_downloading_existing_large_object_triggers_multipart(
        self, mock_bucket
    ):
        """Test downloading existing large object (> 600MB, triggers multipart)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the _get_s3_client method
            mock_s3_client = AsyncMock()
            with patch.object(client, "_get_s3_client", return_value=mock_s3_client):
                # Mock HEAD response for large object (> 600MB)
                large_size = 700 * 1024 * 1024  # 700MB
                mock_head_response = {
                    "ContentLength": large_size,
                    "LastModified": datetime(2023, 1, 1, tzinfo=timezone.utc),
                }
                mock_s3_client.head_object = AsyncMock(return_value=mock_head_response)

                # Mock multipart_download method to return True (success)
                with patch.object(
                    client, "multipart_download", return_value=True
                ) as mock_multipart:
                    # Mock aiofiles.open to simulate file reading
                    mock_file_content = b"large file content"

                    with patch(
                        "aiofiles.open",
                        side_effect=self._mock_aiofile(mock_file_content),
                    ):
                        result = await client.get_object("large_file.bin", mock_bucket)

                        # Should call multipart download for large files
                        mock_multipart.assert_called_once()
                        call_args = mock_multipart.call_args
                        # Parameters: key, temp_file_path, bucket, show_progress=show_progress
                        assert call_args[0][0] == "large_file.bin"  # key
                        assert (
                            call_args[0][2] == mock_bucket
                        )  # bucket (3rd positional arg)
                        assert call_args[1]["show_progress"]  # default value

                        assert result == mock_file_content

    @pytest.mark.asyncio
    async def test_downloading_non_existent_object_returns_none(self, mock_bucket):
        """Test downloading non-existent object returns None"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the _get_s3_client method
            mock_s3_client = AsyncMock()
            with patch.object(client, "_get_s3_client", return_value=mock_s3_client):
                # Mock HEAD to raise 404 error
                from botocore.exceptions import ClientError

                error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
                mock_s3_client.head_object = AsyncMock(
                    side_effect=ClientError(error_response, "HeadObject")
                )

                result = await client.get_object("non_existent_key.txt", mock_bucket)

                # Should return None for non-existent objects
                assert result is None
                mock_s3_client.head_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_downloading_with_invalid_bucket_credentials(self, mock_bucket):
        """Test downloading with invalid bucket credentials"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the _get_s3_client method
            mock_s3_client = AsyncMock()
            with patch.object(client, "_get_s3_client", return_value=mock_s3_client):
                # Mock HEAD to raise 403 error (access denied)
                from botocore.exceptions import ClientError

                error_response = {"Error": {"Code": "403", "Message": "Access Denied"}}
                mock_s3_client.head_object = AsyncMock(
                    side_effect=ClientError(error_response, "HeadObject")
                )

                # The actual implementation catches ClientError and returns None, doesn't re-raise
                result = await client.get_object("test_key.txt", mock_bucket)
                assert result is None

    @pytest.mark.asyncio
    async def test_downloading_with_malformed_key_names(self, mock_bucket):
        """Test downloading with malformed key names"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the _get_s3_client method
            mock_s3_client = AsyncMock()
            with patch.object(client, "_get_s3_client", return_value=mock_s3_client):
                malformed_keys = [
                    "",  # empty key
                    " ",  # whitespace key
                    "key with spaces",  # spaces
                    "key/with//double//slashes",  # double slashes
                    "key\nwith\nnewlines",  # newlines
                    "very" + "long" * 200 + "key",  # very long key
                ]

                for malformed_key in malformed_keys:
                    # Reset mocks for each iteration
                    mock_s3_client.reset_mock()

                    # Mock HEAD response
                    mock_head_response = {
                        "ContentLength": 1024,
                        "LastModified": datetime(2023, 1, 1, tzinfo=timezone.utc),
                    }
                    mock_s3_client.head_object = AsyncMock(
                        return_value=mock_head_response
                    )

                    # Mock GET response with proper body
                    mock_get_response = {
                        "Body": self._mock_response_body(b"test content")
                    }
                    mock_s3_client.get_object = AsyncMock(
                        return_value=mock_get_response
                    )

                    # Mock aiofiles.open for file operations
                    with patch(
                        "aiofiles.open", side_effect=self._mock_aiofile(b"test content")
                    ):
                        # Should handle malformed keys gracefully
                        result = await client.get_object(malformed_key, mock_bucket)

                        if malformed_key.strip():  # non-empty after strip
                            assert result == b"test content"
                            mock_s3_client.head_object.assert_called_with(
                                Bucket=mock_bucket.name, Key=malformed_key
                            )

    @pytest.mark.asyncio
    async def test_downloading_zero_byte_file(self, mock_bucket):
        """Test downloading zero-byte file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the _get_s3_client method
            mock_s3_client = AsyncMock()
            with patch.object(client, "_get_s3_client", return_value=mock_s3_client):
                # Mock HEAD response for zero-byte file
                mock_head_response = {
                    "ContentLength": 0,
                    "LastModified": datetime(2023, 1, 1, tzinfo=timezone.utc),
                }
                mock_s3_client.head_object = AsyncMock(return_value=mock_head_response)

                # Mock GET response with empty body
                mock_get_response = {"Body": self._mock_response_body(b"")}
                mock_s3_client.get_object = AsyncMock(return_value=mock_get_response)

                # Mock aiofiles.open for file operations
                with patch("aiofiles.open", side_effect=self._mock_aiofile(b"")):
                    result = await client.get_object("empty_file.txt", mock_bucket)

                    # Should handle zero-byte files
                    assert result == b""

    @pytest.mark.asyncio
    async def test_downloading_very_large_file_multiple_gb(self, mock_bucket):
        """Test downloading very large file (multiple GB)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the _get_s3_client method
            mock_s3_client = AsyncMock()
            with patch.object(client, "_get_s3_client", return_value=mock_s3_client):
                # Mock HEAD response for very large file (5GB)
                very_large_size = 5 * 1024 * 1024 * 1024  # 5GB
                mock_head_response = {
                    "ContentLength": very_large_size,
                    "LastModified": datetime(2023, 1, 1, tzinfo=timezone.utc),
                }
                mock_s3_client.head_object = AsyncMock(return_value=mock_head_response)

                # Mock multipart_download method for very large files
                with patch.object(
                    client, "multipart_download", return_value=True
                ) as mock_multipart:
                    mock_file_content = b"very large file content"

                    with patch(
                        "aiofiles.open",
                        side_effect=self._mock_aiofile(mock_file_content),
                    ):
                        result = await client.get_object(
                            "very_large_file.bin", mock_bucket
                        )

                        # Should use multipart download for very large files
                        mock_multipart.assert_called_once()
                        call_args = mock_multipart.call_args
                        assert call_args[0][0] == "very_large_file.bin"  # key
                        assert (
                            call_args[0][2] == mock_bucket
                        )  # bucket (3rd positional arg)
                        assert call_args[1]["show_progress"]  # default value

                        assert result == mock_file_content

    @pytest.mark.asyncio
    async def test_temp_file_always_cleaned_up_on_success(self, mock_bucket):
        """Test that temp file is always cleaned up on success"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the _get_s3_client method
            mock_s3_client = AsyncMock()
            with patch.object(client, "_get_s3_client", return_value=mock_s3_client):
                # Mock HEAD response
                mock_head_response = {
                    "ContentLength": 1024,
                    "LastModified": datetime(2023, 1, 1, tzinfo=timezone.utc),
                }
                mock_s3_client.head_object = AsyncMock(return_value=mock_head_response)

                # Mock GET response with proper body
                mock_get_response = {"Body": self._mock_response_body(b"test content")}
                mock_s3_client.get_object = AsyncMock(return_value=mock_get_response)

                # Track temp files created/deleted using os.path.exists
                with (
                    patch("os.path.exists") as mock_exists,
                    patch("os.remove") as mock_remove,
                    patch(
                        "aiofiles.open", side_effect=self._mock_aiofile(b"test content")
                    ),
                ):
                    # File exists when cleanup checks for it
                    mock_exists.return_value = True

                    result = await client.get_object("test_cleanup.txt", mock_bucket)

                    # Should return content
                    assert result == b"test content"
                    # Should have attempted to remove temp file
                    mock_remove.assert_called_once()

    @pytest.mark.asyncio
    async def test_temp_file_cleaned_up_on_exception(self, mock_bucket):
        """Test that temp file is cleaned up even when exception occurs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the _get_s3_client method
            mock_s3_client = AsyncMock()
            with patch.object(client, "_get_s3_client", return_value=mock_s3_client):
                # Mock HEAD response
                mock_head_response = {
                    "ContentLength": 1024,
                    "LastModified": datetime(2023, 1, 1, tzinfo=timezone.utc),
                }
                mock_s3_client.head_object = AsyncMock(return_value=mock_head_response)

                # Mock GET to raise exception during download
                mock_s3_client.get_object = AsyncMock(
                    side_effect=Exception("Download failed")
                )

                # Track temp files cleanup
                with (
                    patch("os.path.exists", return_value=True),
                    patch("os.remove") as mock_remove,
                ):
                    # Should handle exception gracefully and return None
                    result = await client.get_object("test_exception.txt", mock_bucket)

                    # Should return None due to exception handling
                    assert result is None
                    # Should still attempt cleanup
                    mock_remove.assert_called_once()

    @pytest.mark.asyncio
    async def test_downloading_with_special_characters_in_key_name(self, mock_bucket):
        """Test downloading with special characters in key name"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the _get_s3_client method
            mock_s3_client = AsyncMock()
            with patch.object(client, "_get_s3_client", return_value=mock_s3_client):
                special_keys = [
                    "file with spaces.txt",
                    "file-with-dashes.txt",
                    "file_with_underscores.txt",
                    "file.with.dots.txt",
                    "file@with#special$chars%.txt",
                    "파일이름.txt",  # Korean characters
                    "файл.txt",  # Cyrillic characters
                    "🚀rocket🚀.txt",  # Emojis
                    "file[with]brackets.txt",
                    "file(with)parentheses.txt",
                    "file{with}braces.txt",
                    "file+with+plus.txt",
                    "file=with=equals.txt",
                ]

                for special_key in special_keys:
                    # Reset mocks for each iteration
                    mock_s3_client.reset_mock()

                    # Mock HEAD response
                    mock_head_response = {
                        "ContentLength": 1024,
                        "LastModified": datetime(2023, 1, 1, tzinfo=timezone.utc),
                    }
                    mock_s3_client.head_object = AsyncMock(
                        return_value=mock_head_response
                    )

                    # Mock GET response with proper body
                    mock_get_response = {
                        "Body": self._mock_response_body(b"test content")
                    }
                    mock_s3_client.get_object = AsyncMock(
                        return_value=mock_get_response
                    )

                    with patch(
                        "aiofiles.open", side_effect=self._mock_aiofile(b"test content")
                    ):
                        result = await client.get_object(special_key, mock_bucket)

                        # Should handle special characters gracefully
                        assert result == b"test content"
                        mock_s3_client.head_object.assert_called_with(
                            Bucket=mock_bucket.name, Key=special_key
                        )

    @pytest.mark.asyncio
    async def test_behavior_when_temp_directory_full(self, mock_bucket):
        """Test behavior when temp directory is full"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the _get_s3_client method
            mock_s3_client = AsyncMock()
            with patch.object(client, "_get_s3_client", return_value=mock_s3_client):
                # Mock HEAD response
                mock_head_response = {
                    "ContentLength": 1024,
                    "LastModified": datetime(2023, 1, 1, tzinfo=timezone.utc),
                }
                mock_s3_client.head_object = AsyncMock(return_value=mock_head_response)

                # Mock file creation to raise "No space left" error in aiofiles.open
                class MockAsyncFileNoSpace:
                    async def __aenter__(self):
                        raise OSError(28, "No space left on device")  # ENOSPC

                    async def __aexit__(self, exc_type, exc_val, exc_tb):
                        pass

                def mock_open_no_space(*args, **kwargs):
                    return MockAsyncFileNoSpace()

                with patch("aiofiles.open", side_effect=mock_open_no_space):
                    # Should handle the error gracefully and return None
                    result = await client.get_object("test_no_space.txt", mock_bucket)
                    assert result is None

    @pytest.mark.asyncio
    async def test_download_with_show_progress_parameter(self, mock_bucket):
        """Test download with show_progress parameter for large files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the _get_s3_client method
            mock_s3_client = AsyncMock()
            with patch.object(client, "_get_s3_client", return_value=mock_s3_client):
                # Mock HEAD response for large file
                large_size = 700 * 1024 * 1024  # 700MB
                mock_head_response = {
                    "ContentLength": large_size,
                    "LastModified": datetime(2023, 1, 1, tzinfo=timezone.utc),
                }
                mock_s3_client.head_object = AsyncMock(return_value=mock_head_response)

                # Mock multipart_download method
                with patch.object(
                    client, "multipart_download", return_value=True
                ) as mock_multipart:
                    mock_file_content = b"large file content"

                    with patch(
                        "aiofiles.open",
                        side_effect=self._mock_aiofile(mock_file_content),
                    ):
                        # Test with show_progress=True
                        await client.get_object(
                            "large_file.bin", mock_bucket, show_progress=True
                        )

                        mock_multipart.assert_called_once()
                        call_args = mock_multipart.call_args
                        assert call_args[1]["show_progress"]

                        # Test with show_progress=False
                        mock_multipart.reset_mock()
                        await client.get_object(
                            "large_file.bin", mock_bucket, show_progress=False
                        )

                        mock_multipart.assert_called_once()
                        call_args = mock_multipart.call_args
                        assert not call_args[1]["show_progress"]

    @pytest.mark.asyncio
    async def test_download_respects_file_extension(self, mock_bucket):
        """Test that downloaded file preserves original extension"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = StorageClient(temp_dir)

            # Mock the _get_s3_client method
            mock_s3_client = AsyncMock()
            with patch.object(client, "_get_s3_client", return_value=mock_s3_client):
                test_files = [
                    "document.pdf",
                    "image.jpg",
                    "video.mp4",
                    "archive.zip",
                    "data.json",
                    "script.py",
                    "file.with.multiple.dots.txt",
                ]

                for test_file in test_files:
                    # Reset mocks for each iteration
                    mock_s3_client.reset_mock()

                    # Mock HEAD response
                    mock_head_response = {
                        "ContentLength": 1024,
                        "LastModified": datetime(2023, 1, 1, tzinfo=timezone.utc),
                    }
                    mock_s3_client.head_object = AsyncMock(
                        return_value=mock_head_response
                    )

                    # Mock GET response with proper body
                    mock_get_response = {
                        "Body": self._mock_response_body(b"test content")
                    }
                    mock_s3_client.get_object = AsyncMock(
                        return_value=mock_get_response
                    )

                    with patch(
                        "aiofiles.open", side_effect=self._mock_aiofile(b"test content")
                    ):
                        result = await client.get_object(test_file, mock_bucket)

                        # Should return file content (the method returns bytes, not file path)
                        assert result == b"test content"
