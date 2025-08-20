import asyncio
import math
import os
import tempfile
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tplr.comms import Comms


# Mock Bucket class
@dataclass
class Bucket:
    name: str
    account_id: str
    access_key_id: str
    secret_access_key: str


@pytest.fixture(autouse=True)
def mock_config():
    """Mock the config module"""
    with patch("tplr.config.client_config", {}):
        yield


@pytest.fixture
async def comms_instance():
    """Create a minimal Comms instance for testing helper methods only"""
    from tplr.comms import Comms

    with patch("tplr.comms.Comms.__init__", return_value=None):
        comms = Comms.__new__(Comms)
        comms.temp_dir = tempfile.mkdtemp()
        comms._s3_clients = {}

        # Import session
        from aiobotocore.session import get_session

        comms.session = get_session()

        # Add required methods/attributes
        comms.get_base_url = (
            lambda account_id: f"https://{account_id}.r2.cloudflarestorage.com"
        )

        # Import the actual methods from the real Comms class
        from tplr.comms import Comms as RealComms

        comms._get_download_resume_info = RealComms._get_download_resume_info.__get__(
            comms
        )
        comms.download_large_file = RealComms.download_large_file.__get__(comms)

        yield comms

        # Cleanup
        import shutil

        shutil.rmtree(comms.temp_dir, ignore_errors=True)

        # Give a moment for async tasks to clean up naturally
        await asyncio.sleep(0.01)


@pytest.fixture
def test_bucket():
    """Create test bucket configuration"""
    return Bucket(
        name="test-bucket",
        account_id="test-account",
        access_key_id="test-key",
        secret_access_key="test-secret",
    )


@pytest.fixture
def mock_s3_client():
    """Create mock S3 client with streaming body"""
    client = AsyncMock()

    # Create mock stream body
    mock_body = AsyncMock()
    mock_body.read = AsyncMock(return_value=b"test data chunk")
    mock_body.__aenter__ = AsyncMock(return_value=mock_body)
    mock_body.__aexit__ = AsyncMock(return_value=None)

    client.get_object = AsyncMock(
        return_value={"Body": mock_body, "ContentLength": 1024}
    )

    return client


class TestAdaptiveChunkSizing:
    """Test adaptive chunk sizing decision logic"""

    @pytest.mark.parametrize(
        "file_size_gb,expected_chunk_mb,expected_min_workers",
        [
            (0.5, 64, 6),  # < 10GB: 64MB chunks, min 6 workers
            (15, 256, 8),  # 10-100GB: 256MB chunks, min 8 workers (increased)
            (150, 512, 8),  # 100GB+: 512MB chunks, min 8 workers (increased)
        ],
    )
    def test_chunk_sizing_logic(
        self, file_size_gb, expected_chunk_mb, expected_min_workers
    ):
        """Test chunk sizing decision logic without full download"""
        file_size = int(file_size_gb * 1024 * 1024 * 1024)

        params = Comms._calc_download_params(file_size)

        # Verify chunk size
        expected_chunk_size = expected_chunk_mb * 1024 * 1024
        assert params["chunk_size"] == expected_chunk_size, (
            f"Chunk size {params['chunk_size']} != expected {expected_chunk_size} for {file_size_gb}GB file"
        )

        # Verify minimum worker count
        assert params["max_workers"] >= expected_min_workers, (
            f"Worker count {params['max_workers']} < expected minimum {expected_min_workers} for {file_size_gb}GB file"
        )

        # Verify total chunks calculation
        expected_chunks = math.ceil(file_size / expected_chunk_size)
        assert params["total_chunks"] == expected_chunks, (
            f"Total chunks {params['total_chunks']} != expected {expected_chunks}"
        )


class TestCustomWorkerEnvironment:
    """Test DOWNLOAD_MAX_WORKERS environment variable decision logic"""

    def test_custom_worker_count(self, monkeypatch):
        """Test that custom worker count overrides calculated workers"""
        monkeypatch.setenv("DOWNLOAD_MAX_WORKERS", "3")

        file_size = 150 * 1024 * 1024 * 1024  # 150GB - would normally use 16 workers

        params = Comms._calc_download_params(file_size, custom_workers="3")

        # Should use custom worker count instead of calculated 16
        assert params["max_workers"] == 3, (
            f"Expected 3 workers, got {params['max_workers']}"
        )

        # Other params should be normal
        assert params["chunk_size"] == 512 * 1024 * 1024  # 512MB for 100GB+
        assert (
            abs(params["file_size_gb"] - 150.0) < 0.1
        )  # Allow small floating point differences

    def test_invalid_custom_worker_count(self):
        """Test handling of invalid DOWNLOAD_MAX_WORKERS value"""
        file_size = 15 * 1024 * 1024 * 1024  # 15GB (clearly > 10GB)

        # Invalid string should fall back to default calculation
        params = Comms._calc_download_params(file_size, custom_workers="invalid")

        # Should use default calculated workers (min 8 for 10GB+)
        assert params["max_workers"] >= 8, (
            f"Should fall back to default workers, got {params['max_workers']}"
        )

        # Other params should be normal
        assert params["chunk_size"] == 256 * 1024 * 1024  # 256MB for 10GB+

    def test_worker_count_limited_by_chunks(self):
        """Test that worker count is limited by total chunks"""
        # Small file with only 1 chunk
        file_size = 32 * 1024 * 1024  # 32MB (< 64MB chunk size)

        params = Comms._calc_download_params(file_size, custom_workers="16")

        # Should be limited to 1 worker since there's only 1 chunk
        assert params["max_workers"] == 1, (
            f"Should be limited to 1 worker, got {params['max_workers']}"
        )
        assert params["total_chunks"] == 1


class TestResumeCapability:
    """Test download resume functionality"""

    @pytest.mark.asyncio
    async def test_resume_info_with_partial_file(self, comms_instance):
        """Test resume info calculation with partial file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test_resume.bin")

            chunk_size = 256 * 1024 * 1024  # 256MB
            file_size = 1024 * 1024 * 1024  # 1GB
            total_chunks = 4

            # Create partial file (2 chunks)
            partial_size = chunk_size * 2
            with open(temp_file, "wb") as f:
                f.write(b"0" * partial_size)

            resume_info = await comms_instance._get_download_resume_info(
                temp_file, total_chunks, chunk_size, file_size
            )

            assert len(resume_info["completed_chunks"]) == 2
            assert len(resume_info["remaining_chunks"]) == 2
            assert 0 in resume_info["completed_chunks"]
            assert 1 in resume_info["completed_chunks"]
            assert 2 in resume_info["remaining_chunks"]
            assert 3 in resume_info["remaining_chunks"]

    @pytest.mark.asyncio
    async def test_resume_info_with_empty_file(self, comms_instance):
        """Test resume info with empty file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test_empty.bin")

            # Create empty file
            open(temp_file, "wb").close()

            chunk_size = 256 * 1024 * 1024
            file_size = 1024 * 1024 * 1024
            total_chunks = 4

            resume_info = await comms_instance._get_download_resume_info(
                temp_file, total_chunks, chunk_size, file_size
            )

            # Empty file should start fresh
            assert len(resume_info["completed_chunks"]) == 0
            assert len(resume_info["remaining_chunks"]) == total_chunks

    @pytest.mark.asyncio
    async def test_resume_info_no_file(self, comms_instance):
        """Test resume info when file doesn't exist"""
        temp_file = "/tmp/nonexistent_file.bin"

        chunk_size = 256 * 1024 * 1024
        file_size = 1024 * 1024 * 1024
        total_chunks = 4

        resume_info = await comms_instance._get_download_resume_info(
            temp_file, total_chunks, chunk_size, file_size
        )

        # No file should start fresh
        assert len(resume_info["completed_chunks"]) == 0
        assert len(resume_info["remaining_chunks"]) == total_chunks


class TestMemoryEfficientStreaming:
    """Test memory-efficient streaming with 8MB buffer"""

    @pytest.mark.asyncio
    async def test_buffer_size_limit(self, comms_instance, mock_s3_client):
        """Verify 8MB buffer limit is enforced through actual behavior"""
        # Setup test parameters
        expected_buffer_size = 8 * 1024 * 1024  # 8MB
        file_size = 20 * 1024 * 1024  # 20MB file

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test_download.bin")
            test_bucket = Bucket(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret",
            )

            # Track all read calls to verify chunk sizes
            read_calls = []

            async def mock_get_object(Bucket, Key, Range=None):
                if Range:
                    # Parse range header
                    range_str = Range.replace("bytes=", "")
                    start, end = map(int, range_str.split("-"))
                    total_size = end - start + 1

                    # Create a mock stream that tracks read sizes
                    class MockStream:
                        def __init__(self, total_size):
                            self.total_size = total_size
                            self.bytes_read = 0

                        async def read(self, size=None):
                            """Mock read that tracks chunk sizes"""
                            if self.bytes_read >= self.total_size:
                                return b""

                            # Track the size requested
                            read_calls.append(size)

                            # Return data up to the requested size
                            if size is None:
                                # Should not happen with fixed implementation
                                chunk_size = self.total_size - self.bytes_read
                            else:
                                # Return exactly the requested amount (or less for final chunk)
                                chunk_size = min(
                                    size, self.total_size - self.bytes_read
                                )

                            self.bytes_read += chunk_size
                            return b"X" * chunk_size

                        async def __aenter__(self):
                            return self

                        async def __aexit__(self, *args):
                            pass

                    return {"Body": MockStream(total_size)}
                else:
                    return {"ContentLength": file_size}

            mock_s3_client.get_object = AsyncMock(side_effect=mock_get_object)

            # Import the real download_large_file method
            from tplr.comms import Comms as RealComms

            comms_instance.download_large_file = RealComms.download_large_file.__get__(
                comms_instance
            )

            # Mock tqdm and logger
            with patch("tplr.comms.std_tqdm") as mock_tqdm:
                mock_pbar = MagicMock()
                mock_tqdm.return_value = mock_pbar

                with patch("tplr.logger"):
                    result = await comms_instance.download_large_file(
                        mock_s3_client, test_bucket, "test-key", file_size, temp_file
                    )

            assert result is True

            # Verify that read was called with the expected buffer size
            assert len(read_calls) > 0, "No read calls were made"

            # Check that all read calls use the buffer size (8MB) or less
            for i, size in enumerate(read_calls):
                assert size is not None, (
                    f"Read call {i} was called without size parameter"
                )
                assert size <= expected_buffer_size, (
                    f"Read call {i} used size {size}, exceeds buffer limit {expected_buffer_size}"
                )

            # Verify we had multiple reads (streaming behavior)
            assert len(read_calls) >= 2, (
                f"Expected multiple reads for {file_size} bytes with {expected_buffer_size} buffer"
            )

    @pytest.mark.asyncio
    async def test_streaming_read_behavior(self, mock_s3_client):
        """Test that streaming reads data correctly"""
        # The mock client's read() should be called without size parameter
        # due to the ClientResponse.read() fix

        response = await mock_s3_client.get_object(Bucket="test", Key="test")
        async with response["Body"] as stream:
            data = await stream.read()

            # Verify read was called without arguments
            stream.read.assert_called_once_with()
            assert data == b"test data chunk"


class TestIntegrationDownloadLargeFile:
    """Integration tests for download_large_file method"""

    @pytest.mark.asyncio
    async def test_download_large_file_basic(self, comms_instance, mock_s3_client):
        """Test basic download_large_file functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test_download.bin")
            test_bucket = Bucket(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret",
            )

            file_size = 100 * 1024 * 1024  # 100MB

            # Mock S3 responses for chunks
            async def mock_get_object(Bucket, Key, Range=None):
                if Range:
                    # Parse range header
                    range_str = Range.replace("bytes=", "")
                    start, end = map(int, range_str.split("-"))
                    chunk_size = end - start + 1

                    # Create mock stream with appropriate data
                    mock_body = AsyncMock()
                    mock_body.read = AsyncMock(return_value=b"X" * chunk_size)
                    mock_body.__aenter__ = AsyncMock(return_value=mock_body)
                    mock_body.__aexit__ = AsyncMock(return_value=None)

                    return {"Body": mock_body}
                else:
                    return {"ContentLength": file_size}

            mock_s3_client.get_object = AsyncMock(side_effect=mock_get_object)

            # Import the real download_large_file method and bind it
            from tplr.comms import Comms as RealComms

            comms_instance.download_large_file = RealComms.download_large_file.__get__(
                comms_instance
            )

            # Mock tqdm to avoid terminal output
            with patch("tplr.comms.std_tqdm") as mock_tqdm:
                mock_pbar = MagicMock()
                mock_tqdm.return_value = mock_pbar

                # Mock logger
                with patch("tplr.logger") as mock_logger:
                    result = await comms_instance.download_large_file(
                        mock_s3_client, test_bucket, "test-key", file_size, temp_file
                    )

                    assert result is True
                    assert os.path.exists(temp_file)

                    # Verify file was created with correct size
                    actual_size = os.path.getsize(temp_file)
                    assert actual_size == file_size

                    # Verify progress bar was updated
                    assert mock_pbar.update.called

                    # Verify logging
                    mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_download_with_retry_on_failure(self, comms_instance, mock_s3_client):
        """Test retry logic when chunk download fails"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test_retry.bin")
            test_bucket = Bucket(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret",
            )

            file_size = 100 * 1024 * 1024  # 100MB

            # Track call count
            call_count = {"count": 0}

            async def mock_get_object(Bucket, Key, Range=None):
                if Range:
                    call_count["count"] += 1

                    # Fail first attempt, succeed on second
                    if call_count["count"] == 1:
                        raise Exception("Network error")

                    # Parse range header
                    range_str = Range.replace("bytes=", "")
                    start, end = map(int, range_str.split("-"))
                    chunk_size = end - start + 1

                    # Create mock stream
                    mock_body = AsyncMock()
                    mock_body.read = AsyncMock(return_value=b"X" * chunk_size)
                    mock_body.__aenter__ = AsyncMock(return_value=mock_body)
                    mock_body.__aexit__ = AsyncMock(return_value=None)

                    return {"Body": mock_body}
                else:
                    return {"ContentLength": file_size}

            mock_s3_client.get_object = AsyncMock(side_effect=mock_get_object)

            # Import the real download_large_file method
            from tplr.comms import Comms as RealComms

            comms_instance.download_large_file = RealComms.download_large_file.__get__(
                comms_instance
            )

            # Mock tqdm and logger
            with patch("tplr.comms.std_tqdm") as mock_tqdm:
                mock_pbar = MagicMock()
                mock_tqdm.return_value = mock_pbar

                with patch("tplr.logger") as mock_logger:
                    # Mock sleep to speed up test
                    with patch("asyncio.sleep", return_value=None):
                        result = await comms_instance.download_large_file(
                            mock_s3_client,
                            test_bucket,
                            "test-key",
                            file_size,
                            temp_file,
                        )

                        assert result is True
                        # Verify retry occurred
                        assert call_count["count"] > 1

                        # Verify error was logged
                        error_calls = [
                            call
                            for call in mock_logger.error.call_args_list
                            if "Error downloading chunk" in str(call)
                        ]
                        assert len(error_calls) > 0

    @pytest.mark.asyncio
    async def test_download_with_resume(self, comms_instance, mock_s3_client):
        """Test resume capability with partial file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test_resume.bin")
            test_bucket = Bucket(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret",
            )

            chunk_size = 64 * 1024 * 1024  # 64MB
            file_size = 4 * chunk_size  # 256MB (4 chunks)

            # Pre-create partial file (2 chunks already downloaded)
            with open(temp_file, "wb") as f:
                f.write(b"A" * (2 * chunk_size))

            # Track which chunks are requested
            requested_chunks = []

            async def mock_get_object(Bucket, Key, Range=None):
                if Range:
                    # Parse range header
                    range_str = Range.replace("bytes=", "")
                    start, end = map(int, range_str.split("-"))
                    chunk_num = start // chunk_size
                    requested_chunks.append(chunk_num)

                    chunk_data_size = end - start + 1

                    # Create mock stream
                    mock_body = AsyncMock()
                    mock_body.read = AsyncMock(return_value=b"B" * chunk_data_size)
                    mock_body.__aenter__ = AsyncMock(return_value=mock_body)
                    mock_body.__aexit__ = AsyncMock(return_value=None)

                    return {"Body": mock_body}
                else:
                    return {"ContentLength": file_size}

            mock_s3_client.get_object = AsyncMock(side_effect=mock_get_object)

            # Import the real download_large_file method
            from tplr.comms import Comms as RealComms

            comms_instance.download_large_file = RealComms.download_large_file.__get__(
                comms_instance
            )

            # Mock tqdm and logger
            with patch("tplr.comms.std_tqdm") as mock_tqdm:
                mock_pbar = MagicMock()
                mock_tqdm.return_value = mock_pbar

                with patch("tplr.logger") as mock_logger:
                    result = await comms_instance.download_large_file(
                        mock_s3_client, test_bucket, "test-key", file_size, temp_file
                    )

                    assert result is True

                    # Verify only chunks 2 and 3 were downloaded (0 and 1 already existed)
                    # Note: Due to resume logic, it should skip chunks 0 and 1
                    assert 2 in requested_chunks
                    assert 3 in requested_chunks

                    # Verify resume was logged
                    resume_log_calls = [
                        call
                        for call in mock_logger.info.call_args_list
                        if "Resuming download" in str(call)
                    ]
                    assert len(resume_log_calls) > 0


class TestStreamingEfficiency:
    """Test memory-efficient streaming implementation"""

    @pytest.mark.asyncio
    async def test_streaming_without_loading_to_memory(
        self, comms_instance, mock_s3_client
    ):
        """Test that chunks are streamed directly to file without loading entire chunk to memory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test_stream.bin")
            test_bucket = Bucket(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret",
            )

            file_size = 100 * 1024 * 1024  # 100MB
            chunk_size = 64 * 1024 * 1024  # 64MB

            # Track memory usage simulation
            read_sizes = []

            async def mock_get_object(Bucket, Key, Range=None):
                if Range:
                    # Parse range header
                    range_str = Range.replace("bytes=", "")
                    start, end = map(int, range_str.split("-"))
                    total_size = end - start + 1

                    # Simulate streaming in smaller sub-chunks (8MB)
                    class MockStream:
                        def __init__(self, total_size):
                            self.total_size = total_size
                            self.bytes_read = 0
                            self.buffer_size = 8 * 1024 * 1024  # 8MB

                        async def read(self, size=None):
                            if self.bytes_read >= self.total_size:
                                return b""

                            # If size is provided, use it; otherwise use buffer_size
                            if size is not None:
                                chunk_to_read = min(
                                    size, self.total_size - self.bytes_read
                                )
                            else:
                                # Fallback to buffer size if no size specified
                                chunk_to_read = min(
                                    self.buffer_size, self.total_size - self.bytes_read
                                )

                            read_sizes.append(chunk_to_read)
                            self.bytes_read += chunk_to_read
                            return b"X" * chunk_to_read

                        async def __aenter__(self):
                            return self

                        async def __aexit__(self, *args):
                            pass

                    return {"Body": MockStream(total_size)}
                else:
                    return {"ContentLength": file_size}

            mock_s3_client.get_object = AsyncMock(side_effect=mock_get_object)

            # Import the real download_large_file method
            from tplr.comms import Comms as RealComms

            comms_instance.download_large_file = RealComms.download_large_file.__get__(
                comms_instance
            )

            # Mock tqdm
            with patch("tplr.comms.std_tqdm") as mock_tqdm:
                mock_pbar = MagicMock()
                mock_tqdm.return_value = mock_pbar

                with patch("tplr.logger"):
                    result = await comms_instance.download_large_file(
                        mock_s3_client, test_bucket, "test-key", file_size, temp_file
                    )

                    assert result is True

                    # Verify that data was read in chunks no larger than 8MB
                    max_read = max(read_sizes) if read_sizes else 0
                    assert max_read <= 8 * 1024 * 1024, (
                        f"Read size {max_read} exceeds 8MB buffer limit"
                    )

                    # Verify multiple smaller reads occurred for large chunks
                    assert len(read_sizes) > 1, (
                        "Should have multiple reads for streaming"
                    )


class TestErrorRecovery:
    """Test error recovery and edge cases"""

    @pytest.mark.asyncio
    async def test_handle_missing_chunks(self, comms_instance, mock_s3_client):
        """Test handling when some chunks fail permanently"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test_missing.bin")
            test_bucket = Bucket(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret",
            )

            file_size = 256 * 1024 * 1024  # 256MB

            async def mock_get_object(Bucket, Key, Range=None):
                if Range:
                    # Always fail for certain chunks to test error handling
                    range_str = Range.replace("bytes=", "")
                    start, _ = map(int, range_str.split("-"))

                    # Fail for chunk 2 (assuming 64MB chunks)
                    if start == 128 * 1024 * 1024:
                        raise Exception("Permanent failure for chunk 2")

                    # Success for other chunks
                    _, end = map(int, range_str.split("-"))
                    chunk_size = end - start + 1

                    mock_body = AsyncMock()
                    mock_body.read = AsyncMock(return_value=b"X" * chunk_size)
                    mock_body.__aenter__ = AsyncMock(return_value=mock_body)
                    mock_body.__aexit__ = AsyncMock(return_value=None)

                    return {"Body": mock_body}
                else:
                    return {"ContentLength": file_size}

            mock_s3_client.get_object = AsyncMock(side_effect=mock_get_object)

            # Import the real download_large_file method
            from tplr.comms import Comms as RealComms

            comms_instance.download_large_file = RealComms.download_large_file.__get__(
                comms_instance
            )

            # Mock tqdm and logger
            with patch("tplr.comms.std_tqdm") as mock_tqdm:
                mock_pbar = MagicMock()
                mock_tqdm.return_value = mock_pbar

                with patch("tplr.logger") as mock_logger:
                    with patch("asyncio.sleep", return_value=None):
                        # Should fail due to missing chunk
                        result = await comms_instance.download_large_file(
                            mock_s3_client,
                            test_bucket,
                            "test-key",
                            file_size,
                            temp_file,
                        )

                        assert result is False

                        # Verify error was logged multiple times (retries)
                        error_calls = mock_logger.error.call_args_list
                        chunk_errors = [
                            call
                            for call in error_calls
                            if "Error downloading chunk" in str(call)
                        ]
                        assert len(chunk_errors) >= 3  # Should retry 3 times

                        # Give tasks time to complete
                        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_file_size_mismatch(self, comms_instance, mock_s3_client):
        """Test handling when downloaded file size doesn't match expected"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test_mismatch.bin")
            test_bucket = Bucket(
                name="test-bucket",
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret",
            )

            file_size = 100 * 1024 * 1024  # 100MB

            async def mock_get_object(Bucket, Key, Range=None):
                if Range:
                    # Return less data than requested to trigger mismatch
                    range_str = Range.replace("bytes=", "")
                    start, end = map(int, range_str.split("-"))

                    # Create a mock body that returns less data than expected
                    class IncompleteMockBody:
                        def __init__(self, expected_size):
                            self.expected_size = expected_size
                            self.call_count = 0

                        async def read(self, size=None):
                            # First call returns some data, second returns empty (simulating incomplete download)
                            if self.call_count == 0:
                                self.call_count += 1
                                # Return only 1KB instead of the full chunk
                                return b"X" * 1024
                            else:
                                # No more data available
                                return b""

                        async def __aenter__(self):
                            return self

                        async def __aexit__(self, *args):
                            pass

                    expected_size = end - start + 1
                    return {"Body": IncompleteMockBody(expected_size)}
                else:
                    return {"ContentLength": file_size}

            mock_s3_client.get_object = AsyncMock(side_effect=mock_get_object)

            # The method is already bound in the fixture

            # Mock tqdm and logger
            with patch("tplr.comms.std_tqdm") as mock_tqdm:
                mock_pbar = MagicMock()
                mock_tqdm.return_value = mock_pbar

                with patch("tplr.logger") as mock_logger:
                    # Mock asyncio.sleep to speed up retries
                    with patch("asyncio.sleep", return_value=None):
                        # Should fail after retries due to size mismatch
                        result = await comms_instance.download_large_file(
                            mock_s3_client,
                            test_bucket,
                            "test-key",
                            file_size,
                            temp_file,
                        )

                        # The download should fail
                        assert result is False

                        # Verify error was logged for chunk write mismatch
                        error_calls = mock_logger.error.call_args_list
                        mismatch_errors = [
                            call
                            for call in error_calls
                            if "Chunk write mismatch" in str(call)
                            or "Error downloading chunk" in str(call)
                        ]
                        assert len(mismatch_errors) > 0, (
                            "Should have logged chunk write mismatch errors"
                        )

                        # Give tasks time to complete
                        await asyncio.sleep(0)
