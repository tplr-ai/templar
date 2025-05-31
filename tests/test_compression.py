"""Unit tests for the compression module."""

import os
import tempfile
import torch
import pytest

from tplr.compression import FileCompressor, aggregator_compressor


class TestFileCompressor:
    """Test cases for FileCompressor class."""

    @pytest.fixture
    def compressor(self):
        """Create a test compressor instance."""
        return FileCompressor(compression_level=9)

    @pytest.fixture
    def test_data(self):
        """Create test data similar to aggregator files."""
        return {
            "model_state": torch.randn(100, 100),
            "metadata": {
                "window": 42,
                "timestamp": "2025-01-01T00:00:00Z",
                "version": "0.1.0",
                "success_rate": 0.85,
            },
            "gradients": {"layer1": torch.randn(50, 50), "layer2": torch.randn(25, 25)},
        }

    @pytest.fixture
    def temp_file(self, test_data):
        """Create a temporary test file."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(test_data, f.name)
            yield f.name
        if os.path.exists(f.name):
            os.unlink(f.name)

    def test_compress_file(self, compressor, temp_file):
        """Test file compression functionality."""
        compressed_file, ratio = compressor.compress_file(temp_file)

        assert os.path.exists(compressed_file)
        assert compressed_file.endswith(".zst")
        assert 0 < ratio < 1.0
        assert os.path.getsize(compressed_file) < os.path.getsize(temp_file)

        # Clean up
        os.unlink(compressed_file)

    def test_compress_file_custom_output(self, compressor, temp_file):
        """Test file compression with custom output path."""
        custom_output = temp_file + ".custom.zst"
        compressed_file, ratio = compressor.compress_file(temp_file, custom_output)

        assert compressed_file == custom_output
        assert os.path.exists(custom_output)
        assert 0 < ratio < 1.0

        # Clean up
        os.unlink(custom_output)

    def test_compress_file_not_found(self, compressor):
        """Test compression with non-existent file."""
        with pytest.raises(FileNotFoundError):
            compressor.compress_file("/nonexistent/file.pt")

    def test_decompress_file(self, compressor, temp_file, test_data):
        """Test file decompression functionality."""
        # First compress the file
        compressed_file, _ = compressor.compress_file(temp_file)

        # Then decompress it
        decompressed_file = compressor.decompress_file(compressed_file)

        assert os.path.exists(decompressed_file)
        assert not decompressed_file.endswith(".zst")

        # Verify data integrity
        loaded_data = torch.load(decompressed_file, weights_only=False)
        assert torch.equal(test_data["model_state"], loaded_data["model_state"])
        assert test_data["metadata"] == loaded_data["metadata"]

        # Clean up
        os.unlink(compressed_file)
        os.unlink(decompressed_file)

    def test_decompress_file_custom_output(self, compressor, temp_file):
        """Test file decompression with custom output path."""
        compressed_file, _ = compressor.compress_file(temp_file)
        custom_output = temp_file + ".custom.decompressed"

        decompressed_file = compressor.decompress_file(compressed_file, custom_output)

        assert decompressed_file == custom_output
        assert os.path.exists(custom_output)

        # Clean up
        os.unlink(compressed_file)
        os.unlink(custom_output)

    def test_decompress_file_not_found(self, compressor):
        """Test decompression with non-existent file."""
        with pytest.raises(FileNotFoundError):
            compressor.decompress_file("/nonexistent/file.zst")

    def test_compress_decompress_data(self, compressor):
        """Test raw data compression and decompression."""
        original_data = b"test data for compression" * 1000
        compressed_data = compressor.compress_data(original_data)
        decompressed_data = compressor.decompress_data(compressed_data)

        assert len(compressed_data) < len(original_data)
        assert decompressed_data == original_data

    def test_is_compressed_file(self, compressor):
        """Test compressed file detection."""
        assert compressor.is_compressed_file("file.zst")
        assert compressor.is_compressed_file("/path/to/file.pt.zst")
        assert not compressor.is_compressed_file("file.pt")
        assert not compressor.is_compressed_file("file.txt")

    def test_compression_level(self):
        """Test different compression levels."""
        low_compressor = FileCompressor(compression_level=1)
        high_compressor = FileCompressor(compression_level=9)

        assert low_compressor.compression_level == 1
        assert high_compressor.compression_level == 9

    def test_roundtrip_data_integrity(self, compressor, temp_file, test_data):
        """Test complete roundtrip maintains data integrity."""
        # Compress
        compressed_file, ratio = compressor.compress_file(temp_file)

        # Decompress
        decompressed_file = compressor.decompress_file(compressed_file)

        # Load and verify
        loaded_data = torch.load(decompressed_file, weights_only=False)

        # Check all tensor data
        assert torch.equal(test_data["model_state"], loaded_data["model_state"])
        assert torch.equal(
            test_data["gradients"]["layer1"], loaded_data["gradients"]["layer1"]
        )
        assert torch.equal(
            test_data["gradients"]["layer2"], loaded_data["gradients"]["layer2"]
        )

        # Check metadata
        assert test_data["metadata"] == loaded_data["metadata"]

        # Verify compression actually occurred
        assert ratio < 1.0
        assert os.path.getsize(compressed_file) < os.path.getsize(temp_file)

        # Clean up
        os.unlink(compressed_file)
        os.unlink(decompressed_file)


class TestAggregatorCompressor:
    """Test cases for the default aggregator compressor instance."""

    def test_aggregator_compressor_exists(self):
        """Test that the default aggregator compressor is available."""
        assert aggregator_compressor is not None
        assert isinstance(aggregator_compressor, FileCompressor)
        assert aggregator_compressor.compression_level == 9

    def test_aggregator_compressor_functionality(self):
        """Test that the aggregator compressor works correctly."""
        test_data = b"aggregator test data" * 100
        compressed = aggregator_compressor.compress_data(test_data)
        decompressed = aggregator_compressor.decompress_data(compressed)

        assert len(compressed) < len(test_data)
        assert decompressed == test_data
