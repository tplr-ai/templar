# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""File compression utilities for tplr."""

import os
from pathlib import Path
from typing import Optional, Tuple

import zstd

import tplr


class FileCompressor:
    """High-performance file compression using zstd for aggregator files."""

    def __init__(self, compression_level: int = 9):
        """
        Initialize the file compressor.

        Args:
            compression_level: zstd compression level (1-22, higher = better compression)
                             Level 9 provides excellent compression with reasonable speed
        """
        self.compression_level = compression_level

    def compress_file(
        self, input_file_path: str, output_file_path: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Compress a file using zstd compression.

        Args:
            input_file_path: Path to the input file to compress
            output_file_path: Path for the compressed output file. If None, adds .zst extension

        Returns:
            Tuple of (output_file_path, compression_ratio)

        Raises:
            FileNotFoundError: If input file doesn't exist
            IOError: If compression fails
        """
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")

        if output_file_path is None:
            output_file_path = input_file_path + ".zst"

        try:
            with open(input_file_path, "rb") as f_in:
                data = f_in.read()
                compressed_data = zstd.compress(data, self.compression_level)

            with open(output_file_path, "wb") as f_out:
                f_out.write(compressed_data)

            compression_ratio = len(compressed_data) / len(data)

            tplr.logger.info(
                f"Compressed {Path(input_file_path).name}: "
                f"{len(data):,} → {len(compressed_data):,} bytes "
                f"(ratio: {compression_ratio:.3f}, "
                f"reduction: {(1 - compression_ratio) * 100:.1f}%)"
            )

            return output_file_path, compression_ratio

        except Exception as e:
            raise IOError(f"Failed to compress file {input_file_path}: {e}")

    def decompress_file(
        self, input_file_path: str, output_file_path: Optional[str] = None
    ) -> str:
        """
        Decompress a zstd compressed file.

        Args:
            input_file_path: Path to the compressed input file
            output_file_path: Path for the decompressed output file. If None, removes .zst extension

        Returns:
            Path to the decompressed file

        Raises:
            FileNotFoundError: If input file doesn't exist
            IOError: If decompression fails
        """
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")

        if output_file_path is None:
            if input_file_path.endswith(".zst"):
                output_file_path = input_file_path[:-4]  # Remove .zst extension
            else:
                output_file_path = input_file_path + ".decompressed"

        try:
            with open(input_file_path, "rb") as f_in:
                compressed_data = f_in.read()
                decompressed_data = zstd.decompress(compressed_data)

            with open(output_file_path, "wb") as f_out:
                f_out.write(decompressed_data)

            tplr.logger.info(
                f"Decompressed {Path(input_file_path).name}: "
                f"{len(compressed_data):,} → {len(decompressed_data):,} bytes"
            )

            return output_file_path

        except Exception as e:
            raise IOError(f"Failed to decompress file {input_file_path}: {e}")

    def compress_data(self, data: bytes) -> bytes:
        """
        Compress raw bytes using zstd.

        Args:
            data: Raw bytes to compress

        Returns:
            Compressed bytes
        """
        return zstd.compress(data, self.compression_level)

    def decompress_data(self, compressed_data: bytes) -> bytes:
        """
        Decompress zstd compressed bytes.

        Args:
            compressed_data: Compressed bytes to decompress

        Returns:
            Decompressed bytes
        """
        return zstd.decompress(compressed_data)

    def is_compressed_file(self, file_path: str) -> bool:
        """
        Check if a file is zstd compressed based on extension.

        Args:
            file_path: Path to check

        Returns:
            True if file appears to be zstd compressed
        """
        return file_path.endswith(".zst")


# Default compressor instance with high compression level for aggregator files
aggregator_compressor = FileCompressor(compression_level=9)
