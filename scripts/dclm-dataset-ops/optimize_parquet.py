#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "argparse",
#   "pyarrow",
# ]
# ///
"""
Optimize a Parquet file for efficient reading.
This script reads a Parquet file and writes it back with optimized settings for
compression and indexing. It uses the ZSTD compression codec with a specified
compression level and row group size for optimal read performance.

Usage:
    optimize_parquet.py \
        ./input_file.parquet \
        ./optimized_file.parquet \
        --compression_level 3 \
        --row_group_size 1000

"""

import argparse
import os

import pyarrow.parquet as pq


def optimize_parquet_file(
    input_file: str,
    output_file: str,
    compression: str = "ZSTD",
    compression_level: int = 3,
    row_group_size: int = 100_000,
):
    """
    Reads a Parquet file, then writes it with optimized compression and indexing.

    Args:
        input_file (str): Path to the existing Parquet file.
        output_file (str): Path where the optimized Parquet file will be saved.
        compression (str): Compression codec to use (default: ZSTD).
        compression_level (int): Compression level (ZSTD 1-22; default: 3).
        row_group_size (int): Number of rows per row group for optimal read performance.
    """

    print(f"Reading original file: {input_file}")

    # Read entire table from Parquet file
    table = pq.read_table(input_file)

    print(f"File schema:\n{table.schema}")

    # Write optimized Parquet file
    print(f"Writing optimized file: {output_file}")
    pq.write_table(
        table,
        output_file,
        compression=compression,
        compression_level=compression_level,
        use_dictionary=True,  # Enables dictionary encoding
        data_page_size=4 * 1024 * 1024,  # 4MB data pages
        row_group_size=row_group_size,  # rows per group
        version="2.6",  # modern Parquet format
        use_deprecated_int96_timestamps=False,  # recommended setting
    )

    # Report new file details
    optimized_file_size = os.path.getsize(output_file) / (1024 * 1024)
    original_file_size = os.path.getsize(input_file) / (1024 * 1024)
    reduction = (original_file_size - optimized_file_size) / original_file_size * 100

    print("Optimization complete:")
    print(f"- Original size: {original_file_size:.2f} MB")
    print(f"- Optimized size: {optimized_file_size:.2f} MB")
    print(f"- Size reduction: {reduction:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize a Parquet file for efficient reading.")
    parser.add_argument("input_file", help="Input Parquet file path")
    parser.add_argument("output_file", help="Output optimized Parquet file path")
    parser.add_argument(
        "--compression_level", type=int, default=3, help="ZSTD compression level (default: 3)"
    )
    parser.add_argument(
        "--row_group_size", type=int, default=1000, help="Rows per row group (default: 1000)"
    )

    args = parser.parse_args()

    optimize_parquet_file(
        input_file=args.input_file,
        output_file=args.output_file,
        compression="ZSTD",
        compression_level=args.compression_level,
        row_group_size=args.row_group_size,
    )
