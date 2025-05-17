#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "argparse",
#   "asyncio",
#   "pyarrow",
#   "numpy",
#   "s3fs",
# ]
# ///
"""
benchmark_process_page.py

Benchmark read_row_group and simulated _process_page (I/O + processing)
on local Parquet files.

Usage:
    benchmark_process_page.py --parquet-files file1.parquet file2.parquet \
    --iterations 20 \
    --sequence-length 4096 \
    --rows-per-page 200 \
    --output my_benchmark_results.json \
    --debug
"""

import argparse
import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import s3fs

# Configure pyarrow for optimal performance
pyarrow.set_io_thread_count(os.cpu_count())


class TimerProfiler:
    """High-precision profiler for timing function execution"""

    def __init__(self, name):
        self.name = name
        self.stats = {}
        self.lock = threading.Lock()

    def profile(self, func_name):
        """Decorator to profile function execution time"""

        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    status = "success"
                except Exception as e:
                    result = e
                    status = "error"
                    raise
                finally:
                    end_time = time.perf_counter()
                    self._record_time(func_name, start_time, end_time, status)
                return result

            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    status = "success"
                except Exception as e:
                    result = e
                    status = "error"
                    raise
                finally:
                    end_time = time.perf_counter()
                    self._record_time(func_name, start_time, end_time, status)
                return result

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def _record_time(self, func_name, start_time, end_time, status):
        """Record detailed timing information"""
        elapsed_ms = (end_time - start_time) * 1000  # Convert to ms

        with self.lock:
            if func_name not in self.stats:
                self.stats[func_name] = {
                    "calls": [],
                    "success_count": 0,
                    "error_count": 0,
                }

            self.stats[func_name]["calls"].append(
                {"elapsed_ms": elapsed_ms, "status": status, "timestamp": time.time()}
            )

            if status == "success":
                self.stats[func_name]["success_count"] += 1
            else:
                self.stats[func_name]["error_count"] += 1

    def get_stats(self):
        """Get comprehensive statistics for all profiled functions"""
        result = {}

        with self.lock:
            for func_name, data in self.stats.items():
                elapsed_times = [
                    call["elapsed_ms"]
                    for call in data["calls"]
                    if call["status"] == "success"
                ]

                if elapsed_times:
                    result[func_name] = {
                        "count": len(elapsed_times),
                        "success_count": data["success_count"],
                        "error_count": data["error_count"],
                        "mean": sum(elapsed_times) / len(elapsed_times),
                        "median": np.median(elapsed_times),
                        "min": min(elapsed_times),
                        "max": max(elapsed_times),
                        "p90": np.percentile(elapsed_times, 90),
                        "p99": np.percentile(elapsed_times, 99),
                        "total": sum(elapsed_times),
                        "std_dev": np.std(elapsed_times),
                    }
                else:
                    result[func_name] = {
                        "count": 0,
                        "success_count": 0,
                        "error_count": 0,
                        "mean": 0,
                        "median": 0,
                        "min": 0,
                        "max": 0,
                        "p90": 0,
                        "p99": 0,
                        "total": 0,
                        "std_dev": 0,
                    }

        return result

    def log_summary(self):
        """Log a formatted summary of profiling statistics"""
        stats = self.get_stats()

        print(f"\n=== {self.name} Profiling Summary ===")
        print(
            f"{'Function':<30} {'Count':<8} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'P90':<10} {'Total':<12} {'Std Dev':<10}"
        )
        print("-" * 110)

        for func_name, data in sorted(
            stats.items(), key=lambda x: x[1]["total"], reverse=True
        ):
            print(
                f"{func_name:<30} {data['count']:<8} {data['mean']:<10.2f} {data['median']:<10.2f} "
                f"{data['min']:<10.2f} {data['max']:<10.2f} {data['p90']:<10.2f} "
                f"{data['total']:<12.2f} {data['std_dev']:<10.2f}"
            )

    def reset(self, func_name=None):
        """Reset profiling statistics"""
        with self.lock:
            if func_name:
                if func_name in self.stats:
                    del self.stats[func_name]
            else:
                self.stats = {}


class MockShardIndex:
    """Mock implementation of the ShardIndex class"""

    def __init__(self, shard_sizes):
        self.shard_sizes = shard_sizes

    def find_shard(self, config_name, page_number):
        """Find a shard for a given config and page number"""
        if config_name not in self.shard_sizes:
            raise ValueError(f"Config {config_name} not found in shard sizes")

        config_data = self.shard_sizes[config_name]
        shards = config_data.get("shards", [])

        if not shards:
            raise ValueError(f"No shards found for config {config_name}")

        # Find the appropriate shard based on page number
        total_rows = config_data.get("total_rows", 0)
        rows_per_page = 100  # Assuming 100 rows per page

        # Calculate which shard contains this page
        total_pages = total_rows // rows_per_page if rows_per_page else 1
        if total_pages == 0:
            raise ValueError(f"Config {config_name} has no pages")

        # Normalize page number to be within range
        normalized_page = page_number % total_pages

        # Find the shard that contains this page
        current_row = 0
        for shard in shards:
            shard_rows = shard.get("num_rows", 0)
            shard_pages = shard_rows // rows_per_page if rows_per_page else 1

            if normalized_page < current_row + shard_pages:
                # This shard contains the page
                shard_offset = (normalized_page - current_row) * rows_per_page
                return shard, shard_offset, 0

            current_row += shard_pages

        # If we get here, something went wrong
        raise ValueError(
            f"Could not find shard for page {page_number} in config {config_name}"
        )


class MockTokenizer:
    """Mock tokenizer for benchmarking"""

    def __init__(self):
        self.eos_token_id = 1  # Mock EOS token ID

    def __call__(
        self,
        texts,
        padding=False,
        truncation=True,
        max_length=2048,
        return_tensors=None,
    ):
        """Mock tokenization method with realistic performance characteristics"""
        input_ids = []

        for text in texts:
            # Simple mock tokenization based on text length
            if not text:
                tokens = []
            else:
                # Generate pseudo-random tokens based on text content
                text_hash = hash(text) & 0xFFFFFFFF
                rng = np.random.RandomState(text_hash)

                # Token length is roughly proportional to text length (typical tokenization ratio)
                token_len = (
                    min(len(text) // 4 + 1, max_length)
                    if max_length
                    else len(text) // 4 + 1
                )
                tokens = rng.randint(1, 50000, size=token_len).tolist()

            input_ids.append(tokens)

        return {"input_ids": input_ids}


# Create profiler
_profiler = TimerProfiler("R2DatasetBenchmark")


class R2DatasetBenchmark:
    """
    Benchmarking class for R2DatasetLoader methods.

    This class simulates the flow of _process_page and read_row_group methods
    from the R2DatasetLoader class, allowing for performance benchmarking with
    different parquet files.
    """

    MAX_CONCURRENT_REQUESTS = 32
    READ_BUFFER_SIZE = 32 * 1024 * 1024  # 32MB

    def __init__(
        self,
        parquet_files,
        metadata_file=None,
        sequence_length=2048,
        batch_size=32,
        use_s3=False,
        s3_config=None,
        num_rows_per_page=100,
        debug=False,
    ):
        self.parquet_files = parquet_files
        self.metadata_file = metadata_file
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.use_s3 = use_s3
        self.s3_config = s3_config
        self.num_rows_per_page = num_rows_per_page
        self.debug = debug

        # State initialization
        self.shard_sizes = None
        self.metadata_config = None
        self.shard_index = None
        self._token_cache = {}
        self._parquet_cache = {}
        self._fs_cache = {}
        self._executor = None
        self._fs_lock = threading.Lock()
        self.mock_tokenizer = MockTokenizer()

    def debug_log(self, message):
        """Log debug messages if debug mode is enabled"""
        if self.debug:
            print(f"[DEBUG] {message}")

    @contextmanager
    def timer(self, operation):
        """Context manager for timing operations"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            self.debug_log(f"{operation} took {elapsed_time:.2f}ms")

    def get_executor(self):
        """Get or create thread pool executor"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.MAX_CONCURRENT_REQUESTS,
                thread_name_prefix="R2Benchmark",
            )
        return self._executor

    @_profiler.profile("_get_fs")
    def _get_fs(self, account_id=None):
        """Get S3 filesystem instance"""
        if not self.use_s3:
            return None

        with self._fs_lock:
            cache_key = account_id or "default"

            if cache_key in self._fs_cache:
                return self._fs_cache[cache_key]

            # Use provided S3 config
            if not self.s3_config:
                raise ValueError("S3 configuration required for S3 access")

            credentials = self.s3_config.get("credentials", {}).get("read", {})
            if not credentials:
                raise ValueError("S3 credentials not found in configuration")

            # Create filesystem
            fs = s3fs.S3FileSystem(
                key=credentials.get("access_key_id"),
                secret=credentials.get("secret_access_key"),
                client_kwargs={
                    "endpoint_url": self.s3_config.get("endpoint_url")
                    or f"https://{account_id or self.s3_config.get('account_id')}.r2.cloudflarestorage.com",
                    "region_name": self.s3_config.get("region_name", "auto"),
                },
                config_kwargs={
                    "tcp_keepalive": True,
                    "max_pool_connections": 50,
                    "connect_timeout": 5,
                    "read_timeout": 10,
                    "retries": {"max_attempts": 3},
                },
                max_concurrency=self.MAX_CONCURRENT_REQUESTS,
                use_listings_cache=True,
                skip_instance_cache=False,
                default_block_size=self.READ_BUFFER_SIZE,
                default_cache_type="readahead",
            )

            self._fs_cache[cache_key] = fs
            return fs

    @_profiler.profile("load_metadata")
    async def load_metadata(self):
        """Load or generate metadata for benchmarking"""
        if self.metadata_file and os.path.exists(self.metadata_file):
            self.debug_log(f"Loading metadata from file: {self.metadata_file}")

            try:
                with open(self.metadata_file, "r") as f:
                    if self.metadata_file.endswith(".json"):
                        self.shard_sizes = json.load(f)
                    elif self.metadata_file.endswith((".yaml", ".yml")):
                        import yaml

                        self.shard_sizes = yaml.safe_load(f)
                    else:
                        raise ValueError(
                            f"Unsupported metadata file format: {self.metadata_file}"
                        )

                self.debug_log(f"Loaded metadata with {len(self.shard_sizes)} configs")

                # Create shard index
                self.shard_index = MockShardIndex(self.shard_sizes)
                return self.shard_sizes

            except Exception as e:
                print(f"Error loading metadata: {e}")
                print("Generating mock metadata instead")

        # Generate mock metadata
        self.debug_log("Generating mock metadata")
        self.shard_sizes = {}

        for i, parquet_file in enumerate(self.parquet_files):
            config_name = f"config_{i}"

            try:
                # Get parquet file info
                pf_data = await self._get_parquet(parquet_file)
                num_rows = pf_data["parquet"].metadata.num_rows
                num_row_groups = pf_data["parquet"].num_row_groups

                self.debug_log(
                    f"Parquet file {parquet_file}: {num_rows} rows, {num_row_groups} row groups"
                )

                # Create shard entry
                self.shard_sizes[config_name] = {
                    "total_rows": num_rows,
                    "split": "train",
                    "shards": [
                        {
                            "path": parquet_file,
                            "num_rows": num_rows,
                            "row_groups": num_row_groups,
                        }
                    ],
                }

            except Exception as e:
                print(f"Error inspecting parquet file {parquet_file}: {e}")
                # Create a mock entry with sensible defaults
                self.shard_sizes[config_name] = {
                    "total_rows": 10000,
                    "split": "train",
                    "shards": [
                        {"path": parquet_file, "num_rows": 10000, "row_groups": 10}
                    ],
                }

        # Create shard index
        self.shard_index = MockShardIndex(self.shard_sizes)
        return self.shard_sizes

    @_profiler.profile("run_benchmark")
    async def run_benchmark(self, num_iterations=5, page_offsets=None):
        """Run the full benchmark suite"""
        # Load metadata
        await self.load_metadata()

        # Generate page offsets if not provided
        if page_offsets is None:
            page_offsets = [i * 100 for i in range(num_iterations)]

        results = {}
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)

        # Run benchmark for each config
        for config_name, config_data in self.shard_sizes.items():
            print(f"\nBenchmarking config: {config_name}")

            config_results = {
                "process_page_times": [],
                "read_row_group_times": [],
                "tokenize_times": [],
                "parquet_times": [],
                "total_times": [],
                "token_counts": [],
                "io_times": [],
                "processing_times": [],
            }

            for i in range(min(num_iterations, len(page_offsets))):
                page_number = page_offsets[i]
                print(f"  Iteration {i + 1}/{num_iterations} (page: {page_number})")

                # Create page info
                page = (config_name, page_number, "train")

                # Time the full operation
                start_time = time.perf_counter()

                try:
                    # Process the page
                    tokens = await self._process_page(page, semaphore)

                    # Calculate elapsed time
                    end_time = time.perf_counter()
                    elapsed_ms = (end_time - start_time) * 1000

                    # Store results
                    config_results["total_times"].append(elapsed_ms)
                    config_results["token_counts"].append(len(tokens) if tokens else 0)

                    # Get profiler stats for specific functions
                    stats = _profiler.get_stats()

                    # Extract timing data from profiler
                    for metric, stat_key in [
                        ("process_page_times", "_process_page"),
                        ("read_row_group_times", "read_row_group"),
                        ("tokenize_times", "_batch_tokenize"),
                        ("parquet_times", "_get_parquet"),
                    ]:
                        if stat_key in stats and stats[stat_key]["count"] > 0:
                            calls = (
                                stats[stat_key]["calls"]
                                if "calls" in stats[stat_key]
                                else []
                            )
                            if calls and i < len(calls):
                                config_results[metric].append(calls[-1]["elapsed_ms"])

                    # Calculate I/O vs processing time
                    io_time = 0
                    if "read_row_group" in stats and "calls" in stats["read_row_group"]:
                        io_time += (
                            stats["read_row_group"]["calls"][-1]["elapsed_ms"]
                            if stats["read_row_group"]["calls"]
                            else 0
                        )
                    if "_get_parquet" in stats and "calls" in stats["_get_parquet"]:
                        io_time += (
                            stats["_get_parquet"]["calls"][-1]["elapsed_ms"]
                            if stats["_get_parquet"]["calls"]
                            else 0
                        )

                    config_results["io_times"].append(io_time)
                    config_results["processing_times"].append(elapsed_ms - io_time)

                    # Log iteration results
                    io_percent = (io_time / elapsed_ms * 100) if elapsed_ms > 0 else 0
                    processing_percent = 100 - io_percent

                    print(
                        f"    Total: {elapsed_ms:.2f}ms (I/O: {io_time:.2f}ms [{io_percent:.1f}%], "
                        f"Processing: {elapsed_ms - io_time:.2f}ms [{processing_percent:.1f}%]), "
                        f"Tokens: {len(tokens) if tokens else 0}"
                    )

                except Exception as e:
                    print(f"    Error in iteration {i + 1}: {e}")

            # Store results for this config
            results[config_name] = config_results

        return results

    @_profiler.profile("_process_page")
    async def _process_page(self, page, sem):
        """Process a page of data (simulating R2DatasetLoader._process_page)"""
        async with sem:
            config_name, page_number, split = page
            cache_key = f"{config_name}:{page_number}"

            # Check cache
            if cache_key in self._token_cache:
                self.debug_log(f"Cache hit for {cache_key}")
                return self._token_cache[cache_key]

            try:
                # Find shard for this page
                with self.timer("find_shard"):
                    chosen_shard, shard_offset, _ = self.shard_index.find_shard(
                        config_name, page_number
                    )

                self.debug_log(
                    f"Found shard: {chosen_shard['path']}, offset: {shard_offset}"
                )

                # Get parquet file
                with self.timer("get_parquet"):
                    pf_data = await self._get_parquet(chosen_shard["path"])

                # Read row group
                with self.timer("read_row_group"):
                    table = await self.read_row_group(
                        pf_data, chosen_shard, shard_offset
                    )

                # Check for text column
                if "text" not in table.column_names:
                    available_columns = ", ".join(table.column_names)
                    raise ValueError(
                        f"Text column not found. Available columns: {available_columns}"
                    )

                # Extract text data
                with self.timer("extract_texts"):
                    rows_per_group = chosen_shard["num_rows"] // max(
                        pf_data["parquet"].num_row_groups, 1
                    )
                    start_idx = shard_offset % max(rows_per_group, 1)

                    text_column = table["text"]
                    end_idx = min(start_idx + self.num_rows_per_page, len(text_column))

                    texts = text_column.to_pylist()[start_idx:end_idx]
                    self.debug_log(
                        f"Extracted {len(texts)} texts from offset {start_idx}"
                    )

                # Tokenize texts
                with self.timer("tokenize"):
                    all_tokens = await self._batch_tokenize(texts)

                # Cache results
                self._token_cache[cache_key] = all_tokens
                return all_tokens

            except Exception as e:
                print(f"Error processing page {page}: {e}")
                raise

    @_profiler.profile("_get_parquet")
    async def _get_parquet(self, path):
        """Get a parquet file, either from cache or by opening it"""
        # Check cache
        pf_data = self._parquet_cache.get(path)
        if pf_data:
            self.debug_log(f"Cache hit for parquet file: {path}")
            return pf_data

        self.debug_log(f"Opening parquet file: {path}")

        try:
            # Get file
            with self.timer("_get_parquet_file"):
                pf_data = await self._get_parquet_file(path)

            # Cache for future use
            self._parquet_cache[path] = pf_data
            return pf_data

        except Exception as e:
            print(f"Failed to open parquet file {path}: {e}")
            raise

    @_profiler.profile("_get_parquet_file")
    async def _get_parquet_file(self, path):
        """Open a parquet file from local path or S3"""
        try:
            if self.use_s3:
                fs = self._get_fs()
                f = fs.open(path, "rb", buffer_size=self.READ_BUFFER_SIZE)
            else:
                f = open(path, "rb")

            # Run this in the executor to avoid blocking
            def _open_parquet():
                return pq.ParquetFile(
                    f,
                    memory_map=False,
                    pre_buffer=True,
                    buffer_size=self.READ_BUFFER_SIZE,
                )

            executor = self.get_executor()
            pf = await asyncio.get_event_loop().run_in_executor(executor, _open_parquet)

            return {
                "file": f,
                "parquet": pf,
                "lock": threading.Lock(),
            }

        except Exception as e:
            print(f"Error opening parquet file {path}: {e}")
            raise

    @_profiler.profile("read_row_group")
    async def read_row_group(self, pf_data, chosen_shard, shard_offset):
        """Read a row group from a parquet file (simulating R2DatasetLoader.read_row_group)"""

        def _read_group():
            with pf_data["lock"]:
                # Calculate row group index
                num_row_groups = pf_data["parquet"].num_row_groups
                rows_per_group = chosen_shard["num_rows"] // max(num_row_groups, 1)
                group_index = min(
                    shard_offset // max(rows_per_group, 1), num_row_groups - 1
                )

                self.debug_log(f"Reading row group {group_index} of {num_row_groups}")

                # Read the row group
                try:
                    # Try to read 'text' column if available
                    return pf_data["parquet"].read_row_group(
                        group_index,
                        columns=["text"],
                        use_threads=True,
                        use_pandas_metadata=False,
                    )
                except (KeyError, ValueError) as e:
                    self.debug_log(f"Error reading 'text' column: {e}")
                    # If 'text' column not found, read all columns and handle later
                    return pf_data["parquet"].read_row_group(
                        group_index,
                        use_threads=True,
                        use_pandas_metadata=False,
                    )

        executor = self.get_executor()
        return await asyncio.get_event_loop().run_in_executor(executor, _read_group)

    @_profiler.profile("_batch_tokenize")
    async def _batch_tokenize(self, texts):
        """Tokenize a batch of texts (simulating R2DatasetLoader._batch_tokenize)"""

        def _tokenize_batch():
            all_tokens = []

            chunk_size = 128  # Process in chunks for better efficiency
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i : i + chunk_size]

                # Tokenize
                batch_tokens = self.mock_tokenizer(
                    chunk,
                    padding=False,
                    truncation=True,
                    max_length=self.sequence_length,
                    return_tensors=None,
                )

                # Process results
                for tokens in batch_tokens["input_ids"]:
                    if tokens:
                        all_tokens.extend(tokens)
                        if tokens[-1] != self.mock_tokenizer.eos_token_id:
                            all_tokens.append(self.mock_tokenizer.eos_token_id)

            return all_tokens

        executor = self.get_executor()
        return await asyncio.get_event_loop().run_in_executor(executor, _tokenize_batch)

    def __del__(self):
        """Clean up resources"""
        for pf_data in self._parquet_cache.values():
            pf_data["file"].close()

        self._parquet_cache.clear()
        self._token_cache.clear()


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Benchmark R2DatasetLoader methods")

    # Parquet file options
    parser.add_argument(
        "--parquet-files",
        nargs="+",
        required=True,
        help="Paths to parquet files for benchmarking",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata file (JSON or YAML)",
    )

    # Benchmark options
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of iterations per file"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Sequence length for tokenization",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for processing"
    )
    parser.add_argument(
        "--rows-per-page", type=int, default=100, help="Number of rows per page"
    )

    # S3 options
    parser.add_argument(
        "--use-s3", action="store_true", help="Use S3 for accessing parquet files"
    )
    parser.add_argument(
        "--s3-config",
        type=str,
        default=None,
        help="Path to S3 configuration file (JSON)",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for benchmark results",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Load S3 configuration if needed
    s3_config = None
    if args.use_s3:
        if args.s3_config:
            try:
                with open(args.s3_config, "r") as f:
                    s3_config = json.load(f)
            except Exception as e:
                print(f"Error loading S3 config: {e}")
                print("Falling back to local files")
                args.use_s3 = False
        else:
            print("S3 configuration file not provided, falling back to local files")
            args.use_s3 = False

    # Validate parquet files
    valid_files = []
    if not args.use_s3:
        for file_path in args.parquet_files:
            if not os.path.exists(file_path):
                print(f"Warning: Parquet file not found: {file_path}")
                continue
            valid_files.append(file_path)
    else:
        valid_files = args.parquet_files

    if not valid_files:
        print("Error: No valid parquet files provided")
        return

    # Generate random page offsets for benchmarking
    np.random.seed(42)  # For reproducibility
    page_offsets = np.random.randint(0, 1000, args.iterations).tolist()

    print(
        f"Starting benchmark with {len(valid_files)} parquet files, {args.iterations} iterations per file"
    )
    print(
        f"Configuration: sequence_length={args.sequence_length}, rows_per_page={args.rows_per_page}"
    )

    # Create benchmark instance
    benchmark = R2DatasetBenchmark(
        parquet_files=valid_files,
        metadata_file=args.metadata,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        use_s3=args.use_s3,
        s3_config=s3_config,
        num_rows_per_page=args.rows_per_page,
        debug=args.debug,
    )

    # Run benchmark
    start_time = time.perf_counter()
    results = await benchmark.run_benchmark(args.iterations, page_offsets)
    total_time = time.perf_counter() - start_time

    # Log profiling summary
    _profiler.log_summary()

    # Save results
    output_path = Path(args.output)

    # Calculate summary statistics
    summary = {}

    for config_name, config_data in results.items():
        summary[config_name] = {}

        for metric, values in config_data.items():
            if values:
                summary[config_name][f"mean_{metric}"] = sum(values) / len(values)
                summary[config_name][f"median_{metric}"] = np.median(values)
                summary[config_name][f"min_{metric}"] = min(values)
                summary[config_name][f"max_{metric}"] = max(values)

                if metric not in [
                    "token_counts"
                ]:  # Only calculate percentiles for time metrics
                    summary[config_name][f"p90_{metric}"] = np.percentile(values, 90)
                    summary[config_name][f"p99_{metric}"] = np.percentile(values, 99)
                    summary[config_name][f"std_{metric}"] = np.std(values)

    # Save detailed results
    with open(output_path, "w") as f:
        json.dump(
            {
                "summary": summary,
                "raw_data": {
                    config_name: {
                        k: [float(v) for v in values]
                        for k, values in config_data.items()
                    }
                    for config_name, config_data in results.items()
                },
                "profiling": _profiler.get_stats(),
                "configuration": {
                    "parquet_files": valid_files,
                    "iterations": args.iterations,
                    "sequence_length": args.sequence_length,
                    "batch_size": args.batch_size,
                    "rows_per_page": args.rows_per_page,
                    "use_s3": args.use_s3,
                    "timestamp": time.time(),
                    "duration": total_time,
                },
            },
            f,
            indent=2,
        )

    # Print summary table
    print(f"\nBenchmark complete in {total_time:.2f}s! Results saved to {output_path}")
    print("\n=== Summary Results (times in ms) ===")

    headers = [
        "Config",
        "Total",
        "Process Page",
        "Row Group",
        "I/O %",
        "Processing %",
        "Tokens",
    ]
    print(
        f"{headers[0]:<20} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<8} {headers[5]:<12} {headers[6]:<8}"
    )
    print("-" * 85)

    for config_name, config_summary in summary.items():
        total = config_summary.get("mean_total_times", 0)
        io = config_summary.get("mean_io_times", 0)
        io_percent = (io / total * 100) if total > 0 else 0
        processing_percent = 100 - io_percent

        print(
            f"{config_name:<20} "
            f"{total:<12.2f} "
            f"{config_summary.get('mean_process_page_times', 0):<12.2f} "
            f"{config_summary.get('mean_read_row_group_times', 0):<12.2f} "
            f"{io_percent:<8.1f} "
            f"{processing_percent:<12.1f} "
            f"{config_summary.get('mean_token_counts', 0):<8.0f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
