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


import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import s3fs
import yaml

from tplr import logger
from tplr.config import BUCKET_SECRETS
from tplr.dataset import DatasetLoader
from tplr.profilers import get_shard_profiler, get_timer_profiler
from tplr.shard_index import ShardIndex

_timer_profiler = get_timer_profiler("R2DatasetLoader")

pyarrow.set_io_thread_count(os.cpu_count())


class R2DatasetLoader(DatasetLoader):
    """
    A drop-in replacement for DatasetLoader that reads Parquet files from Cloudflare R2 storage.

    This loader handles:
    - Reading and caching metadata from R2 storage
    - Loading data from Parquet files in parallel
    - Tokenizing and batching text data
    - Managing sequence padding and packing

    The loader uses the same credentials logic as comms.py/config.py for R2 access.

    Attributes:
        rows_base_url (str): Base URL for row data (unused)
        size_base_url (str): Base URL for size data (unused)
        _configs_data_cache (dict): Cache for dataset configuration data
        DATASET_SUBFOLDER (str): Subfolder name in R2 bucket containing dataset
        CF_REGION_NAME (str): Cloudflare region name
        _shard_sizes (dict): Cache for shard size metadata
        _metadata_config (dict): Cache for dataset metadata configuration
        _local_cache_dir (Path): Local directory for caching metadata files
    """

    rows_base_url: str = ""  # Empty string to match parent class type
    size_base_url: str = ""  # Empty string to match parent class type
    _configs_data_cache = None
    DATASET_SUBFOLDER = "mlfoundations-dclm-baseline-1.0-parquet-optimized"
    CF_REGION_NAME = "enam"

    # Cache for metadata
    _shard_sizes: dict | None = None
    _metadata_config: dict | None = None
    _local_cache_dir = Path(".cache/tplr")

    _shard_index: "ShardIndex | None" = None

    # Add class-level caching for filesystem and tokenizer results
    _fs_instance = None
    _tokenized_cache = {}
    _buffer_size = 1024 * 1024  # 1MB buffer for reading

    # Class-level caches
    _metadata_cache = {}  # Cache for metadata by config
    _parquet_cache = {}  # Cache ParquetFile objects
    _fs = None  # Single filesystem instance

    # Static configuration
    PREFETCH_SIZE = 3  # Number of pages to prefetch
    MAX_CONCURRENT_REQUESTS = 32  # Number of concurrent requests to R2
    BATCH_SIZE = 128  # Increased batch size for tokenization
    READ_BUFFER_SIZE = 32 * 1024 * 1024  # 32MB read buffer

    # Class-level caches with size limits
    _metadata_cache = {}
    _parquet_cache = {}  # Cache for ParquetFile objects
    _token_cache = {}  # Cache for tokenized results
    _fs = None
    _prefetch_queue = None

    _round_robin_index = 0  # global counter for dataset round-robin selection
    _fs_cache = {}  # maps account_id to a cached s3fs.S3FileSystem
    _fs_lock = threading.Lock()  # lock for fs cache and round robin
    _executor = None  # ThreadPoolExecutor for CPU-bound tasks

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer=None,
        pack_samples=True,
    ):
        """
        Initialize the dataset loader.

        Args:
            batch_size (int, optional): Size of batches to return
            sequence_length (int, optional): Length of sequences to generate
            num_pages (int, optional): Number of pages to load
            tokenizer: Tokenizer instance to use
            pack_samples (bool): Whether to pack samples without padding
        """
        # Validate tokenizer is not None before calling super().__init__
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")

        super().__init__(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_pages=num_pages,
            tokenizer=tokenizer,
            pack_samples=pack_samples,
        )

        # Additional buffers from parent class
        self.used_buffer = []
        self.padded_buffer = []

        # Prefetch setup
        self._prefetch_task = None
        self._current_batch = None
        self._next_batch = None
        self._prefetch_queue = asyncio.Queue(maxsize=self.PREFETCH_SIZE)

        # Add missing attributes
        self.fs = self._get_fs()

    @classmethod
    def get_executor(cls):
        """Get or create a shared ThreadPoolExecutor"""
        if cls._executor is None:
            cls._executor = ThreadPoolExecutor(
                max_workers=cls.MAX_CONCURRENT_REQUESTS,
                thread_name_prefix="R2DatasetLoader",
            )
        return cls._executor

    def _get_pad_size(self, input_ids=None):
        """
        Calculate padding size needed for a sequence.

        Args:
            input_ids: Optional input IDs (for compatibility with parent class)

        Returns:
            int: Number of padding tokens needed
        """
        if self.pack_samples:
            return 1

        if self.sequence_length is None:
            return 0

        sample_size = len(self.token_buffer)
        remainder = sample_size % self.sequence_length
        pad_size = self.sequence_length - remainder
        return pad_size % self.sequence_length

    def _get_shard_path(self, shard_key: str) -> str:
        """Get full R2 path for a shard"""
        dataset_config = BUCKET_SECRETS["dataset"]
        if "multiple" in dataset_config:
            bucket_name = dataset_config["multiple"][0]["name"]
        else:
            bucket_name = dataset_config["name"]
        return f"{bucket_name}/{self.DATASET_SUBFOLDER}/{shard_key}"

    def _refill_padded_buffer(self):
        """Refill buffer with padded sequences."""
        # Check for None values early
        if self.sequence_length is None or self.batch_size is None:
            return

        # Get padding size first
        pad_size = self._get_pad_size()
        if self.padded_buffer and len(self.padded_buffer) >= pad_size:
            return

        # Get tokens from buffer
        if not self.token_buffer:
            return

        # Determine EOS token ID with proper attribute checking
        eos_token_id = None
        if self.tokenizer and hasattr(self.tokenizer, "eos_token_id"):
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        # Pad with EOS tokens if available, otherwise use pad tokens
        if eos_token_id is not None:
            pad_token_id = eos_token_id
        elif self.tokenizer and hasattr(self.tokenizer, "pad_token_id"):
            pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        else:
            pad_token_id = 0

        # Create padded buffer
        remaining_tokens = self.token_buffer[: pad_size - len(self.padded_buffer)]
        self.padded_buffer.extend(remaining_tokens)

        # Pad to required size
        tokens_needed = pad_size - len(self.padded_buffer)
        if tokens_needed > 0:
            self.padded_buffer.extend([pad_token_id] * tokens_needed)

        # Remove processed tokens from buffer
        self.token_buffer = self.token_buffer[len(remaining_tokens) :]

    @staticmethod
    async def fetch_dataset_configs() -> dict:
        """
        Load dataset configurations from cached metadata and shard sizes.
        """
        if R2DatasetLoader._configs_data_cache is not None:
            return R2DatasetLoader._configs_data_cache

        try:
            # Use _load_r2_metadata to get both metadata and shard sizes
            shard_sizes, metadata_config, _ = await R2DatasetLoader._load_r2_metadata()

            # Build configs data from both files
            configs_data = {}
            for config in metadata_config.get("configs", []):
                config_name = config.get("config_name")
                if config_name == "default":
                    continue

                # Get shard info from shard_sizes
                shard_info = shard_sizes.get(config_name, {})
                if not shard_info:
                    continue

                configs_data[config_name] = {
                    "num_rows": shard_info.get("total_rows", 0),
                    "split": shard_info.get("split", "train"),
                    "shards": shard_info.get("shards", []),
                }

            R2DatasetLoader._configs_data_cache = configs_data
            return configs_data

        except Exception as e:
            logger.error(f"Error loading dataset configs: {e}")
            raise

    @staticmethod
    @_timer_profiler.profile("next_pages")
    async def next_pages(
        offset: int, n_pages: int, seed: str, num_rows_per_page: int = 100
    ) -> list:
        """Get next n_pages random pages starting from offset."""
        configs_data = await R2DatasetLoader.fetch_dataset_configs()

        # Create RNG with same method as DatasetLoader
        rng = np.random.default_rng(hash(seed) & 0xFFFFFFFF)

        # Safely advance the bit generator if supported
        try:
            # Use getattr to avoid direct attribute access
            advance_method = getattr(rng.bit_generator, "advance", None)
            if advance_method is not None:
                advance_method(offset)  # Skip ahead by offset
        except (AttributeError, TypeError):
            # Fallback: manually advance by generating and discarding values
            for _ in range(offset):
                rng.random()

        # Sort config keys for consistent ordering
        sorted_keys = sorted(configs_data.keys())

        result = []
        for _ in range(n_pages):
            config = rng.choice(sorted_keys)
            choice = rng.integers(
                0, configs_data[config]["num_rows"] - 1 - num_rows_per_page
            )
            result.append((str(config), int(choice), configs_data[config]["split"]))

        return result

    @staticmethod
    @_timer_profiler.profile("create")
    async def create(
        batch_size, sequence_length, pages_info, tokenizer, pack_samples=True
    ):
        """Create loader with proper initialization"""
        loader = R2DatasetLoader(
            batch_size=batch_size,
            sequence_length=sequence_length,
            tokenizer=tokenizer,
            pack_samples=pack_samples,
        )

        loader.buffer = []
        loader.pages = pages_info.copy()

        await loader._load_r2_metadata()

        sem = asyncio.Semaphore(loader.MAX_CONCURRENT_REQUESTS)

        tasks = [loader._process_page(page, sem) for page in loader.pages]
        tasks_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in tasks_results:
            if isinstance(result, list):
                loader.buffer.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Page processing error: {result}")

        return loader

    @staticmethod
    @_timer_profiler.profile("_load_r2_metadata")
    async def _load_r2_metadata():
        """
        Loads and caches metadata from R2 storage.

        Downloads shard sizes and metadata config files if not cached locally.

        Returns:
            tuple: (shard_sizes dict, metadata_config dict)

        Raises:
            Exception: If metadata loading fails
        """
        if R2DatasetLoader._shard_sizes is not None:
            return (
                R2DatasetLoader._shard_sizes,
                R2DatasetLoader._metadata_config,
                R2DatasetLoader._shard_index,
            )

        fs = R2DatasetLoader._get_fs()
        cache_dir = R2DatasetLoader._local_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Define R2 and local paths
        bucket_name = (
            BUCKET_SECRETS["dataset"].get("name")
            if "name" in BUCKET_SECRETS["dataset"]
            else BUCKET_SECRETS["dataset"]["multiple"][0]["name"]
        )
        bucket_path = f"{bucket_name}/{R2DatasetLoader.DATASET_SUBFOLDER}"
        r2_paths = {
            "shard_sizes": f"{bucket_path}/_shard_sizes.json",
            "metadata": f"{bucket_path}/_metadata.yaml",
        }
        local_paths = {
            "shard_sizes": cache_dir / "shard_sizes.json",
            "metadata": cache_dir / "metadata.yaml",
        }

        try:
            # Download and load shard sizes
            if not local_paths["shard_sizes"].exists():
                logger.info("Downloading shard sizes from R2...")
                fs.get(r2_paths["shard_sizes"], str(local_paths["shard_sizes"]))
            with open(local_paths["shard_sizes"]) as f:
                R2DatasetLoader._shard_sizes = json.load(f)

            # Download and load metadata config
            if not local_paths["metadata"].exists():
                logger.info("Downloading metadata config from R2...")
                fs.get(r2_paths["metadata"], str(local_paths["metadata"]))
            with open(local_paths["metadata"]) as f:
                R2DatasetLoader._metadata_config = yaml.safe_load(f)

            R2DatasetLoader._shard_index = ShardIndex(R2DatasetLoader._shard_sizes)

            return (
                R2DatasetLoader._shard_sizes,
                R2DatasetLoader._metadata_config,
                R2DatasetLoader._shard_index,
            )

        except Exception as e:
            logger.error(f"Failed to load R2 metadata: {e}")
            raise

    @staticmethod
    @_timer_profiler.profile("_get_fs")
    def _get_fs():
        dataset_config = BUCKET_SECRETS["dataset"]
        # For debugging: log the full dataset configuration to check if 'multiple' is present
        logger.debug(f"Dataset config loaded: {dataset_config}")

        with R2DatasetLoader._fs_lock:
            # Pick config in round robin if multiple endpoints are supplied
            if "multiple" in dataset_config:
                configs = dataset_config["multiple"]
                idx = R2DatasetLoader._round_robin_index % len(configs)
                selected_config = configs[idx]
                R2DatasetLoader._round_robin_index += 1
            else:
                selected_config = dataset_config

            # Log the selected bucket name for round robin tracing (should show e.g. "dataset-bucket-1" then "dataset-bucket-2")
            logger.debug(
                f"Using dataset bucket: {selected_config.get('name', 'default')}"
            )

            fs_cache_key = selected_config["account_id"]

            if fs_cache_key not in R2DatasetLoader._fs_cache:
                read_credentials = selected_config["credentials"]["read"]
                fs = s3fs.S3FileSystem(
                    # asynchronous=True,
                    key=read_credentials["access_key_id"],
                    secret=read_credentials["secret_access_key"],
                    client_kwargs={
                        "endpoint_url": f"https://{selected_config['account_id']}.r2.cloudflarestorage.com",
                        "region_name": R2DatasetLoader.CF_REGION_NAME,
                    },
                    config_kwargs={
                        "tcp_keepalive": True,
                        "max_pool_connections": 50,
                        "connect_timeout": 5,
                        "read_timeout": 10,
                        "retries": {"max_attempts": 3},
                    },
                    max_concurrency=R2DatasetLoader.MAX_CONCURRENT_REQUESTS,
                    use_listings_cache=True,
                    skip_instance_cache=False,
                    default_block_size=R2DatasetLoader.READ_BUFFER_SIZE,
                    default_cache_type="readahead",
                )
                R2DatasetLoader._fs_cache[fs_cache_key] = fs
            return R2DatasetLoader._fs_cache[fs_cache_key]

    async def _get_next_page(self):
        """Get next page from the queue"""
        if not self.pages:
            return None
        return self.pages.pop(0)

    async def _prefetch_pages(self):
        """Background task to prefetch pages"""
        try:
            while True:
                page = await self._get_next_page()
                if page is None:
                    break
                await self._prefetch_queue.put(page)  # type: ignore
        except Exception as e:
            logger.error(f"Prefetch error: {e}")
        finally:
            await self._prefetch_queue.put(None)  # type: ignore # Signal completion

    @_timer_profiler.profile("_process_page")
    async def _process_page(self, page, sem):
        """Process page with deterministic shard selection"""
        async with sem:
            config_name, page_number, split = page
            cache_key = f"{config_name}:{page_number}"

            try:
                if cache_key in self._token_cache:
                    return self._token_cache[cache_key]

                metadata = self._metadata_cache.get(config_name)
                if not metadata:
                    shard_sizes, _, _ = await self._load_r2_metadata()
                    metadata = shard_sizes[config_name]
                    self._metadata_cache[config_name] = metadata

                try:
                    # Check if _shard_index is not None before calling find_shard
                    if R2DatasetLoader._shard_index is None:
                        raise ValueError("Shard index not initialized")

                    chosen_shard, shard_offset, _ = (
                        R2DatasetLoader._shard_index.find_shard(
                            config_name, page_number
                        )
                    )
                except ValueError as e:
                    raise ValueError(
                        f"Could not find shard for page {page_number}: {e}"
                    )

                pf_data = await self._get_parquet(chosen_shard["path"])

                table = await self.read_row_group(pf_data, chosen_shard, shard_offset)

                start_idx = shard_offset % (
                    chosen_shard["num_rows"] // pf_data["parquet"].num_row_groups
                )
                texts = table["text"].to_pylist()[
                    start_idx : start_idx + self.num_rows_per_page
                ]

                all_tokens = await self._batch_tokenize(texts)

                self._token_cache[cache_key] = all_tokens
                return all_tokens

            except Exception as e:
                logger.error(f"Error processing page {page}: {e}")
                raise

    @_timer_profiler.profile("_get_parquet")
    async def _get_parquet(self, path: str) -> dict:
        """fetch parquet file handling with connection pooling"""
        pf_data = self._parquet_cache.get(path)
        if pf_data:
            # Check if the cached file is still valid
            if pf_data["file"].closed:
                logger.warning(
                    f"Cached parquet file is closed for {path}, reopening..."
                )
                self._parquet_cache.pop(path, None)
                pf_data = None
            else:
                return pf_data

        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    pf_data = R2DatasetLoader._get_parquet_file(path)
                    self._parquet_cache[path] = pf_data
                    return pf_data
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed to open parquet file {path} with error: {e}. Retrying..."
                        )
                        await asyncio.sleep(2**attempt)  # Exponential backoff
                    else:
                        logger.error(
                            f"Failed to open parquet file {path} after {max_retries} attempts: {e}"
                        )
                        raise

        except Exception as e:
            logger.error(f"Failed to open parquet file {path}: {e}")
            raise

        raise ValueError(f"Failed to get parquet file for {path}")

    @_timer_profiler.profile("read_row_group")
    async def read_row_group(self, pf_data, chosen_shard, shard_offset):
        """row group reading with detailed performance tracking"""
        shard_path = chosen_shard["path"]
        shard_profiler = get_shard_profiler()

        timer_id = shard_profiler.start_read(shard_path, chosen_shard, pf_data)

        def _read_group():
            with pf_data["lock"]:
                # Check if file is still open
                if pf_data["file"].closed:
                    raise IOError(f"Parquet file is closed: {shard_path}")

                num_row_groups = pf_data["parquet"].num_row_groups
                rows_per_group = chosen_shard["num_rows"] // num_row_groups
                group_index = min(shard_offset // rows_per_group, num_row_groups - 1)

                shard_profiler.log_read_details(
                    shard_path,
                    group_index,
                    num_row_groups,
                    shard_offset,
                    rows_per_group,
                )

                return pf_data["parquet"].read_row_group(
                    group_index,
                    columns=["text"],
                    use_threads=False,
                    use_pandas_metadata=False,
                )

        executor = self.get_executor()

        # Retry logic for closed file handles
        max_retries = 3
        result = None
        for attempt in range(max_retries):
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor, _read_group
                )
                break
            except IOError as e:
                if "Parquet file is closed" in str(e) and attempt < max_retries - 1:
                    logger.warning(
                        f"Parquet file was closed during read, attempting to reopen (attempt {attempt + 1}/{max_retries}): {shard_path}"
                    )
                    # Clear from cache and get fresh file handle
                    self._parquet_cache.pop(shard_path, None)
                    pf_data = await self._get_parquet(shard_path)
                else:
                    raise

        if result is None:
            raise RuntimeError(f"Failed to read row group after {max_retries} attempts")

        elapsed = shard_profiler.end_read(
            timer_id,
            shard_path,
            pf_data["parquet"].num_row_groups,
            chosen_shard["num_rows"] // pf_data["parquet"].num_row_groups,
        )

        shard_profiler.log_read_complete(shard_path, elapsed)

        return result

    @_timer_profiler.profile("_batch_tokenize")
    async def _batch_tokenize(self, texts: list[str]) -> list[int]:
        """
        Tokenizes a batch of texts and concatenates them.
        Returns a flat list of token IDs.
        """
        result = []  # Initialize result

        if self.tokenizer is None:
            return result

        # Tokenize all texts in batch - handle tokenizer safely
        try:
            # Use getattr to check if tokenizer is callable
            tokenizer_call = getattr(self.tokenizer, "__call__", None)
            if tokenizer_call is not None:
                encoded = tokenizer_call(
                    texts,
                    truncation=False,
                    padding=False,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )
            else:
                raise AttributeError("Tokenizer is not callable")
        except (TypeError, AttributeError):
            # Fallback for tokenizers without __call__
            encoded = {"input_ids": []}
            for text in texts:
                try:
                    # Use getattr to safely access encode method
                    encode_method = getattr(self.tokenizer, "encode", None)
                    if encode_method is not None:
                        tokens = encode_method(
                            text,
                            add_special_tokens=False,
                        )
                        encoded["input_ids"].append(tokens)
                    else:
                        encoded["input_ids"].append([])
                except (AttributeError, TypeError):
                    # Last resort: empty token list
                    encoded["input_ids"].append([])

        # Concatenate all sequences with EOS tokens
        for input_ids in encoded["input_ids"]:
            result.extend(input_ids)
            # Safely get EOS token ID
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                result.append(eos_token_id)

        return result

    def __iter__(self):
        """Reset buffers and prepare for iteration"""
        self.buffer = self.used_buffer + self.buffer  # Combine buffers
        self.used_buffer = []  # Reset used buffer
        self.padded_buffer = []  # Reset padded buffer
        self._refill_padded_buffer()  # Initial fill
        return self

    def __next__(self):
        """Get next batch, exactly matching DatasetLoader's logic"""
        batch = []

        # Add null checks for comparison operators
        if self.sequence_length is None or self.batch_size is None:
            raise StopIteration

        while len(self.padded_buffer) >= self.sequence_length:
            # Extract sequence_length tokens
            sequence = self.padded_buffer[: self.sequence_length]
            self.padded_buffer = self.padded_buffer[self.sequence_length :]

            batch.append(sequence)

            # Return batch when we have batch_size sequences
            if len(batch) == self.batch_size:
                self._refill_padded_buffer()  # Refill after creating batch
                return np.stack(batch)

            # Refill if needed
            if len(self.padded_buffer) < self.sequence_length:
                self._refill_padded_buffer()

        # No more complete batches
        if batch:  # Partial batch - should not happen with current logic
            raise StopIteration
        raise StopIteration

    async def _read_parquet_table(self, shard_key: str, start_row: int) -> list[str]:
        """Read and return rows from a parquet file."""
        loop = asyncio.get_event_loop()

        # Add null checks for comparison operators
        if self.batch_size is None or self.sequence_length is None:
            return []

        def _read_chunk() -> list[str]:
            shard_path = self._get_shard_path(shard_key)
            table = pq.read_table(
                shard_path,
                filesystem=self.fs,
                columns=["text"],
            )

            # Validate start_row and batch_size before comparisons
            total_rows = len(table)
            if start_row >= total_rows:
                return []

            # Calculate actual batch size with null check
            batch_size = self.batch_size if self.batch_size is not None else 1
            end_row = min(start_row + batch_size, total_rows)

            chunk_table = table.slice(start_row, end_row - start_row)
            df = chunk_table.to_pandas()

            return df["text"].tolist()

        return await loop.run_in_executor(self.get_executor(), _read_chunk)

    def __del__(self):
        """Cleanup resources"""
        if self._prefetch_task:
            self._prefetch_task.cancel()

        for pf_data in self._parquet_cache.values():
            with pf_data["lock"]:
                if pf_data["file"] and not pf_data["file"].closed:
                    try:
                        pf_data["file"].close()  # type: ignore
                    except Exception as e:
                        logger.debug(f"Error closing parquet file: {e}")

        self._parquet_cache.clear()
        self._token_cache.clear()

    @staticmethod
    @_timer_profiler.profile("_get_parquet_file")
    def _get_parquet_file(shard_path: str) -> dict:
        """Cached parquet file access with metadata"""
        fs = R2DatasetLoader._get_fs()
        shard_profiler = get_shard_profiler()

        try:
            file_info = fs.info(shard_path)
            file_size = file_info.get("Size", file_info.get("size", "unknown"))
        except Exception as e:
            logger.warning(f"Could not get file size for {shard_path}: {e}")
            file_size = "unknown"

        f = fs.open(shard_path, "rb", buffer_size=R2DatasetLoader.READ_BUFFER_SIZE)
        pf = pq.ParquetFile(
            f,
            memory_map=False,
            pre_buffer=True,
            buffer_size=R2DatasetLoader.READ_BUFFER_SIZE,
        )

        # Use shard profiler for consistent logging
        shard_profiler.log_parquet_metadata(
            shard_path=shard_path,
            file_size=file_size,
            num_row_groups=pf.num_row_groups,
            total_rows=pf.metadata.num_rows,
        )

        return {
            "file": f,
            "parquet": pf,
            "lock": threading.Lock(),
            "metadata": {
                "path": shard_path,
                "file_size": file_size,
                "num_row_groups": pf.num_row_groups,
                "total_rows": pf.metadata.num_rows,
                "schema": str(pf.schema),
            },
        }

    @staticmethod
    def _get_tokenized_cache(cache_key: str):
        """Cached tokenization results"""
        return R2DatasetLoader._token_cache.get(cache_key)

    @staticmethod
    def get_profiling_stats():
        """Get timing statistics from the profiler"""
        return _timer_profiler.get_stats()

    @staticmethod
    def log_profiling_summary():
        """Log a summary of all timing statistics"""
        _timer_profiler.log_summary()
        ShardIndex.log_profiling_summary()
        get_shard_profiler().log_analysis()

    @staticmethod
    def reset_profiling_stats(func_name: str = ""):
        """Reset profiling statistics"""
        _timer_profiler.reset(func_name)

    @staticmethod
    def get_shard_performance_stats():
        """Get detailed performance statistics for each shard file"""
        return get_shard_profiler().get_stats()

    @staticmethod
    def log_shard_performance_analysis():
        """Log detailed analysis of shard performance"""
        get_shard_profiler().log_analysis()

    @staticmethod
    def export_shard_performance_data(
        output_file: str = "shard_performance_report.json",
    ):
        """Export shard performance data to a JSON file for external analysis"""
        get_shard_profiler().export_data(output_file)
