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


import json
import yaml
import s3fs
import asyncio
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from functools import lru_cache

from tplr import logger
from tplr.config import BUCKET_SECRETS
from tplr.dataset import DatasetLoader


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

    rows_base_url = None
    size_base_url = None
    _configs_data_cache = None
    DATASET_SUBFOLDER = "HuggingFaceFW_fineweb-edu-score-2"
    CF_REGION_NAME = "enam"

    # Cache for metadata
    _shard_sizes = None
    _metadata_config = None
    _local_cache_dir = Path(".cache/tplr")

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
    MAX_CONCURRENT_REQUESTS = 8  # Increased from 4
    BATCH_SIZE = 128  # Increased batch size for tokenization
    READ_BUFFER_SIZE = 4 * 1024 * 1024  # 4MB read buffer

    # Class-level caches with size limits
    _metadata_cache = {}
    _parquet_cache = {}  # Cache for ParquetFile objects
    _token_cache = {}  # Cache for tokenized results
    _fs = None
    _prefetch_queue = None

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

    def _get_pad_size(self, input_ids):
        """
        Calculate padding size needed for a sequence.

        Args:
            input_ids (list): Token IDs to pad

        Returns:
            int: Number of padding tokens needed
        """
        if self.pack_samples:
            return 1

        sample_size = len(input_ids)
        remainder = sample_size % self.sequence_length
        pad_size = self.sequence_length - remainder
        return pad_size % self.sequence_length

    def _refill_padded_buffer(self):
        """Match DatasetLoader's buffer refill logic exactly"""
        while (
            self.buffer
            and len(self.padded_buffer) < self.sequence_length * self.batch_size
        ):
            try:
                # Find next EOS token
                eos_index = self.buffer.index(self.tokenizer.eos_token_id)

                # Get sequence up to and including EOS
                input_ids = self.buffer[: eos_index + 1]
                self.buffer = self.buffer[eos_index + 1 :]

                # Track used tokens
                self.used_buffer.extend(input_ids)

                # Add to padded buffer without the EOS token
                self.padded_buffer.extend(input_ids[:-1])

                # Add padding using EOS tokens (not pad tokens)
                pad_size = self._get_pad_size(input_ids[:-1])
                self.padded_buffer.extend([self.tokenizer.eos_token_id] * pad_size)

            except ValueError:  # No EOS token found
                if self.buffer:  # Add remaining tokens if any
                    self.padded_buffer.extend(self.buffer)
                    self.used_buffer.extend(self.buffer)
                    self.buffer = []

    @staticmethod
    async def fetch_dataset_configs() -> dict:
        """
        Load dataset configurations from cached metadata and shard sizes.
        """
        if R2DatasetLoader._configs_data_cache is not None:
            return R2DatasetLoader._configs_data_cache

        try:
            # Use _load_r2_metadata to get both metadata and shard sizes
            shard_sizes, metadata_config = await R2DatasetLoader._load_r2_metadata()

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
    async def next_pages(
        offset: int, n_pages: int, seed: str, num_rows_per_page: int = 100
    ) -> list:
        """Get next n_pages random pages starting from offset."""
        configs_data = await R2DatasetLoader.fetch_dataset_configs()

        # Create RNG with same method as DatasetLoader
        rng = np.random.default_rng(hash(seed) & 0xFFFFFFFF)
        rng.bit_generator.advance(offset)  # Skip ahead by offset

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

        # Initialize buffers
        loader.buffer = []
        loader.pages = pages_info.copy()

        # Process all pages first
        sem = asyncio.Semaphore(loader.MAX_CONCURRENT_REQUESTS)
        tasks = [
            asyncio.create_task(loader._process_page(page, sem)) for page in pages_info
        ]

        # Gather all tokens
        results = await asyncio.gather(*tasks)
        for tokens in results:
            if tokens:
                loader.buffer.extend(tokens)

        return loader

    @staticmethod
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
            )

        fs = R2DatasetLoader._get_fs()
        cache_dir = R2DatasetLoader._local_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Define R2 and local paths
        r2_base = (
            f"{BUCKET_SECRETS['dataset']['name']}/{R2DatasetLoader.DATASET_SUBFOLDER}"
        )
        r2_paths = {
            "shard_sizes": f"{r2_base}/_shard_sizes.json",
            "metadata": f"{r2_base}/_metadata.yaml",
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

            return (
                R2DatasetLoader._shard_sizes,
                R2DatasetLoader._metadata_config,
            )

        except Exception as e:
            logger.error(f"Failed to load R2 metadata: {e}")
            raise

    @staticmethod
    def _get_fs():
        if not R2DatasetLoader._fs:
            dataset_config = BUCKET_SECRETS["dataset"]
            read_credentials = dataset_config["credentials"]["read"]  # type: ignore

            R2DatasetLoader._fs = s3fs.S3FileSystem(
                key=read_credentials["access_key_id"],
                secret=read_credentials["secret_access_key"],
                client_kwargs={
                    "endpoint_url": f"https://{dataset_config['account_id']}.r2.cloudflarestorage.com",
                    "region_name": R2DatasetLoader.CF_REGION_NAME,
                },
                config_kwargs={
                    "max_pool_connections": 50,
                    "connect_timeout": 5,
                    "read_timeout": 10,
                    "retries": {"max_attempts": 3},
                },
                use_listings_cache=True,
                skip_instance_cache=False,
                default_block_size=R2DatasetLoader.READ_BUFFER_SIZE,
                default_cache_type="readahead",
            )
        return R2DatasetLoader._fs

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
                await self._prefetch_queue.put(page)
        except Exception as e:
            logger.error(f"Prefetch error: {e}")
        finally:
            await self._prefetch_queue.put(None)  # Signal completion

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
                    shard_sizes, _ = await self._load_r2_metadata()
                    metadata = shard_sizes[config_name]
                    self._metadata_cache[config_name] = metadata

                # Find exact shard based on page_number
                cumulative_rows = 0
                chosen_shard = None
                for shard in metadata["shards"]:
                    if (
                        cumulative_rows
                        <= page_number
                        < cumulative_rows + shard["num_rows"]
                    ):
                        chosen_shard = shard
                        break
                    cumulative_rows += shard["num_rows"]

                if not chosen_shard:
                    raise ValueError(f"Could not find shard for page {page_number}")

                # Calculate offset within shard
                shard_offset = page_number - cumulative_rows

                # Read data from exact position
                pf_data = self._parquet_cache.get(chosen_shard["path"])
                if not pf_data:
                    fs = self._get_fs()
                    f = fs.open(
                        chosen_shard["path"], "rb", buffer_size=self.READ_BUFFER_SIZE
                    )
                    pf = pq.ParquetFile(f, memory_map=True)
                    pf_data = {"file": f, "parquet": pf}
                    self._parquet_cache[chosen_shard["path"]] = pf_data

                # Fix: Ensure row group index is within bounds
                num_row_groups = pf_data["parquet"].num_row_groups
                rows_per_group = chosen_shard["num_rows"] // num_row_groups
                group_index = min(shard_offset // rows_per_group, num_row_groups - 1)

                # Read the row group
                table = await asyncio.to_thread(
                    pf_data["parquet"].read_row_group,
                    group_index,
                    columns=["text"],
                    use_threads=True,
                )

                # Adjust start_idx based on actual rows in the group
                start_idx = shard_offset % rows_per_group
                group_rows = len(table)  # Get actual rows in this group
                start_idx = min(start_idx, max(0, group_rows - self.num_rows_per_page))

                texts = table["text"].to_pylist()[
                    start_idx : start_idx + self.num_rows_per_page
                ]  # type: ignore

                # Process texts deterministically
                all_tokens = []
                for text in texts:
                    tokens = await asyncio.to_thread(
                        self.tokenizer,
                        text,
                        padding=False,
                        truncation=True,
                        max_length=self.sequence_length,
                        return_tensors=None,
                    )

                    input_ids = tokens["input_ids"]  # type: ignore
                    if input_ids:
                        all_tokens.extend(input_ids)
                        if input_ids[-1] != self.tokenizer.eos_token_id:
                            all_tokens.append(self.tokenizer.eos_token_id)

                self._token_cache[cache_key] = all_tokens
                return all_tokens

            except Exception as e:
                logger.error(f"Error processing page {page}: {e}")
                raise

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

    def _read_parquet_table(self, fs, path):
        """
        Helper method to read parquet data.

        Args:
            fs: Filesystem instance
            path (str): Path to parquet file

        Returns:
            pyarrow.Table: Table containing text data
        """
        with fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            table = pf.read(columns=["text"])
        return table

    def __del__(self):
        """Cleanup resources"""
        if self._prefetch_task:
            self._prefetch_task.cancel()

        for pf_data in self._parquet_cache.values():
            try:
                pf_data["file"].close()  # type: ignore
            except Exception as e:
                logger.debug(f"Error closing parquet file: {e}")

        self._parquet_cache.clear()
        self._token_cache.clear()

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_parquet_file(shard_path: str):
        """Cached parquet file access"""
        fs = R2DatasetLoader._get_fs()
        f = fs.open(shard_path, "rb", buffer_size=R2DatasetLoader.READ_BUFFER_SIZE)
        return {"file": f, "parquet": pq.ParquetFile(f, memory_map=True)}

    @staticmethod
    @lru_cache(maxsize=1024)
    def _get_tokenized_cache(cache_key: str):
        """Cached tokenization results"""
        return R2DatasetLoader._token_cache.get(cache_key)
