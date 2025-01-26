import json
import yaml
import s3fs
import torch
import random
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
        """Simplified buffer refill"""
        if not self.buffer:
            return

        # Process entire buffer at once
        eos_positions = [
            i
            for i, token in enumerate(self.buffer)
            if token == self.tokenizer.eos_token_id
        ]

        if not eos_positions:
            self.padded_buffer.extend(self.buffer)
            self.buffer = []
            return

        # Process all complete sequences
        last_eos = eos_positions[-1]
        sequences = []
        start = 0

        for pos in eos_positions:
            seq = self.buffer[start:pos]
            if not self.pack_samples:
                # Pad to sequence length
                pad_size = (-len(seq)) % self.sequence_length
                seq.extend([self.tokenizer.pad_token_id] * pad_size)
            sequences.extend(seq)
            start = pos + 1

        # Update buffers
        self.padded_buffer.extend(sequences)
        self.buffer = self.buffer[last_eos + 1 :]

    @staticmethod
    async def fetch_dataset_configs() -> dict:
        """
        Scans the R2 bucket for Parquet dataset configurations.

        Each subfolder represents a config containing 'train.parquet' files.

        Returns:
            dict: Dataset configurations with row counts and shard info

        Raises:
            Exception: If scanning fails
        """
        if R2DatasetLoader._configs_data_cache is not None:
            return R2DatasetLoader._configs_data_cache

        fs = R2DatasetLoader._get_fs()

        # Build the full path including dataset subfolder
        dataset_path = (
            f"{BUCKET_SECRETS['dataset']['name']}/{R2DatasetLoader.DATASET_SUBFOLDER}"
        )

        try:
            print(f"Listing dataset path: {dataset_path}")
            all_paths = fs.ls(dataset_path)
            print("Available config paths:")
            for path in all_paths:
                print(f"  {path}")

            configs_data = {}
            for path in all_paths:
                config_name = path.split("/")[
                    -1
                ]  # This will be CC-MAIN-2017-04 etc. #type: ignore

                # List all parquet files in this config
                parquet_files = [f for f in fs.ls(path) if f.endswith(".parquet")]
                if not parquet_files:
                    print(f"Skipping {config_name} - no parquet files found")
                    continue

                # Count total rows across all parquet files
                total_rows = 0
                for parquet_file in parquet_files:
                    with fs.open(parquet_file, "rb") as f:
                        pf = pq.ParquetFile(f)
                        total_rows += pf.metadata.num_rows

                configs_data[config_name] = {
                    "num_rows": total_rows,
                    "split": "train",
                    "num_shards": len(parquet_files),
                }
                print(
                    f"Added config {config_name} with {total_rows} rows across {len(parquet_files)} shards"
                )

            R2DatasetLoader._configs_data_cache = configs_data
            return configs_data

        except Exception as e:
            print(f"Error scanning dataset folder: {str(e)}")
            raise

    @staticmethod
    async def next_pages(offset: int, n_pages: int, seed: str) -> list:
        """
        Generate next set of pages using cached metadata.

        Pages are selected randomly weighted by row counts.

        Args:
            offset (int): Starting offset (unused)
            n_pages (int): Number of pages to generate
            seed (str): Random seed for reproducibility

        Returns:
            list: List of (config_name, page_number, split) tuples

        Raises:
            RuntimeError: If no configs found
        """
        logger.info(f"Generating {n_pages} pages with seed {seed}")
        rng = random.Random(seed)

        # Load cached metadata
        shard_sizes, _ = await R2DatasetLoader._load_r2_metadata()

        # Get configs with their total rows
        configs = [(name, data["total_rows"]) for name, data in shard_sizes.items()]
        if not configs:
            raise RuntimeError("No configs found in shard sizes data")

        # Generate weighted random pages based on row counts
        total_rows = sum(rows for _, rows in configs)
        pages = []
        for i in range(n_pages):
            # Pick config weighted by row count
            config_name = rng.choices(
                [name for name, _ in configs],
                weights=[rows / total_rows for _, rows in configs],
            )[0]

            # Get random page number within config's row count
            max_rows = shard_sizes[config_name]["total_rows"]
            page_number = rng.randint(0, max_rows - 1)

            pages.append((config_name, page_number, "train"))
            logger.info(f"Generated page {i + 1}: {pages[-1]}")

        return pages

    @staticmethod
    async def create(
        batch_size, sequence_length, pages_info, tokenizer, pack_samples=True
    ):
        """Optimized loader creation with prefetching"""
        loader = R2DatasetLoader(
            batch_size=batch_size,
            sequence_length=sequence_length,
            tokenizer=tokenizer,
            pack_samples=pack_samples,
        )
        loader.buffer = []
        loader.pages = pages_info.copy()

        # Process pages in parallel with increased concurrency
        sem = asyncio.Semaphore(loader.MAX_CONCURRENT_REQUESTS)
        tasks = [
            asyncio.create_task(loader._process_page(page, sem)) for page in pages_info
        ]

        # Wait for all pages and process results
        results = await asyncio.gather(*tasks)
        for tokens in results:
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
        """Process a single page with optimized caching"""
        async with sem:
            config_name, page_number, split = page
            cache_key = f"{config_name}:{page_number}"

            try:
                # Try to get from cache first
                if cache_key in self._token_cache:
                    return self._token_cache[cache_key]

                # Get metadata and choose shard
                metadata = self._metadata_cache.get(config_name)
                if not metadata:
                    shard_sizes, _ = await self._load_r2_metadata()
                    metadata = shard_sizes[config_name]
                    self._metadata_cache[config_name] = metadata

                chosen_shard = random.choice(metadata["shards"])
                shard_path = chosen_shard["path"]

                # Get or create ParquetFile
                pf_data = self._parquet_cache.get(shard_path)
                if not pf_data:
                    fs = self._get_fs()
                    f = fs.open(shard_path, "rb", buffer_size=self.READ_BUFFER_SIZE)
                    pf = pq.ParquetFile(f, memory_map=True)
                    pf_data = {"file": f, "parquet": pf}
                    self._parquet_cache[shard_path] = pf_data

                # Read data efficiently
                selected_group = random.randint(
                    0, pf_data["parquet"].num_row_groups - 1
                )
                table = await asyncio.to_thread(
                    pf_data["parquet"].read_row_group,
                    selected_group,
                    columns=["text"],
                    use_threads=True,
                )

                # Process in large batches
                texts = table["text"].to_pylist()  # type: ignore
                all_tokens = []

                for i in range(0, len(texts), self.BATCH_SIZE):
                    batch = texts[i : i + self.BATCH_SIZE]
                    tokens = await asyncio.to_thread(
                        self.tokenizer,
                        batch,
                        padding=False,
                        truncation=True,
                        max_length=self.sequence_length,
                        return_tensors=None,
                    )

                    for input_ids in tokens["input_ids"]:  # type: ignore
                        all_tokens.extend(input_ids)
                        all_tokens.append(self.tokenizer.eos_token_id)

                # Cache results
                self._token_cache[cache_key] = all_tokens
                return all_tokens

            except Exception as e:
                logger.error(f"Error processing page {page}: {e}")
                raise

    def __iter__(self):
        """Iterator with prefetching"""
        self.buffer = self.used_buffer + self.buffer
        self.padded_buffer = []
        self._refill_padded_buffer()

        # Start prefetching next batch
        if self._next_batch is None:
            self._prefetch_next_batch()

        return self

    def _prefetch_next_batch(self):
        """Prefetch next batch in background"""
        if len(self.padded_buffer) >= self.sequence_length * self.batch_size:
            batch = []
            for _ in range(self.batch_size):
                batch.append(self.padded_buffer[: self.sequence_length])
                self.padded_buffer = self.padded_buffer[self.sequence_length :]
            self._next_batch = torch.tensor(np.stack(batch))

    def __next__(self):
        """Get next batch with prefetching"""
        if self._next_batch is not None:
            result = self._next_batch
            self._next_batch = None
            self._prefetch_next_batch()
            return result

        if len(self.padded_buffer) < self.sequence_length * self.batch_size:
            self._refill_padded_buffer()
            if len(self.padded_buffer) < self.sequence_length * self.batch_size:
                raise StopIteration

        return self.__next__()

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
