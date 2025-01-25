import asyncio
import s3fs
import pyarrow.parquet as pq
import random
import torch
import numpy as np
import json
import yaml
from pathlib import Path

from tplr.config import BUCKET_SECRETS
from tplr.dataset import DatasetLoader
from tplr.logging import logger


# Note: 24 parquet files have a different naming pattern . this should catch that
class LocalParquetDatasetLoader(DatasetLoader):
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

    @staticmethod
    def _get_s3fs():
        """
        Creates a configured S3FileSystem instance for R2 access.

        Returns:
            s3fs.S3FileSystem: Configured filesystem object for R2 access
        """
        return s3fs.S3FileSystem(
            key=BUCKET_SECRETS["read"]["access_key_id"],
            secret=BUCKET_SECRETS["read"]["secret_access_key"],
            client_kwargs={
                "endpoint_url": f"https://{BUCKET_SECRETS['account_id']}.r2.cloudflarestorage.com",
                "region_name": LocalParquetDatasetLoader.CF_REGION_NAME,
            },
            config_kwargs={
                "region_name": LocalParquetDatasetLoader.CF_REGION_NAME,
            },
            use_listings_cache=False,
            skip_instance_cache=True,  # Important: Skip instance cache
            s3_additional_kwargs={"Region": LocalParquetDatasetLoader.CF_REGION_NAME},
        )

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
        if LocalParquetDatasetLoader._shard_sizes is not None:
            return (
                LocalParquetDatasetLoader._shard_sizes,
                LocalParquetDatasetLoader._metadata_config,
            )

        fs = LocalParquetDatasetLoader._get_s3fs()
        cache_dir = LocalParquetDatasetLoader._local_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Define R2 and local paths
        r2_base = f"{BUCKET_SECRETS['bucket_name']}/{LocalParquetDatasetLoader.DATASET_SUBFOLDER}"
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
                LocalParquetDatasetLoader._shard_sizes = json.load(f)

            # Download and load metadata config
            if not local_paths["metadata"].exists():
                logger.info("Downloading metadata config from R2...")
                fs.get(r2_paths["metadata"], str(local_paths["metadata"]))
            with open(local_paths["metadata"]) as f:
                LocalParquetDatasetLoader._metadata_config = yaml.safe_load(f)

            return (
                LocalParquetDatasetLoader._shard_sizes,
                LocalParquetDatasetLoader._metadata_config,
            )

        except Exception as e:
            logger.error(f"Failed to load R2 metadata: {e}")
            raise

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer=None,
        pack_samples: bool = False,
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
        """
        Refill padding buffer from main buffer.

        Processes tokens from main buffer, adds padding as needed,
        and moves processed tokens to used buffer.
        """
        while self.buffer and len(self.padded_buffer) < self.sequence_length:
            input_ids = []

            # Find next EOS token
            try:
                EOS_index = self.buffer.index(self.tokenizer.eos_token_id)
                input_ids = self.buffer[: EOS_index + 1]
                self.buffer = self.buffer[EOS_index + 1 :]
            except ValueError:
                # No EOS found, take all remaining
                input_ids = self.buffer
                self.buffer = []

            self.used_buffer += input_ids

            # Add to padded buffer without EOS
            self.padded_buffer += input_ids[:-1]

            # Add padding
            pad_size = self._get_pad_size(input_ids[:-1])
            self.padded_buffer += [self.tokenizer.pad_token_id] * pad_size

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
        if LocalParquetDatasetLoader._configs_data_cache is not None:
            return LocalParquetDatasetLoader._configs_data_cache

        fs = LocalParquetDatasetLoader._get_s3fs()

        # Build the full path including dataset subfolder
        dataset_path = f"{BUCKET_SECRETS['bucket_name']}/{LocalParquetDatasetLoader.DATASET_SUBFOLDER}"

        try:
            print(f"Listing dataset path: {dataset_path}")
            all_paths = fs.ls(dataset_path)
            print("Available config paths:")
            for path in all_paths:
                print(f"  {path}")

            configs_data = {}
            for path in all_paths:
                config_name = path.split("/")[-1]  # This will be CC-MAIN-2017-04 etc.

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

            LocalParquetDatasetLoader._configs_data_cache = configs_data
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
        shard_sizes, _ = await LocalParquetDatasetLoader._load_r2_metadata()

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

    async def _fetch_data_for_page(self, page, session):
        """
        Fetch and process data for a single page.

        Args:
            page (tuple): (config_name, page_number, split)
            session: Unused session parameter

        Raises:
            RuntimeError: If config not found
            Exception: If shard reading fails
        """
        config_name, page_number, split = page
        logger.info(f"Fetching page: config={config_name}, page={page_number}")

        # Get shard info from cache
        shard_sizes, _ = await self._load_r2_metadata()
        if config_name not in shard_sizes:
            raise RuntimeError(f"Config {config_name} not found in shard sizes")

        # Pick a random shard from this config
        shards = shard_sizes[config_name]["shards"]
        chosen_shard = random.choice(shards)
        logger.info(f"Selected shard: {chosen_shard['path']}")

        fs = self._get_s3fs()
        try:
            with fs.open(chosen_shard["path"], "rb") as f:
                pf = pq.ParquetFile(f)
                table = pf.read_row_group(0, columns=["text"])
                texts = table["text"].to_pylist()
                logger.info(f"Read {len(texts)} texts")

                buffer_to_append = []
                tasks = [self._tokenize_content(text) for text in texts]
                row_input_ids = await asyncio.gather(*tasks)

                for input_ids in row_input_ids:
                    buffer_to_append.extend(input_ids)
                    buffer_to_append.append(self.tokenizer.eos_token_id)

                async with self.lock:
                    self.buffer.extend(buffer_to_append)
                    self.pages.append((config_name, page_number, split))

        except Exception as e:
            logger.error(f"Error reading shard {chosen_shard['path']}: {e}")
            raise

    def __iter__(self):
        """
        Initialize iterator state.

        Returns:
            self: Iterator instance
        """
        logger.info("Starting iteration")
        logger.info(f"Buffer size: {len(self.buffer)}")
        self.buffer = self.used_buffer + self.buffer
        self.padded_buffer = []
        self._refill_padded_buffer()
        return self

    def __next__(self):
        """
        Get next batch of sequences.

        Returns:
            torch.Tensor: Batch of sequences

        Raises:
            StopIteration: When no more batches available
        """
        batch = []
        logger.info(
            f"Getting next batch. Padded buffer size: {len(self.padded_buffer)}"
        )

        while len(self.padded_buffer) >= self.sequence_length:
            batch.append(self.padded_buffer[: self.sequence_length])
            self.padded_buffer = self.padded_buffer[self.sequence_length :]
            self._refill_padded_buffer()

            if len(batch) == self.batch_size:
                logger.info(f"Returning batch of size {len(batch)}")
                return torch.tensor(np.stack(batch))

        logger.info("No more batches available")
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

    @staticmethod
    async def create(
        batch_size: int,
        sequence_length: int,
        pages_info: list,
        tokenizer,
        pack_samples: bool = True,
    ):
        """
        Create and initialize a new loader instance.

        Args:
            batch_size (int): Size of batches to return
            sequence_length (int): Length of sequences to generate
            pages_info (list): List of page information tuples
            tokenizer: Tokenizer instance
            pack_samples (bool): Whether to pack samples without padding

        Returns:
            LocalParquetDatasetLoader: Initialized loader instance
        """
        loader = LocalParquetDatasetLoader()
        loader.batch_size = batch_size
        loader.sequence_length = sequence_length
        loader.tokenizer = tokenizer
        loader.pack_samples = pack_samples
        loader.buffer = []
        loader.pages = []
        loader.lock = asyncio.Lock()

        # Start fetching data for each page
        tasks = []
        for page in pages_info:
            task = asyncio.create_task(loader._fetch_data_for_page(page, None))
            tasks.append(task)

        await asyncio.gather(*tasks)
        return loader
