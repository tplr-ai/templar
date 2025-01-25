import asyncio
import s3fs
import pyarrow.parquet as pq
import random
import torch
import numpy as np

# Pull in the same config used by comms.py
from tplr.config import BUCKET_SECRETS
from tplr.dataset import DatasetLoader
from tplr.comms import CF_REGION_NAME
from tplr.logging import logger

# Note: 24 parquet files have a different naming pattern . this should catch that
class LocalParquetDatasetLoader(DatasetLoader):
    """
    Drop-in replacement for DatasetLoader, but reads Parquet from R2
    using the same credentials logic as comms.py/config.py.
    """

    rows_base_url = None
    size_base_url = None
    _configs_data_cache = None
    DATASET_SUBFOLDER = "HuggingFaceFW_fineweb-edu-score-2"  # Add this constant
    CF_REGION_NAME = "enam"

    @staticmethod
    def _get_s3fs():
        """Helper to create consistent S3FileSystem instances"""
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

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer=None,
        pack_samples: bool = False,
    ):
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
        """Get padding size for sequence."""
        if self.pack_samples:
            return 1

        sample_size = len(input_ids)
        remainder = sample_size % self.sequence_length
        pad_size = self.sequence_length - remainder
        return pad_size % self.sequence_length

    def _refill_padded_buffer(self):
        """Refill padding buffer from main buffer."""
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
        Scans the R2 bucket for your Parquet configs.
        Example approach: each subfolder is a config, containing 'train.parquet'.
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
        """Generate next set of pages to process."""
        logger.info(f"Generating {n_pages} pages with seed {seed}")
        rng = random.Random(seed)

        fs = LocalParquetDatasetLoader._get_s3fs()
        base_path = f"{BUCKET_SECRETS['bucket_name']}/{LocalParquetDatasetLoader.DATASET_SUBFOLDER}"
        logger.info(f"Scanning path: {base_path}")

        try:
            # Get list of all configs (CC-MAIN-*)
            configs = []
            for path in fs.ls(base_path):
                if fs.isdir(path):
                    configs.append(path.split("/")[-1])

            if not configs:
                raise RuntimeError(f"No config directories found in {base_path}")

            logger.info(f"Found configs: {configs}")

            # Generate random pages
            pages = []
            for i in range(n_pages):
                config_name = rng.choice(configs)
                page_number = rng.randint(0, 1000)  # TODO: Get actual row count
                split = "train"
                pages.append((config_name, page_number, split))
                logger.info(f"Generated page {i+1}: {pages[-1]}")

            return pages
        except Exception as e:
            logger.error(f"Error in next_pages: {str(e)}", exc_info=True)
            raise

    async def _fetch_data_for_page(self, page, session):
        """Fetch data for a single page."""
        config_name, page_number, split = page
        logger.info(f"Fetching data for page: {page}")

        fs = LocalParquetDatasetLoader._get_s3fs()
        base_path = (
            f"{BUCKET_SECRETS['bucket_name']}/{self.DATASET_SUBFOLDER}/{config_name}"
        )
        logger.info(f"Listing parquet files in: {base_path}")

        parquet_files = [f for f in fs.ls(base_path) if f.endswith(".parquet")]
        logger.info(f"Found {len(parquet_files)} parquet files")

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {base_path}")

        parquet_path = random.choice(parquet_files)
        logger.info(f"Selected parquet file: {parquet_path}")

        try:
            with fs.open(parquet_path, "rb") as f:
                pf = pq.ParquetFile(f)
                table = pf.read_row_group(0, columns=["text"])
                texts = table["text"].to_pylist()
                logger.info(f"Read {len(texts)} texts from parquet file")

                buffer_to_append = []
                tasks = [self._tokenize_content(text) for text in texts]
                row_input_ids = await asyncio.gather(*tasks)

                for input_ids in row_input_ids:
                    buffer_to_append.extend(input_ids)
                    buffer_to_append.append(self.tokenizer.eos_token_id)

                logger.info(
                    f"Tokenized {len(texts)} texts into {len(buffer_to_append)} tokens"
                )

                async with self.lock:
                    self.buffer.extend(buffer_to_append)
                    self.pages.append((config_name, page_number, split))

        except Exception as e:
            logger.error(f"Error reading parquet data: {str(e)}", exc_info=True)
            raise

    def __iter__(self):
        """Iterator implementation."""
        logger.info("Starting iteration")
        logger.info(f"Buffer size: {len(self.buffer)}")
        self.buffer = self.used_buffer + self.buffer
        self.padded_buffer = []
        self._refill_padded_buffer()
        return self

    def __next__(self):
        """Get next batch."""
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
        """Helper method to read parquet data"""
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
        """Create a new loader instance."""
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
