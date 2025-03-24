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
from transformers import AutoTokenizer
from typing import Optional, List, Any, Union, cast


from tplr import logger
from tplr.config import BUCKET_SECRETS
from tplr.dataset import DatasetLoader
from tplr.logging import T, P

import tplr


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

    _configs_data_cache: Optional[dict] = None
    DATASET_SUBFOLDER: str = "HuggingFaceFW_fineweb-edu-score-2"
    CF_REGION_NAME: str = "enam"

    # Cache for metadata
    _shard_sizes: Optional[dict] = None
    _metadata_config: Optional[dict] = None
    _local_cache_dir: Path = Path(".cache/tplr")

    # Add class-level caching for filesystem and tokenizer results
    _fs_instance: Optional[s3fs.S3FileSystem] = None
    _tokenized_cache: dict = {}
    _buffer_size: int = 1024 * 1024  # 1MB buffer for reading

    # Class-level caches
    _metadata_cache: dict = {}  # Cache for metadata by config
    _parquet_cache: dict = {}  # Cache ParquetFile objects
    _fs: Optional[s3fs.S3FileSystem] = None  # Single filesystem instance

    # Static configuration
    PREFETCH_SIZE: int = 3  # Number of pages to prefetch
    MAX_CONCURRENT_REQUESTS: int = 20
    BATCH_SIZE: int = 128  # Increased batch size for tokenization
    READ_BUFFER_SIZE: int = 4 * 1024 * 1024  # 4MB read buffer

    # Class-level caches with size limits
    _metadata_cache: dict = {}
    _parquet_cache: dict = {}  # Cache for ParquetFile objects
    _token_cache: dict = {}  # Cache for tokenized results
    _fs: Optional[s3fs.S3FileSystem] = None
    _prefetch_queue: Optional[asyncio.Queue] = None

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        batch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        num_pages: Optional[int] = None,
        pack_samples: bool = True,
    ):
        """
        Initialize the dataset loader.

        Args:
            tokenizer: Tokenizer instance to use
            batch_size (int, optional): Size of batches to return
            sequence_length (int, optional): Length of sequences to generate
            num_pages (int, optional): Number of pages to load
            pack_samples (bool): Whether to pack samples without padding
        """
        super().__init__(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_pages=num_pages,
            tokenizer=tokenizer,
            pack_samples=pack_samples,
        )

        # Override parent class variables with same values
        # This avoids the type mismatch while maintaining functionality
        self.rows_base_url = ""  # Use empty string instead of None
        self.size_base_url = ""  # Use empty string instead of None

        # Additional buffers from parent class
        self.used_buffer = []
        self.padded_buffer = []

        # Prefetch setup
        self._prefetch_task = None
        self._current_batch = None
        self._next_batch = None
        self._prefetch_queue = asyncio.Queue(maxsize=self.PREFETCH_SIZE)

    def _get_pad_size(self, input_ids: list) -> int:
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
        if self.sequence_length is None:
            return 0

        remainder = sample_size % self.sequence_length
        pad_size = self.sequence_length - remainder
        return pad_size % self.sequence_length

    def _refill_padded_buffer(self):
        """Match DatasetLoader's buffer refill logic exactly"""
        target_size = (
            self.sequence_length * self.batch_size
            if self.sequence_length is not None
            else 0
        )

        while self.buffer and len(self.padded_buffer) < target_size:
            try:
                # Find next EOS token
                # Fix: Cast tokenizer to Any to bypass type checking
                tokenizer_any = cast(Any, self.tokenizer)
                if (
                    hasattr(tokenizer_any, "eos_token_id")
                    and tokenizer_any.eos_token_id is not None
                ):
                    eos_token_id = tokenizer_any.eos_token_id
                elif hasattr(tokenizer_any, "eos_token") and hasattr(
                    tokenizer_any, "convert_tokens_to_ids"
                ):
                    eos_token_id = tokenizer_any.convert_tokens_to_ids(
                        tokenizer_any.eos_token
                    )
                else:
                    eos_token_id = 2  # Common EOS token ID in many models

                eos_index = self.buffer.index(eos_token_id)

                input_ids = self.buffer[: eos_index + 1]
                self.buffer = self.buffer[eos_index + 1 :]

                self.used_buffer.extend(input_ids)

                self.padded_buffer.extend(input_ids[:-1])

                pad_size = self._get_pad_size(input_ids[:-1])
                if pad_size > 0:
                    self.padded_buffer.extend([eos_token_id] * pad_size)

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
            shard_sizes, metadata_config = await R2DatasetLoader._load_r2_metadata()

            configs_data = {}
            if metadata_config is not None:
                for config in metadata_config.get("configs", []):
                    config_name = config.get("config_name")
                    if config_name == "default":
                        continue

                    shard_info = shard_sizes.get(config_name, {}) if shard_sizes else {}
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
        offset: int, n_pages: int, seed: Union[int, str], num_rows_per_page: int = 100
    ) -> list:
        """Get next n_pages random pages starting from offset."""
        configs_data = await R2DatasetLoader.fetch_dataset_configs()

        seed_int = int(hash(seed) if isinstance(seed, str) else seed) & 0xFFFFFFFF
        rng = np.random.default_rng(seed_int)

        for _ in range(offset):
            _ = rng.random()  # Skip ahead by generating and discarding values

        sorted_keys = sorted(configs_data.keys())

        result = []
        for _ in range(n_pages):
            config = rng.choice(sorted_keys)
            choice = rng.integers(
                0, configs_data[config]["num_rows"] - 1 - num_rows_per_page
            )
            result.append((str(config), int(choice), configs_data[config]["split"]))

        return result

    @classmethod
    async def create(
        cls,
        batch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        num_pages: Optional[int] = None,
        pages_info: Optional[List] = None,
        tokenizer: Any = None,  # Fix: Remove Optional, use Any
        pack_samples: bool = False,
    ):
        """Create loader with proper initialization"""
        # Fix: Ensure tokenizer is not None before creating loader
        if tokenizer is None:
            raise ValueError("Tokenizer cannot be None")

        loader = R2DatasetLoader(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_pages=num_pages,
            tokenizer=tokenizer,  # Now guaranteed not None
            pack_samples=pack_samples,
        )

        loader.buffer = []
        loader.pages = pages_info.copy() if pages_info else []

        sem = asyncio.Semaphore(loader.MAX_CONCURRENT_REQUESTS)
        if pages_info:
            tasks = [
                asyncio.create_task(loader._process_page(page, sem))
                for page in pages_info
            ]

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
            if not local_paths["shard_sizes"].exists():
                logger.info("Downloading shard sizes from R2...")
                fs.get(r2_paths["shard_sizes"], str(local_paths["shard_sizes"]))
            with open(local_paths["shard_sizes"]) as f:
                R2DatasetLoader._shard_sizes = json.load(f)

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
                if self._prefetch_queue is not None:
                    await self._prefetch_queue.put(page)
        except Exception as e:
            logger.error(f"Prefetch error: {e}")
        finally:
            if self._prefetch_queue is not None:
                await self._prefetch_queue.put(None)  # Signal completion

    async def _process_page(self, page, sem):
        """Process page with deterministic shard selection"""
        async with sem:
            config_name, page_number, split = page
            cache_key = f"{config_name}:{page_number}"

            try:
                if hasattr(self, "_token_cache") and cache_key in self._token_cache:
                    return self._token_cache[cache_key]

                if not hasattr(self, "_metadata_cache"):
                    self._metadata_cache = {}

                metadata = self._metadata_cache.get(config_name)
                if not metadata:
                    shard_sizes, _ = await self._load_r2_metadata()
                    if shard_sizes and config_name in shard_sizes:
                        metadata = shard_sizes[config_name]
                        self._metadata_cache[config_name] = metadata
                    else:
                        raise ValueError(f"No metadata found for config {config_name}")

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

                shard_offset = page_number - cumulative_rows

                pf_data = None
                if hasattr(self, "_parquet_cache"):
                    if chosen_shard["path"] in self._parquet_cache:
                        pf_data = self._parquet_cache[chosen_shard["path"]]

                if pf_data is None:
                    fs = self._get_fs()
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            f = fs.open(
                                chosen_shard["path"],
                                "rb",
                                buffer_size=self.READ_BUFFER_SIZE,
                            )
                            pf = pq.ParquetFile(
                                f, memory_map=False
                            )  # Disable memory mapping
                            pf_data = {"file": f, "parquet": pf}
                            if hasattr(self, "_parquet_cache"):
                                self._parquet_cache[chosen_shard["path"]] = pf_data
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"Attempt {attempt + 1} failed to open parquet file {chosen_shard['path']} with error: {e}. Retrying..."
                                )
                                await asyncio.sleep(2**attempt)  # Exponential backoff
                            else:
                                logger.error(
                                    f"Failed to open parquet file {chosen_shard['path']} after {max_retries} attempts: {e}"
                                )
                                raise

                if pf_data is None:
                    raise ValueError(
                        f"Failed to load parquet file for {chosen_shard['path']}"
                    )

                num_row_groups = pf_data["parquet"].num_row_groups
                rows_per_group = chosen_shard["num_rows"] // num_row_groups
                group_index = min(shard_offset // rows_per_group, num_row_groups - 1)

                def safe_read_row_group():
                    fs = R2DatasetLoader._get_fs()
                    file_path = chosen_shard["path"]
                    f = fs.open(
                        file_path, "rb", buffer_size=R2DatasetLoader.READ_BUFFER_SIZE
                    )
                    try:
                        pf = pq.ParquetFile(f, memory_map=True)
                        table = pf.read_row_group(
                            group_index, columns=["text"], use_threads=True
                        )
                    finally:
                        f.close()
                    return table

                table = await asyncio.to_thread(safe_read_row_group)

                start_idx = shard_offset % rows_per_group
                group_rows = len(table)  # Get actual rows in this group
                start_idx = min(start_idx, max(0, group_rows - self.num_rows_per_page))

                texts = table["text"].to_pylist()[
                    start_idx : start_idx + self.num_rows_per_page
                ]  # type: ignore

                # Process texts deterministically
                all_tokens = []
                for text in texts:
                    # Fix: Use a different approach to call tokenizer
                    def tokenize_text(text_to_tokenize):
                        # Cast tokenizer to Any to bypass type checking
                        tokenizer_any = cast(Any, self.tokenizer)

                        # Try different methods to tokenize
                        try:
                            # Try calling directly
                            result = tokenizer_any(
                                text_to_tokenize,
                                padding=False,
                                truncation=True,
                                max_length=self.sequence_length,
                                return_tensors=None,
                            )
                            return result
                        except (TypeError, AttributeError):
                            # Try encode_plus method
                            try:
                                return tokenizer_any.encode_plus(
                                    text_to_tokenize,
                                    padding=False,
                                    truncation=True,
                                    max_length=self.sequence_length,
                                    return_tensors=None,
                                )
                            except (TypeError, AttributeError):
                                # Last resort: encode method
                                input_ids = tokenizer_any.encode(
                                    text_to_tokenize,
                                    truncation=True,
                                    max_length=self.sequence_length,
                                )
                                return {"input_ids": input_ids}

                    tokens = await asyncio.to_thread(tokenize_text, text)

                    # Fix: Handle different return types
                    if isinstance(tokens, dict):
                        # Fix: Use get() to avoid KeyError
                        input_ids = tokens.get("input_ids", [])
                    elif hasattr(tokens, "input_ids"):
                        input_ids = tokens.input_ids
                    else:
                        # Assume tokens is already the input_ids
                        input_ids = tokens if isinstance(tokens, list) else []

                    if input_ids:
                        # Fix: Convert input_ids to list if it's not already
                        if not isinstance(input_ids, list):
                            try:
                                input_ids = (
                                    input_ids.tolist()
                                    if hasattr(input_ids, "tolist")
                                    else list(input_ids)
                                )
                            except (
                                TypeError,
                                ValueError,
                            ):  # Replace bare except with specific exceptions
                                input_ids = [int(x) for x in input_ids]

                        all_tokens.extend(input_ids)

                        # Fix: Cast tokenizer to Any to bypass type checking
                        tokenizer_any = cast(Any, self.tokenizer)
                        if (
                            hasattr(tokenizer_any, "eos_token_id")
                            and tokenizer_any.eos_token_id is not None
                        ):
                            eos_token_id = tokenizer_any.eos_token_id
                        elif hasattr(tokenizer_any, "eos_token") and hasattr(
                            tokenizer_any, "convert_tokens_to_ids"
                        ):
                            eos_token_id = tokenizer_any.convert_tokens_to_ids(
                                tokenizer_any.eos_token
                            )
                        else:
                            eos_token_id = 2  # Common EOS token ID in many models

                        if input_ids[-1] != eos_token_id:
                            all_tokens.append(eos_token_id)

                if not hasattr(self, "_token_cache"):
                    self._token_cache = {}
                self._token_cache[cache_key] = all_tokens
                return all_tokens

            except Exception as e:
                logger.error(f"Error processing page {page}: {e}")
                raise
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

        if self.sequence_length is None:
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

    @classmethod
    async def get_loader(
        cls,
        window: int,
        hparams,
        tokenizer: Any,  # Fix: Use Any type
        seed: Optional[int] = None,
        data_type: str = "training",
        pack_samples: bool = True,
    ):
        """
        Loads data for a given window using the R2DatasetLoader.

        Args:
            window (int): The window offset (e.g. step_window or sync_window).
            hparams: Hyperparameters including pages_per_window, batch_size, sequence_length, etc.
            tokenizer: Tokenizer instance to use.
            seed (int, optional): Seed for deterministic page selection; if None, a random seed is used.
            data_type (str, optional): For logging, e.g. "training" or "evaluation".
            pack_samples (bool, optional): Whether to pack samples without padding.

        Returns:
            tuple: (loader, pages_info)
        """
        seed_val = seed if seed is not None else np.random.randint(0, 10000)

        start_time = T()
        pages = await cls.next_pages(
            offset=window, n_pages=hparams.pages_per_window, seed=seed_val
        )
        loader = await cls.create(
            batch_size=hparams.batch_size,
            sequence_length=hparams.sequence_length,
            num_pages=hparams.pages_per_window,
            pages_info=pages,
            tokenizer=tokenizer,
            pack_samples=pack_samples,
        )
        elapsed = T() - start_time
        logger.info(
            f"Loaded {data_type} data for window {window} with seed: {seed_val}, pages: {[p[1] for p in pages]} "
            + P(window, elapsed)
        )
        return loader, pages


async def retry_call(func, *args, attempts=3, delay=1, context="", **kwargs):
    """
    Calls an async function with retries.
    Args:
        func (Callable): An async function.
        *args: Positional arguments to pass to func.
        attempts (int): Number of retries.
        delay (int): Delay between attempts in seconds.
        context (str): Context description for logging.
        **kwargs: Keyword arguments to pass to func.
    Returns:
        The result of func(*args, **kwargs) or None if all attempts fail.
    """
    for attempt in range(attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            tplr.logger.error(
                f"Attempt {attempt + 1}/{attempts} failed for {context}: {e}"
            )
            await asyncio.sleep(delay)
    tplr.logger.error(f"Failed to complete {context} after {attempts} attempts.")
    return None
