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
from collections import OrderedDict
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

from tplr import logger
from tplr.config import BUCKET_SECRETS
from tplr.dataset import DatasetLoader

import tplr


class R2DatasetLoader(DatasetLoader):
    """
    A drop-in replacement for DatasetLoader that reads Parquet files from Cloudflare R2 storage.

    Optimized for constant-time page loading regardless of dataset size through:
    - Lazy loading with background prefetching
    - Efficient caching with LRU policies
    - Connection pooling and resource management
    - Batch processing of pages

    This loader handles:
    - Reading and caching metadata from R2 storage
    - Loading data from Parquet files in parallel
    - Tokenizing and batching text data
    - Managing sequence padding and packing

    The loader uses the same credentials logic as comms.py/config.py for R2 access.
    """

    # Static configuration
    DATASET_SUBFOLDER = "HuggingFaceFW_fineweb-edu-score-2"
    CF_REGION_NAME = "enam"
    PREFETCH_SIZE = 10  # Number of pages to prefetch ahead
    MAX_CONCURRENT_REQUESTS = 20  # Maximum concurrent R2 requests
    READ_BUFFER_SIZE = 4 * 1024 * 1024  # 4MB read buffer
    BATCH_SIZE = 128  # Batch size for tokenization
    MAX_CACHE_SIZE = 100  # Maximum number of cached pages
    MAX_PARQUET_CACHE = 20  # Maximum number of cached parquet file objects
    LOCAL_CACHE_DIR = Path(".cache/tplr")

    # Class-level caches
    _fs = None  # Single filesystem instance
    _executor = None  # Thread pool executor
    _configs_data_cache = None  # Dataset config cache
    _shard_sizes = None  # Cache for shard size metadata
    _metadata_config = None  # Cache for dataset metadata configuration

    _round_robin_index = 0  # global counter for dataset round-robin selection
    _fs_cache = {}  # maps account_id to a cached s3fs.S3FileSystem
    _fs_lock = threading.Lock()  # lock for fs cache and round robin

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer=None,
        pack_samples=True,
        num_rows_per_page=100,
    ):
        """
        Initialize the dataset loader.

        Args:
            batch_size (int, optional): Size of batches to return
            sequence_length (int, optional): Length of sequences to generate
            num_pages (int, optional): Number of pages to load
            tokenizer: Tokenizer instance to use
            pack_samples (bool): Whether to pack samples without padding
            num_rows_per_page (int): Number of rows to read per page
        """
        super().__init__(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_pages=num_pages,
            tokenizer=tokenizer,
            pack_samples=pack_samples,
        )

        # Additional buffers from parent class
        self.buffer = []
        self.used_buffer = []
        self.padded_buffer = []
        self.pages = []
        self.num_rows_per_page = num_rows_per_page

        # Instance-specific caches
        self._token_cache = OrderedDict()  # LRU cache for tokenized pages
        self._metadata_cache = {}  # Cache for metadata by config

        # Prefetch mechanism
        self._prefetch_queue = asyncio.Queue(maxsize=self.PREFETCH_SIZE)
        self._prefetch_task = None
        self._prefetch_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)
        self._processing = False
        self._shutdown_event = asyncio.Event()  # New event to signal shutdown

        # Ensure thread pool executor is initialized
        if R2DatasetLoader._executor is None:
            R2DatasetLoader._executor = ThreadPoolExecutor(
                max_workers=self.MAX_CONCURRENT_REQUESTS
            )

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
        Refill the padded buffer from the main buffer.
        Matches DatasetLoader's buffer refill logic exactly.
        """
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
    def _get_fs():
        """
        Get or create a shared S3 filesystem instance for R2 access using round-robin selection.
        Uses connection pooling for efficiency and distributes load across multiple buckets.

        Returns:
            s3fs.S3FileSystem: Configured filesystem instance
        """
        dataset_config = BUCKET_SECRETS["dataset"]
        logger.debug(f"Dataset config loaded: {dataset_config}")

        with R2DatasetLoader._fs_lock:
            # Use round robin selection if multiple buckets are configured
            if "multiple" in dataset_config:
                configs = dataset_config["multiple"]
                idx = R2DatasetLoader._round_robin_index % len(configs)
                selected_config = configs[idx]
                R2DatasetLoader._round_robin_index += 1
                logger.debug(
                    f"Round-robin selected bucket {idx}: {selected_config.get('name', 'default')}"
                )
            else:
                selected_config = dataset_config
                logger.debug(
                    f"Using single bucket: {selected_config.get('name', 'default')}"
                )

            # Use account_id as cache key
            fs_cache_key = selected_config["account_id"]

            # Create new filesystem instance if not in cache
            if fs_cache_key not in R2DatasetLoader._fs_cache:
                read_credentials = selected_config["credentials"]["read"]
                fs = s3fs.S3FileSystem(
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
                    use_listings_cache=True,
                    skip_instance_cache=False,
                    default_block_size=R2DatasetLoader.READ_BUFFER_SIZE,
                    default_cache_type="readahead",
                )
                R2DatasetLoader._fs_cache[fs_cache_key] = fs

            return R2DatasetLoader._fs_cache[fs_cache_key]

    @staticmethod
    async def _load_r2_metadata():
        """
        Loads and caches metadata from R2 storage.
        Downloads shard sizes and metadata config files if not cached locally.

        Returns:
            tuple: (shard_sizes dict, metadata_config dict)
        """
        if R2DatasetLoader._shard_sizes is not None:
            return (R2DatasetLoader._shard_sizes, R2DatasetLoader._metadata_config)

        fs = R2DatasetLoader._get_fs()
        cache_dir = R2DatasetLoader.LOCAL_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Define R2 and local paths - handle both single and multiple bucket configurations
        dataset_config = BUCKET_SECRETS["dataset"]
        if "multiple" in dataset_config:
            # Use the first bucket for metadata (consistent location)
            bucket_name = dataset_config["multiple"][0]["name"]
        else:
            bucket_name = dataset_config["name"]

        r2_base = f"{bucket_name}/{R2DatasetLoader.DATASET_SUBFOLDER}"
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

            return (R2DatasetLoader._shard_sizes, R2DatasetLoader._metadata_config)

        except Exception as e:
            logger.error(f"Failed to load R2 metadata: {e}")
            raise

    @staticmethod
    async def fetch_dataset_configs() -> dict:
        """
        Load dataset configurations from cached metadata and shard sizes.

        Returns:
            dict: Dataset configurations
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
        """
        Get next n_pages random pages starting from offset.

        Args:
            offset (int): Starting offset
            n_pages (int): Number of pages to retrieve
            seed (str): Seed for random number generator
            num_rows_per_page (int): Number of rows per page

        Returns:
            list: List of (config_name, page_number, split) tuples
        """
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

    async def _start_prefetching(self):
        """
        Start the background prefetching task to load pages asynchronously.
        This is key to achieving constant-time page loading regardless of total pages.
        """
        if self._prefetch_task is not None and not self._prefetch_task.done():
            return  # Already running

        self._processing = True
        self._shutdown_event.clear()  # Clear shutdown event

        # Get the current event loop safely
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                # If we're in a context where the loop is closed, don't start prefetching
                logger.warning("Cannot start prefetching: event loop is closed")
                return
            self._prefetch_task = asyncio.create_task(self._prefetch_pages())
        except RuntimeError:
            # Handle the case where there is no current event loop
            logger.warning("Cannot start prefetching: no event loop available")
            self._processing = False

    async def _prefetch_pages(self):
        """
        Background task to prefetch pages and process them in parallel.
        Maintains a constant-sized queue of processed pages regardless of total pages.
        """
        try:
            tasks = set()
            pages_to_process = self.pages.copy() if self.pages else []

            # Process initial batch of pages up to prefetch limit
            initial_batch = pages_to_process[: self.PREFETCH_SIZE]
            for page in initial_batch:
                task = asyncio.create_task(self._process_page(page))
                tasks.add(task)
                task.add_done_callback(tasks.remove)

            pages_to_process = pages_to_process[self.PREFETCH_SIZE :]

            # Process remaining pages as queue space becomes available
            while (
                pages_to_process
                and self._processing
                and not self._shutdown_event.is_set()
            ):
                # Wait for queue space or completion
                if (
                    self._prefetch_queue.qsize() < self.PREFETCH_SIZE
                    and len(tasks) < self.MAX_CONCURRENT_REQUESTS
                ):
                    page = pages_to_process.pop(0)
                    task = asyncio.create_task(self._process_page(page))
                    tasks.add(task)
                    task.add_done_callback(tasks.remove)
                else:
                    # Use a short sleep to avoid busy-waiting
                    await asyncio.sleep(0.01)

                    # Check for shutdown signal
                    if self._shutdown_event.is_set():
                        break

            # Wait for remaining tasks to complete if not shutting down
            if tasks and not self._shutdown_event.is_set():
                # Use wait with a timeout to avoid hanging
                done, pending = await asyncio.wait(
                    tasks, timeout=10.0, return_when=asyncio.ALL_COMPLETED
                )

                # Cancel any pending tasks
                for task in pending:
                    task.cancel()

        except asyncio.CancelledError:
            # Handle cancellation gracefully
            logger.debug("Prefetch task was cancelled")
            # Clean up any pending tasks
            for task in tasks:
                task.cancel()
            self._processing = False
            raise
        except Exception as e:
            logger.error(f"Prefetch error: {e}")
        finally:
            # Signal completion
            self._processing = False
            try:
                loop = asyncio.get_running_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop is closed")
                await asyncio.wait_for(self._prefetch_queue.put(None), timeout=1.0)
            except (
                RuntimeError,
                asyncio.TimeoutError,
                asyncio.CancelledError,
                Exception,
            ) as e:
                logger.debug(f"Error finalizing prefetch queue: {e}")

    async def _process_page(self, page):
        """
        Process a single page by loading and tokenizing its content.
        Implements efficient caching and resource management.

        Args:
            page: Tuple of (config_name, page_number, split)

        Returns:
            list: Tokenized content
        """
        # Early check for shutdown
        if self._shutdown_event.is_set():
            return []

        async with self._prefetch_semaphore:
            # Check again after acquiring semaphore
            if self._shutdown_event.is_set():
                return []

            config_name, page_number, split = page
            cache_key = f"{config_name}:{page_number}"

            try:
                # Check token cache first (O(1) operation)
                if cache_key in self._token_cache:
                    tokens = self._token_cache[cache_key]
                    # Move to end of OrderedDict to mark as recently used
                    self._token_cache.pop(cache_key)
                    self._token_cache[cache_key] = tokens

                    # Check for shutdown before putting to queue
                    if not self._shutdown_event.is_set():
                        try:
                            await asyncio.wait_for(
                                self._prefetch_queue.put(tokens), timeout=1.0
                            )
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            # If we can't put to queue, just return
                            pass
                    return tokens

                # Exit early if shutting down
                if self._shutdown_event.is_set():
                    return []

                # Load metadata if not already cached
                if config_name not in self._metadata_cache:
                    shard_sizes, _ = await R2DatasetLoader._load_r2_metadata()
                    if config_name in shard_sizes:
                        self._metadata_cache[config_name] = shard_sizes[config_name]
                    else:
                        raise ValueError(f"Config {config_name} not found in metadata")

                metadata = self._metadata_cache[config_name]

                # Find the exact shard containing this page
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

                # Exit early if shutting down
                if self._shutdown_event.is_set():
                    return []

                # Calculate offset within shard
                shard_offset = page_number - cumulative_rows

                # Read data from the exact shard position using thread pool
                def read_from_shard():
                    fs = R2DatasetLoader._get_fs()
                    shard_path = chosen_shard["path"]

                    # Open file with appropriate buffer size
                    with fs.open(
                        shard_path, "rb", buffer_size=R2DatasetLoader.READ_BUFFER_SIZE
                    ) as f:
                        pf = pq.ParquetFile(f)

                        # Calculate exact row group and offset
                        num_row_groups = pf.num_row_groups
                        rows_per_group = chosen_shard["num_rows"] // num_row_groups
                        group_index = min(
                            shard_offset // rows_per_group, num_row_groups - 1
                        )

                        # Read specific row group
                        table = pf.read_row_group(group_index, columns=["text"])

                        # Calculate start index within group
                        start_idx = shard_offset % rows_per_group
                        group_rows = len(table)
                        start_idx = min(
                            start_idx, max(0, group_rows - self.num_rows_per_page)
                        )

                        # Extract texts
                        texts = table["text"].to_pylist()[
                            start_idx : start_idx + self.num_rows_per_page
                        ]
                        return texts

                # Use thread pool to avoid blocking the event loop
                texts = await asyncio.to_thread(read_from_shard)

                # Exit early if shutting down
                if self._shutdown_event.is_set():
                    return []

                # Process texts in parallel
                all_tokens = []

                # Tokenize texts using thread pool to avoid blocking
                def tokenize_text(text):
                    return self.tokenizer(
                        text,
                        padding=False,
                        truncation=True,
                        max_length=self.sequence_length,
                        return_tensors=None,
                    )["input_ids"]

                # Process texts in parallel batches
                tokenized_results = []
                for text in texts:
                    # Check for shutdown signal
                    if self._shutdown_event.is_set():
                        return []
                    result = await asyncio.to_thread(tokenize_text, text)
                    tokenized_results.append(result)

                # Combine tokenized results and add EOS tokens
                for input_ids in tokenized_results:
                    if input_ids:
                        all_tokens.extend(input_ids)
                        if input_ids[-1] != self.tokenizer.eos_token_id:
                            all_tokens.append(self.tokenizer.eos_token_id)

                # Final shutdown check before caching/adding to queue
                if self._shutdown_event.is_set():
                    return []

                # Update LRU cache
                if len(self._token_cache) >= self.MAX_CACHE_SIZE:
                    # Remove least recently used item
                    self._token_cache.popitem(last=False)
                self._token_cache[cache_key] = all_tokens

                # Add to prefetch queue
                try:
                    await asyncio.wait_for(
                        self._prefetch_queue.put(all_tokens), timeout=1.0
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    # If we can't put to the queue (e.g., during shutdown), just return
                    pass

                return all_tokens

            except Exception as e:
                logger.error(f"Error processing page {page}: {e}")
                # Add empty result to queue to maintain progress, if not shutting down
                if not self._shutdown_event.is_set():
                    try:
                        await asyncio.wait_for(
                            self._prefetch_queue.put([]), timeout=0.5
                        )
                    except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                        # Ignore errors when putting to queue during shutdown
                        pass
                return []

    async def _get_next_tokens(self):
        """
        Get the next batch of tokens from the prefetch queue.
        Ensures constant-time access to the next page of tokens.

        Returns:
            list: Next batch of tokenized content
        """
        # Start prefetching if not already started
        if not self._prefetch_task or self._prefetch_task.done():
            await self._start_prefetching()

        if not self._processing or self._shutdown_event.is_set():
            return []

        try:
            # Get next tokens from queue with timeout to avoid blocking forever
            tokens = await asyncio.wait_for(self._prefetch_queue.get(), timeout=5.0)

            # None signals end of data
            if tokens is None:
                self._processing = False
                return []

            return tokens
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for tokens from prefetch queue")
            return []
        except asyncio.CancelledError:
            self._processing = False
            raise
        except Exception as e:
            logger.error(f"Error getting next tokens: {e}")
            return []

    @staticmethod
    async def create(
        batch_size,
        sequence_length,
        pages_info,
        tokenizer,
        pack_samples=True,
        num_rows_per_page=100,
    ):
        """
        Create loader with proper initialization.
        Implements lazy loading for constant-time initialization.

        Args:
            batch_size (int): Batch size
            sequence_length (int): Sequence length
            pages_info (list): List of page information
            tokenizer: Tokenizer instance
            pack_samples (bool): Whether to pack samples
            num_rows_per_page (int): Number of rows per page

        Returns:
            R2DatasetLoader: Initialized loader
        """
        loader = R2DatasetLoader(
            batch_size=batch_size,
            sequence_length=sequence_length,
            tokenizer=tokenizer,
            pack_samples=pack_samples,
            num_rows_per_page=num_rows_per_page,
        )

        # Store pages for later processing
        loader.pages = pages_info.copy()

        # Register finalizer to handle cleanup when the object is garbage collected
        weakref.finalize(loader, R2DatasetLoader._cleanup_resources, loader)

        # Start prefetching in background
        await loader._start_prefetching()

        # Process first batch immediately for faster response
        try:
            # Wait with a timeout to avoid blocking indefinitely
            first_batch = await asyncio.wait_for(loader._get_next_tokens(), timeout=5.0)
            if first_batch:
                loader.buffer.extend(first_batch)
                loader._refill_padded_buffer()
        except asyncio.TimeoutError:
            logger.warning(
                "Timeout waiting for initial tokens, continuing with empty buffer"
            )
        except Exception as e:
            logger.error(f"Error loading initial tokens: {e}")

        return loader

    def _flush_prefetch_queue_sync(self):
        """
        Synchronously drain the asyncio prefetch queue (if any tokens are available)
        and add them to self.buffer. Safe to call from __next__ which must be synchronous.
        """
        # Check if we're shutting down
        if self._shutdown_event.is_set():
            return

        # Use get_nowait() to flush the queue without blocking
        while True:
            try:
                if not hasattr(self, "_prefetch_queue") or self._prefetch_queue is None:
                    return

                tokens = self._prefetch_queue.get_nowait()
                if tokens is None:  # end signal
                    self._processing = False
                    break
                self.buffer.extend(tokens)
                # Mark task as done
                self._prefetch_queue.task_done()
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                # Any other errors should be ignored to maintain sync behavior
                logger.debug(f"Error in _flush_prefetch_queue_sync: {e}")
                break

    @staticmethod
    def _cleanup_resources(loader):
        """
        Static method for cleaning up resources safely, used by the finalizer.

        Args:
            loader: The loader instance to be cleaned up
        """
        # Signal shutdown
        if hasattr(loader, "_shutdown_event"):
            loader._shutdown_event.set()

        # Cancel the prefetch task
        if (
            hasattr(loader, "_prefetch_task")
            and loader._prefetch_task
            and not loader._prefetch_task.done()
        ):
            try:
                # Check if we can safely cancel the task
                loop = (
                    asyncio.get_event_loop()
                    if hasattr(loader._prefetch_task, "_loop")
                    else None
                )
                if loop and loop.is_running() and not loop.is_closed():
                    loader._prefetch_task.cancel()
            except Exception:
                # Ignore any errors during cleanup
                pass

        # Clear any references to help garbage collection
        if hasattr(loader, "_token_cache"):
            loader._token_cache.clear()
        if hasattr(loader, "_metadata_cache"):
            loader._metadata_cache.clear()
        if hasattr(loader, "buffer"):
            loader.buffer = []
        if hasattr(loader, "used_buffer"):
            loader.used_buffer = []
        if hasattr(loader, "padded_buffer"):
            loader.padded_buffer = []

    async def shutdown(self):
        """
        Properly shut down the prefetch mechanism and clean up resources.
        This should be called explicitly when done with the loader.
        """
        # Signal processing to stop
        self._processing = False
        self._shutdown_event.set()

        # Cancel the prefetch task if it exists and is running
        if hasattr(self, "_prefetch_task") and self._prefetch_task:
            if not self._prefetch_task.done():
                try:
                    # Try to cancel the task
                    self._prefetch_task.cancel()
                    # Wait for cancellation to complete with timeout
                    try:
                        await asyncio.wait_for(self._prefetch_task, timeout=1.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        # Ignore timeout or cancellation exceptions
                        pass
                except Exception as e:
                    logger.debug(f"Error cancelling prefetch task: {e}")

        # Clear the prefetch queue
        if hasattr(self, "_prefetch_queue") and self._prefetch_queue:
            try:
                # Empty the queue
                while not self._prefetch_queue.empty():
                    try:
                        self._prefetch_queue.get_nowait()
                        self._prefetch_queue.task_done()
                    except asyncio.QueueEmpty:
                        break
                    except Exception:
                        # Ignore errors when emptying queue
                        pass
            except Exception as e:
                logger.debug(f"Error clearing prefetch queue: {e}")

        # Clear caches to free memory
        self._token_cache.clear()


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
