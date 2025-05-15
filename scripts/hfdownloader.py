#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "argparse",
#   "boto3",
#   "dnspython",
#   "requests",
#   "tqdm",
#   "botocore",
#   "pyarrow",
# ]
# ///
"""
High-performance Hugging Face to R2 file transfer utility.

This module provides a robust, concurrent solution for transferring datasets
from Hugging Face to Cloudflare R2 storage with optimized throughput and reliability.
"""

import concurrent.futures
import json
import math
import os
import queue
import random
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import boto3
import dns.resolver
import pyarrow.parquet as pq
import requests
from botocore.client import BaseClient
from botocore.config import Config
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

STREAM_BUFFER_SIZE = 256 * 1024 * 1024  # 256MB buffer
MULTIPART_THRESHOLD = 1024 * 1024 * 1024  # 1GB threshold
CHUNK_SIZE = 256 * 1024 * 1024  # 256MB chunks
MAX_CONCURRENT = 16  # 16 concurrent files
MAX_PARTS_PER_FILE = 32  # 32 concurrent chunks per file
BUFFER_SIZE = 128 * 1024 * 1024  # 128MB buffer
MAX_RETRIES = 3  # Retries for failures
RETRY_DELAY = 0.5  # Base delay in seconds

AGREEMENT_MODEL_URL = "https://huggingface.co/{}"
AGREEMENT_DATASET_URL = "https://huggingface.co/datasets/{}"
RAW_MODEL_FILE_URL = "https://huggingface.co/{}/raw/{}/{}"
RAW_DATASET_FILE_URL = "https://huggingface.co/datasets/{}/raw/{}/{}"
LFS_MODEL_RESOLVER_URL = "https://huggingface.co/{}/resolve/{}/{}"
LFS_DATASET_RESOLVER_URL = "https://huggingface.co/datasets/{}/resolve/{}/{}"
JSON_MODELS_FILE_TREE_URL = "https://huggingface.co/api/models/{}/tree/{}/{}"
JSON_DATASET_FILE_TREE_URL = "https://huggingface.co/api/datasets/{}/tree/{}/{}"

NUM_CONNECTIONS = 64
REQUIRES_AUTH = False
AUTH_TOKEN = ""


class FileStatus(Enum):
    """File processing status enum."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"


@dataclass
class HFLfs:
    """LFS file metadata structure."""

    oid: str = ""  # in lfs, oid is sha256 of the file
    size: int = 0
    pointer_size: int = 0


@dataclass
class HFModel:
    """HuggingFace model/dataset file metadata."""

    type: str = ""
    oid: str = ""
    size: int = 0
    path: str = ""
    local_size: int = 0
    needs_download: bool = True
    is_directory: bool = False
    is_lfs: bool = False
    appended_path: str = ""
    skip_downloading: bool = False
    filter_skip: bool = False
    download_link: str = ""
    lfs: Optional[HFLfs] = None


@dataclass
class R2Config:
    """Configuration for Cloudflare R2 storage."""

    account_id: str
    access_key_id: str
    access_key_secret: str
    bucket_name: str
    region: str = "auto"
    subfolder: str = ""


@dataclass
class FileProgress:
    """Progress tracking for individual files."""

    status: FileStatus
    error_message: Optional[str] = None
    retry_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    downloaded_bytes: int = 0
    uploaded_bytes: int = 0


@dataclass
class DownloadState:
    """Enhanced download state with detailed tracking."""

    model_name: str
    branch: str
    total_files: int = 0
    file_progress: Dict[str, FileProgress] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)
    start_time: datetime = field(default_factory=datetime.now)
    corrupted_files: List[str] = field(default_factory=list)
    processing_files: Dict[str, int] = field(default_factory=dict)  # file -> worker_id


class R2FileCache:
    """Thread-safe cache with atomic operations."""

    def __init__(self):
        self.files: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._processing_files: Set[str] = set()

    def try_claim_file(self, key: str, worker_id: int) -> bool:
        """Atomically claim a file for processing."""
        with self._lock:
            if key in self._processing_files:
                return False
            self._processing_files.add(key)
            return True

    def release_file(self, key: str) -> None:
        """Release a file after processing."""
        with self._lock:
            self._processing_files.discard(key)

    def exists_with_size_atomic(
        self, key: str, expected_size: int
    ) -> Tuple[bool, bool]:
        """Check existence and claim in one atomic operation.
        Returns: (exists_with_correct_size, successfully_claimed)
        """
        with self._lock:
            if key in self._processing_files:
                return False, False

            size = self.files.get(key)
            exists_with_size = size is not None and size == expected_size

            if not exists_with_size:
                self._processing_files.add(key)

            return exists_with_size, not exists_with_size

    def update_file(self, key: str, size: int) -> None:
        """Update file in cache."""
        with self._lock:
            self.files[key] = size
            self._processing_files.discard(key)

    def exists(self, key: str) -> bool:
        """Check if a file exists in the cache."""
        with self._lock:
            return key in self.files

    def exists_with_size(self, key: str, expected_size: int) -> bool:
        """Check if a file exists with the expected size."""
        with self._lock:
            size = self.files.get(key)
            return size is not None and size == expected_size

    def get_size(self, key: str) -> Tuple[int, bool]:
        """Get the file size and existence boolean."""
        with self._lock:
            size = self.files.get(key)
            return (size, size is not None)  # type: ignore


class ProgressTracker:
    """Thread-safe progress bar manager."""

    def __init__(self, total: int, description: str):
        """Initialize progress tracker.

        Args:
            total: Total size in bytes
            description: Description for the progress bar
        """
        self.progress_bar = tqdm(
            total=total,
            desc=description,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            mininterval=0.1,
            maxinterval=1.0,
            smoothing=0.1,
        )
        self._lock = threading.Lock()

    def update(self, n: int) -> None:
        """Update progress by n bytes."""
        with self._lock:
            self.progress_bar.update(n)

    def close(self) -> None:
        """Close the progress bar."""
        with self._lock:
            self.progress_bar.close()


class CorruptionRecoveryManager:
    """Manages automatic retry of corrupted files."""

    def __init__(self, max_retries: int = 3):
        self.retry_queue = queue.Queue()
        self.max_retries = max_retries
        self.retry_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self.shutdown_event = threading.Event()

    def add_corrupted_file(self, file_path: str, file_info: HFModel) -> bool:
        """Add a corrupted file to the retry queue."""
        with self._lock:
            retry_count = self.retry_counts.get(file_path, 0)
            if retry_count >= self.max_retries:
                print(f"File {file_path} exceeded max retries ({self.max_retries})")
                return False

            self.retry_counts[file_path] = retry_count + 1
            self.retry_queue.put((file_path, file_info))
            return True

    def get_retry_file(self, timeout: float = 1.0) -> Optional[Tuple[str, HFModel]]:
        """Get a file to retry from the queue."""
        try:
            return self.retry_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def has_pending_retries(self) -> bool:
        """Check if there are pending retries."""
        return not self.retry_queue.empty()


class WorkerManager:
    """Manages worker threads with health monitoring and recovery."""

    def __init__(self, max_workers: int, task_timeout: int = 3600):
        self.max_workers = max_workers
        self.task_timeout = task_timeout
        self.executor = None
        self.active_tasks: Dict[str, Tuple[concurrent.futures.Future, float]] = {}
        self._lock = threading.Lock()
        self.monitor_thread = None
        self.shutdown_event = threading.Event()

    def start(self):
        """Start the worker manager."""
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        self.monitor_thread = threading.Thread(
            target=self._monitor_workers, daemon=True
        )
        self.monitor_thread.start()

    def submit_task(
        self, task_id: str, func, *args, **kwargs
    ) -> concurrent.futures.Future:
        """Submit a task with monitoring."""
        future = self.executor.submit(func, *args, **kwargs)  # type: ignore

        with self._lock:
            self.active_tasks[task_id] = (future, time.time())

        def cleanup_callback(fut):
            with self._lock:
                self.active_tasks.pop(task_id, None)

        future.add_done_callback(cleanup_callback)
        return future

    def _monitor_workers(self):
        """Monitor worker threads and handle timeouts."""
        while not self.shutdown_event.is_set():
            current_time = time.time()
            timed_out_tasks = []

            with self._lock:
                for task_id, (future, start_time) in self.active_tasks.items():
                    if (
                        not future.done()
                        and current_time - start_time > self.task_timeout
                    ):
                        timed_out_tasks.append((task_id, future))

            for task_id, future in timed_out_tasks:
                print(f"Task {task_id} timed out after {self.task_timeout}s")
                future.cancel()

                # Remove from active tasks
                with self._lock:
                    self.active_tasks.pop(task_id, None)

            time.sleep(10)  # Check every 10 seconds

    def wait_all_with_timeout(
        self,
        futures: List[concurrent.futures.Future],
        timeout: int = None,  # type: ignore
    ) -> Tuple[List[Any], int]:
        """Wait for all futures with individual timeout handling.

        Returns:
            Tuple of (results, failure_count)
        """
        results = []
        failure_count = 0

        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except concurrent.futures.TimeoutError:
                print(f"Future {i} timed out after {timeout}s")
                future.cancel()
                results.append(None)
                failure_count += 1
            except Exception as e:
                print(f"Future {i} failed with error: {e}")
                import traceback

                traceback.print_exc()
                results.append(None)
                failure_count += 1

        return results, failure_count

    def shutdown(self):
        """Shutdown the worker manager."""
        self.shutdown_event.set()
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)


def create_optimized_session() -> requests.Session:
    """Create an optimized HTTP session with DNS caching."""
    session = requests.Session()

    # Use a custom resolver that prefers Cloudflare DNS
    dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
    dns.resolver.default_resolver.nameservers = ["1.1.1.1", "1.0.0.1"]

    # Configure the connection pooling and retry strategy
    adapter = HTTPAdapter(
        pool_connections=NUM_CONNECTIONS,
        pool_maxsize=NUM_CONNECTIONS,
        max_retries=Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_DELAY,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        ),
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    if REQUIRES_AUTH:
        session.headers.update({"Authorization": f"Bearer {AUTH_TOKEN}"})

    # Add a user agent to mimic browser behavior
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    return session


def create_r2_client(r2cfg: R2Config) -> BaseClient:
    """Create an optimized R2 client.

    Args:
        r2cfg: R2 configuration parameters

    Returns:
        Configured boto3 S3 client for R2
    """
    return boto3.client(
        "s3",
        endpoint_url=f"https://{r2cfg.account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=r2cfg.access_key_id,
        aws_secret_access_key=r2cfg.access_key_secret,
        region_name=r2cfg.region,
        config=Config(
            max_pool_connections=256,
            connect_timeout=10,
            read_timeout=30 * 60,  # 30 minutes
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def format_size(bytes_: int) -> str:
    """Format bytes into human-readable string.

    Args:
        bytes_: Size in bytes

    Returns:
        Formatted string (e.g., "1.2 MB")
    """
    if bytes_ < 1024:
        return f"{bytes_} B"

    for unit in ["K", "M", "G", "T", "P"]:
        bytes_ = int(bytes_ / 1024)
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}B"

    return f"{bytes_:.1f} EB"


def build_r2_cache(r2cfg: R2Config, prefix: str) -> R2FileCache:
    """Pre-fetch existing files in R2 to a local cache.

    Args:
        r2cfg: R2 configuration
        prefix: Prefix to filter objects

    Returns:
        Cache of file keys and their sizes
    """
    print("Building cache of existing files in R2...")
    client = create_r2_client(r2cfg)
    cache = R2FileCache()

    start_time = time.time()
    count = 0
    paginator = client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=r2cfg.bucket_name, Prefix=prefix):
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            cache.files[obj["Key"]] = obj["Size"]
            count += 1

        if count % 1000 == 0:
            print(f"Cached {count} files...")

    elapsed = time.time() - start_time
    print(f"Cached {count} files in {elapsed:.2f}s")
    return cache


def retry_with_backoff(
    operation: Callable, max_retries: int, initial_backoff: float, max_backoff: float
) -> Any:
    """Retry an operation with exponential backoff.

    Args:
        operation: Function to retry
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds

    Returns:
        Result of the operation if successful

    Raises:
        Exception: Operation failed after all retries
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            last_error = e

            if not is_transient_error(e):
                raise Exception(f"Permanent error (not retrying): {e}")

            if attempt == max_retries - 1:
                break

            backoff = min(initial_backoff * (2**attempt), max_backoff)
            jitter = backoff * (0.8 + 0.4 * random.random())

            print(
                f"Retrying operation after {jitter:.3f}s (attempt {attempt + 1}/{max_retries}): {e}"
            )
            time.sleep(jitter)

    raise Exception(f"Operation failed after {max_retries} retries: {last_error}")


def is_transient_error(error: Exception) -> bool:
    """Determine if an error is transient and retryable.

    Args:
        error: Exception to check

    Returns:
        True if the error is potentially transient
    """
    error_str = str(error)

    # Network errors
    if isinstance(error, (requests.Timeout, requests.ConnectionError)):
        return True

    # HTTP status codes
    if any(code in error_str for code in ["429", "500", "502", "503", "504"]):
        return True

    # AWS/R2 specific errors
    if any(
        err in error_str
        for err in [
            "RequestTimeout",
            "SlowDown",
            "InternalError",
            "connection reset",
            "EOF",
            "broken pipe",
        ]
    ):
        return True

    return False


def save_download_state(state: DownloadState, model_name: str) -> None:
    """Save current download state to a file.

    Args:
        state: Current download state
        model_name: Name of the model/dataset
    """
    state_dir = Path(tempfile.gettempdir()) / "hfdownloader-state"
    state_dir.mkdir(exist_ok=True, parents=True)

    safe_model_name = model_name.replace("/", "_")
    state_file = state_dir / f"{safe_model_name}.json"

    state.last_update = datetime.now()

    # Convert the state to a dictionary with proper serialization
    state_dict = {
        "model_name": state.model_name,
        "branch": state.branch,
        "total_files": state.total_files,
        "file_progress": {},
        "last_update": state.last_update.isoformat(),
        "start_time": state.start_time.isoformat(),
        "corrupted_files": state.corrupted_files,
        "processing_files": state.processing_files,
    }

    # Serialize FileProgress objects
    for path, progress in state.file_progress.items():
        state_dict["file_progress"][path] = {
            "status": progress.status.value,
            "error_message": progress.error_message,
            "retry_count": progress.retry_count,
            "last_update": progress.last_update.isoformat(),
            "downloaded_bytes": progress.downloaded_bytes,
            "uploaded_bytes": progress.uploaded_bytes,
        }

    with open(state_file, "w") as f:
        json.dump(state_dict, f, indent=2)


def load_download_state(model_name: str) -> Optional[DownloadState]:
    """Load download state from a file.

    Args:
        model_name: Name of the model/dataset

    Returns:
        Loaded download state or None if not found
    """
    safe_model_name = model_name.replace("/", "_")
    state_dir = Path(tempfile.gettempdir()) / "hfdownloader-state"
    state_file = state_dir / f"{safe_model_name}.json"

    if not state_file.exists():
        return None

    try:
        with open(state_file, "r") as f:
            state_dict = json.load(f)

        state = DownloadState(
            model_name=state_dict["model_name"],
            branch=state_dict["branch"],
            total_files=state_dict["total_files"],
            file_progress={},
            corrupted_files=state_dict.get("corrupted_files", []),
            processing_files=state_dict.get("processing_files", {}),
        )

        # Load FileProgress objects
        for path, progress_dict in state_dict.get("file_progress", {}).items():
            state.file_progress[path] = FileProgress(
                status=FileStatus(progress_dict["status"]),
                error_message=progress_dict.get("error_message"),
                retry_count=progress_dict.get("retry_count", 0),
                last_update=datetime.fromisoformat(progress_dict["last_update"]),
                downloaded_bytes=progress_dict.get("downloaded_bytes", 0),
                uploaded_bytes=progress_dict.get("uploaded_bytes", 0),
            )

        state.last_update = datetime.fromisoformat(state_dict["last_update"])
        state.start_time = datetime.fromisoformat(state_dict["start_time"])

        if (datetime.now() - state.last_update).days > 7:
            state_file.unlink(missing_ok=True)
            return None

        return state
    except Exception as e:
        print(f"Failed to load download state: {e}")
        return None


def fetch_file_list(url: str, session: requests.Session) -> List[HFModel]:
    """Fetch and parse file list from Hugging Face API.

    Args:
        url: API URL
        session: HTTP session

    Returns:
        List of file models
    """

    def fetch_operation():
        resp = session.get(url, timeout=(10, 60))
        resp.raise_for_status()
        data = resp.json()

        # Convert to our internal models
        return [
            HFModel(
                type=item.get("type", ""),
                oid=item.get("oid", ""),
                size=item.get("size", 0),
                path=item.get("path", ""),
                is_directory=item.get("type") == "directory",
                is_lfs="lfs" in item and item["lfs"] is not None,
                lfs=HFLfs(
                    oid=item.get("lfs", {}).get("oid", ""),
                    size=item.get("lfs", {}).get("size", 0),
                    pointer_size=item.get("lfs", {}).get("pointerSize", 0),
                )
                if "lfs" in item and item["lfs"] is not None
                else None,
            )
            for item in data
        ]

    return retry_with_backoff(
        fetch_operation, max_retries=5, initial_backoff=1.0, max_backoff=10.0
    )


def verify_parquet_file(
    client: BaseClient, bucket: str, key: str, expected_size: int
) -> bool:
    """Verify a parquet file in R2 by checking magic numbers.

    Args:
        client: R2 client
        bucket: Bucket name
        key: Object key
        expected_size: Expected file size

    Returns:
        True if file is valid

    Raises:
        Exception: If verification fails
    """
    # First, get the actual file size from R2
    try:
        head_obj = client.head_object(Bucket=bucket, Key=key)
        actual_size = head_obj["ContentLength"]
    except Exception as e:
        raise Exception(f"Failed to get file size: {e}")

    # Check if file is too small to be a valid parquet file
    if actual_size < 8:
        raise Exception(
            f"File too small to be a valid parquet file (size: {actual_size})"
        )

    # Check if expected size matches actual size
    if expected_size != actual_size:
        raise Exception(
            f"Size mismatch - expected: {expected_size}, actual: {actual_size}"
        )

    # Get first 4 bytes (header)
    header_obj = client.get_object(Bucket=bucket, Key=key, Range="bytes=0-3")
    header = header_obj["Body"].read(4)

    # Get last 4 bytes (footer)
    footer_obj = client.get_object(
        Bucket=bucket, Key=key, Range=f"bytes={actual_size - 4}-{actual_size - 1}"
    )
    footer = footer_obj["Body"].read(4)

    # Verify magic numbers
    expected_magic = b"PAR1"
    if header != expected_magic:
        raise Exception("Invalid parquet header magic number")

    if footer != expected_magic:
        raise Exception("Invalid parquet footer magic number")

    return True


def verify_local_parquet(file_path: str) -> bool:
    """Verify a local parquet file.

    Args:
        file_path: Path to parquet file

    Returns:
        True if file is valid

    Raises:
        Exception: If verification fails
    """
    with open(file_path, "rb") as f:
        # Read header
        header = f.read(4)

        # Seek to end - 4
        f.seek(-4, os.SEEK_END)
        footer = f.read(4)

    # Verify magic numbers
    expected_magic = b"PAR1"
    if header != expected_magic:
        raise Exception("Invalid parquet header magic number")

    if footer != expected_magic:
        raise Exception("Invalid parquet footer magic number")

    return True


def optimize_parquet_file(
    input_file: str,
    output_file: str,
    compression_level: int = 3,
    row_group_size: int = 1000,
) -> None:
    """Optimize a parquet file with ZSTD compression and custom row group size.

    Args:
        input_file: Path to input parquet file
        output_file: Path to output optimized parquet file
        compression_level: ZSTD compression level (1-22, default: 3)
        row_group_size: Number of rows per row group (default: 1000)
    """
    print(f"Optimizing parquet file: {input_file}")

    # Read the original parquet file
    table = pq.read_table(input_file)

    # Write optimized parquet file with ZSTD compression
    pq.write_table(
        table,
        output_file,
        compression="ZSTD",
        compression_level=compression_level,
        use_dictionary=True,  # Enable dictionary encoding
        data_page_size=4 * 1024 * 1024,  # 4MB data pages
        row_group_size=row_group_size,
        version="2.6",  # Modern Parquet format
        use_deprecated_int96_timestamps=False,
    )

    # Report optimization results
    original_size = os.path.getsize(input_file) / (1024 * 1024)
    optimized_size = os.path.getsize(output_file) / (1024 * 1024)

    print("Optimization complete:")
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Optimized size: {optimized_size:.2f} MB")
    print(f"  Compression ratio: {(1 - optimized_size / original_size) * 100:.1f}%")


def stream_simple_to_r2(
    r2cfg: R2Config,
    local_file: str,
    key: str,
    content_length: int,
    progress: Optional[ProgressTracker] = None,
    compression_level: Optional[int] = None,
    row_group_size: Optional[int] = None,
) -> None:
    """Upload a file to R2 in a single operation.

    Args:
        r2cfg: R2 configuration
        local_file: Path to local file
        key: Destination key in R2
        content_length: File size
        progress: Progress tracker
        compression_level: ZSTD compression level for parquet files
        row_group_size: Row group size for parquet files
    """
    client = create_r2_client(r2cfg)
    upload_file = local_file
    actual_content_length = content_length

    # For parquet files, verify and optionally optimize
    if key.endswith(".parquet"):
        verify_local_parquet(local_file)

        # Optimize if compression level or row group size specified
        if compression_level is not None or row_group_size is not None:
            # Create a temporary optimized file
            optimized_file = local_file + ".optimized"
            try:
                # Use defaults if not specified
                comp_level = compression_level if compression_level is not None else 3
                rg_size = row_group_size if row_group_size is not None else 1000

                optimize_parquet_file(local_file, optimized_file, comp_level, rg_size)
                upload_file = optimized_file
                actual_content_length = os.path.getsize(optimized_file)
            except Exception as e:
                # Clean up on error
                if os.path.exists(optimized_file):
                    os.unlink(optimized_file)
                raise Exception(f"Failed to optimize parquet file: {e}")

    if progress is None:
        progress = ProgressTracker(actual_content_length, os.path.basename(key))

    # Create a custom callback to track progress
    def progress_callback(bytes_transferred):
        progress.update(bytes_transferred)

    try:
        # Upload the file
        client.upload_file(
            upload_file, r2cfg.bucket_name, key, Callback=progress_callback
        )

        # Verify after upload for parquet files
        if key.endswith(".parquet"):
            # Use expected size for verification since file was optimized
            verify_size = (
                actual_content_length if upload_file != local_file else content_length
            )
            verify_parquet_file(client, r2cfg.bucket_name, key, verify_size)
    except Exception as e:
        # Delete the failed upload
        try:
            client.delete_object(Bucket=r2cfg.bucket_name, Key=key)
        except Exception:
            pass
        raise Exception(f"Upload failed: {e}")
    finally:
        # Clean up optimized file if it was created
        if upload_file != local_file and os.path.exists(upload_file):
            os.unlink(upload_file)


def stream_multipart_to_r2(
    r2cfg: R2Config,
    local_file: str,
    key: str,
    content_length: int,
    progress: Optional[ProgressTracker] = None,
    max_workers: int = MAX_PARTS_PER_FILE,
    compression_level: Optional[int] = None,
    row_group_size: Optional[int] = None,
) -> None:
    """Upload a large file to R2 using multipart upload.

    Args:
        r2cfg: R2 configuration
        local_file: Path to local file
        key: Destination key in R2
        content_length: File size
        progress: Progress tracker
        max_workers: Maximum concurrent part uploads
        compression_level: ZSTD compression level for parquet files
        row_group_size: Row group size for parquet files
    """
    client = create_r2_client(r2cfg)
    upload_file = local_file
    actual_content_length = content_length

    # For parquet files, verify and optionally optimize
    if key.endswith(".parquet"):
        verify_local_parquet(local_file)

        # Optimize if compression level or row group size specified
        if compression_level is not None or row_group_size is not None:
            # Create a temporary optimized file
            optimized_file = local_file + ".optimized"
            try:
                # Use defaults if not specified
                comp_level = compression_level if compression_level is not None else 3
                rg_size = row_group_size if row_group_size is not None else 1000

                optimize_parquet_file(local_file, optimized_file, comp_level, rg_size)
                upload_file = optimized_file
                actual_content_length = os.path.getsize(optimized_file)
            except Exception as e:
                # Clean up on error
                if os.path.exists(optimized_file):
                    os.unlink(optimized_file)
                raise Exception(f"Failed to optimize parquet file: {e}")

    # Check for existing multipart uploads
    existing_uploads = client.list_multipart_uploads(
        Bucket=r2cfg.bucket_name, Prefix=key
    )

    upload_id = None
    if "Uploads" in existing_uploads:
        for upload in existing_uploads["Uploads"]:
            if upload["Key"] == key:
                upload_id = upload["UploadId"]
                print(f"Found existing multipart upload for {key} (ID: {upload_id})")

                # Get existing parts to potentially resume from
                existing_parts = client.list_parts(
                    Bucket=r2cfg.bucket_name, Key=key, UploadId=upload_id
                )

                if "Parts" in existing_parts and existing_parts["Parts"]:
                    print(
                        f"Found {len(existing_parts['Parts'])} previously uploaded parts"
                    )

                break

    # Start a new multipart upload if we didn't find one to resume
    if not upload_id:
        # Abort any existing uploads for this key
        if "Uploads" in existing_uploads:
            for upload in existing_uploads["Uploads"]:
                if upload["Key"] == key:
                    try:
                        client.abort_multipart_upload(
                            Bucket=r2cfg.bucket_name,
                            Key=upload["Key"],
                            UploadId=upload["UploadId"],
                        )
                    except Exception as e:
                        print(f"Warning: Failed to abort incomplete upload: {e}")

        # Create new multipart upload
        upload = client.create_multipart_upload(Bucket=r2cfg.bucket_name, Key=key)
        upload_id = upload["UploadId"]
        print(f"Created new multipart upload for {key} (ID: {upload_id})")

    # Calculate optimal part size (minimum 5MB, maximum 5GB)
    part_size = max(
        min(actual_content_length // max_workers, 5 * 1024 * 1024 * 1024),
        5 * 1024 * 1024,
    )

    if progress is None:
        progress = ProgressTracker(actual_content_length, os.path.basename(key))

    # Track completed parts
    completed_parts = []
    lock = threading.Lock()

    # Worker function to upload a single part
    def upload_part(part_number, start_byte, end_byte):
        try:
            # Upload the part (use the correct file)
            with open(upload_file, "rb") as f:
                f.seek(start_byte)
                data = f.read(end_byte - start_byte)

            response = client.upload_part(
                Bucket=r2cfg.bucket_name,
                Key=key,
                UploadId=upload_id,
                PartNumber=part_number,
                Body=data,
            )

            with lock:
                completed_parts.append(
                    {"PartNumber": part_number, "ETag": response["ETag"]}
                )
                progress.update(end_byte - start_byte)

            return True
        except Exception as e:
            print(f"Error uploading part {part_number}: {e}")
            return False

    # Generate part information
    part_info = []
    total_parts = math.ceil(actual_content_length / part_size)

    for i in range(1, total_parts + 1):
        start_byte = (i - 1) * part_size
        end_byte = min(i * part_size, actual_content_length)
        part_info.append((i, start_byte, end_byte))

    # Use ThreadPoolExecutor to upload parts in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(upload_part, part_num, start, end)
            for part_num, start, end in part_info
        ]

        # Wait for all uploads to complete
        for future in futures:
            if not future.result():
                # Abort on failure
                client.abort_multipart_upload(
                    Bucket=r2cfg.bucket_name, Key=key, UploadId=upload_id
                )
                raise Exception(f"Failed to upload parts for {key}")

    # Sort completed parts by part number
    completed_parts.sort(key=lambda p: p["PartNumber"])

    # Complete the multipart upload
    client.complete_multipart_upload(
        Bucket=r2cfg.bucket_name,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": completed_parts},
    )

    # Verify the upload for parquet files
    if key.endswith(".parquet"):
        try:
            # Use expected size for verification since file was optimized
            verify_size = (
                actual_content_length if upload_file != local_file else content_length
            )
            verify_parquet_file(client, r2cfg.bucket_name, key, verify_size)
        except Exception as e:
            # Delete the failed upload
            client.delete_object(Bucket=r2cfg.bucket_name, Key=key)
            # Clean up optimized file if it was created
            if upload_file != local_file and os.path.exists(upload_file):
                os.unlink(upload_file)
            raise Exception(f"Post-upload verification failed: {e}")

    # Clean up optimized file if it was created
    if upload_file != local_file and os.path.exists(upload_file):
        os.unlink(upload_file)


def process_hf_folder_tree(
    is_dataset: bool,
    model_dataset_name: str,
    model_branch: str,
    folder_name: str,
    silent_mode: bool,
    r2cfg: R2Config,
    process_files: Callable[[List[HFModel]], None],
    hf_prefix: str,
    session: requests.Session,
) -> None:
    """Process a folder in the Hugging Face repository.

    Args:
        is_dataset: True if processing a dataset, False for model
        model_dataset_name: Model or dataset name
        model_branch: Branch name
        folder_name: Current folder to process
        silent_mode: Suppress verbose output
        r2cfg: R2 configuration
        process_files: Callback to process files
        hf_prefix: Prefix for HF paths
        session: HTTP session
    """
    if not silent_mode:
        print(f"🔍 Scanning: {folder_name}")

    # Build the correct API URL
    url = ""
    if is_dataset:
        if folder_name == "":
            url = JSON_DATASET_FILE_TREE_URL.format(
                model_dataset_name, model_branch, hf_prefix
            )
        else:
            url = JSON_DATASET_FILE_TREE_URL.format(
                model_dataset_name, model_branch, folder_name
            )

    if not silent_mode:
        print(f"📡 API URL: {url}")

    # Make request and get files
    files = fetch_file_list(url, session)

    if not silent_mode:
        print(f"📂 Found {len(files)} items in {folder_name}")

    parquet_files = []
    for file in files:
        if file.path.endswith(".parquet") and file.size > 0:
            if file.is_lfs:
                file.download_link = LFS_DATASET_RESOLVER_URL.format(
                    model_dataset_name, model_branch, file.path
                )
            else:
                file.download_link = RAW_DATASET_FILE_URL.format(
                    model_dataset_name, model_branch, file.path
                )
            parquet_files.append(file)
        elif file.is_directory:
            if not silent_mode:
                print(f"📁 Entering directory: {file.path}")

            process_hf_folder_tree(
                is_dataset,
                model_dataset_name,
                model_branch,
                file.path,
                silent_mode,
                r2cfg,
                process_files,
                hf_prefix,
                session,
            )

    if parquet_files:
        if not silent_mode:
            print(
                f"📦 Processing {len(parquet_files)} parquet files from {folder_name}"
            )
        process_files(parquet_files)


def download_model(
    model_dataset_name: str,
    is_dataset: bool,
    destination_base_path: str,
    model_branch: str,
    token: str,
    silent_mode: bool,
    r2cfg: R2Config,
    hf_prefix: str,
    max_workers: int,
    compression_level: Optional[int] = None,
    row_group_size: Optional[int] = None,
) -> None:
    """Download a model or dataset from Hugging Face and upload to R2.

    Args:
        model_dataset_name: Model or dataset name
        is_dataset: True if downloading a dataset
        destination_base_path: Local path to save files
        model_branch: Branch name
        token: Authentication token
        silent_mode: Suppress verbose output
        r2cfg: R2 configuration
        hf_prefix: Prefix for HF paths
        max_workers: Maximum concurrent workers
        compression_level: ZSTD compression level for parquet files
        row_group_size: Row group size for parquet files
    """
    global REQUIRES_AUTH, AUTH_TOKEN

    # Set authentication globals
    if token:
        REQUIRES_AUTH = True
        AUTH_TOKEN = token

    # Load existing download state
    download_state = load_download_state(model_dataset_name)

    # Initialize new state if needed
    if download_state is None:
        download_state = DownloadState(
            model_name=model_dataset_name,
            branch=model_branch,
            total_files=0,
            file_progress={},
            start_time=datetime.now(),
            last_update=datetime.now(),
        )
        print("🆕 Starting new download session")
    else:
        elapsed = datetime.now() - download_state.start_time
        print(
            f"🔄 Resuming download from previous session (started {elapsed.total_seconds() // 60} minutes ago)"
        )
        completed_count = sum(
            1
            for p in download_state.file_progress.values()
            if p.status == FileStatus.COMPLETED
        )
        print(
            f"💾 Previously completed: {completed_count}/{download_state.total_files} files"
        )

    # Initialize enhanced components
    cache = R2FileCache()  # Use enhanced cache
    recovery_manager = CorruptionRecoveryManager(max_retries=3)
    worker_manager = WorkerManager(max_workers=max_workers, task_timeout=3600)

    # Build initial cache from R2
    r2_cache = build_r2_cache(r2cfg, f"{r2cfg.subfolder}/")
    for key, size in r2_cache.files.items():
        cache.files[key] = size

    # Use the provided maxWorkers parameter
    if max_workers <= 0:
        max_workers = 16  # Default
    print(f"Using {max_workers} worker threads for parallel downloads")

    # Create the session
    session = create_optimized_session()

    # Start worker manager
    worker_manager.start()
    futures = []

    # Stop event for threads
    stop_processing = threading.Event()

    def process_file_enhanced(file: HFModel, worker_id: int) -> bool:
        """Enhanced file processing with proper state tracking and recovery."""
        # Remove the prefix properly - ensure we only remove exact prefix
        if file.path.startswith(hf_prefix):
            path_without_prefix = file.path[len(hf_prefix) :].lstrip("/")
        else:
            path_without_prefix = file.path
        r2_key = f"{r2cfg.subfolder}/{path_without_prefix}"

        # Try to claim the file atomically
        exists_with_size, claimed = cache.exists_with_size_atomic(r2_key, file.size)

        if exists_with_size:
            print(
                f"[Worker {worker_id}] File {r2_key} already exists with correct size"
            )
            download_state.file_progress[file.path] = FileProgress(
                status=FileStatus.COMPLETED
            )
            return True

        if not claimed:
            print(
                f"[Worker {worker_id}] File {r2_key} is being processed by another worker"
            )
            return True

        try:
            # Update state to downloading
            download_state.file_progress[file.path] = FileProgress(
                status=FileStatus.DOWNLOADING
            )
            download_state.processing_files[file.path] = worker_id
            save_download_state(download_state, model_dataset_name)

            # Download file
            download_url = file.download_link or LFS_DATASET_RESOLVER_URL.format(
                model_dataset_name, model_branch, file.path
            )

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp_file = temp.name

            # Setup progress bar
            progress = ProgressTracker(file.size, os.path.basename(file.path))

            # Download file with retry logic
            def download_operation():
                # Stream download to file with progress reporting
                with open(temp_file, "wb") as f:
                    with session.get(
                        download_url, stream=True, timeout=(30, 1800)
                    ) as response:
                        response.raise_for_status()
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                progress.update(len(chunk))

            retry_with_backoff(
                download_operation, max_retries=5, initial_backoff=1.0, max_backoff=30.0
            )

            # Update state to uploading
            download_state.file_progress[file.path].status = FileStatus.UPLOADING
            save_download_state(download_state, model_dataset_name)

            # Upload to R2
            if file.size > MULTIPART_THRESHOLD:
                stream_multipart_to_r2(
                    r2cfg,
                    temp_file,
                    r2_key,
                    file.size,
                    progress,
                    compression_level=compression_level,
                    row_group_size=row_group_size,
                )
            else:
                stream_simple_to_r2(
                    r2cfg,
                    temp_file,
                    r2_key,
                    file.size,
                    progress,
                    compression_level=compression_level,
                    row_group_size=row_group_size,
                )

            # Verify if parquet
            if r2_key.endswith(".parquet"):
                client = create_r2_client(r2cfg)
                # Get actual file size from r2 for verification
                try:
                    head_obj = client.head_object(Bucket=r2cfg.bucket_name, Key=r2_key)
                    actual_uploaded_size = head_obj["ContentLength"]
                    verify_parquet_file(
                        client, r2cfg.bucket_name, r2_key, actual_uploaded_size
                    )
                except Exception as e:
                    print(
                        f"[Worker {worker_id}] Warning: Failed to verify parquet file: {e}"
                    )

            # Mark as completed
            download_state.file_progress[file.path].status = FileStatus.COMPLETED
            cache.update_file(r2_key, file.size)

            # Close progress bar
            progress.close()

            print(f"[Worker {worker_id}] Successfully completed {r2_key}")
            return True

        except Exception as e:
            error_msg = str(e)

            # Print detailed error information
            print(
                f"[Worker {worker_id}] Error processing {file.path}: {type(e).__name__}: {e}"
            )
            if not silent_mode:
                import traceback

                traceback.print_exc()

            if "corrupt" in error_msg.lower() or "invalid parquet" in error_msg.lower():
                # Handle corruption
                download_state.file_progress[file.path].status = FileStatus.CORRUPTED
                download_state.corrupted_files.append(file.path)

                # Add to retry queue
                if recovery_manager.add_corrupted_file(file.path, file):
                    print(
                        f"[Worker {worker_id}] Added corrupted file {file.path} to retry queue"
                    )
                else:
                    print(f"[Worker {worker_id}] File {file.path} exceeded retry limit")
            else:
                # Other errors
                download_state.file_progress[file.path].status = FileStatus.FAILED
                download_state.file_progress[file.path].error_message = error_msg

            return False

        finally:
            # Clean up
            cache.release_file(r2_key)
            download_state.processing_files.pop(file.path, None)
            save_download_state(download_state, model_dataset_name)

            # Clean up temp file
            if "temp_file" in locals() and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:  # noqa: E722
                    pass

    # Setup file processor callback
    def process_files_callback(files: List[HFModel]) -> None:
        """Process a batch of files."""
        pending_files = []
        total_size = 0
        skipped_size = 0
        skipped_count = 0

        # Update total file count in download state
        file_count = 0
        for file in files:
            if not file.is_directory and not file.filter_skip and file.size > 0:
                file_count += 1

        if download_state.total_files == 0:
            download_state.total_files = file_count
        else:
            download_state.total_files += file_count

        # Save state
        try:
            save_download_state(download_state, model_dataset_name)
        except Exception as e:
            print(f"Warning: Failed to save download state: {e}")

        # First, filter files that need to be processed
        for file in files:
            if not file.is_directory and not file.filter_skip and file.size > 0:
                # Remove the prefix properly - ensure we only remove exact prefix
                if file.path.startswith(hf_prefix):
                    path_without_prefix = file.path[len(hf_prefix) :].lstrip("/")
                else:
                    path_without_prefix = file.path
                r2_key = f"{r2cfg.subfolder}/{path_without_prefix}"
                total_size += file.size

                # Check if file is already completed
                if file.path in download_state.file_progress:
                    status = download_state.file_progress[file.path].status
                    if status == FileStatus.COMPLETED:
                        print(
                            f"Skipping {file.path} - marked as completed in saved state"
                        )
                        skipped_size += file.size
                        skipped_count += 1
                        continue
                    elif status == FileStatus.CORRUPTED:
                        # Add corrupted files to retry queue
                        recovery_manager.add_corrupted_file(file.path, file)
                        continue

                if cache.exists_with_size(r2_key, file.size):
                    # File exists in R2 with correct size - mark as completed
                    download_state.file_progress[file.path] = FileProgress(
                        status=FileStatus.COMPLETED
                    )
                    skipped_size += file.size
                    skipped_count += 1
                    continue
                else:
                    existing_size, exists = cache.get_size(r2_key)
                    if exists:
                        # File exists but with incorrect size
                        print(
                            f"File {r2_key} exists with incorrect size (expected: {format_size(file.size)}, "
                            + f"actual: {format_size(existing_size)}). Will be deleted and reuploaded."
                        )

                pending_files.append(file)

        # Print summary
        if not silent_mode:
            print("\n=== Processing Summary ===")
            print(f"Total files found: {len(files)}")
            print(f"Files already in R2: {skipped_count}")
            print(f"Files to process: {len(pending_files)}")
            print(f"Total size: {format_size(total_size)}")
            print(f"Skipped size: {format_size(skipped_size)}")
            print(f"Remaining size: {format_size(total_size - skipped_size)}\n")

        # Submit jobs to worker manager
        for file in pending_files:
            if not silent_mode:
                print(f"Queueing: {file.path} ({format_size(file.size)})")

            task_id = f"{model_dataset_name}_{file.path}"
            worker_id = hash(task_id) % 10000

            future = worker_manager.submit_task(
                task_id, process_file_enhanced, file, worker_id
            )
            futures.append(future)

    # Add retry processing thread
    def process_retry_queue():
        """Process files from the retry queue."""
        while not stop_processing.is_set() or recovery_manager.has_pending_retries():
            retry_file = recovery_manager.get_retry_file(timeout=1.0)
            if retry_file:
                file_path, file_info = retry_file
                print(f"Processing retry for: {file_path}")

                task_id = f"{model_dataset_name}_{file_path}_retry"
                worker_id = hash(task_id) % 10000

                future = worker_manager.submit_task(
                    task_id, process_file_enhanced, file_info, worker_id
                )
                futures.append(future)
            else:
                time.sleep(1)

    retry_thread = threading.Thread(target=process_retry_queue, daemon=True)
    retry_thread.start()

    try:
        # Start processing
        process_hf_folder_tree(
            is_dataset,
            model_dataset_name,
            model_branch,
            "",
            silent_mode,
            r2cfg,
            process_files_callback,
            hf_prefix,
            session,
        )

        # Wait for all futures with timeout
        results, failure_count = worker_manager.wait_all_with_timeout(
            futures, timeout=7200
        )

        if failure_count > 0:
            print(f"⚠️ {failure_count} tasks failed during processing")

        # Wait for retry queue to empty
        print("Waiting for retry queue to complete...")
        retry_attempts = 0
        while (
            recovery_manager.has_pending_retries() and retry_attempts < 60
        ):  # Max 5 minutes
            time.sleep(5)
            retry_attempts += 1

        if recovery_manager.has_pending_retries():
            print("⚠️ Some files are still pending retry after timeout")

        # Stop retry thread
        stop_processing.set()
        retry_thread.join(timeout=5)

        # Check final status
        failed_files = []
        corrupted_files = []
        for path, progress in download_state.file_progress.items():
            if progress.status == FileStatus.FAILED:
                failed_files.append(path)
            elif progress.status == FileStatus.CORRUPTED:
                corrupted_files.append(path)

        if failed_files or corrupted_files:
            print("\n⚠️ WARNING: Some files were not successfully processed:")
            if failed_files:
                print(f"  - {len(failed_files)} files failed")
                for f in failed_files[:5]:  # Show first 5
                    print(f"    - {f}")
                if len(failed_files) > 5:
                    print(f"    ... and {len(failed_files) - 5} more")
            if corrupted_files:
                print(f"  - {len(corrupted_files)} files were corrupted")
                for f in corrupted_files[:5]:  # Show first 5
                    print(f"    - {f}")
                if len(corrupted_files) > 5:
                    print(f"    ... and {len(corrupted_files) - 5} more")

    except Exception as e:
        print(f"Error during processing: {e}")
        # Save state before returning error
        save_download_state(download_state, model_dataset_name)
        raise
    finally:
        # Stop processing
        stop_processing.set()

        # Stop retry thread if still running
        if retry_thread.is_alive():
            retry_thread.join(timeout=5)

        # Shutdown worker manager
        worker_manager.shutdown()

    # Save final state
    print("💾 Saving final download state")
    save_download_state(download_state, model_dataset_name)
    print("✅ Download and upload complete!")


def is_valid_model_name(model_name: str) -> bool:
    """Check if a model name is valid.

    Args:
        model_name: Model name to check

    Returns:
        True if the model name is valid
    """
    pattern = r"^[A-Za-z0-9_\-]+/[A-Za-z0-9\._\-]+$"
    return bool(re.match(pattern, model_name))


def cleanup_corrupted_files(r2cfg: R2Config, prefix: str, concurrency: int) -> None:
    """Clean up corrupted parquet files in R2.

    Args:
        r2cfg: R2 configuration
        prefix: Prefix for files to check
        concurrency: Number of concurrent workers
    """
    client = create_r2_client(r2cfg)
    total_files = 0
    corrupted_files = 0

    # Use ThreadPoolExecutor for parallel verification
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []

        # Function to verify a single file
        def verify_file(obj):
            nonlocal total_files, corrupted_files

            key = obj["Key"]
            size = obj["Size"]

            if not key.endswith(".parquet") or size < 8:
                return

            worker_id = threading.get_ident() % 10000
            print(
                f"[Worker {worker_id}] Checking file: {key} (size: {format_size(size)})"
            )

            try:
                # Check parquet file integrity
                verify_parquet_file(client, r2cfg.bucket_name, key, size)
                print(f"[Worker {worker_id}] ✅ Valid parquet file: {key}")
            except Exception as e:
                print(f"[Worker {worker_id}] ❌ Corrupted file: {key}, error: {e}")

                with threading.Lock():
                    corrupted_files += 1

                if "invalid parquet" in str(e).lower():
                    try:
                        client.delete_object(Bucket=r2cfg.bucket_name, Key=key)
                        print(f"[Worker {worker_id}] Deleted corrupted file: {key}")
                    except Exception as del_e:
                        print(
                            f"[Worker {worker_id}] Warning: Failed to delete file {key}: {del_e}"
                        )

            with threading.Lock():
                total_files += 1

        # List objects from R2
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=r2cfg.bucket_name, Prefix=prefix):
            if "Contents" not in page:
                continue

            print(f"Retrieved {len(page['Contents'])} objects with prefix {prefix}")

            for obj in page["Contents"]:
                futures.append(executor.submit(verify_file, obj))

        # Wait for all verifications to complete
        for future in futures:
            future.result()

    print("\n=== Summary ===")
    print(f"Total parquet files checked: {total_files}")
    print(f"Corrupted files found: {corrupted_files}")

    if total_files == 0:
        print("Warning: No parquet files found! Verify bucket and prefix.")

    print("Verification complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download datasets from Hugging Face to R2"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g. 'mlfoundations/dclm-baseline-1.0-parquet')",
    )
    parser.add_argument("--branch", default="main", help="Branch name")
    parser.add_argument("--r2-account", required=True, help="R2 account ID")
    parser.add_argument("--r2-key", required=True, help="R2 access key ID")
    parser.add_argument("--r2-secret", required=True, help="R2 access key secret")
    parser.add_argument("--r2-bucket", required=True, help="R2 bucket name")
    parser.add_argument("--r2-subfolder", default="hf_dataset", help="R2 subfolder")
    parser.add_argument("--token", help="Hugging Face API token for private datasets")
    parser.add_argument("--workers", type=int, default=16, help="Max worker threads")
    parser.add_argument("--silent", action="store_true", help="Silent mode")
    parser.add_argument("--dest", default="./data", help="Local destination path")
    parser.add_argument("--hf-prefix", default="", help="Hugging Face folder prefix")
    parser.add_argument(
        "--check-corrupted", action="store_true", help="Check for corrupted files only"
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=None,
        help="ZSTD compression level for parquet files (1-22, default: 3 if enabled)",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=None,
        help="Row group size for parquet files (default: 1000 if enabled)",
    )

    args = parser.parse_args()

    r2_config = R2Config(
        account_id=args.r2_account,
        access_key_id=args.r2_key,
        access_key_secret=args.r2_secret,
        bucket_name=args.r2_bucket,
        subfolder=args.r2_subfolder,
    )

    if args.check_corrupted:
        cleanup_corrupted_files(r2_config, args.r2_subfolder, args.workers)
    else:
        download_model(
            model_dataset_name=args.dataset,
            is_dataset=True,
            destination_base_path=args.dest,
            model_branch=args.branch,
            token=args.token,
            silent_mode=args.silent,
            r2cfg=r2_config,
            hf_prefix=args.hf_prefix,
            max_workers=args.workers,
            compression_level=args.compression_level,
            row_group_size=args.row_group_size,
        )
