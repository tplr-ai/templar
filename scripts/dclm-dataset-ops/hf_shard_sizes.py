#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "aiodns",
#   "brotli",
#   "aiohttp",
#   "aiohttp[speedups]",
#   "aiohttp[socks]",
#   "tqdm",
#   "pyarrow",
#   "pandas",
#   "uvloop",
#   "python-dotenv",
#   "fsspec",
#   "aiofiles",
# ]
# ///
"""
HFShardSizes - Hugging Face Parquet File Analyzer to generate the CSV for the shard sizes

Recursively traverses HF repositories, downloads Parquet files,
generates their row counts and checksum, then appends results to a CSV.
"""

import asyncio
import csv
import hashlib
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiofiles
import aiohttp
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()

# API URL templates
AGREEMENT_MODEL_URL = "https://huggingface.co/%s"
AGREEMENT_DATASET_URL = "https://huggingface.co/datasets/%s"
RAW_MODEL_FILE_URL = "https://huggingface.co/%s/raw/%s/%s"
RAW_DATASET_FILE_URL = "https://huggingface.co/datasets/%s/raw/%s/%s"
LFS_MODEL_RESOLVER_URL = "https://huggingface.co/%s/resolve/%s/%s"
LFS_DATASET_RESOLVER_URL = "https://huggingface.co/datasets/%s/resolve/%s/%s"
JSON_MODELS_FILE_TREE_URL = "https://huggingface.co/api/models/%s/tree/%s/%s"
JSON_DATASET_FILE_TREE_URL = "https://huggingface.co/api/datasets/%s/tree/%s/%s"

STREAM_BUFFER_SIZE = 16 * 1024 * 1024  # 16MB buffer for streaming
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for download
MAX_CONCURRENT = 16  # 16 concurrent files
MAX_RETRIES = 3  # 3 retries for faster failure recovery
RETRY_DELAY = 0.5  # 500ms retry delay
NUM_CONNECTIONS = 64  # Max connections for HTTP client
TEMP_DIR = Path(tempfile.gettempdir()) / "hf_parquet_analyzer"


@dataclass
class HFFile:
    """
    Represents a file from Hugging Face's repository structure.
    Contains metadata about the file and download state.
    """

    type: str
    oid: str
    size: int
    path: str
    local_size: int = 0
    needs_download: bool = False
    is_directory: bool = False
    is_lfs: bool = False
    appended_path: str = ""
    skip_downloading: bool = False
    filter_skip: bool = False
    download_link: str = ""
    lfs: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisResult:
    """
    Stores the analysis result for a Parquet file.
    """

    filename: str
    filepath: str
    row_count: int
    byte_size: int
    file_hash: str


class ProcessedFileTracker:
    """Thread-safe tracker of already processed files."""

    def __init__(self, csv_path: str):
        """
        Initialize the tracker with a CSV file path.

        Args:
            csv_path: Path to the CSV file tracking processed files
        """
        self.csv_path = Path(csv_path)
        self._lock = asyncio.Lock()
        self._processed_hashes: Set[str] = set()
        self._initialize()

    def _initialize(self):
        """Initialize the tracker by loading existing processed file hashes."""
        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["filename", "filepath", "row_count", "byte_size", "file_hash"]
                )
        else:
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "file_hash" in row:
                        self._processed_hashes.add(row["file_hash"])

    async def is_processed(self, file_hash: str) -> bool:
        """
        Check if a file has already been processed.

        Args:
            file_hash: Hash of the file to check

        Returns:
            True if the file has been processed, False otherwise
        """
        async with self._lock:
            return file_hash in self._processed_hashes

    async def add_result(self, result: AnalysisResult) -> bool:
        """
        Add an analysis result to the CSV and mark the file as processed.
        Uses atomic writes for crash safety.

        Args:
            result: Analysis result to add

        Returns:
            True if successfully added, False if already exists
        """
        async with self._lock:
            if result.file_hash in self._processed_hashes:
                return False

            temp_csv = self.csv_path.with_suffix(".tmp")

            async with aiofiles.open(temp_csv, "w", newline="") as f:
                if self.csv_path.exists():
                    with open(self.csv_path, "r", newline="") as src:
                        content = src.read()
                        await f.write(content)
                else:
                    await f.write("filename,filepath,row_count,byte_size,file_hash\n")

                line = (
                    f"{result.filename},{result.filepath},{result.row_count},"
                    f"{result.byte_size},{result.file_hash}\n"
                )
                await f.write(line)

            temp_csv.replace(self.csv_path)

            self._processed_hashes.add(result.file_hash)
            return True


def format_size(bytes_size: int) -> str:
    """
    Format size in human-readable format.

    Args:
        bytes_size: The size in bytes

    Returns:
        A human-readable string representation of the size
    """
    if bytes_size < 1024:
        return f"{bytes_size} B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    size = float(bytes_size)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.1f} {units[unit_index]}"


def is_valid_model_name(model_name: str) -> bool:
    """
    Check if a model name matches the expected pattern.

    Args:
        model_name: The model name to validate

    Returns:
        True if the model name is valid, False otherwise
    """
    pattern = r"^[A-Za-z0-9_\-]+/[A-Za-z0-9\._\-]+$"
    return bool(re.match(pattern, model_name))


def compute_file_hash(path: str, size: int) -> str:
    """
    Compute a unique hash for a file based on its path and size.
    This serves as a unique identifier for resumability.

    Args:
        path: File path
        size: File size in bytes

    Returns:
        MD5 hash string
    """
    hasher = hashlib.md5()
    hasher.update(path.encode("utf-8"))
    hasher.update(str(size).encode("utf-8"))
    return hasher.hexdigest()


async def create_http_client() -> aiohttp.ClientSession:
    """
    Create an HTTP client optimized for high throughput parallel connections.
    Uses Cloudflare DNS resolvers for faster DNS resolution.

    Returns:
        An aiohttp ClientSession configured for optimal performance
    """
    connector = aiohttp.TCPConnector(
        limit=NUM_CONNECTIONS,
        limit_per_host=NUM_CONNECTIONS,
        ttl_dns_cache=300,
        # family=aiohttp.TCPConnector.FAMILY_V4,
        resolver=aiohttp.AsyncResolver(nameservers=["1.1.1.1"]),
    )
    timeout = aiohttp.ClientTimeout(total=60, connect=10)  # type: ignore
    return aiohttp.ClientSession(connector=connector, timeout=timeout)


async def fetch_file_list(
    session: aiohttp.ClientSession, url: str, auth_token: Optional[str] = None
) -> List[HFFile]:
    """
    Fetch the file list from Hugging Face API and parse into structured objects.

    Args:
        session: HTTP client session
        url: The API URL to fetch files from
        auth_token: Optional authentication token

    Returns:
        A list of HFFile objects representing files and directories

    Raises:
        Exception: If the API request fails
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    async with session.get(url, headers=headers) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise Exception(
                f"Failed to fetch file list, status: {resp.status}, body: {body}"
            )

        data = await resp.json()
        files = []

        for item in data:
            file = HFFile(
                type=item.get("type", ""),
                oid=item.get("oid", ""),
                size=item.get("size", 0),
                path=item.get("path", ""),
                is_directory=item.get("type") == "directory",
                lfs=item.get("lfs"),
            )

            if file.lfs:
                file.is_lfs = True

            files.append(file)

        return files


def verify_parquet_file(file_path: str) -> bool:
    """
    Verify that a file is a valid Parquet file by checking magic numbers.

    Args:
        file_path: Path to the file to verify

    Returns:
        True if the file is a valid Parquet file, False otherwise
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
            f.seek(-4, os.SEEK_END)
            footer = f.read(4)
            expected_magic = b"PAR1"
            return header == expected_magic and footer == expected_magic
    except Exception as e:
        print(f"Error verifying Parquet file {file_path}: {e}")
        return False


def analyze_parquet_file(file_path: str) -> int:
    """
    Analyze a Parquet file to determine its row count.
    Uses PyArrow for efficient metadata-only reading when possible.

    Args:
        file_path: Path to the Parquet file

    Returns:
        Number of rows in the Parquet file
    """
    try:
        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata

        if metadata is not None:
            return metadata.num_rows

        table = parquet_file.read()
        return table.num_rows
    except Exception:
        try:
            df = pd.read_parquet(file_path)
            return len(df)
        except Exception as inner_e:
            print(f"Error analyzing Parquet file {file_path}: {inner_e}")
            return -1  # Indicates error


async def process_hf_folder_tree(
    session: aiohttp.ClientSession,
    model_path: str,
    is_dataset: bool,
    model_dataset_name: str,
    model_branch: str,
    folder_name: str,
    silent_mode: bool,
    hf_prefix: str,
    file_queue: asyncio.Queue,
    file_tracker: ProcessedFileTracker,
    auth_token: Optional[str] = None,
) -> None:
    """
    Process the Hugging Face folder tree recursively, identifying Parquet files and queueing them for analysis.

    Args:
        session: HTTP client session
        model_path: Base path for the model
        is_dataset: Whether this is a dataset (vs a model)
        model_dataset_name: Model or dataset name
        model_branch: Branch name
        folder_name: Current folder name (empty for root)
        silent_mode: Whether to suppress output
        hf_prefix: Prefix for Hugging Face paths
        file_queue: Queue for files to process
        file_tracker: Tracker for processed files
        auth_token: Optional authentication token
    """
    if not silent_mode:
        print(f"🔍 Scanning: {folder_name or 'root'}")

    if is_dataset:
        if folder_name == "":
            url = JSON_DATASET_FILE_TREE_URL % (
                model_dataset_name,
                model_branch,
                hf_prefix,
            )
        else:
            url = JSON_DATASET_FILE_TREE_URL % (
                model_dataset_name,
                model_branch,
                folder_name,
            )
    else:
        if folder_name == "":
            url = JSON_MODELS_FILE_TREE_URL % (
                model_dataset_name,
                model_branch,
                hf_prefix,
            )
        else:
            url = JSON_MODELS_FILE_TREE_URL % (
                model_dataset_name,
                model_branch,
                folder_name,
            )

    if not silent_mode:
        print(f"📡 API URL: {url}")

    files = await fetch_file_list(session, url, auth_token)

    if not silent_mode:
        print(f"📂 Found {len(files)} items in {folder_name or 'root'}")

    directories = []
    parquet_files = []

    for file in files:
        if file.is_directory:
            directories.append(file)
        elif file.path.endswith(".parquet") and file.size > 0:
            if is_dataset:
                file.download_link = (
                    LFS_DATASET_RESOLVER_URL
                    % (model_dataset_name, model_branch, file.path)
                    if file.is_lfs
                    else RAW_DATASET_FILE_URL
                    % (model_dataset_name, model_branch, file.path)
                )
            else:
                file.download_link = (
                    LFS_MODEL_RESOLVER_URL
                    % (model_dataset_name, model_branch, file.path)
                    if file.is_lfs
                    else RAW_MODEL_FILE_URL
                    % (model_dataset_name, model_branch, file.path)
                )

            parquet_files.append(file)

    for file in parquet_files:
        file_hash = compute_file_hash(file.path, file.size)

        if await file_tracker.is_processed(file_hash):
            if not silent_mode:
                print(f"Skipping {file.path} - already analyzed")
            continue

        if not silent_mode:
            print(f"Queueing: {file.path} ({format_size(file.size)})")

        file.file_hash = file_hash
        await file_queue.put(file)

    for directory in directories:
        await process_hf_folder_tree(
            session,
            model_path,
            is_dataset,
            model_dataset_name,
            model_branch,
            directory.path,
            silent_mode,
            hf_prefix,
            file_queue,
            file_tracker,
            auth_token,
        )


async def download_and_analyze_file(
    session: aiohttp.ClientSession,
    file: HFFile,
    temp_dir: Path,
    auth_token: Optional[str] = None,
    progress_callback=None,
) -> Tuple[Optional[AnalysisResult], Optional[Exception]]:
    """
    Download a file to a temporary location, analyze it, and clean up.

    Args:
        session: HTTP client session
        file: File to download and analyze
        temp_dir: Directory for temporary files
        auth_token: Optional authentication token
        progress_callback: Optional callback for progress updates

    Returns:
        A tuple of (AnalysisResult, Exception) where one is None
    """
    temp_file = temp_dir / f"temp_{os.urandom(8).hex()}.parquet"
    temp_file.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        async with session.get(
            file.download_link,
            headers=headers,
            timeout=300,  # type: ignore
        ) as response:
            if response.status != 200:
                return None, Exception(f"Download failed with status {response.status}")

            with open(temp_file, "wb") as fd:
                downloaded = 0
                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    fd.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(len(chunk))

        if not verify_parquet_file(str(temp_file)):
            return None, Exception(f"Invalid Parquet file: {file.path}")

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            row_count = await loop.run_in_executor(
                pool, analyze_parquet_file, str(temp_file)
            )

        if row_count < 0:
            return None, Exception(f"Failed to analyze Parquet file: {file.path}")

        result = AnalysisResult(
            filename=Path(file.path).name,
            filepath=file.path,
            row_count=row_count,
            byte_size=file.size,
            file_hash=getattr(
                file, "file_hash", compute_file_hash(file.path, file.size)
            ),
        )

        return result, None

    except Exception as e:
        return None, e
    finally:
        if temp_file.exists():
            temp_file.unlink()


async def worker(
    worker_id: int,
    file_queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    temp_dir: Path,
    file_tracker: ProcessedFileTracker,
    auth_token: Optional[str] = None,
    silent_mode: bool = False,
) -> List[Exception]:
    """
    Worker for processing files from the queue. Downloads files from Hugging Face,
    analyzes them, and updates the tracker.

    Args:
        worker_id: Worker identifier
        file_queue: Queue containing files to process
        session: HTTP client session
        temp_dir: Directory for temporary files
        file_tracker: Tracker for processed files
        auth_token: Optional authentication token
        silent_mode: Whether to suppress output

    Returns:
        List of exceptions encountered during processing
    """
    errors = []

    while True:
        try:
            try:
                file = await asyncio.wait_for(file_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if file_queue.empty():
                    break
                continue

            if (
                file.is_directory
                or file.filter_skip
                or file.size <= 0
                or not file.path
                or file.skip_downloading
            ):
                if not silent_mode:
                    print(f"Worker {worker_id}: Skipping {file.path}")

                file_queue.task_done()
                continue

            print(f"Worker {worker_id}: Processing file {file.path}")

            file_hash = getattr(
                file, "file_hash", compute_file_hash(file.path, file.size)
            )
            if await file_tracker.is_processed(file_hash):
                print(f"Worker {worker_id}: Skipping {file.path} - already analyzed")
                file_queue.task_done()
                continue

            progress_bar = tqdm(
                total=file.size,
                desc=f"Worker {worker_id}: {Path(file.path).name}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            )

            result, error = await download_and_analyze_file(
                session,
                file,
                temp_dir,
                auth_token,
                progress_callback=progress_bar.update,
            )

            progress_bar.close()

            if error:
                print(f"Worker {worker_id}: Error processing {file.path}: {error}")
                errors.append(error)
            else:
                # Add result to tracker
                if result is not None:
                    await file_tracker.add_result(result)
                    print(
                        f"✅ Worker {worker_id}: {file.path} - {result.row_count} rows"
                    )
                else:
                    print(f"Worker {worker_id}: No result for {file.path}")

            file_queue.task_done()
        except Exception as e:
            print(f"Worker {worker_id}: Unexpected error: {e}")
            errors.append(e)

    return errors


async def analyze_dataset(
    model_dataset_name: str,
    output_csv: str,
    model_branch: str = "main",
    is_dataset: bool = True,
    auth_token: Optional[str] = None,
    silent_mode: bool = False,
    concurrent_workers: int = MAX_CONCURRENT,
    hf_prefix: str = "",
) -> None:
    """
    Analyze Parquet files in a Hugging Face dataset and record row counts to a CSV file.

    Args:
        model_dataset_name: Model or dataset name
        output_csv: Path to the output CSV file
        model_branch: Branch name (default: main)
        is_dataset: Whether this is a dataset (default: True)
        auth_token: Optional authentication token
        silent_mode: Whether to suppress output
        concurrent_workers: Number of concurrent workers
        hf_prefix: Prefix for Hugging Face paths

    Raises:
        Exception: If any errors occur during analysis
    """
    file_tracker = ProcessedFileTracker(output_csv)

    temp_dir = TEMP_DIR
    os.makedirs(temp_dir, exist_ok=True)

    try:
        file_queue = asyncio.Queue()

        async with await create_http_client() as http_session:
            destination_base_path = "./"
            model_path = os.path.join(
                destination_base_path, model_dataset_name.split(":")[0]
            )

            await process_hf_folder_tree(
                http_session,
                model_path,
                is_dataset,
                model_dataset_name,
                model_branch,
                "",
                silent_mode,
                hf_prefix,
                file_queue,
                file_tracker,
                auth_token,
            )

            queue_size = file_queue.qsize()
            if queue_size == 0:
                print("No new files to analyze.")
                return

            max_workers = min(concurrent_workers, queue_size)
            print(f"Starting {max_workers} workers to process {queue_size} files")

            workers = [
                worker(
                    i,
                    file_queue,
                    http_session,
                    temp_dir,
                    file_tracker,
                    auth_token,
                    silent_mode,
                )
                for i in range(max_workers)
            ]

            worker_results = await asyncio.gather(*workers)
            await file_queue.join()

            errors = [
                error for worker_errors in worker_results for error in worker_errors
            ]

            if errors:
                print(f"Encountered {len(errors)} errors:")
                for i, error in enumerate(errors[:10]):
                    print(f"  {i+1}. {error}")

                if len(errors) > 10:
                    print(f"  ...and {len(errors) - 10} more errors")

                raise Exception(f"Analysis failed with {len(errors)} errors")
            else:
                print(f"✅ Analysis completed successfully: {model_dataset_name}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


async def main():
    """
    Main entry point for the script.
    Parses command-line arguments and initiates the analysis process.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Hugging Face Parquet files and record row counts, checksum and sizes to a CSV file."
    )
    parser.add_argument("model_dataset_name", help="Model or dataset name")
    parser.add_argument(
        "--dataset", action="store_true", help="Analyze a dataset instead of a model"
    )
    parser.add_argument("--branch", default="main", help="Branch name")
    parser.add_argument("--prefix", default="", help="Prefix for Hugging Face paths")
    parser.add_argument("--token", help="Hugging Face API token")
    parser.add_argument("--silent", action="store_true", help="Silent mode")
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_CONCURRENT,
        help="Number of concurrent workers",
    )
    parser.add_argument(
        "--output",
        default="parquet_analysis.csv",
        help="Output CSV file path",
    )

    args = parser.parse_args()

    print(
        f"Analyzing {'dataset' if args.dataset else 'model'}: {args.model_dataset_name}"
    )

    await analyze_dataset(
        model_dataset_name=args.model_dataset_name,
        output_csv=args.output,
        model_branch=args.branch,
        is_dataset=args.dataset,
        auth_token=args.token,
        silent_mode=args.silent,
        concurrent_workers=args.workers,
        hf_prefix=args.prefix,
    )


if __name__ == "__main__":
    import uvloop

    uvloop.install()
    asyncio.run(main())
