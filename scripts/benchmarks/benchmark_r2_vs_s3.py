#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "aiobotocore>=2.5.0",
#   "botocore>=1.29.0",
#   "uvloop>=0.19.0",
#   "tqdm>=4.66.0",
# ]
# ///
"""
Cloudflare R2 vs AWS S3 Performance Benchmark for Templar gradient operations.
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import uvloop
from aiobotocore import session
from botocore.config import Config
from tqdm import tqdm


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator to retry operations with exponential backoff."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (ConnectionError, OSError, Exception) as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        raise last_exception

                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    print(
                        f"Connection error on attempt {attempt + 1}, retrying in {delay:.1f}s: {str(e)[:100]}"
                    )
                    await asyncio.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


@dataclass
class BenchmarkResult:
    """Results for a single benchmark operation."""

    provider: str
    operation: str
    file_size_gb: float
    duration_seconds: float
    throughput_gbps: float
    upload_type: str
    timestamp: str
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark test."""

    provider: str
    operation: str
    file_size_gb: float
    upload_type: str
    num_operations: int
    avg_duration: float
    avg_throughput: float
    min_duration: float
    max_duration: float
    std_duration: float
    success_rate: float
    total_duration: float


class GradientFileGenerator:
    """Generates synthetic gradient files for benchmarking."""

    @staticmethod
    def create_temp_file(file_size_gb: float, temp_dir: str = "/tmp") -> str:
        """Create a temporary file with gradient data."""
        file_id = uuid.uuid4().hex[:8]
        temp_path = f"{temp_dir}/gradient_test_{file_size_gb}gb_{file_id}.bin"
        target_bytes = int(file_size_gb * 1024 * 1024 * 1024)
        chunk_size = 64 * 1024 * 1024  # 64MB chunks for better performance
        total_chunks = (target_bytes + chunk_size - 1) // chunk_size

        with tqdm(
            total=total_chunks,
            desc=f"Generating {file_size_gb}GB test file",
            unit="chunks",
            unit_scale=True,
            leave=False,
            colour="yellow",
        ) as progress:
            with open(temp_path, "wb") as f:
                written = 0
                chunk_data = os.urandom(min(chunk_size, target_bytes))
                while written < target_bytes:
                    remaining = target_bytes - written
                    if remaining >= len(chunk_data):
                        f.write(chunk_data)
                        written += len(chunk_data)
                    else:
                        f.write(chunk_data[:remaining])
                        written += remaining
                    progress.update(1)
        return temp_path


class UnifiedStorageClient:
    """Unified client for Cloudflare R2 and AWS S3 with multipart support."""

    def __init__(
        self,
        provider: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        region: str = "us-east-1",
        account_id: str = "",
    ) -> None:
        self.provider = provider.upper()
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.region = region
        self.account_id = account_id
        self.session = session.get_session()
        self.multipart_threshold = 100 * 1024 * 1024

        # Enhanced connection configuration for stability
        self.client_config = Config(
            max_pool_connections=10,  # Reduced to prevent connection issues
            retries={"max_attempts": 3, "mode": "adaptive"},
            read_timeout=600,  # Increased timeout for large uploads
            connect_timeout=30,
            region_name=self.region,
            parameter_validation=False,  # Slight performance improvement
        )

        if self.provider == "R2":
            if not account_id:
                raise ValueError("account_id is required for R2")
            self.endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
            self.region = "enam"
        else:
            self.endpoint_url = None

    async def put_object_from_file(
        self, key: str, file_path: str, use_multipart: Optional[bool] = None
    ) -> tuple[bool, str]:
        """Upload object from file with optional multipart upload."""
        file_size = os.path.getsize(file_path)
        should_use_multipart = (
            use_multipart
            if use_multipart is not None
            else file_size > self.multipart_threshold
        )
        upload_type = "multipart" if should_use_multipart else "single"

        try:
            async with self.session.create_client(
                "s3",
                endpoint_url=self.endpoint_url,
                region_name=self.region,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=self.client_config,
            ) as client:
                if should_use_multipart:
                    success = await self._multipart_upload(client, key, file_path)
                else:
                    success = await self._single_upload(client, key, file_path)
                return success, upload_type
        except Exception as e:
            print(f"Upload failed for {self.provider}: {str(e)}")
            return False, f"error: {str(e)}"

    async def _single_upload(self, client, key: str, file_path: str) -> bool:
        """Single upload for smaller files."""
        with open(file_path, "rb") as f:
            await client.put_object(Bucket=self.bucket_name, Key=key, Body=f.read())
        return True

    async def _multipart_upload(self, client, key: str, file_path: str) -> bool:
        """Multipart upload for larger files with improved reliability."""
        # Smaller chunk size to avoid connection timeouts
        chunk_size = 50 * 1024 * 1024  # 50MB chunks for better stability
        file_size = os.path.getsize(file_path)
        total_parts = (file_size + chunk_size - 1) // chunk_size

        response = await client.create_multipart_upload(
            Bucket=self.bucket_name, Key=key
        )
        upload_id = response["UploadId"]

        try:
            parts = []
            part_number = 1

            with tqdm(
                total=total_parts,
                desc=f"Uploading {self.provider} multipart",
                unit="parts",
                leave=False,
                colour="magenta",
            ) as upload_progress:
                with open(file_path, "rb") as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break

                        # Retry individual part uploads with better error handling
                        part_uploaded = False
                        for attempt in range(5):  # Increased retries
                            try:
                                part_response = await client.upload_part(
                                    Bucket=self.bucket_name,
                                    Key=key,
                                    PartNumber=part_number,
                                    UploadId=upload_id,
                                    Body=chunk,
                                )
                                parts.append(
                                    {
                                        "ETag": part_response["ETag"],
                                        "PartNumber": part_number,
                                    }
                                )
                                part_uploaded = True
                                break
                            except (ConnectionError, OSError, Exception) as e:
                                if attempt == 4:  # Last attempt
                                    print(
                                        f"Part {part_number} failed after 5 attempts: {str(e)}"
                                    )
                                    raise e
                                delay = 2**attempt + random.uniform(0, 1)
                                print(
                                    f"Part {part_number} failed (attempt {attempt + 1}), retrying in {delay:.1f}s"
                                )
                                await asyncio.sleep(delay)

                        if not part_uploaded:
                            raise Exception(
                                f"Failed to upload part {part_number} after all retries"
                            )

                        part_number += 1
                        upload_progress.update(1)

            await client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
            return True
        except Exception as e:
            try:
                await client.abort_multipart_upload(
                    Bucket=self.bucket_name, Key=key, UploadId=upload_id
                )
            except Exception:
                pass  # Ignore cleanup errors
            raise e

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    async def get_object_to_file(self, key: str, download_path: str) -> bool:
        """Download object to file with retry logic."""
        try:
            async with self.session.create_client(
                "s3",
                endpoint_url=self.endpoint_url,
                region_name=self.region,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=self.client_config,
            ) as client:
                response = await client.get_object(Bucket=self.bucket_name, Key=key)  # type: ignore
                with open(download_path, "wb") as f:
                    async for chunk in response["Body"]:
                        f.write(chunk)
                return True
        except Exception as e:
            print(f"Download failed for {self.provider}: {str(e)}")
            return False


class BenchmarkRunner:
    """Main benchmark runner class."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.results: list[BenchmarkResult] = []
        self.setup_logging()
        self.main_progress: Optional[tqdm] = None
        # Limit concurrent operations to prevent connection pool exhaustion
        self.connection_semaphore = asyncio.Semaphore(5)

        r2_config = config["r2"]
        s3_config = config["s3"]

        self.r2_client = UnifiedStorageClient(
            provider="R2",
            access_key=str(r2_config["access_key"]),
            secret_key=str(r2_config["secret_key"]),
            bucket_name=str(r2_config["bucket_name"]),
            account_id=str(r2_config["account_id"]),
        )

        self.s3_client = UnifiedStorageClient(
            provider="S3",
            access_key=str(s3_config["access_key"]),
            secret_key=str(s3_config["secret_key"]),
            bucket_name=str(s3_config["bucket_name"]),
            region=str(s3_config["region"]),
        )

    def get_client(self, provider: str) -> UnifiedStorageClient:
        """Get the appropriate client for the provider."""
        return self.r2_client if provider == "R2" else self.s3_client

    def setup_logging(self) -> None:
        """Configure logging."""
        test_config = self.config.get("test", {})
        assert isinstance(test_config, dict)
        verbose = bool(test_config.get("verbose", False))
        log_level = logging.DEBUG if verbose else logging.INFO

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("benchmark.log"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    async def benchmark_single_operation(
        self,
        client: UnifiedStorageClient,
        provider: str,
        operation: str,
        key: str,
        file_path: str,
        file_size_gb: float,
        use_multipart: Optional[bool] = None,
    ) -> BenchmarkResult:
        """Benchmark a single PUT or GET operation with connection limiting."""
        timestamp = datetime.now().isoformat()

        async with self.connection_semaphore:  # Limit concurrent operations
            try:
                start_time = time.perf_counter()
                if operation == "PUT":
                    success, upload_type = await client.put_object_from_file(
                        key, file_path, use_multipart
                    )
                else:
                    download_path = f"{file_path}.download"
                    success = await client.get_object_to_file(key, download_path)
                    upload_type = "download"
                    if os.path.exists(download_path):
                        os.remove(download_path)

                end_time = time.perf_counter()
                duration = end_time - start_time
                throughput = file_size_gb / duration if duration > 0 else 0

                return BenchmarkResult(
                    provider=provider,
                    operation=operation,
                    file_size_gb=file_size_gb,
                    duration_seconds=duration,
                    throughput_gbps=throughput,
                    upload_type=upload_type,
                    timestamp=timestamp,
                    success=success,
                )
            except Exception as e:
                return BenchmarkResult(
                    provider=provider,
                    operation=operation,
                    file_size_gb=file_size_gb,
                    duration_seconds=0,
                    throughput_gbps=0,
                    upload_type="error",
                    timestamp=timestamp,
                    success=False,
                    error=str(e),
                )

    async def run_concurrent_operations(
        self,
        provider: str,
        operation: str,
        num_concurrent: int,
        file_size_gb: float,
        test_multipart: bool = True,
        progress_desc: str = "",
    ) -> list[BenchmarkResult]:
        """Run multiple concurrent operations to test throughput."""
        temp_file = GradientFileGenerator.create_temp_file(file_size_gb)

        try:
            client = self.get_client(provider)
            tasks = []

            for i in range(num_concurrent):
                key = f"benchmark/{provider.lower()}/{operation.lower()}/{uuid.uuid4()}.bin"

                if test_multipart and file_size_gb > 0.1:
                    tasks.append(
                        self.benchmark_single_operation(
                            client,
                            provider,
                            operation,
                            key,
                            temp_file,
                            file_size_gb,
                            True,
                        )
                    )
                    if file_size_gb <= 5:
                        key_single = f"benchmark/{provider.lower()}/{operation.lower()}/single_{uuid.uuid4()}.bin"
                        tasks.append(
                            self.benchmark_single_operation(
                                client,
                                provider,
                                operation,
                                key_single,
                                temp_file,
                                file_size_gb,
                                False,
                            )
                        )
                else:
                    tasks.append(
                        self.benchmark_single_operation(
                            client, provider, operation, key, temp_file, file_size_gb
                        )
                    )

            with tqdm(
                total=len(tasks),
                desc=f"{progress_desc or f'{provider} {operation} ops'}",
                unit="ops",
                leave=False,
                colour="green",
            ) as op_progress:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                op_progress.update(len(tasks))

            # Filter out exceptions and log them
            benchmark_results = []
            for result in results:
                if isinstance(result, BenchmarkResult):
                    benchmark_results.append(result)
                elif isinstance(result, Exception):
                    print(f"Operation failed: {str(result)[:100]}")

            return benchmark_results
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    async def run_gradient_simulation(self) -> None:
        """Simulate gradient fetch/aggregation patterns from templar miners."""
        test_config = self.config.get("test", {})
        assert isinstance(test_config, dict)

        max_concurrent = int(test_config.get("max_concurrent", 10))
        skip_large = bool(test_config.get("skip_large", False))
        quick = bool(test_config.get("quick", False))

        scenarios = (
            [{"name": "Quick Test", "file_size_gb": 0.5, "concurrent": [1, 2]}]
            if quick
            else [
                {
                    "name": "Small Gradients (2GB)",
                    "file_size_gb": 2.0,
                    "concurrent": [1, 3, min(5, max_concurrent)],
                },
                {
                    "name": "Medium Gradients (5GB)",
                    "file_size_gb": 5.0,
                    "concurrent": [1, 2, min(3, max_concurrent)],
                },
            ]
        )

        if not quick and not skip_large:
            scenarios.extend(
                [
                    {
                        "name": "Large Gradients (10GB)",
                        "file_size_gb": 10.0,
                        "concurrent": [1, 2],
                    },
                    {
                        "name": "Very Large Gradients (20GB)",
                        "file_size_gb": 20.0,
                        "concurrent": [1],
                    },
                ]
            )

        total_operations = sum(
            len(scenario["concurrent"]) * 4 for scenario in scenarios
        )

        print(
            f"\nStarting GB-scale gradient benchmark with {total_operations} test configurations"
        )
        print("=" * 70)

        with tqdm(
            total=total_operations,
            desc="Overall Benchmark Progress",
            unit="tests",
            colour="cyan",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as main_progress:
            for scenario in scenarios:
                scenario_desc = f"{scenario['name']}"
                main_progress.set_description(scenario_desc)

                for num_concurrent in scenario["concurrent"]:
                    for operation in ["PUT", "GET"]:
                        for provider in ["R2", "S3"]:
                            test_desc = f"{provider} {operation} {num_concurrent}x ({scenario['file_size_gb']}GB)"
                            main_progress.set_postfix_str(test_desc)

                            results = await self.run_concurrent_operations(
                                provider=provider,
                                operation=operation,
                                num_concurrent=num_concurrent,
                                file_size_gb=scenario["file_size_gb"],
                                test_multipart=True,
                                progress_desc=test_desc,
                            )
                            self.results.extend(results)

                            successful = [r for r in results if r.success]
                            if successful:
                                avg_throughput = statistics.mean(
                                    [r.throughput_gbps for r in successful]
                                )
                                main_progress.write(
                                    f"SUCCESS {test_desc}: {len(successful)}/{len(results)} success, "
                                    f"avg {avg_throughput:.3f} GB/s"
                                )
                            else:
                                main_progress.write(
                                    f"FAILED {test_desc}: All operations failed"
                                )

                            main_progress.update(1)

        print(f"\nBenchmark completed! Total operations: {len(self.results)}")
        print("=" * 70)

    def generate_summary_statistics(self) -> list[BenchmarkSummary]:
        """Generate summary statistics from all results."""
        summaries = []
        groups = {}

        for result in self.results:
            key = (
                result.provider,
                result.operation,
                result.file_size_gb,
                result.upload_type,
            )
            if key not in groups:
                groups[key] = []
            groups[key].append(result)

        for (provider, operation, file_size_gb, upload_type), results in groups.items():
            successful_results = [r for r in results if r.success]
            if successful_results:
                durations = [r.duration_seconds for r in successful_results]
                throughputs = [r.throughput_gbps for r in successful_results]

                summaries.append(
                    BenchmarkSummary(
                        provider=provider,
                        operation=operation,
                        file_size_gb=file_size_gb,
                        upload_type=upload_type,
                        num_operations=len(results),
                        avg_duration=statistics.mean(durations),
                        avg_throughput=statistics.mean(throughputs),
                        min_duration=min(durations),
                        max_duration=max(durations),
                        std_duration=statistics.stdev(durations)
                        if len(durations) > 1
                        else 0,
                        success_rate=len(successful_results) / len(results),
                        total_duration=sum(durations),
                    )
                )
        return summaries

    def save_results(self, output_dir: str = "benchmark_results") -> None:
        """Save benchmark results in CSV and JSON formats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_file = output_path / f"r2_vs_s3_detailed_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump([asdict(result) for result in self.results], f, indent=2)

        csv_file = output_path / f"r2_vs_s3_detailed_{timestamp}.csv"
        with open(csv_file, "w", newline="") as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for result in self.results:
                    writer.writerow(asdict(result))

        summaries = self.generate_summary_statistics()
        summary_json_file = output_path / f"r2_vs_s3_summary_{timestamp}.json"
        with open(summary_json_file, "w") as f:
            json.dump([asdict(summary) for summary in summaries], f, indent=2)

        summary_csv_file = output_path / f"r2_vs_s3_summary_{timestamp}.csv"
        with open(summary_csv_file, "w", newline="") as f:
            if summaries:
                writer = csv.DictWriter(f, fieldnames=asdict(summaries[0]).keys())
                writer.writeheader()
                for summary in summaries:
                    writer.writerow(asdict(summary))

        self.logger.info(f"Results saved to {output_path}")
        self.logger.info(
            f"Files: {json_file.name}, {csv_file.name}, {summary_json_file.name}, {summary_csv_file.name}"
        )

    def print_summary_report(self) -> None:
        """Print a formatted summary report to console."""
        summaries = self.generate_summary_statistics()

        print("\n" + "=" * 80)
        print("R2 vs S3 GB-SCALE BENCHMARK SUMMARY REPORT")
        print("=" * 80)

        for operation in ["PUT", "GET"]:
            print(f"\n{operation} OPERATIONS:")
            print("-" * 60)

            op_summaries = [s for s in summaries if s.operation == operation]
            file_sizes = sorted(set([s.file_size_gb for s in op_summaries]))

            for file_size in file_sizes:
                print(f"\nFile Size: {file_size} GB")
                size_summaries = [
                    s for s in op_summaries if s.file_size_gb == file_size
                ]
                upload_types = sorted(set([s.upload_type for s in size_summaries]))

                for upload_type in upload_types:
                    if upload_type in ["error", "download"]:
                        continue

                    print(f"\n  {upload_type.upper()} UPLOADS:")
                    type_summaries = [
                        s for s in size_summaries if s.upload_type == upload_type
                    ]

                    for summary in type_summaries:
                        print(
                            f"    {summary.provider:3s}: {summary.avg_throughput:6.3f} GB/s avg, "
                            f"{summary.success_rate:5.1%} success, {summary.num_operations:3d} ops, "
                            f"{summary.avg_duration:6.1f}s duration"
                        )

        print("\n" + "=" * 80)
        print("OVERALL SUMMARY:")
        print("=" * 80)

        r2_results = [s for s in summaries if s.provider == "R2" and s.success_rate > 0]
        s3_results = [s for s in summaries if s.provider == "S3" and s.success_rate > 0]

        if r2_results and s3_results:
            r2_avg = statistics.mean([s.avg_throughput for s in r2_results])
            s3_avg = statistics.mean([s.avg_throughput for s in s3_results])

            print(f"R2 Average Throughput: {r2_avg:.3f} GB/s")
            print(f"S3 Average Throughput: {s3_avg:.3f} GB/s")

            if r2_avg > s3_avg:
                improvement = (r2_avg / s3_avg - 1) * 100
                print(f"Overall Winner: R2 ({improvement:.1f}% faster on average)")
            else:
                improvement = (s3_avg / r2_avg - 1) * 100
                print(f"Overall Winner: S3 ({improvement:.1f}% faster on average)")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Cloudflare R2 vs AWS S3 for gradient operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    r2_group = parser.add_argument_group("Cloudflare R2 Configuration")
    r2_group.add_argument(
        "--r2-account-id",
        default=os.getenv("R2_ACCOUNT_ID"),
        help="Cloudflare R2 account ID",
    )
    r2_group.add_argument(
        "--r2-access-key",
        default=os.getenv("R2_ACCESS_KEY"),
        help="Cloudflare R2 access key",
    )
    r2_group.add_argument(
        "--r2-secret-key",
        default=os.getenv("R2_SECRET_KEY"),
        help="Cloudflare R2 secret key",
    )
    r2_group.add_argument(
        "--r2-bucket",
        default=os.getenv("R2_BUCKET_NAME"),
        help="Cloudflare R2 bucket name",
    )

    s3_group = parser.add_argument_group("AWS S3 Configuration")
    s3_group.add_argument(
        "--s3-access-key",
        default=os.getenv("AWS_ACCESS_KEY_ID"),
        help="AWS S3 access key ID",
    )
    s3_group.add_argument(
        "--s3-secret-key",
        default=os.getenv("AWS_SECRET_ACCESS_KEY"),
        help="AWS S3 secret access key",
    )
    s3_group.add_argument(
        "--s3-bucket", default=os.getenv("S3_BUCKET_NAME"), help="AWS S3 bucket name"
    )
    s3_group.add_argument(
        "--s3-region",
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS S3 region",
    )

    test_group = parser.add_argument_group("Test Configuration")
    test_group.add_argument(
        "--output-dir", default="benchmark_results", help="Output directory for results"
    )
    test_group.add_argument(
        "--max-concurrent", type=int, default=10, help="Maximum concurrent operations"
    )
    test_group.add_argument(
        "--iterations", type=int, default=3, help="Iterations per test scenario"
    )
    test_group.add_argument(
        "--skip-large", action="store_true", help="Skip large file tests"
    )
    test_group.add_argument(
        "--quick", action="store_true", help="Run quick test with minimal scenarios"
    )
    test_group.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    return parser.parse_args()


async def main() -> None:
    """Main function to run the benchmark."""
    args = parse_arguments()

    required_r2 = [
        args.r2_account_id,
        args.r2_access_key,
        args.r2_secret_key,
        args.r2_bucket,
    ]
    required_s3 = [args.s3_access_key, args.s3_secret_key, args.s3_bucket]

    if not all(required_r2):
        print(
            "Error: Missing R2 configuration. Required: --r2-account-id, --r2-access-key, --r2-secret-key, --r2-bucket"
        )
        return

    if not all(required_s3):
        print(
            "Error: Missing S3 configuration. Required: --s3-access-key, --s3-secret-key, --s3-bucket"
        )
        return

    config = {
        "r2": {
            "account_id": args.r2_account_id,
            "access_key": args.r2_access_key,
            "secret_key": args.r2_secret_key,
            "bucket_name": args.r2_bucket,
        },
        "s3": {
            "access_key": args.s3_access_key,
            "secret_key": args.s3_secret_key,
            "bucket_name": args.s3_bucket,
            "region": args.s3_region,
        },
        "test": {
            "max_concurrent": args.max_concurrent,
            "iterations": args.iterations,
            "skip_large": args.skip_large,
            "quick": args.quick,
            "output_dir": args.output_dir,
            "verbose": args.verbose,
        },
    }

    runner = BenchmarkRunner(config)
    try:
        await runner.run_gradient_simulation()
        runner.save_results(args.output_dir)
        runner.print_summary_report()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        raise


if __name__ == "__main__":
    print("R2 vs S3 Gradient Operations Benchmark")
    print("Use --help for options or set environment variables.")
    uvloop.install()
    asyncio.run(main())
