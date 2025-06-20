"""R2Dataset Live Benchmark Service

Continuously monitors for new windows from the mainnet and benchmarks
R2DatasetLoader performance on each new window arrival.

Key Features:
    - Live window monitoring from Bittensor mainnet
    - Automatic benchmark triggering on new windows
    - Comprehensive performance metrics collection
    - Service-oriented design for continuous operation

Usage:
    Basic run:
        $ uv run scripts/benchmarks/benchmark_r2_dataset.py

    Custom configuration:
        $ uv run scripts/benchmarks/benchmark_r2_dataset.py \\
            --netuid 45 \\
            --batch-size 8
"""

import argparse
import asyncio
import random
import time
from typing import List, Optional

import bittensor as bt
import numpy as np
import uvloop
from transformers.models.auto.tokenization_auto import AutoTokenizer

import tplr
from tplr.r2_dataset import R2DatasetLoader

WINDOW_CHECK_INTERVAL: int = 10  # Check for new windows every 10 seconds


def config() -> bt.Config:
    """Parse command-line arguments and return a configuration object."""

    parser = argparse.ArgumentParser(
        description="R2Dataset Live Benchmark Service. Use --help to display options.",
        add_help=True,
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=45,
        help="Bittensor network UID (subnet ID). Defaults to 45.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Batch size for dataset loading.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Sequence length for tokenization.",
    )
    parser.add_argument(
        "--pages-per-window",
        type=int,
        default=15,
        help="Number of pages to load per window.",
    )
    parser.add_argument(
        "--validator-sample-rate",
        type=float,
        default=0.6,
        help="Sample rate for validator simulation.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="togethercomputer/LLaMA-2-7B-32K",
        help="Tokenizer model name.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run benchmark once on current window and exit.",
    )

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    parser.parse_args()
    return bt.config(parser)


class R2DatasetBenchmark:
    """R2Dataset Live Benchmark Service

    Continuously monitors the Bittensor mainnet for new windows and automatically
    triggers R2DatasetLoader benchmarks when new windows are detected.

    Key Features:
        - Automatic window detection and monitoring
        - Performance metrics collection and logging
        - Service-oriented continuous operation
        - Validator usage pattern simulation

    Attributes:
        config (bt.Config): Configuration object containing CLI arguments
        netuid (int): Network UID for the subnet
        subtensor (bt.Subtensor): Bittensor subtensor connection
        metagraph (bt.Metagraph): Network metagraph
        uid (int): Validator UID for seeding
        hparams: Loaded hyperparameters
        tokenizer: Tokenizer for text processing
        last_benchmark_window (int): Last benchmarked window number
        last_block_number (int): Last processed block number
        stop_event (asyncio.Event): Event for graceful shutdown
    """

    def __init__(self) -> None:
        """Initialize the R2Dataset benchmark service."""
        self.config = config()
        if self.config.netuid is None:
            raise ValueError("No netuid provided")

        self.netuid = self.config.netuid
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.netuid)
        self.hparams = tplr.load_hparams()

        # Configuration parameters with defaults
        self.batch_size = self.config.batch_size or 6
        self.sequence_length = self.config.sequence_length or 2048
        self.pages_per_window = self.config.pages_per_window or 15
        self.validator_sample_rate = self.config.validator_sample_rate or 0.6
        self.tokenizer_name = (
            self.config.tokenizer_name or "togethercomputer/LLaMA-2-7B-32K"
        )

        self.uid = self.netuid

        tplr.logger.info(f"Using netuid/UID: {self.uid}")

        self.current_block = self.subtensor.get_current_block()
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        self.last_benchmark_window = 0
        self.last_block_number = 0
        self.stop_event = asyncio.Event()

        tplr.logger.info(
            f"Connected to netuid {self.config.netuid}, using UID {self.uid}"
        )
        tplr.logger.info(f"Current block: {self.current_block}")
        tplr.logger.info(f"Current window: {self.current_window}")
        tplr.logger.info(f"Blocks per window: {self.hparams.blocks_per_window}")

        tplr.logger.info(f"Loading tokenizer: {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, verbose=False, clean_up_tokenization_spaces=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.stats = {
            "windows_processed": 0,
            "pages_loaded": 0,
            "batches_processed": 0,
            "total_tokens": 0,
            "timing": {
                "page_loading": [],
                "loader_creation": [],
                "batch_processing": [],
                "total_window_time": [],
            },
        }

        self.window_results = []
        self.benchmark_start_time = time.perf_counter()

    async def retry_call(
        self,
        func,
        *args,
        attempts: int = 3,
        delay: float = 1,
        context: str = "",
        **kwargs,
    ):
        """Retry wrapper for async functions with exponential backoff"""
        for attempt in range(attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt < attempts - 1:
                    tplr.logger.warning(
                        f"Attempt {attempt + 1} failed for {context}: {e}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    tplr.logger.error(
                        f"All {attempts} attempts failed for {context}: {e}"
                    )
                    raise

    async def update_state(self) -> None:
        """Refresh the metagraph and current block information."""
        self.metagraph = self.subtensor.metagraph(netuid=self.netuid)
        self.current_block = self.subtensor.get_current_block()
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)

    async def benchmark_window(self) -> dict:
        """Benchmark a single window, matching validator logic"""
        window_start = time.perf_counter()

        target_window = self.current_window
        seed_own = self.uid
        context_id = f"window_{target_window}_block_{self.current_block}"

        tplr.logger.info(
            f"Loading {self.pages_per_window} pages for window {target_window}"
        )
        page_start = time.perf_counter()

        local_pages: List = await self.retry_call(
            R2DatasetLoader.next_pages,
            offset=target_window * self.pages_per_window,
            n_pages=self.pages_per_window,
            seed=str(seed_own),
            attempts=3,
            delay=1,
            context=f"pages for {context_id}",
        )  # type: ignore

        page_time = time.perf_counter() - page_start

        if not local_pages:
            raise ValueError(f"No pages loaded for {context_id}")

        self.stats["timing"]["page_loading"].append(page_time)
        self.stats["pages_loaded"] += len(local_pages)

        tplr.logger.info(f"Loaded {len(local_pages)} pages in {page_time:.2f}s")

        window_batches = 0
        window_tokens = 0

        chosen_page = [random.choice(local_pages)]
        loader_start = time.perf_counter()

        loader = await self.retry_call(
            R2DatasetLoader.create,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            pages_info=chosen_page,
            tokenizer=self.tokenizer,
            attempts=3,
            delay=1,
            context=f"loader for {context_id}",
        )

        loader_time = time.perf_counter() - loader_start
        self.stats["timing"]["loader_creation"].append(loader_time)

        batch_start = time.perf_counter()
        batches = []
        if loader is not None:
            for batch in loader:
                batches.append(batch)

        total_batches = len(batches)
        sample_size = max(1, int(total_batches * self.validator_sample_rate))
        sampled_indices = random.sample(range(total_batches), sample_size)
        sampled_batches = [batches[i] for i in sampled_indices]

        for batch in sampled_batches:
            window_tokens += (
                batch.size if hasattr(batch, "size") else np.prod(batch.shape)
            )

        batch_time = time.perf_counter() - batch_start
        self.stats["timing"]["batch_processing"].append(batch_time)

        window_batches = len(sampled_batches)

        window_time = time.perf_counter() - window_start
        self.stats["timing"]["total_window_time"].append(window_time)

        self.stats["windows_processed"] += 1
        self.stats["batches_processed"] += window_batches
        self.stats["total_tokens"] += window_tokens

        window_stats = {
            "window": target_window,
            "block": self.current_block,
            "batches_processed": window_batches,
            "tokens_processed": window_tokens,
            "timing": {
                "total_time": window_time,
                "page_loading_time": page_time,
                "loader_creation_time": loader_time,
                "batch_processing_time": batch_time,
            },
        }

        tplr.logger.info(
            f"Window {target_window} completed: {window_batches} batches, {window_tokens} tokens, {window_time:.2f}s"
        )

        self.window_results.append(window_stats)

        return window_stats

    async def _benchmark(self) -> Optional[int]:
        """Execute benchmark on the current window.

        Returns:
            Optional[int]: Current window number if successful, None on failure
        """
        await self.update_state()

        if self.current_window <= self.last_benchmark_window:
            tplr.logger.info(
                f"Window already benchmarked (current: {self.current_window}, "
                f"last benchmarked: {self.last_benchmark_window})"
            )
            return self.current_window

        tplr.logger.info(
            f"Starting benchmark for window {self.current_window} (block: {self.current_block})"
        )

        await self.benchmark_window()

        self.last_benchmark_window = self.current_window
        self.last_block_number = self.current_block

        tplr.logger.info(
            f"Successfully benchmarked window {self.current_window} "
            f"(block: {self.current_block})"
        )

        return self.current_window

    async def run(self) -> None:
        """Main benchmark loop.

        Continuously:
        1. Check for new windows from mainnet
        2. Trigger benchmark when new window detected
        3. Handle interrupts and errors
        """
        try:
            if self.config.once:
                tplr.logger.info("Running benchmark once on current window...")
                await self.update_state()
                await self._benchmark()
                tplr.logger.info("Single benchmark completed, exiting.")
                return

            tplr.logger.info("Starting R2Dataset live benchmark service...")

            while not self.stop_event.is_set():
                await self.update_state()

                latest_block = self.current_block
                current_window = self.current_window

                if current_window > self.last_benchmark_window:
                    tplr.logger.info(
                        f"New window detected (block: {latest_block}, window: {current_window}), "
                        f"executing benchmark..."
                    )
                    await self._benchmark()
                else:
                    # Calculate blocks until next window
                    blocks_in_current_window = (
                        latest_block % self.hparams.blocks_per_window
                    )
                    blocks_until_next_window = (
                        self.hparams.blocks_per_window - blocks_in_current_window
                    )
                    tplr.logger.info(
                        f"Waiting for new window (current: {current_window}, "
                        f"last benchmarked: {self.last_benchmark_window}). "
                        f"Need {blocks_until_next_window} more blocks for next window."
                    )

                await asyncio.sleep(WINDOW_CHECK_INTERVAL)

        except KeyboardInterrupt:
            tplr.logger.info("Benchmark service interrupted by user")
            self.stop_event.set()
        except Exception as e:
            tplr.logger.error(f"Benchmark service failed: {e}")

    def cleanup(self) -> None:
        """Cleanup resources before exit."""
        self.stop_event.set()

        # Show final statistics
        self.show_final_statistics()

    def show_final_statistics(self) -> None:
        """Generate and display final benchmark statistics."""
        if not self.window_results:
            tplr.logger.info("No benchmark data to display")
            return

        total_time = time.perf_counter() - self.benchmark_start_time
        final_stats = self._calculate_final_stats(self.window_results, total_time)

        if final_stats:
            self.print_summary(final_stats)

        R2DatasetLoader.log_profiling_summary()

    def _calculate_final_stats(
        self, window_results: List[dict], total_time: float
    ) -> dict:
        """Calculate comprehensive benchmark statistics"""
        if not window_results:
            tplr.logger.error("No successful windows to analyze")
            return {}

        timing_stats = {}
        timing_keys = [
            "total_time",
            "page_loading_time",
            "loader_creation_time",
            "batch_processing_time",
        ]

        for key in timing_keys:
            values = [
                w["timing"][key]
                for w in window_results
                if "timing" in w and key in w["timing"]
            ]
            if values:
                timing_stats[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "total": np.sum(values),
                }

        total_tokens = sum(w["tokens_processed"] for w in window_results)
        total_batches = sum(w["batches_processed"] for w in window_results)

        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        batches_per_second = total_batches / total_time if total_time > 0 else 0

        final_stats = {
            "benchmark_summary": {
                "windows_completed": len(window_results),
                "total_runtime_seconds": total_time,
                "total_tokens_processed": total_tokens,
                "total_batches_processed": total_batches,
                "tokens_per_second": tokens_per_second,
                "batches_per_second": batches_per_second,
            },
            "timing_analysis": timing_stats,
            "window_details": window_results,
            "configuration": {
                "batch_size": self.batch_size,
                "sequence_length": self.sequence_length,
                "pages_per_window": self.pages_per_window,
                "validator_sample_rate": self.validator_sample_rate,
            },
        }

        return final_stats

    def print_summary(self, stats: dict):
        """Print a comprehensive benchmark summary"""
        if not stats:
            tplr.logger.error("No statistics to display")
            return

        summary = stats["benchmark_summary"]
        timing = stats["timing_analysis"]

        print("\n" + "=" * 60)
        print("R2DATASET BENCHMARK SUMMARY")
        print("=" * 60)

        print(f"Windows Completed: {summary['windows_completed']}")
        print(f"Total Runtime: {summary['total_runtime_seconds']:.2f}s")
        print(f"Total Tokens: {summary['total_tokens_processed']:,}")
        print(f"Total Batches: {summary['total_batches_processed']:,}")
        print(f"Throughput: {summary['tokens_per_second']:.0f} tokens/sec")
        print(f"Batch Rate: {summary['batches_per_second']:.2f} batches/sec")

        print("\nTIMING BREAKDOWN (per operation):")
        print("-" * 40)

        for operation, stats_dict in timing.items():
            print(f"{operation.replace('_', ' ').title()}:")
            print(f"  Mean: {stats_dict['mean']:.3f}s")
            print(f"  Range: {stats_dict['min']:.3f}s - {stats_dict['max']:.3f}s")
            print(f"  Std Dev: {stats_dict['std']:.3f}s")

        print("\nCONFIGURATION:")
        print("-" * 20)
        config = stats["configuration"]
        for key, value in config.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        print("=" * 60)


def main() -> None:
    """Entry point for the R2Dataset benchmark service."""
    benchmark = R2DatasetBenchmark()
    try:
        if benchmark.config.debug:
            tplr.logger.setLevel("DEBUG")

        tplr.logger.info(
            "R2Dataset Live Benchmark Service - Using uvloop for enhanced async performance"
        )
        asyncio.run(benchmark.run())

    except KeyboardInterrupt:
        tplr.logger.info("Benchmark service interrupted by user")
    except Exception as e:
        tplr.logger.error(f"Benchmark service terminated with error: {e}")
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    uvloop.install()
    main()
