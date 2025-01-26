# ruff: noqa
# Load .env file before any other imports
import os
from dotenv import load_dotenv

load_dotenv(override=True)

import asyncio
import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
from pathlib import Path
from transformers import AutoTokenizer
import sys

# Check for required environment variables
required_vars = [
    "R2_ACCOUNT_ID",
    "R2_READ_ACCESS_KEY_ID",
    "R2_READ_SECRET_ACCESS_KEY",
    "R2_WRITE_ACCESS_KEY_ID",
    "R2_WRITE_SECRET_ACCESS_KEY",
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print("Please create a .env file with the following variables:")
    for var in missing_vars:
        print(f"{var}=your_{var.lower()}_here")
    print("\nOr export them in your shell:")
    for var in missing_vars:
        print(f"export {var}=your_{var.lower()}_here")
    sys.exit(1)

# Now safe to import tplr
import tplr
from tplr.r2_dataset import R2DatasetLoader
from tplr.logging import logger, debug, T


class ParquetLoaderBenchmark:
    def __init__(self):
        debug()  # Enable debug logging
        self.hparams = tplr.load_hparams()
        self.tokenizer = self.hparams.tokenizer
        self.results = []

    async def benchmark_loader(
        self, n_pages, batch_size, sequence_length, n_iterations=3
    ):
        metrics = []

        for i in tqdm(
            range(n_iterations),
            desc=f"Testing {n_pages} pages, batch={batch_size}, seq={sequence_length}",
        ):
            start_time = T()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            try:
                # Get pages
                pages_start = T()
                pages = await R2DatasetLoader.next_pages(
                    offset=i, n_pages=n_pages, seed=f"benchmark_{i}"
                )
                pages_duration = T() - pages_start
                logger.info(f"Pages generated: {pages}")

                # Create loader
                loader_start = T()
                loader = await R2DatasetLoader.create(
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    pages_info=pages,
                    tokenizer=self.tokenizer,
                    pack_samples=True,
                )
                loader_duration = T() - loader_start

                # Process batches
                process_start = T()
                total_tokens = 0
                batch_times = []
                n_batches = 0

                for batch in loader:
                    batch_start = T()
                    total_tokens += batch.numel()
                    batch_times.append(T() - batch_start)
                    n_batches += 1
                    if n_batches >= 5:  # Limit to 5 batches per iteration
                        break

                process_duration = T() - process_start

                # Calculate memory usage
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = memory_after - memory_before

                metrics.append(
                    {
                        "n_pages": n_pages,
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "iteration": i,
                        "total_duration": T() - start_time,
                        "pages_fetch_time": pages_duration,
                        "loader_creation_time": loader_duration,
                        "processing_time": process_duration,
                        "total_tokens": total_tokens,
                        "tokens_per_second": total_tokens / process_duration
                        if process_duration > 0
                        else 0,
                        "avg_batch_time": np.mean(batch_times),
                        "memory_used_mb": memory_used,
                        "num_batches": n_batches,
                    }
                )

                logger.info(f"Iteration {i} complete: {metrics[-1]}")

            except Exception as e:
                logger.error(f"Error in iteration {i}: {str(e)}", exc_info=True)
                continue

        return metrics

    def plot_results(self, results_df, output_dir="benchmark_results"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create plots for different metrics
        metrics_to_plot = [
            ("tokens_per_second", "Tokens/Second"),
            ("memory_used_mb", "Memory Usage (MB)"),
            ("total_duration", "Total Duration (s)"),
            ("avg_batch_time", "Avg Batch Time (s)"),
        ]

        for metric, title in metrics_to_plot:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=results_df, x="n_pages", y=metric, hue="batch_size")
            plt.title(f"{title} by Pages and Batch Size")
            plt.savefig(Path(output_dir) / f"{metric}_analysis.png")
            plt.close()

        # Create heatmap for sequence length impact
        pivot_data = results_df.pivot_table(
            values="tokens_per_second",
            index="sequence_length",
            columns="batch_size",
            aggfunc="mean",
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt=".0f", cmap="YlOrRd")
        plt.title("Tokens/Second by Sequence Length and Batch Size")
        plt.savefig(Path(output_dir) / "sequence_length_heatmap.png")
        plt.close()


async def main():
    # Create benchmark results directory in scripts/benchmarks
    output_dir = Path(__file__).parent / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = ParquetLoaderBenchmark()
    all_metrics = []

    # Test configurations
    configs = [
        # n_pages, batch_size, sequence_length
        (1, 4, 512),
        (1, 8, 512),
        (2, 4, 512),
        (2, 8, 512),
        (4, 4, 512),
        (5, 6, 2048),
    ]

    for n_pages, batch_size, sequence_length in configs:
        logger.info(
            f"\nTesting configuration: pages={n_pages}, batch={batch_size}, seq={sequence_length}"
        )
        metrics = await benchmark.benchmark_loader(
            n_pages=n_pages, batch_size=batch_size, sequence_length=sequence_length
        )
        all_metrics.extend(metrics)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_metrics)

    # Save raw results
    results_df.to_csv(output_dir / "parquet_loader_results.csv", index=False)

    # Print summary statistics
    print("\nSummary Statistics:")
    summary = (
        results_df.groupby(["n_pages", "batch_size", "sequence_length"])
        .agg(
            {
                "total_duration": ["mean", "std"],
                "tokens_per_second": ["mean", "std"],
                "memory_used_mb": ["mean", "std"],
                "num_batches": "mean",
            }
        )
        .round(2)
    )
    print(summary)

    # Plot results
    benchmark.plot_results(results_df, output_dir=str(output_dir))


if __name__ == "__main__":
    asyncio.run(main())
