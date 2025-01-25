# benchmark_pages.py
# ruff: noqa
import os
from dotenv import load_dotenv

# Load .env file before any other imports
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
import tplr


class PageBenchmark:
    def __init__(self):
        self.hparams = tplr.load_hparams()
        self.tokenizer = self.hparams.tokenizer
        self.results = []

    async def benchmark_pages(self, n_pages, n_iterations=5):
        metrics = []

        for i in tqdm(range(n_iterations), desc=f"Testing {n_pages} pages"):
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            try:
                # Get pages
                pages_start = time.time()
                pages = await tplr.dataset.DatasetLoader.next_pages(
                    offset=i, n_pages=n_pages, seed="benchmark_test"
                )
                pages_duration = time.time() - pages_start

                # Create loader
                loader_start = time.time()
                loader = await tplr.dataset.DatasetLoader.create(
                    batch_size=self.hparams.batch_size,
                    sequence_length=self.hparams.sequence_length,
                    pages_info=pages,
                    tokenizer=self.tokenizer,
                )
                loader_duration = time.time() - loader_start

                # Process batches
                process_start = time.time()
                total_tokens = 0
                batch_times = []

                for batch in loader:
                    batch_start = time.time()
                    input_ids = torch.tensor(batch, dtype=torch.long)
                    total_tokens += input_ids.numel()
                    batch_times.append(time.time() - batch_start)

                process_duration = time.time() - process_start

                # Calculate memory usage
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before

                metrics.append(
                    {
                        "n_pages": n_pages,
                        "iteration": i,
                        "total_duration": time.time() - start_time,
                        "pages_fetch_time": pages_duration,
                        "loader_creation_time": loader_duration,
                        "processing_time": process_duration,
                        "total_tokens": total_tokens,
                        "tokens_per_second": total_tokens / process_duration,
                        "avg_batch_time": np.mean(batch_times),
                        "memory_used_mb": memory_used,
                        "num_batches": len(batch_times),
                    }
                )

            except Exception as e:
                print(f"Error with {n_pages} pages: {str(e)}")
                continue

        return metrics

    def plot_results(self, results_df):
        # Create figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Processing Speed
        sns.boxplot(data=results_df, x="n_pages", y="tokens_per_second", ax=ax1)
        ax1.set_title("Processing Speed by Page Count")
        ax1.set_ylabel("Tokens per Second")

        # 2. Memory Usage
        sns.boxplot(data=results_df, x="n_pages", y="memory_used_mb", ax=ax2)
        ax2.set_title("Memory Usage by Page Count")
        ax2.set_ylabel("Memory Used (MB)")

        # 3. Time Components
        time_components = results_df[
            ["n_pages", "pages_fetch_time", "loader_creation_time", "processing_time"]
        ].melt(id_vars=["n_pages"], var_name="Component", value_name="Time (s)")
        sns.boxplot(
            data=time_components, x="n_pages", y="Time (s)", hue="Component", ax=ax3
        )
        ax3.set_title("Time Components by Page Count")

        # 4. Batch Processing Time
        sns.boxplot(data=results_df, x="n_pages", y="avg_batch_time", ax=ax4)
        ax4.set_title("Average Batch Processing Time")
        ax4.set_ylabel("Time (s)")

        plt.tight_layout()
        plt.savefig("page_benchmark_results.png")
        plt.close()


async def main():
    benchmark = PageBenchmark()
    all_metrics = []

    # Test different page counts
    page_counts = [1, 2, 4, 8, 16, 32]

    for n_pages in page_counts:
        metrics = await benchmark.benchmark_pages(n_pages)
        all_metrics.extend(metrics)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_metrics)

    # Save raw results
    results_df.to_csv("benchmark_results.csv", index=False)

    # Print summary statistics
    print("\nSummary Statistics:")
    summary = (
        results_df.groupby("n_pages")
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
    benchmark.plot_results(results_df)


if __name__ == "__main__":
    asyncio.run(main())
