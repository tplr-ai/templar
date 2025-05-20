#!/usr/bin/env python3
"""
Analyze shard performance data to identify problematic parquet files.
Usage: python analyze_shard_performance.py [shard_performance_report.json]

If no file is provided, it will export data from the current ShardProfiler instance.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tplr.profilers import get_shard_profiler


def analyze_performance(json_file):
    """Analyze and visualize shard performance data"""
    with open(json_file) as f:
        data = json.load(f)

    shards = data["shard_statistics"]

    # Extract metrics
    shard_names = []
    avg_times = []
    max_times = []
    file_sizes = []
    num_rows = []

    for shard_path, stats in shards.items():
        shard_names.append(Path(shard_path).name)
        avg_times.append(stats["avg_time"])
        max_times.append(stats["max_time"])

        # Convert file size to MB if possible
        size = stats["file_size"]
        if isinstance(size, (int, float)):
            file_sizes.append(size / (1024 * 1024))  # Convert to MB
        else:
            file_sizes.append(0)

        num_rows.append(stats["num_rows"])

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Average read time histogram
    ax1.hist(avg_times, bins=50, edgecolor="black")
    ax1.set_xlabel("Average Read Time (seconds)")
    ax1.set_ylabel("Number of Shards")
    ax1.set_title("Distribution of Average Read Times")
    ax1.axvline(
        np.mean(avg_times),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(avg_times):.4f}s",
    )
    ax1.legend()

    # 2. Scatter plot: file size vs avg time
    ax2.scatter(file_sizes, avg_times, alpha=0.6)
    ax2.set_xlabel("File Size (MB)")
    ax2.set_ylabel("Average Read Time (seconds)")
    ax2.set_title("File Size vs Read Time")

    # 3. Top 20 slowest shards
    sorted_indices = np.argsort(avg_times)[-20:]
    ax3.barh(range(20), [avg_times[i] for i in sorted_indices])
    ax3.set_yticks(range(20))
    ax3.set_yticklabels([shard_names[i][:30] for i in sorted_indices], fontsize=8)
    ax3.set_xlabel("Average Read Time (seconds)")
    ax3.set_title("Top 20 Slowest Shards")

    # 4. Box plot of read times
    recent_timings = []
    for stats in shards.values():
        recent_timings.extend(stats.get("recent_timings", []))

    if recent_timings:
        ax4.boxplot([recent_timings])
        ax4.set_ylabel("Read Time (seconds)")
        ax4.set_title("Distribution of Recent Read Times")
        ax4.set_xticklabels(["All Shards"])

    plt.tight_layout()

    # Save the plot
    output_file = json_file.replace(".json", "_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Analysis plot saved to: {output_file}")

    # Print summary statistics
    print("\n=== SHARD PERFORMANCE SUMMARY ===")
    print(f"Total shards: {len(shards)}")
    print(f"Average read time: {np.mean(avg_times):.4f}s")
    print(f"Median read time: {np.median(avg_times):.4f}s")
    print(f"Standard deviation: {np.std(avg_times):.4f}s")
    print(f"Min read time: {np.min(avg_times):.4f}s")
    print(f"Max read time: {np.max(avg_times):.4f}s")

    # Identify outliers (> 3 std deviations)
    mean_time = np.mean(avg_times)
    std_time = np.std(avg_times)
    outliers = [
        (shard, stats)
        for shard, stats in shards.items()
        if stats["avg_time"] > mean_time + 3 * std_time
    ]

    if outliers:
        print(f"\nOUTLIERS (>{mean_time + 3 * std_time:.4f}s):")
        for shard, stats in outliers[:10]:
            print(f"  {shard}")
            print(f"    Avg: {stats['avg_time']:.4f}s")
            print(f"    Rows: {stats['num_rows']}")
            print(f"    Row groups: {stats.get('row_groups', 'N/A')}")
            print(f"    File size: {stats['file_size']}")

    return data


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print(
            "Usage: python analyze_shard_performance.py [shard_performance_report.json]"
        )
        sys.exit(1)

    if len(sys.argv) == 2:
        json_file = sys.argv[1]
    else:
        # Export current profiler data to a temporary file
        json_file = "shard_performance_report.json"
        get_shard_profiler().export_data(json_file)
        print(f"Exported current profiler data to {json_file}")

    analyze_performance(json_file)
