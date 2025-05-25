"""
Shard performance profiler for tracking parquet file read performance.
"""

import json
import time
from typing import Any, Dict, Optional

import tplr

from .base_profiler import BaseProfiler, DummyProfiler, create_profiler_getter


class ShardProfiler(BaseProfiler[Dict[str, Dict]]):
    """Profile shard read performance to identify bottlenecks"""

    def __init__(self, name: str = "ShardProfiler"):
        super().__init__(name)
        self.shard_performance: Dict[str, Dict] = {}
        self._active_timers: Dict[str, float] = {}

    def start_read(
        self,
        shard_path: str,
        chosen_shard: dict,
        pf_data: dict = None,  # type: ignore
    ) -> str:
        """
        Start timing a shard read operation.

        Args:
            shard_path: Path to the shard file
            chosen_shard: Shard metadata dictionary
            pf_data: Optional parquet file data

        Returns:
            Timer ID for this operation
        """
        timer_id = f"{shard_path}_{time.time()}"
        self._active_timers[timer_id] = time.time()

        if shard_path not in self.shard_performance:
            self.shard_performance[shard_path] = {
                "reads": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "num_rows": chosen_shard["num_rows"],
                "file_size": chosen_shard.get("file_size", "unknown"),
                "timings": [],
                "row_groups": None,
                "rows_per_group": None,
                "metadata": pf_data.get("metadata", {}) if pf_data else {},
            }

        return timer_id

    def end_read(
        self,
        timer_id: str,
        shard_path: str,
        num_row_groups: int = None,  # type: ignore
        rows_per_group: int = None,  # type: ignore
    ) -> float:
        """
        End timing a shard read operation.

        Args:
            timer_id: Timer ID from start_read
            shard_path: Path to the shard file
            num_row_groups: Number of row groups in the file
            rows_per_group: Number of rows per group

        Returns:
            Elapsed time for this read
        """
        if timer_id not in self._active_timers:
            tplr.logger.error(f"Timer {timer_id} not found")
            return 0.0

        start_time = self._active_timers.pop(timer_id)
        elapsed = time.time() - start_time

        perf_data = self.shard_performance[shard_path]
        perf_data["reads"] += 1
        perf_data["total_time"] += elapsed
        perf_data["min_time"] = min(perf_data["min_time"], elapsed)
        perf_data["max_time"] = max(perf_data["max_time"], elapsed)
        perf_data["timings"].append(elapsed)

        if num_row_groups is not None and perf_data["row_groups"] is None:
            perf_data["row_groups"] = num_row_groups
            perf_data["rows_per_group"] = rows_per_group

        if len(perf_data["timings"]) > 100:
            perf_data["timings"] = perf_data["timings"][-100:]

        avg_time = perf_data["total_time"] / perf_data["reads"]

        if elapsed > avg_time * 2 and perf_data["reads"] > 10:
            tplr.logger.warning(
                f"SLOW READ DETECTED: {shard_path} took {elapsed:.4f}s "
                f"(2x slower than avg: {avg_time:.4f}s)"
            )

        return elapsed

    def log_read_details(
        self,
        shard_path: str,
        group_index: int,
        num_row_groups: int,
        offset: int,
        rows_per_group: int,
    ):
        """Log detailed information about a row group read"""
        tplr.logger.info(
            f"Reading row group {group_index}/{num_row_groups} from shard: {shard_path}, "
            f"offset: {offset}, rows_per_group: {rows_per_group}"
        )

    def log_parquet_metadata(
        self,
        shard_path: str,
        file_size,
        num_row_groups: int,
        total_rows: int,
    ):
        """Log parquet file metadata"""
        tplr.logger.debug(
            f"Opening parquet file: {shard_path} - "
            f"Size: {file_size}, Row groups: {num_row_groups}, "
            f"Total rows: {total_rows}"
        )

        if shard_path in self.shard_performance:
            perf_data = self.shard_performance[shard_path]
            if file_size != "unknown":
                perf_data["file_size"] = file_size
            if perf_data["row_groups"] is None:
                perf_data["row_groups"] = num_row_groups

    def log_read_complete(self, shard_path: str, elapsed: float):
        """Log completion of a read operation"""
        perf_data = self.shard_performance.get(shard_path)
        if not perf_data:
            return

        avg_time = perf_data["total_time"] / perf_data["reads"]
        tplr.logger.info(
            f"Row group read completed from {shard_path} in {elapsed:.4f}s "
            f"(avg: {avg_time:.4f}s, min: {perf_data['min_time']:.4f}s, "
            f"max: {perf_data['max_time']:.4f}s, reads: {perf_data['reads']})"
        )

    def get_stats(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Get shard performance statistics for a specific shard or all shards"""
        if key:
            return self.shard_performance.get(
                key, {"error": f"No stats found for {key}"}
            )
        return self.shard_performance

    def log_analysis(self):
        """Log detailed analysis of shard performance"""
        if not self.shard_performance:
            tplr.logger.info("No shard performance data collected yet")
            return

        tplr.logger.info("=== SHARD PERFORMANCE ANALYSIS ===")

        # Collect all shard stats
        shard_stats = []
        for shard_path, stats in self.shard_performance.items():
            shard_stats.append(
                {
                    "path": shard_path,
                    "avg_time": stats["total_time"] / stats["reads"]
                    if stats["reads"] > 0
                    else 0,
                    "max_time": stats["max_time"],
                    "min_time": stats["min_time"],
                    "reads": stats["reads"],
                    "num_rows": stats["num_rows"],
                    "file_size": stats["file_size"],
                    "total_time": stats["total_time"],
                }
            )

        shard_stats.sort(key=lambda x: x["avg_time"], reverse=True)

        tplr.logger.info(f"Total shards analyzed: {len(shard_stats)}")
        tplr.logger.info("\nTop 10 SLOWEST shards by average read time:")
        for i, shard in enumerate(shard_stats[:10]):
            tplr.logger.info(
                f"{i + 1}. {shard['path']}\n"
                f"   Avg: {shard['avg_time']:.4f}s, Max: {shard['max_time']:.4f}s, "
                f"Min: {shard['min_time']:.4f}s\n"
                f"   Reads: {shard['reads']}, Rows: {shard['num_rows']}, "
                f"Size: {shard['file_size']}"
            )

        if shard_stats:
            global_avg = sum(s["avg_time"] for s in shard_stats) / len(shard_stats)
            global_total_time = sum(s["total_time"] for s in shard_stats)
            global_total_reads = sum(s["reads"] for s in shard_stats)

            tplr.logger.info("\nGLOBAL STATISTICS:")
            tplr.logger.info(f"Overall average read time: {global_avg:.4f}s")
            tplr.logger.info(f"Total cumulative read time: {global_total_time:.2f}s")
            tplr.logger.info(f"Total number of reads: {global_total_reads}")

            outliers = [s for s in shard_stats if s["avg_time"] > global_avg * 2]
            if outliers:
                tplr.logger.warning("\nPERFORMANCE OUTLIERS (2x slower than average):")
                for outlier in outliers:
                    tplr.logger.warning(
                        f"- {outlier['path']}: {outlier['avg_time']:.4f}s "
                        f"({outlier['avg_time'] / global_avg:.1f}x slower)"
                    )

        tplr.logger.info("=== END SHARD PERFORMANCE ANALYSIS ===")

    def export_data(self, output_file: str = "shard_performance_report.json"):
        """Export shard performance data to a JSON file"""
        if not self.shard_performance:
            tplr.logger.info("No shard performance data to export")
            return

        export_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "shard_statistics": {},
            "summary": {
                "total_shards": 0,
                "total_reads": 0,
                "total_time": 0.0,
                "slowest_shards": [],
            },
        }

        for shard_path, stats in self.shard_performance.items():
            avg_time = stats["total_time"] / stats["reads"] if stats["reads"] > 0 else 0
            export_data["shard_statistics"][shard_path] = {
                "reads": stats["reads"],
                "total_time": stats["total_time"],
                "avg_time": avg_time,
                "min_time": stats["min_time"],
                "max_time": stats["max_time"],
                "num_rows": stats["num_rows"],
                "file_size": stats["file_size"],
                "row_groups": stats.get("row_groups"),
                "rows_per_group": stats.get("rows_per_group"),
                "metadata": stats.get("metadata", {}),
                "recent_timings": stats["timings"][-10:],  # Last 10 timings
            }

            export_data["summary"]["total_reads"] += stats["reads"]
            export_data["summary"]["total_time"] += stats["total_time"]

        export_data["summary"]["total_shards"] = len(self.shard_performance)

        sorted_shards = sorted(
            export_data["shard_statistics"].items(),
            key=lambda x: x[1]["avg_time"],
            reverse=True,
        )

        export_data["summary"]["slowest_shards"] = [
            {"path": path, "avg_time": data["avg_time"], "reads": data["reads"]}
            for path, data in sorted_shards[:10]
        ]

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        tplr.logger.info(f"Shard performance data exported to {output_file}")

    def reset(self, key: Optional[str] = None) -> None:
        """Reset performance data for a specific shard or all shards"""
        if key:
            self.shard_performance.pop(key, None)
        else:
            self.shard_performance.clear()
            self._active_timers.clear()

    def _log_stats_details(self, stats: Dict[str, Any]) -> None:
        """Log the detailed shard performance statistics"""
        # This is handled by log_analysis() method
        self.log_analysis()


# Dummy profiler implementation for when profiling is disabled
class DummyShardProfiler(DummyProfiler):
    """No-op implementation of ShardProfiler when profiling is disabled"""

    def start_read(self, *args, **kwargs) -> str:
        return "dummy_timer"

    def end_read(self, *args, **kwargs) -> float:
        return 0.0

    def log_read_details(self, *args, **kwargs) -> None:
        pass

    def log_parquet_metadata(self, *args, **kwargs) -> None:
        pass

    def log_read_complete(self, *args, **kwargs) -> None:
        pass

    def log_analysis(self) -> None:
        pass

    def export_data(self, *args, **kwargs) -> None:
        pass


# Create the getter function using the factory
get_profiler = create_profiler_getter(
    profiler_class=ShardProfiler,
    dummy_class=DummyShardProfiler,
    enable_flag_getter=lambda: __import__(
        "tplr.profilers", fromlist=["ENABLE_SHARD_PROFILER"]
    ).ENABLE_SHARD_PROFILER,
    global_instance_name="_shard_profiler",
)
