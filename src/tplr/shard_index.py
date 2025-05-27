from functools import lru_cache
from typing import List, Tuple

import numpy as np

from tplr.profilers import get_timer_profiler

_timer_profiler = get_timer_profiler("ShardIndex")


class ShardIndex:
    def __init__(self, shard_data):
        """
        Initialize shard index with precomputed cumulative row counts
        for efficient binary search operations.

        Args:
            shard_data (dict): The loaded _shard_sizes.json data
        """
        self.shard_data = shard_data
        self.indices = {}
        self._numpy_indices = {}

        for config_name, config_data in shard_data.items():
            self._build_index(config_name, config_data)

    def _build_index(self, config_name, config_data):
        """
        Build cumulative index for a specific configuration.

        Args:
            config_name (str): The configuration name
            config_data (dict): The configuration metadata
        """
        shards = config_data.get("shards", [])
        if not shards:
            self.indices[config_name] = {
                "cum_rows": np.array([], dtype=np.int64),
                "shards": [],
                "total_rows": 0,
            }
            self._numpy_indices[config_name] = {
                "cum_rows": np.array([], dtype=np.int64),
                "shard_indices": np.array([], dtype=np.int32),
                "row_counts": np.array([], dtype=np.int64),
            }
            return

        num_shards = len(shards)
        cum_rows = np.zeros(num_shards + 1, dtype=np.int64)
        shard_indices = np.arange(num_shards, dtype=np.int32)

        row_counts = np.array([shard["num_rows"] for shard in shards], dtype=np.int64)
        cum_rows[1:] = np.cumsum(row_counts)

        self.indices[config_name] = {
            "cum_rows": cum_rows.tolist(),
            "shards": shards,
            "total_rows": int(cum_rows[-1]),
        }

        self._numpy_indices[config_name] = {
            "cum_rows": cum_rows,
            "shard_indices": shard_indices,
            "row_counts": row_counts,
        }

    @lru_cache(maxsize=2048)
    @_timer_profiler.profile("find_shard")
    def find_shard(self, config_name: str, page_number: int) -> Tuple[dict, int, int]:
        """
        Find shard
        """
        numpy_data = self._numpy_indices.get(config_name)
        if numpy_data is None:
            raise ValueError(f"No index found for config '{config_name}'")

        cum_rows = numpy_data["cum_rows"]

        if len(cum_rows) == 0 or page_number >= cum_rows[-1]:
            raise ValueError(
                f"Page {page_number} out of bounds for config '{config_name}'"
            )

        shard_idx = np.searchsorted(cum_rows, page_number, side="right") - 1

        if shard_idx < 0 or shard_idx >= len(self.indices[config_name]["shards"]):
            raise ValueError(f"Invalid shard index {shard_idx} for page {page_number}")

        shard = self.indices[config_name]["shards"][shard_idx]
        shard_offset = page_number - cum_rows[shard_idx]

        return shard, int(shard_offset), int(shard_idx)

    def find_shard_batch(
        self, config_name: str, page_numbers: List[int]
    ) -> List[Tuple[dict, int, int]]:
        """
        Batch find operation for multiple pages
        """
        numpy_data = self._numpy_indices.get(config_name)
        if numpy_data is None:
            raise ValueError(f"No index found for config '{config_name}'")

        cum_rows = numpy_data["cum_rows"]
        shards = self.indices[config_name]["shards"]

        page_array = np.array(page_numbers, dtype=np.int64)
        shard_indices = np.searchsorted(cum_rows, page_array, side="right") - 1

        results = []
        shards_len = len(shards)
        for i, (page_num, shard_idx) in enumerate(zip(page_numbers, shard_indices)):
            if shard_idx < 0 or shard_idx >= shards_len or page_num >= cum_rows[-1]:
                raise ValueError(f"Page {page_num} out of bounds")

            shard = shards[shard_idx]
            shard_offset = page_num - cum_rows[shard_idx]
            results.append((shard, int(shard_offset), int(shard_idx)))

        return results

    def get_shard_range(self, config_name: str, shard_idx: int) -> Tuple[int, int]:
        """
        Get the row range for a specific shard - useful for prefetching.
        """
        numpy_data = self._numpy_indices.get(config_name)
        if numpy_data is None:
            raise ValueError(f"No index found for config '{config_name}'")

        cum_rows = numpy_data["cum_rows"]
        if shard_idx < 0 or shard_idx >= len(cum_rows) - 1:
            raise ValueError(f"Invalid shard index {shard_idx}")

        return int(cum_rows[shard_idx]), int(cum_rows[shard_idx + 1])

    def clear_cache(self):
        """Clear the LRU cache if needed."""
        self.find_shard.cache_clear()

    def warm_cache(self, config_name: str, page_numbers: List[int]):
        """Pre-warm the cache with expected page lookups."""
        for page_num in page_numbers:
            try:
                self.find_shard(config_name, page_num)
            except ValueError:
                pass

    @staticmethod
    def get_profiling_stats():
        """Get timing statistics from the profiler"""
        return _timer_profiler.get_stats()

    @staticmethod
    def log_profiling_summary():
        """Log a summary of all timing statistics"""
        _timer_profiler.log_summary()

    @staticmethod
    def reset_profiling_stats(func_name: str = ""):
        """Reset profiling statistics"""
        _timer_profiler.reset(func_name)


def get_shard_index_profiler():
    """Get the ShardIndex timer profiler instance"""
    return _timer_profiler
