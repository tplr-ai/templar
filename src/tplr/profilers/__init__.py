"""
Profiling utilities for performance tracking and analysis.

Environment variables:
    TPLR_ENABLE_PROFILERS: Set to '1' to enable all profilers (default: '0')
    TPLR_ENABLE_TIMER_PROFILER: Set to '1' to enable timer profiler (default: '0')
    TPLR_ENABLE_SHARD_PROFILER: Set to '1' to enable shard profiler (default: '0')
"""

import os

# Global profiling flags
ENABLE_PROFILERS = os.environ.get("TPLR_ENABLE_PROFILERS", "0") == "1"
ENABLE_TIMER_PROFILER = (
    ENABLE_PROFILERS and os.environ.get("TPLR_ENABLE_TIMER_PROFILER", "0") == "1"
)
ENABLE_SHARD_PROFILER = (
    ENABLE_PROFILERS and os.environ.get("TPLR_ENABLE_SHARD_PROFILER", "0") == "1"
)

# Import profilers after defining flags to avoid circular imports
from .timer_profiler import TimerProfiler, get_timer_profiler  # noqa: E402
from .shard_profiler import ShardProfiler, get_profiler as get_shard_profiler  # noqa: E402

__all__ = [
    "TimerProfiler",
    "get_timer_profiler",
    "ShardProfiler",
    "get_shard_profiler",
    "ENABLE_PROFILERS",
    "ENABLE_TIMER_PROFILER",
    "ENABLE_SHARD_PROFILER",
]
