import asyncio
import time
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from tplr import logger


class TimerProfiler:
    """A modular timer profiler for performance monitoring"""

    def __init__(self, name: str = "TimerProfiler"):
        self.name = name
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.active_timers: Dict[str, float] = {}
        self.counts: Dict[str, int] = defaultdict(int)

    def profile(self, func_name: str = "") -> Callable[..., Any]:
        """Decorator to profile function execution time"""
        from . import ENABLE_TIMER_PROFILER

        def decorator(func):
            # If profiling is disabled, return the original function unchanged
            if not ENABLE_TIMER_PROFILER:
                return func

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                name = func_name or func.__name__
                start_time = time.time()

                # Track in active timers
                timer_id = f"{name}_{id(args)}"
                self.active_timers[timer_id] = start_time

                try:
                    result = await func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    self._record_timing(name, elapsed)
                    logger.info(f"[TIMER] {self.name}.{name}: {elapsed:.4f}s")
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    self._record_timing(name, elapsed, error=True)
                    logger.error(
                        f"[TIMER] {self.name}.{name}: {elapsed:.4f}s (error: {e})"
                    )
                    raise
                finally:
                    # Clean up active timer
                    self.active_timers.pop(timer_id, None)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                name = func_name or func.__name__
                start_time = time.time()

                # Track in active timers
                timer_id = f"{name}_{id(args)}"
                self.active_timers[timer_id] = start_time

                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    self._record_timing(name, elapsed)
                    logger.info(f"[TIMER] {self.name}.{name}: {elapsed:.4f}s")
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    self._record_timing(name, elapsed, error=True)
                    logger.error(
                        f"[TIMER] {self.name}.{name}: {elapsed:.4f}s (error: {e})"
                    )
                    raise
                finally:
                    # Clean up active timer
                    self.active_timers.pop(timer_id, None)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def _record_timing(self, name: str, elapsed: float, error: bool = False) -> None:
        """Record timing information"""
        self.timings[name].append(elapsed)
        self.counts[name] += 1
        if error:
            self.counts[f"{name}_errors"] += 1

    def get_stats(self, func_name: Optional[str] = None) -> Dict:
        """Get timing statistics for a specific function or all functions"""
        if func_name:
            timings = self.timings.get(func_name, [])
            if not timings:
                return {"error": f"No timings found for {func_name}"}

            return {
                "function": func_name,
                "count": self.counts[func_name],
                "errors": self.counts.get(f"{func_name}_errors", 0),
                "min": min(timings),
                "max": max(timings),
                "avg": sum(timings) / len(timings),
                "total": sum(timings),
                "timings": timings[-10:],
            }

        all_stats = {}
        for name in self.timings:
            all_stats[name] = self.get_stats(name)
        return all_stats

    def reset(self, func_name: Optional[str] = None) -> None:
        """Reset timing data for a specific function or all functions"""
        if func_name:
            self.timings.pop(func_name, None)
            self.counts.pop(func_name, None)
            self.counts.pop(f"{func_name}_errors", None)
        else:
            self.timings.clear()
            self.counts.clear()
            self.active_timers.clear()

    def log_summary(self) -> None:
        """Log a summary of all timing statistics"""
        logger.info(f"[TIMER] {self.name} Summary:")
        stats = self.get_stats()

        for func_name, func_stats in stats.items():
            if isinstance(func_stats, dict) and "count" in func_stats:
                logger.info(
                    f"  {func_name}: "
                    f"count={func_stats['count']}, "
                    f"avg={func_stats['avg']:.4f}s, "
                    f"min={func_stats['min']:.4f}s, "
                    f"max={func_stats['max']:.4f}s, "
                    f"total={func_stats['total']:.4f}s"
                )


# Dummy profiler implementation for when profiling is disabled
class DummyTimerProfiler:
    """No-op implementation of TimerProfiler when profiling is disabled"""

    def __init__(self, name: str = "DummyTimerProfiler"):
        self.name = name

    def profile(self, func_name: str = "") -> Callable[..., Any]:
        """No-op decorator that returns the original function unchanged"""

        def decorator(func):
            return func

        return decorator

    def _record_timing(self, *args, **kwargs) -> None:
        pass

    def get_stats(self, *args, **kwargs) -> Dict:
        return {}

    def reset(self, *args, **kwargs) -> None:
        pass

    def log_summary(self) -> None:
        pass


# Global singleton instances
_timer_profiler: Optional[TimerProfiler] = None
_dummy_profiler = DummyTimerProfiler()


def get_timer_profiler(name: str = "TimerProfiler") -> TimerProfiler:
    """
    Get or create a named timer profiler instance.
    Returns a dummy profiler if profiling is disabled.

    Args:
        name: Name for the profiler instance

    Returns:
        TimerProfiler instance (or dummy instance if disabled)
    """
    from . import ENABLE_TIMER_PROFILER

    if not ENABLE_TIMER_PROFILER:
        return _dummy_profiler  # type: ignore

    global _timer_profiler
    if _timer_profiler is None or _timer_profiler.name != name:
        _timer_profiler = TimerProfiler(name)
    return _timer_profiler
