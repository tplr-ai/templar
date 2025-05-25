"""
Base profiler class providing common functionality for all profilers.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

import tplr

T = TypeVar("T")


class BaseProfiler(ABC, Generic[T]):
    """
    Abstract base class for all profilers in the system.

    Provides common functionality like stats management, logging,
    and reset capabilities that all profilers share.
    """

    def __init__(self, name: str = "BaseProfiler"):
        """
        Initialize the base profiler.

        Args:
            name: Name identifier for this profiler instance
        """
        self.name = name
        self._stats_data: Dict[str, Any] = {}
        self._otel_enabled = self._should_enable_otel()

    @abstractmethod
    def get_stats(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a specific key or all statistics.

        Args:
            key: Optional specific key to get stats for

        Returns:
            Dictionary containing statistics
        """
        pass

    @abstractmethod
    def reset(self, key: Optional[str] = None) -> None:
        """
        Reset statistics for a specific key or all statistics.

        Args:
            key: Optional specific key to reset stats for
        """
        pass

    def log_summary(self) -> None:
        """
        Log a summary of all statistics.
        Default implementation that can be overridden.
        """
        tplr.logger.info(f"[{self.name}] Summary:")
        stats = self.get_stats()

        if not stats:
            tplr.logger.info("  No data collected yet")
            return

        self._log_stats_details(stats)

    def _should_enable_otel(self) -> bool:
        """Check if OpenTelemetry should be enabled for this profiler."""
        return os.environ.get("TPLR_ENABLE_OTEL_PROFILING", "0") == "1"

    def _get_otel_integration(self):
        """Get OpenTelemetry integration if enabled."""
        if not self._otel_enabled:
            return None
        from .otel_integration import get_otel_integration

        return get_otel_integration()

    @abstractmethod
    def _log_stats_details(self, stats: Dict[str, Any]) -> None:
        """
        Log the detailed statistics. Must be implemented by subclasses.

        Args:
            stats: Statistics dictionary to log
        """
        pass


class DummyProfiler(BaseProfiler[None]):
    """
    No-op implementation of BaseProfiler for when profiling is disabled.
    Provides all methods but does nothing.
    """

    def __init__(self, name: str = "DummyProfiler"):
        # Don't call super().__init__ to avoid any overhead
        self.name = name

    def get_stats(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Return empty stats"""
        return {}

    def reset(self, key: Optional[str] = None) -> None:
        """No-op reset"""
        pass

    def log_summary(self) -> None:
        """No-op log summary"""
        pass

    def _log_stats_details(self, stats: Dict[str, Any]) -> None:
        """No-op log details"""
        pass


# Type for profiler getter functions
ProfilerGetter = TypeVar("ProfilerGetter", bound=BaseProfiler)


def create_profiler_getter(
    profiler_class: type[ProfilerGetter],
    dummy_class: type[BaseProfiler],
    enable_flag_getter: Callable[[], bool],
    global_instance_name: str,
) -> Callable[[], ProfilerGetter]:
    """
    Factory function to create profiler getter functions with consistent behavior.

    Args:
        profiler_class: The actual profiler class to instantiate
        dummy_class: The dummy profiler class to use when disabled
        enable_flag_getter: Function that returns whether profiler is enabled
        global_instance_name: Name for the global instance variable

    Returns:
        Getter function for the profiler
    """
    _instance: Optional[ProfilerGetter] = None
    _dummy = dummy_class()

    def get_profiler(name: Optional[str] = None) -> ProfilerGetter:
        """Get or create profiler instance"""
        nonlocal _instance

        if not enable_flag_getter():
            return _dummy  # type: ignore

        if _instance is None or (name and _instance.name != name):
            _instance = profiler_class(name or profiler_class.__name__)

        return _instance

    # Store instance reference on the function for testing
    get_profiler._instance = lambda: _instance
    get_profiler._reset = lambda: setattr(get_profiler, "_instance", None)

    return get_profiler
