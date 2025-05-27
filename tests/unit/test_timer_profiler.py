"""
Unit tests for TimerProfiler in tplr/profilers/timer_profiler.py.
"""

import asyncio
import os
from unittest import mock

import pytest

from tplr.profilers import (
    ENABLE_PROFILERS,
    ENABLE_TIMER_PROFILER,
    TimerProfiler,
    get_timer_profiler,
)


class TestTimerProfiler:
    """Test the TimerProfiler class and its functionality."""

    def setup_method(self):
        """Set up the test environment to enable profilers for testing."""
        # Mock ENABLE_TIMER_PROFILER to be True for testing
        self.orig_enable_flag = mock.patch("tplr.profilers.ENABLE_TIMER_PROFILER", True)
        self.orig_enable_flag.start()

    def teardown_method(self):
        """Restore original settings after tests."""
        self.orig_enable_flag.stop()

    def test_timer_profiler_init(self):
        """Test that the TimerProfiler initializes correctly."""
        profiler = TimerProfiler(name="TestProfiler")
        assert profiler.name == "TestProfiler"
        assert isinstance(profiler.timings, dict)
        assert isinstance(profiler.active_timers, dict)
        assert isinstance(profiler.counts, dict)

    def test_timer_profiler_profile_sync(self):
        """Test that the TimerProfiler's profile decorator works with sync functions."""
        profiler = TimerProfiler(name="TestProfiler")

        @profiler.profile()
        def test_function():
            return "test"

        result = test_function()
        assert result == "test"
        assert "test_function" in profiler.timings
        assert len(profiler.timings["test_function"]) == 1
        assert profiler.counts["test_function"] == 1
        assert not profiler.active_timers

    @pytest.mark.asyncio
    async def test_timer_profiler_profile_async(self):
        """Test that the TimerProfiler's profile decorator works with async functions."""
        profiler = TimerProfiler(name="TestProfiler")

        @profiler.profile()
        async def test_async_function():
            await asyncio.sleep(0.01)
            return "test_async"

        result = await test_async_function()
        assert result == "test_async"
        assert "test_async_function" in profiler.timings
        assert len(profiler.timings["test_async_function"]) == 1
        assert profiler.counts["test_async_function"] == 1
        assert not profiler.active_timers

    def test_timer_profiler_profile_with_error(self):
        """Test that the TimerProfiler records errors correctly."""
        profiler = TimerProfiler(name="TestProfiler")

        @profiler.profile()
        def test_error_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            test_error_function()

        assert "test_error_function" in profiler.timings
        assert len(profiler.timings["test_error_function"]) == 1
        assert profiler.counts["test_error_function"] == 1
        assert profiler.counts["test_error_function_errors"] == 1
        assert not profiler.active_timers

    def test_timer_profiler_get_stats(self):
        """Test that the TimerProfiler's get_stats method works correctly."""
        profiler = TimerProfiler(name="TestProfiler")

        @profiler.profile()
        def test_function():
            return "test"

        # Run multiple times to get multiple timings
        for _ in range(3):
            test_function()

        # Test getting stats for specific function
        stats = profiler.get_stats("test_function")
        assert stats["function"] == "test_function"
        assert stats["count"] == 3
        assert "min" in stats
        assert "max" in stats
        assert "avg" in stats
        assert "total" in stats
        assert len(stats["timings"]) == 3

        # Test getting stats for all functions
        all_stats = profiler.get_stats()
        assert "test_function" in all_stats
        assert all_stats["test_function"]["count"] == 3

    def test_timer_profiler_reset(self):
        """Test that the TimerProfiler's reset method works correctly."""
        profiler = TimerProfiler(name="TestProfiler")

        @profiler.profile()
        def test_function_1():
            return "test1"

        @profiler.profile()
        def test_function_2():
            return "test2"

        test_function_1()
        test_function_2()

        assert "test_function_1" in profiler.timings
        assert "test_function_2" in profiler.timings

        # Test resetting a single function
        profiler.reset("test_function_1")
        assert "test_function_1" not in profiler.timings
        assert "test_function_2" in profiler.timings

        # Test resetting all functions
        profiler.reset()
        assert not profiler.timings
        assert not profiler.counts
        assert not profiler.active_timers

    def test_timer_profiler_log_summary(self, caplog):
        """Test that the TimerProfiler's log_summary method works correctly."""
        profiler = TimerProfiler(name="TestProfiler")

        @profiler.profile()
        def test_function():
            return "test"

        test_function()

        # Mock the logger to capture log output
        with mock.patch("tplr.profilers.timer_profiler.logger") as mock_logger:
            profiler.log_summary()
            assert mock_logger.info.call_count >= 2
            # First call should be the summary header
            assert mock_logger.info.call_args_list[0][0][0].startswith(
                "[TIMER] TestProfiler Summary:"
            )
            # There should be one call with test_function stats
            function_log_calls = [
                call[0][0]
                for call in mock_logger.info.call_args_list
                if "test_function" in call[0][0]
            ]
            assert len(function_log_calls) == 1


class TestTimerProfilerDisabled:
    """Test the TimerProfiler with profiling disabled."""

    @mock.patch.dict(os.environ, {"TPLR_ENABLE_TIMER_PROFILER": "0"})
    def test_disabled_timer_profiler(self):
        """Test that the TimerProfiler does nothing when disabled."""
        # Simulate a fresh import with environment variable set
        with mock.patch("tplr.profilers.ENABLE_TIMER_PROFILER", False):
            # Get a profiler instance - should be a dummy
            profiler = get_timer_profiler("DisabledTest")

            # The dummy should have the methods but do nothing
            assert hasattr(profiler, "profile")
            assert hasattr(profiler, "get_stats")
            assert hasattr(profiler, "reset")
            assert hasattr(profiler, "log_summary")

            # Create a profiled function
            @profiler.profile()
            def test_function():
                return "test"

            # Function should work normally
            result = test_function()
            assert result == "test"

            # But no stats should be collected
            assert not profiler.get_stats()

            # Make sure we can call other methods without errors
            profiler.reset()
            profiler.log_summary()


class TestTimerProfilerEnableDisable:
    """Test that timer profiler can be enabled and disabled via environment variables."""

    def test_global_disable_timer_profiler(self, monkeypatch):
        """Test that timer profiler can be disabled via TPLR_ENABLE_PROFILERS."""
        # Save original values
        orig_enable_profilers = ENABLE_PROFILERS
        orig_enable_timer_profiler = ENABLE_TIMER_PROFILER

        try:
            # Disable all profilers
            monkeypatch.setenv("TPLR_ENABLE_PROFILERS", "0")

            # Reload the module to pick up environment changes
            import importlib
            import tplr.profilers

            importlib.reload(tplr.profilers)

            # Timer profiler should be disabled
            assert not tplr.profilers.ENABLE_PROFILERS
            assert not tplr.profilers.ENABLE_TIMER_PROFILER

            # Get instance and verify it's a dummy profiler
            timer_profiler = tplr.profilers.get_timer_profiler()

            # Test that it's a dummy implementation
            assert timer_profiler.get_stats() == {}
        finally:
            # Restore original values
            monkeypatch.setattr(
                "tplr.profilers.ENABLE_PROFILERS", orig_enable_profilers
            )
            monkeypatch.setattr(
                "tplr.profilers.ENABLE_TIMER_PROFILER", orig_enable_timer_profiler
            )
