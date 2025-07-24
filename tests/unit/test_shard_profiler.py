"""
Unit tests for ShardProfiler in tplr/profilers/shard_profiler.py.
"""

import os
import tempfile
from unittest import mock

from tplr.profilers import (
    ENABLE_PROFILERS,
    ENABLE_SHARD_PROFILER,
    ShardProfiler,
    get_shard_profiler,
)


class TestShardProfiler:
    """Test the ShardProfiler class and its functionality."""

    def setup_method(self):
        """Set up the test environment to enable profilers for testing."""
        # Mock ENABLE_SHARD_PROFILER to be True for testing
        self.orig_enable_flag = mock.patch("tplr.profilers.ENABLE_SHARD_PROFILER", True)
        self.orig_enable_flag.start()

    def teardown_method(self):
        """Restore original settings after tests."""
        self.orig_enable_flag.stop()

    def test_shard_profiler_init(self):
        """Test that the ShardProfiler initializes correctly."""
        profiler = ShardProfiler(name="TestShardProfiler")
        assert profiler.name == "TestShardProfiler"
        assert isinstance(profiler.shard_performance, dict)
        assert isinstance(profiler._active_timers, dict)

    def test_shard_profiler_read_tracking(self):
        """Test that the ShardProfiler tracks shard reads correctly."""
        profiler = ShardProfiler(name="TestShardProfiler")
        shard_path = "/test/shard.parquet"
        chosen_shard = {"num_rows": 1000, "file_size": 1024 * 1024}

        # Start timing a read
        timer_id = profiler.start_read(shard_path, chosen_shard)
        assert timer_id in profiler._active_timers
        assert shard_path in profiler.shard_performance
        assert profiler.shard_performance[shard_path]["reads"] == 0
        assert profiler.shard_performance[shard_path]["num_rows"] == 1000
        assert profiler.shard_performance[shard_path]["file_size"] == 1024 * 1024

        # End timing the read
        elapsed = profiler.end_read(
            timer_id, shard_path, num_row_groups=10, rows_per_group=100
        )
        assert elapsed > 0
        assert timer_id not in profiler._active_timers
        assert profiler.shard_performance[shard_path]["reads"] == 1
        assert profiler.shard_performance[shard_path]["row_groups"] == 10
        assert profiler.shard_performance[shard_path]["rows_per_group"] == 100
        assert len(profiler.shard_performance[shard_path]["timings"]) == 1

    def test_shard_profiler_multiple_reads(self):
        """Test that the ShardProfiler correctly handles multiple reads."""
        profiler = ShardProfiler(name="TestShardProfiler")
        shard_path = "/test/shard.parquet"
        chosen_shard = {"num_rows": 1000, "file_size": 1024 * 1024}

        # Perform multiple reads
        for _ in range(3):
            timer_id = profiler.start_read(shard_path, chosen_shard)
            profiler.end_read(timer_id, shard_path)

        assert profiler.shard_performance[shard_path]["reads"] == 3
        assert len(profiler.shard_performance[shard_path]["timings"]) == 3

    def test_shard_profiler_get_stats(self):
        """Test that the ShardProfiler's get_stats method works correctly."""
        profiler = ShardProfiler(name="TestShardProfiler")
        shard_path = "/test/shard.parquet"
        chosen_shard = {"num_rows": 1000, "file_size": 1024 * 1024}

        timer_id = profiler.start_read(shard_path, chosen_shard)
        profiler.end_read(timer_id, shard_path)

        stats = profiler.get_stats()
        assert shard_path in stats
        assert stats[shard_path]["reads"] == 1
        assert stats[shard_path]["num_rows"] == 1000
        assert stats[shard_path]["file_size"] == 1024 * 1024
        assert len(stats[shard_path]["timings"]) == 1

    def test_shard_profiler_reset(self):
        """Test that the ShardProfiler's reset method works correctly."""
        profiler = ShardProfiler(name="TestShardProfiler")
        shard_path_1 = "/test/shard1.parquet"
        shard_path_2 = "/test/shard2.parquet"
        chosen_shard = {"num_rows": 1000, "file_size": 1024 * 1024}

        profiler.start_read(shard_path_1, chosen_shard)
        profiler.start_read(shard_path_2, chosen_shard)

        assert shard_path_1 in profiler.shard_performance
        assert shard_path_2 in profiler.shard_performance

        # Test resetting a single shard
        profiler.reset(shard_path_1)
        assert shard_path_1 not in profiler.shard_performance
        assert shard_path_2 in profiler.shard_performance

        # Test resetting all shards
        profiler.reset()
        assert not profiler.shard_performance
        assert not profiler._active_timers

    def test_shard_profiler_log_analysis(self, caplog):
        """Test that the ShardProfiler's log_analysis method works correctly."""
        profiler = ShardProfiler(name="TestShardProfiler")
        shard_path = "/test/shard.parquet"
        chosen_shard = {"num_rows": 1000, "file_size": 1024 * 1024}

        timer_id = profiler.start_read(shard_path, chosen_shard)
        profiler.end_read(timer_id, shard_path)

        # Mock the logger to capture log output
        with mock.patch("tplr.profilers.shard_profiler.logger") as mock_logger:
            profiler.log_analysis()
            # Verify that log_analysis produces output
            assert mock_logger.info.call_count > 0
            # First call should be the analysis header
            assert (
                "SHARD PERFORMANCE ANALYSIS" in mock_logger.info.call_args_list[0][0][0]
            )

    def test_shard_profiler_export_data(self):
        """Test that the ShardProfiler's export_data method works correctly."""
        profiler = ShardProfiler(name="TestShardProfiler")
        shard_path = "/test/shard.parquet"
        chosen_shard = {"num_rows": 1000, "file_size": 1024 * 1024}

        timer_id = profiler.start_read(shard_path, chosen_shard)
        profiler.end_read(timer_id, shard_path)

        # Use a temporary file for testing export
        with tempfile.NamedTemporaryFile(suffix=".json") as temp_file:
            profiler.export_data(temp_file.name)

            # Verify the exported data
            with open(temp_file.name, "r") as f:
                data = f.read()
                assert "shard_statistics" in data
                assert "summary" in data
                assert shard_path in data

    def test_shard_profiler_log_parquet_metadata(self):
        """Test that the ShardProfiler's log_parquet_metadata method works correctly."""
        profiler = ShardProfiler(name="TestShardProfiler")
        shard_path = "/test/shard.parquet"

        # Add a shard to performance data
        chosen_shard = {"num_rows": 1000, "file_size": "unknown"}
        timer_id = profiler.start_read(shard_path, chosen_shard)
        profiler.end_read(timer_id, shard_path)

        # Test logging metadata
        with mock.patch("tplr.profilers.shard_profiler.logger") as mock_logger:
            profiler.log_parquet_metadata(
                shard_path, file_size=2048, num_row_groups=5, total_rows=1000
            )

            # Verify debug message was logged
            assert mock_logger.debug.call_count == 1
            assert shard_path in mock_logger.debug.call_args[0][0]

            # Verify file_size was updated
            assert profiler.shard_performance[shard_path]["file_size"] == 2048
            assert profiler.shard_performance[shard_path]["row_groups"] == 5

    def test_shard_profiler_log_read_details(self):
        """Test that the ShardProfiler's log_read_details method works correctly."""
        profiler = ShardProfiler(name="TestShardProfiler")
        shard_path = "/test/shard.parquet"

        with mock.patch("tplr.profilers.shard_profiler.logger") as mock_logger:
            profiler.log_read_details(
                shard_path=shard_path,
                group_index=2,
                num_row_groups=5,
                offset=100,
                rows_per_group=50,
            )

            # Verify info message was logged
            assert mock_logger.info.call_count == 1
            log_message = mock_logger.info.call_args[0][0]
            assert shard_path in log_message
            assert "Reading row group 2/5" in log_message


class TestShardProfilerDisabled:
    """Test the ShardProfiler with profiling disabled."""

    @mock.patch.dict(os.environ, {"TPLR_ENABLE_SHARD_PROFILER": "0"})
    def test_disabled_shard_profiler(self):
        """Test that the ShardProfiler does nothing when disabled."""
        # Simulate a fresh import with environment variable set
        with mock.patch("tplr.profilers.ENABLE_SHARD_PROFILER", False):
            # Get a profiler instance - should be a dummy
            profiler = get_shard_profiler()

            # The dummy should have the methods but do nothing
            assert hasattr(profiler, "start_read")
            assert hasattr(profiler, "end_read")
            assert hasattr(profiler, "get_stats")
            assert hasattr(profiler, "log_analysis")

            # Start and end a read
            shard_path = "/test/shard.parquet"
            chosen_shard = {"num_rows": 1000, "file_size": 1024 * 1024}

            timer_id = profiler.start_read(shard_path, chosen_shard)
            assert timer_id == "dummy_timer"

            elapsed = profiler.end_read(timer_id, shard_path)
            assert elapsed == 0.0

            # No stats should be collected
            assert not profiler.get_stats()

            # Make sure we can call other methods without errors
            profiler.reset()
            profiler.log_analysis()
            profiler.export_data("test.json")  # This should not create a file


class TestShardProfilerEnableDisable:
    """Test that shard profiler can be enabled and disabled via environment variables."""

    def test_global_disable_shard_profiler(self, monkeypatch):
        """Test that shard profiler can be disabled via TPLR_ENABLE_PROFILERS."""
        # Save original values
        orig_enable_profilers = ENABLE_PROFILERS
        orig_enable_shard_profiler = ENABLE_SHARD_PROFILER

        try:
            # Disable all profilers
            monkeypatch.setenv("TPLR_ENABLE_PROFILERS", "0")

            # Reload the module to pick up environment changes
            import importlib

            import tplr.profilers

            importlib.reload(tplr.profilers)

            # Shard profiler should be disabled
            assert not tplr.profilers.ENABLE_PROFILERS
            assert not tplr.profilers.ENABLE_SHARD_PROFILER

            # Get instance and verify it's a dummy profiler
            shard_profiler = tplr.profilers.get_shard_profiler()

            # Test that it's a dummy implementation
            assert shard_profiler.get_stats() == {}
        finally:
            # Restore original values
            monkeypatch.setattr(
                "tplr.profilers.ENABLE_PROFILERS", orig_enable_profilers
            )
            monkeypatch.setattr(
                "tplr.profilers.ENABLE_SHARD_PROFILER", orig_enable_shard_profiler
            )
