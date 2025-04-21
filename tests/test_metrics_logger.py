"""
Unit tests for the MetricsLogger class.
"""

import json
import unittest.mock as mock
import asyncio
import concurrent.futures
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from bittensor import Config as BT_Config
from influxdb_client.client.write.point import Point

import tplr
from tplr.metrics import MetricsLogger


# patch the constructor before importing MetricsLogger (avoids real HTTP)
with patch("tplr.metrics.InfluxDBClient", autospec=True):
    from tplr.metrics import MetricsLogger


# ---------------------------------------------------------------------------
# helper: make the current event‑loop synchronous for this test only
# ---------------------------------------------------------------------------
def _patch_loop_run_sync(monkeypatch):
    loop = asyncio.get_event_loop()

    def _run_sync(executor, func, *args, **kwargs):
        """Immediate, in‑thread replacement for run_in_executor."""
        func(*args, **kwargs)

        class _Done:
            def result(self, _=None):  # noqa: D401
                return None

        return _Done()

    monkeypatch.setattr(loop, "run_in_executor", _run_sync, raising=True)
    return loop


# ---------------------------------------------------------------------------
# fixture: returns a fresh logger + its write() mock each time
# ---------------------------------------------------------------------------
@pytest.fixture
def logger(monkeypatch):
    _patch_loop_run_sync(monkeypatch)  # make run_in_executor sync

    write_mock = MagicMock()
    log = MetricsLogger(prefix="test")
    log.write_api = SimpleNamespace(write=write_mock)  # no real I/O
    return log, write_mock


@pytest.fixture(autouse=True)
def _sync_run_in_executor(monkeypatch):
    """
    • Ensure the main thread has an event‑loop (needed by MetricsLogger).
    • Patch BaseEventLoop.run_in_executor so it executes the task immediately.
    """

    # ------------------------------------------------------------------
    # 1.  guarantee an event‑loop is present
    # ------------------------------------------------------------------
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:  # no loop yet
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # ------------------------------------------------------------------
    # 2.  patch run_in_executor → synchronous
    # ------------------------------------------------------------------
    def _run_sync(self, executor, func, *args, **kwargs):
        try:
            res = func(*args, **kwargs)  # run immediately
        except Exception as exc:
            fut = concurrent.futures.Future()
            fut.set_exception(exc)
            return fut

        fut = concurrent.futures.Future()
        fut.set_result(res)
        return fut

    monkeypatch.setattr(
        asyncio.BaseEventLoop, "run_in_executor", _run_sync, raising=True
    )


class TestMetricsLogger:
    """Test suite for the MetricsLogger class."""

    @pytest.fixture
    def mock_influxdb_client(self):
        """Mock the InfluxDB client and its write API."""
        with mock.patch("tplr.metrics.InfluxDBClient") as mock_client:
            mock_write_api = mock.MagicMock()
            mock_client.return_value.write_api.return_value = mock_write_api
            mock_client.return_value.ping.return_value = True
            yield mock_client

    @pytest.fixture
    def mock_cuda_functions(self):
        """Mock CUDA-related functions."""
        with (
            mock.patch("tplr.metrics.torch.cuda.memory_stats") as mock_memory_stats,
            mock.patch(
                "tplr.metrics.torch.cuda.memory_allocated"
            ) as mock_memory_allocated,
            mock.patch(
                "tplr.metrics.torch.cuda.memory_reserved"
            ) as mock_memory_reserved,
            mock.patch(
                "tplr.metrics.torch.cuda.max_memory_allocated"
            ) as mock_max_memory_allocated,
            mock.patch("tplr.metrics.torch.cuda.current_device") as mock_current_device,
        ):
            mock_current_device.return_value = 0
            mock_memory_stats.return_value = {"segment.all.current": 10000}
            mock_memory_allocated.return_value = 1024 * 1024 * 100  # 100 MB
            mock_memory_reserved.return_value = 1024 * 1024 * 200  # 200 MB
            mock_max_memory_allocated.return_value = 1024 * 1024 * 150  # 150 MB

            yield {
                "memory_stats": mock_memory_stats,
                "memory_allocated": mock_memory_allocated,
                "memory_reserved": mock_memory_reserved,
                "max_memory_allocated": mock_max_memory_allocated,
                "current_device": mock_current_device,
            }

    @pytest.fixture
    def mock_system_metrics(self):
        """Mock system metrics functions."""
        with (
            mock.patch("tplr.metrics.psutil.cpu_percent") as mock_cpu_percent,
            mock.patch("tplr.metrics.psutil.virtual_memory") as mock_virtual_memory,
        ):
            mock_cpu_percent.return_value = 35.5

            mock_memory = mock.MagicMock()
            mock_memory.used = 8 * 1024 * 1024 * 1024  # 8 GB
            mock_memory.total = 16 * 1024 * 1024 * 1024  # 16 GB
            mock_virtual_memory.return_value = mock_memory

            yield {
                "cpu_percent": mock_cpu_percent,
                "virtual_memory": mock_virtual_memory,
            }

    @pytest.fixture
    def bt_config(self):
        """Create a bittensor Config object for testing."""
        config = BT_Config()
        config.netuid = 268
        config.neuron_name = "test_miner"
        config.logging = mock.MagicMock()
        config.logging.debug = False
        config.logging.trace = False
        return config

    @pytest.fixture
    def metrics_logger(self, mock_influxdb_client):
        """Create a MetricsLogger instance for testing."""
        logger = MetricsLogger(
            prefix="test",
            uid=123,  # type: ignore
            role="miner",
            group="test_group",
            job_type="training",
        )
        return logger

    def test_init(self, mock_influxdb_client):
        """Test MetricsLogger initialization."""
        logger = MetricsLogger()

        mock_influxdb_client.assert_called_once()
        assert logger.prefix == ""
        assert logger.database == tplr.metrics.DEFAULT_DATABASE
        assert logger.org == tplr.metrics.DEFAULT_ORG

        test_prefix = "custom_prefix"
        test_uid = 42
        test_role = "validator"

        logger = MetricsLogger(
            prefix=test_prefix,
            uid=test_uid,  # type: ignore
            role=test_role,
            host="custom_host",
            port=9999,
            database="custom_db",
            org="custom_org",
        )

        assert logger.prefix == test_prefix
        assert logger.uid == test_uid
        assert logger.role == test_role
        assert logger.database == "custom_db"
        assert logger.org == "custom_org"

    def test_process_value(self, metrics_logger):
        """Test processing of different value types."""
        assert metrics_logger.process_value(42) == 42.0
        assert metrics_logger.process_value(3.14) == 3.14

        peer_ids = [1, 2, 3, 4, 5]
        assert metrics_logger.process_value(peer_ids) == str(peer_ids)

        values = [1.1, 2.2, 3.3, 4.4, 5.5]
        result = metrics_logger.process_value(values)
        assert isinstance(result, dict)
        assert "mean" in result
        assert "min" in result
        assert "max" in result
        assert "median" in result
        assert result["mean"] == pytest.approx(3.3)
        assert result["min"] == 1.1
        assert result["max"] == 5.5
        assert result["median"] == 3.3

        assert metrics_logger.process_value([]) == 0.0

        assert metrics_logger.process_value("test") == "test"

    def test_log_basic(self, metrics_logger, mock_influxdb_client):
        """Test basic logging functionality."""
        measurement = "test_measurement"
        tags = {"tag1": "value1", "tag2": "value2"}
        fields = {"field1": 1.0, "field2": 2.0}

        metrics_logger.log(
            measurement=measurement,
            tags=tags,
            fields=fields,
            with_system_metrics=False,
            with_gpu_metrics=False,
        )

        write_api = mock_influxdb_client.return_value.write_api.return_value
        write_api.write.assert_called_once()

        call_args = write_api.write.call_args
        assert call_args is not None

        record = call_args[1]["record"]
        assert isinstance(record, Point)
        assert record._name == f"test{measurement}"

        line_protocol = record.to_line_protocol()

        for tag_key, tag_value in tags.items():
            assert f"{tag_key}={tag_value}" in line_protocol

        for field_key, field_value in fields.items():
            # In InfluxDB line protocol, float values may be written as integers if they are whole numbers
            if isinstance(field_value, float) and field_value.is_integer():
                assert f"{field_key}={int(field_value)}" in line_protocol
            else:
                assert f"{field_key}={field_value}" in line_protocol

        assert f"uid={metrics_logger.uid}" in line_protocol
        assert f"role={metrics_logger.role}" in line_protocol

    def test_log_with_system_metrics(
        self, metrics_logger, mock_influxdb_client, mock_system_metrics
    ):
        """Test logging with system metrics."""
        measurement = "test_with_system"
        tags = {"tag1": "value1"}
        fields = {"field1": 1.0}

        metrics_logger.log(
            measurement=measurement,
            tags=tags,
            fields=fields,
            with_system_metrics=True,
            with_gpu_metrics=False,
        )

        write_api = mock_influxdb_client.return_value.write_api.return_value
        write_api.write.assert_called_once()

        mock_system_metrics["cpu_percent"].assert_called_once()
        mock_system_metrics["virtual_memory"].assert_called_once()

        record = write_api.write.call_args[1]["record"]
        line_protocol = record.to_line_protocol()

        assert "sys_cpu_usage=35.5" in line_protocol
        assert "sys_mem_used=" in line_protocol
        assert "sys_mem_total=" in line_protocol

    def test_log_with_gpu_metrics(
        self, metrics_logger, mock_influxdb_client, mock_cuda_functions
    ):
        """Test logging with GPU metrics."""
        measurement = "test_with_gpu"
        tags = {"tag1": "value1"}
        fields = {"field1": 1.0}

        metrics_logger.log(
            measurement=measurement,
            tags=tags,
            fields=fields,
            with_system_metrics=False,
            with_gpu_metrics=True,
        )

        write_api = mock_influxdb_client.return_value.write_api.return_value
        write_api.write.assert_called_once()

        mock_cuda_functions["current_device"].assert_called_once()
        mock_cuda_functions["memory_stats"].assert_called_once()
        mock_cuda_functions["memory_allocated"].assert_called_once()
        mock_cuda_functions["memory_reserved"].assert_called_once()
        mock_cuda_functions["max_memory_allocated"].assert_called_once()

        record = write_api.write.call_args[1]["record"]
        line_protocol = record.to_line_protocol()

        assert "gpu_id=0" in line_protocol
        assert "gpu_name=CUDA:0" in line_protocol
        assert "gpu_mem_allocated_mb=100" in line_protocol
        assert "gpu_mem_cached_mb=200" in line_protocol
        assert "gpu_mem_total_mb=150" in line_protocol
        assert "gpu_mem_segments=10000" in line_protocol

    def test_log_with_list_fields(self, metrics_logger, mock_influxdb_client):
        """Test logging with list fields that should be processed."""
        measurement = "test_with_lists"
        tags = {"tag1": "value1"}
        # Create a list of values to be processed into statistics
        list_field = [1.0, 2.0, 3.0, 4.0, 5.0]
        fields = {"list_field": list_field}

        metrics_logger.log(
            measurement=measurement,
            tags=tags,
            fields=fields,
            with_system_metrics=False,
            with_gpu_metrics=False,
        )

        write_api = mock_influxdb_client.return_value.write_api.return_value
        write_api.write.assert_called_once()

        record = write_api.write.call_args[1]["record"]
        line_protocol = record.to_line_protocol()

        assert "list_field_mean=3" in line_protocol
        assert "list_field_min=1" in line_protocol
        assert "list_field_max=5" in line_protocol
        assert "list_field_median=3" in line_protocol

    def test_log_with_config_tags(
        self, metrics_logger, mock_influxdb_client, bt_config
    ):
        """Test logging with BT_Config object to add config tags."""
        metrics_logger.config = bt_config

        measurement = "test_with_config"
        tags = {"tag1": "value1"}
        fields = {"field1": 1.0}

        metrics_logger.log(
            measurement=measurement,
            tags=tags,
            fields=fields,
            with_system_metrics=False,
            with_gpu_metrics=False,
        )

        write_api = mock_influxdb_client.return_value.write_api.return_value
        write_api.write.assert_called_once()

        record = write_api.write.call_args[1]["record"]
        line_protocol = record.to_line_protocol()

        assert f"config_netuid={bt_config.netuid}" in line_protocol
        assert f"config_neuron_name={bt_config.neuron_name}" in line_protocol

    def test_log_with_exception(self, metrics_logger, mock_influxdb_client):
        """Test logging when an exception occurs."""

        write_api = mock_influxdb_client.return_value.write_api.return_value
        write_api.write.side_effect = Exception("Test exception")

        metrics_logger.log(
            measurement="test_exception",
            tags={},
            fields={"field1": 1.0},
            with_system_metrics=False,
            with_gpu_metrics=False,
        )

        write_api.write.assert_called_once()

    def test_miner_metrics_pattern(self, metrics_logger, mock_influxdb_client):
        """Test that the metrics logger handles the pattern used by the miner."""
        # This replicates the metrics logging pattern used in miner.py
        measurement = "training_step"
        tags = {
            "window": 42,
            "global_step": 100,
        }
        fields = {
            "loss": 0.75,
            "tokens_per_sec": 1234.56,
            "batch_tokens": 25,
            "grad_norm_std": 0.1,
            "mean_weight_norm": 0.5,
            "mean_momentum_norm": 0.02,
            "batch_duration": 10.5,
            "total_tokens": 50000,
            "active_peers": 8,
            "effective_batch_size": 256,
            "learning_rate": 0.001,
            "mean_grad_norm": 0.3,
            "gather_success_rate": 85.5,
            "max_grad_norm": 0.5,
            "min_grad_norm": 0.1,
            "gather_peers": json.dumps([1, 2, 3, 4, 5, 6, 7, 8]),
            "skipped_peers": json.dumps([9, 10]),
            "window_total_time": 25.5,
            "peer_update_time": 1.2,
            "data_loading_time": 2.5,
            "training_time": 10.5,
            "compression_time": 3.2,
            "gather_time": 5.6,
            "model_update_time": 2.5,
        }

        metrics_logger.log(
            measurement=measurement,
            tags=tags,
            fields=fields,
            with_system_metrics=True,
            with_gpu_metrics=True,
        )

        write_api = mock_influxdb_client.return_value.write_api.return_value
        write_api.write.assert_called_once()

        call_args = write_api.write.call_args
        assert call_args is not None
        assert call_args[1]["bucket"] == metrics_logger.database
        assert call_args[1]["org"] == metrics_logger.org
        assert isinstance(call_args[1]["record"], Point)
        assert call_args[1]["record"]._name == f"test{measurement}"

    def test_log_with_wait(self, metrics_logger, mock_influxdb_client):
        """Test logging with wait."""
        measurement = "test_with_wait"
        tags = {"tag1": "value1"}
        fields = {"field1": 42}

        metrics_logger.log(
            measurement=measurement,
            tags=tags,
            fields=fields,
            with_system_metrics=False,
            with_gpu_metrics=False,
        )

        # --------------------------------------------------------------
        # run_in_executor schedules _write_point on a background thread.
        # Wait (≤2 s) until that thread calls write_api.write, then check.
        # --------------------------------------------------------------
        def _wait_for_call(m, timeout=2.0):
            end = time.time() + timeout
            while time.time() < end and m.call_count == 0:
                time.sleep(0.01)
            assert m.call_count == 1, (
                f"Expected 'write' to be called once; got {m.call_count}"
            )

        mock_write = mock_influxdb_client.return_value.write_api.return_value.write
        _wait_for_call(mock_write)

    def test_log_invokes_write_once(self, logger):
        log, write_mock = logger

        log.log("train_step", tags={}, fields={"loss": 0.123})

        write_mock.assert_called_once()
