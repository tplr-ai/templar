# test_metrics_logger.py

import asyncio
import logging
import time  # Need time for sleep
import unittest.mock as mock
from unittest.mock import MagicMock, Mock, patch  # Import Mock

import pytest
from bittensor import Config as BT_Config

# Assuming tplr is importable
from tplr.metrics import MetricsLogger

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)


# --- Fixture to ENSURE an event loop exists ---
@pytest.fixture(autouse=True)
def ensure_event_loop():
    """
    Ensure the main thread has an event loop for MetricsLogger.
    Needed because MetricsLogger.log() calls asyncio.get_event_loop().
    """
    log.debug("Ensuring event loop exists for the current thread...")
    try:
        loop = asyncio.get_event_loop()
        log.debug(f"Event loop already exists: {loop}")
    except RuntimeError:
        log.info("No event loop found, creating and setting a new one.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        log.debug(f"Set new event loop: {loop}")


# --- Helper to wait for mock call ---
def wait_for_mock_call(mock_object: Mock, timeout: float = 3.0):
    """Waits for a mock object to be called at least once."""
    mock_name = getattr(mock_object, "_mock_name", repr(mock_object))
    log.debug(f"Waiting up to {timeout}s for mock '{mock_name}' to be called...")
    start_time = time.monotonic()
    while time.monotonic() < start_time + timeout:
        if mock_object.call_count > 0:
            log.debug(
                f"Mock '{mock_name}' called after {time.monotonic() - start_time:.3f}s. Count: {mock_object.call_count}"
            )
            return True
        time.sleep(0.05)
    log.warning(
        f"Timeout ({timeout}s) waiting for mock '{mock_name}'. Call count remained {mock_object.call_count}."
    )
    return False


class TestMetricsLogger:
    @pytest.fixture
    def mock_influxdb_client(self):
        """Patch InfluxDBClient, configure mocks, and return the mock class."""
        # Target the location where InfluxDBClient is imported in the metrics module
        with patch("tplr.metrics.InfluxDBClient", autospec=True) as mock_client_class:
            log.debug(
                f"Patching tplr.metrics.InfluxDBClient [ID: {id(mock_client_class)}]"
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.ping.return_value = (
                True  # Mock ping if called during init
            )
            mock_write_api_instance = MagicMock(
                spec_set=["write"]
            )  # Use spec_set if methods are known
            mock_write_api_instance.write._mock_name = (
                "mock_write_api_instance.write"  # For logs
            )
            mock_write_api_instance.write.side_effect = None

            log.debug(
                f" Configured Mock Write API Instance [ID: {id(mock_write_api_instance)}], Write Method [ID: {id(mock_write_api_instance.write)}]"
            )
            mock_client_instance.write_api.return_value = mock_write_api_instance
            yield mock_client_class

            mock_write_api_instance.write.side_effect = None
            mock_write_api_instance.write.reset_mock()

            log.debug("Exiting mock_influxdb_client fixture, patch deactivated.")

    @pytest.fixture
    def metrics_logger(
        self, mock_influxdb_client
    ):  # Depends on the mock client fixture
        """Create a MetricsLogger instance for testing, using the mocked client."""
        log.debug("Creating MetricsLogger instance...")
        # Explicitly pass connection details to avoid default/env var issues at import time
        logger = MetricsLogger(
            prefix="test",
            uid="123",
            role="miner",
            group="test_group",
            job_type="training",
            host="dummyhost.invalid",
            port=8086,
            token="unit-test-token",
            org="unit-test-org",
            database="unit-test-bucket",
        )
        log.debug(f"MetricsLogger instance created [ID: {id(logger)}]")
        # Verify internal mock references immediately
        assert (
            logger.client is mock_influxdb_client.return_value
        ), "Logger client mismatch"
        assert (
            logger.write_api is mock_influxdb_client.return_value.write_api.return_value
        ), "Logger write_api mismatch"
        assert isinstance(
            logger.write_api.write, MagicMock
        ), "Logger write_api.write is not a mock"
        return logger

    # --- Other fixtures (CUDA, System, BT_Config) - Keep as they were ---
    @pytest.fixture
    def mock_cuda_functions(self):
        try:
            with mock.patch.multiple(
                "tplr.metrics.torch.cuda",
                **{
                    "memory_stats": mock.DEFAULT,
                    "memory_allocated": mock.DEFAULT,
                    "memory_reserved": mock.DEFAULT,
                    "max_memory_allocated": mock.DEFAULT,
                    "current_device": mock.DEFAULT,
                    "is_available": mock.DEFAULT,
                },
            ) as mocks:
                mocks["is_available"].return_value = True
                mocks["current_device"].return_value = 0
                mocks["memory_stats"].return_value = {"segment.all.current": 10000}
                mocks["memory_allocated"].return_value = 1024 * 1024 * 100
                mocks["memory_reserved"].return_value = 1024 * 1024 * 200
                mocks["max_memory_allocated"].return_value = 1024 * 1024 * 150
                yield mocks
        except ImportError:
            log.warning("torch not found, skipping CUDA mocks.")
            yield {
                name: MagicMock()
                for name in [
                    "memory_stats",
                    "memory_allocated",
                    "memory_reserved",
                    "max_memory_allocated",
                    "current_device",
                    "is_available",
                ]
            }

    @pytest.fixture
    def mock_system_metrics(self):
        try:
            with mock.patch.multiple("tplr.metrics", psutil=mock.DEFAULT) as mocks:
                mock_psutil = mocks["psutil"]
                mock_psutil.cpu_percent.return_value = 35.5
                mock_memory = MagicMock()
                mock_memory.used = 8 * 1024 * 1024 * 1024
                mock_memory.total = 16 * 1024 * 1024 * 1024
                mock_psutil.virtual_memory.return_value = mock_memory
                yield {
                    "cpu_percent": mock_psutil.cpu_percent,
                    "virtual_memory": mock_psutil.virtual_memory,
                }
        except ImportError:
            log.warning("psutil not found, skipping system mocks.")
            yield {name: MagicMock() for name in ["cpu_percent", "virtual_memory"]}

    @pytest.fixture
    def bt_config(self):
        config = BT_Config()
        config.netuid = 268
        config.neuron_name = "test_miner"
        config.logging = MagicMock()
        config.logging.debug = False
        config.logging.trace = False
        return config

    # --- Test Methods ---

    def get_write_method_mock(self, logger_instance):
        """Helper to reliably get the write method mock from the logger instance."""
        assert hasattr(logger_instance, "write_api"), "Logger missing 'write_api'"
        assert hasattr(
            logger_instance.write_api, "write"
        ), "Logger's write_api missing 'write'"
        write_mock = logger_instance.write_api.write
        assert isinstance(
            write_mock, MagicMock
        ), f"write_api.write is not a MagicMock, it's {type(write_mock)}"
        log.debug(
            f"Retrieved write mock [ID: {id(write_mock)}] Name: {getattr(write_mock, '_mock_name', 'N/A')}"
        )
        return write_mock

    # test_init and test_process_value remain the same
    def test_init(self, mock_influxdb_client):
        """Test MetricsLogger initialization calls InfluxDBClient with correct args."""
        log.debug("Running test_init...")
        dummy_host = "dummyhost.invalid"
        dummy_port = 8086
        test_token = "init-test-token"
        test_org = "unit-test-org"
        _ = MetricsLogger(
            token=test_token, host=dummy_host, port=dummy_port, org=test_org
        )
        mock_influxdb_client.assert_called_once()
        _, call_kwargs = mock_influxdb_client.call_args
        expected_url = f"https://{dummy_host}:{dummy_port}"
        expected_org = test_org
        assert call_kwargs.get("url") == expected_url
        assert call_kwargs.get("token") == test_token
        assert call_kwargs.get("org") == expected_org
        log.debug("test_init completed successfully.")

    def test_process_value(self, metrics_logger):
        """Test processing of different value types."""
        log.debug("Running test_process_value...")
        assert metrics_logger.process_value(42) == 42
        assert metrics_logger.process_value(3.14) == 3.14

        peer_ids = [1, 2, 3]
        assert metrics_logger.process_value(peer_ids) == str(peer_ids)

        values = [1.1, 2.2, 3.3]
        expected_stats = {"mean": 2.2, "min": 1.1, "max": 3.3, "median": 2.2}
        assert metrics_logger.process_value(values) == pytest.approx(expected_stats)

        assert metrics_logger.process_value([]) == 0.0
        assert metrics_logger.process_value("test") == "test"
        assert metrics_logger.process_value(None) == 0.0
        log.debug("test_process_value completed successfully.")

    # --- Logging Tests: Apply wait_for_mock_call ---

    def test_log_basic(self, metrics_logger):
        log.debug("Running test_log_basic...")
        metrics_logger.log(measurement="test_m", tags={"t": "v"}, fields={"f": 1})
        write_mock = self.get_write_method_mock(metrics_logger)
        assert wait_for_mock_call(
            write_mock
        ), f"'{write_mock._mock_name}' not called within timeout"
        write_mock.assert_called_once()
        # Add further checks on call_args if needed
        log.debug("test_log_basic completed.")

    def test_log_with_system_metrics(self, metrics_logger, mock_system_metrics):
        log.debug("Running test_log_with_system_metrics...")
        metrics_logger.log("sys_test", {}, {}, with_system_metrics=True)
        write_mock = self.get_write_method_mock(metrics_logger)
        assert wait_for_mock_call(
            write_mock
        ), f"'{write_mock._mock_name}' not called within timeout"
        write_mock.assert_called_once()
        # Check system mocks were called
        if hasattr(mock_system_metrics["cpu_percent"], "assert_called_once"):
            mock_system_metrics["cpu_percent"].assert_called_once()
            mock_system_metrics["virtual_memory"].assert_called_once()
        log.debug("test_log_with_system_metrics completed.")

    def test_log_with_gpu_metrics(self, metrics_logger, mock_cuda_functions):
        log.debug("Running test_log_with_gpu_metrics...")
        metrics_logger.log("gpu_test", {}, {}, with_gpu_metrics=True)
        write_mock = self.get_write_method_mock(metrics_logger)
        assert wait_for_mock_call(
            write_mock
        ), f"'{write_mock._mock_name}' not called within timeout"
        write_mock.assert_called_once()
        # Check CUDA mocks were called (ensure mock fixture provides actual mocks)
        if hasattr(mock_cuda_functions["current_device"], "assert_called"):
            mock_cuda_functions["current_device"].assert_called()
            mock_cuda_functions["memory_stats"].assert_called_once()
            mock_cuda_functions["memory_allocated"].assert_called_once()
            mock_cuda_functions["memory_reserved"].assert_called_once()
            mock_cuda_functions["max_memory_allocated"].assert_called_once()
        log.debug("test_log_with_gpu_metrics completed.")

    def test_log_with_list_fields(self, metrics_logger):
        log.debug("Running test_log_with_list_fields...")
        metrics_logger.log("list_test", {}, {"numeric": [1.0, 2.0], "peers": [1, 2]})
        write_mock = self.get_write_method_mock(metrics_logger)
        assert wait_for_mock_call(
            write_mock
        ), f"'{write_mock._mock_name}' not called within timeout"
        write_mock.assert_called_once()
        log.debug("test_log_with_list_fields completed.")

    def test_log_with_config_tags(self, metrics_logger, bt_config):
        log.debug("Running test_log_with_config_tags...")
        metrics_logger.config = bt_config
        metrics_logger.log("config_test", {}, {})
        write_mock = self.get_write_method_mock(metrics_logger)
        assert wait_for_mock_call(
            write_mock
        ), f"'{write_mock._mock_name}' not called within timeout"
        write_mock.assert_called_once()
        log.debug("test_log_with_config_tags completed.")

    @pytest.fixture
    def exception_mock_metrics_logger(self, metrics_logger):
        """Create a separate metrics logger instance for the exception test."""
        write_mock = self.get_write_method_mock(metrics_logger)
        write_mock.reset_mock()
        test_exception = Exception("InfluxDB write failed in test")
        write_mock.side_effect = test_exception

        yield metrics_logger

        write_mock.side_effect = None
        write_mock.reset_mock()

    def test_log_with_exception(self, exception_mock_metrics_logger):
        log.debug("Running test_log_with_exception...")

        metrics_logger = exception_mock_metrics_logger
        write_mock = self.get_write_method_mock(metrics_logger)

        # The logger internally catches the exception from write_api.write
        # but the call to run_in_executor should still succeed.
        metrics_logger.log("exception_test", {}, {})
        # We wait for the _write_point function to be called, which then calls the mock
        assert wait_for_mock_call(
            write_mock
        ), f"'{write_mock._mock_name}' not called within timeout (even with exception)"
        write_mock.assert_called_once()  # Check it was attempted

        log.debug("test_log_with_exception completed.")

    def test_miner_metrics_pattern(
        self, metrics_logger, mock_system_metrics, mock_cuda_functions
    ):
        log.debug("Running test_miner_metrics_pattern...")
        metrics_logger.log(
            "training_step",
            {"w": 42},
            {"l": 0.75},
            with_system_metrics=True,
            with_gpu_metrics=True,
        )
        write_mock = self.get_write_method_mock(metrics_logger)
        assert wait_for_mock_call(
            write_mock
        ), f"'{write_mock._mock_name}' not called within timeout"
        write_mock.assert_called_once()
        log.debug("test_miner_metrics_pattern completed.")

    @pytest.fixture
    def clean_mock_metrics_logger(self, metrics_logger):
        """Create a clean metrics logger instance with reset mocks."""
        write_mock = self.get_write_method_mock(metrics_logger)

        write_mock.reset_mock()
        write_mock.side_effect = None

        yield metrics_logger

    def test_log_call_invokes_write_once(self, clean_mock_metrics_logger):
        log.debug("Running test_log_call_invokes_write_once...")

        metrics_logger = clean_mock_metrics_logger
        write_mock = self.get_write_method_mock(metrics_logger)

        metrics_logger.log("single_call_test", {}, {})
        assert wait_for_mock_call(
            write_mock
        ), f"'{write_mock._mock_name}' not called within timeout"
        write_mock.assert_called_once()
        log.debug("test_log_call_invokes_write_once completed.")

    def test_log_with_sample_rate(self, metrics_logger):
        """Test that sample_rate parameter controls logging frequency."""
        log.debug("Running test_log_with_sample_rate...")

        write_mock = self.get_write_method_mock(metrics_logger)

        write_mock.side_effect = None
        write_mock.reset_mock()

        with patch("random.random", return_value=0.5):
            write_mock.reset_mock()

            # Call log with sample_rate=0.8 (should log because 0.5 < 0.8)
            metrics_logger.log(
                measurement="test_sample",
                tags={"test": "value"},
                fields={"field": 1.0},
                sample_rate=0.8,
            )

            assert wait_for_mock_call(
                write_mock
            ), f"'{write_mock._mock_name}' not called within timeout"
            write_mock.assert_called_once()

        with patch("random.random", return_value=0.9):
            write_mock.reset_mock()

            metrics_logger.log(
                measurement="test_sample",
                tags={"test": "value"},
                fields={"field": 1.0},
                sample_rate=0.8,
            )

            time.sleep(0.1)
            write_mock.assert_not_called()

        log.debug("test_log_with_sample_rate completed.")
