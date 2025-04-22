# test_metrics_logger.py

import unittest.mock as mock
import asyncio
import concurrent.futures
from unittest.mock import MagicMock, patch
import logging
import statistics

import pytest
from bittensor import Config as BT_Config
from influxdb_client.client.write.point import Point


from tplr.metrics import MetricsLogger

# Configure basic logging for debugging test issues
# Run pytest with -s --log-cli-level=DEBUG
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _sync_run_in_executor_simple(monkeypatch):
    """
    Ensure event loop exists.
    Simplified patch for run_in_executor to run func immediately.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        log.debug("No event loop found, creating a new one.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    else:
        log.debug(f"Using existing event loop: {loop}")

    original_run_in_executor = asyncio.BaseEventLoop.run_in_executor
    log.debug(f"Original run_in_executor: {original_run_in_executor}")

    def _run_sync_direct(self, executor, func, *args, **kwargs):
        log.debug(
            f"SYNC PATCH (Direct): Running {func.__name__} with args: {args}, kwargs: {kwargs}"
        )
        try:
            # Call the function directly, ignore the executor
            result = func(*args, **kwargs)
            log.debug(
                f"SYNC PATCH (Direct): {func.__name__} executed successfully, result type: {type(result)}"
            )
            # Return a completed future as expected by await loop.run_in_executor(...)
            fut = concurrent.futures.Future()
            fut.set_result(result)
            return fut
        except Exception as exc:
            log.exception(
                f"SYNC PATCH (Direct): Exception during execution of {func.__name__}"
            )  # Log the exception
            fut = concurrent.futures.Future()
            fut.set_exception(exc)
            return fut

    monkeypatch.setattr(asyncio.BaseEventLoop, "run_in_executor", _run_sync_direct)
    log.debug(
        f"Patched run_in_executor with _run_sync_direct: {asyncio.BaseEventLoop.run_in_executor}"
    )


@pytest.fixture(autouse=True)
def _dummy_influx_env(monkeypatch):
    """Provide dummy InfluxDB env vars."""
    env_vars = {
        "INFLUXDB_HOST": "dummyhost.invalid",
        "INFLUXDB_PORT": "8086",
        "INFLUXDB_TOKEN": "unit-test-token",
        "INFLUXDB_ORG": "unit-test-org",
        "INFLUXDB_DATABASE": "unit-test-bucket",
    }
    log.debug(f"Setting dummy env vars: {env_vars}")
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)


class TestMetricsLogger:
    @pytest.fixture
    def mock_influxdb_client(self):
        """Patch InfluxDBClient, configure mocks, and return the mock class."""
        # Target the correct location where InfluxDBClient is imported in tplr.metrics
        with patch("tplr.metrics.InfluxDBClient", autospec=True) as mock_client_class:
            log.debug(
                f"Patching tplr.metrics.InfluxDBClient [ID: {id(mock_client_class)}]"
            )

            mock_client_instance = mock_client_class.return_value
            log.debug(f" Mock Client Instance [ID: {id(mock_client_instance)}]")
            mock_client_instance.ping.return_value = True

            mock_write_api_instance = MagicMock(
                spec_set=["write"]
            )  # Be specific about write method
            log.debug(f" Mock Write API Instance [ID: {id(mock_write_api_instance)}]")
            mock_client_instance.write_api.return_value = mock_write_api_instance
            # Add name for easier debugging
            mock_write_api_instance.write._mock_name = "mock_write_api_instance.write"

            log.debug(
                f" mock_client_class.return_value.write_api.return_value.write [ID: {id(mock_client_class.return_value.write_api.return_value.write)}]"
            )

            yield mock_client_class  # Yield the mock *class*
            log.debug("Exiting mock_influxdb_client fixture, patch deactivated.")

    @pytest.fixture
    def metrics_logger(
        self, mock_influxdb_client
    ):  # Depends on the client mock fixture
        """Create a MetricsLogger instance for testing, using the mocked client."""
        log.debug(
            f"Creating MetricsLogger instance... mock_influxdb_client [ID: {id(mock_influxdb_client)}]"
        )
        # Constructor uses the mocked InfluxDBClient because the patch from mock_influxdb_client fixture is active
        logger = MetricsLogger(
            prefix="test",
            uid="123",
            role="miner",
            group="test_group",
            job_type="training",
        )
        log.debug(f"MetricsLogger instance created [ID: {id(logger)}]")
        log.debug(
            f" logger.client [ID: {id(logger.client)}], Expected mock instance [ID: {id(mock_influxdb_client.return_value)}]"
        )
        log.debug(
            f" logger.write_api [ID: {id(logger.write_api)}], Expected mock write_api [ID: {id(mock_influxdb_client.return_value.write_api.return_value)}]"
        )
        log.debug(
            f" logger.write_api.write [ID: {id(logger.write_api.write)}], Expected mock write method [ID: {id(mock_influxdb_client.return_value.write_api.return_value.write)}]"
        )

        assert logger.client is mock_influxdb_client.return_value, (
            "Logger client is not the expected mock instance"
        )
        assert (
            logger.write_api is mock_influxdb_client.return_value.write_api.return_value
        ), "Logger write_api is not the expected mock instance"
        assert hasattr(logger.write_api, "write"), (
            "Mock write_api instance is missing the 'write' method"
        )
        assert isinstance(logger.write_api.write, MagicMock), (
            "logger.write_api.write is not a mock!"
        )

        return logger

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
        # (Keep the version from the previous response)
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
        """Create a bittensor Config object for testing."""
        config = BT_Config()
        config.netuid = 268
        config.neuron_name = "test_miner"
        config.logging = MagicMock()
        config.logging.debug = False
        config.logging.trace = False
        return config

    # --- Test Methods ---

    def test_init(self, mock_influxdb_client):
        """Test MetricsLogger initialization calls InfluxDBClient."""
        log.debug("Running test_init...")
        # Get dummy values (could read from os.environ but safer to hardcode them here
        # as they are defined in the _dummy_influx_env fixture)
        dummy_host = "dummyhost.invalid"
        dummy_port = 8086
        test_token = "init-test-token"
        test_org = "unit-test-org"  # Defined in dummy env

        # Explicitly pass host and port to override defaults captured at import time
        MetricsLogger(
            token=test_token,
            host=dummy_host,
            port=dummy_port,
            org=test_org,  # Also pass org if needed for the check
        )

        # Assert that the mock InfluxDBClient class was called
        mock_influxdb_client.assert_called_once()

        # Check specific args used for initialization
        _, call_kwargs = mock_influxdb_client.call_args

        # Now the expected URL should match the dummy host/port
        expected_url = f"https://{dummy_host}:{dummy_port}"
        expected_org = test_org  # From dummy env / passed arg

        assert call_kwargs.get("url") == expected_url
        assert call_kwargs.get("token") == test_token
        assert call_kwargs.get("org") == expected_org

        log.debug("test_init completed successfully.")

    def test_process_value(self, metrics_logger):
        """Test processing of different value types."""
        log.debug("Running test_process_value...")
        assert metrics_logger.process_value(42) == 42
        assert metrics_logger.process_value(3.14) == 3.14

        peer_ids = [1, 2, 3, 4, 5]
        assert metrics_logger.process_value(peer_ids) == str(peer_ids)

        values = [1.1, 2.2, 3.3, 4.4, 5.5]
        expected_stats = {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
        }
        result = metrics_logger.process_value(values)
        # FIX: Use pytest.approx for dictionary comparison involving floats
        assert result == pytest.approx(expected_stats)

        # Check implementation detail for empty list
        assert metrics_logger.process_value([]) == 0.0

        assert metrics_logger.process_value("test") == "test"
        assert metrics_logger.process_value(None) == 0.0
        log.debug("test_process_value completed successfully.")

    # --- Helper to get the specific mock method to check ---
    def get_write_method_mock(self, logger_instance):
        """Gets the mock for the 'write' method on the logger's write_api."""
        # Defensive checks
        assert hasattr(logger_instance, "write_api"), (
            "Logger instance has no 'write_api' attribute"
        )
        assert hasattr(logger_instance.write_api, "write"), (
            "Logger's write_api has no 'write' method"
        )
        write_method_mock = logger_instance.write_api.write
        assert isinstance(write_method_mock, MagicMock), (
            f"Expected write_api.write to be a mock, but got {type(write_method_mock)}"
        )
        log.debug(
            f"Retrieved write method mock [ID: {id(write_method_mock)}], Name: {getattr(write_method_mock, '_mock_name', 'N/A')}"
        )
        return write_method_mock

    # --- Test log methods ---
    # All log tests use the metrics_logger fixture

    def test_log_basic(self, metrics_logger):  # No need for mock_influxdb_client here
        """Test basic logging functionality."""
        log.debug("Running test_log_basic...")
        measurement = "test_measurement"
        tags = {"tag1": "value1", "tag2": "value2"}
        fields = {"field1": 1.0, "field2": 2}

        # Call the method under test
        metrics_logger.log(measurement=measurement, tags=tags, fields=fields)

        # Get the specific mock method and assert
        write_method_mock = self.get_write_method_mock(metrics_logger)
        try:
            write_method_mock.assert_called_once()
            log.info("✅ test_log_basic: write_method_mock.assert_called_once() PASSED")
        except AssertionError as e:
            log.error(
                f"❌ test_log_basic: write_method_mock.assert_called_once() FAILED. Calls: {write_method_mock.call_args_list}"
            )
            raise e

        # Check arguments (only if called)
        call_args = write_method_mock.call_args
        assert call_args is not None
        assert call_args.kwargs["bucket"] == metrics_logger.database
        assert call_args.kwargs["org"] == metrics_logger.org
        record = call_args.kwargs["record"]
        assert isinstance(record, Point)
        assert record._name == f"test{measurement}"
        # ... other checks ...
        log.debug("test_log_basic completed.")

    def test_log_with_system_metrics(self, metrics_logger, mock_system_metrics):
        log.debug("Running test_log_with_system_metrics...")
        metrics_logger.log("sys_test", {}, {}, with_system_metrics=True)
        write_method_mock = self.get_write_method_mock(metrics_logger)
        try:
            write_method_mock.assert_called_once()
            log.info(
                "✅ test_log_with_system_metrics: write_method_mock.assert_called_once() PASSED"
            )
        except AssertionError as e:
            log.error(
                f"❌ test_log_with_system_metrics: FAILED. Calls: {write_method_mock.call_args_list}"
            )
            raise e
        # ... other checks ...
        mock_system_metrics["cpu_percent"].assert_called_once()
        mock_system_metrics["virtual_memory"].assert_called_once()
        log.debug("test_log_with_system_metrics completed.")

    def test_log_with_gpu_metrics(self, metrics_logger, mock_cuda_functions):
        log.debug("Running test_log_with_gpu_metrics...")
        metrics_logger.log("gpu_test", {}, {}, with_gpu_metrics=True)
        write_method_mock = self.get_write_method_mock(metrics_logger)
        try:
            write_method_mock.assert_called_once()
            log.info(
                "✅ test_log_with_gpu_metrics: write_method_mock.assert_called_once() PASSED"
            )
        except AssertionError as e:
            log.error(
                f"❌ test_log_with_gpu_metrics: FAILED. Calls: {write_method_mock.call_args_list}"
            )
            raise e

        # Check that the *actually called* CUDA mocks were called
        if hasattr(
            mock_cuda_functions["current_device"], "assert_called"
        ):  # Check if it's a real mock
            # Keep assertions for functions that ARE called by get_gpu_metrics:
            mock_cuda_functions["current_device"].assert_called()
            mock_cuda_functions["memory_stats"].assert_called_once()
            mock_cuda_functions["memory_allocated"].assert_called_once()
            mock_cuda_functions["memory_reserved"].assert_called_once()
            mock_cuda_functions["max_memory_allocated"].assert_called_once()

        log.debug("test_log_with_gpu_metrics completed.")

    def test_log_with_list_fields(self, metrics_logger):
        log.debug("Running test_log_with_list_fields...")
        metrics_logger.log(
            "list_test", {}, {"numeric": [1.0, 2.0], "peers": [1, 2], "empty": []}
        )
        write_method_mock = self.get_write_method_mock(metrics_logger)
        try:
            write_method_mock.assert_called_once()
            log.info(
                "✅ test_log_with_list_fields: write_method_mock.assert_called_once() PASSED"
            )
        except AssertionError as e:
            log.error(
                f"❌ test_log_with_list_fields: FAILED. Calls: {write_method_mock.call_args_list}"
            )
            raise e
        # ... other checks ...
        log.debug("test_log_with_list_fields completed.")

    def test_log_with_config_tags(self, metrics_logger, bt_config):
        log.debug("Running test_log_with_config_tags...")
        metrics_logger.config = bt_config
        metrics_logger.log("config_test", {}, {})
        write_method_mock = self.get_write_method_mock(metrics_logger)
        try:
            write_method_mock.assert_called_once()
            log.info(
                "✅ test_log_with_config_tags: write_method_mock.assert_called_once() PASSED"
            )
        except AssertionError as e:
            log.error(
                f"❌ test_log_with_config_tags: FAILED. Calls: {write_method_mock.call_args_list}"
            )
            raise e
        # ... other checks ...
        log.debug("test_log_with_config_tags completed.")

    def test_log_with_exception(self, metrics_logger):
        log.debug("Running test_log_with_exception...")
        write_method_mock = self.get_write_method_mock(metrics_logger)
        write_method_mock.side_effect = Exception("InfluxDB write failed")

        metrics_logger.log("exception_test", {}, {})

        try:
            write_method_mock.assert_called_once()  # Should still be called once
            log.info(
                "✅ test_log_with_exception: write_method_mock.assert_called_once() PASSED"
            )
        except AssertionError as e:
            log.error(
                f"❌ test_log_with_exception: FAILED. Calls: {write_method_mock.call_args_list}"
            )
            raise e
        # Note: The sync patch will catch the exception and log it. The test won't see it directly.
        log.debug("test_log_with_exception completed.")

    def test_miner_metrics_pattern(
        self, metrics_logger, mock_system_metrics, mock_cuda_functions
    ):
        log.debug("Running test_miner_metrics_pattern...")
        # Simplified fields for brevity
        measurement = "training_step"
        tags = {"window": 42}
        fields = {"loss": 0.75, "active_peers": 8}
        metrics_logger.log(
            measurement, tags, fields, with_system_metrics=True, with_gpu_metrics=True
        )

        write_method_mock = self.get_write_method_mock(metrics_logger)
        try:
            write_method_mock.assert_called_once()
            log.info(
                "✅ test_miner_metrics_pattern: write_method_mock.assert_called_once() PASSED"
            )
        except AssertionError as e:
            log.error(
                f"❌ test_miner_metrics_pattern: FAILED. Calls: {write_method_mock.call_args_list}"
            )
            raise e
        # ... detailed checks ...
        log.debug("test_miner_metrics_pattern completed.")

    def test_log_call_invokes_write_once(self, metrics_logger):
        log.debug("Running test_log_call_invokes_write_once...")
        metrics_logger.log("single_call_test", {}, {})
        write_method_mock = self.get_write_method_mock(metrics_logger)
        try:
            write_method_mock.assert_called_once()
            log.info(
                "✅ test_log_call_invokes_write_once: write_method_mock.assert_called_once() PASSED"
            )
        except AssertionError as e:
            log.error(
                f"❌ test_log_call_invokes_write_once: FAILED. Calls: {write_method_mock.call_args_list}"
            )
            raise e
        log.debug("test_log_call_invokes_write_once completed.")
