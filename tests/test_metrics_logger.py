# test_metrics_logger.py

import json
import unittest.mock as mock
import asyncio
import concurrent.futures
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import os
import logging
import statistics
import sys # For checking platform
import inspect

import pytest
from bittensor import Config as BT_Config
from influxdb_client.client.write.point import Point
import tplr
from tplr.metrics import MetricsLogger, DEFAULT_DATABASE, DEFAULT_ORG

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
log = logging.getLogger(__name__)

# --- Robust Sync Patch with CI Debugging ---
@pytest.fixture(autouse=True)
def _sync_run_in_executor_robust(monkeypatch):
    """
    Ensure event loop exists.
    Robust patch for run_in_executor to run func immediately, with CI debugging.
    """
    log.info(f"[CI_DEBUG] Applying sync patch. Platform: {sys.platform}")
    loop_instance = None
    try:
        # Try to get the current running loop if one exists
        loop_instance = asyncio.get_running_loop()
        log.debug(f"[CI_DEBUG] Found running event loop via get_running_loop(): {loop_instance} (type: {type(loop_instance)})")
    except RuntimeError:
        # If no running loop, try get_event_loop (might create one depending on policy)
        try:
             loop_instance = asyncio.get_event_loop()
             log.debug(f"[CI_DEBUG] Found event loop via get_event_loop(): {loop_instance} (type: {type(loop_instance)})")
        except RuntimeError:
            # If get_event_loop also fails (shouldn't with default policy after Python 3.7?), create new.
            log.warning("[CI_DEBUG] No event loop found via get_running_loop/get_event_loop, creating a new one.")
            loop_instance = asyncio.new_event_loop()
            asyncio.set_event_loop(loop_instance)
            log.debug(f"[CI_DEBUG] Set new event loop: {loop_instance} (type: {type(loop_instance)})")

    # Ensure a loop object was obtained
    if loop_instance is None:
         pytest.fail("[CI_DEBUG] Failed to obtain an event loop instance.")

    # Decide what to patch: Base class is generally preferred.
    target_loop_class = asyncio.BaseEventLoop
    original_run_in_executor = getattr(target_loop_class, "run_in_executor", None)

    if original_run_in_executor is None:
         log.error(f"[CI_DEBUG] Critical: Could not find {target_loop_class.__name__}.run_in_executor to patch!")
         pytest.fail(f"Failed to find {target_loop_class.__name__}.run_in_executor method to patch.")

    log.debug(f"[CI_DEBUG] Original {target_loop_class.__name__}.run_in_executor [ID: {id(original_run_in_executor)}]: {original_run_in_executor}")

    # Use a flag to see if our patch function actually runs during a test
    patch_was_entered_flag = {"entered": False}

    def _run_sync_patched_for_ci(self, executor, func, *args, **kwargs):
        # 'self' is the event loop instance calling this method
        patch_was_entered_flag["entered"] = True # Mark entry
        log.info(f"[CI_DEBUG] ---> SYNC PATCH ENTERED by loop {id(self)} (type: {type(self)}) <---")
        log.info(f"[CI_DEBUG]        Executor: {executor}")
        log.info(f"[CI_DEBUG]        Function: {getattr(func, '__name__', repr(func))}")
        log.info(f"[CI_DEBUG]        Args: {args}")
        log.info(f"[CI_DEBUG]        Kwargs: {kwargs}")

        fut = concurrent.futures.Future()
        try:
            # --- Execute the function directly ---
            result = func(*args, **kwargs)
            # --- ----------------------------- ---
            log.info(f"[CI_DEBUG]        Function {getattr(func, '__name__', repr(func))} executed. Result type: {type(result)}")
            fut.set_result(result)
        except Exception as exc:
            log.exception(f"[CI_DEBUG]      ! EXCEPTION during sync execution of {getattr(func, '__name__', repr(func))}")
            fut.set_exception(exc)

        log.info(f"[CI_DEBUG] ---> SYNC PATCH EXITING. Returning future: {fut} (State: {fut._state}) <---")
        return fut

    # Apply the patch
    monkeypatch.setattr(
        target_loop_class,
        "run_in_executor",
        _run_sync_patched_for_ci
    )

    # Verify patch application
    patched_method = getattr(target_loop_class, "run_in_executor", None)
    log.debug(f"[CI_DEBUG] Patched {target_loop_class.__name__}.run_in_executor [ID: {id(patched_method)}]: {patched_method}")
    if not hasattr(patched_method, '__code__') or patched_method.__code__ != _run_sync_patched_for_ci.__code__:
        log.error("[CI_DEBUG] !!! CRITICAL: Post-patch check failed! run_in_executor does NOT seem to be the patched function.")
        # Optionally fail fast in CI if the patch seems broken
        # pytest.fail("Patch verification failed.")
    else:
        log.info("[CI_DEBUG] Post-patch check successful: run_in_executor appears to be the patched function.")

    yield # Test runs here

    log.info(f"[CI_DEBUG] Teardown: Patch function was entered during test? {patch_was_entered_flag['entered']}")
    # Monkeypatch handles restoration
    log.debug(f"[CI_DEBUG] Teardown: Restoring {target_loop_class.__name__}.run_in_executor.")


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
        with patch("tplr.metrics.InfluxDBClient", autospec=True) as mock_client_class:
            log.debug(f"Patching tplr.metrics.InfluxDBClient [ID: {id(mock_client_class)}]")
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.ping.return_value = True
            mock_write_api_instance = MagicMock(spec_set=["write"])
            # Give the mock write method a clear name for logs/errors
            mock_write_api_instance.write._mock_name = "mock_write_api_instance.write"
            log.debug(f" Configured Mock Write API Instance [ID: {id(mock_write_api_instance)}], Write Method [ID: {id(mock_write_api_instance.write)}]")
            mock_client_instance.write_api.return_value = mock_write_api_instance
            yield mock_client_class
            log.debug(f"Exiting mock_influxdb_client fixture, patch deactivated.")


    @pytest.fixture
    def metrics_logger(self, mock_influxdb_client):
        """Create a MetricsLogger instance for testing, using the mocked client."""
        log.debug(f"Creating MetricsLogger instance...")
        # Instantiate using dummy values matching _dummy_influx_env to ensure consistency
        logger = MetricsLogger(
            prefix="test",
            uid="123",
            role="miner",
            group="test_group",
            job_type="training",
            host="dummyhost.invalid", # Explicitly pass to avoid default capture issue
            port=8086,              # Explicitly pass
            token="unit-test-token",  # Explicitly pass
            org="unit-test-org",      # Explicitly pass
            database="unit-test-bucket" # Explicitly pass
        )
        log.debug(f"MetricsLogger instance created [ID: {id(logger)}]")
        # Verify internal mocks are correct IMMEDIATELY after creation
        log.debug(f" logger.client is mock_client_instance? {logger.client is mock_influxdb_client.return_value}")
        log.debug(f" logger.write_api is mock_write_api_instance? {logger.write_api is mock_influxdb_client.return_value.write_api.return_value}")
        log.debug(f" logger.write_api.write is mock_write_api_instance.write? {logger.write_api.write is mock_influxdb_client.return_value.write_api.return_value.write}")
        assert logger.client is mock_influxdb_client.return_value
        assert logger.write_api is mock_influxdb_client.return_value.write_api.return_value
        assert isinstance(logger.write_api.write, MagicMock)
        return logger

    # --- Other fixtures (CUDA, System, BT_Config) remain the same ---
    @pytest.fixture
    def mock_cuda_functions(self):
        try:
            import torch
            # Use patch.multiple for cleaner setup
            with mock.patch.multiple("tplr.metrics.torch.cuda", **{
                "memory_stats": mock.DEFAULT, "memory_allocated": mock.DEFAULT,
                "memory_reserved": mock.DEFAULT, "max_memory_allocated": mock.DEFAULT,
                "current_device": mock.DEFAULT, "is_available": mock.DEFAULT, # Keep is_available mockable if needed elsewhere
            }) as mocks:
                # Configure mocks
                mocks["is_available"].return_value = True # Assume available for tests needing GPU path
                mocks["current_device"].return_value = 0
                mocks["memory_stats"].return_value = {"segment.all.current": 10000}
                mocks["memory_allocated"].return_value = 1024 * 1024 * 100
                mocks["memory_reserved"].return_value = 1024 * 1024 * 200
                mocks["max_memory_allocated"].return_value = 1024 * 1024 * 150
                yield mocks
        except ImportError:
            log.warning("torch not found, skipping CUDA mocks.")
            # Provide dummy mocks if torch isn't present
            yield { name: MagicMock() for name in ["memory_stats", "memory_allocated", "memory_reserved", "max_memory_allocated", "current_device", "is_available"] }


    @pytest.fixture
    def mock_system_metrics(self):
        try:
            import psutil
            # Patch psutil functions used in get_system_metrics
            with mock.patch.multiple('tplr.metrics', psutil=mock.DEFAULT) as mocks:
                mock_psutil = mocks['psutil']
                # Configure mocks
                mock_psutil.cpu_percent.return_value = 35.5
                mock_memory = MagicMock()
                mock_memory.used = 8 * 1024 * 1024 * 1024 # 8 GB in bytes
                mock_memory.total = 16 * 1024 * 1024 * 1024 # 16 GB in bytes
                mock_psutil.virtual_memory.return_value = mock_memory
                # Yield the individual mocks for checking calls
                yield {
                    "cpu_percent": mock_psutil.cpu_percent,
                    "virtual_memory": mock_psutil.virtual_memory,
                }
        except ImportError:
             log.warning("psutil not found, skipping system mocks.")
             # Provide dummy mocks if psutil isn't present
             yield { name: MagicMock() for name in ["cpu_percent", "virtual_memory"] }


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
        assert hasattr(logger_instance, 'write_api'), "Logger missing 'write_api'"
        assert hasattr(logger_instance.write_api, 'write'), "Logger's write_api missing 'write'"
        write_mock = logger_instance.write_api.write
        # Check it's the mock we expect
        assert isinstance(write_mock, MagicMock), f"write_api.write is not a MagicMock, it's {type(write_mock)}"
        log.debug(f"Retrieved write mock [ID: {id(write_mock)}] Name: {getattr(write_mock, '_mock_name', 'N/A')}")
        return write_mock

    # test_init and test_process_value remain the same as the previous working version

    def test_init(self, mock_influxdb_client):
        """Test MetricsLogger initialization calls InfluxDBClient with correct args."""
        log.debug("Running test_init...")
        dummy_host = "dummyhost.invalid"
        dummy_port = 8086
        test_token = "init-test-token"
        test_org = "unit-test-org"

        # Explicitly pass args to override defaults/env vars during test
        logger_instance = MetricsLogger(
            token=test_token, host=dummy_host, port=dummy_port, org=test_org
        )
        mock_influxdb_client.assert_called_once()
        call_args, call_kwargs = mock_influxdb_client.call_args
        expected_url = f"https://{dummy_host}:{dummy_port}"
        assert call_kwargs.get('url') == expected_url
        assert call_kwargs.get('token') == test_token
        assert call_kwargs.get('org') == test_org
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
            'mean': statistics.mean(values), 'min': min(values),
            'max': max(values), 'median': statistics.median(values)
            }
        assert metrics_logger.process_value(values) == pytest.approx(expected_stats)
        assert metrics_logger.process_value([]) == 0.0
        assert metrics_logger.process_value("test") == "test"
        assert metrics_logger.process_value(None) == 0.0
        log.debug("test_process_value completed successfully.")

    # --- Logging Tests ---
    # Use the same structure for all log tests

    def test_log_basic(self, metrics_logger):
        """Test basic logging functionality."""
        log.debug("Running test_log_basic...")
        measurement = "test_measurement"
        tags = {"tag1": "value1"}
        fields = {"field1": 1.0}

        metrics_logger.log(measurement=measurement, tags=tags, fields=fields)

        write_method_mock = self.get_write_method_mock(metrics_logger)
        try:
            # The core assertion: Was write called?
            write_method_mock.assert_called_once()
            log.info(f"✅ {inspect.currentframe().f_code.co_name}: Write mock called.")
        except AssertionError as e:
            log.error(f"❌ {inspect.currentframe().f_code.co_name}: Write mock NOT called. Details: {e}")
            log.error(f"   Mock call count: {write_method_mock.call_count}")
            log.error(f"   Mock call args list: {write_method_mock.call_args_list}")
            raise # Re-raise the assertion error
        # Other checks (bucket, org, record content) can follow if needed
        call_args = write_method_mock.call_args
        assert call_args.kwargs["bucket"] == metrics_logger.database
        assert call_args.kwargs["org"] == metrics_logger.org
        # ... etc ...
        log.debug("test_log_basic completed.")

    def test_log_with_system_metrics(self, metrics_logger, mock_system_metrics):
        log.debug("Running test_log_with_system_metrics...")
        metrics_logger.log("sys_test", {}, {}, with_system_metrics=True)
        write_method_mock = self.get_write_method_mock(metrics_logger)
        try:
            write_method_mock.assert_called_once()
            log.info(f"✅ {inspect.currentframe().f_code.co_name}: Write mock called.")
        except AssertionError as e:
            log.error(f"❌ {inspect.currentframe().f_code.co_name}: Write mock NOT called. Details: {e}")
            raise
        # Check system mocks were called
        if hasattr(mock_system_metrics["cpu_percent"], 'assert_called_once'): # Check if real mock
             mock_system_metrics["cpu_percent"].assert_called_once()
             mock_system_metrics["virtual_memory"].assert_called_once()
        log.debug("test_log_with_system_metrics completed.")

    def test_log_with_gpu_metrics(self, metrics_logger, mock_cuda_functions):
        log.debug("Running test_log_with_gpu_metrics...")
        metrics_logger.log("gpu_test", {}, {}, with_gpu_metrics=True)
        write_method_mock = self.get_write_method_mock(metrics_logger)
        try:
            write_method_mock.assert_called_once()
            log.info(f"✅ {inspect.currentframe().f_code.co_name}: Write mock called.")
        except AssertionError as e:
            log.error(f"❌ {inspect.currentframe().f_code.co_name}: Write mock NOT called. Details: {e}")
            raise
        # Check CUDA mocks were called (excluding is_available)
        if hasattr(mock_cuda_functions["current_device"], 'assert_called'):
             mock_cuda_functions["current_device"].assert_called()
             mock_cuda_functions["memory_stats"].assert_called_once()
             mock_cuda_functions["memory_allocated"].assert_called_once()
             mock_cuda_functions["memory_reserved"].assert_called_once()
             mock_cuda_functions["max_memory_allocated"].assert_called_once()
        log.debug("test_log_with_gpu_metrics completed.")

    # ... apply the same try/except assertion pattern to other log tests ...
    # (test_log_with_list_fields, test_log_with_config_tags, test_log_with_exception,
    # test_miner_metrics_pattern, test_log_call_invokes_write_once)

    def test_log_with_list_fields(self, metrics_logger):
        log.debug("Running test_log_with_list_fields...")
        metrics_logger.log("list_test", {}, {"numeric": [1.0, 2.0], "peers": [1,2]})
        write_method_mock = self.get_write_method_mock(metrics_logger)
        try:
            write_method_mock.assert_called_once()
            log.info(f"✅ {inspect.currentframe().f_code.co_name}: Write mock called.")
        except AssertionError as e:
            log.error(f"❌ {inspect.currentframe().f_code.co_name}: Write mock NOT called. Details: {e}")
            raise
        log.debug("test_log_with_list_fields completed.")

    def test_log_with_config_tags(self, metrics_logger, bt_config):
        log.debug("Running test_log_with_config_tags...")
        metrics_logger.config = bt_config
        metrics_logger.log("config_test", {}, {})
        write_method_mock = self.get_write_method_mock(metrics_logger)
        try:
            write_method_mock.assert_called_once()
            log.info(f"✅ {inspect.currentframe().f_code.co_name}: Write mock called.")
        except AssertionError as e:
            log.error(f"❌ {inspect.currentframe().f_code.co_name}: Write mock NOT called. Details: {e}")
            raise
        log.debug("test_log_with_config_tags completed.")

    def test_log_with_exception(self, metrics_logger):
        log.debug("Running test_log_with_exception...")
        write_method_mock = self.get_write_method_mock(metrics_logger)
        write_method_mock.side_effect = Exception("InfluxDB write failed in test")

        metrics_logger.log("exception_test", {}, {})

        try:
            write_method_mock.assert_called_once() # Should still be called once
            log.info(f"✅ {inspect.currentframe().f_code.co_name}: Write mock called (and raised exception).")
        except AssertionError as e:
            log.error(f"❌ {inspect.currentframe().f_code.co_name}: Write mock NOT called. Details: {e}")
            raise
        log.debug("test_log_with_exception completed.")

    def test_miner_metrics_pattern(self, metrics_logger, mock_system_metrics, mock_cuda_functions):
        # Need to import inspect
        import inspect
        log.debug("Running test_miner_metrics_pattern...")
        measurement = "training_step"
        tags = {"window": 42}
        fields = {"loss": 0.75, "active_peers": 8}
        metrics_logger.log(measurement, tags, fields, with_system_metrics=True, with_gpu_metrics=True)

        write_method_mock = self.get_write_method_mock(metrics_logger)
        try:
            write_method_mock.assert_called_once()
            log.info(f"✅ {inspect.currentframe().f_code.co_name}: Write mock called.")
        except AssertionError as e:
            log.error(f"❌ {inspect.currentframe().f_code.co_name}: Write mock NOT called. Details: {e}")
            raise
        log.debug("test_miner_metrics_pattern completed.")


    def test_log_call_invokes_write_once(self, metrics_logger):
        import inspect
        log.debug("Running test_log_call_invokes_write_once...")
        metrics_logger.log("single_call_test", {}, {})
        write_method_mock = self.get_write_method_mock(metrics_logger)
        try:
            write_method_mock.assert_called_once()
            log.info(f"✅ {inspect.currentframe().f_code.co_name}: Write mock called.")
        except AssertionError as e:
            log.error(f"❌ {inspect.currentframe().f_code.co_name}: Write mock NOT called. Details: {e}")
            raise
        log.debug("test_log_call_invokes_write_once completed.")