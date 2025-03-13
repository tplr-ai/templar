# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, Final, List
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write.point import Point
from influxdb_client.domain.write_precision import WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import time
import logging
import psutil
import GPUtil
from threading import Lock
import statistics
from . import __version__
import uuid
from bittensor import Config as BT_Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUNTIME_ID: Final[str] = str(uuid.uuid4())

CACHED_GPU_METRICS_INTERVAL: Final[int] = 2


class MetricsLogger:
    """
    Metrics Logger for Distributed Training using InfluxDB.
    Logs training metrics such as loss, gradients, GPU utilization, memory usage.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8086,
        database: str = "tplr_metrics",
        token: str | None = None,
        org: str = "templar",
        prefix: str = "",
        uid: str | None = None,
        version: str = __version__,
        runtime_id: str = RUNTIME_ID,
        config: BT_Config | None = None,
        group: str = "",
        job_type: str = "",
    ):
        """
        Initializes the InfluxDB client and prepares metadata for logging.

        Args:
            host (str, optional): Hostname of the InfluxDB server. Defaults to "localhost".
            port (int, optional): Port number of the InfluxDB server. Defaults to 8086.
            database (str, optional): Name of the InfluxDB database. Defaults to "tplr_metrics".
            token (str, optional): InfluxDB token. Defaults to None.
            org (str, optional): InfluxDB organization. Defaults to "templar".
            prefix (str, optional): Prefix to add to all metric names. Defaults to ""
            uid (str, optional): Unique identifier for the training run. Defaults to None.
            version (str, optional): Version of the templar library. Defaults to __version__.
            runtime_id (str, optional): Unique identifier for the runtime. Defaults to RUNTIME_ID.
            config (BT_Config, optional): Bittensor configuration object. Defaults to None.
            group (str, optional): Group name for the run. Defaults to "".
            job_type (str, optional): Job type for the run. Defaults to "".
        """

        if not token or not token.strip():
            raise ValueError("InfluxDB token must be provided and non-empty")

        url = f"https://{host}:{port}"
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.database = database
        self.org = org
        self.prefix = prefix
        self.uid = uid
        self.version = version
        self.runtime_id = runtime_id
        self.config = config
        self.group = group
        self.job_type = job_type
        self.lock = Lock()

    def process_value(self, v):
        if isinstance(v, (int, float)):
            return float(v)
        elif isinstance(v, list):
            # If the list appears to be a list of peer UIDs, log the raw list as a string.
            if v and all(isinstance(item, int) for item in v):
                return str(v)  # Alternatively, return len(v) if count is preferred
            # For lists of numbers that are not peer IDs, compute stats.
            if not v:  # Empty list
                return 0.0
            return {
                "mean": float(statistics.mean(v)),
                "min": float(min(v)),
                "max": float(max(v)),
                "median": float(statistics.median(v)),
            }
        return v

    def log(
        self,
        measurement: str,
        tags: dict,
        fields: dict,
        timestamp=None,
        with_system_metrics=True,
        with_gpu_metrics=True,
    ):
        """
        Logs metrics to InfluxDB.

        Args:
            measurement: Name of the measurement
            tags: Dictionary of tags to attach to the data point
            fields: Dictionary of fields to record
            timestamp: Optional timestamp (nanoseconds)
            with_system_metrics: Whether to include system metrics
            with_gpu_metrics: Whether to include GPU metrics
        """
        try:
            timestamp = timestamp or int(time.time_ns())

            point = Point(f"{self.prefix}{measurement}")

            processed_fields = self._process_fields(fields)
            if with_system_metrics:
                self._add_system_metrics(processed_fields)
            if with_gpu_metrics:
                self._add_gpu_metrics(tags, processed_fields)

            self._add_tags(point, tags)
            self._add_standard_tags(point)
            self._add_config_tags(point)

            for field_key, field_value in processed_fields.items():
                point = point.field(field_key, field_value)
            point = point.time(timestamp, WritePrecision.NS)

            with self.lock:
                self.write_api.write(bucket=self.database, org=self.org, record=point)

        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

    def _process_fields(self, fields):
        """Process field values and handle lists"""
        processed_fields = {}
        for k, v in fields.items():
            processed_value = self.process_value(v)
            if isinstance(processed_value, dict):
                for stat_name, stat_value in processed_value.items():
                    processed_fields[f"{k}_{stat_name}"] = stat_value
            else:
                processed_fields[k] = processed_value
        return processed_fields

    def _add_system_metrics(self, fields):
        """Add system metrics to fields"""
        system_metrics = get_system_metrics()
        for key, value in system_metrics.items():
            fields[f"sys_{key}"] = value

    def _get_cached_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Get cached GPU metrics
        Cache GPU metrics for `CACHED_GPU_METRICS_INTERVAL` seconds to improve
        performance during rapid logging.
        """
        current_time = time.time()
        if (
            not hasattr(self, "_gpu_metrics_cache")
            or not hasattr(self, "_gpu_metrics_last_update")
            or current_time - self._gpu_metrics_last_update
            >= CACHED_GPU_METRICS_INTERVAL
        ):
            self._gpu_metrics_cache = get_gpu_metrics()
            self._gpu_metrics_last_update = current_time

        return self._gpu_metrics_cache

    def _add_gpu_metrics(self, tags, fields):
        """Add GPU metrics to fields and tags"""
        gpu_metrics_list = self._get_cached_gpu_metrics()
        for i, gpu_metrics in enumerate(gpu_metrics_list):
            for key, value in gpu_metrics.items():
                if key in ("gpu_id", "gpu_name"):
                    tags[key] = value
                else:
                    fields[f"gpu{i}_{key}"] = value

    def _add_tags(self, point, tags):
        """Add custom tags to point"""
        for tag_key, tag_value in tags.items():
            point.tag(tag_key, str(tag_value))
        return point

    def _add_standard_tags(self, point):
        """Add standard tags from class attributes"""
        standard_tags = {
            "uid": self.uid,
            "prefix": self.prefix,
            "version": self.version,
            "runtime_id": self.runtime_id,
            "group": self.group,
            "job_type": self.job_type,
        }
        for tag_name, tag_value in standard_tags.items():
            if tag_value:
                point.tag(tag_name, str(tag_value))
        return point

    def _add_config_tags(self, point):
        """Add configuration tags if available"""
        if self.config is not None:
            try:
                config_dict = (
                    vars(self.config) if hasattr(self.config, "__dict__") else {}
                )
                for key, value in config_dict.items():
                    if (
                        not key.startswith("_")
                        and value is not None
                        and isinstance(value, (str, int, float, bool))
                    ):
                        point.tag(f"config_{key}", str(value))
            except Exception as e:
                logger.warning(f"Error adding config tags: {e}")
        return point


def get_gpu_metrics() -> List[Dict[str, Any]]:
    """
    Retrieves real-time GPU utilization metrics for all available GPUs.

    Returns:
    List[Dict[str, Any]]: List of GPU utilization metrics for each GPU.
    """
    gpus = GPUtil.getGPUs()
    if not gpus:
        return [
            {
                "gpu_usage": 0.0,
                "gpu_mem_used": 0.0,
                "gpu_mem_total": 0.0,
                "gpu_name": "",
                "gpu_id": 0,
            }
        ]

    result = []
    for i, gpu in enumerate(gpus):
        result.append(
            {
                "gpu_id": i,
                "gpu_usage": gpu.load * 100,
                "gpu_mem_used": gpu.memoryUsed,
                "gpu_mem_total": gpu.memoryTotal,
                "gpu_name": gpu.name,
            }
        )

    return result


def get_system_metrics() -> Dict[str, Any]:
    """
    Retrieves CPU and RAM usage metrics.

    Returns:
        Dict[str, Any]: Dictionary containing CPU and RAM usage metrics.
    """
    cpu_usage = psutil.cpu_percent()
    mem = psutil.virtual_memory()

    return {
        "cpu_usage": cpu_usage,
        "mem_used": mem.used / (1024**2),
        "mem_total": mem.total / (1024**2),
    }
