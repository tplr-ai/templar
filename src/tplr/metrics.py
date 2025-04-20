import asyncio
import datetime
import io
import json
import logging
import statistics
import threading
import time
import uuid
from datetime import timezone
from typing import Any

import psutil
import torch
import zstd
from bittensor import Config as BT_Config
from boto3.s3.transfer import TransferConfig
from boto3.session import Session
from botocore.config import Config as BotoConfig

# Unique runtime identifier
RUNTIME_ID = str(uuid.uuid4())

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_system_metrics() -> dict:
    """
    Retrieves CPU and RAM usage metrics.
    """
    cpu_usage = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    return {
        "cpu_usage": cpu_usage,
        "mem_used": mem.used / (1024**2),
        "mem_total": mem.total / (1024**2),
    }


def get_gpu_metrics() -> dict:
    """
    Retrieves GPU utilization metrics.
    """
    current_device = torch.cuda.current_device()
    stats = torch.cuda.memory_stats(current_device)
    return {
        "gpu_mem_segments": stats.get("segment.all.current", 0),
        "gpu_mem_allocated_mb": torch.cuda.memory_allocated(current_device) / 1024**2,
        "gpu_mem_cached_mb": torch.cuda.memory_reserved(current_device) / 1024**2,
        "gpu_mem_total_mb": torch.cuda.max_memory_allocated(current_device) / 1024**2,
        "gpu_name": f"CUDA:{current_device}",
        "gpu_id": current_device,
    }


def process_value(v):
    if isinstance(v, int):
        return int(v)
    elif isinstance(v, float):
        return float(v)
    elif isinstance(v, list):
        if v and all(isinstance(item, int) for item in v):
            return str(v)
        if not v:
            return 0.0
        return {
            "mean": float(statistics.mean(v)),
            "min": float(min(v)),
            "max": float(max(v)),
            "median": float(statistics.median(v)),
        }
    elif isinstance(v, str):
        return v
    elif v is None:
        return 0.0
    else:
        return str(v)


def process_fields(fields: dict) -> dict:
    processed = {}
    for k, v in fields.items():
        pv = process_value(v)
        if isinstance(pv, dict):
            for stat, val in pv.items():
                processed[f"{k}_{stat}"] = val
        else:
            processed[k] = pv
    return processed


class MetricsLogger:
    """
    Async metrics logger to S3-compatible storage.
    """

    def __init__(
        self,
        bucket: str,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        region_name: str | None = None,
        prefix: str | None = None,
        uid: str | None = None,
        version: str | None = None,
        runtime_id: str = RUNTIME_ID,
        role: str | None = None,
        config: BT_Config | None = None,
        group: str | None = None,
        job_type: str | None = None,
        max_queue_size: int = 600_000,
        flush_interval: float = 10.0,
        http_max_pool_connections: int = 100,
        http_connect_timeout: float = 2.0,
        http_read_timeout: float = 30.0,
        http_max_retries: int = 3,
        multipart_threshold: int = 1 * 1024 * 1024,
        multipart_chunksize: int = 1 * 1024 * 1024,
        s3_max_concurrency: int = 50,
    ):
        self.bucket = bucket
        self.s3_prefix = prefix or "telemetry"
        self.uid = uid or str(uuid.uuid4())
        self.version = version or ""
        self.runtime_id = runtime_id
        self.role = role or ""
        self.group = group or ""
        self.job_type = job_type or ""
        self.config = config
        self.flush_interval = flush_interval

        boto_config = BotoConfig(
            max_pool_connections=http_max_pool_connections,
            connect_timeout=http_connect_timeout,
            read_timeout=http_read_timeout,
            retries={"max_attempts": http_max_retries, "mode": "standard"},
        )

        self.transfer_config = TransferConfig(
            multipart_threshold=multipart_threshold,
            multipart_chunksize=multipart_chunksize,
            max_concurrency=s3_max_concurrency,
            use_threads=True,
        )

        session = Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region_name,
        )
        self.s3_client = session.client(
            "s3", endpoint_url=endpoint_url, config=boto_config
        )

        self._max_queue = max_queue_size
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            logger.warning("MetricsLogger worker not ready in 5s")

    def log(
        self,
        measurement: str,
        tags: dict,
        fields: dict,
        timestamp: int | None = None,
        with_system_metrics: bool = False,
        with_gpu_metrics: bool = False,
    ) -> None:
        ts = timestamp or int(time.time_ns())
        data_fields = process_fields(fields)
        if with_system_metrics:
            data_fields.update(get_system_metrics())
        if with_gpu_metrics:
            for k, v in get_gpu_metrics().items():
                if k in ("gpu_id", "gpu_name"):
                    tags[k] = v
                else:
                    data_fields[k] = v

        all_tags = {**tags}
        for t_key, t_val in (
            ("uid", self.uid),
            ("version", self.version),
            ("runtime_id", self.runtime_id),
            ("role", self.role),
            ("group", self.group),
            ("job_type", self.job_type),
        ):
            if t_val:
                all_tags[t_key] = t_val
        if self.config:
            for k, v in vars(self.config).items():
                if (
                    not k.startswith("_")
                    and v is not None
                    and isinstance(v, (str, int, float, bool))
                ):
                    all_tags[f"config_{k}"] = v

        record = {
            "measurement": measurement,
            "timestamp": ts,
            "tags": all_tags,
            "fields": data_fields,
        }
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, record)
        except Exception as e:
            logger.warning(f"Dropping metric {measurement}: {e}")

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        self._buffers: dict[str, dict] = {}
        self._buffers_lock = asyncio.Lock()
        self._loop.create_task(self._consumer())
        self._loop.create_task(self._periodic_flush())
        self._ready.set()
        self._loop.run_forever()

    async def _consumer(self):
        while True:
            record = await self._queue.get()
            await self._handle(record)

    async def _periodic_flush(self):
        """Flush expired buffers at configured intervals."""
        while True:
            await asyncio.sleep(self.flush_interval)
            await self._flush_expired()

    async def _handle(self, record: dict[str, Any]) -> None:
        minute = datetime.datetime.fromtimestamp(
            record["timestamp"] / 1e9, tz=timezone.utc
        )
        bucket_time = minute.replace(second=0, microsecond=0)
        key = (
            f"{self.s3_prefix}/uid={self.uid}/version={self.version}/role={self.role}/"
            f"year={bucket_time.year}/month={bucket_time.month:02}/day={bucket_time.day:02}/"
            f"hour={bucket_time.hour:02}/minute={bucket_time.minute:02}/"
            f"{record['measurement']}_{self.runtime_id}.jsonl.zstd"
        )
        line = json.dumps(
            {
                "timestamp": record["timestamp"],
                "tags": record["tags"],
                "fields": record["fields"],
            }
        )
        async with self._buffers_lock:
            buf = self._buffers.get(key)
            if not buf:
                buf = {"minute": bucket_time, "lines": []}
                self._buffers[key] = buf
            buf["lines"].append(line)
        await self._flush_expired()

    async def _flush_expired(self):
        now = datetime.datetime.utcnow().replace(second=0, microsecond=0)
        to_flush = []
        async with self._buffers_lock:
            for key, buf in list(self._buffers.items()):
                if buf["minute"] < now:
                    to_flush.append((key, buf["lines"]))
                    del self._buffers[key]
        for key, lines in to_flush:
            asyncio.create_task(self._upload(key, lines))

    async def _upload(self, key: str, lines: list[str]) -> None:
        try:
            data = ("\n".join(lines) + "\n").encode("utf-8")
            compressed = zstd.compress(data)
            await self._loop.run_in_executor(None, self._s3_put, key, compressed)
        except Exception as e:
            logger.error(f"Failed to upload metrics {key}: {e}")

    def _s3_put(self, key: str, data: bytes) -> None:
        """Blocking S3 upload with multipart and persistent connections."""
        # TransferConfig will handle Multipart uploads
        self.s3_client.upload_fileobj(
            io.BytesIO(data), Bucket=self.bucket, Key=key, Config=self.transfer_config
        )

    def close(self) -> None:
        if getattr(self, "_loop", None):
            future = asyncio.run_coroutine_threadsafe(self._flush_all(), self._loop)
            try:
                future.result(timeout=10)
            except Exception as e:
                logger.warning(f"Error flushing metrics on close: {e}")
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    async def _flush_all(self) -> None:
        to_flush = []
        async with self._buffers_lock:
            for key, buf in list(self._buffers.items()):
                to_flush.append((key, buf["lines"]))
                del self._buffers[key]
        await asyncio.gather(*(self._upload(k, lines) for k, lines in to_flush))
