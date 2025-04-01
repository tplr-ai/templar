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

# Global imports
import json
import logging
import os
import socket
import time
import uuid
from datetime import datetime
from multiprocessing.queues import Queue
from typing import Final

import bittensor as bt
from logging_loki import LokiQueueHandler
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler

LOKI_URL: Final[str] = os.environ.get(
    "LOKI_URL", "https://logs.tplr.ai/loki/api/v1/push"
)
TRACE_ID: Final[str] = str(uuid.uuid4())


def T() -> float:
    """
    Returns the current time in seconds since the epoch.

    Returns:
        float: Current time in seconds.
    """
    return time.time()


def P(window: int, duration: float) -> str:
    """
    Formats a log prefix with the window number and duration.

    Args:
        window (int): The current window index.
        duration (float): The duration in seconds.

    Returns:
        str: A formatted string for log messages.
    """
    return f"[steel_blue]{window}[/steel_blue] ([grey63]{duration:.2f}s[/grey63])"


# Configure the root logger
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(
            markup=True,  # Enable markup parsing to allow color rendering
            rich_tracebacks=True,
            highlighter=NullHighlighter(),
            show_level=False,
            show_time=True,
            show_path=False,
        )
    ],
)

# Create a logger instance
logger = logging.getLogger("templar")
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Custom Logging Filter to silence subtensor warnings
# -----------------------------------------------------------------------------
class NoSubtensorWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Return False if the record contains the undesired subtensor warning
        return (
            "Verify your local subtensor is running on port" not in record.getMessage()
        )


# Apply our custom filter to both the root logger and all attached handlers.
logging.getLogger().addFilter(NoSubtensorWarning())
logger.addFilter(NoSubtensorWarning())
for handler in logging.getLogger().handlers:
    handler.addFilter(NoSubtensorWarning())


def debug() -> None:
    """
    Sets the logger level to DEBUG.
    """
    logger.setLevel(logging.DEBUG)


def trace() -> None:
    """
    Sets the logger level to TRACE.

    Note:
        The TRACE level is not standard in the logging module.
        You may need to add it explicitly if required.
    """
    TRACE_LEVEL_NUM = 5
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

    def trace_method(self, message, *args, **kws) -> None:
        if self.isEnabledFor(TRACE_LEVEL_NUM):
            self._log(TRACE_LEVEL_NUM, message, args, **kws)

    logging.Logger.trace = trace_method
    logger.setLevel(TRACE_LEVEL_NUM)


bt.logging.off()

logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
logger.addHandler(
    RichHandler(
        markup=True,
        rich_tracebacks=True,
        highlighter=NullHighlighter(),
        show_level=False,
        show_time=True,
        show_path=False,
    )
)


def setup_loki_logger(
    service: str,
    uid: str,
    version: str,
    environment="finney",
    url=LOKI_URL,
):
    """
    Hijack the templar logger with Loki logging.

    Args:
        uid: UID identifier for filtering logs
        version: Version identifier for filtering logs
        url: Loki server URL
    """

    host = socket.gethostname()
    pid = os.getpid()

    class StructuredLogFormatter(logging.Formatter):
        """Custom formatter that outputs logs in a structured format with metadata."""

        def format(self, record: logging.LogRecord) -> str:
            log_data = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "pathname": record.pathname,
                "filename": record.filename,
                "module": record.module,
                "lineno": record.lineno,
                "exc_info": record.exc_info,
                "funcName": record.funcName,
                "stack_info": record.stack_info,
                "thread": record.threadName,
                "thread_id": record.thread,
                "time": T(),
                "host": host,
                "pid": pid,
                "process": pid,
                "trace_id": TRACE_ID,
                "service": service,
                "environment": environment,
                "version": version,
                "uid": uid,
                # this duplication is needed for loki to work with labels
                "severity": record.levelname.lower(),
                "tags": {
                    "service": service,
                    "environment": environment,
                    "version": version,
                    "uid": uid,
                },
            }

            return json.dumps(log_data)

    try:
        log_queue = Queue(-1)

        loki_handler = LokiQueueHandler(
            queue=log_queue,  # type: ignore
            url=url,
            tags={},
            version="1",
            # TODO Add auth=(username, password) when available
            auth=None,
        )

        loki_handler.setFormatter(StructuredLogFormatter())

        loki_handler.setLevel(logger.level)

        loki_handler.addFilter(NoSubtensorWarning())

        logger.addHandler(loki_handler)
        return True
    except Exception as e:
        logger.error(f"Failed to add Loki logging: {e}")
        return False


__all__ = ["logger", "debug", "trace", "P", "T", "setup_loki_logger"]
