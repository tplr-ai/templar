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
import logging.handlers
import os
import socket
import time
import uuid
from datetime import datetime
from queue import Queue
from typing import Final

import bittensor as bt
import logging_loki
from rich import highlighter, logging as richlogging

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
        richlogging.RichHandler(
            markup=True,  # Enable markup parsing to allow color rendering
            rich_tracebacks=True,
            highlighter=highlighter.NullHighlighter(),
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
logger.propagate = True
logger.handlers.clear()
logger.addHandler(
    richlogging.RichHandler(
        markup=True,
        rich_tracebacks=True,
        highlighter=highlighter.NullHighlighter(),
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
) -> logging.Logger:
    """
    Add Loki logging to templar logger.

    Configures the logger to send logs to both Loki (asynchronously) and stdout.
    Uses a Queue for asynchronous logging to avoid blocking the main thread.

    Args:
        service: Service name (e.g., 'miner', 'validator')
        uid: UID identifier for filtering logs
        version: Version identifier for filtering logs
        environment: Environment name
        url: Loki server URL

    Returns:
        The configured logger instance
    """
    host = socket.gethostname()
    pid = os.getpid()
    tags = {
        "service": service,
        "host": host,
        "pid": pid,
        "environment": environment,
        "version": version,
        "uid": uid,
        "trace_id": TRACE_ID,
    }

    class StructuredLogFormatter(logging.Formatter):
        """Custom formatter that outputs logs in a structured format with metadata."""

        def format(self, record: logging.LogRecord) -> str:
            log_data = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "host": host,
                "pid": pid,
                "service": service,
                "environment": environment,
                "version": version,
                "uid": uid,
                "trace_id": TRACE_ID,
            }

            if hasattr(record, "extra_data") and record.extra_data:  # type: ignore
                log_data.update(record.extra_data)  # type: ignore

            return json.dumps(log_data)

    def log_with_context(logger, level, message, **context):
        """Log a message with additional context data."""
        record = logging.LogRecord(
            name=logger.name,
            level=getattr(logging, level.upper()),
            pathname=__file__,
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )

        record.extra_data = context

        for handler in logger.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    try:
        logger = logging.getLogger("templar")

        log_queue = Queue(-1)

        queue_handler = logging.handlers.QueueHandler(log_queue)
        listener = logging.handlers.QueueListener(log_queue, respect_handler_level=True)

        loki_handler = logging_loki.LokiHandler(
            url=url,
            tags=tags,
            auth=None,  # TODO Add auth=(username, password) when available
            version="1",
        )

        listener.handlers = [loki_handler]  # type: ignore

        console_handler = richlogging.RichHandler(
            markup=True,
            rich_tracebacks=True,
            highlighter=highlighter.NullHighlighter(),
            show_level=False,
            show_time=True,
            show_path=False,
        )

        loki_handler.setFormatter(StructuredLogFormatter())

        logger.setLevel(logging.INFO)

        logger.handlers.clear()
        listener.start()

        logger.addHandler(queue_handler)

        logger.addHandler(console_handler)

        logger.log_with_context = lambda level, message, **kwargs: log_with_context(  # type: ignore
            logger, level, message, **kwargs
        )

        logger.propagate = False

        logger._listener = listener  # type: ignore

        return logger
    except Exception as e:
        logger = logging.getLogger("templar")
        logger.error(f"Failed to add Loki logging: {e}")

        if not logger.handlers:
            logger.addHandler(
                richlogging.RichHandler(
                    markup=True,
                    rich_tracebacks=True,
                    highlighter=highlighter.NullHighlighter(),
                    show_level=False,
                    show_time=True,
                    show_path=False,
                )
            )

        return logger


def log_with_context(level, message, **context):
    """
    Log a message with additional context data.

    This is a convenience function for logging with extra context data
    when the logger hasn't been initialized with setup_loki_logger.

    Args:
        level: Log level (e.g., 'info', 'error', 'warning')
        message: The log message
        **context: Additional context data to include with the log

    Example:
        log_with_context('info', 'Processing batch', batch_size=32, batch_id='abc123')
    """
    if not hasattr(logger, "log_with_context"):
        getattr(logger, level.lower())(message)
        return

    logger.log_with_context(level, message, **context)  # type: ignore


__all__ = [
    "logger",
    "debug",
    "trace",
    "P",
    "T",
    "setup_loki_logger",
    "log_with_context",
]
