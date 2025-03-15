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
import logging
import os
import time

import bittensor as bt
from logging_loki import LokiHandler
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler

from . import __version__


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

    logging.Logger.trace = trace_method  # type: ignore
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


# Check if Loki logging is enabled via environment variable (default is disabled)
enable_loki = os.environ.get("ENABLE_LOKI", "false").lower() in ["true", "1", "yes"]

# Only add LokiHandler if explicitly enabled
if enable_loki:
    try:
        loki_handler = LokiHandler(
            url="http://localhost:3100/loki/api/v1/push",
            tags={
                "app": "templar",
                "env": "prod",
                "version": __version__,
                "uid": "default_uid",
            },
            version="1",
        )
        loki_handler.setLevel(logging.INFO)
        logger.addHandler(loki_handler)
        logger.info("Loki logging enabled")
    except Exception as e:
        logger.error(f"Failed to add LokiHandler: {e}")
else:
    logger.debug("Loki logging disabled")

__all__ = ["logger", "debug", "trace", "P", "T"]
