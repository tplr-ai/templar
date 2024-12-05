# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Global imports
import logging
import time
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler


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
            show_time=False,
            show_path=False,
        )
    ],
)

# Create a logger instance
logger = logging.getLogger("templar")
logger.setLevel(logging.INFO)


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


__all__ = ["logger", "debug", "trace", "P", "T"]
