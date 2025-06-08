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

import json
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Dict

import tplr


def log_with_context(level: str, message: str, **kwargs) -> None:
    """
    Generic function to log messages with dynamic keyword arguments formatted into the log string.
    
    Args:
        level: Log level (e.g., "info", "debug", "warning", "error")
        message: The main log message
        **kwargs: Additional context to include in the log
    """
    # Sanitize wandb and metrics_logger keys to avoid logging them as context
    sanitized_kwargs = {
        k: v for k, v in kwargs.items() 
        if k not in ["wandb", "metrics_logger"] and not k.startswith("_")
    }
    
    if sanitized_kwargs:
        # Format context as key=value pairs
        context_str = " | ".join(f"{k}={v}" for k, v in sanitized_kwargs.items())
        full_message = f"{message} | {context_str}"
    else:
        full_message = message
    
    # Get the appropriate logger method
    log_method = getattr(tplr.logger, level.lower(), tplr.logger.info)
    log_method(full_message)


def json_encode_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper to safely JSON encode complex data types for logging.
    
    Args:
        data: Dictionary with potentially complex values
        
    Returns:
        Dictionary with JSON-safe values
    """
    result = {}
    for key, value in data.items():
        try:
            # Try to JSON encode to check if it's serializable
            json.dumps(value)
            result[key] = value
        except (TypeError, ValueError):
            # If not serializable, convert to string
            result[key] = str(value)
    return result


@contextmanager
def timer(name: str, wandb_obj=None, step=None, metrics_logger=None):
    """
    Context manager for timing operations and optionally logging to WandB and metrics.
    
    Args:
        name: Name of the operation being timed
        wandb_obj: Optional WandB object for logging
        step: Optional step number for WandB logging
        metrics_logger: Optional metrics logger for InfluxDB
    """
    start = perf_counter()
    try:
        yield
    finally:
        duration = perf_counter() - start
        
        # Use the enhanced log_with_context for debug logging
        log_with_context("debug", f"{name} took {duration:.2f}s")
        
        # Log to WandB if provided
        if wandb_obj and step is not None:
            wandb_obj.log({f"validator/{name}": duration}, step=step)
            
        # Log to metrics logger if provided
        if metrics_logger and step is not None:
            metrics_logger.log(
                measurement="timing", 
                tags={"window": step}, 
                fields={name: duration}
            ) 