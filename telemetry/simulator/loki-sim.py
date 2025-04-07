#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "python-logging-loki>=0.3.1",
#   "requests>=2.32.3",
#   "multiprocessing-logging>=0.3.4",
# ]
# ///

"""
Loki Logging Simulator - A tool to test Loki log ingestion with different patterns.

example usage:

```
./loki-sim.py --url https://logs.tplr.ai/loki/api/v1/push --rate 10 --duration 300
```

This script provides a flexible way to test Loki log ingestion with:
- Configurable log volume and frequency
- Multiple log levels
- Structured logging with metadata
- Trace IDs for distributed tracing
- Random errors and warnings
- Performance metrics simulation
"""

import argparse
import json
import logging
import logging.handlers
import os
import random
import signal
import socket
import sys
import time
import uuid
from datetime import datetime
from multiprocessing import Queue
from threading import Thread

import logging_loki


class StructuredLogFormatter(logging.Formatter):
    """Custom formatter that outputs logs in a structured format with metadata."""

    def format(self, record):
        # Standard log attributes
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }

        # Add any extra attributes provided in the log record
        if hasattr(record, "extra_data") and record.extra_data:
            log_data.update(record.extra_data)

        # Format as JSON
        return json.dumps(log_data)


def setup_logger(loki_url, service_name, environment, version="1", add_console=True):
    """
    Set up a configurable logger that sends log messages to Loki using a Queue for performance.

    Using LokiQueueHandler for asynchronous logging to avoid blocking the main thread
    during network operations. The queue handler sends messages to a separate thread
    that handles the actual HTTP requests to Loki.

    Args:
        loki_url: URL for Loki's HTTP push endpoint
        service_name: Name of the service (used in tags)
        environment: Environment name (used in tags)
        version: Loki API version
        add_console: Whether to also log to console

    Returns:
        Configured logger
    """
    # Create a logger
    logger = logging.getLogger("loki-simulator")
    logger.setLevel(logging.DEBUG)

    # Create a message queue for asynchronous logging
    log_queue = Queue(-1)  # Unbounded queue

    # Configure Loki queue handler for non-blocking performance
    loki_handler = logging_loki.LokiQueueHandler(
        queue=log_queue,
        url=loki_url,
        tags={
            "service": service_name,
            "environment": environment,
            "host": socket.gethostname(),
        },
        version=version,
        # Additional options for resilience
        auth=None,  # Add (username, password) tuple if authentication is needed
    )

    # Use structured logging format
    loki_handler.setFormatter(StructuredLogFormatter())
    logger.addHandler(loki_handler)

    # Add console handler if requested
    if add_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)

    return logger


def generate_trace_id():
    """Generate a random trace ID for distributed tracing simulation."""
    return str(uuid.uuid4())


def log_with_context(logger, level, message, **context):
    """Log a message with additional context data."""
    # Create a log record with extra data
    record = logging.LogRecord(
        name=logger.name,
        level=getattr(logging, level.upper()),
        pathname=__file__,
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )

    # Add extra context to the record
    record.extra_data = context

    # Pass the record to each handler
    for handler in logger.handlers:
        if record.levelno >= handler.level:
            handler.handle(record)


def simulate_api_request(logger, endpoints, trace_id=None):
    """Simulate an API request with multiple log entries."""
    if not trace_id:
        trace_id = generate_trace_id()

    endpoint = random.choice(endpoints)
    method = random.choice(["GET", "POST", "PUT", "DELETE"])
    status_code = random.choices([200, 404, 500], weights=[0.9, 0.05, 0.05])[0]
    duration = random.uniform(0.05, 2.0)

    # Log request start
    log_with_context(
        logger,
        "info",
        f"Started {method} request to {endpoint}",
        trace_id=trace_id,
        method=method,
        endpoint=endpoint,
        event="request_start",
    )

    # Simulate processing time
    time.sleep(random.uniform(0.01, 0.1))

    # Maybe log some debug info
    if random.random() < 0.3:
        log_with_context(
            logger,
            "debug",
            f"Processing {method} request parameters",
            trace_id=trace_id,
            method=method,
            endpoint=endpoint,
            event="request_processing",
            parameters={"query": "example", "limit": 10},
        )

    # Maybe log a database query
    if random.random() < 0.7:
        query_time = random.uniform(0.01, 0.5)
        log_with_context(
            logger,
            "debug",
            f"Database query executed in {query_time:.2f}s",
            trace_id=trace_id,
            method=method,
            endpoint=endpoint,
            event="database_query",
            query_time=query_time,
            rows_returned=random.randint(1, 100),
        )

    # Add an error/warning based on status code
    if status_code == 404:
        log_with_context(
            logger,
            "warning",
            f"Resource not found for {endpoint}",
            trace_id=trace_id,
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            event="resource_not_found",
        )
    elif status_code == 500:
        error_type = random.choice(["DatabaseError", "TimeoutError", "ValidationError"])
        log_with_context(
            logger,
            "error",
            f"{error_type} occurred while processing request",
            trace_id=trace_id,
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            error_type=error_type,
            event="request_error",
        )

    # Log request completion
    log_with_context(
        logger,
        "info",
        f"Completed {method} request to {endpoint}",
        trace_id=trace_id,
        method=method,
        endpoint=endpoint,
        status_code=status_code,
        duration_ms=duration * 1000,
        event="request_complete",
    )


def simulate_background_task(logger, tasks, duration=10):
    """Simulate a background task with progress updates."""
    task_name = random.choice(tasks)
    trace_id = generate_trace_id()

    # Log task start
    log_with_context(
        logger,
        "info",
        f"Started background task: {task_name}",
        trace_id=trace_id,
        task=task_name,
        event="task_start",
        uid="999",
        version="0.2.62",
    )

    # Random number of progress updates
    steps = random.randint(3, 8)
    for i in range(steps):
        # Wait a bit
        time.sleep(duration / steps)

        # Log progress
        progress = (i + 1) / steps * 100
        log_with_context(
            logger,
            "debug",
            f"Task progress: {progress:.1f}%",
            trace_id=trace_id,
            task=task_name,
            progress=progress,
            step=i + 1,
            total_steps=steps,
            event="task_progress",
            uid="999",
            version="0.2.62",
        )

        # Random warning
        if random.random() < 0.2:
            log_with_context(
                logger,
                "warning",
                "Task processing slower than expected",
                trace_id=trace_id,
                task=task_name,
                progress=progress,
                event="task_slow",
                uid="999",
                version="0.2.62",
            )

    # Success or failure
    if random.random() < 0.9:  # 90% success rate
        log_with_context(
            logger,
            "info",
            f"Completed background task: {task_name}",
            trace_id=trace_id,
            task=task_name,
            duration_seconds=duration,
            event="task_complete",
            items_processed=random.randint(10, 1000),
            uid="999",
            version="0.2.62",
        )
    else:
        error_type = random.choice(["MemoryError", "TimeoutError", "ResourceError"])
        log_with_context(
            logger,
            "error",
            f"Background task failed: {task_name}",
            trace_id=trace_id,
            task=task_name,
            error=error_type,
            event="task_failure",
            duration_seconds=duration,
            uid="999",
            version="0.2.62",
        )


def simulate_system_metrics(logger, interval=30, duration=300):
    """Simulate system metrics logging."""
    start_time = time.time()
    end_time = start_time + duration

    while time.time() < end_time:
        # Generate fake metrics
        cpu_usage = random.uniform(5, 95)
        memory_usage = random.uniform(20, 85)
        disk_usage = random.uniform(30, 90)

        # Log metrics
        log_with_context(
            logger,
            "info",
            "System metrics collected",
            event="system_metrics",
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory_usage,
            disk_usage_percent=disk_usage,
            network_throughput_mbps=random.uniform(0.1, 50),
            open_file_descriptors=random.randint(10, 500),
            metric_type="system",
            uid="999",
            version="0.2.62",
        )

        # Random warnings based on thresholds
        if cpu_usage > 80:
            log_with_context(
                logger,
                "warning",
                "High CPU usage detected",
                event="high_cpu_usage",
                cpu_usage_percent=cpu_usage,
                metric_type="system_alert",
                uid="999",
                version="0.2.62",
            )

        if memory_usage > 80:
            log_with_context(
                logger,
                "warning",
                "High memory usage detected",
                event="high_memory_usage",
                memory_usage_percent=memory_usage,
                metric_type="system_alert",
                uid="999",
                version="0.2.62",
            )

        if disk_usage > 85:
            log_with_context(
                logger,
                "warning",
                "High disk usage detected",
                event="high_disk_usage",
                disk_usage_percent=disk_usage,
                metric_type="system_alert",
                uid="999",
                version="0.2.62",
            )

        time.sleep(interval)


def run_load_simulation(
    logger, requests_per_second, duration_seconds, endpoints, tasks
):
    """Run a load simulation with the specified parameters."""
    print(
        f"Starting load simulation: {requests_per_second} req/s for {duration_seconds} seconds"
    )

    start_time = time.time()
    end_time = start_time + duration_seconds
    request_count = 0

    # Start system metrics thread
    metrics_thread = Thread(
        target=simulate_system_metrics, args=(logger, 15, duration_seconds)
    )
    metrics_thread.daemon = True
    metrics_thread.start()

    # Start background tasks thread
    def run_tasks():
        while time.time() < end_time:
            task_duration = random.uniform(5, 20)
            simulate_background_task(logger, tasks, task_duration)
            time.sleep(random.uniform(5, 15))

    tasks_thread = Thread(target=run_tasks)
    tasks_thread.daemon = True
    tasks_thread.start()

    # Main loop for API requests
    try:
        while time.time() < end_time:
            cycle_start = time.time()

            # Calculate how many requests to send in this cycle
            requests_this_second = int(requests_per_second)
            if random.random() < (requests_per_second - requests_this_second):
                requests_this_second += 1

            # Send the requests
            for _ in range(requests_this_second):
                simulate_api_request(logger, endpoints)
                request_count += 1

            # Sleep to maintain the requested rate
            elapsed = time.time() - cycle_start
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")

    # Final statistics
    actual_duration = time.time() - start_time
    print("\nSimulation complete:")
    print(f"- Total requests: {request_count}")
    print(f"- Actual duration: {actual_duration:.2f} seconds")
    print(f"- Average rate: {request_count / actual_duration:.2f} req/s")


def main():
    """Parse command line arguments and run the simulation."""
    parser = argparse.ArgumentParser(description="Loki Logging Simulator")
    parser.add_argument(
        "--url",
        default="https://logs.tplr.ai/loki/api/v1/push",
        help="Loki push API URL (default: https://logs.tplr.ai/loki/api/v1/push)",
    )
    parser.add_argument(
        "--service",
        default="python-simulator",
        help="Service name for log tags (default: python-simulator)",
    )
    parser.add_argument(
        "--env", default="dev", help="Environment name for log tags (default: dev)"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=5.0,
        help="Simulated requests per second (default: 5.0)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Simulation duration in seconds (default: 60)",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable console logging")

    args = parser.parse_args()

    # Sample endpoints and tasks to simulate
    endpoints = [
        "/api/users",
        "/api/items",
        "/api/orders",
        "/api/search",
        "/api/auth/login",
        "/api/auth/logout",
        "/api/reports/daily",
        "/api/reports/monthly",
        "/api/status",
    ]

    tasks = [
        "data-export",
        "email-notification",
        "report-generation",
        "data-cleanup",
        "user-sync",
        "backup",
        "index-rebuild",
    ]

    # Set up signal handler
    def signal_handler(sig, frame):
        print("\nShutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Setup logger
    logger = setup_logger(args.url, args.service, args.env, add_console=not args.quiet)

    print(f"Logging to Loki at: {args.url}")
    print(f"Service: {args.service}, Environment: {args.env}")

    # Initial heartbeat log
    log_with_context(
        logger,
        "info",
        "Loki simulator starting up",
        event="startup",
        simulator_version="1.0.0",
        python_version=sys.version,
        params={"rate": args.rate, "duration": args.duration, "loki_url": args.url},
    )

    # Run the simulation
    run_load_simulation(logger, args.rate, args.duration, endpoints, tasks)

    # Final log
    log_with_context(logger, "info", "Loki simulator shutting down", event="shutdown")


if __name__ == "__main__":
    main()
