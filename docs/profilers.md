# Profilers

The TPLR project includes profiling utilities for performance tracking and analysis. These profilers allow you to measure and analyze various aspects of the system's performance, and can be enabled or disabled through environment variables. The profilers also support OpenTelemetry integration for exporting metrics to observability platforms.

## Available Profilers

1. **Timer Profiler** - Tracks function execution times
2. **Shard Profiler** - Measures parquet file read performance

## Profiled Components

The following components in the TPLR system are instrumented with profilers:

### R2DatasetLoader

- **Profiler Type**: Timer Profiler
- **Instance Name**: `R2DatasetLoader`
- **Tracked Operations**: Data loading, parquet file processing, tokenization, metadata operations

### Comms (Communication Layer)

- **Profiler Type**: Timer Profiler
- **Instance Name**: `Comms`
- **Tracked Operations**:
  - **S3 Operations**: `s3_put_object`, `s3_get_object`, `upload_large_file`, `download_large_file`
  - **Data Operations**: `put`, `get`, `gather`
  - **Checkpoint Operations**: `get_latest_checkpoint`, `load_checkpoint`, `save_checkpoint`
  - **Peer Management**: `is_miner_active`, `get_peer_list`
  - **Aggregation**: `load_aggregation`

## Configuration via Environment Variables

By default, all profilers are **disabled** to minimize overhead in production environments. You can control profiler behavior using the following environment variables:

### Profiler Control

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TPLR_ENABLE_PROFILERS` | `0` | Master switch to enable/disable all profilers |
| `TPLR_ENABLE_TIMER_PROFILER` | `0` | Enable/disable the timer profiler specifically |
| `TPLR_ENABLE_SHARD_PROFILER` | `0` | Enable/disable the shard profiler specifically |

### OpenTelemetry Integration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TPLR_ENABLE_OTEL_PROFILING` | `0` | Enable/disable OpenTelemetry metrics export |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OpenTelemetry OTLP collector endpoint |

Set the value to `1` to enable a profiler, or `0` to disable it.

## Usage Examples

### Enabling All Profilers

```bash
export TPLR_ENABLE_PROFILERS=1
export TPLR_ENABLE_TIMER_PROFILER=1
export TPLR_ENABLE_SHARD_PROFILER=1
python neurons/miner.py
```

### Enabling All Profilers with OpenTelemetry

```bash
export TPLR_ENABLE_PROFILERS=1
export TPLR_ENABLE_TIMER_PROFILER=1
export TPLR_ENABLE_SHARD_PROFILER=1
export TPLR_ENABLE_OTEL_PROFILING=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://your-collector:4317
python neurons/miner.py
```

### Enabling Only Timer Profiler

```bash
export TPLR_ENABLE_PROFILERS=1
export TPLR_ENABLE_TIMER_PROFILER=1
export TPLR_ENABLE_SHARD_PROFILER=0
python neurons/miner.py
```

### Enabling Only Shard Profiler

```bash
export TPLR_ENABLE_PROFILERS=1
export TPLR_ENABLE_SHARD_PROFILER=1
python neurons/miner.py
```

## API Usage

The profilers are designed to be used as singletons, available throughout the codebase:

```python
from tplr.profilers import get_timer_profiler, get_shard_profiler

# Get the timer profiler instance
timer_profiler = get_timer_profiler("MyModule")

# Use as a decorator
@timer_profiler.profile("my_function")
def my_function():
    # Code to profile
    pass

# Get the shard profiler instance
shard_profiler = get_shard_profiler()

# Track shard read performance
timer_id = shard_profiler.start_read(shard_path, chosen_shard)
# ... perform read operation
elapsed = shard_profiler.end_read(timer_id, shard_path)
```

## OpenTelemetry Metrics

When OpenTelemetry integration is enabled, the profilers export the following metrics:

### Timer Profiler Metrics

| Metric Name | Type | Description | Attributes |
|-------------|------|-------------|------------|
| `profiler_function_duration_seconds` | Histogram | Function execution duration | `function_name`, `profiler_name` |
| `profiler_function_calls_total` | Counter | Total number of function calls | `function_name`, `profiler_name` |
| `profiler_function_errors_total` | Counter | Total number of function errors | `function_name`, `profiler_name` |

### Shard Profiler Metrics

| Metric Name | Type | Description | Attributes |
|-------------|------|-------------|------------|
| `profiler_shard_read_duration_seconds` | Histogram | Shard read operation duration | `shard_path`, `profiler_name`, `row_groups`, `rows_per_group` |
| `profiler_shard_reads_total` | Counter | Total number of shard read operations | `shard_path`, `profiler_name`, `row_groups`, `rows_per_group` |
| `profiler_shard_file_size_bytes` | Histogram | Shard file size in bytes | `shard_path`, `profiler_name`, `row_groups`, `rows_per_group` |
| `profiler_shard_rows_processed` | Histogram | Number of rows processed per operation | `shard_path`, `profiler_name`, `row_groups`, `rows_per_group` |

## Performance Impact

When profilers are disabled via environment variables, they are replaced with dummy implementations that have negligible performance impact. This allows profiling code to remain in place without affecting production performance.

The OpenTelemetry integration adds minimal overhead when enabled, as metrics are exported asynchronously in the background.

## Local Development with Jaeger

For local development and testing, you can use the provided Jaeger setup to collect and visualize profiler metrics.

### Starting the Telemetry Stack

```bash
# Start Jaeger with OpenTelemetry support
cd telemetry/tracing
docker compose up -d

# Verify services are running
docker compose ps
```

This will start:

- **Jaeger All-in-One** with OpenTelemetry collector support
- **Jaeger UI** available at <http://localhost:16686>

### Running with Telemetry

```bash
# Enable profilers and OpenTelemetry
export TPLR_ENABLE_PROFILERS=1
export TPLR_ENABLE_TIMER_PROFILER=1
export TPLR_ENABLE_SHARD_PROFILER=1
export TPLR_ENABLE_OTEL_PROFILING=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Run your application
python neurons/miner.py
```

### Viewing Metrics

1. Open the Jaeger UI at <http://localhost:16686>
2. Select the service `tplr-profiler.profiler` from the dropdown
3. View timing traces and performance metrics
4. Use the search functionality to filter by specific functions or shards

### Example Metrics in Jaeger

When running with profilers enabled, you can expect to see metrics like:

**R2DatasetLoader Operations:**

- `R2DatasetLoader.next_pages` - Time to load next batch of data pages
- `R2DatasetLoader._get_parquet` - Time to fetch parquet files from R2
- `R2DatasetLoader._batch_tokenize` - Time for tokenization operations

**Comms Operations:**

- `Comms.s3_put_object` - Time to upload objects to S3/R2
- `Comms.s3_get_object` - Time to download objects from S3/R2
- `Comms.gather` - Time to gather gradients from multiple peers
- `Comms.save_checkpoint` - Time to save model checkpoints
- `Comms.load_checkpoint` - Time to load model checkpoints
- `Comms.is_miner_active` - Time to check if a miner is active

These metrics help identify performance bottlenecks in:

- Network I/O operations (S3/R2 transfers)
- Peer communication and gradient gathering
- Checkpoint operations
- Data loading and processing

### Alternative Endpoints

The Jaeger setup provides multiple endpoints:

| Protocol | Endpoint | Use Case |
|----------|----------|----------|
| OTLP gRPC | `http://localhost:4317` | Default for profiler metrics (recommended) |
| OTLP HTTP | `http://localhost:4318` | Alternative HTTP-based export |
| Jaeger UI | `http://localhost:16686` | Web interface for viewing traces and metrics |

### Testing the Integration

A test script is available to verify the profiler and telemetry integration:

```bash
# Run the test script
python test_telemetry_setup.py
```

This will:

- Generate sample timer and shard profiler metrics
- Export metrics to the Jaeger collector (if running)
- Display test results and usage instructions

### Stopping the Telemetry Stack

```bash
cd telemetry/tracing
docker compose down

# To also remove data volumes
docker compose down -v
```

## Troubleshooting

### Common Issues

**Jaeger not receiving metrics:**

- Ensure Jaeger is running: `docker compose ps`
- Check the OTLP endpoint: `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317`
- Verify profilers are enabled: `TPLR_ENABLE_OTEL_PROFILING=1`

**No data in Jaeger UI:**

- Wait 30 seconds for metrics export (default interval)
- Check Jaeger logs: `docker compose logs jaeger`
- Verify service name: Look for `tplr-profiler.profiler` in the service dropdown
- Check for specific operations: Look for `Comms.*` or `R2DatasetLoader.*` in the traces

**Performance impact:**

- Disable profilers in production: `TPLR_ENABLE_PROFILERS=0`
- Use selective profiling: Enable only needed profilers
- Monitor export frequency and adjust if needed
