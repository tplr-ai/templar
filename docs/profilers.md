# Profilers

The TPLR project includes profiling utilities for performance tracking and analysis. These profilers allow you to measure and analyze various aspects of the system's performance, and can be enabled or disabled through environment variables.

## Available Profilers

1. **Timer Profiler** - Tracks function execution times
2. **Shard Profiler** - Measures parquet file read performance

## Configuration via Environment Variables

By default, all profilers are **disabled** to minimize overhead in production environments. You can control profiler behavior using the following environment variables:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TPLR_ENABLE_PROFILERS` | `0` | Master switch to enable/disable all profilers |
| `TPLR_ENABLE_TIMER_PROFILER` | `0` | Enable/disable the timer profiler specifically |
| `TPLR_ENABLE_SHARD_PROFILER` | `0` | Enable/disable the shard profiler specifically |

Set the value to `1` to enable a profiler, or `0` to disable it.

## Usage Examples

### Enabling All Profilers

```bash
export TPLR_ENABLE_PROFILERS=1
export TPLR_ENABLE_TIMER_PROFILER=1
export TPLR_ENABLE_SHARD_PROFILER=1
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

## Performance Impact

When profilers are disabled via environment variables, they are replaced with dummy implementations that have negligible performance impact. This allows profiling code to remain in place without affecting production performance.