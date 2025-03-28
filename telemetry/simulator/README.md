# Metrics Simulator

This directory contains tools for simulating metrics data for testing and development of the telemetry system.

## Miner Metrics Simulator

The `miner_metrics_simulator.py` script simulates metrics for a Templar miner node and logs them to InfluxDB.

### Features

- Generates realistic miner metrics based on patterns from actual miner operation
- Simulates training windows with appropriate metrics progression
- Logs metrics to InfluxDB for Grafana visualization
- Configurable interval and window duration

### Usage

```bash
# Set InfluxDB connection details (or use defaults)
export INFLUXDB_HOST=localhost
export INFLUXDB_PORT=8086
export INFLUXDB_ORG=tplr
export INFLUXDB_BUCKET=tplr
export INFLUXDB_TOKEN=your_influxdb_token

# Run the simulator
./miner_metrics_simulator.py
```

### Configuration

The simulator can be configured through environment variables:

- `INFLUXDB_HOST`: InfluxDB host (default: localhost)
- `INFLUXDB_PORT`: InfluxDB port (default: 8086)
- `INFLUXDB_ORG`: InfluxDB organization (default: tplr)
- `INFLUXDB_BUCKET`: InfluxDB bucket (default: tplr)
- `INFLUXDB_TOKEN`: Authentication token for InfluxDB (required)

### Metrics Generated

The simulator generates a wide range of metrics similar to those in the actual miner, including:

- Training metrics (loss, tokens per second, batch time, etc.)
- Gradient and weight statistics
- Network and peer information
- Timing breakdowns for various operations
- System resource usage (CPU, memory, GPU)
- Miner rewards

### Development

To add new simulated metrics or modify existing ones, edit the `generate_miner_metrics()` and 
`add_system_metrics()` methods in the `MetricsSimulator` class.