# Templar Metrics Flux Queries

This directory contains Flux queries used in the Templar Grafana dashboards. The queries are organized by measurement type and metric name.

## Metric Prefixes

Metrics in the system use prefixes to identify the source component:

- `M` prefix: Miner metrics (from `neurons/miner.py`)
- `V` prefix: Validator metrics (from `neurons/validator.py`)
- `E` prefix: Evaluator metrics (from `scripts/evaluator.py`)

## Miner Metrics

Miner metrics are typically prefixed with `M` and include:

- `Mtraining_step_active_peers.flux`: Active peers count
- `Mtraining_step_gather_peers.flux`: List of gathered peers
- `Mtraining_step_batch_size.flux`: Batch size metric
- `Mtraining_step_batch_time.flux`: Time taken per batch
- `Mtraining_step_grad_step_time.flux`: Time for gradient steps
- `Mtraining_step_loss.flux`: Training loss
- `Mtraining_step_optimizer_state_size.flux`: Size of optimizer state
- `Mtraining_step_tokens_per_sec.flux`: Training throughput

## Validator Metrics

Validator metrics are typically prefixed with `V` and include:

- `Vvalidator_window_loss_comparison.flux`: Loss before/after gradient application
- `Vvalidator_window_improvement.flux`: Loss improvement metrics
- `Vvalidator_operation_times.flux`: Operation timing metrics
- `Vvalidator_slashing.flux`: Validator slashing activity
- `Vtemplar_metrics_v2_weight.flux`: Peer weights
- `Vtemplar_metrics_v2_binary_indicators.flux`: Binary indicators for validation

## Evaluator Metrics

Evaluator metrics are typically prefixed with `E` and include:

- `Ebenchmark_task_score.flux`: Benchmark task scores
- `Ebenchmark_metrics_runtime.flux`: Benchmark runtime metrics

## System Metrics

System metrics are collected from all components:

- `cpu_usage.flux`: CPU utilization (sys_cpu_usage or cpu_usage)
- `memory_usage.flux`: Memory usage metrics (sys_mem_used, mem_used, etc.)
- `gpu_utilization.flux`: GPU utilization percentage
- `gpu_memory_usage.flux`: GPU memory usage metrics (allocated, cached, total)
- `gpu_mem_segments.flux`: GPU memory segments count

## Usage

These queries can be used as a reference for creating new dashboards or troubleshooting existing metrics. When adding new metrics, follow the same naming convention with appropriate component prefixes.

Most queries follow a standard pattern with:
1. Source bucket selection
2. Time range filtering
3. Measurement filtering with appropriate prefix
4. Field filtering for specific metrics
5. Tag filtering for specific components (uid, role, etc.)
6. Grouping by relevant dimensions
7. Aggregation over time windows

## Important Notes

When working with numeric vs string fields:
- Use `mean()` or other numeric aggregations for numeric fields (like `active_peers`, `loss`, etc.)
- Use `last()` for string fields (like `gather_peers`) that cannot be aggregated numerically
- String fields are often stored as JSON strings and require special handling

For testing queries outside of Grafana:
- Replace `v.timeRangeStart` with a relative time like `-12h`
- Replace `v.timeRangeStop` with `now()`
- Replace `v.windowPeriod` with a specific interval like `5m`
- Replace Grafana variables with concrete values or remove those filter conditions
