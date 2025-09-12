# Templar Model Evaluator

The Templar Model Evaluator is an autonomous service that continuously evaluates model checkpoints using standardized benchmark tasks. It monitors for new checkpoints, automatically downloads them, runs comprehensive evaluations, and logs results to metrics systems.

## Overview

The evaluator script (`scripts/evaluator.py`) performs:
- Automatic checkpoint detection by window number
- Multi-task model evaluation using lm-eval
- Distributed metrics logging to InfluxDB
- Resource management and cleanup
- Service-oriented design for continuous operation

## Features

- **Automated Evaluation**: Continuously monitors for new model checkpoints
- **Benchmark Suite**: Runs multiple evaluation tasks (ARC, Winogrande, MMLU, etc.)
- **Metrics Logging**: Reports results to InfluxDB for monitoring
- **Resource Management**: Handles GPU memory and temporary file cleanup
- **Configurable Intervals**: Customizable evaluation frequency
- **Multi-GPU Support**: Can run on specified GPU devices

## Prerequisites

### System Requirements

- Registered Bittensor wallet
- **NVIDIA GPU with CUDA support (H200 required - 141GB VRAM minimum)**
- Python 3.8+
- uv package manager
- lm-eval benchmark tool installed
- Access to model checkpoints on the network
- **CPU**: 32 cores, 3.5 GHz minimum
- **RAM**: 800 GB minimum
- **Network**: 1024 Mbps download/upload bandwidth minimum

### Environment Variables

The evaluator requires:
- Standard Templar environment configuration
- InfluxDB token (optional, uses default if not provided)

## Installation

### Option 1: Docker (Recommended)

Run the evaluator using Docker Compose:

1. **Configure environment variables** in your `.env` file:
   ```bash
   # Required variables
   WALLET_NAME=your_wallet_name
   WALLET_HOTKEY=your_hotkey
   HF_TOKEN=your_huggingface_token  # Required for tokenizer access
   WANDB_API_KEY=your_wandb_key
   NETUID=268  # or your target netuid
   
   # R2 Dataset Access (required)
   R2_DATASET_ACCOUNT_ID=your_account_id
   R2_DATASET_BUCKET_NAME=your_bucket_name
   R2_DATASET_READ_ACCESS_KEY_ID=your_access_key
   R2_DATASET_READ_SECRET_ACCESS_KEY=your_secret_key
   
   # Optional evaluator-specific variables
   EVALUATOR_GPU_IDS="4,5,6,7"  # GPU device IDs (default: 4,5,6,7)
   INFLUXDB_TOKEN=your_influxdb_token  # For metrics logging
   CUSTOM_EVAL_PATH=eval_dataset  # Custom evaluation dataset path
   EVAL_INTERVAL=600  # Seconds between evaluations (default: 10 min)
   EVAL_BATCH_SIZE=8  # Batch size for evaluation
   EVAL_TASKS=arc_challenge,arc_easy,openbookqa,winogrande,piqa,hellaswag,mmlu
   ```

2. **Start the evaluator service**:
   ```bash
   docker compose -f docker/compose.yml up -d evaluator
   ```

3. **Monitor logs**:
   ```bash
   docker compose -f docker/compose.yml logs -f evaluator
   ```

### Option 2: Local Installation

1. **Install dependencies**:
   ```bash
   # Install uv if not already installed
   pip install uv
   
   # Install project dependencies
   uv sync
   ```

2. **Install lm-eval**:
   ```bash
   pip install lm-eval
   ```

3. **Configure environment** following the standard Templar setup

## Usage

### Docker Usage

When using Docker, the evaluator runs automatically based on your configuration. To modify behavior:

```bash
# Stop the evaluator
docker compose -f docker/compose.yml stop evaluator

# Update .env file with new configuration
# Then restart
docker compose -f docker/compose.yml up -d evaluator
```

### Local Usage

#### Basic Execution

Run the evaluator with default settings:
```bash
uv run ./scripts/evaluator.py
```

#### Custom Configuration

Run with specific parameters:
```bash
uv run scripts/evaluator.py \
  --netuid 3 \
  --device cuda:0 \
  --tasks "arc_challenge,winogrande,piqa" \
  --eval_interval 300 \
  --actual_batch_size 16
```

#### Custom Evaluation Dataset

Run with a custom evaluation dataset for loss and perplexity calculation:
```bash
uv run scripts/evaluator.py \
  --netuid 3 \
  --device cuda:0 \
  --custom_eval_path eval_dataset \
  --eval_interval 600
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--netuid` | 3 | Bittensor network UID |
| `--device` | cuda:7 | GPU device to use for evaluation |
| `--tasks` | Multiple tasks | Comma-separated list of evaluation tasks |
| `--eval_interval` | 600 | Seconds between evaluation checks (10 minutes) |
| `--actual_batch_size` | 8 | Batch size for evaluation |
| `--checkpoint_path` | checkpoints/ | Directory for checkpoint storage |
| `--uid` | None | Override the wallet's UID |
| `--skip-gaps` | False | Skip gaps in the evaluation process |
| `--custom_eval_path` | None | Relative path to custom evaluation dataset bins for loss and perplexity calculation |

### Evaluation Tasks

Default evaluation tasks include:
- `arc_challenge` - AI2 Reasoning Challenge
- `arc_easy` - AI2 Reasoning Challenge (Easy)
- `openbookqa` - Open Book Question Answering
- `winogrande` - Winograd Schema Challenge
- `piqa` - Physical Interaction QA
- `hellaswag` - Commonsense NLI
- `mmlu` - Massive Multitask Language Understanding

## Architecture

### Evaluation Flow

1. **Checkpoint Detection**: Monitors blockchain for new checkpoints by window number
2. **Model Loading**: Downloads and loads checkpoint when new version is detected
3. **Benchmark Execution**: Runs lm-eval benchmark suite
4. **Results Processing**: Parses benchmark results and extracts metrics
5. **Metrics Logging**: Sends results to InfluxDB
6. **Cleanup**: Removes temporary files and frees GPU memory
7. **Wait**: Sleeps until next evaluation interval

### Key Components

- **Evaluator Class**: Main orchestrator for the evaluation process
- **CommsClass**: Handles network communication and checkpoint retrieval
- **MetricsLogger**: Manages InfluxDB metric submission
- **LM-Eval Integration**: Executes standardized language model benchmarks

### Special Handling

**MMLU Task**: The evaluator runs MMLU (a computationally intensive task) only every 4th evaluation cycle to balance resource usage. It uses:
- Reduced dataset sampling (15% limit)
- 5-shot evaluation
- BFloat16 precision for efficiency

## Monitoring

### Metrics Reported

The evaluator reports the following metrics to InfluxDB:

1. **Benchmark Metrics**:
   - `lm_eval_exit_code`: Success/failure of evaluation
   - `benchmark_runtime_s`: Time taken for evaluation

2. **Task Scores**:
   - Individual scores for each evaluation task
   - Metric types: `acc_norm` (normalized accuracy) or `acc` (accuracy)

3. **Summary Information**:
   - `num_tasks`: Number of tasks evaluated
   - `global_step`: Training step of the checkpoint
   - `window`: Window number of the checkpoint
   - `block`: Block number when evaluated

### Viewing Metrics

Access metrics through your InfluxDB dashboard or Grafana visualization to track:
- Model performance over time
- Evaluation success rates
- Task-specific improvements
- System performance

## Troubleshooting

### Common Issues

1. **No Checkpoints Found**:
   - Verify network connectivity
   - Check wallet registration
   - Ensure correct netuid

2. **GPU Memory Errors**:
   - Reduce batch size
   - Ensure GPU has sufficient memory
   - Check for other processes using GPU

3. **Evaluation Failures**:
   - Verify lm-eval installation
   - Check task names are valid
   - Ensure model files are properly saved

4. **Metrics Not Appearing**:
   - Verify InfluxDB token
   - Check network connectivity to InfluxDB
   - Review logs for connection errors

### Debug Mode

Enable detailed logging:
```bash
export RUST_LOG=debug
uv run scripts/evaluator.py
```

## Best Practices

1. **Resource Management**:
   - Monitor GPU memory usage
   - Clean up temporary files regularly
   - Use appropriate batch sizes

2. **Scheduling**:
   - Set evaluation intervals based on checkpoint frequency
   - Consider system resources when choosing intervals
   - Use dedicated GPU for evaluations

3. **Task Selection**:
   - Choose tasks relevant to your use case
   - Balance evaluation comprehensiveness with runtime
   - Consider computational cost of each task

4. **Monitoring**:
   - Set up alerts for evaluation failures
   - Track metrics trends over time
   - Monitor system resource usage

## Integration with Templar

The evaluator integrates with the Templar ecosystem by:
- Using the same wallet system as miners/validators
- Accessing checkpoints from the distributed network
- Reporting metrics to the shared monitoring infrastructure
- Following the same version and configuration patterns

## Development

### Extending Evaluations

To add new evaluation tasks:
1. Verify task is supported by lm-eval
2. Add task name to the `--tasks` parameter
3. Ensure proper metric extraction in `_process_results()`

### Custom Metrics

To report additional metrics:
1. Extend the `MetricsLogger` calls
2. Add new fields to the measurement dictionaries
3. Update InfluxDB schema if needed

## Advanced Configuration

### Running Multiple Evaluators

To run evaluators on different tasks or intervals:
1. Use different UIDs for each instance
2. Configure unique task sets
3. Set different evaluation intervals
4. Assign to different GPUs

### Integration with CI/CD

The evaluator can be integrated into continuous integration pipelines:
1. Set up as a scheduled job
2. Report results to monitoring dashboards
3. Trigger alerts on performance regressions

## Additional Resources

- [LM-Eval Documentation](https://github.com/EleutherAI/lm-evaluation-harness)
- [InfluxDB Documentation](https://docs.influxdata.com/)
- [Templar Miner Setup](./miner.md)
- [Templar Validator Setup](./validator.md)
