# Local Model Evaluator

The Local Model Evaluator (`scripts/evaluator-local.py`) is a utility script that allows you to evaluate model checkpoints offline without connecting to the Bittensor network or Templar chain. This is useful for testing checkpoints locally, benchmarking models during development, or validating saved checkpoints before using them in production (through model_converter).

## Overview

Unlike the main evaluator script which pulls checkpoints from the network, this local evaluator:

- Accepts checkpoint files directly from your local filesystem
- Runs standardized benchmark tasks using the `lm-eval` library
- Provides immediate feedback on model performance
- Requires no network connection or chain registration

## Prerequisites

Before using the local evaluator, ensure you have:

1. A valid model checkpoint file (`.pt` format)
2. The `lm-eval` library installed: `pip install lm-eval`
3. CUDA-enabled GPU (optional but recommended)
4. Sufficient disk space for temporary model files

## Installation

The evaluator is part of the Templar repository. No additional installation is required beyond the standard Templar setup:

```bash
git clone <templar-repo>
cd templar
uv pip install -e ".[dev]"
```

## Usage

### Basic Usage

Evaluate a checkpoint using the default tasks:

```bash
python scripts/evaluator-local.py --checkpoint_path path/to/checkpoint.pt
```

### Custom Tasks

Specify which benchmark tasks to run:

```bash
python scripts/evaluator-local.py \
    --checkpoint_path checkpoint.pt \
    --tasks arc_easy,winogrande,piqa
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint_path` | Required | Path to the model checkpoint file |
| `--device` | cuda | Device to use for evaluation (cuda/cpu) |
| `--tasks` | arc_challenge,arc_easy,openbookqa,winogrande,piqa,hellaswag | Comma-separated list of evaluation tasks |
| `--batch_size` | 8 | Evaluation batch size |
| `--limit` | 1.0 | Fraction of dataset to evaluate (0.0-1.0) |
| `--num_fewshot` | 0 | Number of few-shot examples |
| `--output_dir` | evaluation_results | Directory to save results |
| `--cleanup` | False | Clean up model files after evaluation |

### Available Tasks

The evaluator supports all tasks available in `lm-eval`. Common options include:

- `arc_challenge`: AI2 Reasoning Challenge (challenge set)
- `arc_easy`: AI2 Reasoning Challenge (easy set)
- `openbookqa`: Open Book Question Answering
- `winogrande`: Commonsense reasoning
- `piqa`: Physical Interaction QA
- `hellaswag`: Commonsense natural language inference
- `truthfulqa_mc`: TruthfulQA multiple choice
- `mmlu`: Massive Multitask Language Understanding

## Step-by-Step Example

### 1. Download or Prepare a Checkpoint

First, obtain a checkpoint file. You can either:

- Download from cloud storage:

```bash
rclone copy templar-validator:/80f15715bb0b882c9e967c13e677ed7d/checkpoint-792063-1-v0.2.81.pt ./checkpoints/
```

- Or copy from a local validator:

```bash
cp /path/to/validator/checkpoint.pt ./checkpoints/
```

### 2. Run the Evaluation

Execute the evaluator with your desired configuration:

```bash
python scripts/evaluator-local.py \
    --checkpoint_path ./checkpoints/checkpoint-792063-1-v0.2.81.pt \
    --tasks arc_easy \
    --device cuda
```

### 3. Monitor Progress

The script will display progress as it:

1. Loads the checkpoint
2. Saves the model in HuggingFace format
3. Runs the lm-eval benchmark
4. Processes results

### 4. Review Results

After completion, you'll see output like:

```text
hf (pretrained=models/eval,tokenizer=models/eval), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 8
| Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|--------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_easy|      1|none  |     0|acc     |↑  |0.5922|±  |0.0101|
|        |       |none  |     0|acc_norm|↑  |0.5236|±  |0.0102|

[13:48:23] Model files kept at: models/eval

==================================================
EVALUATION RESULTS
==================================================

Runtime: 48.35 seconds
Config: {
  "checkpoint_path": "../experimental/checkpoint-792063-1-v0.2.81.pt/checkpoint-792063-1-v0.2.81.pt",
  "device": "cuda",
  "tasks": "arc_easy",
  "batch_size": 8,
  "limit": 1.0,
  "num_fewshot": 0,
  "output_dir": "evaluation_results",
  "cleanup": false
}

Task Scores:
------------------------------
arc_easy (acc_norm,none): 0.5236
==================================================

Results saved to evaluation_results/evaluation_summary.json
```

## Output Files

The evaluator creates several output files:

1. **Summary JSON**: `evaluation_results/evaluation_summary.json`
   - Contains runtime, configuration, and aggregated scores
   - Useful for programmatic analysis

2. **Detailed Results**: `evaluation_results/models__eval/<timestamp>.json`
   - Complete evaluation metrics from lm-eval
   - Includes per-task breakdowns and statistics

3. **Model Files**: `models/eval/` (unless `--cleanup` is used)
   - Temporary HuggingFace format model files
   - Can be reused for further evaluation

## Advanced Usage

### Quick Evaluation with Limited Samples

For faster testing, evaluate on a subset of the data:

```bash
python scripts/evaluator-local.py \
    --checkpoint_path checkpoint.pt \
    --limit 0.1 \
    --batch_size 16
```

### Multi-Task Evaluation

Run multiple benchmarks in one command:

```bash
python scripts/evaluator-local.py \
    --checkpoint_path checkpoint.pt \
    --tasks arc_challenge,arc_easy,winogrande,piqa,hellaswag \
    --output_dir comprehensive_eval
```

### Few-Shot Evaluation

Test few-shot learning capabilities:

```bash
python scripts/evaluator-local.py \
    --checkpoint_path checkpoint.pt \
    --tasks arc_easy \
    --num_fewshot 5
```

### CPU-Only Evaluation

For systems without CUDA:

```bash
python scripts/evaluator-local.py \
    --checkpoint_path checkpoint.pt \
    --device cpu \
    --batch_size 4
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `--batch_size`
   - Use `--device cpu` for CPU evaluation
   - Enable `--cleanup` to free resources

2. **Missing Dependencies**

   ```bash
   pip install lm-eval transformers torch
   ```

3. **Checkpoint Loading Errors**
   - Ensure checkpoint is compatible with the model architecture
   - Check that the file path is correct and accessible

4. **Slow Evaluation**
   - Use `--limit 0.1` for quick tests
   - Increase `--batch_size` if GPU memory allows
   - Select fewer tasks

### Error Messages

- `FileNotFoundError: Checkpoint not found`: Verify the checkpoint path
- `RuntimeError: Evaluation failed`: Check lm-eval installation and task names
- `CUDA out of memory`: Reduce batch size or use CPU

## Performance Tips

1. **GPU Optimization**
   - Use the largest batch size that fits in memory
   - Ensure CUDA is properly configured

2. **Quick Testing**
   - Start with `--limit 0.1` to verify setup
   - Run single tasks before comprehensive evaluation

3. **Resource Management**
   - Use `--cleanup` to remove temporary files
   - Monitor GPU memory during evaluation

## Integration with CI/CD

The local evaluator can be integrated into automated workflows:

```bash
#!/bin/bash
# CI evaluation script

CHECKPOINT=$1
THRESHOLD=0.5

# Run evaluation
python scripts/evaluator-local.py \
    --checkpoint_path "$CHECKPOINT" \
    --tasks arc_easy \
    --output_dir ci_results \
    --cleanup

# Check results
SCORE=$(jq '.results.arc_easy."acc_norm,none"' ci_results/evaluation_summary.json)
if (( $(echo "$SCORE < $THRESHOLD" | bc -l) )); then
    echo "Model performance below threshold!"
    exit 1
fi
```

## Best Practices

1. **Version Control**: Track evaluation results alongside model versions
2. **Consistent Settings**: Use the same batch size and device for comparable results
3. **Regular Testing**: Evaluate checkpoints periodically during training
4. **Documentation**: Record evaluation settings with results

## See Also

- [Evaluator Documentation](evaluator.md) - Network-based model evaluation
- [Model Converter](model_converter.md) - Convert between checkpoint formats
- [Miner Setup](miner.md) - Generate checkpoints for evaluation
