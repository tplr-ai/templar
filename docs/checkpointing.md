# Checkpointing in Miners and Validators

This document explains the checkpointing mechanism used in both miners and validators, highlighting the process and key differences between them.

## Overview

Checkpointing is a crucial feature that allows miners and validators to save their current state periodically. This enables them to resume operations seamlessly after interruptions, such as crashes or restarts, without losing significant progress.

- **Miners** save their training state, including the model parameters, optimizer state, scheduler state, and current training step.
- **Validators** save their evaluation state, including the model parameters, evaluation scores, weights, and current evaluation step.

## Asynchronous Checkpoint Saving

Both miners and validators utilize asynchronous checkpoint saving to prevent blocking the main training or evaluation loops. By saving checkpoints asynchronously, the processes can continue their operations without waiting for the checkpoint to be saved, enhancing performance and efficiency.

### Key Features

- **Non-Blocking**: Checkpoint saving runs in the background, allowing the main loop to proceed without delays.
- **Regular Intervals**:
  - **Miners**: Save checkpoints every **500 training steps** (`global_step`).
  - **Validators**: Save checkpoints every **500 blocks** based on the blockchain's block number.

## Checkpointing in Miners

### What is Saved

- **Model State**: The current state of the model's parameters.
- **Optimizer State**: State of the optimizer to resume training seamlessly.
- **Scheduler State**: If a learning rate scheduler is used.
- **Global Step**: The current training step (`global_step`).
- **Additional State**: Any other variables necessary for training.

### Saving Mechanism

- Checkpoints are saved asynchronously every 500 training steps.
- The saving process uses asynchronous tasks to offload the I/O operations.
- Default checkpoint file is `checkpoint-M<UID>.pth`.

### Restoring from Checkpoint

- On startup, the miner checks for the existence of the checkpoint file.
- If found, it loads the saved states and resumes training from the saved `global_step`.
- No additional flags are required for checkpoint loading.

## Checkpointing in Validators

### What is Saved

- **Model State**: The current state of the model's parameters.
- **Global Step**: The current evaluation step (`global_step`).
- **Scores**: Evaluation scores for different miners.
- **Weights**: Assigned weights based on evaluation.
- **Additional State**: Any other variables necessary for evaluation.

### Saving Mechanism

- Checkpoints are saved asynchronously every 500 blocks.
- The checkpointing is triggered based on the blockchain's block number.
- Uses asynchronous tasks to prevent blocking the evaluation loop.
- Default checkpoint file is `checkpoint-V<UID>.pth`.

### Restoring from Checkpoint

- On startup, the validator checks for the existence of the checkpoint file.
- If found, it loads the saved states and resumes evaluation from the saved `global_step`.
- No additional flags are required for checkpoint loading.

## Differences Between Miner and Validator Checkpoints

| Aspect               | Miner                                                | Validator                                              |
|----------------------|------------------------------------------------------|--------------------------------------------------------|
| **Saving Frequency** | Every 500 **training steps** (`global_step`)         | Every 500 **blocks** (blockchain block number)         |
| **Trigger Condition**| `global_step % 500 == 0`                             | `current_block % 500 == 0`                             |
| **Saved States**     | Model state, optimizer, scheduler, global step, etc. | Model state, global step, scores, weights, etc.        |
| **Checkpoint File**  | `miner_checkpoint.pth`                               | `validator_checkpoint.pth`                             |
| **Restoration**      | Resumes training from saved `global_step`            | Resumes evaluation from saved `global_step`            |

## Configuration

### Setting the Checkpoint Path

By default, the checkpoint files are saved with the names `checkpoint-M<UID>.pth` and `checkpoint-V<UID>.pth`. You can customize the checkpoint path using the `--checkpoint_path` argument when running the miner or validator.

**Example**:

```bash
# For Miner
python neurons/miner.py --checkpoint_path /path/to/custom_miner_checkpoint.pth

# For Validator
python neurons/validator.py --checkpoint_path /path/to/custom_validator_checkpoint.pth
```

### Ensure Write Permissions

Make sure that the process has read and write permissions to the directory where the checkpoint files are stored.

## Best Practices

- **Regular Monitoring**: Check logs to ensure that checkpoints are being saved and loaded correctly.
- **Avoid Overwriting**: Ensure that `global_step` is not being unintentionally reset after loading from a checkpoint.
- **Backup Checkpoints**: Periodically back up checkpoint files to prevent data loss.
- **Consistent Paths**: Use consistent checkpoint paths when running multiple processes to avoid confusion.

## Troubleshooting

- **Checkpoint Not Saving**:
  - Verify that the checkpoint path is correct.
  - Ensure the process has write permissions to the checkpoint location.
  - Check for any errors in the logs during the checkpoint saving steps.
- **Global Step Reset to Zero**:
  - Check that `global_step` is not being reinitialized after loading the checkpoint.
  - Remove any code that sets `global_step = 0` after loading.
- **Checkpoint Not Loading**:
  - Ensure the checkpoint file exists at the specified path.
  - Verify that the file is not corrupted.
  - Check logs for any exceptions during the loading process.
- **Asynchronous Saving Issues**:
  - Ensure that the event loop is running correctly.
  - Check for exceptions in the asynchronous tasks.

## Conclusion

Checkpointing is essential for maintaining continuity in mining and validation operations. By understanding the differences and properly configuring your setup, you can ensure efficient and reliable performance of your miner and validator nodes.
