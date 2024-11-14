# Global Step Synchronization


## Introduction

This document explains how the system achieves **global step synchronization** among miners and validators. Synchronizing the `global_step` is crucial to ensure consistent training progression, learning rate scheduling, and coordinated model updates across all nodes participating in the decentralized training process.

## Motivation

Without global step synchronization, several issues can arise:

- **Learning Rate Inconsistency**: New nodes or restarted nodes may start with `global_step` at zero, causing them to be in the warm-up phase of the learning rate scheduler while others are in stable or decay phases.
- **Training Progress Discrepancies**: Nodes operating at different `global_step` values may apply updates that are out of sync, leading to conflicts and suboptimal model performance.
- **Optimization Conflicts**: Asynchronous steps can cause conflicting gradients and hinder convergence.

To address these issues, the system implements a mechanism to synchronize `global_step` across all nodes, ensuring cohesive training and optimal model updates.

## Synchronization Mechanism Overview

The synchronization of `global_step` is achieved through the following steps:

1. **Embedding `global_step` in Model Slices**: Whenever a miner or validator uploads a model slice (state or delta), they include their current `global_step` in the data.
2. **Extracting `global_step` from Received Slices**: When nodes receive slices from others, they extract the `global_step` and keep track of the maximum value.
3. **Updating Local `global_step`**: Nodes update their local `global_step` to be the maximum between their current value and the maximum `global_step` from the received slices.
4. **Adjusting Learning Rate Schedulers**: After updating `global_step`, nodes adjust their learning rate schedulers to align with the new `global_step`.

## Detailed Implementation

### Including `global_step` in Uploaded Slices

When miners and validators upload their model slices, they include the `global_step` as metadata. This is implemented in the `upload_slice_for_window` function.

```python
# In src/templar/comms.py

async def upload_slice_for_window(
    bucket: str,
    model: torch.nn.Module,
    window: int,
    seed: str,
    wallet: 'bt.wallet',
    compression: int,
    key: str = 'slice',
    global_step: int = 0
):
    filename = f'{key}-{window}-{wallet.hotkey.ss58_address}.pt'
    logger.debug(f"Uploading slice to S3: {filename}")

    # Get indices for slicing
    indices = await get_indices_for_window(model, seed, compression)

    # Create the slice data with global_step
    slice_data = {'global_step': global_step}
    for name, param in model.named_parameters():
        slice_data[name] = param.data.view(-1)[indices[name].to(model.device)].cpu()

    # Save and upload the slice_data
    # ... existing code to save and upload to S3 ...
```

### Extracting and Updating `global_step` from Received Slices

When applying slices from other nodes, the system extracts `global_step` and updates the local `global_step` accordingly.

```python
# In src/templar/comms.py

async def apply_slices_to_model(
    model: torch.nn.Module,
    window: int,
    seed: str,
    compression: int,
    key: str = 'slice'
) -> int:
    indices_dict = await get_indices_for_window(model, seed, compression)
    slice_files = await load_files_for_window(window=window, key=key)

    max_global_step = 0  # Initialize max_global_step

    # Iterate over each slice file
    for file_i in slice_files:
        try:
            slice_i = await get_slices(file_i, model.device)
            slice_global_step = slice_i.get('global_step', 0)  # Default to 0 if not present
            max_global_step = max(max_global_step, slice_global_step)

            # Apply the slice to the model
            # ... existing code to apply parameter slices ...
        except Exception as e:
            logger.exception(f"Error applying slice from {file_i}: {e}")

    # Return the maximum global_step found
    return max_global_step
```

### Updating Local `global_step` and Adjusting the Scheduler

After applying slices, nodes update their local `global_step` and adjust their learning rate schedulers to reflect the new training progress.

```python
# In neurons/miner.py or neurons/validator.py

# Apply slices and get max_global_step
max_global_step = await apply_slices_to_model(
    model=self.model,
    window=window,
    seed=window,
    compression=self.hparams.compression,
    key='state'
)

# Update local global_step
self.global_step = max(self.global_step, max_global_step)
self.scheduler.last_epoch = self.global_step - 1  # Update scheduler to match global_step
tplr.logger.info(f"Updated global step to {self.global_step}")
```

### Initializing or Loading `global_step` from Checkpoints

When nodes start or restart, they load the `global_step` from saved checkpoints if available.

```python
# In neurons/miner.py or neurons/validator.py

# Load checkpoint if it exists
if os.path.exists(self.checkpoint_path):
    tplr.logger.info(f"Loading checkpoint from {self.checkpoint_path}")
    global_step, _ = asyncio.run(load_checkpoint(
        filename=self.checkpoint_path,
        model=self.model,
        optimizer=self.optimizer,      # For miners
        scheduler=None,                # Scheduler will be initialized later
        device=self.config.device
    ))
    self.global_step = global_step
    tplr.logger.info(f"Resumed from global step {self.global_step}")
else:
    tplr.logger.info("No checkpoint found. Starting from scratch.")
    self.global_step = 0
```

### Adjusting the Learning Rate Scheduler

When initializing the learning rate scheduler, the `last_epoch` parameter is set to `self.global_step - 1` to ensure the learning rate matches the current training stage.

```python
# In neurons/miner.py or neurons/validator.py

self.scheduler = get_wsd_scheduler(
    optimizer=self.optimizer,
    num_warmup_steps=self.hparams.num_warmup_steps,
    num_stable_steps=self.hparams.num_stable_steps,
    num_decay_steps=self.hparams.num_decay_steps,
    last_epoch=self.global_step - 1  # Set to global_step - 1
)
```

### Saving Checkpoints with `global_step`

Nodes save their `global_step` along with other state information in checkpoints.

```python
# In src/templar/comms.py

async def save_checkpoint(
    filename,
    model,
    optimizer=None,
    scheduler=None,
    global_step=0,
    **kwargs
):
    checkpoint = {
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        # Include optimizer and scheduler states if available
        # ... existing code ...
    }
    # Save the checkpoint asynchronously
    await loop.run_in_executor(None, torch.save, checkpoint, filename)
```

## Handling Possible Scenarios

### New Nodes Joining the Network

- **Scenario**: A new miner or validator joins the network without prior checkpoints.
- **Handling**:
  - The node starts with `global_step = 0`.
  - Upon applying slices from other nodes, it updates its `global_step` to match the network.
  - The learning rate scheduler is adjusted accordingly.

### Node Restarts

- **Scenario**: A node restarts due to a crash or manual restart.
- **Handling**:
  - The node loads its saved `global_step` from the checkpoint.
  - After applying new slices, it updates `global_step` if higher steps are found.
  - The learning rate scheduler is realigned.

### Missing `global_step` in Slices

- **Scenario**: Some slices do not contain `global_step` (e.g., due to older software versions).
- **Handling**:
  - Slices without `global_step` default to zero.
  - The system uses the maximum `global_step` from all slices.
  - Nodes avoid regressing `global_step` to a lower value.

## Benefits of Global Step Synchronization

- **Consistent Learning Rate Scheduling**: Ensures all nodes are in the same phase (warm-up, stable, decay) of the learning rate schedule.
- **Aligned Training Progress**: Nodes update and apply model parameters coherently.
- **Improved Model Convergence**: Synchronization reduces conflicting updates and promotes efficient training.
- **Enhanced Collaboration**: Facilitates smoother integration of contributions from various nodes.

## Conclusion

By embedding `global_step` within model slices and updating the local `global_step` based on the maximum received value, the system achieves effective synchronization across miners and validators. This mechanism ensures consistent training progression, coordinated updates, and optimal performance of the decentralized model training process.

---

**Note**: For more details on checkpointing and learning rate scheduling, refer to the following documents:

- [Checkpointing in Miners and Validators](checkpointing.md)
- [Learning Rate Scheduler Implementation](../src/templar/learning_rates.py)