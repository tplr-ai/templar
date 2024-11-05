# Validator Setup

This document provides a guide on how to set up and run a validator using `validator.py`. It explains the workflow, configuration options, and step-by-step instructions to get a validator up and running. We'll also highlight important flags such as `--sync_state` that are crucial for proper synchronization and operation within the network.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Understanding `validator.py`](#understanding-validatorpy)
  - [Overview](#overview)
  - [Key Components](#key-components)
- [Setting Up a Validator](#setting-up-a-validator)
  - [Step 1: Install Dependencies](#step-1-install-dependencies)
  - [Step 2: Configure Wallet and Hotkey](#step-2-configure-wallet-and-hotkey)
  - [Step 3: Running the Validator](#step-3-running-the-validator)
  - [Example Command](#example-command)
- [Important Flags](#important-flags)
  - [`--sync_state`](#--sync_state)
- [Additional Configuration](#additional-configuration)
- [Logging and Monitoring](#logging-and-monitoring)
- [Conclusion](#conclusion)

## Introduction

The validator is a crucial component of the protocol, responsible for evaluating miners' contributions by comparing their uploaded deltas with the validator's locally computed gradients. This ensures only high-quality updates are incorporated into the global model. This guide will help you set up a validator, understand its workflow, and ensure it operates correctly within the network.

## Prerequisites

- **Python 3.8 or higher**
- **CUDA-compatible GPU** (if using GPU acceleration)
- **Installation of Required Python Packages**:
  - `bittensor`
  - `torch`
  - `transformers`
  - `numpy`
  - `wandb` (if using Weights and Biases for logging)
- **Access to the Bittensor Network** (`subtensor`)
- **Registered Wallet and Hotkey**
- **Access to S3-Compatible Storage** (e.g., AWS S3 bucket)
- **PM2 Process Manager** (if running multiple validators or for process management)

## Understanding `validator.py`

### Overview

The `validator.py` script initializes and runs a validator node that participates in the evaluation of miners' contributions. Here's what it does:

- **Synchronizes the Model State**: Downloads the latest model state slices and applies miners' deltas from the previous window.
- **Evaluation**: Computes local gradients on specific data subsets and compares them with miners' uploaded deltas.
- **Scoring**: Calculates scores (e.g., cosine similarity) to evaluate the quality of miners' updates.
- **Weight Assignment**: Sets weights on the chain based on the evaluation, influencing the aggregation of model updates.

### Key Components

- **Configuration (`config`)**: Sets up configuration parameters using `argparse` and `bittensor` utilities.
- **Wallet and Subtensor**: Initializes the wallet and connects to the subtensor (blockchain node).
- **Model Initialization**: Loads the Llama model using `transformers`.
- **Buckets**: Manages the S3 buckets used for storing and retrieving model slices and deltas.
- **Evaluation Loop**: The main asynchronous loop where evaluation and synchronization happen.
- **Event Handling**: Listens to blockchain events to synchronize blocks and windows.

## Setting Up a Validator

### Step 1: Install Dependencies

Ensure all required packages are installed:

```bash
pip install bittensor torch transformers numpy wandb
```

Alternatively, if your project has a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 2: Configure Wallet and Hotkey

You need a registered wallet and hotkey on the network:

1. **Create a Wallet**:

   ```bash
   btcli new_coldkey --name <wallet_name>
   btcli new_hotkey --name <wallet_name> --hotkey <hotkey_name>
   ```

2. **Register on the Network** (if not already registered):

   ```bash
   btcli register --wallet.name <wallet_name> --wallet.hotkey <hotkey_name> --netuid <netuid>
   ```

   Replace `<netuid>` with the desired network UID (e.g., `223` for test networks).

### Step 3: Running the Validator

Run the validator script using Python, providing the necessary arguments.

#### Example Command

Using `start.sh` as inspiration:

```bash
python3 neurons/validator.py \
  --actual_batch_size 1 \
  --wallet.name <wallet_name> \
  --wallet.hotkey <hotkey_name> \
  --bucket <bucket_name> \
  --device cuda:0 \
  --use_wandb \
  --project <wandb_project_name> \
  --netuid 223 \
  --sync_state
```

Replace placeholders with your specific values:

- `<wallet_name>`: Your wallet name.
- `<hotkey_name>`: Your hotkey name.
- `<bucket_name>`: The S3 bucket name you'll use.
- `<wandb_project_name>`: Your Weights and Biases project name.

### Running with PM2 (Optional)

If you plan to manage your validator processes using PM2:

```bash
pm2 start neurons/validator.py --interpreter python3 --name Validator1 -- \
  --actual_batch_size 1 \
  --wallet.name <wallet_name> \
  --wallet.hotkey <hotkey_name> \
  --bucket <bucket_name> \
  --device cuda:0 \
  --use_wandb \
  --project <wandb_project_name> \
  --netuid 223 \
  --sync_state
```

## Important Flags

### `--sync_state`

- **Usage**: `--sync_state`
- **Description**: When set, the validator synchronizes the model state by pulling from the history of uploaded states and deltas. This ensures your validator's model is up-to-date with the global state and can correctly evaluate miners' contributions.
- **Default**: `False`

**Note**: It's important to use `--sync_state` to ensure your validator operates correctly within the network.

## Additional Configuration

- **Project Name (`--project`)**: Specify the Weights and Biases project name if you're using WandB for logging.
- **Device (`--device`)**: Set to `cuda` or `cuda:<gpu_id>` to use GPU acceleration.
- **Batch Size (`--actual_batch_size`)**: Determines the batch size for evaluation. Ensure it's appropriate for your GPU memory.
- **Network UID (`--netuid`)**: Specify the network UID you are connecting to (e.g., `223` for test networks).
- **Debugging Flags**:
  - `--debug`: Enables debug-level logging.
  - `--trace`: Enables trace-level logging (very verbose).
- **Automatic Updates (`--autoupdate`)**: Enables automatic updates of the validator script.

## Logging and Monitoring

- **Weights and Biases (WandB)**:
  - If `--use_wandb` is set, the validator will log metrics to WandB.
  - Log in to WandB using `wandb login` before running the validator.
- **Console Logs**:
  - The validator outputs logs to the console, including synchronization status, evaluation progress, and any errors.
  - Monitor these logs to ensure your validator is operating correctly.
- **PM2 Logs** (if using PM2):
  - Use `pm2 logs <name>` to view real-time logs.
  - Use `pm2 monit` for monitoring CPU and memory usage.

## Conclusion

By following this guide, you should be able to set up and run a validator that participates in the network's protocol by evaluating miners' contributions. Remember to use the `--sync_state` flag to ensure your validator stays synchronized with the global model state and performs accurate evaluations.

Feel free to customize the configurations and explore additional flags as needed. For any issues or further customization, refer to the `validator.py` code and adjust the configurations accordingly.

---

**Helpful Tips**:

- **Synchronization**: The initial synchronization might take some time, especially if the model state is large. Ensure your network connection is stable.
- **Testing**: Use the `--test` flag if you're connecting to a test network.
- **Automatic Updates**: The `--autoupdate` flag enables automatic updates of the validator script.
