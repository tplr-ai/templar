# Miner Setup

This document provides a guide on how to set up and run a miner using `miner.py`. It explains the workflow, configuration options, and step-by-step instructions to get a miner up and running. Additionally, it highlights important flags such as `--remote` and `--sync_state` that are crucial for proper synchronization and operation within the network.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Understanding `miner.py`](#understanding-minerpy)
  - [Overview](#overview)
  - [Key Components](#key-components)
- [Setting Up a Miner](#setting-up-a-miner)
  - [Step 1: Install Dependencies](#step-1-install-dependencies)
  - [Step 2: Configure Wallet and Hotkey](#step-2-configure-wallet-and-hotkey)
  - [Step 3: Running the Miner](#step-3-running-the-miner)
  - [Example Command](#example-command)
- [Important Flags](#important-flags)
  - [`--remote`](#--remote)
  - [`--sync_state`](#--sync_state)
- [Additional Configuration](#additional-configuration)
- [Logging and Monitoring](#logging-and-monitoring)
- [Conclusion](#conclusion)

## Introduction

The miner is a crucial component of the protocol, responsible for training the model on designated data subsets and uploading updates to be integrated into the global model. By following this guide, you'll learn how to set up a miner, understand its workflow, and ensure it operates correctly within the network.

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
- **PM2 Process Manager** (if running multiple miners or for process management)

## Understanding `miner.py`

### Overview

The `miner.py` script initializes and runs a miner node that participates in collaborative model training. Here's what it does:

- **Synchronizes the Model State**: Downloads the latest model state slices from other miners.
- **Training**: Trains the model on assigned data subsets.
- **Computes and Uploads Deltas**: Calculates the changes (deltas) in the model parameters and uploads them.
- **Progresses Through Windows**: Operates in synchronized windows determined by the blockchain.
- **Handles Network Interactions**: Manages connections to the Bittensor network, including the blockchain and other miners.

### Key Components

- **Configuration (`config`)**: Sets up configuration parameters using `argparse` and `bittensor` utilities.
- **Wallet and Subtensor**: Initializes the wallet and connects to the subtensor (blockchain node).
- **Model Initialization**: Loads the Llama model using `transformers`.
- **Optimizer and Scheduler**: Sets up the optimizer (`AdamW`) and learning rate scheduler (`CosineAnnealingLR`).
- **Buckets**: Manages the S3 buckets used for storing and retrieving model slices.
- **Training Loop**: The main asynchronous loop where training and synchronization happen.
- **Event Handling**: Listens to blockchain events to synchronize blocks and windows.

## Setting Up a Miner

### Step 1: Install Dependencies

Ensure all required packages are installed. Use `pip` to install the necessary Python packages:

```bash
pip install bittensor torch transformers numpy wandb
```

Alternatively, if your project has a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 2: Configure Wallet and Hotkey

Before running the miner, you need a registered wallet and hotkey on the network:

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

### Step 3: Running the Miner

You can run the miner script using Python, providing the necessary arguments.

#### Example Command

Using the `start.sh` script as inspiration, here's how to run a single miner:

```bash
python3 neurons/miner.py \
  --actual_batch_size 1 \
  --wallet.name <wallet_name> \
  --wallet.hotkey <hotkey_name> \
  --bucket <bucket_name> \
  --device cuda:0 \
  --use_wandb \
  --project <wandb_project_name> \
  --netuid 223 \
  --remote \
  --sync_state
```

Replace placeholders with your specific values:

- `<wallet_name>`: Your wallet name.
- `<hotkey_name>`: Your hotkey name.
- `<bucket_name>`: The S3 bucket name you'll use.
- `<wandb_project_name>`: Your Weights and Biases project name.

### Running with PM2 (Optional)

If you plan to manage your miner processes using PM2 (as in `start.sh`), you can start the miner like this:

```bash
pm2 start neurons/miner.py --interpreter python3 --name Miner1 -- \
  --actual_batch_size 1 \
  --wallet.name <wallet_name> \
  --wallet.hotkey <hotkey_name> \
  --bucket <bucket_name> \
  --device cuda:0 \
  --use_wandb \
  --project <wandb_project_name> \
  --netuid 223 \
  --remote \
  --sync_state
```

## Important Flags

### `--remote`

- **Usage**: `--remote`
- **Description**: Enables the miner to connect to other miners' buckets. This allows your miner to download updates (model slices) from other miners, ensuring better synchronization and collaboration.
- **Default**: `False` (If not specified, the miner will only connect to its own bucket)

### `--sync_state`

- **Usage**: `--sync_state`
- **Description**: When set, the miner synchronizes the model state by pulling from the history of uploaded states. This ensures your miner's model is up-to-date with the global state and is crucial when joining an existing network.
- **Default**: `False`

**Note**: It's important to use both `--remote` and `--sync_state` to effectively participate in the collaborative training and maintain synchronization with other miners.

## Additional Configuration

- **Project Name (`--project`)**: Specify the Weights and Biases project name if you're using WandB for logging.
- **Device (`--device`)**: Set to `cuda` or `cuda:<gpu_id>` to use GPU acceleration.
- **Batch Size (`--actual_batch_size`)**: Determines the batch size for training. Ensure it's appropriate for your GPU memory.
- **Network UID (`--netuid`)**: Specify the network UID you are connecting to (e.g., `223` for test networks).
- **Debugging Flags**:
  - `--debug`: Enables debug-level logging.
  - `--trace`: Enables trace-level logging (very verbose).
- **Random Training Data (`--random`)**: If set, the miner trains on random data subsets.
- **Baseline Mode (`--baseline`)**: If set, the miner will not perform synchronization with other peers and will train independently.

## Logging and Monitoring

- **Weights and Biases (WandB)**:
  - If `--use_wandb` is set, the miner will log metrics to WandB.
  - Log in to WandB using `wandb login` before running the miner.
- **Console Logs**:
  - The miner outputs logs to the console, including synchronization status, training progress, and any errors.
  - Monitor these logs to ensure your miner is operating correctly.
- **PM2 Logs** (if using PM2):
  - Use `pm2 logs <name>` to view real-time logs.
  - Use `pm2 monit` for monitoring CPU and memory usage.

## Conclusion

By following this guide, you should be able to set up and run a miner that participates in the network's collaborative training protocol. Remember to use the `--remote` and `--sync_state` flags to ensure your miner stays synchronized with the global model state and contributes effectively.

Feel free to customize the configurations and explore additional flags as needed. For any issues or further customization, refer to the `miner.py` code and adjust the configurations accordingly.

---

**Helpful Tips**:

- **Synchronization**: The initial synchronization might take some time, especially if the model state is large. Be patient and ensure your network connection is stable.
- **Testing**: Use the `--test` flag if you're connecting to a test network.
- **Automatic Updates**: The `--autoupdate` flag enables automatic updates of the miner script. Use it if you want your miner to stay up-to-date with the latest code changes.
