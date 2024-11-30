
# Miner Setup

This document provides a guide on how to set up and run a miner using `miner.py`. It explains the workflow, configuration options, and step-by-step instructions to get a miner up and running.

## Table of Contents

- [Miner Setup](#miner-setup)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [Automated Installation (Recommended)](#automated-installation-recommended)
    - [Manual Installation](#manual-installation)
  - [Running the Miner](#running-the-miner)
    - [Using PM2 (Recommended)](#using-pm2-recommended)
    - [Important Flags](#important-flags)
  - [Configuration](#configuration)
    - [Hardware Requirements](#hardware-requirements)
    - [Network Options](#network-options)
    - [AWS Setup](#aws-setup)
  - [Monitoring](#monitoring)
    - [Logs](#logs)
    - [Performance](#performance)
  - [Troubleshooting](#troubleshooting)

## Prerequisites

- **NVIDIA GPU** with CUDA support
  - Minimum 24GB VRAM recommended
  - Multiple GPUs supported
- **Ubuntu** (or Ubuntu-based Linux distribution)
- **Python 3.12**
- **CUDA-compatible drivers**


## Cloudflare R2 Bucket Configuration
  To use buckets for sharing model slices, do the following:
  1. **Navigate to R2 Object Storage and Create a Bucket**:
     - Name the bucket the same as your CloudFlare **account ID**. This can be found on the your [Cloudflare Dashboard](https://dash.cloudflare.com) in the lower right corner or the right side of the R2 Object Storage Overview page. Account IDs are not sensitive and are safe to share. 
     - Set the **region** to **ENAM** (Eastern North America).

  2. **Generate Tokens**:
     - Navigate to the R2 Object Storage Overview page, on the left side, click "Manage R2 API Tokens".
     - Create seperate **read** and  **read/write**  tokens.
     - Note down the access key IDs and secret access keys for each token. These can also be retrieved at any time from your R2 API Token Management page
     - ***Heads up***: The access key id and secret access key for your *read* token will be shared
  with other neurons through commits to the network. The secrets for your write
  token will stay secret.

  3. **Update `.env.yaml`**:
     - Create the file `.env.yaml` by copying [`.env-template.yaml`](../.env-template.yaml)
       and populate it with values from the previous steps:
       ```
         cp .env-template.yaml .env.yaml
       ```
     



## Installation

<!-- ### Automated Installation (Recommended)

The easiest way to set up a miner is using the automated installation script:

```bash
# Clone the repository
git clone https://github.com/RaoFoundation/templar
cd templar

# Make the script executable
chmod +x scripts/run.sh

# Run the installation script
./scripts/run.sh --neuron miner --network <network> \
  --aws-access-key-id <your-key> \
  --aws-secret-access-key <your-secret> \
  --bucket <your-bucket>
```

The script will:
1. Install all required dependencies (Git, npm, pm2, Rust, uv, Python 3.12)
2. Set up AWS credentials
3. Create and register Bittensor wallets
4. Configure wandb for logging
5. Start miners on all available GPUs -->

### Manual Installation

If you prefer to install manually, follow these steps:

1. **Install System Dependencies**:
```bash
# Add Python 3.12 repository
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update

# Install required packages
sudo apt-get install python3.12 python3.12-venv git npm
```

2. **Install Node.js and PM2**:
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo bash
sudo apt-get install -y nodejs
npm install pm2 -g
```

3. **Install Rust and uv**:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

4. **Set Up Python Environment**:
```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install PyTorch
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install requirements
uv sync --extra all 

# Install flash-attn
uv pip install flash-attn --no-build-isolation
```


5. **Create and Register Wallets**:
```bash
# Create coldkey
btcli wallet new_coldkey --wallet.name default --n-words 12


# Create and register hotkey
btcli wallet new_hotkey --wallet.name default --wallet.hotkey <name> --n-words 12
btcli subnet pow_register --wallet.name default --wallet.hotkey <name> --netuid <netuid> --subtensor.network <network>
```

6. **Log into Weights & Biases (WandB)**
```bash
# Log into WandB
wandb login <your_api_key>
```

## Running the Miner

### Using PM2 (Recommended)

PM2 automatically manages your miner processes and restarts them if they crash:

```bash
# Start a miner on each GPU
  pm2 start neurons/miner.py --interpreter python3 --name miner -- \
    --actual_batch_size <batch_size> \
    --wallet.name default \
    --wallet.hotkey "name" \
    --device "cuda" \
    --use_wandb \
    --netuid <netuid> \
    --subtensor.network <network> \
    --process_name miner \  # Must match PM2's --name
    --sync_state


# Monitor logs
pm2 logs

# Check status
pm2 list
```

> **Important**: When using PM2, the `--process_name` argument must match the PM2 process name specified by `--name`. For example, if PM2 process is named `miner_C0`, use `--process_name miner_C0`.

### Important Flags
- **`--process_name`**: (Required) Must match the PM2 process name when using PM2
- **`--sync_state`**: Synchronizes model state with network history
- **`--actual_batch_size`**: Set based on GPU memory:
  - 80GB+ VRAM: batch size 6
  - 40GB VRAM: batch size 3
  - 24GB VRAM: batch size 1
- **`--netuid`**: Network subnet ID (e.g., 223 for testnet)
- **`--subtensor.network`**: Network name (finney/test/local)
- **`--no_autoupdate`**: Disable automatic code updates

## Configuration

### Hardware Requirements

- **GPU Memory Requirements**:
  - Minimum: 24GB VRAM
  - Recommended: 40GB+ VRAM
  - Optimal: 80GB+ VRAM
- **Storage**: 100GB+ recommended for model and data
- **RAM**: 32GB+ recommended
- **Network**: Stable internet connection with good bandwidth

### Network Options

- **Mainnet (Finney)**:
  - Network: `finney`
  - Netuid: 3
- **Testnet**:
  - Network: `test`
  - Netuid: 223
  - Endpoint: `wss://test.finney.opentensor.ai:443/`
- **Local**:
  - Network: `local`
  - Netuid: 3
  - Endpoint: `wss://localhost:9944`

<!-- ### AWS Setup

1. Create an S3 bucket with public read access
2. Configure CORS for validator access
3. Set up IAM user with S3 access
4. Export credentials in environment -->

## Monitoring

### Logs

- **PM2 Logs**: `pm2 logs [miner_name]`
- **System Monitoring**: `pm2 monit`
- **Weights & Biases**: Enable with `--use_wandb`

### Performance

Monitor key metrics:
- GPU utilization
- Memory usage
- Network bandwidth
- Training progress
- Rewards and weights

<!-- ## Troubleshooting

Common issues and solutions:
- CUDA out of memory
- Network synchronization issues
- Registration failures -->
