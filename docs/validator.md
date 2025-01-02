# Validator Setup

This document provides a comprehensive guide on how to set up and run a validator using `validator.py`. Validators are crucial components of **τemplar**, responsible for evaluating miners' contributions by assessing their uploaded gradients.

## Table of Contents

- [Validator Setup](#validator-setup)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [Using Docker Compose (Recommended)](#using-docker-compose-recommended)
    - [Manual Installation](#manual-installation)
  - [Running the Validator](#running-the-validator)
    - [Using Docker Compose](#using-docker-compose)
    - [Running Without Docker](#running-without-docker)
  - [Configuration](#configuration)
    - [Environment Variables](#environment-variables)
    - [Hardware Requirements](#hardware-requirements)
    - [Network Options](#network-options)
  - [Monitoring](#monitoring)
    - [Logs](#logs)
    - [Performance](#performance)
  - [Troubleshooting](#troubleshooting)
  - [Validator Operations](#validator-operations)
    - [State Synchronization](#state-synchronization)
    - [Evaluation Process](#evaluation-process)
    - [Weight Setting](#weight-setting)

---

## Introduction

This guide will help you set up and run a validator for **τemplar**. Validators play a critical role in maintaining the integrity of the network by evaluating miners' contributions and updating weights accordingly.

---

## Prerequisites

- **NVIDIA GPU** with CUDA support
  - Minimum H100 recommended
- **Ubuntu** (or Ubuntu-based Linux distribution)
- **Docker** and **Docker Compose**
- **Git**
- **Cloudflare R2 Bucket Configuration**:
  - **Bucket Setup**:
    1. **Create a Bucket**: Name it the same as your **account ID** and set the **region** to **ENAM**.
    2. **Generate Tokens**:
       - **Read Token**: Admin Read permissions.
       - **Write Token**: Admin Read & Write permissions.
    3. **Store Credentials**: You'll need these for the `.env` file.

---

## Installation

### Using Docker Compose (Recommended)

1. **Install Docker and Docker Compose**:

   Follow the same steps as in the [Miner Setup](#using-docker-compose-recommended) section.

2. **Clone the Repository**:

   ```bash
   git clone https://github.com/tplr-ai/templar.git
   cd templar
   ```

3. **Navigate to the Docker Directory**:

   ```bash
   cd docker
   ```

4. **Create and Populate the `.env` File**:

   Create a `.env` file in the `docker` directory by copying the `.env.example`:

   ```bash
   cp .env.example .env
   ```

   Populate the `.env` file with your configuration. Variables to set:

   ```dotenv:docker/.env
   WANDB_API_KEY=your_wandb_api_key

   # Cloudflare R2 Credentials
   R2_ACCOUNT_ID=your_r2_account_id

   R2_READ_ACCESS_KEY_ID=your_r2_read_access_key_id
   R2_READ_SECRET_ACCESS_KEY=your_r2_read_secret_access_key

   R2_WRITE_ACCESS_KEY_ID=your_r2_write_access_key_id
   R2_WRITE_SECRET_ACCESS_KEY=your_r2_write_secret_access_key

   # Wallet Configuration
   WALLET_NAME=default
   WALLET_HOTKEY=your_validator_hotkey_name

   # Network Configuration
   NETWORK=finney
   NETUID=3

   # GPU Configuration
   CUDA_DEVICE=cuda:0

   # Node Type
   NODE_TYPE=validator

   # Additional Settings
   DEBUG=false
   ```

   **Note**: Set `NODE_TYPE` to `validator`.

5. **Update `docker-compose.yml`**:

   Ensure that the `docker-compose.yml` file is correctly configured for your setup.

6. **Run Docker Compose**:

   Start the validator using Docker Compose:

   ```bash
   docker compose -f docker/compose.yml up -d
   ```

### Manual Installation

If you prefer to run the validator without Docker, follow the instructions in the [Running Without Docker](#running-without-docker) section.

---

## Running the Validator

### Using Docker Compose

After completing the installation steps, your validator should be running. Check it with:

```bash
docker ps
```

You should see a container named `templar-validator-<WALLET_HOTKEY>`.

### Running Without Docker

1. **Install System Dependencies**:

   Same as in the miner setup.

2. **Install NVIDIA CUDA Drivers**:

   Install the appropriate NVIDIA CUDA drivers.

3. **Clone the Repository**:

   ```bash
   git clone https://github.com/tplr-ai/templar.git
   cd templar
   ```

4. **Set Up Python Environment**:

   ```bash
   export WANDB_API_KEY=your_wandb_api_key
   export NODE_TYPE=your_node_type
   export WALLET_NAME=your_wallet_name
   export WALLET_HOTKEY=your_wallet_hotkey
   export CUDA_DEVICE=your_cuda_device
   export NETWORK=your_network
   export NETUID=your_netuid
   export DEBUG=your_debug_setting
   export R2_ACCOUNT_ID=your_r2_account_id
   export R2_READ_ACCESS_KEY_ID=your_r2_read_access_key_id
   export R2_READ_SECRET_ACCESS_KEY=your_r2_read_secret_access_key
   export R2_WRITE_ACCESS_KEY_ID=your_r2_write_access_key_id
   export R2_WRITE_SECRET_ACCESS_KEY=your_r2_write_secret_access_key
   export GITHUB_USER=your_github_username
   ```

5. **Create and Register Validator Wallet**:

   ```bash
   # Create coldkey if not already created
   btcli wallet new_coldkey --wallet.name default --n-words 12

   # Create and register validator hotkey
   btcli wallet new_hotkey --wallet.name default --wallet.hotkey validator --n-words 12
   btcli subnet pow_register --wallet.name default --wallet.hotkey validator --netuid <netuid> --subtensor.network <network>
   ```

6. **Log into Weights & Biases (WandB)**:

   ```bash
   wandb login your_wandb_api_key
   ```

7. **Set Environment Variables**:

   Export necessary environment variables as in the miner setup.

8. **Run the Validator**:

   ```bash
   python neurons/validator.py \
     --actual_batch_size 6 \
     --wallet.name default \
     --wallet.hotkey validator \
     --device cuda \
     --use_wandb \
     --netuid <netuid> \
     --subtensor.network <network> \
     --sync_state
   ```

---

## Configuration

### Environment Variables

Set the following in the `docker/.env` file when using Docker Compose:

```dotenv:docker/.env
WANDB_API_KEY=your_wandb_api_key

# Cloudflare R2 Credentials
R2_ACCOUNT_ID=your_r2_account_id

R2_READ_ACCESS_KEY_ID=your_r2_read_access_key_id
R2_READ_SECRET_ACCESS_KEY=your_r2_read_secret_access_key

R2_WRITE_ACCESS_KEY_ID=your_r2_write_access_key_id
R2_WRITE_SECRET_ACCESS_KEY=your_r2_write_secret_access_key

# Wallet Configuration
WALLET_NAME=default
WALLET_HOTKEY=your_validator_hotkey_name

# Network Configuration
NETWORK=finney
NETUID=3

# GPU Configuration
CUDA_DEVICE=cuda:0

# Node Type
NODE_TYPE=validator

# Additional Settings
DEBUG=false
```

**Note**: The R2 permissions remain unchanged.

### Hardware Requirements

- **GPU Requirements**:
  - Minimum: NVIDIA H100 with 80GB VRAM
- **Storage**: 200GB+ recommended for model and evaluation data
- **RAM**: 32GB+ recommended
- **Network**: High-bandwidth, stable connection for state synchronization

### Network Options

- **Mainnet (Finney)**:
  - Network: `finney`
  - Netuid: `3`
- **Testnet**:
  - Network: `test`
  - Netuid: `223`
- **Local**:
  - Network: `local`
  - Netuid: `1`

---

## Monitoring

### Logs

- **Docker Logs**:

  ```bash
  docker logs -f templar-validator-${WALLET_HOTKEY}
  ```

- **Weights & Biases**:

  - Ensure `--use_wandb` is enabled
  - Monitor evaluation metrics and network statistics

### Performance

Key metrics to monitor:

- GPU utilization
- Memory usage
- Network bandwidth
- Evaluation throughput
- Weight setting frequency

---

## Troubleshooting

- **State Synchronization Failures**: Check network settings and ensure the validator is properly registered and connected.
- **Out of Memory Errors**: Reduce `--actual_batch_size`.
- **Network Connectivity Issues**: Verify firewall settings and network configurations.

---

## Validator Operations

### State Synchronization

- The validator synchronizes its model with the latest global state.
- It gathers and applies gradients from miners to maintain consistency.

### Evaluation Process

1. **Collect Miner Gradients**: Gathers compressed gradients submitted by miners.
2. **Evaluate Contributions**: Assesses the impact of each miner's gradient on model performance.
3. **Compute Scores**: Calculates scores based on loss improvement.
4. **Update Weights**: Adjusts miners' weights on the blockchain accordingly.

### Weight Setting

- **Scoring Mechanism**: Based on the performance improvement contributed by miners.
- **Update Frequency**: Weights are periodically updated on the blockchain.
- **Impact**: Influences reward distribution and miner reputation in the network.

---

# `.env.example`

Here's the updated `.env.example` file populated with all the required variables:

```dotenv:docker/.env.example
# Weights & Biases API Key
WANDB_API_KEY=your_wandb_api_key

# Cloudflare R2 Credentials
R2_ACCOUNT_ID=your_r2_account_id

# Read Token (Shared with other neurons)
R2_READ_ACCESS_KEY_ID=your_r2_read_access_key_id
R2_READ_SECRET_ACCESS_KEY=your_r2_read_secret_access_key

# Write Token (Keep Secret)
R2_WRITE_ACCESS_KEY_ID=your_r2_write_access_key_id
R2_WRITE_SECRET_ACCESS_KEY=your_r2_write_secret_access_key

# Wallet Configuration
WALLET_NAME=default
WALLET_HOTKEY=your_hotkey_name

# Network Configuration
NETWORK=finney
NETUID=3

# GPU Configuration
CUDA_DEVICE=cuda:0

# Node Type (miner or validator)
NODE_TYPE=miner

# Additional Settings
DEBUG=false
```

**Note**:

- Replace all placeholders with your actual credentials.
- The `.env` file should be placed in the `docker` directory (`templar/docker/.env`).
- Permissions for R2 remain the same; read credentials are shared, write credentials are kept secret.

