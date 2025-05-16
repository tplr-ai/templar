# Templar Localnet Setup

This guide covers setting up a local development environment for Templar using the Ansible-based deployment in `scripts/localnet/`. This provides a complete automated solution for running validators and miners on a local Bittensor network.

## Overview

The Ansible playbook automates the deployment of a Bittensor "Templar" subnet for local development and testing. It sets up all necessary components for running Templar validators and miners, providing a smooth developer experience.

## Features

- Automated deployment of validators and miners
- Multi-GPU support with automatic assignment
- PM2 process management
- Automatic wallet creation and funding
- Network verification
- Support for both local and remote deployments
- Passwordless wallet generation on target machines

## Prerequisites

- Ubuntu 24.04 (or compatible Linux distribution)
- NVIDIA GPU(s) with compatible drivers for CUDA
- Ansible installed on your control machine
- Existing Bittensor wallet with coldkey and hotkeys for validators/miners

## Quick Start

1. **Navigate to the localnet directory:**
   ```bash
   cd scripts/localnet
   ```

2. **Configure your inventory:**
   ```bash
   cp inventory.example inventory
   ```

   Edit `inventory` for local deployment:
   ```ini
   [localnet]
   localhost ansible_connection=local remote_mode=false
   ```

   Or for remote deployment:
   ```ini
   [localnet]
   myserver ansible_host=192.168.1.100 ansible_user=ubuntu remote_mode=true
   ```

3. **Configure deployment settings:**
   ```bash
   cp group_vars/all/vault.yml.example group_vars/all/vault.yml
   ```

   Edit `group_vars/all/vault.yml` with your configuration:
   ```yaml
   # Deployment mode
   remote_mode: false  # Set to true for remote deployment
   
   # Wallet configuration
   cold_wallet_name: "owner"
   owner_hotkey: "default"
   validator_hotkeys: ["validator1", "validator2"]
   miner_hotkeys: ["miner1", "miner2"]
   
   # Network settings
   network: "local"
   netuid: 2
   
   # Development version
   templar_version: "0.2.58dev"
   
   # Staking amount per validator
   stake_amount: 1000
   
   # R2 Credentials (add your own dataset credentials)
   R2_DATASET_ACCOUNT_ID: "your_dataset_account_id"
   R2_DATASET_BUCKET_NAME: "your_dataset_bucket_name"
   R2_DATASET_READ_ACCESS_KEY_ID: "your_dataset_read_key"
   R2_DATASET_READ_SECRET_ACCESS_KEY: "your_dataset_secret_key"
   
   # Gradient bucket credentials
   R2_GRADIENTS_ACCOUNT_ID: "your_gradients_account_id"
   R2_GRADIENTS_BUCKET_NAME: "your_gradients_bucket_name"
   # ... other R2 credentials ...
   ```

4. **Run the deployment:**
   ```bash
   ansible-playbook -i inventory playbook.yml
   ```

## What the Playbook Does

The deployment process follows these steps:

1. **Common Setup** - Installs system packages, Python, Node.js, PM2, Rust toolchain, and Bittensor CLI
2. **Subtensor Setup** - Configures the local Bittensor network
3. **Wallet Creation** - Creates passwordless wallets on the target machine
4. **Templar Deployment** - Sets up virtual environment, installs dependencies, configures processes
5. **Process Management** - Starts validators and miners using PM2
6. **Network Verification** - Validates all processes are running correctly

## Process Management

View all running processes:
```bash
pm2 list
```

Monitor process logs:
```bash
pm2 logs TV1  # Validator 1 logs
pm2 logs TM1  # Miner 1 logs
```

Restart processes:
```bash
pm2 restart TV1
pm2 restart all
```

Stop all processes:
```bash
pm2 stop all
pm2 delete all
```

## Advanced Usage

### Selective Deployment with Tags

Deploy only system dependencies:
```bash
ansible-playbook -i inventory playbook.yml --tags common
```

Deploy only Templar processes:
```bash
ansible-playbook -i inventory playbook.yml --tags templar
```

Restart processes after code changes:
```bash
ansible-playbook -i inventory playbook.yml --tags templar_restart
```

### Multi-GPU Configuration

For hosts with multiple GPUs, specify arrays in your inventory:

```ini
multi_gpu_host cuda_devices='["cuda:0", "cuda:1", "cuda:2"]' wallet_hotkeys='["validator1", "validator2", "miner1"]'
```

The playbook automatically:
- Creates separate directories for each GPU instance
- Assigns processes to specific GPUs
- Manages each instance independently

### Custom GPU Assignments

Add to `vault.yml` for manual GPU assignment:
```yaml
validator_devices: [0, 1]  # Assign validators to specific GPUs
miner_devices: [2, 3]      # Assign miners to specific GPUs
```

## Wallet Management

The playbook uses a secure approach to wallet management:
- Creates passwordless wallets directly on the target machine
- Uses `owner` as the main wallet
- Creates sub-wallets like `owner_validator_1`, `owner_miner_1`
- Only the owner wallet is funded via faucet
- All wallets are registered to the subnet
- Validators are staked automatically

## Troubleshooting

### GPU Issues
```bash
nvidia-smi  # Check GPU availability
```

### Process Failures
```bash
pm2 logs <process_name>  # Check specific process logs
pm2 describe <process_name>  # View process details
```

### Wallet Problems
```bash
btcli wallet list  # List available wallets
ls ~/.bittensor/wallets/  # Check wallet files
```

### Network Status
```bash
btcli subnet list --subtensor.network local
btcli subnet info 2 --subtensor.network local
btcli wallet overview --subtensor.network local
```

### Common Errors

1. **No GPUs Found:**
   - Install NVIDIA drivers and CUDA toolkit
   - Verify with `nvidia-smi`

2. **Process Startup Failures:**
   - Check logs with `pm2 logs`
   - Verify environment variables in `.env`
   - Ensure dependencies are installed

3. **Port Conflicts:**
   - Stop conflicting processes
   - Change ports in configuration

## Directory Structure

The Ansible playbook creates the following structure on target machines:
```
~/templar/                    # Main Templar directory
├── .venv/                   # Python virtual environment
├── .env                     # Environment variables
├── ecosystem.config.js      # PM2 configuration
└── templar-{gpu_id}/        # GPU-specific instances
```

## Configuration Files

- `inventory` - Ansible inventory with host definitions
- `group_vars/all/vault.yml` - Configuration variables and secrets
- `playbook.yml` - Main Ansible playbook
- `roles/` - Ansible roles for different deployment aspects

## Best Practices

1. **Development Workflow:**
   - Use localnet for initial development
   - Test thoroughly before moving to testnet
   - Keep development and production configurations separate

2. **Security:**
   - Never commit vault files with real credentials
   - Use separate wallets for different networks
   - Keep sensitive files in `.gitignore`

3. **Resource Management:**
   - Monitor GPU usage with `nvidia-smi`
   - Check PM2 process health regularly
   - Ensure adequate disk space for logs

## Additional Resources

- [Detailed Ansible Localnet README](../scripts/localnet/README.md)
- [Miner Setup Guide](./miner.md)
- [Validator Setup Guide](./validator.md)
- [Dataset Setup Guide](./r2_dataset.md)