# TEMPLAR Localnet Ansible Deployment

This Ansible project automates the deployment of a Bittensor "Templar" subnet for local development and testing. The playbook sets up all necessary components for running Templar validators and miners, focusing on a smooth developer experience.

## System Requirements

- **Operating System**: Ubuntu 24.04 (or compatible Linux distribution)
- **Hardware**: NVIDIA GPU(s) with compatible drivers for CUDA
- **Local Machine**: Ansible installed on the control machine
- **Bittensor Wallet**: Existing wallet with coldkey and hotkeys for validators/miners

## Detailed Setup Guide

### 1. Prepare Your Configuration

#### Create Your Inventory File
```bash
cp inventory.example inventory
```

For local deployment (on your current machine):
```
[localnet]
localhost ansible_connection=local remote_mode=false
```

For remote deployment (to another server):
```
[localnet]
myserver ansible_host=203.0.113.10 ansible_user=ubuntu remote_mode=true
```

#### Configure Vault Variables
```bash
cp group_vars/all/vault.yml.example group_vars/all/vault.yml
```

Then edit `group_vars/all/vault.yml` with the following critical settings:

```yaml
# Deployment Mode
remote_mode: false  # Set to true for remote server deployment

# Wallet Configuration - MUST be configured correctly
cold_wallet_name: "your_wallet_folder_name"  # Name of folder under ~/.bittensor/wallets
bittensor_wallet_password: "your_password"   # Password for your wallet (if encrypted)
owner_hotkey: "primary_hotkey_name"          # Hotkey that will create the subnet
validator_hotkeys: ["validator1", "validator2"]  # List of hotkeys for validators
miner_hotkeys: ["miner1", "miner2"]          # List of hotkeys for miners

# Network Settings
network: "local"  # Network name (local, test, finney, etc)
netuid: 2         # Subnet ID to create or use

# Version for development
templar_version: "0.2.58dev"  # Development version string

# Staking Settings
stake_amount: 1000  # TAO amount to stake on each validator
```

### 2. Running the Deployment

#### Full Deployment
```bash
ansible-playbook -i inventory playbook.yml
```

#### Selective Deployment with Tags
Deploy only system dependencies:
```bash
ansible-playbook -i inventory playbook.yml --tags common
```

Deploy or update only Templar processes:
```bash
ansible-playbook -i inventory playbook.yml --tags templar
```

Restart existing Templar processes (after code changes):
```bash
ansible-playbook -i inventory playbook.yml --tags templar_restart
```

### 3. What The Playbook Does

The deployment process follows these steps in sequence:

1. **Common Setup** (`common` role):
   - Installs all required system packages (build tools, Python, Node.js, etc.)
   - Sets up Rust toolchain via rustup
   - Installs PM2 for process management
   - Installs Bittensor CLI and Python dependencies
   - Installs uv package manager for Python

2. **Templar Deployment** (`templar` role):
   - Checks for available GPUs on the system
   - Creates Python virtual environment using uv
   - Installs project dependencies with uv sync
   - Configures environment variables via .env file
   - Generates PM2 ecosystem.config.js for process management
   - Starts validator processes with proper GPU assignments
   - Starts miner processes with proper GPU assignments

3. **Network Verification** (`network_verify` role):
   - Validates that all processes are running correctly
   - Ensures proper network connectivity

### 4. Technical Implementation Details

#### Python Environment
The playbook uses modern Python tooling:
- **uv**: Fast, reliable package installer and environment manager
- **Virtual Environment**: Isolated environment in `~/templar/.venv/`
- **Dependency Resolution**: Uses `uv sync` for efficient installation

#### Process Management
All processes are managed through PM2:
- **Validators**: Named as "TV1", "TV2", etc.
- **Miners**: Named as "TM1", "TM2", etc.
- **GPU Assignment**: Processes are automatically assigned to GPUs in sequence
- **Project Naming**: Random suffix generated for each deployment

#### Ecosystem Configuration
The PM2 ecosystem file (`ecosystem.config.js`) contains:
- Process definitions for all validators and miners
- Environment variable loading from .env
- Command-line arguments for each process
- Device assignments (CUDA devices)

## Managing Your Deployment

### Process Control

Check status of all processes:
```bash
pm2 list
```

View logs for a specific process:
```bash
pm2 logs TV1  # Validator 1
pm2 logs TM2  # Miner 2
```

Restart specific processes:
```bash
pm2 restart TV1
```

Stop and remove all processes:
```bash
pm2 stop all
pm2 delete all
```

### Troubleshooting Steps

1. **GPU Issues**
   - Verify GPU availability: `nvidia-smi`
   - Check process-to-GPU mapping in ecosystem.config.js
   - Ensure CUDA libraries are properly installed

2. **Process Failures**
   - Check PM2 process status: `pm2 list`
   - Examine logs: `pm2 logs <process_name>`
   - Verify environment variables in .env file

3. **Wallet Problems**
   - Ensure wallet files exist in ~/.bittensor/wallets/
   - Verify hotkeys are correctly named in vault.yml
   - Check wallet connectivity with `btcli wallet list`

4. **Python Dependency Issues**
   - Activate virtual environment: `source ~/templar/.venv/bin/activate`
   - Check installed packages: `pip list`
   - Manually reinstall dependencies if needed: `uv sync`

### Common Errors and Solutions

1. **No GPUs Found**:
   - Error: "Found 0 GPU(s)"
   - Solution: Install NVIDIA drivers and CUDA toolkit

2. **Process Startup Failures**:
   - Error: PM2 shows processes in error state
   - Solution: Check logs with `pm2 logs` and resolve dependency issues

3. **Wallet File Not Found**:
   - Error: Cannot find wallet files
   - Solution: Ensure wallet paths and names are correct, and files exist

4. **Port Conflicts**:
   - Error: Address already in use
   - Solution: Stop conflicting processes or change ports in configuration

## Advanced Usage

### Customizing GPU Assignments

To manually assign specific GPUs to processes, add these variables to your vault.yml:

```yaml
# Custom GPU device assignments (device indices)
validator_devices: [0, 1]  # Assign validators to specific GPUs
miner_devices: [2, 3]      # Assign miners to specific GPUs
```

### Restarting After Code Changes

After making changes to the Templar code:

```bash
# Pull latest code changes
cd ~/templar
git pull

# Restart processes with the ansible playbook
ansible-playbook -i inventory playbook.yml --tags templar_restart
```

### Verifying Network Status

To check the status of your local subnet:

```bash
btcli subnet list --subtensor.network local
btcli subnet info 2 --subtensor.network local  # Replace 2 with your netuid
btcli wallet overview --subtensor.network local
```