# Miner Setup with Ansible

This guide provides an alternative method for setting up Templar miners using Ansible automation. The Ansible playbook automates the deployment process, making it easier to provision miners across multiple hosts or manage multi-GPU configurations.

## Overview

The Ansible playbook (`scripts/miner-setup-ansible/playbook.yml`) automates the following tasks:
- Clones the Templar repository
- Sets up the required Python virtual environment with CUDA support
- Installs necessary system and Python packages
- Configures environment variables and credentials
- Deploys miners as managed services (systemd or nohup)
- Supports multi-GPU configurations with separate instances per GPU

## Prerequisites

Before using the Ansible playbook, ensure you have:

- **Control Machine Requirements**:
  - [Ansible](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) installed
  - SSH access configured to your target hosts
  - Python 3 and pip
  - Unix-like environment (Linux/macOS)

- **Target Host Requirements**:
  - Ubuntu (22.04 recommended)
  - NVIDIA GPU with CUDA support already installed
  - SSH access enabled
  - Python installed (the playbook will install if missing)

- **Templar Requirements**:
  - Wallet credentials (coldkey and hotkey)
  - Cloudflare R2 bucket configuration
  - Weights & Biases API key
  - Dataset setup following the [R2 Dataset Guide](./r2_dataset.md)

## Directory Structure

The Ansible setup is located in `scripts/miner-setup-ansible/`:

```
scripts/miner-setup-ansible/
├── README.md
├── playbook.yml
├── inventory.example
├── group_vars/
│   └── all/
│       └── vault.yml.example
└── roles/
    └── templar/
        ├── defaults/
        │   └── main.yml
        ├── tasks/
        │   └── main.yml
        └── templates/
            ├── miner.service.j2
            └── run.sh.j2
```

## Configuration

### 1. Create Inventory File

Create an inventory file defining your target hosts and their GPU configurations:

```bash
cd scripts/miner-setup-ansible
cp inventory.example inventory
```

Edit the `inventory` file:

```ini
[bittensor_subnet]
# Single GPU host example
192.168.1.100 ansible_user=ubuntu ansible_port=22 wallet_hotkeys='["miner"]' cuda_devices='["cuda"]'

# Multi-GPU host example
192.168.1.101 ansible_user=ubuntu ansible_port=22 wallet_hotkeys='["miner_1", "miner_2"]' cuda_devices='["cuda:0", "cuda:1"]'
```

**Important**: The `wallet_hotkeys` and `cuda_devices` arrays must have matching lengths.

### 2. Configure Secrets with Ansible Vault

Create an encrypted vault file for sensitive credentials:

```bash
# Create the directory if it doesn't exist
mkdir -p group_vars/all/

# Copy the example file
cp group_vars/all/vault.yml.example group_vars/all/vault.yml

# Encrypt the vault file
ansible-vault encrypt group_vars/all/vault.yml
```

Edit the vault file with your credentials:

```bash
ansible-vault edit group_vars/all/vault.yml
```

Configure the following variables:

```yaml
env_vars:
  WANDB_API_KEY: "your_wandb_api_key"
  INFLUXDB_TOKEN: "your_influxdb_token"  # Optional
  
  # Dataset R2 credentials - Set up your own dataset
  # See: docs/r2_dataset.md for instructions
  R2_DATASET_ACCOUNT_ID: "your_dataset_account_id"
  R2_DATASET_BUCKET_NAME: "your_dataset_bucket_name"
  R2_DATASET_READ_ACCESS_KEY_ID: "your_dataset_read_access_key_id"
  R2_DATASET_READ_SECRET_ACCESS_KEY: "your_dataset_read_secret_access_key"
  
  # Gradient bucket credentials
  R2_GRADIENTS_ACCOUNT_ID: "your_gradients_account_id"
  R2_GRADIENTS_BUCKET_NAME: "your_gradients_bucket_name"
  R2_GRADIENTS_READ_ACCESS_KEY_ID: "your_gradients_read_access_key_id"
  R2_GRADIENTS_READ_SECRET_ACCESS_KEY: "your_gradients_read_secret_access_key"
  R2_GRADIENTS_WRITE_ACCESS_KEY_ID: "your_gradients_write_access_key_id"
  R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY: "your_gradients_write_secret_access_key"
  
  # The aggregator credentials use the provided values
  R2_AGGREGATOR_ACCOUNT_ID: "c17ae32eab2883094481526a22e0dfa1"
  R2_AGGREGATOR_BUCKET_NAME: "aggregator"
  R2_AGGREGATOR_READ_ACCESS_KEY_ID: "421a6f9f494fa3ee2329ec8ab8a66446"
  R2_AGGREGATOR_READ_SECRET_ACCESS_KEY: "523737ec0be417851b45aaf7ad6adf0aca8fed49729c2e99b81827e5d9a9ddd0"
  
  WALLET_NAME: "default"
  NETWORK: "finney"
  NETUID: "3"
  GITHUB_USER: "your_github_username"

# Miner parameters
actual_batch_size: 5
netuid: "3"
subtensor_network: "finney"
wallet_name: "default"

# GPU configuration (defaults, can be overridden in inventory)
cuda_devices: ["cuda:0"]
wallet_hotkeys: ["miner_0"]
```

### 3. Configure the Playbook

The default configuration can be found in `roles/templar/defaults/main.yml`. You can override these values in your vault file or pass them as extra variables when running the playbook.

Key configuration options:
- `use_systemd`: Whether to use systemd services (default: false)
- `actual_batch_size`: Batch size for training (default: 5)
- System packages to install (customizable via `additional_apt_packages`)
- Python packages to install (customizable via `additional_pip_packages`)

## Running the Playbook

### Basic Usage

From the `scripts/miner-setup-ansible/` directory:

```bash
ansible-playbook -i inventory playbook.yml --ask-vault-pass
```

This will:
1. Prompt for your vault password
2. Connect to the hosts defined in your inventory
3. Execute the provisioning tasks

### Advanced Usage

#### Override Variables

You can override default variables via command line:

```bash
ansible-playbook -i inventory playbook.yml \
  -e "actual_batch_size=6 wallet_name=my_wallet" \
  --ask-vault-pass
```

#### Enable Systemd Services

To use systemd instead of nohup for process management:

```bash
ansible-playbook -i inventory playbook.yml \
  -e "use_systemd=true" \
  --ask-vault-pass
```

#### Verbose Output

For debugging, use the `-vvv` flag:

```bash
ansible-playbook -i inventory playbook.yml --ask-vault-pass -vvv
```

## Multi-GPU Support

The playbook supports running multiple miner instances on hosts with multiple GPUs:

1. **Configure in Inventory**:
   ```ini
   multi_gpu_host cuda_devices='["cuda:0", "cuda:1", "cuda:2"]' wallet_hotkeys='["miner_1", "miner_2", "miner_3"]'
   ```

2. **Automatic Instance Creation**:
   - Separate directories: `templar-0`, `templar-1`, `templar-2`
   - Individual environment configurations per GPU
   - Dedicated services for each GPU instance

3. **Service Management**:
   - Systemd: `miner-0.service`, `miner-1.service`, etc.
   - Nohup: Separate processes per GPU

## Post-Installation

After successful deployment:

1. **Check Service Status** (if using systemd):
   ```bash
   sudo systemctl status miner.service
   ```

2. **View Logs**:
   - Systemd: `sudo journalctl -u miner.service -f`
   - Nohup: Check `miner_loop.log` in the templar directory

3. **Monitor Performance**:
   - Use Weights & Biases dashboard
   - Check GPU utilization with `nvidia-smi` or `nvtop`

## Customization

### Adding Packages

In your vault file or via command line:

```yaml
additional_apt_packages:
  - tmux
  - nethogs

additional_pip_packages:
  - numpy
  - pandas
```

### Modifying Templates

The playbook uses Jinja2 templates for:
- `.env` file configuration
- `run.sh` script
- systemd service files

These can be customized in the `roles/templar/templates/` directory.

## Troubleshooting

### Common Issues

1. **SSH Connection Failed**:
   - Verify SSH access: `ssh ubuntu@host_ip`
   - Check inventory file for correct user and port

2. **Vault Password Issues**:
   - Ensure vault file is encrypted: `ansible-vault view group_vars/all/vault.yml`
   - Re-encrypt if needed: `ansible-vault rekey group_vars/all/vault.yml`

3. **CUDA Not Found**:
   - Verify CUDA installation on target host
   - Check `nvidia-smi` output

4. **Service Fails to Start**:
   - Check logs for errors
   - Verify all environment variables are set correctly
   - Ensure wallet keys exist on the host

### Security Best Practices

1. **Keep Sensitive Files Secure**:
   - Add to `.gitignore`:
     ```
     inventory
     group_vars/all/vault.yml
     ```

2. **Use Strong Vault Passwords**:
   - Generate secure passwords for vault encryption
   - Store vault password securely

3. **Limit SSH Access**:
   - Use SSH keys instead of passwords
   - Configure firewall rules appropriately

## Additional Resources

- [Ansible Documentation](https://docs.ansible.com/ansible/latest/)
- [Templar Documentation](./README.md)
- [Dataset Setup Guide](./r2_dataset.md)
- [Manual Miner Setup](./miner.md)