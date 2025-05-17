# Ansible Provisioning Playbook

This Ansible project provisions templar. It clones the repository, sets up a the required venv with CUDA packages, installs needed packages, and deploys a continuously running miner service.

_**Note**: currently we only support Ubuntu (expected 22.04.) and miner nodes with CUDA support already installed._

## Table of Contents

- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
  - [Inventory File](#inventory-file)
  - [Vault File for Secrets](#vault-file-for-secrets)
- [Usage](#usage)
  - [Running the Playbook](#running-the-playbook)
  - [Overriding Default Variables](#overriding-default-variables)
- [Customization](#customization)
- [Additional Notes](#additional-notes)

## Prerequisites

Before running the playbook, ensure you have the following installed on your control machine:

- [Ansible](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html)
- A Unix-like environment (Linux/macOS) with SSH access configured to your target hosts.
- Python 3 and pip (if you plan to run additional pip-based tasks locally).
- [Ansible Vault](https://docs.ansible.com/ansible/latest/user_guide/vault.html) for securely storing sensitive data.
- A target host with SSH access and Python installed (the playbook will install Python if it’s not present).
- A CUDA-enabled GPU for running the miner.

## Configuration

### Inventory File

Create an inventory file defining your target hosts and their GPU configurations:

```ini
[bittensor_subnet]
# Single GPU host example
192.168.123.213 ansible_user=root ansible_port=12345 wallet_hotkeys='["miner"]' cuda_devices='["cuda"]'

# Multi-GPU host example
192.168.222.111 ansible_user=root ansible_port=23456 wallet_hotkeys='["miner_1", "miner_2", "miner_3", "miner_4"]' cuda_devices='["cuda:0", "cuda:1", "cuda:2", "cuda:3"]'
```

Note: The `wallet_hotkeys` and `cuda_devices` arrays must have matching lengths.

**Steps to create the inventory file:**

1. In the root of your project, create a file named `inventory`.
2. Add the host definition as shown above.
3. Confirm that `inventory` is listed in your `.gitignore` so it is not checked into version control.

### Vault File for Secrets

To securely store sensitive environment variables (e.g., API keys, wallet credentials), use Ansible Vault. Create a vault file (for example, `group_vars/all/vault.yml`) containing your secrets. **Add this file to your `.gitignore`**.

**Steps to create the vault file:**

1. Create the directory `group_vars/all/` if it doesn’t already exist.

2. Run the following command to create and encrypt the file:

   ```bash
   ansible-vault create group_vars/all/vault.yml
   ```

3. In the editor that opens, define your sensitive variables under the `env_vars` key. For example:

   ```yaml
   env_vars:
     WANDB_API_KEY: "your_wandb_key"
     INFLUXDB_TOKEN: "your_influxdb_token"  # Optional, falls back to default if not provided
     R2_ACCOUNT_ID: "your_r2_account_id"
     R2_GRADIENTS_ACCOUNT_ID: "your_r2_gradients_account_id"
     # ... other env vars ...
     WALLET_NAME: "default"
     NETWORK: "finney"
     NETUID: "3"
     # Note: CUDA_DEVICE and WALLET_HOTKEY are now managed per-instance

   # GPU configuration (defaults, can be overridden in inventory)
   cuda_devices: ["cuda:0"]
   wallet_hotkeys: ["miner_0"]
   ```

4. Save and exit the editor. When you run the playbook, you’ll be prompted for the vault password.

## Usage

### Running the Playbook

From the root directory (where `provision.yml` is located), run:

```bash
ansible-playbook -i inventory provision.yml --ask-vault-pass
```

- The `-i inventory` option tells Ansible which inventory file to use.
- The `--ask-vault-pass` (optional) flag will prompt you for the vault password to decrypt the sensitive data in your vault file. _**Note**: use this flag only if you have encrypted your vault file._

### Overriding Default Variables

The playbook is designed to be modular and configurable. You can override default variables in several ways:

1. **Inventory or Group/Host Variables:**
   Override variables by defining them in a host or group variable file (e.g., `host_vars/your_host.yml`).

2. **Extra Variables on the Command Line:**
   Pass variables directly via the command line:

   ```bash
   ansible-playbook -i inventory provision.yml -e "actual_batch_size=5 wallet_name=default wallet_hotkey=miner" --ask-vault-pass
   ```

3. **Custom Defaults:**
   Modify the `roles/templar/defaults/main.yml` file as needed, though it is recommended to override via inventory or extra-vars for production use.

## Customization

- **APT and Pip Packages:**
  In `roles/templar/defaults/main.yml`, you can add additional packages:
  - `additional_apt_packages`: List any extra APT packages.
  - `additional_pip_packages`: List extra pip packages to install globally.
  - `additional_uv_pip_packages`: List extra pip packages to install in the virtual environment.

- **Miner Parameters:**
  Default miner parameters (e.g., `actual_batch_size`, `device`, `netuid`, `subtensor_network`, `wallet_name`, `wallet_hotkey`) are defined in the defaults file.

- **Templates:**
  Both `.env` and `run.sh` are generated via Jinja2 templates (`.env.j2` and `run.sh.j2`). Adjust these templates if your application requires additional logic or different formatting.

## Multi-GPU Support

The playbook supports running multiple miner instances on hosts with multiple GPUs. For each GPU:

- A separate clone of the repository is created in a unique directory
- Environment variables are templated with GPU-specific settings
- A dedicated systemd service is created (when systemd is enabled)

To configure multiple GPUs:

1. In your inventory, specify arrays for `cuda_devices` and `wallet_hotkeys`:

   ```ini
   miner2 cuda_devices='["cuda:0", "cuda:1"]' wallet_hotkeys='["miner_1", "miner_2"]'
   ```

2. The playbook will automatically:
   - Create separate directories (e.g., templar-cuda0, templar-cuda1)
   - Configure each instance with its own GPU and wallet
   - Start separate services for each GPU

Note: Ensure you have unique wallet hotkeys for each GPU to avoid conflicts.

## Additional Notes

- **Security:**
  Always keep sensitive files (inventory, vault files) out of version control. Ensure your `.gitignore` file contains entries for these files. For example, your `.gitignore` might include:

  ```text
  inventory
  group_vars/all/vault.yml
  ```

- **Idempotency:**
  The playbook is designed to be idempotent where possible. You can re-run the playbook to update your configuration without re-cloning repositories or reinstalling packages unnecessarily.

- **Troubleshooting:**
  - Use the `-vvv` flag with `ansible-playbook` for verbose output if you run into issues.
  - Ensure your SSH keys and network connectivity are correctly configured when connecting to your remote host.
