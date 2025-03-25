# GPU Box Setup

This Ansible playbook automates the setup of a GPU server for AI workloads, including NVIDIA drivers, CUDA, Docker with NVIDIA runtime, and system optimizations.

## Features

- Installs and configures NVIDIA drivers
- Sets up NVIDIA Docker runtime
- Installs Docker and Docker Compose
- Configures system for optimal GPU performance
- Implements basic security hardening
- Sets up Python environment for AI development
- Optimizes memory and swap settings
- Verifies GPU functionality

## Prerequisites

- Target server with NVIDIA GPU(s)
- Ubuntu or RedHat/CentOS Linux
- SSH access with sudo privileges
- Ansible 2.9+ installed on your control machine

## Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd gpu-box-setup
   ```

2. **Create and customize your inventory file**
   ```bash
   cp inventory.example inventory
   ```
   Edit the inventory file with your server details.

3. **Create Vault for Sensitive Information**
   ```bash
   mkdir -p group_vars/all
   ansible-vault create group_vars/all/vault.yml
   ```
   Add the following to your vault file:
   ```yaml
   ---
   vault_sudo_password: "your-sudo-password"
   ```

4. **Run the playbook**
   ```bash
   ansible-playbook -i inventory playbook.yml --ask-vault-pass
   ```

## Configuration Options

The playbook is highly configurable through role variables. Key settings include:

### Common Role
- `python_version`: Python version to install (default: "3.12")
- `system_upgrade`: Whether to upgrade system packages (default: true)
- `swap_size`: Swap size in GB (default: 64)
- `pytorch_version`: PyTorch version to install (default: "2.2.0")
- `pytorch_cuda_version`: CUDA version for PyTorch (default: "12.1")

### Security Role
- `ssh_permit_root_login`: Allow root SSH login (default: "no")
- `enable_firewall`: Enable firewall (default: true)
- `firewall_allowed_tcp_ports`: List of allowed TCP ports (default includes 22)

### NVIDIA Driver Role
- `nvidia_driver_persistence_mode_on`: Enable driver persistence (default: yes)
- `nvidia_driver_ubuntu_install_from_cuda_repo`: Install driver from CUDA repo (default: yes)

## Verification

After the playbook completes, the following verification steps are performed:
- Tests NVIDIA driver installation with `nvidia-smi`
- Verifies Docker can access NVIDIA runtime
- Runs a simple CUDA test container

## Troubleshooting

- **SSH Connection Issues**: Verify SSH connectivity and key authentication
- **Sudo Password Errors**: Ensure the vault contains the correct sudo password
- **NVIDIA Driver Issues**: Check system compatibility and kernel headers
- **Docker Issues**: Verify user is in the docker group