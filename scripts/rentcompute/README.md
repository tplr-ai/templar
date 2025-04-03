# RentCompute CLI

A command line utility for renting and managing GPU compute instances from cloud providers.

## Installation

```bash
pip install .
```

## Usage Guide

### Setting up API credentials

```bash
rentcompute login
```

This will prompt you to enter your API key, which will be securely stored in `~/.rentcompute/credentials.yaml`.

### Finding Available Instances

```bash
# Search with specific requirements
rentcompute search --gpu-min=2 --gpu-type=h100 --price-max=5

# Filter by name pattern
rentcompute search --name=gpu

# View all available instances
rentcompute search
```

This displays available instances matching your criteria without starting them.

### Starting a Compute Instance

```bash
# Start with specific requirements and custom name
rentcompute start --name="my-gpu-server" --gpu-min=2 --gpu-max=8 --gpu-type=h100 --price-max=5

# Start with specific GPU requirements
rentcompute start --gpu-min=4 --gpu-type=h100

# Start with price constraints
rentcompute start --price-max=3.50

# Start any available instance (lowest cost option)
rentcompute start

# Start and automatically provision using .rentcompute.yml
rentcompute start --gpu-type=h100 --provision
```

After starting, the tool will display SSH connection details for accessing your instance.

### Managing Active Instances

List all your active instances:

```bash
rentcompute list
```

This shows all running instances with their details:
- Instance name and ID
- SSH connection details (host, user, port)
- Status
- GPU specifications
- Hourly price
- Ready-to-use SSH command

### Provisioning Instances

Provisioning allows you to automatically configure new or existing instances.

#### Provisioning During Start

```bash
# Start and provision a new instance
rentcompute start --gpu-type=h100 --provision
```

#### Provisioning Existing Instances

```bash
# Provision an existing instance (with confirmation)
rentcompute provision --id <instance-id>

# Provision without confirmation
rentcompute provision --id <instance-id> -y
```

#### Provisioning Configuration

Provisioning uses the `.rentcompute.yml` file in your current directory. Example:

```yaml
# Instance provisioning configuration
provisioning:
  # Type: ansible, script, or docker
  type: ansible
  
  # Ansible configuration
  playbook: ./playbook.yml
  # Target hosts group (should match hosts: in playbook)
  hosts_group: localnet
  # Root directory where Ansible files are located
  root_dir: ../scripts/localnet
  # Path to vars file (relative to root_dir)
  vars_file: group_vars/all/vault.yml
  # Extra variables for ansible-playbook
  extra_vars:
    app_env: development
    gpu_driver: nvidia-latest
  
  # For script provisioning (uncomment to use)
  # type: script
  # script: ./setup.sh
  
  # For docker provisioning (uncomment to use)
  # type: docker
  # compose_file: ./docker-compose.yml
```

Supported provisioning methods:
- **ansible**: Runs Ansible playbooks on the instance
- **script**: Executes shell scripts with instance details as environment variables
- **docker**: Copies and runs docker-compose files on the instance

### Syncing Files with Instances

Sync your local files with running instances:

```bash
# Sync with all instances (with confirmation)
rentcompute rsync

# Sync with a specific instance
rentcompute rsync --id <instance-id>

# Sync without confirmation
rentcompute rsync -y

# Sync and reload instances after sync
rentcompute rsync --reload

# Use a custom config file
rentcompute rsync --config custom-config.yml
```

This uses rsync with `-avzP --delete` options and automatically excludes common development directories like `node_modules`, `target`, `venv`, etc.

Sync configuration in `.rentcompute.yml`:

```yaml
# Directories to sync
sync:
  - source: ./data
    destination: ~/data
  - source: ./src
    destination: ~/project/src
  - source: ./scripts
    destination: ~/scripts
```

### Reloading Instances

Reload running instances after making changes:

```bash
# Reload all instances (with confirmation)
rentcompute reload --all

# Reload a specific instance
rentcompute reload --id <instance-id>

# Reload without confirmation
rentcompute reload --all -y

# Use a custom config file
rentcompute reload --all --config custom-config.yml
```

You can also reload instances immediately after syncing files by adding the `--reload` flag to the rsync command:

```bash
rentcompute rsync --reload
```

Reload configuration in `.rentcompute.yml`:

```yaml
# Configuration for reloading instances
reload:
  type: ansible
  playbook: ./reload.yml
  root_dir: ../localnet
  hosts_group: localnet
  vars_file: group_vars/all/vault.yml
  extra_vars:
    remote_mode: true
    gpu_driver: nvidia-latest
```

### Stopping Instances

Stop a specific instance:

```bash
# Stop with confirmation
rentcompute stop --id <instance-id>

# Skip confirmation
rentcompute stop --id <instance-id> -y
```

Stop all running instances:

```bash
# Stop all with confirmation
rentcompute stop --all

# Stop all without confirmation
rentcompute stop --all -y
```

The stop command:
1. Verifies instance existence and shows details
2. Asks for confirmation (unless `-y` is used)
3. Sends stop requests to the provider
4. Shows results summary

## Configuration Files

RentCompute uses the following configuration files:

1. **API Credentials**: `~/.rentcompute/credentials.yaml`
   - Contains provider API keys

2. **Instance Configuration**: `.rentcompute.yml` (in working directory)
   - Provisioning configuration
   - Sync directory mappings
   - Environment variables

## Command Reference

| Command | Description |
|---------|-------------|
| `login` | Set API credentials |
| `search` | Find available instances |
| `start` | Start a new instance |
| `list` | List active instances |
| `provision` | Provision an existing instance |
| `rsync` | Sync files with instances |
| `reload` | Reload instances after changes |
| `stop` | Stop instance(s) |

## Development

To set up the development environment:

```bash
pip install -e .
```

Running the tool during development:

```bash
# Using uv run (recommended)
uv run rentcompute [command]

# Enable debugging
uv run rentcompute --debug [command]
```

## Provider Support

Currently supported providers:
- **Celium**: GPU cloud provider
- **Mock**: Local testing provider