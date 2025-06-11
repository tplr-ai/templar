# Subtensor Production Deployment

Ansible deployment for highly available Subtensor infrastructure.

## 🚀 Quick Start

### Simple Deployment
```bash
# 1. Prepare Ubuntu 24.04 server with Docker
# 2. Configure inventory and settings
cp inventory.example inventory
cp group_vars/all.yml.example group_vars/all.yml

# 3. Deploy everything
ansible-playbook -i inventory playbook.yml

# 4. Verify deployment
ansible your-server -i inventory -m shell -a "subtensor-admin health" --become
```

**📖 For detailed step-by-step instructions, see [REPLICATION.md](REPLICATION.md)**

## Architecture

This deployment creates a **2-node high availability** Subtensor setup:

- **subtensor-0**: Primary node (ports: 9933, 9944, 30333)
- **subtensor-1**: Secondary node (ports: 9934, 9945, 30334)

All nodes sync with the official Finney mainnet using warp sync and official bootnodes. Running Subtensor v2.0.10 in production mode.

## Configuration

Edit `group_vars/all.yml` for network, replicas, and resource limits.

### Chainspec Management

The deployment automatically downloads the official chainspec from:

```
https://raw.githubusercontent.com/opentensor/subtensor/v2.0.10/chainspecs/raw_spec_finney.json
```

This ensures consistency with the official Subtensor network configuration.

## Commands

```bash
# Full deployment
ansible-playbook -i inventory playbook.yml --ask-vault-pass

# Specific components
ansible-playbook -i inventory playbook.yml --tags nginx
ansible-playbook -i inventory playbook.yml --tags subtensor
ansible-playbook -i inventory playbook.yml --tags monitoring
ansible-playbook -i inventory playbook.yml --tags tools
```

## Admin Tools

The deployment includes comprehensive admin tools for day-to-day management:

### Health & Monitoring
```bash
subtensor-admin health          # Check node health and peer count
subtensor-admin status          # Quick Docker container status
subtensor-monitor sync          # Check sync status across nodes
subtensor-monitor dashboard     # Real-time monitoring overview
subtensor-monitor metrics       # System CPU, memory, disk usage
```

### Backup & Snapshots
```bash
subtensor-admin backup          # Simple container backup
subtensor-admin snapshot        # Compressed LZ4 snapshot (recommended)
subtensor-admin restore         # Interactive restore from snapshots
```

**Snapshot Process:**
- Creates `snapshot_YYYY-MM-DD.tar.lz4` files
- Temporarily stops services for clean snapshots
- Stores in `/opt/subtensor/backups/`
- Typical size: ~10GB compressed

### Service Management
```bash
subtensor-admin restart         # Restart Subtensor services
subtensor-admin logs            # View recent container logs
```

### Daily Operations Example
```bash
# Morning health check
subtensor-admin health

# Create snapshot before maintenance
subtensor-admin snapshot

# Monitor sync progress
subtensor-monitor sync

# Check system resources
subtensor-monitor metrics
```

## Troubleshooting

### Common Issues

**Nodes not syncing:**
```bash
# Check peer connections
subtensor-admin health

# View sync status
subtensor-monitor sync

# Check container logs
subtensor-admin logs
```

**High resource usage:**
```bash
# Check system metrics
subtensor-monitor metrics

# Monitor in real-time
subtensor-monitor dashboard
```

**Service failures:**
```bash
# Check service status
systemctl status subtensor

# Restart services
subtensor-admin restart

# View detailed logs
journalctl -u subtensor -f
```

### Recovery Procedures

**Restore from snapshot:**
```bash
# List available snapshots
subtensor-admin restore

# Follow interactive prompts to select and restore
```

**Manual service recovery:**
```bash
# Stop services
systemctl stop subtensor

# Check docker containers
docker ps -a

# Remove corrupted containers
docker rm subtensor-0 subtensor-1

# Restart deployment
ansible-playbook -i inventory playbook.yml --tags subtensor
```

## Monitoring

Basic monitoring is included with Prometheus and AlertManager:
- **Prometheus**: `http://SERVER:9090`
- **AlertManager**: `http://SERVER:9093`

```bash
# Check monitoring services
systemctl status prometheus alertmanager node-exporter
```

## Configuration Files

Key configuration files to customize for your deployment:

### `group_vars/all.yml`
```yaml
# Basic settings
subtensor_version: "v2.0.10"
subtensor_replicas: 2
subtensor_sync_mode: "fast"

# Network settings
subtensor_network: "finney"
subtensor_bootnodes: "..."

# Resource limits
subtensor_memory_limit: "8g"
subtensor_cpu_limit: "4"

# Tools settings  
tools_enabled: true
```

### `inventory`
```ini
[subtensor]
your-server.example.com ansible_user=ubuntu

[subtensor:vars]
ansible_ssh_private_key_file=~/.ssh/your-key.pem
```

## Documentation

- **[REPLICATION.md](REPLICATION.md)** - Complete step-by-step replication guide
- **[Admin Tools](roles/tools/README.md)** - Simple command reference
- **Configuration examples** in `group_vars/all.yml.example`
- **Inventory template** in `inventory.example`

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the detailed replication guide
3. Verify configuration against examples
