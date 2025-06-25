# Subtensor Node Replication Guide

Quick guide to replicate this Subtensor production setup on any Ubuntu server.

## 🚀 Quick Setup (10 minutes)

### 1. Prepare Your Server

```bash
# Ubuntu 24.04 LTS server required
# Minimum: 16GB RAM, 500GB storage

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

### 2. Configure Deployment

```bash
# Clone this repository
git clone <repo-url>
cd templar-prod-subtensor/scripts/prod-subtensor

# Copy example files
cp inventory.example inventory
cp group_vars/all.yml.example group_vars/all.yml

# Edit inventory with your server IP
vim inventory
```

### 3. Customize Configuration

Edit `group_vars/all.yml`:

```yaml
# Required settings
subtensor_version: "v2.0.10"
subtensor_replicas: 2
subtensor_user: "subtensor"
subtensor_group: "subtensor"

# Network (use defaults for mainnet)
subtensor_network: "finney" 
subtensor_sync_mode: "fast"

# Optional: Resource limits
subtensor_memory_limit: "8g"
subtensor_cpu_limit: "4"

# Optional: Enable monitoring
monitoring_enabled: true
prometheus_enabled: true

# Admin tools (recommended)
tools_enabled: true
```

### 4. Deploy

```bash
# Deploy everything
ansible-playbook -i inventory playbook.yml

# Or deploy incrementally
ansible-playbook -i inventory playbook.yml --tags subtensor
ansible-playbook -i inventory playbook.yml --tags tools
ansible-playbook -i inventory playbook.yml --tags monitoring
```

### 5. Verify

```bash
# Check node health
ansible your-server -i inventory -m shell -a "subtensor-admin health" --become

# Monitor sync status
ansible your-server -i inventory -m shell -a "subtensor-monitor sync" --become

# Create first snapshot
ansible your-server -i inventory -m shell -a "subtensor-admin snapshot" --become
```

## 📊 Expected Results

After successful deployment:

- **2 Subtensor nodes** running in Docker containers
- **Load balancer** (nginx) distributing traffic
- **Monitoring** with Prometheus/AlertManager  
- **Admin tools** for management and backups
- **Automatic startup** via systemd

### Ports Used

- `9933, 9934` - RPC endpoints
- `9944, 9945` - WebSocket endpoints  
- `30333, 30334` - P2P networking
- `9615, 9616` - Prometheus metrics
- `80, 443` - HTTP/HTTPS (nginx)

### Data Storage

- Blockchain data: `/var/snap/docker/common/var-lib-docker/volumes/`
- Backups: `/opt/subtensor/backups/`
- Logs: `/var/log/subtensor/`

## 🔧 Daily Operations

```bash
# Health check
subtensor-admin health

# Create backup
subtensor-admin snapshot

# Check sync status
subtensor-monitor sync

# View system metrics
subtensor-monitor metrics

# Check logs if issues
subtensor-admin logs
```

## 📋 Requirements Checklist

- [ ] Ubuntu 24.04 LTS server
- [ ] 16GB+ RAM
- [ ] 500GB+ storage  
- [ ] Docker installed
- [ ] Ansible configured locally
- [ ] SSH access to target server
- [ ] Ports 9933-9945, 30333-30334 open
- [ ] Internet connectivity for sync

## 🆘 Troubleshooting

**Deployment fails:**

- Check SSH connectivity: `ansible your-server -i inventory -m ping`
- Verify sudo access: `ansible your-server -i inventory -m shell -a "sudo whoami"`

**Nodes not syncing:**

- Check peer connections: `subtensor-admin health`
- View container logs: `subtensor-admin logs`
- Restart services: `subtensor-admin restart`

**Out of disk space:**

- Check usage: `subtensor-monitor metrics`
- Create snapshot and archive old data
- Consider increasing storage

---

**Estimated sync time:** 6-12 hours for full sync  
**Disk usage:** ~50GB blockchain data + snapshots  
**Memory usage:** ~8GB per node during sync
