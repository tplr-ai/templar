# Subtensor Production Deployment

Ansible deployment for highly available Subtensor infrastructure.

## Quick Start

```bash
cd scripts/prod-subtensor
ansible-vault encrypt group_vars/vault.yml
ansible-playbook -i inventory playbook.yml --ask-vault-pass
```

## Architecture

This deployment creates a **3-node high availability** Subtensor setup:

- **subtensor-0**: Primary node (ports: 9933, 9944, 30333)
- **subtensor-1**: Secondary node (ports: 9934, 9945, 30334)  
- **subtensor-2**: Tertiary node (ports: 9935, 9946, 30335)

All nodes sync with the official Finney mainnet using warp sync and official bootnodes.

## Configuration

Edit `group_vars/all.yml` for network, replicas, and resource limits.

### Chainspec Management

The deployment automatically downloads the official chainspec from:

```
https://raw.githubusercontent.com/opentensor/subtensor/refs/heads/main/chainspecs/raw_spec_finney.json
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
```

## Management

```bash
# Service status
systemctl status subtensor nginx prometheus alertmanager

# Health checks
/opt/subtensor/scripts/health_check.sh        # Subtensor nodes health
/opt/monitoring/subtensor_health_check.sh     # Monitoring health check

# Logs
journalctl -u subtensor -f                    # Subtensor service logs
journalctl -u subtensor-monitor -f            # Monitoring logs
docker logs subtensor-0 -f                    # Individual node logs

# Watchtower management
/opt/subtensor/scripts/watchtower-manage.sh status       # Check Watchtower status
/opt/subtensor/scripts/watchtower-manage.sh monitor      # Enable monitor-only mode
/opt/subtensor/scripts/watchtower-manage.sh update       # Enable automatic updates
/opt/subtensor/scripts/watchtower-manage.sh force-update # Force immediate update
```

## Monitoring

The deployment includes comprehensive monitoring with Discord alerts.

```bash
# Access monitoring interfaces
# Prometheus: http://SERVER:9090
# AlertManager: http://SERVER:9093

# Monitor all services
systemctl status prometheus alertmanager node-exporter subtensor-monitor
```

For detailed monitoring setup and Discord integration, see [MONITORING.md](MONITORING.md).

## Automated Updates with Blue/Green Deployment

This deployment includes Docker Watchtower for automated container updates using a blue/green rollout strategy.

### Features

- **Zero-downtime updates**: Rolling updates ensure continuous service availability
- **Health checks**: Automated verification of container health before and after updates
- **Automatic rollback**: Failed updates are automatically rolled back
- **Discord notifications**: Real-time alerts for update status
- **Backup creation**: Pre-update backups for easy recovery

### Configuration

Edit `group_vars/all.yml` to configure Watchtower behavior:

```yaml
watchtower_enabled: true
watchtower_poll_interval: 3600  # Check for updates every hour
watchtower_monitor_only: false  # Set to true to only monitor, not update
bluegreen_enabled: true
bluegreen_rollback_on_failure: true
```

### Update Process

1. **Pre-update**: Creates backup, drains traffic, verifies cluster health
2. **Update**: Downloads new image, stops old container, starts new one
3. **Health check**: Verifies new container is healthy and syncing
4. **Post-update**: Restores traffic, sends notifications
5. **Rollback**: If health checks fail, automatically rolls back to previous version

### Management Commands

```bash
# Check update status
/opt/subtensor/scripts/watchtower-manage.sh status

# Force immediate update check
/opt/subtensor/scripts/watchtower-manage.sh force-update

# Switch to monitor-only mode (no automatic updates)
/opt/subtensor/scripts/watchtower-manage.sh monitor

# Enable automatic updates
/opt/subtensor/scripts/watchtower-manage.sh update

# Switch deployment group
/opt/subtensor/scripts/watchtower-manage.sh switch-group green

# View Watchtower logs
/opt/subtensor/scripts/watchtower-manage.sh logs
```

### Safety Features

- **Primary node protection**: Secondary nodes must be healthy before updating primary
- **Gradual rollout**: Updates one node at a time with health verification
- **Traffic management**: Temporarily removes updating nodes from load balancer
- **Backup restoration**: Automatic restoration from backup on critical failures
- **Timeout protection**: Updates that take too long are automatically failed
