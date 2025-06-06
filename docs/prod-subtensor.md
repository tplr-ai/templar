# Production Subtensor Deployment

Ansible-based deployment for production-ready Subtensor infrastructure with monitoring, SSL, and high availability.

## Overview

The prod-subtensor deployment provides:

- Dockerized Subtensor nodes with automatic health checks
- Nginx reverse proxy with SSL termination
- Prometheus monitoring with AlertManager and Discord integration
- Automated backup and maintenance scripts
- Security hardening with fail2ban

## Prerequisites

- Ubuntu 20.04+ target servers
- Ansible 2.9+
- Docker and Docker Compose on target hosts
- SSL certificates (Let's Encrypt or custom)

## Quick Start

```bash
cd scripts/prod-subtensor
cp inventory.example inventory
cp group_vars/all.yml.example group_vars/all.yml
ansible-vault create group_vars/vault.yml
ansible-playbook -i inventory playbook.yml --ask-vault-pass
```

## Configuration

### Main Configuration (`group_vars/all.yml`)

```yaml
# Network configuration
subtensor_network: "finney"
subtensor_chain_endpoint: "wss://entrypoint-finney.opentensor.ai:443"

# Resource limits
subtensor_replicas: 2
subtensor_memory_limit: "8g"
subtensor_cpu_limit: "4"

# Monitoring
monitoring_enabled: true
discord_webhook_enabled: true
```

### Secrets (`group_vars/vault.yml`)

```yaml
vault_discord_webhook_url: "https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN"
vault_ssl_email: "admin@example.com"
vault_backup_s3_key: "backup_access_key"
```

## Deployment Commands

```bash
# Full deployment
ansible-playbook -i inventory playbook.yml --ask-vault-pass

# Deploy specific components
ansible-playbook -i inventory playbook.yml --tags subtensor
ansible-playbook -i inventory playbook.yml --tags nginx
ansible-playbook -i inventory playbook.yml --tags monitoring

# Update configuration only
ansible-playbook -i inventory playbook.yml --tags config
```

## Service Management

### System Services

```bash
# Subtensor service
sudo systemctl status subtensor
sudo systemctl restart subtensor
sudo systemctl logs -f subtensor

# Nginx service
sudo systemctl status nginx
sudo systemctl reload nginx

# Docker services
docker-compose -f /opt/subtensor/docker-compose.yml ps
docker-compose -f /opt/subtensor/docker-compose.yml logs -f
```

### Health Monitoring

```bash
# Run Subtensor health check
/opt/subtensor/scripts/health_check.sh

# Run monitoring health check (includes Discord alerts)
/opt/monitoring/subtensor_health_check.sh

# Check individual node status
curl -s -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"system_health","params":[],"id":1}' http://localhost:9933

# View Prometheus metrics
curl -s http://localhost:9090/metrics
curl -s http://localhost:8080/metrics  # Custom Subtensor metrics
```

## Backup and Maintenance

### Automated Backups

Backups run daily via cron:

```bash
# View backup status
/opt/subtensor/scripts/backup.sh --status

# Manual backup
/opt/subtensor/scripts/backup.sh --manual
```

### Maintenance Tasks

```bash
# Database maintenance
/opt/subtensor/scripts/maintenance.sh --vacuum

# Update subtensor image
/opt/subtensor/scripts/maintenance.sh --update

# Rotate logs
/opt/subtensor/scripts/maintenance.sh --rotate-logs
```

## Monitoring and Alerting

### Monitoring Stack

Access monitoring interfaces:

- **Prometheus**: `http://your-server:9090` - Metrics collection and querying
- **AlertManager**: `http://your-server:9093` - Alert routing and management
- **Node Exporter**: `http://your-server:9100` - System metrics
- **Custom Metrics**: `http://your-server:8080` - Subtensor-specific metrics

### Discord Integration

Real-time alerts sent to Discord with severity levels:

- 🚨 **Critical**: All nodes down, major service failures
- ⚠️ **Warning**: Individual node issues, high resource usage
- ℹ️ **Info**: Service restarts, maintenance notifications

```bash
# Test Discord webhook manually
python3 /opt/monitoring/discord_notifier.py \
  --webhook-url "YOUR_WEBHOOK_URL" \
  --severity "info" \
  --title "Test Alert" \
  --message "Testing Discord integration"
```

### Key Metrics

- **Blockchain Metrics**: Best block height, finalized blocks, sync status
- **Network Metrics**: Peer count, RPC latency, network connectivity
- **System Metrics**: Memory/CPU usage, disk space, network I/O
- **Custom Metrics**: Subtensor-specific blockchain metrics

### Alerting Rules

Configured alerts for:

- All Subtensor nodes offline (Critical)
- Individual node failures (Warning)
- High block lag (>10 blocks behind)
- Low peer count (<5 peers)
- High resource usage (CPU/Memory/Disk >80%)
- RPC endpoint failures

## Security Features

### SSL/TLS Configuration

- Automatic Let's Encrypt certificate generation
- SSL certificate auto-renewal
- Strong cipher suites and HSTS headers

### Network Security

- Fail2ban protection against brute force
- UFW firewall with minimal open ports
- Docker network isolation

### Access Control

- SSH key-based authentication only
- Sudo access logging
- Regular security updates

## Troubleshooting

### Common Issues

1. **Node not syncing**

   ```bash
   # Check network connectivity
   curl -s https://entrypoint-finney.opentensor.ai:443
   
   # Restart subtensor service
   sudo systemctl restart subtensor
   ```

2. **High memory usage**

   ```bash
   # Check database size
   du -sh /opt/subtensor/data/
   
   # Run maintenance
   /opt/subtensor/scripts/maintenance.sh --vacuum
   ```

3. **SSL certificate issues**

   ```bash
   # Renew certificates
   certbot renew --dry-run
   
   # Check certificate status
   openssl x509 -in /etc/letsencrypt/live/domain/cert.pem -text -noout
   ```

4. **Discord alerts not working**

   ```bash
   # Check Discord webhook URL in vault
   ansible-vault view group_vars/vault.yml
   
   # Test webhook manually
   python3 /opt/monitoring/discord_notifier.py \
     --webhook-url "YOUR_URL" --severity info --title "Test" --message "Test"
   
   # Check monitoring service status
   systemctl status subtensor-monitor
   journalctl -u subtensor-monitor -n 50
   ```

5. **Monitoring services not starting**

   ```bash
   # Check all monitoring services
   systemctl status prometheus alertmanager node-exporter subtensor-monitor
   
   # Validate Prometheus configuration
   /usr/local/bin/promtool check config /opt/monitoring/config/prometheus.yml
   
   # Check monitoring logs
   journalctl -u subtensor-monitor -f
   ```

### Log Locations

- **Subtensor logs**: `journalctl -u subtensor -f`
- **Monitoring logs**: `journalctl -u subtensor-monitor -f`
- **Prometheus logs**: `journalctl -u prometheus -f`
- **AlertManager logs**: `journalctl -u alertmanager -f`
- **Nginx logs**: `/var/log/nginx/access.log` and `/var/log/nginx/error.log`
- **Docker logs**: `docker-compose -f /opt/subtensor/docker-compose.yml logs -f`
- **Health check logs**: `/opt/monitoring/health_check.log`

## Performance Tuning

### Database Optimization

```yaml
# In group_vars/all.yml
subtensor_db_cache_size: "2GB"
subtensor_db_connections: 100
subtensor_pruning_mode: "archive"
```

### Resource Limits

```yaml
# Memory and CPU limits
subtensor_memory_limit: "8g"
subtensor_memory_reservation: "4g"
subtensor_cpu_limit: "4"
subtensor_cpu_reservation: "2"
```

## Development and Testing

### Local Testing

```bash
# Test playbook syntax
ansible-playbook --syntax-check playbook.yml

# Dry run deployment
ansible-playbook -i inventory playbook.yml --check

# Test specific roles
ansible-playbook -i inventory playbook.yml --tags subtensor --check
```

### Staging Environment

Copy production configuration for staging:

```bash
cp group_vars/all.yml group_vars/staging.yml
# Modify staging-specific settings
ansible-playbook -i staging_inventory playbook.yml -e @group_vars/staging.yml
```

## Related Documentation

- [Validator Setup](validator.md) - For validator-specific configuration
- [Monitoring Details](../scripts/prod-subtensor/MONITORING.md) - Complete monitoring and Discord setup guide
- [Telemetry System](../telemetry/README.md) - General telemetry and metrics collection
- [Local Development](localnet.md) - For local testing setup
