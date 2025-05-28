# Production Subtensor Deployment

Ansible-based deployment for production-ready Subtensor infrastructure with monitoring, SSL, and high availability.

## Overview

The prod-subtensor deployment provides:
- Dockerized Subtensor nodes with automatic health checks
- Nginx reverse proxy with SSL termination
- Prometheus monitoring with Grafana dashboards
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
enable_monitoring: true
grafana_admin_password: "{{ vault_grafana_password }}"
```

### Secrets (`group_vars/vault.yml`)

```yaml
vault_grafana_password: "secure_password"
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
# Run health check
/opt/subtensor/scripts/health_check.sh

# Check node sync status
curl -s http://localhost:9944/health | jq .

# View metrics
curl -s http://localhost:9615/metrics
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

### Grafana Dashboards

Access Grafana at `https://your-domain/grafana`
- Subtensor Node Dashboard
- System Metrics Dashboard
- Network Health Dashboard

### Key Metrics

- Block height and sync status
- Memory and CPU usage
- Network connections
- Transaction pool size
- Database size and performance

### Alerting Rules

Configured alerts for:
- Node offline/unhealthy
- High memory usage (>80%)
- Sync lag (>10 blocks behind)
- SSL certificate expiration

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

### Log Locations

- Subtensor logs: `/var/log/subtensor/`
- Nginx logs: `/var/log/nginx/`
- System logs: `journalctl -u subtensor`
- Docker logs: `docker-compose logs`

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
- [Monitoring Setup](../telemetry/README.md) - For detailed monitoring configuration
- [Local Development](localnet.md) - For local testing setup