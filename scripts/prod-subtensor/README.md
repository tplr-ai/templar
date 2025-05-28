# Subtensor Production Deployment

Ansible deployment for highly available Subtensor infrastructure.

## Quick Start

```bash
cd scripts/prod-subtensor
ansible-vault encrypt group_vars/vault.yml
ansible-playbook -i inventory playbook.yml --ask-vault-pass
```

## Configuration

Edit `group_vars/all.yml` for network, replicas, and resource limits.

## Commands

```bash
# Full deployment
ansible-playbook -i inventory playbook.yml --ask-vault-pass

# Specific components
ansible-playbook -i inventory playbook.yml --tags nginx
ansible-playbook -i inventory playbook.yml --tags subtensor
```

## Management

```bash
# Service status
systemctl status subtensor nginx

# Health check
/opt/subtensor/scripts/health_check.sh

# Logs
journalctl -u subtensor -f
```
