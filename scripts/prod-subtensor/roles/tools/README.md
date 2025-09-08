# Subtensor Admin Tools

Simple command-line tools for managing Subtensor nodes.

## Installation

Deploy via Ansible:

```bash
ansible-playbook -i inventory playbook.yml --tags tools
```

## Commands

### subtensor-admin

- `health` - Check node health
- `status` - Docker container status  
- `logs` - View recent logs
- `restart` - Restart services
- `backup` - Create backup
- `snapshot` - Take compressed snapshot
- `restore` - Restore from snapshot

### subtensor-monitor

- `sync` - Check sync status
- `metrics` - System metrics
- `dashboard` - Quick overview

## Examples

```bash
subtensor-admin health
subtensor-monitor sync
subtensor-admin snapshot
subtensor-admin restore
```
