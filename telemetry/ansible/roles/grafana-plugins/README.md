# Grafana Plugins Role

This Ansible role installs and configures custom Grafana plugins for the Templar project.

## Plugins Installed

- **marcusolsson-json-datasource**: A plugin that fetches and displays Templar version information from the API endpoint.

## Requirements

- Grafana should be installed and running (provided by the `grafana` role)
- Plugin should be built before deployment (compiled version in the `dist` directory)

## Role Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `templar_version_api_url` | URL for the Templar version API | `http://18.217.218.11/api/templar/version` |
| `grafana_plugins_to_allow` | List of unsigned plugins to allow | `["marcusolsson-json-datasource"]` |

## How It Works

1. Creates necessary directory structure for the plugin
2. Copies the plugin files to the Grafana plugins directory
3. Updates Grafana configuration to allow unsigned plugins
4. Restarts Grafana service if configuration changed
5. Adds the Templar Version datasource automatically

## Dependencies

- `grafana` role must be applied before this role

## Example Usage

To use this role, include it in your playbook after the Grafana role:

```yaml
- hosts: grafana_servers
  roles:
    - role: grafana
    - role: grafana-plugins
```
