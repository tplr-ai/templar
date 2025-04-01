# Templar Telemetry

This directory contains the telemetry infrastructure for the Templar project, allowing you to monitor performance metrics via public-facing Grafana dashboards.

## Components

- **Ansible**: Automation for deploying Grafana with InfluxDB integration
- **Grafana**: Dashboard configurations and provisioning
- **Documentation**: Setup guides and troubleshooting

## Directory Structure

```
telemetry/
├── ansible/                   # Ansible automation for deployment
├── grafana/                   # Grafana dashboard configurations
│   └── provisioning/
│       ├── dashboards/        # JSON dashboard definitions
│       └── datasources/       # Data source configurations
```

## Getting Started

1. **Prerequisites**:
   - A server with SSH access
   - InfluxDB instance with telemetry data
   - Basic knowledge of Ansible

2. **Deployment**:
   ```bash
   cd telemetry/ansible
   ansible-playbook -i inventories/production playbooks/deploy_grafana.yml
   ```

3. **Accessing Dashboards**:
   - After deployment, dashboards will be available at: `http://your-server-ip/`
   - No login required - dashboards are publicly accessible in read-only mode

## Custom Dashboards

To add a custom dashboard:

1. Export your dashboard from Grafana as JSON
2. Place the JSON file in `grafana/provisioning/dashboards/`
3. Run the update playbook:
   ```bash
   ansible-playbook -i inventories/production playbooks/update_dashboards.yml
   ```

## Documentation

For detailed documentation on the deployment process, see:
- [Ansible README](ansible/README.md) - Deployment instructions
- [Grafana Setup](grafana/README.md) - Dashboard configuration

## Security Considerations

- The Grafana instance is configured for public read-only access
- Admin interfaces are protected from public access
- All sensitive endpoints are restricted
- Regular security updates are applied through the maintenance playbook