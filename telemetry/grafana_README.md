# Grafana Configuration

This document describes the Grafana dashboard configuration that will be deployed by the Ansible scripts.

## Dashboard Structure

The deployment will configure Grafana to display:

- **Overview Metrics**: Active peers, gather success rate
- **Validator Metrics**: Loss measurements, improvements, weights
- **Miner Metrics**: Gradient norms, learning rates, operation times
- **Model Evaluation**: Benchmark results, gradient analysis
- **System Metrics**: CPU usage, memory, GPU resources

## Provisioning Process

During deployment:

1. Grafana is installed and configured for anonymous access
2. Dashboard provisioning is set up to automatically load dashboard JSON files
3. The existing dashboard JSON files from the Templar project are copied to Grafana
4. InfluxDB is configured as a data source

## Dashboard Provider Configuration

The Ansible deployment configures the dashboard provider with:

```yaml
apiVersion: 1
providers:
  - name: 'default'
    orgId: 1
    folder: 'Public Dashboards'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: false  # Prevent UI modifications since dashboards are managed by Ansible
    options:
      path: /etc/grafana/provisioning/dashboards
      foldersFromFilesStructure: true
```

## InfluxDB Integration

The dashboards are configured to query data from InfluxDB, using:

- Bucket: "tplr"
- Measurements: "training_step", "validator_window", "benchmark_task", etc.
- Fields: Various metrics including loss, gradients, system metrics, etc.

## Adding New Dashboards

To add a new dashboard to the deployment:

1. Create the dashboard JSON file
2. Place it in the source dashboards directory
3. Deploy using the update_dashboards playbook
4. For manual updates, import the dashboard through the Grafana UI

## Anonymous Access Configuration

Grafana is configured for anonymous access with:

```yaml
grafana_auth_anonymous_enabled: true
grafana_auth_anonymous_org_name: "Main Org."
grafana_auth_anonymous_org_role: "Viewer"
grafana_auth_anonymous_hide_version: true
```

This allows public read-only access to all dashboards without requiring login.
