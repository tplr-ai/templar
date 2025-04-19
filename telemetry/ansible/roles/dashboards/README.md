# Dashboards Role

This role manages Grafana dashboards and additional metrics collection tools for the Templar project.

## Features

1. **Dashboard Provisioning**: Automatically deploys JSON dashboard definitions to Grafana
2. **Version Reporting**: Collects and displays the current Templar version from GitHub
3. **Metrics Visualization**: Provides comprehensive dashboards for all Templar metrics

## Version Reporter

The Version Reporter is a Python-based utility that:

1. Fetches the current Templar version from GitHub by reading the `__init__.py` file
2. Reports this version to InfluxDB for display in the Grafana dashboard
3. Runs hourly via cron to keep the version information current

### Components

- `version_reporter.py`: Python script that fetches and reports the version
- `report_version.sh`: Shell wrapper for running the Python script
- Cron job: Scheduled task that runs hourly to update the version

### Configuration

The version reporter uses the following environment variables:

- `INFLUXDB_URL`: URL of the InfluxDB server (default: http://localhost:8086)
- `INFLUXDB_TOKEN`: Authentication token for InfluxDB
- `INFLUXDB_ORG`: Organization name in InfluxDB (default: templar)
- `INFLUXDB_BUCKET`: Bucket name in InfluxDB (default: tplr)

These variables can be defined in `/etc/templar/environment` which is sourced by the shell script.

### Dashboard Integration

The Templar version appears at the top of the dashboard in a stat panel that shows:
- The current version number
- When the version was last checked
