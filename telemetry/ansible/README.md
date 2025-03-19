# Grafana Deployment with Ansible

This Ansible project provides a clean, modular approach to deploying Grafana with InfluxDB integration on Ubuntu 24.04, optimized for public read-only access with high traffic capabilities.

## Directory Structure

```
ansible/
├── ansible.cfg                  # Ansible configuration
├── inventory                    # Server inventory 
├── playbook.yml                 # Main playbook
├── roles/                       # Modular roles
│   ├── grafana/                 # Grafana installation and configuration
│   │   ├── tasks/               # Tasks for Grafana setup
│   │   ├── templates/           # Grafana configuration templates
│   │   ├── files/               # Static files for Grafana
│   │   └── handlers/            # Handlers for Grafana service
│   ├── nginx/                   # NGINX reverse proxy
│   │   ├── tasks/               # NGINX installation and setup
│   │   ├── templates/           # NGINX configuration templates
│   │   ├── files/               # SSL certificates and static files
│   │   └── handlers/            # Handlers for NGINX service
│   └── dashboards/              # Dashboard provisioning
│       ├── tasks/               # Dashboard deployment tasks
│       ├── files/               # Dashboard JSON files
│       └── handlers/            # Dashboard reload handlers
├── group_vars/                  # Global variables
│   ├── all.yml                  # Common settings
│   └── vault.yml                # Encrypted secrets (not in git)
└── host_vars/                   # Host-specific variables
    └── grafana_prod.yml         # Server-specific configuration
```

## Features

- **Public Read-Only Access**: Grafana is configured to allow anonymous users to view dashboards without login
- **InfluxDB Integration**: Connects to your existing InfluxDB instance
- **Dashboard Provisioning**: Automatically loads dashboards from JSON files
- **NGINX Reverse Proxy**: Handles access control and performance optimization
- **Security**: Restricts access to admin areas while allowing public read-only access
- **Performance Tuning**: Optimized for hundreds of concurrent users
- **Live Updates**: Configured for real-time data visualization

## Prerequisites

1. Ubuntu 24.04 server with SSH access
2. Python 3.x installed on the control node
3. InfluxDB running and accessible (managed service)
4. Dashboard designs in JSON format

## Quick Start

1. **Configure the inventory**:
   - Edit `inventory` to set your server IP
   - Update InfluxDB connection in `group_vars/all.yml` (already configured for AWS InfluxDB)
   - Create `group_vars/vault.yml` from the example and encrypt it with `ansible-vault encrypt group_vars/vault.yml`
   - Customize server-specific settings in `host_vars/grafana_prod.yml`

2. **Add your dashboards**:
   - Place dashboard JSON files in `roles/dashboards/files/`
   - Make sure to include `templar_metrics.json` as the home dashboard

3. **Deploy Grafana**:
   ```bash
   # Run the playbook with your SSH key
   ansible-playbook -i inventory playbook.yml --private-key=~/.ssh/your_key.pem

   # Or if using a vault password file
   ansible-playbook -i inventory playbook.yml --private-key=~/.ssh/your_key.pem --vault-password-file=./vault_pass.txt
   ```

4. **Update dashboards only**:
   ```bash
   ansible-playbook -i inventory playbook.yml --tags dashboards --private-key=~/.ssh/your_key.pem
   ```

5. **Access Grafana**:
   - Public access (no login required): http://your-server-ip
   - Direct Grafana access: http://your-server-ip:3000

## Configuration Options

### Main Configuration Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `grafana_version` | Grafana version to install | 10.4.0 |
| `grafana_http_port` | Grafana service port | 3000 |
| `grafana_domain` | Server domain/hostname | localhost |
| `grafana_influxdb_host` | InfluxDB host | localhost |
| `grafana_influxdb_port` | InfluxDB port | 8086 |
| `grafana_influxdb_database` | InfluxDB database name | metrics |

### Performance Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `grafana_memory_limit` | Memory limit for Grafana | 2GB |
| `grafana_cpu_quota` | CPU quota for Grafana | 200% |
| `grafana_max_connections` | Max connections for live updates | 500 |
| `nginx_worker_connections` | NGINX worker connections | 4096 |
| `grafana_concurrent_render_limit` | Concurrent rendering limit | 30 |

### Security Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `grafana_auth_anonymous_enabled` | Enable anonymous access | true |
| `grafana_auth_anonymous_org_role` | Role for anonymous users | Viewer |
| `nginx_ssl_enabled` | Enable SSL | false |

## Managing Dashboards

Your dashboard JSON files should be placed in `roles/dashboards/files/`. The Ansible playbook will automatically:
1. Copy these dashboards to the Grafana server
2. Set proper permissions
3. Configure Grafana to load them
4. Restart/reload services as needed

## Troubleshooting

- **NGINX or Grafana not starting**: Check logs with `journalctl -u nginx` or `journalctl -u grafana-server`
- **Dashboards not showing**: Verify file permissions and dashboard JSON format
- **Performance issues**: Adjust the performance parameters in `group_vars/all.yml`
- **Connection errors**: Check InfluxDB connection settings

### Troubleshooting Server Issues

If you encounter errors with Grafana:

1. **Basic Grafana Service Troubleshooting**:
   ```bash
   # Check service status
   systemctl status grafana-server
   
   # View logs
   journalctl -u grafana-server -n 50
   
   # Check if Grafana is listening
   netstat -tulpn | grep 3000
   ```

2. **Permission Issues**:
   If you see permission-related errors:
   ```bash
   # Fix ownership
   sudo chown -R grafana:grafana /var/lib/grafana /etc/grafana /var/log/grafana
   
   # Fix permissions
   sudo chmod -R 755 /var/lib/grafana /etc/grafana /var/log/grafana
   
   # Restart service
   sudo systemctl restart grafana-server
   ```

3. **NGINX Issues**:
   If you see 502 Bad Gateway errors:
   ```bash
   # Check NGINX config
   sudo nginx -t
   
   # Check NGINX logs
   sudo tail -f /var/log/nginx/error.log
   
   # Test direct Grafana access
   curl -v http://localhost:3000/api/health
   ```

4. **Basic Recovery Steps**:
   If needed, restart the services:
   ```bash
   sudo systemctl restart grafana-server
   sudo systemctl restart nginx
   ```

### Authentication and Access Issues

If you encounter 403 Forbidden errors or redirection to login page when trying to access dashboards as an anonymous user:

1. **Check NGINX Configuration**:
   - Ensure `/api/datasources/proxy` and `/api/ds/query` endpoints are accessible
   - The updated `roles/nginx/templates/grafana-nginx.conf.j2` should handle this correctly:
     - Added WebSocket support for live updates
     - Set proper buffer settings for high traffic
     - Set appropriate timeouts for query operations
     - Properly configured proxy headers for auth and CORS

2. **Ensure Complete Public Access (No Login Required)**:
   - The updated configuration handles this through multiple layers:
   
   a. Environment variables (via grafana-env.j2):
   ```
   GF_AUTH_ANONYMOUS_ENABLED=true
   GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
   GF_AUTH_DISABLE_LOGIN_FORM=true
   GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/etc/grafana/dashboards/templar_metrics.json
   ```
   
   b. Grafana.ini settings (already updated):
   ```ini
   [auth.anonymous]
   enabled = true
   org_name = Templar AI
   org_role = Viewer
   hide_version = true
   
   [auth]
   disable_login_form = true
   
   [dashboards]
   default_home_dashboard_path = /etc/grafana/dashboards/templar_metrics.json
   ```
   
   c. NGINX configuration (improved to handle WebSockets properly):
   - Added WebSocket proxy settings for live updates
   - Optimized buffer settings for better performance
   - Set appropriate timeouts for database queries

3. **Ensure Dashboard is Accessible**:
   - Make sure the dashboard JSON is deployed to the exact location specified in the default_home_dashboard_path
   - Verify permissions on the dashboard file

4. **Check InfluxDB Connection**:
   - Verify correct token authentication in `influxdb-datasource.yml.j2`
   - Ensure organization and bucket settings match your InfluxDB instance

5. **After Making Changes**:
   - Redeploy the configuration: `ansible-playbook -i inventory playbook.yml`
   - Check NGINX logs: `tail -f /var/log/nginx/grafana.error.log`
   - Check Grafana logs: `tail -f /var/log/grafana/grafana.log`

## Security Notes

- Admin areas are protected from public access
- API access is limited to read-only operations for anonymous users
- Write operations are restricted to localhost
- Anonymous users only have Viewer permissions