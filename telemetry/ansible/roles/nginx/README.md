# NGINX Role

This role configures NGINX as a reverse proxy for Grafana and other services in the Templar telemetry stack.

## Features

1. **Reverse Proxy for Grafana**: Routes web traffic to the Grafana instance
2. **SSL Support**: Optional HTTPS configuration with SSL certificates
3. **Kiosk Mode**: Automatic redirection to dashboard kiosk mode for clean UI
4. **Firewall Configuration**: Sets up UFW rules for secure access
5. **Version API**: Provides endpoints to retrieve the current Templar version

## Version API

The Version API is a lightweight HTTP service that:

1. Fetches the current Templar version from GitHub by reading the `__init__.py` file
2. Caches the version information to reduce GitHub API usage
3. Provides both JSON and plain text endpoints for consuming the version

### API Endpoints

The following endpoints are available:

- `/api/templar/version`: Returns JSON format `{"version": "X.Y.Z"}`
- `/api/templar/version/plain`: Returns plain text version number
- `/api/templar/health`: Health check endpoint that returns 200 OK if the service is running

### Implementation

1. **Python Service**: Simple HTTP server running on port 8585 (only accessible locally)
2. **NGINX Proxy**: Routes external requests to the internal service with caching
3. **Caching**: Responses are cached for 1 hour to reduce load on GitHub and the API service
4. **Auto-start**: Configured as a systemd service that starts on boot

### Dashboard Integration

The Templar version appears at the top of the dashboard in a text panel that uses JavaScript to:

1. Fetch the version from the API endpoint
2. Update the HTML display with the current version
3. Refresh periodically to ensure the version is current

This implementation works in Grafana's text panels with HTML mode, as it uses the browser's fetch API to get the version directly from the same server hosting Grafana.
