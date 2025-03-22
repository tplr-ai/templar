# Templar Version Plugin for Grafana

This Grafana plugin displays the Templar server version information at the top of your Grafana dashboards. The plugin fetches version data from the Templar API endpoint and makes it available in Grafana.

## Features

- Fetches version information from the Templar API
- Displays version number in dashboards
- Configurable API endpoint

## Development

### Prerequisites

- Node.js 16+
- npm or yarn

### Getting Started

1. Install dependencies
   ```bash
   npm install
   ```

2. Build the plugin
   ```bash
   npm run build
   ```

3. For development with hot-reloading:
   ```bash
   npm run dev
   ```

## Installation in Grafana

### Local Installation

1. Build the plugin as described above
2. Create a directory for the plugin in your Grafana plugins directory:
   ```bash
   mkdir -p /var/lib/grafana/plugins/templar-version-datasource
   ```
3. Copy the contents of the `dist` directory to the plugin directory:
   ```bash
   cp -r dist/* /var/lib/grafana/plugins/templar-version-datasource/
   ```
4. Add the plugin to Grafana's allowed unsigned plugins list in `grafana.ini`:
   ```ini
   [plugins]
   allow_loading_unsigned_plugins = templar-version-datasource
   ```
5. Restart Grafana:
   ```bash
   sudo systemctl restart grafana-server
   ```

### Docker Installation

If you're running Grafana in Docker, you can mount the plugin directory:

```bash
docker run -d -p 3000:3000 \
  -v /path/to/plugin/dist:/var/lib/grafana/plugins/templar-version-datasource \
  -e "GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=templar-version-datasource" \
  grafana/grafana
```

## Usage

### Adding the Data Source

1. In Grafana, go to Configuration > Data Sources > Add data source
2. Search for "Templar Version" and select it
3. Configure the URL (default is http://18.217.218.11/api/templar/version)
4. Click "Save & Test" to verify the connection

### Adding to a Dashboard

1. Create a new dashboard or edit an existing one
2. Add a new panel
3. In the panel configuration, select "Templar Version" as the data source
4. Choose "Text" visualization to display the version information
5. In the text panel settings, use the following formatting template:
   ```
   Templar Version: ${version}
   ```
6. Position the panel at the top of your dashboard
7. Save the dashboard