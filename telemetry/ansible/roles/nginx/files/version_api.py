#!/usr/bin/env python3
"""
Simple Version API for Templar

This script fetches the current Templar version from GitHub and serves it via a simple HTTP API.
"""

import json
import logging
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# Handle missing requests library gracefully
try:
    import requests
except ImportError:
    print("Error: 'requests' package not found. Installing...")
    try:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests

        print("Successfully installed 'requests' package.")
    except Exception as e:
        print(f"Failed to install 'requests' package: {e}")
        print("Please install it manually with: pip3 install requests")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("templar_version_api")

# Configuration
GITHUB_URL = "https://raw.githubusercontent.com/tplr-ai/templar/refs/heads/main/src/tplr/__init__.py"
CACHE_TTL = 3600  # Cache version for 1 hour
PORT = 8585  # The port the API will run on

# Cache for the version
cached_version = None
last_fetch_time = 0


def fetch_version():
    """Fetch the current Templar version from GitHub."""
    global cached_version, last_fetch_time

    current_time = time.time()

    # Return cached version if available and not expired
    if cached_version and (current_time - last_fetch_time) < CACHE_TTL:
        return cached_version

    try:
        response = requests.get(GITHUB_URL, timeout=10)
        response.raise_for_status()

        # Extract version using regex
        version_match = re.search(r'__version__\s*=\s*"([^"]+)"', response.text)
        if version_match:
            cached_version = version_match.group(1)
            last_fetch_time = current_time
            logger.info(f"Updated version cache to: {cached_version}")

            # Update dashboard JSON file with the current version
            try:
                update_dashboard_version(cached_version)
            except Exception as e:
                logger.error(f"Error updating dashboard version: {e}")

            return cached_version
        else:
            logger.error("Version pattern not found in response")
            return cached_version or "Unknown"
    except requests.RequestException as e:
        logger.error(f"Error fetching version from GitHub: {e}")
        return cached_version or "Unknown"


def update_dashboard_version(version):
    """Update the dashboard JSON file with the current version."""
    # Check both possible dashboard paths
    dashboard_paths = [
        "/etc/grafana/dashboards/templar_metrics.json",  # Standard path
        "/var/lib/grafana/dashboards/templar_metrics.json",  # Alternative path
    ]

    updated = False

    for dashboard_path in dashboard_paths:
        try:
            if os.path.exists(dashboard_path):
                with open(dashboard_path, "r") as f:
                    dashboard_json = json.load(f)

                # Find the version panel
                for panel in dashboard_json.get("panels", []):
                    if panel.get("id") == 1:  # ID of the version panel (first panel)
                        content = panel.get("options", {}).get("content", "")

                        # Don't update the content as it's using template variables
                        # Just keep it as is - Grafana will handle variable substitution
                        updated_content = content

                        # Don't modify the panel content - Grafana will handle variable substitution
                        # Just keep the content as is to avoid interfering with Grafana variables

                        panel["options"]["content"] = updated_content
                        break

                # Write the updated dashboard back to the file
                with open(dashboard_path, "w") as f:
                    json.dump(dashboard_json, f, indent=2)

                logger.info(
                    f"Updated dashboard version to: v{version} in {dashboard_path}"
                )
                updated = True

        except Exception as e:
            logger.error(f"Failed to update dashboard version in {dashboard_path}: {e}")

    if not updated:
        error_msg = "Could not update dashboard in any of the expected locations"
        logger.error(error_msg)
        raise Exception(error_msg)


class VersionHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests to the API."""
        if self.path == "/version":
            # JSON response
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            version = fetch_version()
            response = json.dumps({"version": version})
            self.wfile.write(response.encode())

        elif self.path == "/version/plain":
            # Plain text response
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            version = fetch_version()
            self.wfile.write(version.encode())

        elif self.path == "/health":
            # Health check endpoint
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")

        else:
            # Not found
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

    pass


def main():
    """Run the HTTP server."""
    server = ThreadedHTTPServer(("0.0.0.0", PORT), VersionHandler)
    logger.info(f"Starting version API server on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
