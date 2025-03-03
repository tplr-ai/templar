const fs = require('fs');
const path = require('path');
require('dotenv').config();

// Debug log: Print the critical environment variables to verify .env is loaded.
console.debug("Loaded INFLUXDB_URL:", process.env.INFLUXDB_URL);
console.debug("Loaded INFLUXDB_TOKEN:", process.env.INFLUXDB_TOKEN);
console.debug("Loaded INFLUXDB_ORG:", process.env.INFLUXDB_ORG);
console.debug("Loaded INFLUXDB_BUCKET:", process.env.INFLUXDB_BUCKET);

/**
 * Parse version from src/tplr/__init__.py
 */
function getVersionFromInit() {
  try {
    // Adjust relative path: from /api/src/config to /src/tplr/__init__.py
    const initPath = path.join(__dirname, '../../../src/tplr/__init__.py');
    const content = fs.readFileSync(initPath, 'utf8');
    const match = content.match(/__version__\s*=\s*["']([^"']+)["']/);
    if (match) {
      return match[1];
    }
  } catch (err) {
    console.error('Error reading __init__.py:', err);
  }
  return 'default_version';
}

/**
 * Adds protocol if missing.
 */
function addProtocolIfMissing(url) {
  if (!url.match(/^(https?:)\/\//)) {
    return `https://${url}`;
  }
  return url;
}

/**
 * Normalizes the URL by adding the default port (8086) if not set.
 */
function normalizeUrl(url) {
  let normalized = addProtocolIfMissing(url);
  try {
    const parsed = new URL(normalized);
    if (!parsed.port) {
      parsed.port = "8086";
      return parsed.toString();
    }
    return normalized;
  } catch (e) {
    console.error("Error normalizing URL:", e);
  }
  return normalized;
}

const influxConfig = {
  url: normalizeUrl(process.env.INFLUXDB_URL || 'https://default-url:8086'),
  token: process.env.INFLUXDB_TOKEN,
  org: process.env.INFLUXDB_ORG || 'templar',
  bucket: process.env.INFLUXDB_BUCKET || 'tplr',
  version: process.env.INFLUXDB_VERSION || getVersionFromInit(),
};

module.exports = influxConfig;