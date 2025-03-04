import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get the directory name of the current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Debug log: Print the critical environment variables to verify .env is loaded.
console.debug("Loaded INFLUXDB_URL:", process.env.INFLUXDB_URL);
console.debug("Loaded INFLUXDB_TOKEN:", process.env.INFLUXDB_TOKEN);
console.debug("Loaded INFLUXDB_ORG:", process.env.INFLUXDB_ORG);
console.debug("Loaded INFLUXDB_BUCKET:", process.env.INFLUXDB_BUCKET);

/**
 * Parse version from __init__.py if it exists
 */
let minerVersion = '0.2.29t'; // Default version
try {
  const initPath = path.join(__dirname, '../../../src/tplr/__init__.py');
  if (fs.existsSync(initPath)) {
    const initContent = fs.readFileSync(initPath, 'utf8');
    const versionMatch = initContent.match(/__version__\s*=\s*["']([^"']+)["']/);
    if (versionMatch && versionMatch[1]) {
      minerVersion = versionMatch[1];
    }
  }
} catch (error) {
  console.warn('Could not parse version from __init__.py:', error.message);
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

// Default configuration for InfluxDB
const influxConfig = {
  url: normalizeUrl(process.env.INFLUXDB_URL || 'https://uaepr2itgl-tzxeth774u3fvf.timestream-influxdb.us-east-2.on.aws'),
  token: process.env.INFLUXDB_TOKEN,
  org: process.env.INFLUXDB_ORG || 'tplr',
  bucket: process.env.INFLUXDB_BUCKET || 'tplr',
  version: process.env.MINER_VERSION || minerVersion
};

export default influxConfig;