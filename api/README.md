
# TPLR Metrics API

API service for querying TPLR miner metrics from InfluxDB.

## Installation

1. Clone the repository and navigate to the api directory:
```bash
cd api
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file in the api directory:
```bash
INFLUXDB_TOKEN=your_influxdb_token_here
PORT=3050  # Optional, defaults to 3000
```

## Configuration

The InfluxDB configuration is in `src/config/influxdb.js`. Default settings:
- URL: AWS InfluxDB instance
- Organization: templar
- Bucket: tplr
- Default Version: 0.2.29t

## Running the API

Development mode:
```bash
npm run dev
```

Production mode:
```bash
npm start
```

## API Routes

### Get Losses
Retrieves loss metrics for miners.

```bash
GET /api/metrics/losses
```

Query Parameters:
- `window` (optional): Filter by specific window
- `version` (optional): Filter by version (defaults to 0.2.29t)
- `timeRange` (optional): Time range to query (defaults to 24h)

Example:
```bash
curl "http://localhost:3050/api/metrics/losses" | json_pp
curl "http://localhost:3050/api/metrics/losses?window=123&timeRange=12h" | json_pp
```

### Get Tokens Per Second
Retrieves tokens per second metrics for miners.

```bash
GET /api/metrics/tokens-per-sec
```

Query Parameters:
- `uid` (optional): Filter by specific miner UID
- `version` (optional): Filter by version (defaults to 0.2.29t)
- `timeRange` (optional): Time range to query (defaults to 24h)
- `window` (optional): Filter by specific window
- `aggregate` (optional): If 'true', returns mean across all UIDs

Examples:
```bash
# Get all UIDs tokens/sec
curl "http://localhost:3050/api/metrics/tokens-per-sec" | json_pp

# Get mean tokens/sec across all UIDs
curl "http://localhost:3050/api/metrics/tokens-per-sec?aggregate=true" | json_pp

# Get specific UID tokens/sec
curl "http://localhost:3050/api/metrics/tokens-per-sec?uid=76" | json_pp

# Get tokens/sec for last 12 hours
curl "http://localhost:3050/api/metrics/tokens-per-sec?timeRange=12h" | json_pp
```

## Response Formats

### Losses Response
```json
{
  "losses": [
    {
      "time": "2024-03-21T10:00:00Z",
      "window": 205245,
      "losses": {
        "76": 0.834,
        "77": 0.756
      }
    }
  ],
  "version": "0.2.29t",
  "params": {
    "window": null,
    "timeRange": "24h"
  }
}
```

### Tokens Per Second Response
```json
{
  "success": true,
  "data": [
    {
      "time": "2024-03-21T10:00:00Z",
      "tokens_per_sec": 1234.56,
      "uid": "76"  // Only present when not aggregated
    }
  ],
  "timestamp": "2024-03-21T10:01:00Z"
}
```

## Error Handling

All endpoints return appropriate HTTP status codes and error messages:

```json
{
  "success": false,
  "error": "Error message here",
  "timestamp": "2024-03-21T10:01:00Z"
}
```
