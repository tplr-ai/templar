# Accessing Templar Protocol Log Archives

This document explains how to access the raw logs collected from the Templar Protocol network. While real-time logs and filtering are available through our Grafana dashboard, we also provide direct access to the raw log archives for greater transparency and detailed analysis.

## Overview

As part of our commitment to transparency, all logs from the Templar Protocol are:

1. Collected via Loki in real-time
2. Archived to Cloudflare R2 storage in JSONL format
3. Made available through public read-only credentials

This system allows anyone to verify network operations, troubleshoot issues, or perform historical analysis of the protocol.

## Log Archive Structure

The logs are stored in a time-partitioned directory structure:

```
logs/version={version}/year={year}/month={month}/day={day}/hour={hour}/{service}_{uid}_{random_hex}.jsonl
```

For example:
```
logs/version=0.2.69/year=2025/month=04/day=11/hour=21/validator_1_1e73e016.jsonl
```

This structure enables efficient querying by:

- Protocol version
- Time period (year, month, day, hour)
- Service type (validator, miner, aggregator)
- Node UID

## Access Credentials

The log archives are available in a public R2 bucket with read-only access:

- **Bucket URL**: https://8af7f92a8a0661cf7f1ac0420c932980.r2.cloudflarestorage.com/loki-logs
- **Access Key ID**: `0a87c1d3f104b47aa8ad711e94ef401c`
- **Secret Access Key**: `54d37a9105144363b5b5108b24af5ffdf574c1f540b2d36dbdd5c5cac738743e`

These credentials provide read-only access to the archive.

## Accessing the Logs

You can access the logs using any S3-compatible tool. Here are some examples:

### Using rclone

1. Configure rclone with the provided credentials:

```bash
rclone config create templar-logs s3 \
  provider=Cloudflare \
  endpoint=https://8af7f92a8a0661cf7f1ac0420c932980.r2.cloudflarestorage.com \
  access_key_id=0a87c1d3f104b47aa8ad711e94ef401c \
  secret_access_key=54d37a9105144363b5b5108b24af5ffdf574c1f540b2d36dbdd5c5cac738743e \
  region=auto
```

2. List available logs:

```bash
# List all logs
rclone ls templar-logs:loki-logs/logs

# List logs for a specific version
rclone ls templar-logs:loki-logs/logs/version=0.2.71/

# List logs for a specific day
rclone ls templar-logs:loki-logs/logs/version=0.2.71/year=2025/month=04/day=13/
```

3. Download logs for analysis:

```bash
# Download all logs from a specific hour for a validator with UID=1
rclone copy templar-logs:loki-logs/logs/version=0.2.71/year=2025/month=04/day=13/hour=11/validator_1_ ./local-logs/

# Download all logs from a specific day
rclone copy templar-logs:loki-logs/logs/version=0.2.71/year=2025/month=04/day=13/ ./local-logs/day-13/
```

## Log Format

The logs are stored in JSONL (JSON Lines) format, where each line is a valid JSON object representing a single log entry. This format makes it easy to parse and analyze the logs programmatically.

Example of parsing log files:

```python
import json

# Open a log file
with open('validator_1_1e73e016.jsonl', 'r') as f:
    # Process each line as a JSON object
    for line in f:
        log_entry = json.loads(line)
        # Process the log entry
        print(f"Timestamp: {log_entry.get('ts')}, Level: {log_entry.get('level')}, Message: {log_entry.get('message')}")
```

## Retention Policy

Please note that log archives are subject to a retention policy. Older logs may be purged periodically to manage storage costs. If you need specific historical logs for research or analysis, we recommend downloading and storing them locally.

## Additional Resources

- For real-time log viewing and filtering, please visit our [Grafana Dashboard](https://grafana.tplr.ai)
- For questions or issues with log access, please contact the Templar team or open an issue on our GitHub repository
