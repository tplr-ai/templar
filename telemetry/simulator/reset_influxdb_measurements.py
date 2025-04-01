#!/usr/bin/env python3
# Script to reset problematic InfluxDB measurements

import os
import argparse
from influxdb_client import InfluxDBClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default InfluxDB settings
DEFAULT_HOST = "pliftu8n85-tzxeth774u3fvf.timestream-influxdb.us-east-2.on.aws"
DEFAULT_PORT = 8086
DEFAULT_DATABASE = "tplr"
DEFAULT_ORG = "tplr"
FALLBACK_INFLUXDB_TOKEN = "lTRclLtRXOJWGOB-vr1mhtp5SholImgBH705pMgK1_0sCzTzAXivhd4gPwJhRoK6HLRvG8cxjhOTEy1hlm4D3Q=="


def reset_measurements(measurement, field=None):
    """Reset problematic InfluxDB measurements to resolve type conflicts"""
    # Get InfluxDB connection parameters
    host = os.environ.get("INFLUXDB_HOST", DEFAULT_HOST)
    port = int(os.environ.get("INFLUXDB_PORT", DEFAULT_PORT))
    database = os.environ.get("INFLUXDB_DATABASE", DEFAULT_DATABASE)
    org = os.environ.get("INFLUXDB_ORG", DEFAULT_ORG)
    token = os.environ.get("INFLUXDB_TOKEN", FALLBACK_INFLUXDB_TOKEN)

    url = f"https://{host}:{port}"
    client = InfluxDBClient(url=url, token=token, org=org)

    # Build delete query
    delete_query = f'from(bucket: "{database}") |> range(start: 0)'

    if measurement:
        delete_query += f' |> filter(fn: (r) => r._measurement == "{measurement}")'

    if field:
        delete_query += f' |> filter(fn: (r) => r._field == "{field}")'

    # Delete data
    logger.info(f"Executing delete query: {delete_query}")
    query_api = client.query_api()
    delete_api = client.delete_api()

    # First get some data to confirm we have the right measurement/field
    try:
        result = query_api.query(delete_query)
        if not result:
            logger.warning(
                f"No data found for measurement={measurement}, field={field}"
            )
            return

        # Now delete the data
        logger.info("Found data to delete. Proceeding with deletion...")

        # Delete for all time
        start = "1970-01-01T00:00:00Z"
        stop = "2099-12-31T23:59:59Z"

        delete_api.delete(
            start=start,
            stop=stop,
            predicate=f'_measurement="{measurement}"'
            + (f' AND _field="{field}"' if field else ""),
            bucket=database,
        )

        logger.info(
            f"Successfully deleted data for measurement={measurement}, field={field}"
        )
    except Exception as e:
        logger.error(f"Error deleting data: {e}")
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Reset InfluxDB measurements to resolve type conflicts"
    )
    parser.add_argument(
        "--measurement", type=str, required=True, help="Measurement to reset"
    )
    parser.add_argument("--field", type=str, help="Specific field to reset (optional)")
    args = parser.parse_args()

    reset_measurements(args.measurement, args.field)


if __name__ == "__main__":
    main()
