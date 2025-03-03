# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write.point import Point
from influxdb_client.domain.write_precision import WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import time
from threading import Lock
import statistics
from . import __version__


class MetricsLogger:
    def __init__(
        self,
        host="localhost",
        port=8086,
        database="tplr_metrics",
        token=None,
        org="templar",
    ):
        if token is None:
            raise ValueError("InfluxDB token must be provided")

        url = f"https://{host}:{port}"
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.database = database  # In InfluxDB 2.x, this is called "bucket"
        self.org = org
        self.lock = Lock()

    def process_value(self, v):
        if isinstance(v, (int, float)):
            return float(v)
        elif isinstance(v, list):
            # If the list appears to be a list of peer UIDs, log the raw list as a string.
            if v and all(isinstance(item, int) for item in v):
                return str(v)  # Alternatively, return len(v) if count is preferred
            # For lists of numbers that are not peer IDs, compute stats.
            if not v:  # Empty list
                return 0.0
            return {
                "mean": float(statistics.mean(v)),
                "min": float(min(v)),
                "max": float(max(v)),
                "median": float(statistics.median(v)),
            }
        return v

    def log(self, measurement: str, tags: dict, fields: dict, timestamp=None):
        try:
            if timestamp is None:
                timestamp = int(time.time_ns())

            if "version" not in tags:
                tags["version"] = __version__

            # Process fields and handle lists
            processed_fields = {}
            for k, v in fields.items():
                processed_value = self.process_value(v)
                if isinstance(processed_value, dict):
                    # For list values, create multiple fields with prefixes
                    for stat_name, stat_value in processed_value.items():
                        processed_fields[f"{k}_{stat_name}"] = stat_value
                else:
                    processed_fields[k] = processed_value

            point = Point(measurement)

            # Add tags
            for tag_key, tag_value in tags.items():
                point = point.tag(tag_key, str(tag_value))

            # Add fields
            for field_key, field_value in processed_fields.items():
                point = point.field(field_key, field_value)

            # Add timestamp
            point = point.time(timestamp, WritePrecision.NS)

            with self.lock:
                self.write_api.write(bucket=self.database, org=self.org, record=point)
        except Exception as e:
            print(f"Error logging metrics: {str(e)}")
