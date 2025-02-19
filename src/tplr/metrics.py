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

from influxdb import InfluxDBClient
import time
from threading import Lock

class MetricsLogger:
    def __init__(self, host="localhost", port=8086, database="tplr_metrics"):
        self.client = InfluxDBClient(host=host, port=port)
        self.database = database
        self.client.create_database(self.database)
        self.client.switch_database(self.database)
        self.lock = Lock()  # ensure thread-safety

    def log(self, measurement: str, tags: dict, fields: dict, timestamp=None):
        if timestamp is None:
            # Use nanosecond precision timestamp
            timestamp = int(time.time() * 1e9)
        data_point = [
            {
                "measurement": measurement,
                "tags": tags,
                "time": timestamp,
                "fields": fields,
            }
        ]
        with self.lock:
            self.client.write_points(data_point)