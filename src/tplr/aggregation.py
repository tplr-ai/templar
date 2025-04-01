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
import json
from .schemas import Bucket
from . import config
from tplr import logger
from . import __version__

class AggregationManager:
    def __init__(self, comms):
        """
        Initialize the aggregation manager.
        Args:
            comms: Comms instance to access storage and configuration.
        """
        self.comms = comms
        self.logger = comms.logger

    async def load_aggregation(self, window: int):
        """
        Load aggregated gradient data for a given window.
        Args:
            window: Window number for which to load aggregation.
        Returns:
            Aggregated data if found; otherwise, None.
        """
        bucket = self._get_aggregator_bucket()
        filename = f"aggregator-{window}-v{__version__}.pt"
        self.logger.info(f"Attempting to download aggregation file: {filename}")
        result = await self.comms.storage.s3_get_object(key=filename, bucket=bucket, timeout=20)
        if result is None:
            self.logger.warning(f"No aggregation file found for window {window}")
        else:
            self.logger.info(f"Successfully loaded aggregation data for window {window}")
        return result

    def _get_aggregator_bucket(self) -> Bucket:
        bucket_config = config.BUCKET_SECRETS.get("aggregator")
        if not bucket_config:
            raise ValueError("Aggregator bucket configuration not found.")
        creds = bucket_config.get("credentials", {}).get("read", {})
        return Bucket(
            name=bucket_config.get("name", "").strip(),
            account_id=bucket_config.get("account_id", "").strip(),
            access_key_id=creds.get("access_key_id", "").strip(),
            secret_access_key=creds.get("secret_access_key", "").strip(),
        )
