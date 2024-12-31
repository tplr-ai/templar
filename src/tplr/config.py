# The MIT License (MIT)
# © 2024 templar.tech

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
# fmt: off

# Global imports
import os
import sys
import botocore.config

# Local imports
from .logging import logger

# Configure bucket secrets from environment variables
BUCKET_SECRETS = {
    "account_id": os.environ.get("R2_ACCOUNT_ID"),
    "bucket_name": os.environ.get("R2_ACCOUNT_ID"),  # Using account_id as bucket name
    "read": {
        "access_key_id": os.environ.get("R2_READ_ACCESS_KEY_ID"),
        "secret_access_key": os.environ.get("R2_READ_SECRET_ACCESS_KEY")
    },
    "write": {
        "access_key_id": os.environ.get("R2_WRITE_ACCESS_KEY_ID"),
        "secret_access_key": os.environ.get("R2_WRITE_SECRET_ACCESS_KEY")
    }
}

# Validate required environment variables
required_vars = [
    "R2_ACCOUNT_ID",
    "R2_READ_ACCESS_KEY_ID",
    "R2_READ_SECRET_ACCESS_KEY",
    "R2_WRITE_ACCESS_KEY_ID",
    "R2_WRITE_SECRET_ACCESS_KEY"
]

missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# Configure the S3 client
client_config = botocore.config.Config(
    max_pool_connections=256,
)
