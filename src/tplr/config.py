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
from pathlib import Path

# Local imports
import botocore.config
from dotenv import load_dotenv
from .logging import logger

# Load environment variables
env_path = Path(__file__).parents[2] / ".env"
if not env_path.exists():
    logger.error(
        f"{env_path} not found. Please create it with the required R2 configuration."
    )
    sys.exit(1)

load_dotenv(env_path)

# Configure bucket secrets
BUCKET_SECRETS = {
    "account_id": os.getenv("R2_ACCOUNT_ID"),
    "bucket_name": os.getenv("R2_ACCOUNT_ID"),  # Using account_id as bucket name
    "read": {
        "access_key_id": os.getenv("R2_READ_ACCESS_KEY_ID"),
        "secret_access_key": os.getenv("R2_READ_SECRET_ACCESS_KEY")
    },
    "write": {
        "access_key_id": os.getenv("R2_WRITE_ACCESS_KEY_ID"),
        "secret_access_key": os.getenv("R2_WRITE_SECRET_ACCESS_KEY")
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

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# Configure the S3 client
client_config = botocore.config.Config(
    max_pool_connections=256,
)
