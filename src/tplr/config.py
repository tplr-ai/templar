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
import botocore.config

# Local imports
from .logging import logger

def load_bucket_secrets():
    secrets = {
        "gradients": {
            "account_id": os.environ.get("R2_GRADIENTS_ACCOUNT_ID"),
            "name": os.environ.get("R2_GRADIENTS_BUCKET_NAME"),
            "credentials": {
                "read": {
                    "access_key_id": os.environ.get("R2_GRADIENTS_READ_ACCESS_KEY_ID"),
                    "secret_access_key": os.environ.get("R2_GRADIENTS_READ_SECRET_ACCESS_KEY")
                },
                "write": {
                    "access_key_id": os.environ.get("R2_GRADIENTS_WRITE_ACCESS_KEY_ID"),
                    "secret_access_key": os.environ.get("R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY")
                }
            }
        },
        "dataset": {
            "account_id": os.environ.get("R2_DATASET_ACCOUNT_ID"), 
            "name": os.environ.get("R2_DATASET_BUCKET_NAME"),
            "credentials": {
                "read": {
                    "access_key_id": os.environ.get("R2_DATASET_READ_ACCESS_KEY_ID"),
                    "secret_access_key": os.environ.get("R2_DATASET_READ_SECRET_ACCESS_KEY")
                },
                "write": {
                    "access_key_id": os.environ.get("R2_DATASET_WRITE_ACCESS_KEY_ID"),
                    "secret_access_key": os.environ.get("R2_DATASET_WRITE_SECRET_ACCESS_KEY")
                }
            }
        }
    }

    required_vars = [
        "R2_GRADIENTS_ACCOUNT_ID",
        "R2_GRADIENTS_BUCKET_NAME",
        "R2_GRADIENTS_READ_ACCESS_KEY_ID",
        "R2_GRADIENTS_READ_SECRET_ACCESS_KEY", 
        "R2_GRADIENTS_WRITE_ACCESS_KEY_ID",
        "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY",
        "R2_DATASET_ACCOUNT_ID",
        "R2_DATASET_BUCKET_NAME",
        "R2_DATASET_READ_ACCESS_KEY_ID",
        "R2_DATASET_READ_SECRET_ACCESS_KEY"
    ]

    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise ImportError(f"Required environment variables missing: {missing_vars}")
    
    return secrets

# Initialize config after env vars are loaded
client_config = botocore.config.Config(max_pool_connections=256)
BUCKET_SECRETS = load_bucket_secrets()
