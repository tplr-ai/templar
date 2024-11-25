# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Global imports
import os
import sys
import yaml
from pathlib import Path

# Local imports
import botocore.config
from dotenv import dotenv_values
from loguru import logger

# Load environment variables
env_config = {**dotenv_values(".env"), **os.environ}
AWS_ACCESS_KEY_ID = env_config.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_config.get("AWS_SECRET_ACCESS_KEY")

envfile_path = Path(__file__).parents[2] / ".env.yaml"
try:
    with open(envfile_path, "r") as file:
        BUCKET_SECRETS = yaml.safe_load(file)
except FileNotFoundError:
    logger.error(
        f"{envfile_path} not found. Please create it with the help of `.env-template.yaml`."
    )
    sys.exit()
BUCKET_SECRETS["bucket_name"] = BUCKET_SECRETS["account_id"]

# Configure the S3 client
client_config = botocore.config.Config(
    max_pool_connections=256,
)
