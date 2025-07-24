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

import os
from unittest.mock import patch

from src.tplr.config import format_bucket_secrets


def test_format_bucket_secrets_with_valid_inputs():
    """
    Tests that format_bucket_secrets returns a correctly structured dictionary
    when all environment variables are present.
    """
    with patch.dict(
        os.environ,
        {
            "TEST_SECRET_ACCOUNT_ID": "test_account_id",
            "TEST_SECRET_BUCKET_NAME": "test_bucket_name",
            "TEST_SECRET_READ_ACCESS_KEY_ID": "test_read_access_key",
            "TEST_SECRET_READ_SECRET_ACCESS_KEY": "test_read_secret_key",
            "TEST_SECRET_WRITE_ACCESS_KEY_ID": "test_write_access_key",
            "TEST_SECRET_WRITE_SECRET_ACCESS_KEY": "test_write_secret_key",
        },
    ):
        expected_output = {
            "account_id": "test_account_id",
            "name": "test_bucket_name",
            "credentials": {
                "read": {
                    "access_key_id": "test_read_access_key",
                    "secret_access_key": "test_read_secret_key",
                },
                "write": {
                    "access_key_id": "test_write_access_key",
                    "secret_access_key": "test_write_secret_key",
                },
            },
        }
        assert format_bucket_secrets("TEST_SECRET") == expected_output


def test_format_bucket_secrets_with_missing_inputs():
    """
    Tests that format_bucket_secrets handles missing environment variables
    gracefully by returning None for the corresponding values.
    """
    with patch.dict(os.environ, {}, clear=True):
        expected_output = {
            "account_id": None,
            "name": None,
            "credentials": {
                "read": {
                    "access_key_id": None,
                    "secret_access_key": None,
                },
                "write": {
                    "access_key_id": None,
                    "secret_access_key": None,
                },
            },
        }
        assert format_bucket_secrets("TEST_SECRET") == expected_output
