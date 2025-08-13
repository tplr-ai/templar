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
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT of OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import unittest
from pydantic import ValidationError
from tplr.schemas import Bucket, CommsGetResult

class TestBucket(unittest.TestCase):

    def test_bucket_creation(self):
        b = Bucket(name="test", account_id="test", access_key_id="test", secret_access_key="test")
        self.assertEqual(b.name, "test")

    def test_bucket_missing_fields(self):
        with self.assertRaises(ValidationError):
            Bucket(name="test", account_id="test", access_key_id="test")

    def test_bucket_empty_fields(self):
        with self.assertRaises(ValidationError):
            Bucket(name="", account_id="test", access_key_id="test", secret_access_key="test")

    def test_bucket_equality(self):
        b1 = Bucket(name="test", account_id="test", access_key_id="test", secret_access_key="test")
        b2 = Bucket(name="test", account_id="test", access_key_id="test", secret_access_key="test")
        self.assertEqual(b1, b2)

    def test_bucket_inequality(self):
        b1 = Bucket(name="test1", account_id="test", access_key_id="test", secret_access_key="test")
        b2 = Bucket(name="test2", account_id="test", access_key_id="test", secret_access_key="test")
        self.assertNotEqual(b1, b2)

    def test_bucket_hash(self):
        b1 = Bucket(name="test", account_id="test", access_key_id="test", secret_access_key="test")
        b2 = Bucket(name="test", account_id="test", access_key_id="test", secret_access_key="test")
        self.assertEqual(hash(b1), hash(b2))

class TestCommsGetResult(unittest.TestCase):

    def test_comms_get_result_ok(self):
        res = CommsGetResult(data={"key": "value"}, global_step=1, status="OK")
        self.assertTrue(res.success)
        self.assertEqual(res.status, "OK")
        self.assertEqual(res.data, {"key": "value"})

    def test_comms_get_result_not_ok(self):
        res = CommsGetResult(status="NOT_FOUND")
        self.assertFalse(res.success)

    def test_comms_get_result_ok_no_data(self):
        res = CommsGetResult(status="OK")
        self.assertFalse(res.success)

if __name__ == '__main__':
    unittest.main()