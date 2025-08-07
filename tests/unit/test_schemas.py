import unittest

from tplr.schemas import Bucket


class TestBucket(unittest.TestCase):
    def test_hash(self):
        bucket1 = Bucket(
            name="test",
            account_id="123",
            access_key_id="abc",
            secret_access_key="def",
        )
        bucket2 = Bucket(
            name="test",
            account_id="123",
            access_key_id="abc",
            secret_access_key="def",
        )
        self.assertEqual(hash(bucket1), hash(bucket2))

    def test_eq(self):
        bucket1 = Bucket(
            name="test",
            account_id="123",
            access_key_id="abc",
            secret_access_key="def",
        )
        bucket2 = Bucket(
            name="test",
            account_id="123",
            access_key_id="abc",
            secret_access_key="def",
        )
        self.assertEqual(bucket1, bucket2)


if __name__ == "__main__":
    unittest.main()
