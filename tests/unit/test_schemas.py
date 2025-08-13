import unittest

from pydantic import ValidationError

from tplr.schemas import Bucket, CommsGetResult


class TestBucket(unittest.TestCase):
    def test_bucket_creation(self):
        b = Bucket(
            name="test",
            account_id="test",
            access_key_id="test",
            secret_access_key="test",
        )
        self.assertEqual(b.name, "test")

    def test_bucket_missing_fields(self):
        with self.assertRaises(ValidationError):
            Bucket(name="test", account_id="test", access_key_id="test")

    def test_bucket_empty_fields(self):
        with self.assertRaises(ValidationError):
            Bucket(
                name="",
                account_id="test",
                access_key_id="test",
                secret_access_key="test",
            )

    def test_bucket_equality(self):
        b1 = Bucket(
            name="test",
            account_id="test",
            access_key_id="test",
            secret_access_key="test",
        )
        b2 = Bucket(
            name="test",
            account_id="test",
            access_key_id="test",
            secret_access_key="test",
        )
        self.assertEqual(b1, b2)

    def test_bucket_inequality(self):
        b1 = Bucket(
            name="test1",
            account_id="test",
            access_key_id="test",
            secret_access_key="test",
        )
        b2 = Bucket(
            name="test2",
            account_id="test",
            access_key_id="test",
            secret_access_key="test",
        )
        self.assertNotEqual(b1, b2)

    def test_bucket_hash(self):
        b1 = Bucket(
            name="test",
            account_id="test",
            access_key_id="test",
            secret_access_key="test",
        )
        b2 = Bucket(
            name="test",
            account_id="test",
            access_key_id="test",
            secret_access_key="test",
        )
        self.assertEqual(hash(b1), hash(b2))


class TestCommsGetResult(unittest.TestCase):
    def test_comms_get_result_ok(self):
        res = CommsGetResult(data={"key": "value"}, global_step=1, status="OK")
        self.assertTrue(res.success)
        self.assertEqual(res.status, "OK")
        self.assertEqual(res.data, {"key": "value"})

    def test_comms_get_result_non_ok_statuses(self):
        for status in ("TOO_EARLY", "TOO_LATE", "NOT_FOUND", "ERROR"):
            with self.subTest(status=status):
                res = CommsGetResult(status=status)
                self.assertFalse(res.success)

    def test_comms_get_result_ok_no_data(self):
        res = CommsGetResult(status="OK")
        self.assertFalse(res.success)

    def test_comms_get_result_default_status_is_ok(self):
        res = CommsGetResult(data={"k": "v"})
        self.assertEqual(res.status, "OK")
        self.assertTrue(res.success)

    def test_comms_get_result_invalid_status_raises(self):
        with self.assertRaises(ValidationError):
            CommsGetResult(status="INVALID")

    def test_comms_get_result_ok_with_empty_dict_data(self):
        res = CommsGetResult(status="OK", data={})
        # Current semantics: success if status == "OK" and data is not None
        self.assertTrue(res.success)
