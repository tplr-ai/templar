import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tplr import decos, exceptions


class TestDecos(unittest.TestCase):
    def test_retry_on_failure(self):
        """Tests the retry_on_failure decorator."""

        @decos.retry_on_failure(retries=3, delay=0.01)
        async def my_func():
            my_func.call_count += 1
            if my_func.call_count < 3:
                return None
            return MagicMock(success=True)

        my_func.call_count = 0

        result = asyncio.run(my_func())
        self.assertEqual(my_func.call_count, 3)
        self.assertTrue(result.success)

    def test_retry_on_failure_timeout(self):
        """Tests the timeout functionality of the retry_on_failure decorator."""

        @decos.retry_on_failure(retries=5, timeout=0.05, delay=0.02)
        async def my_func():
            my_func.call_count += 1
            await asyncio.sleep(0.03)
            return None

        my_func.call_count = 0

        result = asyncio.run(my_func())
        self.assertIsNone(result)
        self.assertLess(my_func.call_count, 5)

    # @patch("tplr.decos.exceptions.handle_s3_exceptions")
    # def test_async_s3_exception_catcher(self, mock_handler):
    #     """Tests the async_s3_exception_catcher decorator."""

    #     @decos.async_s3_exception_catcher
    #     async def my_func():
    #         raise ValueError("test error")

    #     asyncio.run(my_func())
    #     mock_handler.assert_called_once()

    # @patch("tplr.decos.exceptions.handle_s3_exceptions")
    # def test_s3_exception_catcher(self, mock_handler):
    #     """Tests the s3_exception_catcher decorator."""

    #     @decos.s3_exception_catcher
    #     def my_func():
    #         raise ValueError("test error")

    #     my_func()
    #     mock_handler.assert_called_once()

    # def test_general_exception_catcher(self):
    #     """Tests the general_exception_catcher decorator."""

    #     mock_handler = MagicMock()

    #     @decos.general_exception_catcher
    #     async def my_func():
    #         raise ValueError("test error")

    #     with patch("tplr.decos.exceptions.handle_general_exceptions", new=mock_handler):
    #         asyncio.run(my_func())
    #         mock_handler.assert_called_once()
