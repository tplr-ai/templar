import asyncio
import unittest
from unittest.mock import MagicMock, patch

from botocore import exceptions as boto_excepts

from tplr import exceptions


class TestExceptions(unittest.TestCase):
    def test_handle_s3_exceptions_client_error(self):
        """Tests that ClientError is handled correctly."""
        mock_logger = MagicMock()
        with patch("tplr.exceptions.tplr.logger", mock_logger):
            error = boto_excepts.ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject"
            )
            purge = exceptions.handle_s3_exceptions(error, "test_func")
            self.assertTrue(purge)
            mock_logger.exception.assert_called_once()

    def test_handle_s3_exceptions_connection_closed(self):
        """Tests that ConnectionClosedError is handled correctly."""
        mock_logger = MagicMock()
        with patch("tplr.exceptions.tplr.logger", mock_logger):
            error = boto_excepts.ConnectionClosedError(endpoint_url="some_url")
            purge = exceptions.handle_s3_exceptions(error, "test_func")
            self.assertTrue(purge)
            mock_logger.exception.assert_called_once()

    def test_handle_s3_exceptions_no_credentials(self):
        """Tests that NoCredentialsError is handled correctly."""
        mock_logger = MagicMock()
        with patch("tplr.exceptions.tplr.logger", mock_logger):
            error = boto_excepts.NoCredentialsError()
            purge = exceptions.handle_s3_exceptions(error, "test_func")
            self.assertFalse(purge)
            mock_logger.exception.assert_called_once()

    def test_handle_s3_exceptions_param_validation(self):
        """Tests that ParamValidationError is handled correctly."""
        mock_logger = MagicMock()
        with patch("tplr.exceptions.tplr.logger", mock_logger):
            error = boto_excepts.ParamValidationError(report="some_report")
            purge = exceptions.handle_s3_exceptions(error, "test_func")
            self.assertFalse(purge)
            mock_logger.exception.assert_called_once()

    def test_handle_s3_exceptions_timeout(self):
        """Tests that asyncio.TimeoutError is handled correctly."""
        mock_logger = MagicMock()
        with patch("tplr.exceptions.tplr.logger", mock_logger):
            error = asyncio.TimeoutError()
            purge = exceptions.handle_s3_exceptions(error, "test_func")
            self.assertFalse(purge)
            mock_logger.exception.assert_called_once()

    def test_handle_s3_exceptions_unexpected(self):
        """Tests that other exceptions are handled correctly."""
        mock_logger = MagicMock()
        with patch("tplr.exceptions.tplr.logger", mock_logger):
            error = Exception("Some other error")
            purge = exceptions.handle_s3_exceptions(error, "test_func")
            self.assertFalse(purge)
            mock_logger.exception.assert_called_once()

    def test_handle_general_exceptions(self):
        """Tests that general exceptions are handled correctly."""
        mock_logger = MagicMock()
        with patch("tplr.exceptions.tplr.logger", mock_logger):
            exceptions.handle_general_exceptions(IndexError("test"), "test_func")
            mock_logger.exception.assert_called_with(
                "Function 'test_func' encountered an error: IndexError - likely access to a list/tuple with an out-of-bounds index: test"
            )

            exceptions.handle_general_exceptions(ValueError("test"), "test_func")
            mock_logger.exception.assert_called_with(
                "Function 'test_func' encountered an error: ValueError - an argument is of the correct type but has an inappropriate value: test"
            )

            exceptions.handle_general_exceptions(KeyError("test"), "test_func")
            mock_logger.exception.assert_called_with(
                "Function 'test_func' encountered an error: KeyError - likely access to a dictionary with a non-existent key: 'test'"
            )

            exceptions.handle_general_exceptions(TypeError("test"), "test_func")
            mock_logger.exception.assert_called_with(
                "Function 'test_func' encountered an error: TypeError - an operation or function is applied to an object of inappropriate type: test"
            )

            exceptions.handle_general_exceptions(AttributeError("test"), "test_func")
            mock_logger.exception.assert_called_with(
                "Function 'test_func' encountered an error: AttributeError - an attribute reference or assignment fails: test"
            )

            exceptions.handle_general_exceptions(FileNotFoundError("test"), "test_func")
            mock_logger.exception.assert_called_with(
                "Function 'test_func' encountered an error: FileNotFoundError - a file or directory is requested but doesn't exist: test"
            )

            exceptions.handle_general_exceptions(Exception("test"), "test_func")
            mock_logger.exception.assert_called_with(
                "Function 'test_func' encountered an error: An unexpected error occurred: test"
            )
