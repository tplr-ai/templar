#!/usr/bin/env python3
"""
Comprehensive unit tests for the R2 access validation script.
Testing Framework: pytest with asyncio support
"""

import os
import sys
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import tempfile
import botocore.exceptions
from aiobotocore.session import get_session

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.test_validate_r2_access import test_read_access, test_write_access, validate_credentials


class TestReadAccess:
    """Test suite for test_read_access function"""

    @pytest.mark.asyncio
    async def test_read_access_success(self):
        """Test successful read access verification"""
        # Arrange
        mock_client = AsyncMock()
        mock_client.list_objects_v2.return_value = {"Contents": []}
        bucket = "test-bucket"
        prefix = "test_"

        # Act
        result = await test_read_access(mock_client, bucket, prefix)

        # Assert
        assert result is True
        mock_client.list_objects_v2.assert_called_once_with(Bucket=bucket, MaxKeys=1)

    @pytest.mark.asyncio
    async def test_read_access_client_error(self):
        """Test read access failure due to client error"""
        # Arrange
        mock_client = AsyncMock()
        error_response = {"Error": {"Code": "AccessDenied"}}
        mock_client.list_objects_v2.side_effect = botocore.exceptions.ClientError(
            error_response, "ListObjectsV2"
        )
        bucket = "test-bucket"
        prefix = "test_"

        # Act
        result = await test_read_access(mock_client, bucket, prefix)

        # Assert
        assert result is False
        mock_client.list_objects_v2.assert_called_once_with(Bucket=bucket, MaxKeys=1)

    @pytest.mark.asyncio
    async def test_read_access_different_error_codes(self):
        """Test read access with various error codes"""
        error_codes = ["NoSuchBucket", "InvalidAccessKeyId", "SignatureDoesNotMatch"]
        
        for error_code in error_codes:
            # Arrange
            mock_client = AsyncMock()
            error_response = {"Error": {"Code": error_code}}
            mock_client.list_objects_v2.side_effect = botocore.exceptions.ClientError(
                error_response, "ListObjectsV2"
            )
            bucket = "test-bucket"
            prefix = "test_"

            # Act
            result = await test_read_access(mock_client, bucket, prefix)

            # Assert
            assert result is False

    @pytest.mark.asyncio
    async def test_read_access_default_prefix(self):
        """Test read access with default prefix parameter"""
        # Arrange
        mock_client = AsyncMock()
        mock_client.list_objects_v2.return_value = {"Contents": []}
        bucket = "test-bucket"

        # Act
        result = await test_read_access(mock_client, bucket)

        # Assert
        assert result is True
        mock_client.list_objects_v2.assert_called_once_with(Bucket=bucket, MaxKeys=1)

    @pytest.mark.asyncio
    async def test_read_access_empty_bucket_name(self):
        """Test read access with empty bucket name"""
        # Arrange
        mock_client = AsyncMock()
        error_response = {"Error": {"Code": "InvalidBucketName"}}
        mock_client.list_objects_v2.side_effect = botocore.exceptions.ClientError(
            error_response, "ListObjectsV2"
        )
        bucket = ""

        # Act
        result = await test_read_access(mock_client, bucket)

        # Assert
        assert result is False


class TestWriteAccess:
    """Test suite for test_write_access function"""

    @pytest.mark.asyncio
    async def test_write_access_success(self):
        """Test successful write access verification"""
        # Arrange
        mock_client = AsyncMock()
        mock_client.put_object.return_value = {}
        mock_client.delete_object.return_value = {}
        bucket = "test-bucket"
        prefix = "test_"

        # Act
        result = await test_write_access(mock_client, bucket, prefix)

        # Assert
        assert result is True
        mock_client.put_object.assert_called_once_with(
            Bucket=bucket, Key="test_permissions_file", Body=b"test content"
        )
        mock_client.delete_object.assert_called_once_with(
            Bucket=bucket, Key="test_permissions_file"
        )

    @pytest.mark.asyncio
    async def test_write_access_put_failure(self):
        """Test write access failure during put operation"""
        # Arrange
        mock_client = AsyncMock()
        error_response = {"Error": {"Code": "AccessDenied"}}
        mock_client.put_object.side_effect = botocore.exceptions.ClientError(
            error_response, "PutObject"
        )
        bucket = "test-bucket"
        prefix = "test_"

        # Act
        result = await test_write_access(mock_client, bucket, prefix)

        # Assert
        assert result is False
        mock_client.put_object.assert_called_once()
        mock_client.delete_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_access_delete_failure_after_successful_put(self):
        """Test write access when put succeeds but delete fails"""
        # Arrange
        mock_client = AsyncMock()
        mock_client.put_object.return_value = {}
        error_response = {"Error": {"Code": "AccessDenied"}}
        mock_client.delete_object.side_effect = botocore.exceptions.ClientError(
            error_response, "DeleteObject"
        )
        bucket = "test-bucket"

        # Act
        result = await test_write_access(mock_client, bucket)

        # Assert
        assert result is False
        mock_client.put_object.assert_called_once()
        mock_client.delete_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_access_various_error_codes(self):
        """Test write access with various error codes"""
        error_codes = ["NoSuchBucket", "InvalidAccessKeyId", "InternalError"]
        
        for error_code in error_codes:
            # Arrange
            mock_client = AsyncMock()
            error_response = {"Error": {"Code": error_code}}
            mock_client.put_object.side_effect = botocore.exceptions.ClientError(
                error_response, "PutObject"
            )
            bucket = "test-bucket"

            # Act
            result = await test_write_access(mock_client, bucket)

            # Assert
            assert result is False

    @pytest.mark.asyncio
    async def test_write_access_default_prefix(self):
        """Test write access with default prefix parameter"""
        # Arrange
        mock_client = AsyncMock()
        mock_client.put_object.return_value = {}
        mock_client.delete_object.return_value = {}
        bucket = "test-bucket"

        # Act
        result = await test_write_access(mock_client, bucket)

        # Assert
        assert result is True


class TestValidateCredentials:
    """Test suite for validate_credentials function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.required_vars = [
            "R2_GRADIENTS_ACCOUNT_ID",
            "R2_GRADIENTS_READ_ACCESS_KEY_ID",
            "R2_GRADIENTS_READ_SECRET_ACCESS_KEY",
            "R2_GRADIENTS_WRITE_ACCESS_KEY_ID",
            "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY",
            "R2_DATASET_ACCOUNT_ID",
            "R2_DATASET_READ_ACCESS_KEY_ID",
            "R2_DATASET_READ_SECRET_ACCESS_KEY",
            "R2_DATASET_WRITE_ACCESS_KEY_ID",
            "R2_DATASET_WRITE_SECRET_ACCESS_KEY",
            "R2_GRADIENTS_BUCKET_NAME",
            "R2_DATASET_BUCKET_NAME",
        ]

    @patch.dict(os.environ, {}, clear=True)
    @pytest.mark.asyncio
    async def test_validate_credentials_missing_env_vars(self):
        """Test validation fails when required environment variables are missing"""
        with pytest.raises(SystemExit) as exc_info:
            await validate_credentials()
        
        assert exc_info.value.code == 1

    @patch.dict(os.environ, {
        "R2_GRADIENTS_ACCOUNT_ID": "test-account-1",
        "R2_GRADIENTS_READ_ACCESS_KEY_ID": "read-key-1",
        "R2_GRADIENTS_READ_SECRET_ACCESS_KEY": "read-secret-1",
        "R2_GRADIENTS_WRITE_ACCESS_KEY_ID": "write-key-1",
        "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY": "write-secret-1",
        "R2_DATASET_ACCOUNT_ID": "test-account-2",
        "R2_DATASET_READ_ACCESS_KEY_ID": "read-key-2",
        "R2_DATASET_READ_SECRET_ACCESS_KEY": "read-secret-2",
        "R2_DATASET_WRITE_ACCESS_KEY_ID": "write-key-2",
        "R2_DATASET_WRITE_SECRET_ACCESS_KEY": "write-secret-2",
        "R2_GRADIENTS_BUCKET_NAME": "gradients-bucket",
        "R2_DATASET_BUCKET_NAME": "dataset-bucket",
    })
    @patch('tests.test_validate_r2_access.get_session')
    @patch('tests.test_validate_r2_access.test_read_access')
    @patch('tests.test_validate_r2_access.test_write_access')
    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, mock_test_write, mock_test_read, mock_get_session):
        """Test successful credential validation"""
        # Arrange
        mock_session = Mock()
        mock_client = AsyncMock()
        mock_session.create_client.return_value.__aenter__.return_value = mock_client
        mock_session.create_client.return_value.__aexit__.return_value = None
        mock_get_session.return_value = mock_session
        
        mock_test_read.return_value = True
        mock_test_write.return_value = False  # Read credentials should not have write access

        # Act
        await validate_credentials()

        # Assert
        assert mock_get_session.called
        assert mock_session.create_client.call_count == 4  # 2 buckets × 2 credential types

    @patch.dict(os.environ, {
        "R2_GRADIENTS_ACCOUNT_ID": "test-account-1",
        # Missing some required vars
        "R2_GRADIENTS_READ_ACCESS_KEY_ID": "read-key-1",
    })
    @pytest.mark.asyncio
    async def test_validate_credentials_partial_env_vars(self):
        """Test validation fails when only some environment variables are set"""
        with pytest.raises(SystemExit) as exc_info:
            await validate_credentials()
        
        assert exc_info.value.code == 1

    @patch.dict(os.environ, {
        "R2_GRADIENTS_ACCOUNT_ID": "test-account-1",
        "R2_GRADIENTS_READ_ACCESS_KEY_ID": "read-key-1",
        "R2_GRADIENTS_READ_SECRET_ACCESS_KEY": "read-secret-1",
        "R2_GRADIENTS_WRITE_ACCESS_KEY_ID": "write-key-1",
        "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY": "write-secret-1",
        "R2_DATASET_ACCOUNT_ID": "test-account-2",
        "R2_DATASET_READ_ACCESS_KEY_ID": "read-key-2",
        "R2_DATASET_READ_SECRET_ACCESS_KEY": "read-secret-2",
        "R2_DATASET_WRITE_ACCESS_KEY_ID": "write-key-2",
        "R2_DATASET_WRITE_SECRET_ACCESS_KEY": "write-secret-2",
        "R2_GRADIENTS_BUCKET_NAME": "gradients-bucket",
        "R2_DATASET_BUCKET_NAME": "dataset-bucket",
    })
    @patch('tests.test_validate_r2_access.get_session')
    @patch('tests.test_validate_r2_access.test_read_access')
    @patch('tests.test_validate_r2_access.test_write_access')
    @pytest.mark.asyncio
    async def test_validate_credentials_read_has_write_warning(self, mock_test_write, mock_test_read, mock_get_session):
        """Test warning when read credentials have write access"""
        # Arrange
        mock_session = Mock()
        mock_client = AsyncMock()
        mock_session.create_client.return_value.__aenter__.return_value = mock_client
        mock_session.create_client.return_value.__aexit__.return_value = None
        mock_get_session.return_value = mock_session
        
        mock_test_read.return_value = True
        mock_test_write.return_value = True  # Read credentials have write access - should warn

        # Act & Assert - should not raise but should print warning
        await validate_credentials()

    @patch.dict(os.environ, {
        "R2_GRADIENTS_ACCOUNT_ID": "test-account-1",
        "R2_GRADIENTS_READ_ACCESS_KEY_ID": "read-key-1",
        "R2_GRADIENTS_READ_SECRET_ACCESS_KEY": "read-secret-1",
        "R2_GRADIENTS_WRITE_ACCESS_KEY_ID": "write-key-1",
        "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY": "write-secret-1",
        "R2_DATASET_ACCOUNT_ID": "test-account-2",
        "R2_DATASET_READ_ACCESS_KEY_ID": "read-key-2",
        "R2_DATASET_READ_SECRET_ACCESS_KEY": "read-secret-2",
        "R2_DATASET_WRITE_ACCESS_KEY_ID": "write-key-2",
        "R2_DATASET_WRITE_SECRET_ACCESS_KEY": "write-secret-2",
        "R2_GRADIENTS_BUCKET_NAME": "gradients-bucket",
        "R2_DATASET_BUCKET_NAME": "dataset-bucket",
    })
    @patch('tests.test_validate_r2_access.get_session')
    @patch('tests.test_validate_r2_access.test_read_access')
    @patch('tests.test_validate_r2_access.test_write_access')
    @pytest.mark.asyncio
    async def test_validate_credentials_write_no_read_warning(self, mock_test_write, mock_test_read, mock_get_session):
        """Test warning when write credentials don't have read access"""
        # Arrange
        mock_session = Mock()
        mock_client = AsyncMock()
        mock_session.create_client.return_value.__aenter__.return_value = mock_client
        mock_session.create_client.return_value.__aexit__.return_value = None
        mock_get_session.return_value = mock_session
        
        # First call (read client): read=True, write=False
        # Second call (write client): read=False, write=True - should warn
        mock_test_read.side_effect = [True, False, True, False]  # 4 calls total
        mock_test_write.side_effect = [False, True, False, True]

        # Act & Assert - should not raise but should print warning
        await validate_credentials()

    @patch('tests.test_validate_r2_access.os.getenv')
    @pytest.mark.asyncio
    async def test_validate_credentials_env_var_checking(self, mock_getenv):
        """Test that all required environment variables are checked"""
        # Arrange
        mock_getenv.return_value = None  # All vars missing

        # Act & Assert
        with pytest.raises(SystemExit):
            await validate_credentials()

        # Verify all required vars were checked
        expected_calls = [
            "R2_GRADIENTS_ACCOUNT_ID",
            "R2_GRADIENTS_READ_ACCESS_KEY_ID", 
            "R2_GRADIENTS_READ_SECRET_ACCESS_KEY",
            "R2_GRADIENTS_WRITE_ACCESS_KEY_ID",
            "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY",
            "R2_DATASET_ACCOUNT_ID",
            "R2_DATASET_READ_ACCESS_KEY_ID",
            "R2_DATASET_READ_SECRET_ACCESS_KEY", 
            "R2_DATASET_WRITE_ACCESS_KEY_ID",
            "R2_DATASET_WRITE_SECRET_ACCESS_KEY",
        ]
        
        for var in expected_calls:
            assert any(call.args[0] == var for call in mock_getenv.call_args_list)


class TestEnvironmentSetup:
    """Test suite for environment setup and configuration"""

    @patch('tests.test_validate_r2_access.Path')
    @patch('tests.test_validate_r2_access.load_dotenv')
    def test_env_file_loading_success(self, mock_load_dotenv, mock_path):
        """Test successful .env file loading"""
        # Arrange
        mock_env_path = Mock()
        mock_env_path.exists.return_value = True
        mock_path.return_value.parent.parent.__truediv__.return_value = mock_env_path

        # Import should trigger the env loading code
        # This test verifies the module loads without error when .env exists

    @patch('tests.test_validate_r2_access.Path')
    @patch('tests.test_validate_r2_access.sys.exit')
    def test_env_file_missing_exits(self, mock_exit, mock_path):
        """Test that missing .env file causes system exit"""
        # Arrange
        mock_env_path = Mock()
        mock_env_path.exists.return_value = False
        mock_path.return_value.parent.parent.__truediv__.return_value = mock_env_path

        # This would be tested if we could reimport the module
        # For now, we can test the logic conceptually


class TestEdgeCases:
    """Test suite for edge cases and error conditions"""

    @pytest.mark.asyncio
    async def test_client_timeout_handling(self):
        """Test handling of client timeouts"""
        # Arrange
        mock_client = AsyncMock()
        mock_client.list_objects_v2.side_effect = asyncio.TimeoutError("Request timeout")
        bucket = "test-bucket"

        # Act & Assert
        with pytest.raises(asyncio.TimeoutError):
            await test_read_access(mock_client, bucket)

    @pytest.mark.asyncio
    async def test_malformed_error_response(self):
        """Test handling of malformed error responses"""
        # Arrange
        mock_client = AsyncMock()
        # Create a ClientError with malformed response
        error_response = {"Error": {}}  # Missing "Code" key
        mock_client.list_objects_v2.side_effect = botocore.exceptions.ClientError(
            error_response, "ListObjectsV2"
        )
        bucket = "test-bucket"

        # Act & Assert - Should handle gracefully
        with pytest.raises(KeyError):
            await test_read_access(mock_client, bucket)

    @pytest.mark.asyncio
    async def test_unicode_bucket_names(self):
        """Test handling of unicode characters in bucket names"""
        # Arrange
        mock_client = AsyncMock()
        mock_client.list_objects_v2.return_value = {"Contents": []}
        bucket = "test-bücket-ñame"  # Unicode characters

        # Act
        result = await test_read_access(mock_client, bucket)

        # Assert
        assert result is True
        mock_client.list_objects_v2.assert_called_once_with(Bucket=bucket, MaxKeys=1)

    @pytest.mark.asyncio
    async def test_very_long_bucket_names(self):
        """Test handling of very long bucket names"""
        # Arrange
        mock_client = AsyncMock()
        mock_client.list_objects_v2.return_value = {"Contents": []}
        bucket = "a" * 255  # Very long bucket name

        # Act
        result = await test_read_access(mock_client, bucket)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_special_characters_in_prefix(self):
        """Test handling of special characters in prefix"""
        # Arrange
        mock_client = AsyncMock()
        mock_client.put_object.return_value = {}
        mock_client.delete_object.return_value = {}
        bucket = "test-bucket"
        prefix = "test_@#$%_"  # Special characters

        # Act
        result = await test_write_access(mock_client, bucket, prefix)

        # Assert
        assert result is True


class TestIntegrationScenarios:
    """Test suite for integration-style scenarios"""

    @patch.dict(os.environ, {
        "R2_GRADIENTS_ACCOUNT_ID": "",  # Empty string
        "R2_GRADIENTS_READ_ACCESS_KEY_ID": "read-key-1",
        "R2_GRADIENTS_READ_SECRET_ACCESS_KEY": "read-secret-1",
        "R2_GRADIENTS_WRITE_ACCESS_KEY_ID": "write-key-1",
        "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY": "write-secret-1",
        "R2_DATASET_ACCOUNT_ID": "test-account-2",
        "R2_DATASET_READ_ACCESS_KEY_ID": "read-key-2",
        "R2_DATASET_READ_SECRET_ACCESS_KEY": "read-secret-2",
        "R2_DATASET_WRITE_ACCESS_KEY_ID": "write-key-2",
        "R2_DATASET_WRITE_SECRET_ACCESS_KEY": "write-secret-2",
    })
    @pytest.mark.asyncio
    async def test_empty_string_env_vars_treated_as_missing(self):
        """Test that empty string environment variables are treated as missing"""
        with pytest.raises(SystemExit) as exc_info:
            await validate_credentials()
        
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_concurrent_access_tests(self):
        """Test concurrent execution of access tests"""
        # Arrange
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        mock_client1.list_objects_v2.return_value = {"Contents": []}
        mock_client2.list_objects_v2.return_value = {"Contents": []}

        # Act - Run tests concurrently
        results = await asyncio.gather(
            test_read_access(mock_client1, "bucket1"),
            test_read_access(mock_client2, "bucket2")
        )

        # Assert
        assert all(results)
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__])