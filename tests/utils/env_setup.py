"""Test environment setup utilities"""
import os
from pathlib import Path

def setup_test_environment():
    """Setup test environment variables"""
    # Set default test environment variables
    test_env = {
        # Gradients bucket config
        "R2_GRADIENTS_ACCOUNT_ID": "test_account",
        "R2_GRADIENTS_BUCKET_NAME": "test-bucket",
        "R2_GRADIENTS_READ_ACCESS_KEY_ID": "test_read_key",
        "R2_GRADIENTS_READ_SECRET_ACCESS_KEY": "test_read_secret",
        "R2_GRADIENTS_WRITE_ACCESS_KEY_ID": "test_write_key",
        "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY": "test_write_secret",
        
        # Dataset bucket config
        "R2_DATASET_BUCKET_NAME": "test-dataset-bucket",
        "R2_DATASET_ACCOUNT_ID": "test_dataset_account",
        "R2_DATASET_READ_ACCESS_KEY_ID": "test_dataset_read_key",
        "R2_DATASET_READ_SECRET_ACCESS_KEY": "test_dataset_read_secret",
        
        # Additional configs
        "WANDB_MODE": "disabled",  # Disable wandb during tests
        "PYTEST_RUNNING": "1",
        
        # Mock API endpoints
        "R2_API_ENDPOINT": "https://test-endpoint.com",
        "R2_DATASET_API_ENDPOINT": "https://test-dataset-endpoint.com",
        
        # Optional configs with defaults
        "MOCK_RESPONSES": "1",
        "TEST_MODE": "1",
        "DISABLE_WANDB": "1"
    }
    
    # Set environment variables if not already set
    for key, value in test_env.items():
        if key not in os.environ:
            os.environ[key] = value

    # Verify required variables are set
    required_vars = [
        "R2_DATASET_ACCOUNT_ID",
        "R2_DATASET_READ_ACCESS_KEY_ID",
        "R2_DATASET_READ_SECRET_ACCESS_KEY",
        "R2_DATASET_BUCKET_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        raise ImportError(f"Required environment variables missing: {missing_vars}") 