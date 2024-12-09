import asyncio
import os
import threading
import time
import torch
import unittest
from unittest import mock
import tempfile

# Import your CheckpointManager and necessary components
from templar.checkpoint import CheckpointManager
from templar import __version__
from templar.config import BUCKET_SECRETS

# Use aiobotocore for asyncio-compatible S3 client
from aiobotocore.session import get_session

CF_REGION_NAME = 'auto'  # Replace with your region if needed

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

async def upload_exists_in_s3(bucket_name, key, access_key, secret_key, endpoint_url):
    """Check if the uploaded object exists in S3."""
    session = get_session()
    async with session.create_client(
        's3',
        endpoint_url=endpoint_url,
        region_name=CF_REGION_NAME,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    ) as s3_client:
        try:
            response = await s3_client.head_object(Bucket=bucket_name, Key=key)
            return True
        except Exception as e:
            print(f"Error checking S3 object existence: {e}")
            return False

def test_checkpoint_manager_actual_s3_upload():
    # Create a temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize a dummy model
        model = DummyModel()

        # Define checkpoint path
        checkpoint_path = os.path.join(temp_dir, 'checkpoint.pth')

        # Mock wallet
        wallet = mock.Mock()
        wallet.hotkey.ss58_address = 'dummy_hotkey_address'

        # Initialize CheckpointManager
        checkpoint_manager = CheckpointManager(
            model=model,
            checkpoint_path=checkpoint_path,
            wallet=wallet,
            device='cpu'
        )

        # Save and upload checkpoint
        checkpoint_manager.save_and_upload(global_step=1)

        # Allow some time for the background thread to process the upload
        time.sleep(5)

        # Shutdown the manager to process remaining uploads
        checkpoint_manager.cleanup()

        # Assert that the checkpoint file was created locally
        assert os.path.exists(checkpoint_path), "Checkpoint file should exist locally"

        # Check if the file exists in S3
        filename = f"neuron_checkpoints_{wallet.hotkey.ss58_address}_v{__version__}.pth"
        bucket_name = BUCKET_SECRETS['bucket_name']
        access_key = BUCKET_SECRETS['write']['access_key_id']
        secret_key = BUCKET_SECRETS['write']['secret_access_key']
        endpoint_url = f"https://{BUCKET_SECRETS['account_id']}.r2.cloudflarestorage.com"

        loop = asyncio.get_event_loop()
        s3_exists = loop.run_until_complete(
            upload_exists_in_s3(bucket_name, filename, access_key, secret_key, endpoint_url)
        )

        assert s3_exists, "Checkpoint file should be uploaded to S3"

        # Clean up: Delete the object from S3
        session = get_session()
        loop.run_until_complete(async_delete_from_s3(
            bucket_name, filename, access_key, secret_key, endpoint_url
        ))

async def async_delete_from_s3(bucket_name, key, access_key, secret_key, endpoint_url):
    session = get_session()
    async with session.create_client(
        's3',
        endpoint_url=endpoint_url,
        region_name=CF_REGION_NAME,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    ) as s3_client:
        try:
            await s3_client.delete_object(Bucket=bucket_name, Key=key)
            print(f"Deleted {key} from bucket {bucket_name}")
        except Exception as e:
            print(f"Error deleting S3 object: {e}")
