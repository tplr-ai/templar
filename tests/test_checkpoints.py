# ruff: noqa
# pylint: disable=all
# mypy: ignore-errors
# type: ignore

import asyncio
import os
import time
import torch
import unittest
from unittest import mock
import tempfile
import glob

from aiobotocore.session import get_session

from templar.checkpoint import CheckpointManager, get_base_url
from templar import __version__
from templar.config import BUCKET_SECRETS
from templar.constants import CF_REGION_NAME

CF_REGION_NAME = "auto"  # Replace with your region if needed


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)


async def upload_exists_in_s3(bucket_name, key, access_key, secret_key, endpoint_url):
    """Check if the uploaded object exists in S3."""
    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=CF_REGION_NAME,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ) as s3_client:
        try:
            await s3_client.head_object(Bucket=bucket_name, Key=key)
            return True
        except Exception as e:
            print(f"Error checking S3 object existence: {e}")
            return False


async def async_delete_from_s3(bucket_name, key, access_key, secret_key, endpoint_url):
    """Asynchronously delete an object from S3."""
    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=CF_REGION_NAME,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ) as s3_client:
        try:
            await s3_client.delete_object(Bucket=bucket_name, Key=key)
            print(f"Deleted {key} from bucket {bucket_name}")
        except Exception as e:
            print(f"Error deleting S3 object: {e}")


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for checkpoints
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(self.temp_dir.name, "checkpoint.pth")

        # Initialize a dummy model
        self.model = DummyModel()

        # Mock wallet
        self.wallet = mock.Mock()
        self.wallet.hotkey.ss58_address = "dummy_hotkey_address"

        # Endpoint URL for S3
        self.endpoint_url = get_base_url(BUCKET_SECRETS["account_id"])

        # S3 credentials
        self.bucket_name = BUCKET_SECRETS["bucket_name"].split("/")[-1]
        self.access_key = BUCKET_SECRETS["write"]["access_key_id"]
        self.secret_key = BUCKET_SECRETS["write"]["secret_access_key"]

    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()

    def test_checkpoint_manager_actual_s3_upload(self):
        checkpoint_manager = CheckpointManager(
            model=self.model,
            checkpoint_path=self.checkpoint_path,
            wallet=self.wallet,
            device="cpu",
        )

        # Save and upload checkpoint
        checkpoint_manager.save_and_upload(global_step=1, block_number=100)

        # Wait for the upload to complete
        checkpoint_manager.thread_pool.shutdown(wait=True)
        checkpoint_manager.cleanup()

        # Assert that the checkpoint file was created locally
        self.assertTrue(
            os.path.exists(checkpoint_manager.checkpoint_path),
            "Checkpoint file should exist locally",
        )

        # Check if the file exists in S3
        filename = os.path.basename(checkpoint_manager.checkpoint_path)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            s3_exists = loop.run_until_complete(
                upload_exists_in_s3(
                    self.bucket_name,
                    filename,
                    self.access_key,
                    self.secret_key,
                    self.endpoint_url,
                )
            )
            self.assertTrue(s3_exists, "Checkpoint file should be uploaded to S3")

            # Clean up: Delete the object from S3
            loop.run_until_complete(
                async_delete_from_s3(
                    self.bucket_name,
                    filename,
                    self.access_key,
                    self.secret_key,
                    self.endpoint_url,
                )
            )
        finally:
            loop.close()

    def test_checkpoint_upload_failure(self):
        # Simulate a failure in uploading by providing incorrect S3 credentials
        faulty_access_key = "invalid_access_key"
        faulty_secret_key = "invalid_secret_key"

        checkpoint_manager = CheckpointManager(
            model=self.model,
            checkpoint_path=self.checkpoint_path,
            wallet=self.wallet,
            device="cpu",
        )

        # Patch the BUCKET_SECRETS to use faulty credentials
        with mock.patch(
            "templar.checkpoint.BUCKET_SECRETS",
            {
                "bucket_name": BUCKET_SECRETS["bucket_name"],
                "account_id": BUCKET_SECRETS["account_id"],
                "write": {
                    "access_key_id": faulty_access_key,
                    "secret_access_key": faulty_secret_key,
                },
            },
        ):
            # Save and upload checkpoint
            checkpoint_manager.save_and_upload(global_step=1, block_number=100)
            checkpoint_manager.thread_pool.shutdown(wait=True)
            checkpoint_manager.cleanup()

            # Ensure that the checkpoint file was saved locally
            self.assertTrue(
                os.path.exists(checkpoint_manager.checkpoint_path),
                "Checkpoint file should exist locally",
            )

            # Check that the file was not uploaded to S3
            filename = os.path.basename(checkpoint_manager.checkpoint_path)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                s3_exists = loop.run_until_complete(
                    upload_exists_in_s3(
                        self.bucket_name,
                        filename,
                        self.access_key,
                        self.secret_key,
                        self.endpoint_url,
                    )
                )
                self.assertFalse(
                    s3_exists,
                    "Checkpoint file should not be uploaded to S3 due to faulty credentials",
                )
            finally:
                loop.close()

    def test_checkpoint_cleanup_old_checkpoints(self):
        checkpoint_manager = CheckpointManager(
            model=self.model,
            checkpoint_path=self.checkpoint_path,
            wallet=self.wallet,
            device="cpu",
        )

        # Save multiple checkpoints
        for i in range(5):
            checkpoint_manager.save_and_upload(global_step=i, block_number=100 + i)

        # Wait for all uploads and cleanups to complete
        checkpoint_manager.thread_pool.shutdown(wait=True)
        checkpoint_manager.cleanup()

        # Wait a bit to ensure all background tasks have completed
        time.sleep(5)

        # Check that only the latest 3 checkpoints are kept locally
        pattern = os.path.join(
            checkpoint_manager.checkpoint_dir,
            f"neuron_checkpoint_{self.wallet.hotkey.ss58_address}_b*_v{__version__}.pth",
        )
        checkpoint_files = glob.glob(pattern)
        self.assertEqual(
            len(checkpoint_files),
            3,
            "Only the latest 3 checkpoints should be kept locally",
        )

        # Check that only the latest 3 checkpoints exist in S3
        filenames = [os.path.basename(f) for f in checkpoint_files]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            tasks = [
                upload_exists_in_s3(
                    self.bucket_name,
                    filename,
                    self.access_key,
                    self.secret_key,
                    self.endpoint_url,
                )
                for filename in filenames
            ]
            results = loop.run_until_complete(asyncio.gather(*tasks))
            self.assertTrue(all(results), "Latest checkpoints should exist in S3")

            # Check that older checkpoints are deleted from S3
            # Assuming block numbers 100 to 104, checkpoints with block_number 100 and 101 should be deleted
            old_filenames = [
                f"neuron_checkpoint_{self.wallet.hotkey.ss58_address}_b{100 + i}_v{__version__}.pth"
                for i in range(2)
            ]
            tasks = [
                upload_exists_in_s3(
                    self.bucket_name,
                    filename,
                    self.access_key,
                    self.secret_key,
                    self.endpoint_url,
                )
                for filename in old_filenames
            ]
            results = loop.run_until_complete(asyncio.gather(*tasks))
            self.assertFalse(any(results), "Old checkpoints should be deleted from S3")
        finally:
            # Clean up: Delete remaining objects from S3
            tasks = [
                async_delete_from_s3(
                    self.bucket_name,
                    filename,
                    self.access_key,
                    self.secret_key,
                    self.endpoint_url,
                )
                for filename in filenames + old_filenames
            ]
            loop.run_until_complete(asyncio.gather(*tasks))
            loop.close()

    def test_checkpoint_save_failure(self):
        # Simulate a failure in saving the checkpoint locally by mocking torch.save
        checkpoint_manager = CheckpointManager(
            model=self.model,
            checkpoint_path=self.checkpoint_path,
            wallet=self.wallet,
            device="cpu",
        )

        with mock.patch("torch.save", side_effect=IOError("Simulated IO Error")):
            # Attempt to save and upload checkpoint
            checkpoint_manager.save_and_upload(global_step=1, block_number=100)
            checkpoint_manager.thread_pool.shutdown(wait=True)
            checkpoint_manager.cleanup()

            # Ensure that the checkpoint file was not created
            self.assertFalse(
                os.path.exists(checkpoint_manager.checkpoint_path),
                "Checkpoint file should not exist due to save failure",
            )

    def test_loading_from_highest_stake(self):
        # This test would require a mock of the metagraph and buckets
        # For simplicity, let's assume the methods return expected values
        checkpoint_manager = CheckpointManager(
            model=self.model,
            checkpoint_path=self.checkpoint_path,
            wallet=self.wallet,
            device="cpu",
        )

        # Mocking methods
        metagraph = mock.Mock()
        metagraph.hotkeys = ["dummy_hotkey_address", "other_hotkey"]
        metagraph.uids = [0, 1]
        metagraph.S = torch.tensor([10, 5])  # Highest stake at index 0

        buckets = [mock.Mock(), None]  # Only the first neuron has a valid bucket

        # Mock download_checkpoint_from_neuron to simulate downloading a checkpoint
        with mock.patch(
            "templar.checkpoint.download_checkpoint_from_neuron",
            return_value=self.checkpoint_path,
        ):
            # Mock load_checkpoint to simulate loading the checkpoint
            with mock.patch(
                "templar.checkpoint.load_checkpoint", return_value=(100, {})
            ):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    global_step = loop.run_until_complete(
                        checkpoint_manager.load_from_highest_stake(metagraph, buckets)
                    )
                    self.assertEqual(
                        global_step, 100, "Global step should be loaded from checkpoint"
                    )
                finally:
                    loop.close()
