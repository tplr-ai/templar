# ruff: noqa
# pylint: disable=all
# mypy: ignore-errors
# type: ignore

import asyncio
import os
import torch
import unittest
from unittest import mock
import tempfile
import glob
from aiobotocore.session import get_session

from templar.checkpoint import (
    CheckpointManager,
    get_base_url,
    load_checkpoint,
    download_checkpoint_from_neuron,
)
from templar import __version__
from templar.config import BUCKET_SECRETS
from templar.constants import CF_REGION_NAME


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)


async def upload_exists_in_s3(bucket_name, key, access_key, secret_key, endpoint_url):
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
        except Exception:
            return False


async def async_delete_from_s3(bucket_name, key, access_key, secret_key, endpoint_url):
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
        except Exception as e:
            print(f"Error deleting S3 object: {e}")


class TestCheckpointManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(self.temp_dir.name, "checkpoint.pth")
        self.model = DummyModel()
        self.wallet = mock.Mock()
        self.wallet.hotkey.ss58_address = "dummy_hotkey_address"
        self.endpoint_url = get_base_url(BUCKET_SECRETS["account_id"])
        self.bucket_name = BUCKET_SECRETS["bucket_name"].split("/")[-1]
        self.access_key = BUCKET_SECRETS["write"]["access_key_id"]
        self.secret_key = BUCKET_SECRETS["write"]["secret_access_key"]
        self.original_bucket_secrets = BUCKET_SECRETS.copy()

    def tearDown(self):
        self.temp_dir.cleanup()
        for key in self.original_bucket_secrets:
            BUCKET_SECRETS[key] = self.original_bucket_secrets[key]

    async def test_async_behavior(self):
        # Example test that ensures `save_and_upload` doesn't block
        checkpoint_manager = CheckpointManager(
            model=self.model,
            checkpoint_path=self.checkpoint_path,
            wallet=self.wallet,
            device="cpu",
        )

        await checkpoint_manager.save_and_upload(global_step=1, block_number=100)
        self.assertTrue(os.path.exists(checkpoint_manager.checkpoint_path))
        checkpoint_manager.cleanup()

    async def test_checkpoint_cleanup(self):
        checkpoint_manager = CheckpointManager(
            model=self.model,
            checkpoint_path=self.checkpoint_path,
            wallet=self.wallet,
            device="cpu",
        )

        # Save multiple checkpoints
        for i in range(5):
            await checkpoint_manager.save_and_upload(
                global_step=i, block_number=100 + i
            )

        # Wait a moment for async tasks to finish
        await asyncio.sleep(5)

        # Check that only the latest 3 checkpoints remain locally
        pattern = os.path.join(
            checkpoint_manager.checkpoint_dir,
            f"neuron_checkpoint_{self.wallet.hotkey.ss58_address}_b*_v{__version__}.pth",
        )
        files = glob.glob(pattern)
        self.assertEqual(len(files), 3)
        checkpoint_manager.cleanup()

    async def test_checkpoint_local_save(self):
        # If you previously tested `save_checkpoint` directly, now test `save_and_upload`
        # to ensure a checkpoint is saved locally without error.
        checkpoint_manager = CheckpointManager(
            model=self.model,
            checkpoint_path=self.checkpoint_path,
            wallet=self.wallet,
            device="cpu",
        )
        await checkpoint_manager.save_and_upload(global_step=1, block_number=100)
        self.assertTrue(os.path.exists(checkpoint_manager.checkpoint_path))
        checkpoint_manager.cleanup()

    async def test_checkpoint_s3_upload(self):
        checkpoint_manager = CheckpointManager(
            model=self.model,
            checkpoint_path=self.checkpoint_path,
            wallet=self.wallet,
            device="cpu",
        )
        await checkpoint_manager.save_and_upload(global_step=1, block_number=100)

        # Wait for the upload to complete
        if checkpoint_manager.upload_task:
            await checkpoint_manager.upload_task

        filename = os.path.basename(checkpoint_manager.checkpoint_path)
        s3_exists = await upload_exists_in_s3(
            self.bucket_name,
            filename,
            self.access_key,
            self.secret_key,
            self.endpoint_url,
        )
        self.assertTrue(s3_exists)

        # Cleanup
        await async_delete_from_s3(
            self.bucket_name,
            filename,
            self.access_key,
            self.secret_key,
            self.endpoint_url,
        )
        checkpoint_manager.cleanup()

    async def test_checkpoint_upload_with_invalid_credentials(self):
        faulty_access_key = "invalid_access_key"
        faulty_secret_key = "invalid_secret_key"

        with mock.patch.dict(
            "templar.config.BUCKET_SECRETS",
            {
                "bucket_name": BUCKET_SECRETS["bucket_name"],
                "account_id": BUCKET_SECRETS["account_id"],
                "write": {
                    "access_key_id": faulty_access_key,
                    "secret_access_key": faulty_secret_key,
                },
            },
            clear=False,
        ):
            checkpoint_manager = CheckpointManager(
                model=self.model,
                checkpoint_path=self.checkpoint_path,
                wallet=self.wallet,
                device="cpu",
            )
            await checkpoint_manager.save_and_upload(global_step=1, block_number=100)

            # Check if NOT uploaded with correct creds
            filename = os.path.basename(checkpoint_manager.checkpoint_path)
            s3_exists = await upload_exists_in_s3(
                self.bucket_name,
                filename,
                self.access_key,
                self.secret_key,
                self.endpoint_url,
            )
            self.assertFalse(s3_exists)
            checkpoint_manager.cleanup()

    async def test_invalid_checkpoint_path(self):
        # Test what happens if the checkpoint directory is invalid
        invalid_path = "/invalid_dir/checkpoint.pth"

        with self.assertRaises(PermissionError):
            checkpoint_manager = CheckpointManager(
                model=self.model,
                checkpoint_path=invalid_path,
                wallet=self.wallet,
                device="cpu",
            )
            # No need to proceed if initialization fails
            await checkpoint_manager.save_and_upload(global_step=1, block_number=100)

        # No cleanup needed since initialization failed
