import asyncio
import unittest
from unittest.mock import MagicMock, patch

import pytest

from tplr.chain import ChainManager
from tplr.schemas import Bucket


class TestChainManager(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_metagraph = MagicMock()
        self.mock_hparams = MagicMock()
        self.mock_wallet = MagicMock()
        self.mock_bucket = Bucket(
            name="test_bucket",
            account_id="test_account_id",
            access_key_id="test_access_key_id",
            secret_access_key="test_secret_access_key",
        )

        with patch("bittensor.subtensor") as mock_subtensor:
            self.mock_config.netuid = 1
            self.chain_manager = ChainManager(
                config=self.mock_config,
                hparams=self.mock_hparams,
                wallet=self.mock_wallet,
                bucket=self.mock_bucket,
            )
            self.mock_subtensor_instance = mock_subtensor.return_value
            self.chain_manager.metagraph = self.mock_metagraph

    def test_init(self):
        self.assertEqual(self.chain_manager.config, self.mock_config)
        self.assertEqual(self.chain_manager.netuid, 1)
        self.assertEqual(self.chain_manager.metagraph, self.mock_metagraph)
        self.assertEqual(self.chain_manager.hparams, self.mock_hparams)
        self.assertEqual(self.chain_manager.wallet, self.mock_wallet)
        self.assertEqual(self.chain_manager.bucket, self.mock_bucket)

    def test_get_bucket(self):
        self.chain_manager.commitments = {1: self.mock_bucket}
        bucket = self.chain_manager.get_bucket(1)
        self.assertEqual(bucket, self.mock_bucket)

    def test_get_all_buckets(self):
        self.mock_metagraph.uids = [1, 2]
        self.chain_manager.commitments = {
            1: self.mock_bucket,
            2: self.mock_bucket,
        }
        buckets = self.chain_manager.get_all_buckets()
        self.assertEqual(len(buckets), 2)
        self.assertEqual(buckets[1], self.mock_bucket)
        self.assertEqual(buckets[2], self.mock_bucket)

    def test_commit(self):
        self.chain_manager.commit(self.mock_wallet, self.mock_bucket)
        self.mock_subtensor_instance.commit.assert_called_once()

    def test_try_commit_no_change(self):
        self.mock_wallet.hotkey.ss58_address = "test_address"
        self.mock_metagraph.hotkeys = ["test_address"]
        with patch.object(
            self.chain_manager,
            "get_commitment",
            return_value=self.mock_bucket,
        ) as mock_get_commitment:
            self.chain_manager.try_commit(self.mock_wallet, self.mock_bucket)
            mock_get_commitment.assert_called_once()
            self.mock_subtensor_instance.commit.assert_not_called()

    def test_try_commit_with_change(self):
        self.mock_wallet.hotkey.ss58_address = "test_address"
        self.mock_metagraph.hotkeys = ["test_address"]
        changed_bucket = Bucket(
            name="changed_bucket",
            account_id="changed_account_id",
            access_key_id="changed_access_key_id",
            secret_access_key="changed_secret_access_key",
        )
        with patch.object(
            self.chain_manager, "get_commitment", return_value=changed_bucket
        ) as mock_get_commitment:
            self.chain_manager.try_commit(self.mock_wallet, self.mock_bucket)
            mock_get_commitment.assert_called_once()
            self.mock_subtensor_instance.commit.assert_called_once()

    def test_get_commitment(self):
        self.mock_subtensor_instance.get_commitment.return_value = (
            "a" * 32 + "b" * 32 + "c" * 64
        )
        bucket = self.chain_manager.get_commitment(1)
        self.assertEqual(bucket.name, "a" * 32)
        self.assertEqual(bucket.account_id, "a" * 32)
        self.assertEqual(bucket.access_key_id, "b" * 32)
        self.assertEqual(bucket.secret_access_key, "c" * 64)

    def test_decode_metadata(self):
        encoded_ss58 = (b"\x01" * 32,)
        metadata = {
            "info": {
                "fields": [
                    [
                        {
                            "Commitment": (
                                b"\x02" * 128,
                                0,
                            )
                        }
                    ]
                ]
            }
        }
        with patch("tplr.chain.decode_account_id", return_value="decoded_address"):
            decoded_key, commitment = self.chain_manager.decode_metadata(
                encoded_ss58, metadata
            )
            self.assertEqual(decoded_key, "decoded_address")
            self.assertEqual(commitment, "\x02" * 128)

    @pytest.mark.asyncio
    async def test_get_commitments(self):
        self.mock_subtensor_instance.substrate.query_map.return_value = []
        commitments = await self.chain_manager.get_commitments()
        self.assertEqual(commitments, {})

    @pytest.mark.asyncio
    async def test_get_bucket_for_neuron(self):
        self.mock_wallet.hotkey.ss58_address = "test_address"
        self.mock_metagraph.hotkeys = ["test_address"]
        self.chain_manager.commitments = {0: self.mock_bucket}
        bucket = await self.chain_manager.get_bucket_for_neuron(self.mock_wallet)
        self.assertEqual(bucket, self.mock_bucket)

    @pytest.mark.asyncio
    async def test_fetch_commitments(self):
        with patch.object(
            self.chain_manager, "get_commitments", return_value={1: self.mock_bucket}
        ) as mock_get_commitments:
            await self.chain_manager.fetch_commitments()
            mock_get_commitments.assert_called_once()
            self.assertEqual(self.chain_manager.commitments, {1: self.mock_bucket})

    def test_get_hotkey(self):
        self.mock_metagraph.uids = [1, 2]
        self.mock_metagraph.hotkeys = ["hotkey1", "hotkey2"]
        hotkey = self.chain_manager.get_hotkey(1)
        self.assertEqual(hotkey, "hotkey1")

    def test_update_peers_with_buckets(self):
        self.mock_metagraph.uids.tolist.return_value = [1, 2]
        self.mock_metagraph.S.tolist.return_value = [100, 200]
        self.chain_manager.active_peers = {1, 2}
        self.chain_manager.eval_peers = {1: 1}
        self.chain_manager.update_peers_with_buckets()
        self.assertEqual(self.chain_manager.eval_peers, {1: 1, 2: 1})
        self.assertEqual(self.chain_manager.inactive_peers, set())


if __name__ == "__main__":
    unittest.main()
