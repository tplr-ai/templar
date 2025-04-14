import os
import time
import torch
import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Set required environment variables before importing tplr
required_env_vars = [
    "R2_GRADIENTS_ACCOUNT_ID",
    "R2_GRADIENTS_BUCKET_NAME",
    "R2_GRADIENTS_READ_ACCESS_KEY_ID",
    "R2_GRADIENTS_READ_SECRET_ACCESS_KEY",
    "R2_GRADIENTS_WRITE_ACCESS_KEY_ID",
    "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY",
    "R2_AGGREGATOR_ACCOUNT_ID",
    "R2_AGGREGATOR_BUCKET_NAME",
    "R2_AGGREGATOR_READ_ACCESS_KEY_ID",
    "R2_AGGREGATOR_READ_SECRET_ACCESS_KEY",
    "R2_DATASET_ACCOUNT_ID",
    "R2_DATASET_BUCKET_NAME",
    "R2_DATASET_READ_ACCESS_KEY_ID",
    "R2_DATASET_READ_SECRET_ACCESS_KEY",
]

# Set mock values for testing
for var in required_env_vars:
    os.environ[var] = "mock_value"

# Now we can import tplr modules
with patch(
    "tplr.config.load_bucket_secrets",
    return_value={
        "gradients": {
            "account_id": "mock_account",
            "name": "mock_bucket",
            "credentials": {
                "read": {
                    "access_key_id": "mock_key",
                    "secret_access_key": "mock_secret",
                },
                "write": {
                    "access_key_id": "mock_key",
                    "secret_access_key": "mock_secret",
                },
            },
        },
        "aggregator": {
            "account_id": "mock_account",
            "name": "mock_bucket",
            "credentials": {
                "read": {
                    "access_key_id": "mock_key",
                    "secret_access_key": "mock_secret",
                },
                "write": {
                    "access_key_id": "mock_key",
                    "secret_access_key": "mock_secret",
                },
            },
        },
        "dataset": {
            "account_id": "mock_account",
            "name": "mock_bucket",
            "credentials": {
                "read": {
                    "access_key_id": "mock_key",
                    "secret_access_key": "mock_secret",
                }
            },
        },
    },
):
    import tplr
    from tplr.comms import Comms


class TestBatchTensorProcessing:
    """Tests for the batch tensor processing in gather."""

    @pytest.fixture
    def mock_comms(self):
        """Create a mock Comms instance with required attributes."""
        comms = MagicMock(spec=Comms)
        comms._validate_response = Comms._validate_response
        comms._batch_validate_tensors = Comms._batch_validate_tensors
        comms._batch_normalize_tensors = Comms._batch_normalize_tensors
        comms.gather = Comms.gather
        comms.check_compressed_indices = MagicMock()
        comms.client_semaphore = MagicMock()
        comms.client_semaphore.__aenter__ = AsyncMock()
        comms.client_semaphore.__aexit__ = AsyncMock()
        comms.hparams = SimpleNamespace(topk_compression=5)

        # Mock logger
        tplr.logger = MagicMock()

        return comms

    @pytest.fixture
    def mock_responses(self):
        """Create mock tensor responses for testing."""
        # Create some test tensors
        idxs1 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int16)
        vals1 = torch.randn(5)
        idxs2 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int16)
        vals2 = torch.randn(5)

        # Create responses with tensor dictionaries
        responses = [
            (
                {
                    "model.weight.0idxs": idxs1,
                    "model.weight.0vals": vals1,
                    "metadata": {"value": 1},
                },
                0,  # global_step
            ),
            (
                {
                    "model.weight.0idxs": idxs2,
                    "model.weight.0vals": vals2,
                    "metadata": {"value": 2},
                },
                1,  # global_step
            ),
            None,  # Missing response
            Exception("Test exception"),  # Error response
        ]
        return responses

    @pytest.mark.asyncio
    async def test_gather_with_batch_processing(self, mock_comms, mock_responses):
        """Test the gather function with batch tensor processing."""
        # Set up the mock for get_with_retry
        mock_comms.get_with_retry = AsyncMock(side_effect=mock_responses)

        # Set up totalks
        totalks = {"model.weight.0": 10}

        # Test gather
        result = await mock_comms.gather(
            my_uid=0,
            uids=[1, 2, 3, 4],
            window=5,
            key="gradient",
            timeout=10,
            device="cpu",
            totalks=totalks,
            local=True,
            stale_retention=10,
            time_min=datetime.now(timezone.utc),
            time_max=datetime.now(timezone.utc),
        )

        # Verify result structure
        assert result is not None
        assert hasattr(result, "state_dict")
        assert hasattr(result, "uids")
        assert hasattr(result, "global_steps")
        assert hasattr(result, "skipped_uids")

        # Verify that get_with_retry was called for each UID
        assert mock_comms.get_with_retry.call_count == 4

    @pytest.mark.asyncio
    async def test_tensor_validation(self, mock_comms):
        """Test the tensor validation helper function."""
        # Create test tensors with some invalid values
        valid_tensor = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int16)
        invalid_tensor = torch.tensor(
            [0, 1, 2, 100, 4], dtype=torch.int16
        )  # 100 is out of bounds
        nan_tensor = torch.tensor([1.0, 2.0, float("nan"), 4.0, 5.0])

        # Set up test data
        param_name = "model.weight.0idxs"
        tensors = [valid_tensor, invalid_tensor, nan_tensor]
        uids = [1, 2, 3]
        device = "cpu"
        totalks = {"model.weight.0": 10}

        # Test batch validation
        valid_tensors, valid_uids = mock_comms._batch_validate_tensors(
            param_name, tensors, uids, device, totalks
        )

        # Only the valid tensor should pass validation
        assert len(valid_tensors) == 1
        assert valid_uids == [1]

        # Test with vals parameters
        param_name = "model.weight.0vals"
        tensors = [torch.tensor([1.0, 2.0]), nan_tensor]
        uids = [1, 3]

        valid_tensors, valid_uids = mock_comms._batch_validate_tensors(
            param_name, tensors, uids, device, totalks
        )

        # Only the valid tensor should pass validation
        assert len(valid_tensors) == 1
        assert valid_uids == [1]

    def test_batch_normalization(self, mock_comms):
        """Test the batch normalization helper function."""
        # Create test tensors with different shapes
        tensor1 = torch.tensor([1.0, 0.0, 0.0])  # Unit vector in x direction
        tensor2 = torch.tensor([0.0, 2.0, 0.0])  # Vector in y direction
        tensor3 = torch.tensor([0.0, 0.0, 3.0])  # Vector in z direction

        # Same shape tensors can be batch processed
        tensors = [tensor1, tensor2, tensor3]
        normalized = mock_comms._batch_normalize_tensors(tensors, "cpu")

        # Verify each vector is normalized (unit length)
        for tensor in normalized:
            # Allow small floating point error
            assert abs(torch.norm(tensor).item() - 1.0) < 1e-6

        # Different shape tensors should be processed individually
        tensors = [tensor1, torch.tensor([[1.0, 2.0], [3.0, 4.0]])]
        normalized = mock_comms._batch_normalize_tensors(tensors, "cpu")

        # Still should have same number of tensors
        assert len(normalized) == 2
        # First tensor still normalized
        assert abs(torch.norm(normalized[0]).item() - 1.0) < 1e-6
        # Second tensor normalized (matrix norm)
        assert abs(torch.norm(normalized[1]).item() - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_gather_performance(self, mock_comms):
        """Test that batch processing improves performance."""
        # Create a larger number of test responses
        num_responses = 20

        # Generate test data
        idxs_list = [
            torch.tensor([i, i + 1, i + 2, i + 3, i + 4], dtype=torch.int16)
            for i in range(num_responses)
        ]
        vals_list = [torch.randn(5) for _ in range(num_responses)]

        responses = [
            (
                {
                    "model.weight.0idxs": idxs,
                    "model.weight.0vals": vals,
                    "metadata": {"value": i},
                },
                i,  # global_step
            )
            for i, (idxs, vals) in enumerate(zip(idxs_list, vals_list))
        ]

        # Set up the mock
        mock_comms.get_with_retry = AsyncMock(side_effect=responses)
        mock_comms.check_compressed_indices = MagicMock(
            return_value=None
        )  # Always pass validation

        # Set up totalks
        totalks = {"model.weight.0": 100}

        # Measure time for gather
        start_time = time.time()
        result = await mock_comms.gather(
            my_uid=0,
            uids=list(range(num_responses)),
            window=5,
            key="gradient",
            timeout=10,
            device="cpu",
            totalks=totalks,
            local=True,
            stale_retention=10,
            time_min=datetime.now(timezone.utc),
            time_max=datetime.now(timezone.utc),
        )
        end_time = time.time()

        # Verify we got results for all tensors
        assert result is not None
        assert len(result.uids) == num_responses

        # Log the time taken - we can't assert a specific time improvement
        # but this will be useful for manual comparison
        print(
            f"Time taken for batch processing {num_responses} tensors: {end_time - start_time:.4f}s"
        )
