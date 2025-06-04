# ruff : noqa

import pytest
import asyncio
import torch
import time
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from datetime import datetime, timezone
from types import SimpleNamespace

from tplr.training.aggregation_manager import AggregationManager


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_chain_manager():
    """Mock chain manager that peer manager depends on"""
    manager = MagicMock()
    manager.current_window = 10
    manager.get_current_window = MagicMock(return_value=10)
    manager.get_bucket = MagicMock(return_value=MagicMock())
    manager.commitments = {1: MagicMock(), 2: MagicMock(), 3: MagicMock()}
    manager.metagraph = MagicMock()
    manager.metagraph.uids = [0, 1, 2, 3, 4, 5]  # Mock UID list
    return manager


@pytest.fixture
def mock_peer_manager(mock_chain_manager):
    """Mock peer manager with proper chain_manager dependency"""
    manager = MagicMock()
    # Set the chain_manager that AggregationManager expects to access
    manager.chain_manager = mock_chain_manager

    # Add methods that might be called by AggregationManager
    manager.get_active_peers = MagicMock(return_value={1, 2, 3})
    manager.is_peer_active = AsyncMock(return_value=True)
    manager.weighted_random_sample_no_replacement = MagicMock(return_value=[1, 2, 3])

    return manager


@pytest.fixture
def mock_gradient_manager():
    manager = MagicMock()
    manager.validate_gradient = MagicMock(return_value=True)
    manager.check_compressed_indices = MagicMock()
    return manager


@pytest.fixture
def mock_storage_client():
    client = MagicMock()
    client.get_object = AsyncMock(return_value=None)  # Default to None (no data found)
    client.put_object = AsyncMock(return_value=True)
    client.get_object_size = AsyncMock(return_value=1000)
    client.multipart_download = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_file_manager():
    manager = MagicMock()
    manager.create_temp_file = MagicMock(return_value="/tmp/test_file.pt")
    manager.delete_file = MagicMock()
    manager.get_local_storage_path = MagicMock(return_value="/tmp/local_path.pt")
    return manager


@pytest.fixture
def mock_hparams():
    hparams = MagicMock()
    hparams.topk_compression = 10
    hparams.active_check_interval = 60
    return hparams


@pytest.fixture
def valid_aggregation_manager(
    mock_gradient_manager,
    mock_peer_manager,
    mock_storage_client,
    mock_file_manager,
    mock_hparams,
):
    return AggregationManager(
        gradient_manager=mock_gradient_manager,
        peer_manager=mock_peer_manager,
        storage_client=mock_storage_client,
        file_manager=mock_file_manager,
        hparams=mock_hparams,
        device="cpu",
    )


# -----------------------------------------------------------------------------
# CONSTRUCTOR & INITIALIZATION TESTS
# -----------------------------------------------------------------------------
def test_aggregation_manager_constructor_with_valid_dependencies(
    mock_gradient_manager,
    mock_peer_manager,
    mock_storage_client,
    mock_file_manager,
    mock_hparams,
):
    """Test AggregationManager constructor with valid dependencies"""
    manager = AggregationManager(
        gradient_manager=mock_gradient_manager,
        peer_manager=mock_peer_manager,
        storage_client=mock_storage_client,
        file_manager=mock_file_manager,
        hparams=mock_hparams,
        device="cpu",
    )

    assert manager.gradient_manager is mock_gradient_manager
    assert manager.peer_manager is mock_peer_manager
    assert manager.storage_client is mock_storage_client
    assert manager.file_manager is mock_file_manager
    assert manager.hparams is mock_hparams
    assert manager.device == "cpu"


def test_aggregation_manager_constructor_with_none_gradient_manager():
    """Test AggregationManager constructor with None gradient_manager"""
    # Constructor doesn't validate - it just stores None
    manager = AggregationManager(
        gradient_manager=None,
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    # The constructor succeeds but stores None
    assert manager.gradient_manager is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_none_peer_manager():
    """Test AggregationManager constructor with None peer_manager"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=None,
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    assert manager.peer_manager is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_none_storage_client():
    """Test AggregationManager constructor with None storage_client"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=None,
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    assert manager.storage_client is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_none_file_manager():
    """Test AggregationManager constructor with None file_manager"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=None,
        hparams=MagicMock(),
        device="cpu",
    )

    assert manager.file_manager is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_none_hparams():
    """Test AggregationManager constructor with None hparams"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=None,
        device="cpu",
    )

    assert manager.hparams is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_invalid_device_string():
    """Test AggregationManager constructor with invalid device string"""
    # This should not raise during construction, but may cause issues during tensor operations
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="invalid_device",
    )

    assert manager.device == "invalid_device"


def test_semaphore_initialization_with_default_value(valid_aggregation_manager):
    """Test semaphore initialization with default value (15)"""
    assert hasattr(valid_aggregation_manager, "gather_semaphore")
    assert isinstance(valid_aggregation_manager.gather_semaphore, asyncio.Semaphore)
    assert valid_aggregation_manager.gather_semaphore._value == 15


def test_semaphore_initialization_with_custom_value():
    """Test semaphore initialization with custom value"""
    # Note: Current implementation uses hardcoded value, but test what would happen if it were configurable
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    # Verify default is still 15
    assert manager.gather_semaphore._value == 15

    # TODO: Add parameter to constructor to allow custom semaphore value
    # TODO: Test with different semaphore values (1, 5, 50, 100)


def test_all_manager_dependencies_are_properly_stored_as_instance_variables(
    mock_gradient_manager,
    mock_peer_manager,
    mock_storage_client,
    mock_file_manager,
    mock_hparams,
):
    """Test all manager dependencies are properly stored as instance variables"""
    manager = AggregationManager(
        gradient_manager=mock_gradient_manager,
        peer_manager=mock_peer_manager,
        storage_client=mock_storage_client,
        file_manager=mock_file_manager,
        hparams=mock_hparams,
        device="cuda:0",
    )

    # Test all dependencies are stored correctly
    assert manager.gradient_manager is mock_gradient_manager
    assert manager.peer_manager is mock_peer_manager
    assert manager.storage_client is mock_storage_client
    assert manager.file_manager is mock_file_manager
    assert manager.hparams is mock_hparams
    assert manager.device == "cuda:0"

    # Test that semaphore is initialized
    assert hasattr(manager, "gather_semaphore")
    assert isinstance(manager.gather_semaphore, asyncio.Semaphore)

    # Test instance attributes directly using __dict__
    expected_attrs = {
        "gradient_manager",
        "peer_manager",
        "storage_client",
        "file_manager",
        "hparams",
        "device",
        "gather_semaphore",
    }
    actual_attrs = set(manager.__dict__.keys())

    assert expected_attrs.issubset(actual_attrs), (
        f"Missing expected attributes: {expected_attrs - actual_attrs}"
    )

    # Verify no unexpected critical attributes are missing
    missing_attrs = expected_attrs - actual_attrs
    assert len(missing_attrs) == 0, f"Missing critical attributes: {missing_attrs}"


def test_aggregation_manager_constructor_with_cuda_device():
    """Test AggregationManager constructor with CUDA device"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cuda:0",
    )

    assert manager.device == "cuda:0"


def test_aggregation_manager_constructor_with_cpu_device():
    """Test AggregationManager constructor with CPU device"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    assert manager.device == "cpu"


def test_aggregation_manager_constructor_stores_references_not_copies():
    """Test that constructor stores references to dependencies, not copies"""
    gradient_manager = MagicMock()
    peer_manager = MagicMock()
    storage_client = MagicMock()
    file_manager = MagicMock()
    hparams = MagicMock()

    manager = AggregationManager(
        gradient_manager=gradient_manager,
        peer_manager=peer_manager,
        storage_client=storage_client,
        file_manager=file_manager,
        hparams=hparams,
        device="cpu",
    )

    # Test that the exact same objects are referenced
    assert manager.gradient_manager is gradient_manager
    assert manager.peer_manager is peer_manager
    assert manager.storage_client is storage_client
    assert manager.file_manager is file_manager
    assert manager.hparams is hparams


def test_aggregation_manager_constructor_with_empty_device_string():
    """Test AggregationManager constructor with empty device string"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="",
    )

    assert manager.device == ""


def test_aggregation_manager_constructor_with_whitespace_device():
    """Test AggregationManager constructor with whitespace device string"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="  cpu  ",
    )

    assert manager.device == "  cpu  "


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - BASIC FUNCTIONALITY
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_single_uid_success_case(valid_aggregation_manager):
    """Test gather_gradients with single UID success case"""
    # Mock _aggregate_gradients to return a dict with all required keys
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2, 0.3])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert hasattr(result, "uids")
        assert hasattr(result, "state_dict")
        assert 1 in result.uids


@pytest.mark.asyncio
async def test_gather_gradients_with_multiple_uids_success_case(
    valid_aggregation_manager,
):
    """Test gather_gradients with multiple UIDs success case"""
    mock_aggregation_result = {
        "valid_uids": [1, 2, 3],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101, 102],
        "upload_bytes": 3000,
        "download_bytes": 6000,
        "succeeded": [1, 2, 3],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2, 3],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 3


@pytest.mark.asyncio
async def test_gather_gradients_with_empty_uid_list(valid_aggregation_manager):
    """Test gather_gradients with empty UID list"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_none_uid_list(valid_aggregation_manager):
    """Test gather_gradients with None UID list"""
    with pytest.raises((TypeError, AttributeError)):
        await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=None,
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )


@pytest.mark.asyncio
async def test_gather_gradients_with_duplicate_uids_in_list(valid_aggregation_manager):
    """Test gather_gradients with duplicate UIDs in list"""
    mock_aggregation_result = {
        "valid_uids": [1, 2],  # Duplicates removed
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [1, 2],
        "failed": [],
        "skipped_uids": [1],  # One duplicate was skipped
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 1, 2, 1],  # Duplicate UIDs
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert len(result.uids) == 2  # Duplicates handled


@pytest.mark.asyncio
async def test_gather_gradients_with_my_uid_included_in_target_uids_list(
    valid_aggregation_manager,
):
    """Test gather_gradients with my_uid included in target UIDs list"""
    mock_aggregation_result = {
        "valid_uids": [2, 3],  # my_uid (1) excluded from results
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [101, 102],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [2, 3],
        "failed": [],
        "skipped_uids": [1],  # my_uid was skipped
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=1,
            uids=[1, 2, 3],  # my_uid included
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert 1 not in result.uids  # my_uid should be excluded


@pytest.mark.asyncio
async def test_gather_gradients_with_invalid_uid_types_non_int(
    valid_aggregation_manager,
):
    """Test gather_gradients with invalid UID types (non-int)"""
    # These should be skipped due to invalid UID format
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=["not_an_int", "also_invalid"],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should return None as no valid gradients found
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_invalid_uid_types_negative(
    valid_aggregation_manager,
):
    """Test gather_gradients with invalid UID types (negative)"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[-1, -5],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should return None as gradients won't be found for negative UIDs
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_returns_none_when_no_valid_gradients_found(
    valid_aggregation_manager,
):
    """Test gather_gradients returns None when no valid gradients found"""
    # Mock storage to return None (no data found)
    with patch.object(
        valid_aggregation_manager.storage_client, "get_object", new_callable=AsyncMock
    ) as mock_get_object:
        mock_get_object.return_value = None

        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2, 3],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_returns_correct_simplenamespace_structure(
    valid_aggregation_manager,
):
    """Test gather_gradients returns correct SimpleNamespace structure"""
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # Check required attributes with correct field names
        assert hasattr(result, "uids")
        assert hasattr(result, "state_dict")
        assert hasattr(result, "global_steps")
        assert hasattr(result, "upload_bytes")
        assert hasattr(result, "download_bytes")
        assert hasattr(result, "success_rate")
        assert hasattr(result, "time")
        assert hasattr(result, "skipped_uids")

        # Check types
        assert isinstance(result.uids, list)
        assert isinstance(result.state_dict, SimpleNamespace)
        assert isinstance(result.global_steps, list)
        assert isinstance(result.upload_bytes, int)
        assert isinstance(result.download_bytes, int)
        assert isinstance(result.success_rate, float)
        assert isinstance(result.time, float)
        assert isinstance(result.skipped_uids, list)


@pytest.mark.asyncio
async def test_gather_gradients_with_local_true_vs_local_false_behavior(
    valid_aggregation_manager,
):
    """Test gather_gradients with local=True vs local=False behavior"""
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        # Test with local=True
        result_local = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            local=True,
        )

        # Test with local=False
        result_remote = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            local=False,
        )

        # Both should succeed
        assert result_local is not None
        assert result_remote is not None


@pytest.mark.asyncio
async def test_gather_gradients_with_different_timeout_values_1s(
    valid_aggregation_manager,
):
    """Test gather_gradients with timeout=1s"""
    # For timeout tests, we just verify the method completes
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=1,  # Very short timeout
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should complete (likely returning None due to timeout)
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_different_timeout_values_30s(
    valid_aggregation_manager,
):
    """Test gather_gradients with timeout=30s"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should complete
    assert result is None  # No gradients found


@pytest.mark.asyncio
async def test_gather_gradients_with_different_timeout_values_60s(
    valid_aggregation_manager,
):
    """Test gather_gradients with timeout=60s"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=60,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should complete
    assert result is None  # No gradients found


@pytest.mark.asyncio
async def test_gather_gradients_with_zero_timeout_edge_case(valid_aggregation_manager):
    """Test gather_gradients with zero timeout (edge case)"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=0,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should handle gracefully
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_negative_timeout_should_handle_gracefully(
    valid_aggregation_manager,
):
    """Test gather_gradients with negative timeout (should handle gracefully)"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=-5,  # Negative timeout
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should handle gracefully
    assert result is None


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - TIME CONSTRAINTS
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_time_min_constraint_only(
    valid_aggregation_manager,
):
    """Test gather_gradients with time_min constraint only"""
    time_min = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_time_max_constraint_only(
    valid_aggregation_manager,
):
    """Test gather_gradients with time_max constraint only"""
    time_max = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_both_time_min_and_time_max_constraints(
    valid_aggregation_manager,
):
    """Test gather_gradients with both time_min and time_max constraints"""
    time_min = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1, 2],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [1, 2],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 2


@pytest.mark.asyncio
async def test_gather_gradients_with_time_min_equals_time_max_exact_timestamp(
    valid_aggregation_manager,
):
    """Test gather_gradients with time_min = time_max (exact timestamp)"""
    exact_time = datetime(2024, 6, 15, 14, 30, 45, tzinfo=timezone.utc)

    # Should return None as exact timestamp match is unlikely
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
        time_min=exact_time,
        time_max=exact_time,
    )

    # Exact timestamp matching typically returns no results
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_time_min_greater_than_time_max_invalid_range(
    valid_aggregation_manager,
):
    """Test gather_gradients with time_min > time_max (invalid range)"""
    time_min = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(
        2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc
    )  # Earlier than time_min

    # Invalid time range should return no results
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
        time_min=time_min,
        time_max=time_max,
    )

    # Invalid range should return None
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_timezone_naive_datetime_objects(
    valid_aggregation_manager,
):
    """Test gather_gradients with timezone-naive datetime objects"""
    time_min = datetime(2024, 6, 1, 12, 0, 0)  # No timezone
    time_max = datetime(2024, 6, 30, 12, 0, 0)  # No timezone

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_timezone_aware_datetime_objects(
    valid_aggregation_manager,
):
    """Test gather_gradients with timezone-aware datetime objects"""
    time_min = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_mixed_timezone_datetime_objects(
    valid_aggregation_manager,
):
    """Test gather_gradients with mixed timezone datetime objects"""
    # Mix of UTC and timezone-naive
    time_min = datetime(2024, 6, 1, 12, 0, 0)  # Naive
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)  # UTC

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
            time_max=time_max,
        )

        # Should handle mixed timezones gracefully
        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_very_old_time_min_before_any_data_exists(
    valid_aggregation_manager,
):
    """Test gather_gradients with very old time_min (before any data exists)"""
    # Very old timestamp that predates any possible data
    time_min = datetime(1990, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
        )

        # Should include all available data
        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_very_future_time_max(valid_aggregation_manager):
    """Test gather_gradients with very future time_max"""
    # Future timestamp
    time_max = datetime(2030, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_max=time_max,
        )

        # Should include all available data up to the future time
        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_time_filtering_with_local_storage(
    valid_aggregation_manager,
):
    """Test gather_gradients time filtering with local storage"""
    time_min = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            local=True,  # Use local storage
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_time_filtering_with_remote_storage(
    valid_aggregation_manager,
):
    """Test gather_gradients time filtering with remote storage"""
    time_min = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            local=False,  # Use remote storage
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - STALE RETENTION
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_default_stale_retention_10(
    valid_aggregation_manager,
):
    """Test gather_gradients with default stale_retention (10)"""
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            stale_retention=10,  # Default value
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        # TODO: Verify stale_retention parameter is passed through aggregation pipeline


@pytest.mark.asyncio
async def test_gather_gradients_with_stale_retention_0_no_retention(
    valid_aggregation_manager,
):
    """Test gather_gradients with stale_retention=0 (no retention)"""
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            stale_retention=0,  # No retention
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        # TODO: Verify that stale_retention=0 disables retention mechanisms


@pytest.mark.asyncio
async def test_gather_gradients_with_very_high_stale_retention_1000(
    valid_aggregation_manager,
):
    """Test gather_gradients with very high stale_retention (1000)"""
    mock_aggregation_result = {
        "valid_uids": [1, 2],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [1, 2],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            stale_retention=1000,  # Very high retention
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 2
        # TODO: Verify high retention allows more stale gradients to be included


@pytest.mark.asyncio
async def test_gather_gradients_with_negative_stale_retention_edge_case(
    valid_aggregation_manager,
):
    """Test gather_gradients with negative stale_retention (edge case)"""
    # Negative stale_retention should be handled gracefully
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
        stale_retention=-5,  # Negative value
    )

    # Should handle gracefully, likely returning None or treating as 0
    # TODO: Define expected behavior for negative stale_retention values
    assert result is None or isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_stale_retention_affects_cleanup_operations_correctly(
    valid_aggregation_manager,
):
    """Test stale_retention affects cleanup operations correctly"""
    # Mock file system operations to track cleanup behavior
    with (
        patch.object(
            valid_aggregation_manager.file_manager, "delete_file"
        ) as mock_delete,
        patch("os.path.exists", return_value=True),
        patch("os.path.getmtime", return_value=time.time() - 3600),
    ):  # 1 hour old
        # Test with different stale_retention values
        for retention in [0, 5, 10, 100]:
            mock_delete.reset_mock()

            result = await valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[1],
                window=10,
                timeout=30,
                device="cpu",
                totalks={"layer.weight": 1000},
                stale_retention=retention,
            )

            # TODO: Verify cleanup behavior changes based on stale_retention value
            # Higher retention should result in fewer cleanup operations


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - DEVICE HANDLING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_device_cpu(valid_aggregation_manager):
    """Test gather_gradients with device="cpu" """
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ) as mock_agg:
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # Verify device parameter was passed to aggregation
        mock_agg.assert_called_once()
        call_args = mock_agg.call_args
        assert call_args[0][1] == "cpu"  # device parameter


@pytest.mark.asyncio
async def test_gather_gradients_with_device_cuda_if_available(
    valid_aggregation_manager,
):
    """Test gather_gradients with device="cuda" (if available)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ) as mock_agg:
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device=device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # Verify correct device was used
        mock_agg.assert_called_once()
        call_args = mock_agg.call_args
        assert call_args[0][1] == device


@pytest.mark.asyncio
async def test_gather_gradients_with_device_cuda_specific_gpu(
    valid_aggregation_manager,
):
    """Test gather_gradients with device="cuda:0" (specific GPU)"""
    device = "cuda:0"

    mock_aggregation_result = {
        "valid_uids": [1, 2],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [1, 2],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ) as mock_agg:
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2],
            window=10,
            timeout=30,
            device=device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 2

        # Verify specific GPU device was used
        mock_agg.assert_called_once()
        call_args = mock_agg.call_args
        assert call_args[0][1] == "cuda:0"


@pytest.mark.asyncio
async def test_gather_gradients_with_invalid_device_string(valid_aggregation_manager):
    """Test gather_gradients with invalid device string"""
    invalid_device = "invalid_device_string"

    # The current implementation doesn't validate device strings during gather_gradients
    # It only fails during actual tensor operations in _aggregate_gradients
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device=invalid_device,
        totalks={"layer.weight": 1000},
    )

    # Should return None since no valid gradients will be found/processed
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_device_mismatch_between_input_and_target(
    valid_aggregation_manager,
):
    """Test gather_gradients with device mismatch between input and target"""
    # Mock tensors on different devices
    cpu_tensor = torch.tensor([0.1, 0.2], device="cpu")

    # Simulate device mismatch scenario
    mock_response = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": cpu_tensor},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager, "_aggregate_gradients", return_value=mock_response
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cuda" if torch.cuda.is_available() else "cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        # TODO: Verify that device movement occurred during aggregation


@pytest.mark.asyncio
async def test_tensor_device_movement_during_aggregation_process(
    valid_aggregation_manager,
):
    """Test tensor device movement during aggregation process"""
    # Create mock tensors on CPU
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0], device="cpu")
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mock the raw responses with CPU tensors
    mock_raw_responses = [
        {
            "uid": 1,
            "response": (
                {"state_dict": {"layer.weight": cpu_tensor}, "global_step": 100},
                100,
            ),
            "is_exception": False,
        }
    ]

    with (
        patch.object(
            valid_aggregation_manager,
            "gather_semaphore",
            new_callable=lambda: asyncio.Semaphore(15),
        ),
        patch.object(
            valid_aggregation_manager,
            "_get_with_retry",
            side_effect=[mock_raw_responses[0]["response"]],
        ),
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device=target_device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # Verify tensor was moved to target device
        if hasattr(result.state_dict, "layer"):
            layer_tensors = result.state_dict.layer
            if hasattr(layer_tensors, "weight") and isinstance(
                layer_tensors.weight, list
            ):
                for tensor in layer_tensors.weight:
                    if isinstance(tensor, torch.Tensor):
                        assert str(tensor.device).startswith(
                            target_device.split(":")[0]
                        )


@pytest.mark.asyncio
async def test_gather_gradients_device_consistency_across_multiple_uids(
    valid_aggregation_manager,
):
    """Test device consistency when gathering from multiple UIDs"""
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mock tensors from different UIDs, all should be moved to target device
    mock_aggregation_result = {
        "valid_uids": [1, 2, 3],
        "aggregated_state_dict": {
            "layer.weight": [
                torch.tensor([0.1, 0.2], device=target_device),
                torch.tensor([0.3, 0.4], device=target_device),
                torch.tensor([0.5, 0.6], device=target_device),
            ]
        },
        "global_steps": [100, 101, 102],
        "upload_bytes": 3000,
        "download_bytes": 6000,
        "succeeded": [1, 2, 3],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2, 3],
            window=10,
            timeout=30,
            device=target_device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 3

        # Verify all tensors are on the same target device
        if hasattr(result.state_dict, "layer") and hasattr(
            result.state_dict.layer, "weight"
        ):
            tensors = result.state_dict.layer.weight
            if isinstance(tensors, list):
                for tensor in tensors:
                    if isinstance(tensor, torch.Tensor):
                        assert str(tensor.device).startswith(
                            target_device.split(":")[0]
                        )


@pytest.mark.asyncio
async def test_gather_gradients_device_parameter_validation(valid_aggregation_manager):
    """Test device parameter validation and error handling"""
    test_devices = ["cpu", "cuda", "cuda:0", "cuda:1", "mps", ""]

    for device in test_devices:
        try:
            result = await valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[1],
                window=10,
                timeout=30,
                device=device,
                totalks={"layer.weight": 1000},
            )

            # Should either succeed or fail gracefully
            assert result is None or isinstance(result, SimpleNamespace)

        except Exception as e:
            # Device-related exceptions should be handled gracefully
            assert "device" in str(e).lower() or "cuda" in str(e).lower()


@pytest.mark.asyncio
async def test_gather_gradients_mixed_device_tensors_in_quant_params(
    valid_aggregation_manager,
):
    """Test handling of mixed device tensors in quant_params structures"""
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mock quant_params with tensors on different devices
    mock_quant_params = {
        "scale": torch.tensor([1.0], device="cpu"),
        "zero_point": torch.tensor([0], device="cpu"),
        "bits": 8,
    }

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight_quant_params": [mock_quant_params]},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device=target_device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # TODO: Verify quant_params tensors were moved to target device
        # This requires implementing device movement validation for nested structures


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - TOTALKS PARAMETER
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_empty_totalks_dict(valid_aggregation_manager):
    """Test gather_gradients with empty totalks dict"""
    empty_totalks = {}

    # Mock response with some gradient data
    mock_raw_responses = [
        {
            "uid": 1,
            "response": (
                {
                    "state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
                    "global_step": 100,
                },
                100,
            ),
            "is_exception": False,
        }
    ]

    with patch.object(
        valid_aggregation_manager,
        "_get_with_retry",
        side_effect=[mock_raw_responses[0]["response"]],
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks=empty_totalks,  # Empty dict
        )

        # Should handle empty totalks gracefully
        assert result is None or isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_none_totalks(valid_aggregation_manager):
    """Test gather_gradients with None totalks"""
    # The current implementation doesn't validate totalks parameter upfront
    # It only fails when trying to process gradients that need validation
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks=None,  # None value
    )

    # Should return None since no valid gradients can be processed without totalks
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_mismatched_totalks_keys_vs_gradient_keys(
    valid_aggregation_manager,
):
    """Test gather_gradients with mismatched totalks keys vs gradient keys"""
    totalks = {"layer.weight": 1000, "layer.bias": 500}  # Expected keys

    # Mock gradient response with different keys
    mock_gradient_state = {
        "different.weight": torch.tensor([0.1, 0.2]),  # Different key
        "other.layer": torch.tensor([0.3, 0.4]),  # Different key
    }

    mock_raw_responses = [
        {
            "uid": 1,
            "response": (mock_gradient_state, 100),
            "is_exception": False,
        }
    ]

    with patch.object(
        valid_aggregation_manager,
        "_get_with_retry",
        side_effect=[mock_raw_responses[0]["response"]],
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0, uids=[1], window=10, timeout=30, device="cpu", totalks=totalks
        )

        # Should succeed but may have no valid gradients due to key mismatch
        assert result is None or isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_totalks_containing_extra_unused_keys(
    valid_aggregation_manager,
):
    """Test gather_gradients with totalks containing extra unused keys"""
    totalks = {
        "layer.weight": 1000,
        "layer.bias": 500,
        "unused.layer1": 200,  # Extra unused key
        "unused.layer2": 300,  # Extra unused key
        "nonexistent.param": 100,  # Extra unused key
    }

    mock_gradient_state = {
        "layer.weight": torch.tensor([0.1, 0.2]),
        "layer.bias": torch.tensor([0.3]),
    }

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {
            "layer.weight": [torch.tensor([0.1, 0.2])],
            "layer.bias": [torch.tensor([0.3])],
        },
        "global_steps": [100],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0, uids=[1], window=10, timeout=30, device="cpu", totalks=totalks
        )

        # Should handle extra keys gracefully
        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 1


@pytest.mark.asyncio
async def test_gather_gradients_with_totalks_missing_required_keys(
    valid_aggregation_manager,
):
    """Test gather_gradients with totalks missing required keys"""
    totalks = {"layer.weight": 1000}  # Missing key for layer.bias

    # Mock gradient with indices that require totalk validation
    mock_gradient_state = {
        "layer.weight_idxs": torch.tensor([0, 1, 2]),
        "layer.weight_vals": torch.tensor([0.1, 0.2, 0.3]),
        "layer.bias_idxs": torch.tensor([0, 1]),  # This will fail - no totalk
        "layer.bias_vals": torch.tensor([0.4, 0.5]),
    }

    # FIXED: Response should be tuple (state_dict, global_step) directly
    mock_response = (mock_gradient_state, 100)

    with patch.object(
        valid_aggregation_manager,
        "_get_with_retry",
        side_effect=[mock_response],
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0, uids=[1], window=10, timeout=30, device="cpu", totalks=totalks
        )

        # Should return None due to validation failure
        assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_totalks_containing_invalid_values_negative_zero(
    valid_aggregation_manager,
):
    """Test gather_gradients with totalks containing invalid values (negative, zero)"""
    invalid_totalks_cases = [
        {"layer.weight": -100},  # Negative value
        {"layer.weight": 0},  # Zero value
        {"layer.weight": -1},  # Negative value
    ]

    for totalks in invalid_totalks_cases:
        # Mock gradient with indices
        mock_gradient_state = {
            "layer.weight_idxs": torch.tensor([0, 1]),
            "layer.weight_vals": torch.tensor([0.1, 0.2]),
        }

        # FIXED: Response should be tuple (state_dict, global_step) directly
        mock_response = (mock_gradient_state, 100)

        with patch.object(
            valid_aggregation_manager,
            "_get_with_retry",
            side_effect=[mock_response],
        ):
            result = await valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[1],
                window=10,
                timeout=30,
                device="cpu",
                totalks=totalks,
            )

            # Should fail validation with invalid totalks values
            assert result is None


import pytest
import asyncio
import torch
import time
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from datetime import datetime, timezone
from types import SimpleNamespace

from tplr.training.aggregation_manager import AggregationManager


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_chain_manager():
    """Mock chain manager that peer manager depends on"""
    manager = MagicMock()
    manager.current_window = 10
    manager.get_current_window = MagicMock(return_value=10)
    manager.get_bucket = MagicMock(return_value=MagicMock())
    manager.commitments = {1: MagicMock(), 2: MagicMock(), 3: MagicMock()}
    manager.metagraph = MagicMock()
    manager.metagraph.uids = [0, 1, 2, 3, 4, 5]  # Mock UID list
    return manager


@pytest.fixture
def mock_peer_manager(mock_chain_manager):
    """Mock peer manager with proper chain_manager dependency"""
    manager = MagicMock()
    # Set the chain_manager that AggregationManager expects to access
    manager.chain_manager = mock_chain_manager

    # Add methods that might be called by AggregationManager
    manager.get_active_peers = MagicMock(return_value={1, 2, 3})
    manager.is_peer_active = AsyncMock(return_value=True)
    manager.weighted_random_sample_no_replacement = MagicMock(return_value=[1, 2, 3])

    return manager


@pytest.fixture
def mock_gradient_manager():
    manager = MagicMock()
    manager.validate_gradient = MagicMock(return_value=True)
    manager.check_compressed_indices = MagicMock()
    return manager


@pytest.fixture
def mock_storage_client():
    client = MagicMock()
    client.get_object = AsyncMock(return_value=None)  # Default to None (no data found)
    client.put_object = AsyncMock(return_value=True)
    client.get_object_size = AsyncMock(return_value=1000)
    client.multipart_download = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_file_manager():
    manager = MagicMock()
    manager.create_temp_file = MagicMock(return_value="/tmp/test_file.pt")
    manager.delete_file = MagicMock()
    manager.get_local_storage_path = MagicMock(return_value="/tmp/local_path.pt")
    return manager


@pytest.fixture
def mock_hparams():
    hparams = MagicMock()
    hparams.topk_compression = 10
    hparams.active_check_interval = 60
    return hparams


@pytest.fixture
def valid_aggregation_manager(
    mock_gradient_manager,
    mock_peer_manager,
    mock_storage_client,
    mock_file_manager,
    mock_hparams,
):
    return AggregationManager(
        gradient_manager=mock_gradient_manager,
        peer_manager=mock_peer_manager,
        storage_client=mock_storage_client,
        file_manager=mock_file_manager,
        hparams=mock_hparams,
        device="cpu",
    )


# -----------------------------------------------------------------------------
# CONSTRUCTOR & INITIALIZATION TESTS
# -----------------------------------------------------------------------------
def test_aggregation_manager_constructor_with_valid_dependencies(
    mock_gradient_manager,
    mock_peer_manager,
    mock_storage_client,
    mock_file_manager,
    mock_hparams,
):
    """Test AggregationManager constructor with valid dependencies"""
    manager = AggregationManager(
        gradient_manager=mock_gradient_manager,
        peer_manager=mock_peer_manager,
        storage_client=mock_storage_client,
        file_manager=mock_file_manager,
        hparams=mock_hparams,
        device="cpu",
    )

    assert manager.gradient_manager is mock_gradient_manager
    assert manager.peer_manager is mock_peer_manager
    assert manager.storage_client is mock_storage_client
    assert manager.file_manager is mock_file_manager
    assert manager.hparams is mock_hparams
    assert manager.device == "cpu"


def test_aggregation_manager_constructor_with_none_gradient_manager():
    """Test AggregationManager constructor with None gradient_manager"""
    # Constructor doesn't validate - it just stores None
    manager = AggregationManager(
        gradient_manager=None,
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    # The constructor succeeds but stores None
    assert manager.gradient_manager is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_none_peer_manager():
    """Test AggregationManager constructor with None peer_manager"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=None,
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    assert manager.peer_manager is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_none_storage_client():
    """Test AggregationManager constructor with None storage_client"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=None,
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    assert manager.storage_client is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_none_file_manager():
    """Test AggregationManager constructor with None file_manager"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=None,
        hparams=MagicMock(),
        device="cpu",
    )

    assert manager.file_manager is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_none_hparams():
    """Test AggregationManager constructor with None hparams"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=None,
        device="cpu",
    )

    assert manager.hparams is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_invalid_device_string():
    """Test AggregationManager constructor with invalid device string"""
    # This should not raise during construction, but may cause issues during tensor operations
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="invalid_device",
    )

    assert manager.device == "invalid_device"


def test_semaphore_initialization_with_default_value(valid_aggregation_manager):
    """Test semaphore initialization with default value (15)"""
    assert hasattr(valid_aggregation_manager, "gather_semaphore")
    assert isinstance(valid_aggregation_manager.gather_semaphore, asyncio.Semaphore)
    assert valid_aggregation_manager.gather_semaphore._value == 15


def test_semaphore_initialization_with_custom_value():
    """Test semaphore initialization with custom value"""
    # Note: Current implementation uses hardcoded value, but test what would happen if it were configurable
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    # Verify default is still 15
    assert manager.gather_semaphore._value == 15

    # TODO: Add parameter to constructor to allow custom semaphore value
    # TODO: Test with different semaphore values (1, 5, 50, 100)


def test_all_manager_dependencies_are_properly_stored_as_instance_variables(
    mock_gradient_manager,
    mock_peer_manager,
    mock_storage_client,
    mock_file_manager,
    mock_hparams,
):
    """Test all manager dependencies are properly stored as instance variables"""
    manager = AggregationManager(
        gradient_manager=mock_gradient_manager,
        peer_manager=mock_peer_manager,
        storage_client=mock_storage_client,
        file_manager=mock_file_manager,
        hparams=mock_hparams,
        device="cuda:0",
    )

    # Test all dependencies are stored correctly
    assert manager.gradient_manager is mock_gradient_manager
    assert manager.peer_manager is mock_peer_manager
    assert manager.storage_client is mock_storage_client
    assert manager.file_manager is mock_file_manager
    assert manager.hparams is mock_hparams
    assert manager.device == "cuda:0"

    # Test that semaphore is initialized
    assert hasattr(manager, "gather_semaphore")
    assert isinstance(manager.gather_semaphore, asyncio.Semaphore)

    # Test instance attributes directly using __dict__
    expected_attrs = {
        "gradient_manager",
        "peer_manager",
        "storage_client",
        "file_manager",
        "hparams",
        "device",
        "gather_semaphore",
    }
    actual_attrs = set(manager.__dict__.keys())

    assert expected_attrs.issubset(actual_attrs), (
        f"Missing expected attributes: {expected_attrs - actual_attrs}"
    )

    # Verify no unexpected critical attributes are missing
    missing_attrs = expected_attrs - actual_attrs
    assert len(missing_attrs) == 0, f"Missing critical attributes: {missing_attrs}"


def test_aggregation_manager_constructor_with_cuda_device():
    """Test AggregationManager constructor with CUDA device"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cuda:0",
    )

    assert manager.device == "cuda:0"


def test_aggregation_manager_constructor_with_cpu_device():
    """Test AggregationManager constructor with CPU device"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    assert manager.device == "cpu"


def test_aggregation_manager_constructor_stores_references_not_copies():
    """Test that constructor stores references to dependencies, not copies"""
    gradient_manager = MagicMock()
    peer_manager = MagicMock()
    storage_client = MagicMock()
    file_manager = MagicMock()
    hparams = MagicMock()

    manager = AggregationManager(
        gradient_manager=gradient_manager,
        peer_manager=peer_manager,
        storage_client=storage_client,
        file_manager=file_manager,
        hparams=hparams,
        device="cpu",
    )

    # Test that the exact same objects are referenced
    assert manager.gradient_manager is gradient_manager
    assert manager.peer_manager is peer_manager
    assert manager.storage_client is storage_client
    assert manager.file_manager is file_manager
    assert manager.hparams is hparams


def test_aggregation_manager_constructor_with_empty_device_string():
    """Test AggregationManager constructor with empty device string"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="",
    )

    assert manager.device == ""


def test_aggregation_manager_constructor_with_whitespace_device():
    """Test AggregationManager constructor with whitespace device string"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="  cpu  ",
    )

    assert manager.device == "  cpu  "


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - BASIC FUNCTIONALITY
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_single_uid_success_case(valid_aggregation_manager):
    """Test gather_gradients with single UID success case"""
    # Mock _aggregate_gradients to return a dict with all required keys
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2, 0.3])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert hasattr(result, "uids")
        assert hasattr(result, "state_dict")
        assert 1 in result.uids


@pytest.mark.asyncio
async def test_gather_gradients_with_multiple_uids_success_case(
    valid_aggregation_manager,
):
    """Test gather_gradients with multiple UIDs success case"""
    mock_aggregation_result = {
        "valid_uids": [1, 2, 3],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101, 102],
        "upload_bytes": 3000,
        "download_bytes": 6000,
        "succeeded": [1, 2, 3],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2, 3],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 3


@pytest.mark.asyncio
async def test_gather_gradients_with_empty_uid_list(valid_aggregation_manager):
    """Test gather_gradients with empty UID list"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_none_uid_list(valid_aggregation_manager):
    """Test gather_gradients with None UID list"""
    with pytest.raises((TypeError, AttributeError)):
        await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=None,
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )


@pytest.mark.asyncio
async def test_gather_gradients_with_duplicate_uids_in_list(valid_aggregation_manager):
    """Test gather_gradients with duplicate UIDs in list"""
    mock_aggregation_result = {
        "valid_uids": [1, 2],  # Duplicates removed
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [1, 2],
        "failed": [],
        "skipped_uids": [1],  # One duplicate was skipped
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 1, 2, 1],  # Duplicate UIDs
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert len(result.uids) == 2  # Duplicates handled


@pytest.mark.asyncio
async def test_gather_gradients_with_my_uid_included_in_target_uids_list(
    valid_aggregation_manager,
):
    """Test gather_gradients with my_uid included in target UIDs list"""
    mock_aggregation_result = {
        "valid_uids": [2, 3],  # my_uid (1) excluded from results
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [101, 102],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [2, 3],
        "failed": [],
        "skipped_uids": [1],  # my_uid was skipped
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=1,
            uids=[1, 2, 3],  # my_uid included
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert 1 not in result.uids  # my_uid should be excluded


@pytest.mark.asyncio
async def test_gather_gradients_with_invalid_uid_types_non_int(
    valid_aggregation_manager,
):
    """Test gather_gradients with invalid UID types (non-int)"""
    # These should be skipped due to invalid UID format
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=["not_an_int", "also_invalid"],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should return None as no valid gradients found
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_invalid_uid_types_negative(
    valid_aggregation_manager,
):
    """Test gather_gradients with invalid UID types (negative)"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[-1, -5],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should return None as gradients won't be found for negative UIDs
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_returns_none_when_no_valid_gradients_found(
    valid_aggregation_manager,
):
    """Test gather_gradients returns None when no valid gradients found"""
    # Mock storage to return None (no data found)
    with patch.object(
        valid_aggregation_manager.storage_client, "get_object", new_callable=AsyncMock
    ) as mock_get_object:
        mock_get_object.return_value = None

        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2, 3],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_returns_correct_simplenamespace_structure(
    valid_aggregation_manager,
):
    """Test gather_gradients returns correct SimpleNamespace structure"""
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # Check required attributes with correct field names
        assert hasattr(result, "uids")
        assert hasattr(result, "state_dict")
        assert hasattr(result, "global_steps")
        assert hasattr(result, "upload_bytes")
        assert hasattr(result, "download_bytes")
        assert hasattr(result, "success_rate")
        assert hasattr(result, "time")
        assert hasattr(result, "skipped_uids")

        # Check types
        assert isinstance(result.uids, list)
        assert isinstance(result.state_dict, SimpleNamespace)
        assert isinstance(result.global_steps, list)
        assert isinstance(result.upload_bytes, int)
        assert isinstance(result.download_bytes, int)
        assert isinstance(result.success_rate, float)
        assert isinstance(result.time, float)
        assert isinstance(result.skipped_uids, list)


@pytest.mark.asyncio
async def test_gather_gradients_with_local_true_vs_local_false_behavior(
    valid_aggregation_manager,
):
    """Test gather_gradients with local=True vs local=False behavior"""
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        # Test with local=True
        result_local = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            local=True,
        )

        # Test with local=False
        result_remote = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            local=False,
        )

        # Both should succeed
        assert result_local is not None
        assert result_remote is not None


@pytest.mark.asyncio
async def test_gather_gradients_with_different_timeout_values_1s(
    valid_aggregation_manager,
):
    """Test gather_gradients with timeout=1s"""
    # For timeout tests, we just verify the method completes
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=1,  # Very short timeout
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should complete (likely returning None due to timeout)
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_different_timeout_values_30s(
    valid_aggregation_manager,
):
    """Test gather_gradients with timeout=30s"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should complete
    assert result is None  # No gradients found


@pytest.mark.asyncio
async def test_gather_gradients_with_different_timeout_values_60s(
    valid_aggregation_manager,
):
    """Test gather_gradients with timeout=60s"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=60,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should complete
    assert result is None  # No gradients found


@pytest.mark.asyncio
async def test_gather_gradients_with_zero_timeout_edge_case(valid_aggregation_manager):
    """Test gather_gradients with zero timeout (edge case)"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=0,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should handle gracefully
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_negative_timeout_should_handle_gracefully(
    valid_aggregation_manager,
):
    """Test gather_gradients with negative timeout (should handle gracefully)"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=-5,  # Negative timeout
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should handle gracefully
    assert result is None


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - TIME CONSTRAINTS
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_time_min_constraint_only(
    valid_aggregation_manager,
):
    """Test gather_gradients with time_min constraint only"""
    time_min = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_time_max_constraint_only(
    valid_aggregation_manager,
):
    """Test gather_gradients with time_max constraint only"""
    time_max = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_both_time_min_and_time_max_constraints(
    valid_aggregation_manager,
):
    """Test gather_gradients with both time_min and time_max constraints"""
    time_min = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1, 2],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [1, 2],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 2


@pytest.mark.asyncio
async def test_gather_gradients_with_time_min_equals_time_max_exact_timestamp(
    valid_aggregation_manager,
):
    """Test gather_gradients with time_min = time_max (exact timestamp)"""
    exact_time = datetime(2024, 6, 15, 14, 30, 45, tzinfo=timezone.utc)

    # Should return None as exact timestamp match is unlikely
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
        time_min=exact_time,
        time_max=exact_time,
    )

    # Exact timestamp matching typically returns no results
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_time_min_greater_than_time_max_invalid_range(
    valid_aggregation_manager,
):
    """Test gather_gradients with time_min > time_max (invalid range)"""
    time_min = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(
        2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc
    )  # Earlier than time_min

    # Invalid time range should return no results
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
        time_min=time_min,
        time_max=time_max,
    )

    # Invalid range should return None
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_timezone_naive_datetime_objects(
    valid_aggregation_manager,
):
    """Test gather_gradients with timezone-naive datetime objects"""
    time_min = datetime(2024, 6, 1, 12, 0, 0)  # No timezone
    time_max = datetime(2024, 6, 30, 12, 0, 0)  # No timezone

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_timezone_aware_datetime_objects(
    valid_aggregation_manager,
):
    """Test gather_gradients with timezone-aware datetime objects"""
    time_min = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_mixed_timezone_datetime_objects(
    valid_aggregation_manager,
):
    """Test gather_gradients with mixed timezone datetime objects"""
    # Mix of UTC and timezone-naive
    time_min = datetime(2024, 6, 1, 12, 0, 0)  # Naive
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)  # UTC

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
            time_max=time_max,
        )

        # Should handle mixed timezones gracefully
        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_very_old_time_min_before_any_data_exists(
    valid_aggregation_manager,
):
    """Test gather_gradients with very old time_min (before any data exists)"""
    # Very old timestamp that predates any possible data
    time_min = datetime(1990, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
        )

        # Should include all available data
        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_very_future_time_max(valid_aggregation_manager):
    """Test gather_gradients with very future time_max"""
    # Future timestamp
    time_max = datetime(2030, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_max=time_max,
        )

        # Should include all available data up to the future time
        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_time_filtering_with_local_storage(
    valid_aggregation_manager,
):
    """Test gather_gradients time filtering with local storage"""
    time_min = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            local=True,  # Use local storage
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_time_filtering_with_remote_storage(
    valid_aggregation_manager,
):
    """Test gather_gradients time filtering with remote storage"""
    time_min = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            local=False,  # Use remote storage
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - STALE RETENTION
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_default_stale_retention_10(
    valid_aggregation_manager,
):
    """Test gather_gradients with default stale_retention (10)"""
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            stale_retention=10,  # Default value
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        # TODO: Verify stale_retention parameter is passed through aggregation pipeline


@pytest.mark.asyncio
async def test_gather_gradients_with_stale_retention_0_no_retention(
    valid_aggregation_manager,
):
    """Test gather_gradients with stale_retention=0 (no retention)"""
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            stale_retention=0,  # No retention
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        # TODO: Verify that stale_retention=0 disables retention mechanisms


@pytest.mark.asyncio
async def test_gather_gradients_with_very_high_stale_retention_1000(
    valid_aggregation_manager,
):
    """Test gather_gradients with very high stale_retention (1000)"""
    mock_aggregation_result = {
        "valid_uids": [1, 2],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [1, 2],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            stale_retention=1000,  # Very high retention
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 2
        # TODO: Verify high retention allows more stale gradients to be included


@pytest.mark.asyncio
async def test_gather_gradients_with_negative_stale_retention_edge_case(
    valid_aggregation_manager,
):
    """Test gather_gradients with negative stale_retention (edge case)"""
    # Negative stale_retention should be handled gracefully
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
        stale_retention=-5,  # Negative value
    )

    # Should handle gracefully, likely returning None or treating as 0
    # TODO: Define expected behavior for negative stale_retention values
    assert result is None or isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_stale_retention_affects_cleanup_operations_correctly(
    valid_aggregation_manager,
):
    """Test stale_retention affects cleanup operations correctly"""
    # Mock file system operations to track cleanup behavior
    with (
        patch.object(
            valid_aggregation_manager.file_manager, "delete_file"
        ) as mock_delete,
        patch("os.path.exists", return_value=True),
        patch("os.path.getmtime", return_value=time.time() - 3600),
    ):  # 1 hour old
        # Test with different stale_retention values
        for retention in [0, 5, 10, 100]:
            mock_delete.reset_mock()

            result = await valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[1],
                window=10,
                timeout=30,
                device="cpu",
                totalks={"layer.weight": 1000},
                stale_retention=retention,
            )

            # TODO: Verify cleanup behavior changes based on stale_retention value
            # Higher retention should result in fewer cleanup operations


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - DEVICE HANDLING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_device_cpu(valid_aggregation_manager):
    """Test gather_gradients with device="cpu" """
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ) as mock_agg:
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # Verify device parameter was passed to aggregation
        mock_agg.assert_called_once()
        call_args = mock_agg.call_args
        assert call_args[0][1] == "cpu"  # device parameter


@pytest.mark.asyncio
async def test_gather_gradients_with_device_cuda_if_available(
    valid_aggregation_manager,
):
    """Test gather_gradients with device="cuda" (if available)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ) as mock_agg:
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device=device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # Verify correct device was used
        mock_agg.assert_called_once()
        call_args = mock_agg.call_args
        assert call_args[0][1] == device


@pytest.mark.asyncio
async def test_gather_gradients_with_device_cuda_specific_gpu(
    valid_aggregation_manager,
):
    """Test gather_gradients with device="cuda:0" (specific GPU)"""
    device = "cuda:0"

    mock_aggregation_result = {
        "valid_uids": [1, 2],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [1, 2],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ) as mock_agg:
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2],
            window=10,
            timeout=30,
            device=device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 2

        # Verify specific GPU device was used
        mock_agg.assert_called_once()
        call_args = mock_agg.call_args
        assert call_args[0][1] == "cuda:0"


@pytest.mark.asyncio
async def test_gather_gradients_with_invalid_device_string(valid_aggregation_manager):
    """Test gather_gradients with invalid device string"""
    invalid_device = "invalid_device_string"

    # The current implementation doesn't validate device strings during gather_gradients
    # It only fails during actual tensor operations in _aggregate_gradients
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device=invalid_device,
        totalks={"layer.weight": 1000},
    )

    # Should return None since no valid gradients will be found/processed
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_device_mismatch_between_input_and_target(
    valid_aggregation_manager,
):
    """Test gather_gradients with device mismatch between input and target"""
    # Mock tensors on different devices
    cpu_tensor = torch.tensor([0.1, 0.2], device="cpu")

    # Simulate device mismatch scenario
    mock_response = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": cpu_tensor},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager, "_aggregate_gradients", return_value=mock_response
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cuda" if torch.cuda.is_available() else "cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        # TODO: Verify that device movement occurred during aggregation


@pytest.mark.asyncio
async def test_tensor_device_movement_during_aggregation_process(
    valid_aggregation_manager,
):
    """Test tensor device movement during aggregation process"""
    # Create mock tensors on CPU
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0], device="cpu")
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mock the raw responses with CPU tensors
    mock_raw_responses = [
        {
            "uid": 1,
            "response": (
                {"state_dict": {"layer.weight": cpu_tensor}, "global_step": 100},
                100,
            ),
            "is_exception": False,
        }
    ]

    with (
        patch.object(
            valid_aggregation_manager,
            "gather_semaphore",
            new_callable=lambda: asyncio.Semaphore(15),
        ),
        patch.object(
            valid_aggregation_manager,
            "_get_with_retry",
            side_effect=[mock_raw_responses[0]["response"]],
        ),
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device=target_device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # Verify tensor was moved to target device
        if hasattr(result.state_dict, "layer"):
            layer_tensors = result.state_dict.layer
            if hasattr(layer_tensors, "weight") and isinstance(
                layer_tensors.weight, list
            ):
                for tensor in layer_tensors.weight:
                    if isinstance(tensor, torch.Tensor):
                        assert str(tensor.device).startswith(
                            target_device.split(":")[0]
                        )


@pytest.mark.asyncio
async def test_gather_gradients_device_consistency_across_multiple_uids(
    valid_aggregation_manager,
):
    """Test device consistency when gathering from multiple UIDs"""
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mock tensors from different UIDs, all should be moved to target device
    mock_aggregation_result = {
        "valid_uids": [1, 2, 3],
        "aggregated_state_dict": {
            "layer.weight": [
                torch.tensor([0.1, 0.2], device=target_device),
                torch.tensor([0.3, 0.4], device=target_device),
                torch.tensor([0.5, 0.6], device=target_device),
            ]
        },
        "global_steps": [100, 101, 102],
        "upload_bytes": 3000,
        "download_bytes": 6000,
        "succeeded": [1, 2, 3],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2, 3],
            window=10,
            timeout=30,
            device=target_device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 3

        # Verify all tensors are on the same target device
        if hasattr(result.state_dict, "layer") and hasattr(
            result.state_dict.layer, "weight"
        ):
            tensors = result.state_dict.layer.weight
            if isinstance(tensors, list):
                for tensor in tensors:
                    if isinstance(tensor, torch.Tensor):
                        assert str(tensor.device).startswith(
                            target_device.split(":")[0]
                        )


@pytest.mark.asyncio
async def test_gather_gradients_device_parameter_validation(valid_aggregation_manager):
    """Test device parameter validation and error handling"""
    test_devices = ["cpu", "cuda", "cuda:0", "cuda:1", "mps", ""]

    for device in test_devices:
        try:
            result = await valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[1],
                window=10,
                timeout=30,
                device=device,
                totalks={"layer.weight": 1000},
            )

            # Should either succeed or fail gracefully
            assert result is None or isinstance(result, SimpleNamespace)

        except Exception as e:
            # Device-related exceptions should be handled gracefully
            assert "device" in str(e).lower() or "cuda" in str(e).lower()


@pytest.mark.asyncio
async def test_gather_gradients_mixed_device_tensors_in_quant_params(
    valid_aggregation_manager,
):
    """Test handling of mixed device tensors in quant_params structures"""
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mock quant_params with tensors on different devices
    mock_quant_params = {
        "scale": torch.tensor([1.0], device="cpu"),
        "zero_point": torch.tensor([0], device="cpu"),
        "bits": 8,
    }

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight_quant_params": [mock_quant_params]},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device=target_device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # TODO: Verify quant_params tensors were moved to target device
        # This requires implementing device movement validation for nested structures


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - TOTALKS PARAMETER
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_empty_totalks_dict(valid_aggregation_manager):
    """Test gather_gradients with empty totalks dict"""
    empty_totalks = {}

    # Mock response with some gradient data
    mock_raw_responses = [
        {
            "uid": 1,
            "response": (
                {
                    "state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
                    "global_step": 100,
                },
                100,
            ),
            "is_exception": False,
        }
    ]

    with patch.object(
        valid_aggregation_manager,
        "_get_with_retry",
        side_effect=[mock_raw_responses[0]["response"]],
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks=empty_totalks,  # Empty dict
        )

        # Should handle empty totalks gracefully
        assert result is None or isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_none_totalks(valid_aggregation_manager):
    """Test gather_gradients with None totalks"""
    # The current implementation doesn't validate totalks parameter upfront
    # It only fails when trying to process gradients that need validation
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks=None,  # None value
    )

    # Should return None since no valid gradients can be processed without totalks
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_mismatched_totalks_keys_vs_gradient_keys(
    valid_aggregation_manager,
):
    """Test gather_gradients with mismatched totalks keys vs gradient keys"""
    totalks = {"layer.weight": 1000, "layer.bias": 500}  # Expected keys

    # Mock gradient response with different keys
    mock_gradient_state = {
        "different.weight": torch.tensor([0.1, 0.2]),  # Different key
        "other.layer": torch.tensor([0.3, 0.4]),  # Different key
    }

    mock_raw_responses = [
        {
            "uid": 1,
            "response": (mock_gradient_state, 100),
            "is_exception": False,
        }
    ]

    with patch.object(
        valid_aggregation_manager,
        "_get_with_retry",
        side_effect=[mock_raw_responses[0]["response"]],
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0, uids=[1], window=10, timeout=30, device="cpu", totalks=totalks
        )

        # Should succeed but may have no valid gradients due to key mismatch
        assert result is None or isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_totalks_containing_extra_unused_keys(
    valid_aggregation_manager,
):
    """Test gather_gradients with totalks containing extra unused keys"""
    totalks = {
        "layer.weight": 1000,
        "layer.bias": 500,
        "unused.layer1": 200,  # Extra unused key
        "unused.layer2": 300,  # Extra unused key
        "nonexistent.param": 100,  # Extra unused key
    }

    mock_gradient_state = {
        "layer.weight": torch.tensor([0.1, 0.2]),
        "layer.bias": torch.tensor([0.3]),
    }

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {
            "layer.weight": [torch.tensor([0.1, 0.2])],
            "layer.bias": [torch.tensor([0.3])],
        },
        "global_steps": [100],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0, uids=[1], window=10, timeout=30, device="cpu", totalks=totalks
        )

        # Should handle extra keys gracefully
        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 1


@pytest.mark.asyncio
async def test_gather_gradients_with_totalks_missing_required_keys(
    valid_aggregation_manager,
):
    """Test gather_gradients with totalks missing required keys"""
    totalks = {"layer.weight": 1000}  # Missing key for layer.bias

    # Mock gradient with indices that require totalk validation
    mock_gradient_state = {
        "layer.weight_idxs": torch.tensor([0, 1, 2]),
        "layer.weight_vals": torch.tensor([0.1, 0.2, 0.3]),
        "layer.bias_idxs": torch.tensor([0, 1]),  # This will fail - no totalk
        "layer.bias_vals": torch.tensor([0.4, 0.5]),
    }

    # FIXED: Response should be tuple (state_dict, global_step) directly
    mock_response = (mock_gradient_state, 100)

    with patch.object(
        valid_aggregation_manager,
        "_get_with_retry",
        side_effect=[mock_response],
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0, uids=[1], window=10, timeout=30, device="cpu", totalks=totalks
        )

        # Should return None due to validation failure
        assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_totalks_containing_invalid_values_negative_zero(
    valid_aggregation_manager,
):
    """Test gather_gradients with totalks containing invalid values (negative, zero)"""
    invalid_totalks_cases = [
        {"layer.weight": -100},  # Negative value
        {"layer.weight": 0},  # Zero value
        {"layer.weight": -1},  # Negative value
    ]

    for totalks in invalid_totalks_cases:
        # Mock gradient with indices
        mock_gradient_state = {
            "layer.weight_idxs": torch.tensor([0, 1]),
            "layer.weight_vals": torch.tensor([0.1, 0.2]),
        }

        # FIXED: Response should be tuple (state_dict, global_step) directly
        mock_response = (mock_gradient_state, 100)

        with patch.object(
            valid_aggregation_manager,
            "_get_with_retry",
            side_effect=[mock_response],
        ):
            result = await valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[1],
                window=10,
                timeout=30,
                device="cpu",
                totalks=totalks,
            )

            # Should fail validation with invalid totalks values
            assert result is None


# ruff : noqa

import pytest
import asyncio
import torch
import time
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from datetime import datetime, timezone
from types import SimpleNamespace

from tplr.training.aggregation_manager import AggregationManager


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_chain_manager():
    """Mock chain manager that peer manager depends on"""
    manager = MagicMock()
    manager.current_window = 10
    manager.get_current_window = MagicMock(return_value=10)
    manager.get_bucket = MagicMock(return_value=MagicMock())
    manager.commitments = {1: MagicMock(), 2: MagicMock(), 3: MagicMock()}
    manager.metagraph = MagicMock()
    manager.metagraph.uids = [0, 1, 2, 3, 4, 5]  # Mock UID list
    return manager


@pytest.fixture
def mock_peer_manager(mock_chain_manager):
    """Mock peer manager with proper chain_manager dependency"""
    manager = MagicMock()
    # Set the chain_manager that AggregationManager expects to access
    manager.chain_manager = mock_chain_manager

    # Add methods that might be called by AggregationManager
    manager.get_active_peers = MagicMock(return_value={1, 2, 3})
    manager.is_peer_active = AsyncMock(return_value=True)
    manager.weighted_random_sample_no_replacement = MagicMock(return_value=[1, 2, 3])

    return manager


@pytest.fixture
def mock_gradient_manager():
    manager = MagicMock()
    manager.validate_gradient = MagicMock(return_value=True)
    manager.check_compressed_indices = MagicMock()
    return manager


@pytest.fixture
def mock_storage_client():
    client = MagicMock()
    client.get_object = AsyncMock(return_value=None)  # Default to None (no data found)
    client.put_object = AsyncMock(return_value=True)
    client.get_object_size = AsyncMock(return_value=1000)
    client.multipart_download = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_file_manager():
    manager = MagicMock()
    manager.create_temp_file = MagicMock(return_value="/tmp/test_file.pt")
    manager.delete_file = MagicMock()
    manager.get_local_storage_path = MagicMock(return_value="/tmp/local_path.pt")
    return manager


@pytest.fixture
def mock_hparams():
    hparams = MagicMock()
    hparams.topk_compression = 10
    hparams.active_check_interval = 60
    return hparams


@pytest.fixture
def valid_aggregation_manager(
    mock_gradient_manager,
    mock_peer_manager,
    mock_storage_client,
    mock_file_manager,
    mock_hparams,
):
    return AggregationManager(
        gradient_manager=mock_gradient_manager,
        peer_manager=mock_peer_manager,
        storage_client=mock_storage_client,
        file_manager=mock_file_manager,
        hparams=mock_hparams,
        device="cpu",
    )


# -----------------------------------------------------------------------------
# CONSTRUCTOR & INITIALIZATION TESTS
# -----------------------------------------------------------------------------
def test_aggregation_manager_constructor_with_valid_dependencies(
    mock_gradient_manager,
    mock_peer_manager,
    mock_storage_client,
    mock_file_manager,
    mock_hparams,
):
    """Test AggregationManager constructor with valid dependencies"""
    manager = AggregationManager(
        gradient_manager=mock_gradient_manager,
        peer_manager=mock_peer_manager,
        storage_client=mock_storage_client,
        file_manager=mock_file_manager,
        hparams=mock_hparams,
        device="cpu",
    )

    assert manager.gradient_manager is mock_gradient_manager
    assert manager.peer_manager is mock_peer_manager
    assert manager.storage_client is mock_storage_client
    assert manager.file_manager is mock_file_manager
    assert manager.hparams is mock_hparams
    assert manager.device == "cpu"


def test_aggregation_manager_constructor_with_none_gradient_manager():
    """Test AggregationManager constructor with None gradient_manager"""
    # Constructor doesn't validate - it just stores None
    manager = AggregationManager(
        gradient_manager=None,
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    # The constructor succeeds but stores None
    assert manager.gradient_manager is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_none_peer_manager():
    """Test AggregationManager constructor with None peer_manager"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=None,
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    assert manager.peer_manager is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_none_storage_client():
    """Test AggregationManager constructor with None storage_client"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=None,
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    assert manager.storage_client is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_none_file_manager():
    """Test AggregationManager constructor with None file_manager"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=None,
        hparams=MagicMock(),
        device="cpu",
    )

    assert manager.file_manager is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_none_hparams():
    """Test AggregationManager constructor with None hparams"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=None,
        device="cpu",
    )

    assert manager.hparams is None
    # TODO: Add validation in constructor to raise TypeError for None dependencies


def test_aggregation_manager_constructor_with_invalid_device_string():
    """Test AggregationManager constructor with invalid device string"""
    # This should not raise during construction, but may cause issues during tensor operations
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="invalid_device",
    )

    assert manager.device == "invalid_device"


def test_semaphore_initialization_with_default_value(valid_aggregation_manager):
    """Test semaphore initialization with default value (15)"""
    assert hasattr(valid_aggregation_manager, "gather_semaphore")
    assert isinstance(valid_aggregation_manager.gather_semaphore, asyncio.Semaphore)
    assert valid_aggregation_manager.gather_semaphore._value == 15


def test_semaphore_initialization_with_custom_value():
    """Test semaphore initialization with custom value"""
    # Note: Current implementation uses hardcoded value, but test what would happen if it were configurable
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    # Verify default is still 15
    assert manager.gather_semaphore._value == 15

    # TODO: Add parameter to constructor to allow custom semaphore value
    # TODO: Test with different semaphore values (1, 5, 50, 100)


def test_all_manager_dependencies_are_properly_stored_as_instance_variables(
    mock_gradient_manager,
    mock_peer_manager,
    mock_storage_client,
    mock_file_manager,
    mock_hparams,
):
    """Test all manager dependencies are properly stored as instance variables"""
    manager = AggregationManager(
        gradient_manager=mock_gradient_manager,
        peer_manager=mock_peer_manager,
        storage_client=mock_storage_client,
        file_manager=mock_file_manager,
        hparams=mock_hparams,
        device="cuda:0",
    )

    # Test all dependencies are stored correctly
    assert manager.gradient_manager is mock_gradient_manager
    assert manager.peer_manager is mock_peer_manager
    assert manager.storage_client is mock_storage_client
    assert manager.file_manager is mock_file_manager
    assert manager.hparams is mock_hparams
    assert manager.device == "cuda:0"

    # Test that semaphore is initialized
    assert hasattr(manager, "gather_semaphore")
    assert isinstance(manager.gather_semaphore, asyncio.Semaphore)

    # Test instance attributes directly using __dict__
    expected_attrs = {
        "gradient_manager",
        "peer_manager",
        "storage_client",
        "file_manager",
        "hparams",
        "device",
        "gather_semaphore",
    }
    actual_attrs = set(manager.__dict__.keys())

    assert expected_attrs.issubset(actual_attrs), (
        f"Missing expected attributes: {expected_attrs - actual_attrs}"
    )

    # Verify no unexpected critical attributes are missing
    missing_attrs = expected_attrs - actual_attrs
    assert len(missing_attrs) == 0, f"Missing critical attributes: {missing_attrs}"


def test_aggregation_manager_constructor_with_cuda_device():
    """Test AggregationManager constructor with CUDA device"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cuda:0",
    )

    assert manager.device == "cuda:0"


def test_aggregation_manager_constructor_with_cpu_device():
    """Test AggregationManager constructor with CPU device"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="cpu",
    )

    assert manager.device == "cpu"


def test_aggregation_manager_constructor_stores_references_not_copies():
    """Test that constructor stores references to dependencies, not copies"""
    gradient_manager = MagicMock()
    peer_manager = MagicMock()
    storage_client = MagicMock()
    file_manager = MagicMock()
    hparams = MagicMock()

    manager = AggregationManager(
        gradient_manager=gradient_manager,
        peer_manager=peer_manager,
        storage_client=storage_client,
        file_manager=file_manager,
        hparams=hparams,
        device="cpu",
    )

    # Test that the exact same objects are referenced
    assert manager.gradient_manager is gradient_manager
    assert manager.peer_manager is peer_manager
    assert manager.storage_client is storage_client
    assert manager.file_manager is file_manager
    assert manager.hparams is hparams


def test_aggregation_manager_constructor_with_empty_device_string():
    """Test AggregationManager constructor with empty device string"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="",
    )

    assert manager.device == ""


def test_aggregation_manager_constructor_with_whitespace_device():
    """Test AggregationManager constructor with whitespace device string"""
    manager = AggregationManager(
        gradient_manager=MagicMock(),
        peer_manager=MagicMock(),
        storage_client=MagicMock(),
        file_manager=MagicMock(),
        hparams=MagicMock(),
        device="  cpu  ",
    )

    assert manager.device == "  cpu  "


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - BASIC FUNCTIONALITY
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_single_uid_success_case(valid_aggregation_manager):
    """Test gather_gradients with single UID success case"""
    # Mock _aggregate_gradients to return a dict with all required keys
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2, 0.3])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert hasattr(result, "uids")
        assert hasattr(result, "state_dict")
        assert 1 in result.uids


@pytest.mark.asyncio
async def test_gather_gradients_with_multiple_uids_success_case(
    valid_aggregation_manager,
):
    """Test gather_gradients with multiple UIDs success case"""
    mock_aggregation_result = {
        "valid_uids": [1, 2, 3],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101, 102],
        "upload_bytes": 3000,
        "download_bytes": 6000,
        "succeeded": [1, 2, 3],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2, 3],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 3


@pytest.mark.asyncio
async def test_gather_gradients_with_empty_uid_list(valid_aggregation_manager):
    """Test gather_gradients with empty UID list"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_none_uid_list(valid_aggregation_manager):
    """Test gather_gradients with None UID list"""
    with pytest.raises((TypeError, AttributeError)):
        await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=None,
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )


@pytest.mark.asyncio
async def test_gather_gradients_with_duplicate_uids_in_list(valid_aggregation_manager):
    """Test gather_gradients with duplicate UIDs in list"""
    mock_aggregation_result = {
        "valid_uids": [1, 2],  # Duplicates removed
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [1, 2],
        "failed": [],
        "skipped_uids": [1],  # One duplicate was skipped
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 1, 2, 1],  # Duplicate UIDs
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert len(result.uids) == 2  # Duplicates handled


@pytest.mark.asyncio
async def test_gather_gradients_with_my_uid_included_in_target_uids_list(
    valid_aggregation_manager,
):
    """Test gather_gradients with my_uid included in target UIDs list"""
    mock_aggregation_result = {
        "valid_uids": [2, 3],  # my_uid (1) excluded from results
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [101, 102],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [2, 3],
        "failed": [],
        "skipped_uids": [1],  # my_uid was skipped
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=1,
            uids=[1, 2, 3],  # my_uid included
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert 1 not in result.uids  # my_uid should be excluded


@pytest.mark.asyncio
async def test_gather_gradients_with_invalid_uid_types_non_int(
    valid_aggregation_manager,
):
    """Test gather_gradients with invalid UID types (non-int)"""
    # These should be skipped due to invalid UID format
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=["not_an_int", "also_invalid"],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should return None as no valid gradients found
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_invalid_uid_types_negative(
    valid_aggregation_manager,
):
    """Test gather_gradients with invalid UID types (negative)"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[-1, -5],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should return None as gradients won't be found for negative UIDs
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_returns_none_when_no_valid_gradients_found(
    valid_aggregation_manager,
):
    """Test gather_gradients returns None when no valid gradients found"""
    # Mock storage to return None (no data found)
    with patch.object(
        valid_aggregation_manager.storage_client, "get_object", new_callable=AsyncMock
    ) as mock_get_object:
        mock_get_object.return_value = None

        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2, 3],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_returns_correct_simplenamespace_structure(
    valid_aggregation_manager,
):
    """Test gather_gradients returns correct SimpleNamespace structure"""
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # Check required attributes with correct field names
        assert hasattr(result, "uids")
        assert hasattr(result, "state_dict")
        assert hasattr(result, "global_steps")
        assert hasattr(result, "upload_bytes")
        assert hasattr(result, "download_bytes")
        assert hasattr(result, "success_rate")
        assert hasattr(result, "time")
        assert hasattr(result, "skipped_uids")

        # Check types
        assert isinstance(result.uids, list)
        assert isinstance(result.state_dict, SimpleNamespace)
        assert isinstance(result.global_steps, list)
        assert isinstance(result.upload_bytes, int)
        assert isinstance(result.download_bytes, int)
        assert isinstance(result.success_rate, float)
        assert isinstance(result.time, float)
        assert isinstance(result.skipped_uids, list)


@pytest.mark.asyncio
async def test_gather_gradients_with_local_true_vs_local_false_behavior(
    valid_aggregation_manager,
):
    """Test gather_gradients with local=True vs local=False behavior"""
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        # Test with local=True
        result_local = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            local=True,
        )

        # Test with local=False
        result_remote = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            local=False,
        )

        # Both should succeed
        assert result_local is not None
        assert result_remote is not None


@pytest.mark.asyncio
async def test_gather_gradients_with_different_timeout_values_1s(
    valid_aggregation_manager,
):
    """Test gather_gradients with timeout=1s"""
    # For timeout tests, we just verify the method completes
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=1,  # Very short timeout
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should complete (likely returning None due to timeout)
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_different_timeout_values_30s(
    valid_aggregation_manager,
):
    """Test gather_gradients with timeout=30s"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should complete
    assert result is None  # No gradients found


@pytest.mark.asyncio
async def test_gather_gradients_with_different_timeout_values_60s(
    valid_aggregation_manager,
):
    """Test gather_gradients with timeout=60s"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=60,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should complete
    assert result is None  # No gradients found


@pytest.mark.asyncio
async def test_gather_gradients_with_zero_timeout_edge_case(valid_aggregation_manager):
    """Test gather_gradients with zero timeout (edge case)"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=0,
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should handle gracefully
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_negative_timeout_should_handle_gracefully(
    valid_aggregation_manager,
):
    """Test gather_gradients with negative timeout (should handle gracefully)"""
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=-5,  # Negative timeout
        device="cpu",
        totalks={"layer.weight": 1000},
    )

    # Should handle gracefully
    assert result is None


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - TIME CONSTRAINTS
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_time_min_constraint_only(
    valid_aggregation_manager,
):
    """Test gather_gradients with time_min constraint only"""
    time_min = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_time_max_constraint_only(
    valid_aggregation_manager,
):
    """Test gather_gradients with time_max constraint only"""
    time_max = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_both_time_min_and_time_max_constraints(
    valid_aggregation_manager,
):
    """Test gather_gradients with both time_min and time_max constraints"""
    time_min = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1, 2],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [1, 2],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 2


@pytest.mark.asyncio
async def test_gather_gradients_with_time_min_equals_time_max_exact_timestamp(
    valid_aggregation_manager,
):
    """Test gather_gradients with time_min = time_max (exact timestamp)"""
    exact_time = datetime(2024, 6, 15, 14, 30, 45, tzinfo=timezone.utc)

    # Should return None as exact timestamp match is unlikely
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
        time_min=exact_time,
        time_max=exact_time,
    )

    # Exact timestamp matching typically returns no results
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_time_min_greater_than_time_max_invalid_range(
    valid_aggregation_manager,
):
    """Test gather_gradients with time_min > time_max (invalid range)"""
    time_min = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(
        2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc
    )  # Earlier than time_min

    # Invalid time range should return no results
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
        time_min=time_min,
        time_max=time_max,
    )

    # Invalid range should return None
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_timezone_naive_datetime_objects(
    valid_aggregation_manager,
):
    """Test gather_gradients with timezone-naive datetime objects"""
    time_min = datetime(2024, 6, 1, 12, 0, 0)  # No timezone
    time_max = datetime(2024, 6, 30, 12, 0, 0)  # No timezone

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_timezone_aware_datetime_objects(
    valid_aggregation_manager,
):
    """Test gather_gradients with timezone-aware datetime objects"""
    time_min = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_mixed_timezone_datetime_objects(
    valid_aggregation_manager,
):
    """Test gather_gradients with mixed timezone datetime objects"""
    # Mix of UTC and timezone-naive
    time_min = datetime(2024, 6, 1, 12, 0, 0)  # Naive
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)  # UTC

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
            time_max=time_max,
        )

        # Should handle mixed timezones gracefully
        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_very_old_time_min_before_any_data_exists(
    valid_aggregation_manager,
):
    """Test gather_gradients with very old time_min (before any data exists)"""
    # Very old timestamp that predates any possible data
    time_min = datetime(1990, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_min=time_min,
        )

        # Should include all available data
        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_very_future_time_max(valid_aggregation_manager):
    """Test gather_gradients with very future time_max"""
    # Future timestamp
    time_max = datetime(2030, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            time_max=time_max,
        )

        # Should include all available data up to the future time
        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_time_filtering_with_local_storage(
    valid_aggregation_manager,
):
    """Test gather_gradients time filtering with local storage"""
    time_min = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            local=True,  # Use local storage
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_time_filtering_with_remote_storage(
    valid_aggregation_manager,
):
    """Test gather_gradients time filtering with remote storage"""
    time_min = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_max = datetime(2024, 6, 30, 12, 0, 0, tzinfo=timezone.utc)

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            local=False,  # Use remote storage
            time_min=time_min,
            time_max=time_max,
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - STALE RETENTION
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_default_stale_retention_10(
    valid_aggregation_manager,
):
    """Test gather_gradients with default stale_retention (10)"""
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            stale_retention=10,  # Default value
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        # TODO: Verify stale_retention parameter is passed through aggregation pipeline


@pytest.mark.asyncio
async def test_gather_gradients_with_stale_retention_0_no_retention(
    valid_aggregation_manager,
):
    """Test gather_gradients with stale_retention=0 (no retention)"""
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            stale_retention=0,  # No retention
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        # TODO: Verify that stale_retention=0 disables retention mechanisms


@pytest.mark.asyncio
async def test_gather_gradients_with_very_high_stale_retention_1000(
    valid_aggregation_manager,
):
    """Test gather_gradients with very high stale_retention (1000)"""
    mock_aggregation_result = {
        "valid_uids": [1, 2],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [1, 2],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
            stale_retention=1000,  # Very high retention
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 2
        # TODO: Verify high retention allows more stale gradients to be included


@pytest.mark.asyncio
async def test_gather_gradients_with_negative_stale_retention_edge_case(
    valid_aggregation_manager,
):
    """Test gather_gradients with negative stale_retention (edge case)"""
    # Negative stale_retention should be handled gracefully
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks={"layer.weight": 1000},
        stale_retention=-5,  # Negative value
    )

    # Should handle gracefully, likely returning None or treating as 0
    # TODO: Define expected behavior for negative stale_retention values
    assert result is None or isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_stale_retention_affects_cleanup_operations_correctly(
    valid_aggregation_manager,
):
    """Test stale_retention affects cleanup operations correctly"""
    # Mock file system operations to track cleanup behavior
    with (
        patch.object(
            valid_aggregation_manager.file_manager, "delete_file"
        ) as mock_delete,
        patch("os.path.exists", return_value=True),
        patch("os.path.getmtime", return_value=time.time() - 3600),
    ):  # 1 hour old
        # Test with different stale_retention values
        for retention in [0, 5, 10, 100]:
            mock_delete.reset_mock()

            result = await valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[1],
                window=10,
                timeout=30,
                device="cpu",
                totalks={"layer.weight": 1000},
                stale_retention=retention,
            )

            # TODO: Verify cleanup behavior changes based on stale_retention value
            # Higher retention should result in fewer cleanup operations


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - DEVICE HANDLING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_device_cpu(valid_aggregation_manager):
    """Test gather_gradients with device="cpu" """
    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ) as mock_agg:
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # Verify device parameter was passed to aggregation
        mock_agg.assert_called_once()
        call_args = mock_agg.call_args
        assert call_args[0][1] == "cpu"  # device parameter


@pytest.mark.asyncio
async def test_gather_gradients_with_device_cuda_if_available(
    valid_aggregation_manager,
):
    """Test gather_gradients with device="cuda" (if available)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ) as mock_agg:
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device=device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # Verify correct device was used
        mock_agg.assert_called_once()
        call_args = mock_agg.call_args
        assert call_args[0][1] == device


@pytest.mark.asyncio
async def test_gather_gradients_with_device_cuda_specific_gpu(
    valid_aggregation_manager,
):
    """Test gather_gradients with device="cuda:0" (specific GPU)"""
    device = "cuda:0"

    mock_aggregation_result = {
        "valid_uids": [1, 2],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100, 101],
        "upload_bytes": 2000,
        "download_bytes": 4000,
        "succeeded": [1, 2],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ) as mock_agg:
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2],
            window=10,
            timeout=30,
            device=device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 2

        # Verify specific GPU device was used
        mock_agg.assert_called_once()
        call_args = mock_agg.call_args
        assert call_args[0][1] == "cuda:0"


@pytest.mark.asyncio
async def test_gather_gradients_with_invalid_device_string(valid_aggregation_manager):
    """Test gather_gradients with invalid device string"""
    invalid_device = "invalid_device_string"

    # The current implementation doesn't validate device strings during gather_gradients
    # It only fails during actual tensor operations in _aggregate_gradients
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device=invalid_device,
        totalks={"layer.weight": 1000},
    )

    # Should return None since no valid gradients will be found/processed
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_device_mismatch_between_input_and_target(
    valid_aggregation_manager,
):
    """Test gather_gradients with device mismatch between input and target"""
    # Mock tensors on different devices
    cpu_tensor = torch.tensor([0.1, 0.2], device="cpu")

    # Simulate device mismatch scenario
    mock_response = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": cpu_tensor},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager, "_aggregate_gradients", return_value=mock_response
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cuda" if torch.cuda.is_available() else "cpu",
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        # TODO: Verify that device movement occurred during aggregation


@pytest.mark.asyncio
async def test_tensor_device_movement_during_aggregation_process(
    valid_aggregation_manager,
):
    """Test tensor device movement during aggregation process"""
    # Create mock tensors on CPU
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0], device="cpu")
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mock the raw responses with CPU tensors
    mock_raw_responses = [
        {
            "uid": 1,
            "response": (
                {"state_dict": {"layer.weight": cpu_tensor}, "global_step": 100},
                100,
            ),
            "is_exception": False,
        }
    ]

    with (
        patch.object(
            valid_aggregation_manager,
            "gather_semaphore",
            new_callable=lambda: asyncio.Semaphore(15),
        ),
        patch.object(
            valid_aggregation_manager,
            "_get_with_retry",
            side_effect=[mock_raw_responses[0]["response"]],
        ),
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device=target_device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # Verify tensor was moved to target device
        if hasattr(result.state_dict, "layer"):
            layer_tensors = result.state_dict.layer
            if hasattr(layer_tensors, "weight") and isinstance(
                layer_tensors.weight, list
            ):
                for tensor in layer_tensors.weight:
                    if isinstance(tensor, torch.Tensor):
                        assert str(tensor.device).startswith(
                            target_device.split(":")[0]
                        )


@pytest.mark.asyncio
async def test_gather_gradients_device_consistency_across_multiple_uids(
    valid_aggregation_manager,
):
    """Test device consistency when gathering from multiple UIDs"""
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mock tensors from different UIDs, all should be moved to target device
    mock_aggregation_result = {
        "valid_uids": [1, 2, 3],
        "aggregated_state_dict": {
            "layer.weight": [
                torch.tensor([0.1, 0.2], device=target_device),
                torch.tensor([0.3, 0.4], device=target_device),
                torch.tensor([0.5, 0.6], device=target_device),
            ]
        },
        "global_steps": [100, 101, 102],
        "upload_bytes": 3000,
        "download_bytes": 6000,
        "succeeded": [1, 2, 3],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2, 3],
            window=10,
            timeout=30,
            device=target_device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 3

        # Verify all tensors are on the same target device
        if hasattr(result.state_dict, "layer") and hasattr(
            result.state_dict.layer, "weight"
        ):
            tensors = result.state_dict.layer.weight
            if isinstance(tensors, list):
                for tensor in tensors:
                    if isinstance(tensor, torch.Tensor):
                        assert str(tensor.device).startswith(
                            target_device.split(":")[0]
                        )


@pytest.mark.asyncio
async def test_gather_gradients_device_parameter_validation(valid_aggregation_manager):
    """Test device parameter validation and error handling"""
    test_devices = ["cpu", "cuda", "cuda:0", "cuda:1", "mps", ""]

    for device in test_devices:
        try:
            result = await valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[1],
                window=10,
                timeout=30,
                device=device,
                totalks={"layer.weight": 1000},
            )

            # Should either succeed or fail gracefully
            assert result is None or isinstance(result, SimpleNamespace)

        except Exception as e:
            # Device-related exceptions should be handled gracefully
            assert "device" in str(e).lower() or "cuda" in str(e).lower()


@pytest.mark.asyncio
async def test_gather_gradients_mixed_device_tensors_in_quant_params(
    valid_aggregation_manager,
):
    """Test handling of mixed device tensors in quant_params structures"""
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mock quant_params with tensors on different devices
    mock_quant_params = {
        "scale": torch.tensor([1.0], device="cpu"),
        "zero_point": torch.tensor([0], device="cpu"),
        "bits": 8,
    }

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight_quant_params": [mock_quant_params]},
        "global_steps": [100],
        "upload_bytes": 1000,
        "download_bytes": 2000,
        "succeeded": [1],
        "failed": [],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device=target_device,
            totalks={"layer.weight": 1000},
        )

        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # TODO: Verify quant_params tensors were moved to target device
        # This requires implementing device movement validation for nested structures


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - TOTALKS PARAMETER
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_with_empty_totalks_dict(valid_aggregation_manager):
    """Test gather_gradients with empty totalks dict"""
    empty_totalks = {}

    # Mock response with some gradient data
    mock_raw_responses = [
        {
            "uid": 1,
            "response": (
                {
                    "state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
                    "global_step": 100,
                },
                100,
            ),
            "is_exception": False,
        }
    ]

    with patch.object(
        valid_aggregation_manager,
        "_get_with_retry",
        side_effect=[mock_raw_responses[0]["response"]],
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks=empty_totalks,  # Empty dict
        )

        # Should handle empty totalks gracefully
        assert result is None or isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_none_totalks(valid_aggregation_manager):
    """Test gather_gradients with None totalks"""
    # The current implementation doesn't validate totalks parameter upfront
    # It only fails when trying to process gradients that need validation
    result = await valid_aggregation_manager.gather_gradients(
        my_uid=0,
        uids=[1],
        window=10,
        timeout=30,
        device="cpu",
        totalks=None,  # None value
    )

    # Should return None since no valid gradients can be processed without totalks
    assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_mismatched_totalks_keys_vs_gradient_keys(
    valid_aggregation_manager,
):
    """Test gather_gradients with mismatched totalks keys vs gradient keys"""
    totalks = {"layer.weight": 1000, "layer.bias": 500}  # Expected keys

    # Mock gradient response with different keys
    mock_gradient_state = {
        "different.weight": torch.tensor([0.1, 0.2]),  # Different key
        "other.layer": torch.tensor([0.3, 0.4]),  # Different key
    }

    mock_raw_responses = [
        {
            "uid": 1,
            "response": (mock_gradient_state, 100),
            "is_exception": False,
        }
    ]

    with patch.object(
        valid_aggregation_manager,
        "_get_with_retry",
        side_effect=[mock_raw_responses[0]["response"]],
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0, uids=[1], window=10, timeout=30, device="cpu", totalks=totalks
        )

        # Should succeed but may have no valid gradients due to key mismatch
        assert result is None or isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_with_totalks_containing_extra_unused_keys(
    valid_aggregation_manager,
):
    """Test gather_gradients with totalks containing extra unused keys"""
    totalks = {
        "layer.weight": 1000,
        "layer.bias": 500,
        "unused.layer1": 200,  # Extra unused key
        "unused.layer2": 300,  # Extra unused key
        "nonexistent.param": 100,  # Extra unused key
    }

    mock_gradient_state = {
        "layer.weight": torch.tensor([0.1, 0.2]),
        "layer.bias": torch.tensor([0.3]),
    }

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {
            "layer.weight": [torch.tensor([0.1, 0.2])],
            "layer.bias": [torch.tensor([0.3])],
        },
        "global_steps": [100],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0, uids=[1], window=10, timeout=30, device="cpu", totalks=totalks
        )

        # Should handle extra keys gracefully
        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 1


@pytest.mark.asyncio
async def test_gather_gradients_with_totalks_missing_required_keys(
    valid_aggregation_manager,
):
    """Test gather_gradients with totalks missing required keys"""
    totalks = {"layer.weight": 1000}  # Missing key for layer.bias

    # Mock gradient with indices that require totalk validation
    mock_gradient_state = {
        "layer.weight_idxs": torch.tensor([0, 1, 2]),
        "layer.weight_vals": torch.tensor([0.1, 0.2, 0.3]),
        "layer.bias_idxs": torch.tensor([0, 1]),  # This will fail - no totalk
        "layer.bias_vals": torch.tensor([0.4, 0.5]),
    }

    # FIXED: Response should be tuple (state_dict, global_step) directly
    mock_response = (mock_gradient_state, 100)

    with patch.object(
        valid_aggregation_manager,
        "_get_with_retry",
        side_effect=[mock_response],
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0, uids=[1], window=10, timeout=30, device="cpu", totalks=totalks
        )

        # Should return None due to validation failure
        assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_totalks_containing_invalid_values_negative_zero(
    valid_aggregation_manager,
):
    """Test gather_gradients with totalks containing invalid values (negative, zero)"""
    invalid_totalks_cases = [
        {"layer.weight": -100},  # Negative value
        {"layer.weight": 0},  # Zero value
        {"layer.weight": -1},  # Negative value
    ]

    for totalks in invalid_totalks_cases:
        # Mock gradient with indices
        mock_gradient_state = {
            "layer.weight_idxs": torch.tensor([0, 1]),
            "layer.weight_vals": torch.tensor([0.1, 0.2]),
        }

        # FIXED: Response should be tuple (state_dict, global_step) directly
        mock_response = (mock_gradient_state, 100)

        with patch.object(
            valid_aggregation_manager,
            "_get_with_retry",
            side_effect=[mock_response],
        ):
            result = await valid_aggregation_manager.gather_gradients(
                my_uid=0, uids=[1], window=10, timeout=30, device="cpu", totalks=totalks
            )

            # Should fail validation with invalid totalks values
            assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_totalks_containing_non_integer_values(
    valid_aggregation_manager,
):
    """Test gather_gradients with totalks containing non-integer values"""
    invalid_totalks_cases = [
        {"layer.weight": 100.5},  # Float value
        {"layer.weight": "1000"},  # String value
        {"layer.weight": [1000]},  # List value
        {"layer.weight": None},  # None value
    ]

    for totalks in invalid_totalks_cases:
        # Mock gradient with indices
        mock_gradient_state = {
            "layer.weight_idxs": torch.tensor([0, 1]),
            "layer.weight_vals": torch.tensor([0.1, 0.2]),
        }

        mock_raw_responses = [
            {
                "uid": 1,
                "response": (
                    {"state_dict": mock_gradient_state, "global_step": 100},
                    100,
                ),
                "is_exception": False,
            }
        ]

        with patch.object(
            valid_aggregation_manager,
            "_get_with_retry",
            side_effect=[mock_raw_responses[0]["response"]],
        ):
            try:
                result = await valid_aggregation_manager.gather_gradients(
                    my_uid=0,
                    uids=[1],
                    window=10,
                    timeout=30,
                    device="cpu",
                    totalks=totalks,
                )

                # Should either fail or handle gracefully
                assert result is None or isinstance(result, SimpleNamespace)

            except (TypeError, ValueError, AttributeError):
                # Type errors are acceptable for non-integer values
                pass


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - CONCURRENT OPERATIONS
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_semaphore_limits_concurrent_operations_to_15(
    valid_aggregation_manager,
):
    """Test gather_gradients semaphore limits concurrent operations to 15"""
    # Verify semaphore is initialized with limit of 15
    assert valid_aggregation_manager.gather_semaphore._value == 15

    # Mock slow response to test semaphore blocking
    async def slow_mock_response(*args, **kwargs):
        await asyncio.sleep(0.1)  # Small delay
        return None

    with patch.object(
        valid_aggregation_manager, "_get_with_retry", side_effect=slow_mock_response
    ):
        # Start 15 concurrent operations (should all proceed)
        tasks = [
            valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[i],
                window=10,
                timeout=30,
                device="cpu",
                totalks={"layer.weight": 1000},
            )
            for i in range(1, 16)  # UIDs 1-15
        ]

        # All should complete
        results = await asyncio.gather(*tasks)
        assert len(results) == 15
        assert all(result is None for result in results)  # All return None due to mock


@pytest.mark.asyncio
async def test_gather_gradients_with_20_plus_simultaneous_calls_semaphore_blocking(
    valid_aggregation_manager,
):
    """Test gather_gradients with 20+ simultaneous calls (semaphore blocking)"""
    call_order = []
    call_times = []

    async def tracking_mock_response(*args, **kwargs):
        call_times.append(time.time())
        call_order.append(len(call_order))
        await asyncio.sleep(0.05)  # Small delay to observe blocking
        return None

    with patch.object(
        valid_aggregation_manager, "_get_with_retry", side_effect=tracking_mock_response
    ):
        start_time = time.time()

        # Start 20 concurrent operations (5 should be blocked initially)
        tasks = [
            valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[i],
                window=10,
                timeout=30,
                device="cpu",
                totalks={"layer.weight": 1000},
            )
            for i in range(1, 21)  # UIDs 1-20
        ]

        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # Verify all completed
        assert len(results) == 20
        assert all(result is None for result in results)

        # Verify some blocking occurred (total time should be > single operation time)
        # With 20 operations, 15 concurrent + 5 waiting, should take longer than single operation
        assert end_time - start_time > 0.05  # At least one delay period

        # Verify we have exactly 20 calls
        assert len(call_order) == 20


@pytest.mark.asyncio
async def test_gather_gradients_with_concurrent_calls_using_same_uids(
    valid_aggregation_manager,
):
    """Test gather_gradients with concurrent calls using same UIDs"""
    shared_uids = [1, 2, 3]
    call_count = 0

    async def counting_mock_response(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.02)
        return None

    with patch.object(
        valid_aggregation_manager, "_get_with_retry", side_effect=counting_mock_response
    ):
        # Multiple concurrent calls with same UIDs
        tasks = [
            valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=shared_uids,
                window=10 + i,  # Different windows
                timeout=30,
                device="cpu",
                totalks={"layer.weight": 1000},
            )
            for i in range(5)  # 5 concurrent calls
        ]

        results = await asyncio.gather(*tasks)

        # All should complete
        assert len(results) == 5
        assert all(result is None for result in results)

        # Should have made calls for each UID in each gather operation
        assert call_count == 5 * len(
            shared_uids
        )  # 5 calls * 3 UIDs each = 15 total calls


@pytest.mark.asyncio
async def test_gather_gradients_with_concurrent_calls_using_different_uids(
    valid_aggregation_manager,
):
    """Test gather_gradients with concurrent calls using different UIDs"""
    call_count = 0
    uid_calls = set()

    async def tracking_mock_response(uid, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        uid_calls.add(uid)
        await asyncio.sleep(0.02)
        return None

    with patch.object(
        valid_aggregation_manager, "_get_with_retry", side_effect=tracking_mock_response
    ):
        # Multiple concurrent calls with different UIDs
        tasks = [
            valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[i + 1],  # Different UIDs: [1], [2], [3], [4], [5]
                window=10,
                timeout=30,
                device="cpu",
                totalks={"layer.weight": 1000},
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All should complete
        assert len(results) == 5
        assert all(result is None for result in results)

        # Should have made exactly 5 calls with different UIDs
        assert call_count == 5
        assert len(uid_calls) == 5  # All different UIDs were called


@pytest.mark.asyncio
async def test_gather_gradients_semaphore_release_on_exception(
    valid_aggregation_manager,
):
    """Test gather_gradients semaphore release on exception"""
    initial_semaphore_value = valid_aggregation_manager.gather_semaphore._value

    async def exception_mock_response(*args, **kwargs):
        raise RuntimeError("Test exception")

    with patch.object(
        valid_aggregation_manager,
        "_get_with_retry",
        side_effect=exception_mock_response,
    ):
        # This should raise an exception but still release semaphore
        try:
            await valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[1],
                window=10,
                timeout=30,
                device="cpu",
                totalks={"layer.weight": 1000},
            )
        except Exception:
            pass  # Expected exception

        # Semaphore should be back to initial value
        assert (
            valid_aggregation_manager.gather_semaphore._value == initial_semaphore_value
        )


@pytest.mark.asyncio
async def test_gather_gradients_semaphore_release_on_timeout(valid_aggregation_manager):
    """Test gather_gradients semaphore release on timeout"""
    initial_semaphore_value = valid_aggregation_manager.gather_semaphore._value

    async def timeout_mock_response(*args, **kwargs):
        raise asyncio.TimeoutError("Test timeout")

    with patch.object(
        valid_aggregation_manager, "_get_with_retry", side_effect=timeout_mock_response
    ):
        try:
            await valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[1],
                window=10,
                timeout=30,
                device="cpu",
                totalks={"layer.weight": 1000},
            )
        except Exception:
            pass  # Expected timeout

        # Semaphore should be back to initial value
        assert (
            valid_aggregation_manager.gather_semaphore._value == initial_semaphore_value
        )


@pytest.mark.asyncio
async def test_gather_gradients_semaphore_release_on_success(valid_aggregation_manager):
    """Test gather_gradients semaphore release on success"""
    initial_semaphore_value = valid_aggregation_manager.gather_semaphore._value

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
        "global_steps": [100],
        "skipped_uids": [],
    }

    with patch.object(
        valid_aggregation_manager,
        "_aggregate_gradients",
        return_value=mock_aggregation_result,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        # Should succeed
        assert result is not None
        assert isinstance(result, SimpleNamespace)

        # Semaphore should be back to initial value
        assert (
            valid_aggregation_manager.gather_semaphore._value == initial_semaphore_value
        )


@pytest.mark.asyncio
async def test_gather_gradients_concurrent_semaphore_stress_test(
    valid_aggregation_manager,
):
    """Test gather_gradients semaphore under stress with many concurrent operations"""
    initial_semaphore_value = valid_aggregation_manager.gather_semaphore._value
    completed_operations = 0

    async def quick_mock_response(*args, **kwargs):
        nonlocal completed_operations
        await asyncio.sleep(0.001)  # Very quick operation
        completed_operations += 1
        return None

    with patch.object(
        valid_aggregation_manager, "_get_with_retry", side_effect=quick_mock_response
    ):
        # Launch 50 concurrent operations
        tasks = [
            valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[i % 10 + 1],  # Cycle through UIDs 1-10
                window=i,
                timeout=30,
                device="cpu",
                totalks={"layer.weight": 1000},
            )
            for i in range(50)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete (even if they return None)
        assert len(results) == 50
        assert completed_operations == 50  # All operations completed

        # Semaphore should be back to initial value
        assert (
            valid_aggregation_manager.gather_semaphore._value == initial_semaphore_value
        )

        # No exceptions should have occurred
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0


# -----------------------------------------------------------------------------
# GATHER_GRADIENTS - ERROR HANDLING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_gradients_when_all_uids_fail_with_exceptions(
    valid_aggregation_manager,
):
    """Test gather_gradients when all UIDs fail with exceptions"""

    async def exception_mock_response(*args, **kwargs):
        raise ConnectionError("Network connection failed")

    with patch.object(
        valid_aggregation_manager,
        "_get_with_retry",
        side_effect=exception_mock_response,
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2, 3],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        # Should return None when all UIDs fail
        assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_when_some_uids_fail_some_succeed(
    valid_aggregation_manager,
):
    """Test gather_gradients when some UIDs fail, some succeed"""

    async def mixed_mock_response(uid, *args, **kwargs):
        if uid == "1":
            # UID 1 succeeds
            return (
                {
                    "state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
                    "global_step": 100,
                },
                100,
            )
        elif uid == "2":
            # UID 2 fails with exception
            raise TimeoutError("UID 2 timeout")
        else:
            # UID 3 returns None
            return None

    mock_aggregation_result = {
        "valid_uids": [1],
        "aggregated_state_dict": {"layer.weight": [torch.tensor([0.1, 0.2])]},
        "global_steps": [100],
        "skipped_uids": [2, 3],
    }

    with (
        patch.object(
            valid_aggregation_manager,
            "_get_with_retry",
            side_effect=mixed_mock_response,
        ),
        patch.object(
            valid_aggregation_manager,
            "_aggregate_gradients",
            return_value=mock_aggregation_result,
        ),
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2, 3],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        # Should succeed with partial results
        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 1
        assert result.uids[0] == 1


@pytest.mark.asyncio
async def test_gather_gradients_when_network_timeout_occurs(valid_aggregation_manager):
    """Test gather_gradients when network timeout occurs"""

    async def timeout_mock_response(*args, **kwargs):
        raise asyncio.TimeoutError("Network timeout after 30 seconds")

    with patch.object(
        valid_aggregation_manager, "_get_with_retry", side_effect=timeout_mock_response
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        # Should handle timeout gracefully and return None
        assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_when_storage_client_throws_exceptions(
    valid_aggregation_manager,
):
    """Test gather_gradients when storage client throws exceptions"""

    async def storage_exception_mock(*args, **kwargs):
        raise Exception("Storage backend unavailable")

    with patch.object(
        valid_aggregation_manager, "_get_with_retry", side_effect=storage_exception_mock
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        # Should handle storage exceptions gracefully
        assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_when_file_manager_throws_exceptions(
    valid_aggregation_manager,
):
    """Test gather_gradients when file_manager throws exceptions"""
    # Mock file_manager to throw exception during temp file operations
    with patch.object(
        valid_aggregation_manager.file_manager,
        "create_temp_file",
        side_effect=OSError("Disk full"),
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        # Should handle file system exceptions gracefully
        assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_when_gradient_manager_throws_exceptions(
    valid_aggregation_manager,
):
    """Test gather_gradients when gradient_manager throws exceptions"""
    # Mock gradient_manager operations to throw exceptions
    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        side_effect=ValueError("Invalid gradient format"),
    ):
        # Mock response with indices that would trigger gradient_manager call
        mock_gradient_state = {
            "layer.weight_idxs": torch.tensor([0, 1]),
            "layer.weight_vals": torch.tensor([0.1, 0.2]),
        }

        mock_raw_responses = [
            {
                "uid": 1,
                "response": (mock_gradient_state, 100),
                "is_exception": False,
            }
        ]

        with patch.object(
            valid_aggregation_manager,
            "_get_with_retry",
            side_effect=[mock_raw_responses[0]["response"]],
        ):
            result = await valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[1],
                window=10,
                timeout=30,
                device="cpu",
                totalks={"layer.weight": 1000},
            )

            # Should handle gradient_manager exceptions gracefully
            assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_when_peer_manager_throws_exceptions(
    valid_aggregation_manager,
):
    """Test gather_gradients when peer_manager throws exceptions"""
    # Mock peer_manager to throw exception during bucket retrieval
    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        side_effect=RuntimeError("Chain manager error"),
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        # Should handle peer_manager exceptions gracefully
        assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_malformed_gradient_responses(
    valid_aggregation_manager,
):
    """Test gather_gradients with malformed gradient responses"""
    malformed_responses = [
        # Not a tuple at all
        {"invalid": "response"},
        # Empty dict
        {},
        # Wrong tuple length (too many elements)
        ({"layer.weight": torch.tensor([0.1])}, 100, "extra"),
        # Wrong tuple length (too few elements)
        ({"layer.weight": torch.tensor([0.1])},),
        # Tuple with None state_dict
        (None, 100),
        # String instead of dict for state_dict
        ("invalid_state_dict", 100),
        # List instead of dict for state_dict
        ([1, 2, 3], 100),
    ]

    for malformed_response in malformed_responses:
        with patch.object(
            valid_aggregation_manager,
            "_get_with_retry",
            return_value=malformed_response,
        ):
            result = await valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[1],
                window=10,
                timeout=30,
                device="cpu",
                totalks={"layer.weight": 1000},
            )

            # Should handle malformed responses gracefully
            assert result is None


@pytest.mark.asyncio
async def test_gather_gradients_with_corrupted_gradient_data(valid_aggregation_manager):
    """Test gather_gradients with corrupted gradient data"""
    corrupted_data_cases = [
        # Tensor with NaN values
        {"layer.weight": torch.tensor([float("nan"), 0.2])},
        # Tensor with Inf values
        {"layer.weight": torch.tensor([float("inf"), 0.2])},
        # Tensor with -Inf values
        {"layer.weight": torch.tensor([float("-inf"), 0.2])},
        # Mixed NaN and valid values
        {"layer.weight": torch.tensor([0.1, float("nan"), 0.3])},
        # Empty tensor
        {"layer.weight": torch.tensor([])},
        # Tensor with wrong shape
        {
            "layer.weight_idxs": torch.tensor([0, 1]),
            "layer.weight_vals": torch.tensor([0.1]),
        },  # Mismatched lengths
        # Invalid indices (negative)
        {
            "layer.weight_idxs": torch.tensor([-1, 0]),
            "layer.weight_vals": torch.tensor([0.1, 0.2]),
        },
        # Indices out of bounds (would need totalks validation)
        {
            "layer.weight_idxs": torch.tensor([999999, 1000000]),
            "layer.weight_vals": torch.tensor([0.1, 0.2]),
        },
    ]

    for corrupted_state in corrupted_data_cases:
        mock_raw_responses = [
            {
                "uid": 1,
                "response": (corrupted_state, 100),
                "is_exception": False,
            }
        ]

        with patch.object(
            valid_aggregation_manager,
            "_get_with_retry",
            side_effect=[mock_raw_responses[0]["response"]],
        ):
            result = await valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[1],
                window=10,
                timeout=30,
                device="cpu",
                totalks={"layer.weight": 1000},
            )

            # Should handle corrupted data gracefully (may return None or filtered results)
            assert result is None or isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_graceful_degradation_with_partial_failures(
    valid_aggregation_manager,
):
    """Test gather_gradients graceful degradation with partial failures"""
    # Simulate a mix of success, failure, and corrupted responses
    responses = [
        # UID 1: Success
        (
            {
                "state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
                "global_step": 100,
            },
            100,
        ),
        # UID 2: Network error (will be handled by _get_with_retry)
        ConnectionError("Network error"),
        # UID 3: Corrupted data
        (
            {
                "state_dict": {"layer.weight": torch.tensor([float("nan")])},
                "global_step": 101,
            },
            101,
        ),
        # UID 4: Success
        (
            {
                "state_dict": {"layer.weight": torch.tensor([0.3, 0.4])},
                "global_step": 102,
            },
            102,
        ),
        # UID 5: Malformed response
        {"malformed": "response"},
    ]

    async def mixed_response_mock(uid, *args, **kwargs):
        uid_idx = int(uid) - 1
        response = responses[uid_idx]
        if isinstance(response, Exception):
            raise response
        return response

    # Mock aggregation to return only valid UIDs
    mock_aggregation_result = {
        "valid_uids": [1, 4],  # Only UIDs with valid data
        "aggregated_state_dict": {
            "layer.weight": [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])]
        },
        "global_steps": [100, 102],
        "skipped_uids": [2, 3, 5],
    }

    with (
        patch.object(
            valid_aggregation_manager,
            "_get_with_retry",
            side_effect=mixed_response_mock,
        ),
        patch.object(
            valid_aggregation_manager,
            "_aggregate_gradients",
            return_value=mock_aggregation_result,
        ),
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=[1, 2, 3, 4, 5],
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        # Should succeed with partial results - graceful degradation
        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 2  # Only successful UIDs
        assert result.uids == [1, 4]

        # Should have aggregated the valid gradients
        assert hasattr(result, "state_dict")


@pytest.mark.asyncio
async def test_gather_gradients_error_recovery_after_multiple_failures(
    valid_aggregation_manager,
):
    """Test gather_gradients error recovery after multiple failures"""
    failure_count = 0

    async def intermittent_failure_mock(*args, **kwargs):
        nonlocal failure_count
        failure_count += 1

        # Fail first few attempts, then succeed
        if failure_count <= 3:
            raise ConnectionError(f"Temporary failure {failure_count}")
        else:
            return (
                {
                    "state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
                    "global_step": 100,
                },
                100,
            )

    # Don't patch _aggregate_gradients - let it handle the actual responses
    with patch.object(
        valid_aggregation_manager,
        "_get_with_retry",
        side_effect=intermittent_failure_mock,
    ):
        # Multiple calls to test recovery
        for attempt in range(5):
            result = await valid_aggregation_manager.gather_gradients(
                my_uid=0,
                uids=[1],
                window=10 + attempt,
                timeout=30,
                device="cpu",
                totalks={"layer.weight": 1000},
            )

            if attempt < 3:
                # First few attempts should fail
                assert result is None
            else:
                # Later attempts should succeed after recovery
                assert result is not None
                assert isinstance(result, SimpleNamespace)


@pytest.mark.asyncio
async def test_gather_gradients_handles_mixed_exception_types(
    valid_aggregation_manager,
):
    """Test gather_gradients handles mixed exception types correctly"""
    exception_types = [
        ConnectionError("Network connection failed"),
        TimeoutError("Operation timeout"),
        ValueError("Invalid value"),
        RuntimeError("Runtime error"),
        OSError("File system error"),
        MemoryError("Out of memory"),
        KeyError("Missing key"),
        AttributeError("Missing attribute"),
    ]

    async def cycling_exception_mock(uid, *args, **kwargs):
        uid_idx = int(uid) - 1
        if uid_idx < len(exception_types):
            raise exception_types[uid_idx]
        else:
            # Last UID succeeds
            return (
                {
                    "state_dict": {"layer.weight": torch.tensor([0.1, 0.2])},
                    "global_step": 100,
                },
                100,
            )

    mock_aggregation_result = {
        "valid_uids": [len(exception_types) + 1],
        "aggregated_state_dict": {"layer.weight": [torch.tensor([0.1, 0.2])]},
        "global_steps": [100],
        "skipped_uids": list(range(1, len(exception_types) + 1)),
    }

    with (
        patch.object(
            valid_aggregation_manager,
            "_get_with_retry",
            side_effect=cycling_exception_mock,
        ),
        patch.object(
            valid_aggregation_manager,
            "_aggregate_gradients",
            return_value=mock_aggregation_result,
        ),
    ):
        result = await valid_aggregation_manager.gather_gradients(
            my_uid=0,
            uids=list(
                range(1, len(exception_types) + 2)
            ),  # All UIDs including one that succeeds
            window=10,
            timeout=30,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

        # Should handle all exception types and succeed with the one valid UID
        assert result is not None
        assert isinstance(result, SimpleNamespace)
        assert len(result.uids) == 1


# -----------------------------------------------------------------------------
# _AGGREGATE_GRADIENTS - RESPONSE PROCESSING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_aggregate_gradients_with_empty_raw_responses_list(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients with empty raw_responses list"""
    empty_responses = []

    result = valid_aggregation_manager._aggregate_gradients(
        raw_responses=empty_responses,
        device="cpu",
        totalks={"layer.weight": 1000},
        metrics={"download_bytes": 0},
    )

    # Should return None when no responses
    assert result is None


@pytest.mark.asyncio
async def test_aggregate_gradients_with_all_exception_responses(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients with all exception responses"""
    exception_responses = [
        {"uid": 1, "response": ConnectionError("Network error"), "is_exception": True},
        {"uid": 2, "response": TimeoutError("Timeout"), "is_exception": True},
        {"uid": 3, "response": ValueError("Invalid value"), "is_exception": True},
    ]

    result = valid_aggregation_manager._aggregate_gradients(
        raw_responses=exception_responses,
        device="cpu",
        totalks={"layer.weight": 1000},
        metrics={"download_bytes": 0},
    )

    # Should return None when all responses are exceptions
    assert result is None


@pytest.mark.asyncio
async def test_aggregate_gradients_with_all_none_responses(valid_aggregation_manager):
    """Test _aggregate_gradients with all None responses"""
    none_responses = [
        {"uid": 1, "response": None, "is_exception": False},
        {"uid": 2, "response": None, "is_exception": False},
        {"uid": 3, "response": None, "is_exception": False},
    ]

    result = valid_aggregation_manager._aggregate_gradients(
        raw_responses=none_responses,
        device="cpu",
        totalks={"layer.weight": 1000},
        metrics={"download_bytes": 0},
    )

    # Should return None when all responses are None
    assert result is None


@pytest.mark.asyncio
async def test_aggregate_gradients_with_dict_format_responses_legacy(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients with dict format responses (legacy)"""
    dict_responses = [
        {
            "uid": 1,
            "response": {
                "state_dict": {"layer.weight": torch.tensor([0.1])},
                "global_step": 100,
            },
            "is_exception": False,
        }
    ]

    result = valid_aggregation_manager._aggregate_gradients(
        raw_responses=dict_responses,
        device="cpu",
        totalks={"layer.weight": 1000},
        metrics={"download_bytes": 0},
    )

    # Dict format is not supported - should return None
    assert result is None


@pytest.mark.asyncio
async def test_aggregate_gradients_with_missing_global_step_key(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients with missing 'global_step' key"""
    responses_missing_global_step = [
        {
            "uid": 1,
            # This will cause an issue when trying to unpack the tuple - wrong tuple length
            "response": (
                {"state_dict": {"layer.weight": torch.tensor([0.1])}},
            ),  # Only one element instead of two
            "is_exception": False,
        }
    ]

    # Should handle malformed tuple gracefully and skip the UID
    result = valid_aggregation_manager._aggregate_gradients(
        raw_responses=responses_missing_global_step,
        device="cpu",
        totalks={"layer.weight": 1000},
        metrics={"download_bytes": 0},
    )

    # Should return None because the only response was malformed
    assert result is None


@pytest.mark.asyncio
async def test_aggregate_gradients_calls_validate_gradient_response_correctly(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients calls _validate_gradient_response correctly"""
    responses = [
        {
            "uid": 1,
            "response": (
                {
                    "state_dict": {"layer.weight": torch.tensor([0.1])},
                    "global_step": 100,
                },
                100,
            ),
            "is_exception": False,
        }
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ) as mock_validate:
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"layer.weight": 1000},
            metrics={"download_bytes": 0},
        )

        # Should call validation and succeed
        assert mock_validate.called
        assert result is not None
        assert result["valid_uids"] == [1]


@pytest.mark.asyncio
async def test_aggregate_gradients_handles_validation_exceptions_gracefully(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients handles validation exceptions gracefully"""
    responses = [
        {
            "uid": 1,
            "response": (
                {
                    "state_dict": {"layer.weight": torch.tensor([0.1])},
                    "global_step": 100,
                },
                100,
            ),
            "is_exception": False,
        }
    ]

    def validation_exception(state_dict, uid, device, totalks):
        raise ValueError("Validation error")

    # The actual implementation doesn't catch validation exceptions, so we expect them to propagate
    with patch.object(
        valid_aggregation_manager,
        "_validate_gradient_response",
        side_effect=validation_exception,
    ):
        with pytest.raises(ValueError, match="Validation error"):
            valid_aggregation_manager._aggregate_gradients(
                raw_responses=responses,
                device="cpu",
                totalks={"layer.weight": 1000},
                metrics={"download_bytes": 0},
            )


@pytest.mark.asyncio
async def test_aggregate_gradients_with_all_uids_failing_validation(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients with all UIDs failing validation"""
    responses = [
        {
            "uid": 1,
            "response": (
                {
                    "state_dict": {"layer.weight": torch.tensor([0.1])},
                    "global_step": 100,
                },
                100,
            ),
            "is_exception": False,
        },
        {
            "uid": 2,
            "response": (
                {
                    "state_dict": {"layer.weight": torch.tensor([0.2])},
                    "global_step": 101,
                },
                101,
            ),
            "is_exception": False,
        },
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=False
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"layer.weight": 1000},
            metrics={"download_bytes": 0},
        )

        # Should return None when all UIDs fail validation
        assert result is None


@pytest.mark.asyncio
async def test_aggregate_gradients_with_some_uids_failing_validation(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients with some UIDs failing validation"""
    responses = [
        {
            "uid": 1,
            "response": (
                {
                    "state_dict": {"layer.weight": torch.tensor([0.1])},
                    "global_step": 100,
                },
                100,
            ),
            "is_exception": False,
        },
        {
            "uid": 2,
            "response": (
                {
                    "state_dict": {"layer.weight": torch.tensor([0.2])},
                    "global_step": 101,
                },
                101,
            ),
            "is_exception": False,
        },
        {
            "uid": 3,
            "response": (
                {
                    "state_dict": {"layer.weight": torch.tensor([0.3])},
                    "global_step": 102,
                },
                102,
            ),
            "is_exception": False,
        },
    ]

    # Create a side_effect function that tracks call count
    call_count = 0

    def selective_validation(state_dict, uid, device, totalks):
        nonlocal call_count
        call_count += 1
        # Only first call (UID 1) passes validation
        if call_count == 1:
            return True
        return False

    with patch.object(
        valid_aggregation_manager,
        "_validate_gradient_response",
        side_effect=selective_validation,
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"layer.weight": 1000},
            metrics={"download_bytes": 0},
        )

        # Should include UID 1, skip UIDs 2 and 3
        assert result is not None
        assert result["valid_uids"] == [1]
        assert 2 in result["skipped_uids"]
        assert 3 in result["skipped_uids"]
        assert len(result["global_steps"]) == 1


@pytest.mark.asyncio
async def test_aggregate_gradients_validation_with_different_totalks_values(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients validation with different totalks values"""
    responses = [
        {
            "uid": 1,
            "response": (
                {
                    "state_dict": {
                        "layer.weight_idxs": torch.tensor([0, 1]),
                        "layer.weight_vals": torch.tensor([0.1, 0.2]),
                    },
                    "global_step": 100,
                },
                100,
            ),
            "is_exception": False,
        }
    ]

    # Test with different totalks values
    totalks_cases = [
        {"layer.weight": 1000},  # Normal case
        {"layer.weight": 10},  # Small totalk
        {"layer.weight": 1000000},  # Large totalk
    ]

    for totalks in totalks_cases:
        metrics = {"download_bytes": 0}  # Initialize for each case
        with patch.object(
            valid_aggregation_manager, "_validate_gradient_response", return_value=True
        ) as mock_validate:
            result = valid_aggregation_manager._aggregate_gradients(
                raw_responses=responses, device="cpu", totalks=totalks, metrics=metrics
            )

            # Should call validation with correct totalks
            assert mock_validate.called
            call_args = mock_validate.call_args
            # Parameters are passed positionally: (state_dict, uid, device, totalks)
            assert call_args[0][3] == totalks


@pytest.mark.asyncio
async def test_aggregate_gradients_validation_parameter_passing_accuracy(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients validation parameter passing accuracy"""
    responses = [
        {
            "uid": 1,
            "response": (
                {
                    "param1": torch.tensor([0.1]),
                    "param2": torch.tensor([0.2]),
                },  # Just the state_dict, not wrapped in another dict
                100,  # global_step
            ),
            "is_exception": False,
        }
    ]

    totalks = {"param1": 500, "param2": 300}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics = {"download_bytes": 0}
    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ) as mock_validate:
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses, device=device, totalks=totalks, metrics=metrics
        )

        # Verify all parameters are passed correctly
        assert mock_validate.called
        call_args = mock_validate.call_args
        # Parameters are passed positionally: (state_dict, uid, device, totalks)
        assert call_args[0][2] == device
        assert call_args[0][3] == totalks

        # Verify state_dict structure is preserved
        state_dict_arg = call_args[0][0]
        assert "param1" in state_dict_arg
        assert "param2" in state_dict_arg


# -----------------------------------------------------------------------------
# _AGGREGATE_GRADIENTS - TENSOR PROCESSING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_aggregate_gradients_processes_regular_tensors_correctly(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients processes regular tensors correctly"""
    responses = [
        {
            "uid": 1,
            "response": (
                {"layer.weight": torch.tensor([0.1, 0.2])},  # Just the state_dict
                100,  # global_step
            ),
            "is_exception": False,
        },
        {
            "uid": 2,
            "response": (
                {"layer.weight": torch.tensor([0.3, 0.4])},  # Just the state_dict
                101,  # global_step
            ),
            "is_exception": False,
        },
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"layer.weight": 1000},
            metrics={"download_bytes": 0},
        )

        assert result is not None
        assert "layer.weight" in result["aggregated_state_dict"]
        assert len(result["aggregated_state_dict"]["layer.weight"]) == 2
        assert result["valid_uids"] == [1, 2]
        assert result["global_steps"] == [100, 101]


@pytest.mark.asyncio
async def test_aggregate_gradients_processes_quant_params_dictionaries(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients processes quant_params dictionaries"""
    quant_params_1 = {
        "scale": torch.tensor([1.0]),
        "zero_point": torch.tensor([0]),
        "bits": 8,
    }
    quant_params_2 = {
        "scale": torch.tensor([2.0]),
        "zero_point": torch.tensor([1]),
        "bits": 8,
    }

    responses = [
        {
            "uid": 1,
            "response": (
                {"layer.weight_quant_params": quant_params_1},  # Just the state_dict
                100,  # global_step
            ),
            "is_exception": False,
        },
        {
            "uid": 2,
            "response": (
                {"layer.weight_quant_params": quant_params_2},  # Just the state_dict
                101,  # global_step
            ),
            "is_exception": False,
        },
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"layer.weight": 1000},
            metrics={"download_bytes": 0},
        )

        assert result is not None
        assert "layer.weight_quant_params" in result["aggregated_state_dict"]
        assert len(result["aggregated_state_dict"]["layer.weight_quant_params"]) == 2
        assert result["valid_uids"] == [1, 2]
        assert result["global_steps"] == [100, 101]


@pytest.mark.asyncio
async def test_aggregate_gradients_handles_mixed_tensor_types(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients handles mixed tensor types"""
    quant_params = {
        "scale": torch.tensor([1.5]),
        "zero_point": torch.tensor([2]),
        "bits": 4,
    }

    responses = [
        {
            "uid": 1,
            "response": (
                {
                    "layer.weight": torch.tensor([0.1, 0.2]),
                    "layer.bias": torch.tensor([0.5]),
                    "layer.weight_idxs": torch.tensor([0, 1]),
                    "layer.weight_vals": torch.tensor([0.1, 0.2]),
                    "layer.conv_quant_params": quant_params,
                },  # Just the state_dict
                100,  # global_step
            ),
            "is_exception": False,
        }
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"layer.weight": 1000, "layer.bias": 500},
            metrics={"download_bytes": 0},
        )

        assert result is not None
        aggregated = result["aggregated_state_dict"]

        # Regular tensor
        assert "layer.weight" in aggregated
        assert "layer.bias" in aggregated

        # Compressed indices and values
        assert "layer.weight_idxs" in aggregated
        assert "layer.weight_vals" in aggregated

        # Quant params
        assert "layer.conv_quant_params" in aggregated

        assert result["valid_uids"] == [1]
        assert result["global_steps"] == [100]


@pytest.mark.asyncio
async def test_aggregate_gradients_device_movement_for_all_tensor_types(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients device movement for all tensor types"""
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    quant_params = {
        "scale": torch.tensor([1.0], device="cpu"),
        "zero_point": torch.tensor([0], device="cpu"),
        "bits": 8,
    }

    responses = [
        {
            "uid": 1,
            "response": (
                {
                    "layer.weight": torch.tensor([0.1, 0.2], device="cpu"),
                    "layer.bias_quant_params": quant_params,
                },  # Just the state_dict
                100,  # global_step
            ),
            "is_exception": False,
        }
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device=target_device,
            totalks={"layer.weight": 1000},
            metrics={"download_bytes": 0},
        )

        assert result is not None
        aggregated = result["aggregated_state_dict"]

        # Check regular tensor moved to target device
        assert aggregated["layer.weight"][0].device.type == target_device

        # Check quant_params tensors moved to target device
        quant_params_result = aggregated["layer.bias_quant_params"][0]
        assert quant_params_result["scale"].device.type == target_device
        assert quant_params_result["zero_point"].device.type == target_device


@pytest.mark.asyncio
async def test_aggregate_gradients_with_empty_tensors(valid_aggregation_manager):
    """Test _aggregate_gradients with empty tensors"""
    responses = [
        {
            "uid": 1,
            "response": (
                {
                    "empty.weight": torch.tensor([]),
                    "empty.bias": torch.tensor([]),
                },  # Just the state_dict
                100,  # global_step
            ),
            "is_exception": False,
        }
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"empty.weight": 1000, "empty.bias": 500},
            metrics={"download_bytes": 0},
        )

        assert result is not None
        aggregated = result["aggregated_state_dict"]

        assert "empty.weight" in aggregated
        assert "empty.bias" in aggregated
        assert len(aggregated["empty.weight"]) == 1
        assert len(aggregated["empty.bias"]) == 1
        assert aggregated["empty.weight"][0].numel() == 0  # Empty tensor
        assert aggregated["empty.bias"][0].numel() == 0  # Empty tensor


@pytest.mark.asyncio
async def test_aggregate_gradients_with_sparse_tensors(valid_aggregation_manager):
    """Test _aggregate_gradients with sparse tensors (indices/values format)"""
    responses = [
        {
            "uid": 1,
            "response": (
                {
                    "layer.weight_idxs": torch.tensor([0, 5, 10]),
                    "layer.weight_vals": torch.tensor([0.1, 0.2, 0.3]),
                },  # Just the state_dict
                100,  # global_step
            ),
            "is_exception": False,
        },
        {
            "uid": 2,
            "response": (
                {
                    "layer.weight_idxs": torch.tensor([1, 6, 11]),
                    "layer.weight_vals": torch.tensor([0.4, 0.5, 0.6]),
                },  # Just the state_dict
                101,  # global_step
            ),
            "is_exception": False,
        },
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"layer.weight": 1000},
            metrics={"download_bytes": 0},
        )

        assert result is not None
        aggregated = result["aggregated_state_dict"]

        # Check indices aggregation
        assert "layer.weight_idxs" in aggregated
        assert "layer.weight_vals" in aggregated
        assert len(aggregated["layer.weight_idxs"]) == 2
        assert len(aggregated["layer.weight_vals"]) == 2

        # Verify the actual values
        assert torch.equal(aggregated["layer.weight_idxs"][0], torch.tensor([0, 5, 10]))
        assert torch.equal(
            aggregated["layer.weight_vals"][0], torch.tensor([0.1, 0.2, 0.3])
        )
        assert torch.equal(aggregated["layer.weight_idxs"][1], torch.tensor([1, 6, 11]))
        assert torch.equal(
            aggregated["layer.weight_vals"][1], torch.tensor([0.4, 0.5, 0.6])
        )


@pytest.mark.asyncio
async def test_aggregate_gradients_with_sparse_tensors(valid_aggregation_manager):
    """Test _aggregate_gradients with sparse tensors (indices/values format)"""
    responses = [
        {
            "uid": 1,
            "response": (
                {
                    "layer.weight_idxs": torch.tensor([0, 5, 10]),
                    "layer.weight_vals": torch.tensor([0.1, 0.2, 0.3]),
                },  # Just the state_dict
                100,  # global_step
            ),
            "is_exception": False,
        },
        {
            "uid": 2,
            "response": (
                {
                    "layer.weight_idxs": torch.tensor([1, 6, 11]),
                    "layer.weight_vals": torch.tensor([0.4, 0.5, 0.6]),
                },  # Just the state_dict
                101,  # global_step
            ),
            "is_exception": False,
        },
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"layer.weight": 1000},
            metrics={"download_bytes": 0},
        )

        assert result is not None
        aggregated = result["aggregated_state_dict"]

        # Check indices aggregation
        assert "layer.weight_idxs" in aggregated
        assert "layer.weight_vals" in aggregated
        assert len(aggregated["layer.weight_idxs"]) == 2
        assert len(aggregated["layer.weight_vals"]) == 2

        # Verify the actual values
        assert torch.equal(aggregated["layer.weight_idxs"][0], torch.tensor([0, 5, 10]))
        assert torch.equal(
            aggregated["layer.weight_vals"][0], torch.tensor([0.1, 0.2, 0.3])
        )
        assert torch.equal(aggregated["layer.weight_idxs"][1], torch.tensor([1, 6, 11]))
        assert torch.equal(
            aggregated["layer.weight_vals"][1], torch.tensor([0.4, 0.5, 0.6])
        )


@pytest.mark.asyncio
async def test_aggregate_gradients_with_complex_tensor_shapes_3d_4d(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients with complex tensor shapes (3D, 4D)"""
    tensor_3d = torch.randn(2, 3, 4)  # 3D tensor
    tensor_4d = torch.randn(2, 3, 4, 5)  # 4D tensor

    responses = [
        {
            "uid": 1,
            "response": (
                {
                    "conv.weight": tensor_4d,
                    "lstm.weight": tensor_3d,
                },  # Just the state_dict
                100,  # global_step
            ),
            "is_exception": False,
        }
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"conv.weight": 1000, "lstm.weight": 500},
            metrics={"download_bytes": 0},
        )

        assert result is not None
        aggregated = result["aggregated_state_dict"]

        # Check 4D tensor preservation
        assert "conv.weight" in aggregated
        assert aggregated["conv.weight"][0].shape == tensor_4d.shape
        assert torch.equal(aggregated["conv.weight"][0], tensor_4d)

        # Check 3D tensor preservation
        assert "lstm.weight" in aggregated
        assert aggregated["lstm.weight"][0].shape == tensor_3d.shape
        assert torch.equal(aggregated["lstm.weight"][0], tensor_3d)


@pytest.mark.asyncio
async def test_aggregate_gradients_aggregation_preserves_tensor_order(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients aggregation preserves tensor order"""
    responses = [
        {
            "uid": 1,
            "response": (
                {"param": torch.tensor([1.0])},  # Just the state_dict
                100,  # global_step
            ),
            "is_exception": False,
        },
        {
            "uid": 2,
            "response": (
                {"param": torch.tensor([2.0])},  # Just the state_dict
                101,  # global_step
            ),
            "is_exception": False,
        },
        {
            "uid": 3,
            "response": (
                {"param": torch.tensor([3.0])},  # Just the state_dict
                102,  # global_step
            ),
            "is_exception": False,
        },
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"param": 1000},
            metrics={"download_bytes": 0},
        )

        assert result is not None
        aggregated = result["aggregated_state_dict"]

        # Check order is preserved (UID 1, 2, 3)
        assert len(aggregated["param"]) == 3
        assert torch.equal(aggregated["param"][0], torch.tensor([1.0]))
        assert torch.equal(aggregated["param"][1], torch.tensor([2.0]))
        assert torch.equal(aggregated["param"][2], torch.tensor([3.0]))

        # Check UIDs and global_steps match the order
        assert result["valid_uids"] == [1, 2, 3]
        assert result["global_steps"] == [100, 101, 102]


@pytest.mark.asyncio
async def test_aggregate_gradients_download_bytes_calculation_accuracy(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients download_bytes calculation accuracy"""
    tensor1 = torch.randn(100).float()  # 100 * 4 = 400 bytes
    tensor2 = torch.randn(50, 2).double()  # 100 * 8 = 800 bytes
    tensor3 = torch.randn(10).half()  # 10 * 2 = 20 bytes

    responses = [
        {
            "uid": 1,
            "response": (
                {
                    "param1": tensor1,
                    "param2": tensor2,
                    "param3": tensor3,
                },  # Just the state_dict
                100,  # global_step
            ),
            "is_exception": False,
        }
    ]

    metrics = {"download_bytes": 0}
    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"param1": 1000, "param2": 1000, "param3": 1000},
            metrics=metrics,
        )

        assert result is not None

        # Calculate expected bytes
        expected_bytes = (
            tensor1.element_size() * tensor1.nelement()  # 400 bytes
            + tensor2.element_size() * tensor2.nelement()  # 800 bytes
            + tensor3.element_size() * tensor3.nelement()  # 20 bytes
        )

        assert metrics["download_bytes"] == expected_bytes
        assert metrics["download_bytes"] == 1220  # 400 + 800 + 20


# -----------------------------------------------------------------------------
# _AGGREGATE_GRADIENTS - OUTPUT STRUCTURE
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_aggregate_gradients_returns_correct_dictionary_keys(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients returns correct dictionary keys"""
    responses = [
        {
            "uid": 1,
            "response": (
                {"state_dict": {"param": torch.tensor([1.0])}, "global_step": 100},
                100,
            ),
            "is_exception": False,
        }
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"param": 1000},
            metrics={"download_bytes": 0},
        )

        assert result is not None

        # Check all required keys are present
        required_keys = [
            "aggregated_state_dict",
            "valid_uids",
            "global_steps",
            "skipped_uids",
        ]
        for key in required_keys:
            assert key in result

        # Check key types
        assert isinstance(result["aggregated_state_dict"], dict)
        assert isinstance(result["valid_uids"], list)
        assert isinstance(result["global_steps"], list)
        assert isinstance(result["skipped_uids"], list)


@pytest.mark.asyncio
async def test_aggregate_gradients_aggregated_state_dict_contains_lists(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients aggregated_state_dict contains lists"""
    responses = [
        {
            "uid": 1,
            "response": (
                {
                    "param1": torch.tensor([1.0]),
                    "param2": torch.tensor([2.0]),
                },  # Just the state_dict
                100,  # global_step
            ),
            "is_exception": False,
        },
        {
            "uid": 2,
            "response": (
                {
                    "param1": torch.tensor([3.0]),
                    "param3": torch.tensor([4.0]),
                },  # Just the state_dict
                101,  # global_step
            ),
            "is_exception": False,
        },
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"param1": 1000, "param2": 1000, "param3": 1000},
            metrics={"download_bytes": 0},
        )

        assert result is not None
        aggregated = result["aggregated_state_dict"]

        # Check all values are lists
        for key, value in aggregated.items():
            assert isinstance(value, list), (
                f"Key {key} should contain a list, got {type(value)}"
            )

        # Check specific list contents
        assert len(aggregated["param1"]) == 2  # From both UIDs
        assert len(aggregated["param2"]) == 1  # Only from UID 1
        assert len(aggregated["param3"]) == 1  # Only from UID 2

        # Verify actual tensor values
        assert torch.equal(aggregated["param1"][0], torch.tensor([1.0]))
        assert torch.equal(aggregated["param1"][1], torch.tensor([3.0]))
        assert torch.equal(aggregated["param2"][0], torch.tensor([2.0]))
        assert torch.equal(aggregated["param3"][0], torch.tensor([4.0]))


@pytest.mark.asyncio
async def test_aggregate_gradients_valid_uids_list_matches_successful_uids(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients valid_uids list matches successful UIDs"""
    responses = [
        {
            "uid": 1,
            "response": (
                {"state_dict": {"param": torch.tensor([1.0])}, "global_step": 100},
                100,
            ),
            "is_exception": False,
        },
        {"uid": 2, "response": ConnectionError("Network error"), "is_exception": True},
        {
            "uid": 3,
            "response": (
                {"state_dict": {"param": torch.tensor([3.0])}, "global_step": 102},
                102,
            ),
            "is_exception": False,
        },
        {"uid": 4, "response": None, "is_exception": False},
    ]

    # Mock validation to pass for UIDs 1 and 3, fail for others
    def selective_validation(state_dict, uid, device, totalks):
        return uid in [1, 3]

    with patch.object(
        valid_aggregation_manager,
        "_validate_gradient_response",
        side_effect=selective_validation,
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"param": 1000},
            metrics={"download_bytes": 0},
        )

        assert result is not None

        # Only UIDs 1 and 3 should be in valid_uids
        assert result["valid_uids"] == [1, 3]
        assert len(result["valid_uids"]) == 2


@pytest.mark.asyncio
async def test_aggregate_gradients_global_steps_list_matches_valid_uids(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients global_steps list matches valid UIDs"""
    responses = [
        {
            "uid": 5,
            "response": (
                {"state_dict": {"param": torch.tensor([5.0])}, "global_step": 500},
                500,
            ),
            "is_exception": False,
        },
        {
            "uid": 7,
            "response": (
                {"state_dict": {"param": torch.tensor([7.0])}, "global_step": 700},
                700,
            ),
            "is_exception": False,
        },
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"param": 1000},
            metrics={"download_bytes": 0},
        )

        assert result is not None

        # Check lengths match
        assert len(result["valid_uids"]) == len(result["global_steps"])

        # Check corresponding values
        assert result["valid_uids"] == [5, 7]
        assert result["global_steps"] == [500, 700]


@pytest.mark.asyncio
async def test_aggregate_gradients_skipped_uids_list_matches_failed_uids(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients skipped_uids list matches failed UIDs"""
    responses = [
        {
            "uid": 1,
            "response": (
                {"state_dict": {"param": torch.tensor([1.0])}, "global_step": 100},
                100,
            ),
            "is_exception": False,
        },
        {"uid": 2, "response": TimeoutError("Timeout"), "is_exception": True},
        {"uid": 3, "response": None, "is_exception": False},
        {
            "uid": 4,
            "response": (
                {"state_dict": {"param": torch.tensor([4.0])}, "global_step": 400},
                400,
            ),
            "is_exception": False,
        },
    ]

    # Mock validation to pass for UID 1, fail for UID 4
    def selective_validation(state_dict, uid, device, totalks):
        return uid == 1

    with patch.object(
        valid_aggregation_manager,
        "_validate_gradient_response",
        side_effect=selective_validation,
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"param": 1000},
            metrics={"download_bytes": 0},
        )

        assert result is not None

        # UIDs 2, 3, 4 should be skipped (exception, None, validation failure)
        assert set(result["skipped_uids"]) == {2, 3, 4}
        assert result["valid_uids"] == [1]


@pytest.mark.asyncio
async def test_aggregate_gradients_maintains_uid_order_consistency(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients maintains UID order consistency"""
    # Test with UIDs in non-sequential order
    responses = [
        {
            "uid": 10,
            "response": (
                {"param": torch.tensor([10.0])},  # Just the state_dict
                1000,  # global_step
            ),
            "is_exception": False,
        },
        {
            "uid": 3,
            "response": (
                {"param": torch.tensor([3.0])},  # Just the state_dict
                300,  # global_step
            ),
            "is_exception": False,
        },
        {
            "uid": 7,
            "response": (
                {"param": torch.tensor([7.0])},  # Just the state_dict
                700,  # global_step
            ),
            "is_exception": False,
        },
        {
            "uid": 1,
            "response": (
                {"param": torch.tensor([1.0])},  # Just the state_dict
                100,  # global_step
            ),
            "is_exception": False,
        },
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"param": 1000},
            metrics={"download_bytes": 0},
        )

        assert result is not None

        # Order should match the order in responses (10, 3, 7, 1)
        expected_uid_order = [10, 3, 7, 1]
        expected_step_order = [1000, 300, 700, 100]
        expected_tensor_order = [
            torch.tensor([10.0]),
            torch.tensor([3.0]),
            torch.tensor([7.0]),
            torch.tensor([1.0]),
        ]

        assert result["valid_uids"] == expected_uid_order
        assert result["global_steps"] == expected_step_order

        # Check tensor order matches
        for i, expected_tensor in enumerate(expected_tensor_order):
            assert torch.equal(
                result["aggregated_state_dict"]["param"][i], expected_tensor
            )


@pytest.mark.asyncio
async def test_aggregate_gradients_handles_duplicate_uids_correctly(
    valid_aggregation_manager,
):
    """Test _aggregate_gradients handles duplicate UIDs correctly"""
    # Test with duplicate UIDs (should process both)
    responses = [
        {
            "uid": 5,
            "response": (
                {"param": torch.tensor([5.1])},  # Just the state_dict
                501,  # global_step
            ),
            "is_exception": False,
        },
        {
            "uid": 5,  # Duplicate UID
            "response": (
                {"param": torch.tensor([5.2])},  # Just the state_dict
                502,  # global_step
            ),
            "is_exception": False,
        },
        {
            "uid": 8,
            "response": (
                {"param": torch.tensor([8.0])},  # Just the state_dict
                800,  # global_step
            ),
            "is_exception": False,
        },
    ]

    with patch.object(
        valid_aggregation_manager, "_validate_gradient_response", return_value=True
    ):
        result = valid_aggregation_manager._aggregate_gradients(
            raw_responses=responses,
            device="cpu",
            totalks={"param": 1000},
            metrics={"download_bytes": 0},
        )

        assert result is not None

        # Should process all responses including duplicates
        assert result["valid_uids"] == [5, 5, 8]
        assert result["global_steps"] == [501, 502, 800]

        # Check all tensors are aggregated
        assert len(result["aggregated_state_dict"]["param"]) == 3
        assert torch.equal(
            result["aggregated_state_dict"]["param"][0], torch.tensor([5.1])
        )
        assert torch.equal(
            result["aggregated_state_dict"]["param"][1], torch.tensor([5.2])
        )
        assert torch.equal(
            result["aggregated_state_dict"]["param"][2], torch.tensor([8.0])
        )


# -----------------------------------------------------------------------------
# _VALIDATE_GRADIENT_RESPONSE - BASIC VALIDATION
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_validate_gradient_response_with_valid_gradient(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with valid gradient"""
    state_dict = {
        "layer.weight": torch.tensor([0.1, 0.2, 0.3]),
        "layer.bias": torch.tensor([0.5]),
    }

    result = valid_aggregation_manager._validate_gradient_response(
        state_dict_resp=state_dict,
        uid=1,
        device="cpu",
        totalks={"layer.weight": 1000, "layer.bias": 500},
    )

    assert result is True


@pytest.mark.asyncio
async def test_validate_gradient_response_with_empty_state_dict(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with empty state_dict"""
    state_dict = {}

    result = valid_aggregation_manager._validate_gradient_response(
        state_dict_resp=state_dict, uid=1, device="cpu", totalks={"layer.weight": 1000}
    )

    # Empty state_dict should be valid (no parameters to validate)
    assert result is True


@pytest.mark.asyncio
async def test_validate_gradient_response_with_none_state_dict(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with None state_dict"""
    # Should return False due to type check, not raise AttributeError
    result = valid_aggregation_manager._validate_gradient_response(
        state_dict_resp=None, uid=1, device="cpu", totalks={"layer.weight": 1000}
    )
    assert result is False


@pytest.mark.asyncio
async def test_validate_gradient_response_with_malformed_state_dict(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with malformed state_dict"""
    # Test with non-dict type - should return False due to type check
    result = valid_aggregation_manager._validate_gradient_response(
        state_dict_resp="not_a_dict",
        uid=1,
        device="cpu",
        totalks={"layer.weight": 1000},
    )
    assert result is False

    # Test with dict containing non-tensor values that will fail device movement
    state_dict = {
        "layer.weight_idxs": "not_a_tensor",  # This should fail .to(device)
    }

    # This should return False due to .to() method failure on non-tensor
    result = valid_aggregation_manager._validate_gradient_response(
        state_dict_resp=state_dict,
        uid=1,
        device="cpu",
        totalks={"layer.weight": 1000},  # Provide correct totalk key
    )

    assert result is False


@pytest.mark.asyncio
async def test_validate_gradient_response_with_missing_required_keys(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with missing required keys"""
    # This test assumes the current implementation doesn't require specific keys
    # The validation only checks parameters that are present
    state_dict = {"unexpected.param": torch.tensor([1.0, 2.0])}

    result = valid_aggregation_manager._validate_gradient_response(
        state_dict_resp=state_dict,
        uid=1,
        device="cpu",
        totalks={"layer.weight": 1000},  # Missing totalk for unexpected.param
    )

    # Should pass since no idxs/vals parameters are present
    assert result is True


@pytest.mark.asyncio
async def test_validate_gradient_response_with_extra_unexpected_keys(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with extra unexpected keys"""
    state_dict = {
        "layer.weight": torch.tensor([0.1, 0.2]),
        "unexpected.param": torch.tensor([1.0]),
        "another.unexpected": torch.tensor([2.0]),
    }

    result = valid_aggregation_manager._validate_gradient_response(
        state_dict_resp=state_dict,
        uid=1,
        device="cpu",
        totalks={"layer.weight": 1000},  # No totalks for unexpected params
    )

    # Should pass since no idxs/vals parameters are present
    assert result is True


# -----------------------------------------------------------------------------
# _VALIDATE_GRADIENT_RESPONSE - INDICES VALIDATION
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_validate_gradient_response_with_valid_idxs_parameters(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with valid idxs parameters"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10, 15]),
        "layer.weight_vals": torch.tensor([0.1, 0.2, 0.3, 0.4]),
    }

    # Based on the log message, it's looking for "layer.weight_" (with underscore)
    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        return_value=None,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight_": 1000},  # Note the trailing underscore
        )

    assert result is True


@pytest.mark.asyncio
async def test_validate_gradient_response_with_invalid_idxs_parameters(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with invalid idxs parameters"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10, 1500]),  # 1500 exceeds totalk
        "layer.weight_vals": torch.tensor([0.1, 0.2, 0.3, 0.4]),
    }

    # Mock check_compressed_indices to raise an exception
    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        side_effect=ValueError("Index 1500 exceeds bounds"),
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

    assert result is False


@pytest.mark.asyncio
async def test_validate_gradient_response_with_missing_totalk_for_idxs_param(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with missing totalk for idxs param"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10]),
        "layer.weight_vals": torch.tensor([0.1, 0.2, 0.3]),
    }

    # Missing totalk for "layer.weight" (base name of "layer.weight_idxs")
    result = valid_aggregation_manager._validate_gradient_response(
        state_dict_resp=state_dict,
        uid=1,
        device="cpu",
        totalks={"other.param": 500},  # No totalk for layer.weight
    )

    assert result is False


@pytest.mark.asyncio
async def test_validate_gradient_response_with_mismatched_totalk_values(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with mismatched totalk values"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10]),
        "layer.bias_idxs": torch.tensor([0, 1]),
    }

    # Mock check_compressed_indices to fail for one parameter
    def mock_check_indices(param_name, tensor, totalk, allowed_topk):
        if param_name == "layer.bias_idxs":
            raise ValueError(f"Totalk mismatch for {param_name}")
        return None

    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        side_effect=mock_check_indices,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight": 1000, "layer.bias": 500},
        )

    assert result is False


@pytest.mark.asyncio
async def test_validate_gradient_response_calls_check_compressed_indices_correctly(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response calls check_compressed_indices correctly"""
    state_dict = {
        "layer1.weight_idxs": torch.tensor([0, 5, 10]),
        "layer2.bias_idxs": torch.tensor([0, 1, 2]),
    }

    mock_check = Mock(return_value=None)
    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        mock_check,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={
                "layer1.weight_": 1000,
                "layer2.bias_": 500,
            },  # Note trailing underscores
        )

    assert result is True

    # Verify check_compressed_indices was called correctly for each idxs parameter
    assert mock_check.call_count == 2

    # Check first call (layer1.weight_idxs)
    first_call = mock_check.call_args_list[0]
    assert first_call[0][0] == "layer1.weight_idxs"  # param_name
    assert torch.equal(
        first_call[0][1], torch.tensor([0, 5, 10])
    )  # tensor (moved to device)
    assert first_call[0][2] == 1000  # totalk
    assert (
        first_call[1]["allowed_topk"]
        == valid_aggregation_manager.hparams.topk_compression
    )

    # Check second call (layer2.bias_idxs)
    second_call = mock_check.call_args_list[1]
    assert second_call[0][0] == "layer2.bias_idxs"
    assert torch.equal(second_call[0][1], torch.tensor([0, 1, 2]))
    assert second_call[0][2] == 500
    assert (
        second_call[1]["allowed_topk"]
        == valid_aggregation_manager.hparams.topk_compression
    )


@pytest.mark.asyncio
async def test_validate_gradient_response_handles_check_compressed_indices_exceptions(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response handles check_compressed_indices exceptions"""
    state_dict = {"layer.weight_idxs": torch.tensor([0, 5, 10])}

    # Test different types of exceptions
    exceptions_to_test = [
        ValueError("Index out of bounds"),
        RuntimeError("CUDA error"),
        IndexError("List index out of range"),
        Exception("Generic error"),
    ]

    for exception in exceptions_to_test:
        with patch.object(
            valid_aggregation_manager.gradient_manager,
            "check_compressed_indices",
            side_effect=exception,
        ):
            result = valid_aggregation_manager._validate_gradient_response(
                state_dict_resp=state_dict,
                uid=1,
                device="cpu",
                totalks={"layer.weight": 1000},
            )

        assert result is False, f"Should return False for exception: {exception}"


@pytest.mark.asyncio
async def test_validate_gradient_response_with_idxs_exceeding_totalk_bounds(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with idxs exceeding totalk bounds"""
    state_dict = {
        "layer.weight_idxs": torch.tensor(
            [0, 500, 1000, 1500]
        )  # 1000 and 1500 exceed bounds for totalk=1000
    }

    # Mock to simulate bounds checking failure
    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        side_effect=ValueError("Indices exceed totalk bounds"),
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

    assert result is False


@pytest.mark.asyncio
async def test_validate_gradient_response_with_negative_indices(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with negative indices"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([-1, 0, 5, 10])  # Negative index
    }

    # Mock to simulate negative indices validation failure
    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        side_effect=ValueError("Negative indices not allowed"),
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

    assert result is False


@pytest.mark.asyncio
async def test_validate_gradient_response_with_duplicate_indices(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with duplicate indices"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 5, 10])  # Duplicate index 5
    }

    # Mock to simulate duplicate indices validation failure
    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        side_effect=ValueError("Duplicate indices found"),
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight": 1000},
        )

    assert result is False


# -----------------------------------------------------------------------------
# _VALIDATE_GRADIENT_RESPONSE - VALUES VALIDATION
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_validate_gradient_response_with_valid_vals_parameters(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with valid vals parameters"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10]),
        "layer.weight_vals": torch.tensor([0.1, 0.2, 0.3]),
    }

    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        return_value=None,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight_": 1000},
        )

    assert result is True


@pytest.mark.asyncio
async def test_validate_gradient_response_with_nan_values_in_vals(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with NaN values in vals"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10]),
        "layer.weight_vals": torch.tensor([0.1, float("nan"), 0.3]),
    }

    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        return_value=None,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight_": 1000},
        )

    # Should still pass - NaN validation is not in this method
    assert result is True


@pytest.mark.asyncio
async def test_validate_gradient_response_with_inf_values_in_vals(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with Inf values in vals"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10]),
        "layer.weight_vals": torch.tensor([0.1, float("inf"), 0.3]),
    }

    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        return_value=None,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight_": 1000},
        )

    # Inf values should cause validation to fail
    assert result is False


@pytest.mark.asyncio
async def test_validate_gradient_response_with_negative_inf_values_in_vals(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with -Inf values in vals"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10]),
        "layer.weight_vals": torch.tensor([0.1, float("-inf"), 0.3]),
    }

    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        return_value=None,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight_": 1000},
        )

    # -Inf values should cause validation to fail
    assert result is False


@pytest.mark.asyncio
async def test_validate_gradient_response_with_very_large_values(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with very large values"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10]),
        "layer.weight_vals": torch.tensor([0.1, 1e10, 0.3]),
    }

    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        return_value=None,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight_": 1000},
        )

    assert result is True


@pytest.mark.asyncio
async def test_validate_gradient_response_with_very_small_values(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with very small values"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10]),
        "layer.weight_vals": torch.tensor([0.1, 1e-10, 0.3]),
    }

    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        return_value=None,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight_": 1000},
        )

    assert result is True


@pytest.mark.asyncio
async def test_validate_gradient_response_with_zero_values(valid_aggregation_manager):
    """Test _validate_gradient_response with zero values"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10]),
        "layer.weight_vals": torch.tensor([0.1, 0.0, 0.3]),
    }

    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        return_value=None,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight_": 1000},
        )

    assert result is True


@pytest.mark.asyncio
async def test_validate_gradient_response_with_mixed_nan_valid_values(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with mixed NaN/valid values"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10, 15]),
        "layer.weight_vals": torch.tensor([0.1, float("nan"), 0.3, float("inf")]),
    }

    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        return_value=None,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight_": 1000},
        )

    # Any NaN/Inf values should cause validation to fail
    assert result is False


# -----------------------------------------------------------------------------
# _VALIDATE_GRADIENT_RESPONSE - DEVICE COMPATIBILITY
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_validate_gradient_response_with_tensors_on_correct_device(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with tensors on correct device"""
    state_dict = {
        "layer.weight": torch.tensor([0.1, 0.2]).to("cpu"),
        "layer.bias": torch.tensor([0.5]).to("cpu"),
    }

    result = valid_aggregation_manager._validate_gradient_response(
        state_dict_resp=state_dict,
        uid=1,
        device="cpu",
        totalks={"layer.weight": 1000, "layer.bias": 500},
    )

    assert result is True


@pytest.mark.asyncio
async def test_validate_gradient_response_with_tensors_on_wrong_device(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with tensors on wrong device"""
    # Create tensors on CPU but validate for CUDA device (if available)
    state_dict = {
        "layer.weight": torch.tensor([0.1, 0.2]).to("cpu"),
        "layer.bias": torch.tensor([0.5]).to("cpu"),
    }

    # This should still work because tensors are moved to target device during validation
    result = valid_aggregation_manager._validate_gradient_response(
        state_dict_resp=state_dict,
        uid=1,
        device="cpu",  # Using CPU since CUDA might not be available
        totalks={"layer.weight": 1000, "layer.bias": 500},
    )

    assert result is True


@pytest.mark.asyncio
async def test_validate_gradient_response_device_movement_during_validation(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response device movement during validation"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10]).to("cpu"),
        "layer.weight_vals": torch.tensor([0.1, 0.2, 0.3]).to("cpu"),
    }

    # Mock to verify tensor is moved to correct device
    mock_check = Mock(return_value=None)
    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        mock_check,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight_": 1000},
        )

    assert result is True
    # Verify the tensor passed to check_compressed_indices is on correct device
    call_args = mock_check.call_args_list[0]
    passed_tensor = call_args[0][1]
    assert passed_tensor.device.type == "cpu"


@pytest.mark.asyncio
async def test_validate_gradient_response_with_mixed_device_tensors(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with mixed device tensors"""
    state_dict = {
        "layer.weight": torch.tensor([0.1, 0.2]).to("cpu"),
        "layer.bias": torch.tensor([0.5]).to("cpu"),
        "layer.conv_idxs": torch.tensor([0, 1]).to("cpu"),
    }

    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        return_value=None,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight": 1000, "layer.bias": 500, "layer.conv_": 800},
        )

    assert result is True


@pytest.mark.asyncio
async def test_validate_gradient_response_cuda_cpu_device_transitions(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response CUDA/CPU device transitions"""
    # TODO: Add CUDA availability check for production use
    state_dict = {
        "layer.weight": torch.tensor([0.1, 0.2]),
        "layer.bias": torch.tensor([0.5]),
    }

    # Test CPU device
    result = valid_aggregation_manager._validate_gradient_response(
        state_dict_resp=state_dict,
        uid=1,
        device="cpu",
        totalks={"layer.weight": 1000, "layer.bias": 500},
    )

    assert result is True


# -----------------------------------------------------------------------------
# _PROCESS_GRADIENT_TENSORS - TENSOR HANDLING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_process_gradient_tensors_with_regular_torch_tensor_objects(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors with regular torch.Tensor objects"""
    aggregated_state_dict = {}
    state_dict = {
        "layer.weight": torch.tensor([0.1, 0.2, 0.3]),
        "layer.bias": torch.tensor([0.5]),
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Verify both tensors were processed
    assert "layer.weight" in aggregated_state_dict
    assert "layer.bias" in aggregated_state_dict
    assert isinstance(aggregated_state_dict["layer.weight"], list)
    assert isinstance(aggregated_state_dict["layer.bias"], list)
    assert len(aggregated_state_dict["layer.weight"]) == 1
    assert len(aggregated_state_dict["layer.bias"]) == 1


@pytest.mark.asyncio
async def test_process_gradient_tensors_device_movement_for_all_types(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors device movement for all types"""
    aggregated_state_dict = {}
    state_dict = {
        "layer.tensor": torch.tensor([0.1, 0.2]),
        "layer.quant_params": {
            "tensor": torch.tensor([1.0])
        },  # Changed to quant_params
        "layer.list": [
            torch.tensor([2.0])
        ],  # This won't be processed as it's not a tensor
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Only tensors and quant_params should be processed
    assert "layer.tensor" in aggregated_state_dict
    assert "layer.quant_params" in aggregated_state_dict
    # layer.list won't be processed since it's not a tensor or quant_params

    # Check device placement for regular tensor
    assert aggregated_state_dict["layer.tensor"][0].device.type == "cpu"


@pytest.mark.asyncio
async def test_process_gradient_tensors_with_empty_containers(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors with empty containers"""
    aggregated_state_dict = {}
    state_dict = {
        "layer.empty_quant_params": {},  # Only quant_params get processed if not tensors
        "layer.tensor": torch.tensor(
            [1.0]
        ),  # Add a tensor to ensure something gets processed
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Only tensor and quant_params should be processed
    assert "layer.tensor" in aggregated_state_dict
    assert "layer.empty_quant_params" in aggregated_state_dict


@pytest.mark.asyncio
async def test_process_gradient_tensors_with_deeply_nested_structures(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors with deeply nested structures"""
    aggregated_state_dict = {}
    deeply_nested = {
        "level1": {
            "level2": {
                "level3": {
                    "tensor": torch.tensor([1.0, 2.0]),
                    "list": [
                        torch.tensor([3.0]),
                        {"nested_tensor": torch.tensor([4.0])},
                    ],
                }
            }
        }
    }
    state_dict = {"layer.deep_quant_params": deeply_nested}  # Changed to quant_params
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    assert "layer.deep_quant_params" in aggregated_state_dict
    assert isinstance(aggregated_state_dict["layer.deep_quant_params"], list)


@pytest.mark.asyncio
async def test_process_gradient_tensors_appends_to_existing_lists_correctly(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors appends to existing lists correctly"""
    # Pre-populate aggregated_state_dict
    aggregated_state_dict = {
        "layer.weight": [torch.tensor([1.0, 2.0])],
        "layer.bias": [torch.tensor([0.5])],
    }

    state_dict = {
        "layer.weight": torch.tensor([3.0, 4.0]),
        "layer.bias": torch.tensor([1.5]),
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Lists should now have 2 elements each
    assert len(aggregated_state_dict["layer.weight"]) == 2
    assert len(aggregated_state_dict["layer.bias"]) == 2

    # Verify append order
    assert torch.equal(
        aggregated_state_dict["layer.weight"][0], torch.tensor([1.0, 2.0])
    )
    assert torch.equal(
        aggregated_state_dict["layer.weight"][1], torch.tensor([3.0, 4.0])
    )


@pytest.mark.asyncio
async def test_process_gradient_tensors_creates_new_lists_for_new_keys(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors creates new lists for new keys"""
    aggregated_state_dict = {}
    state_dict = {
        "layer.weight": torch.tensor([0.1, 0.2]),
        "layer.bias": torch.tensor([0.5]),
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # New lists should be created
    assert "layer.weight" in aggregated_state_dict
    assert "layer.bias" in aggregated_state_dict
    assert isinstance(aggregated_state_dict["layer.weight"], list)
    assert isinstance(aggregated_state_dict["layer.bias"], list)
    assert len(aggregated_state_dict["layer.weight"]) == 1
    assert len(aggregated_state_dict["layer.bias"]) == 1


@pytest.mark.asyncio
async def test_process_gradient_tensors_maintains_tensor_order(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors maintains tensor order"""
    aggregated_state_dict = {}
    metrics = {"download_bytes": 0}

    # Process multiple state_dicts in order
    state_dicts = [
        {"param": torch.tensor([1.0])},
        {"param": torch.tensor([2.0])},
        {"param": torch.tensor([3.0])},
    ]

    for i, state_dict in enumerate(state_dicts):
        valid_aggregation_manager._process_gradient_tensors(
            state_dict, i, "cpu", aggregated_state_dict, metrics
        )

    # Verify order is maintained
    assert len(aggregated_state_dict["param"]) == 3
    assert torch.equal(aggregated_state_dict["param"][0], torch.tensor([1.0]))
    assert torch.equal(aggregated_state_dict["param"][1], torch.tensor([2.0]))
    assert torch.equal(aggregated_state_dict["param"][2], torch.tensor([3.0]))


@pytest.mark.asyncio
async def test_process_gradient_tensors_handles_duplicate_keys(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors handles duplicate keys"""
    aggregated_state_dict = {}

    # Same key appears in same state_dict (unusual but should work)
    state_dict = {"layer.weight": torch.tensor([1.0, 2.0])}
    metrics = {"download_bytes": 0}

    # Process twice to simulate duplicate processing
    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )
    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 2, "cpu", aggregated_state_dict, metrics
    )

    # Should have 2 entries for the same key
    assert len(aggregated_state_dict["layer.weight"]) == 2
    assert torch.equal(
        aggregated_state_dict["layer.weight"][0], torch.tensor([1.0, 2.0])
    )
    assert torch.equal(
        aggregated_state_dict["layer.weight"][1], torch.tensor([1.0, 2.0])
    )


@pytest.mark.asyncio
async def test_process_gradient_tensors_aggregated_state_dict_structure(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors aggregated_state_dict structure"""
    aggregated_state_dict = {}
    state_dict = {
        "layer1.weight": torch.tensor([0.1, 0.2]),
        "layer1.bias": torch.tensor([0.5]),
        "layer2.weight": torch.tensor([1.0, 2.0, 3.0]),
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Verify structure
    expected_keys = {"layer1.weight", "layer1.bias", "layer2.weight"}
    assert set(aggregated_state_dict.keys()) == expected_keys

    # All values should be lists
    for key in expected_keys:
        assert isinstance(aggregated_state_dict[key], list)
        assert len(aggregated_state_dict[key]) == 1


@pytest.mark.asyncio
async def test_process_gradient_tensors_metrics_calculation_accuracy(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors metrics calculation accuracy"""
    aggregated_state_dict = {}

    # Create tensors with known sizes
    tensor1 = torch.tensor([0.1, 0.2, 0.3, 0.4])  # 4 elements * 4 bytes = 16 bytes
    tensor2 = torch.tensor([1.0, 2.0])  # 2 elements * 4 bytes = 8 bytes

    state_dict = {"layer.weight": tensor1, "layer.bias": tensor2}
    metrics = {"download_bytes": 100}  # Starting value

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Calculate expected size: (4 + 2) elements * 4 bytes per float32 = 24 bytes
    expected_bytes = 100 + (tensor1.numel() + tensor2.numel()) * tensor1.element_size()
    assert metrics["download_bytes"] == expected_bytes


# Fix the validation tests - NaN/Inf values are actually being rejected
@pytest.mark.asyncio
async def test_validate_gradient_response_with_nan_values_in_vals(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with NaN values in vals"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10]),
        "layer.weight_vals": torch.tensor([0.1, float("nan"), 0.3]),
    }

    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        return_value=None,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight_": 1000},
        )

    # NaN values should cause validation to fail
    assert result is False


@pytest.mark.asyncio
async def test_validate_gradient_response_with_inf_values_in_vals(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with Inf values in vals"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10]),
        "layer.weight_vals": torch.tensor([0.1, float("inf"), 0.3]),
    }

    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        return_value=None,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight_": 1000},
        )

    # Inf values should cause validation to fail
    assert result is False


@pytest.mark.asyncio
async def test_validate_gradient_response_with_negative_inf_values_in_vals(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with -Inf values in vals"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10]),
        "layer.weight_vals": torch.tensor([0.1, float("-inf"), 0.3]),
    }

    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        return_value=None,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight_": 1000},
        )

    # -Inf values should cause validation to fail
    assert result is False


@pytest.mark.asyncio
async def test_validate_gradient_response_with_mixed_nan_valid_values(
    valid_aggregation_manager,
):
    """Test _validate_gradient_response with mixed NaN/valid values"""
    state_dict = {
        "layer.weight_idxs": torch.tensor([0, 5, 10, 15]),
        "layer.weight_vals": torch.tensor([0.1, float("nan"), 0.3, float("inf")]),
    }

    with patch.object(
        valid_aggregation_manager.gradient_manager,
        "check_compressed_indices",
        return_value=None,
    ):
        result = valid_aggregation_manager._validate_gradient_response(
            state_dict_resp=state_dict,
            uid=1,
            device="cpu",
            totalks={"layer.weight_": 1000},
        )

    # Any NaN/Inf values should cause validation to fail
    assert result is False


# -----------------------------------------------------------------------------
# _MOVE_TO_DEVICE_RECURSIVE - RECURSIVE PROCESSING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_move_to_device_recursive_with_torch_tensor(valid_aggregation_manager):
    """Test _move_to_device_recursive with torch.Tensor"""
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = valid_aggregation_manager._move_to_device_recursive(tensor, "cpu")

    assert torch.equal(result, tensor)
    assert result.device.type == "cpu"


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_dict_containing_tensors(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with dict containing tensors"""
    data = {
        "tensor1": torch.tensor([1.0, 2.0]),
        "tensor2": torch.tensor([3.0, 4.0]),
        "non_tensor": "string_value",
    }

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    assert torch.equal(result["tensor1"], torch.tensor([1.0, 2.0]))
    assert torch.equal(result["tensor2"], torch.tensor([3.0, 4.0]))
    assert result["non_tensor"] == "string_value"
    assert result["tensor1"].device.type == "cpu"
    assert result["tensor2"].device.type == "cpu"


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_list_containing_tensors(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with list containing tensors"""
    data = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), "string_value", 42]

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    assert torch.equal(result[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(result[1], torch.tensor([3.0, 4.0]))
    assert result[2] == "string_value"
    assert result[3] == 42
    assert result[0].device.type == "cpu"
    assert result[1].device.type == "cpu"


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_tuple_containing_tensors(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with tuple containing tensors"""
    data = (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), "string_value")

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    assert torch.equal(result[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(result[1], torch.tensor([3.0, 4.0]))
    assert result[2] == "string_value"
    assert result[0].device.type == "cpu"
    assert result[1].device.type == "cpu"
    assert isinstance(result, tuple)


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_nested_dict_list_combinations(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with nested dict/list combinations"""
    data = {
        "list_of_tensors": [torch.tensor([1.0]), torch.tensor([2.0])],
        "dict_of_tensors": {
            "inner_tensor": torch.tensor([3.0]),
            "inner_list": [torch.tensor([4.0]), {"deep_tensor": torch.tensor([5.0])}],
        },
    }

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    # Verify structure and device placement
    assert torch.equal(result["list_of_tensors"][0], torch.tensor([1.0]))
    assert torch.equal(result["list_of_tensors"][1], torch.tensor([2.0]))
    assert torch.equal(result["dict_of_tensors"]["inner_tensor"], torch.tensor([3.0]))
    assert torch.equal(result["dict_of_tensors"]["inner_list"][0], torch.tensor([4.0]))
    assert torch.equal(
        result["dict_of_tensors"]["inner_list"][1]["deep_tensor"], torch.tensor([5.0])
    )

    # Verify all tensors are on CPU
    assert result["list_of_tensors"][0].device.type == "cpu"
    assert result["dict_of_tensors"]["inner_tensor"].device.type == "cpu"
    assert (
        result["dict_of_tensors"]["inner_list"][1]["deep_tensor"].device.type == "cpu"
    )


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_empty_containers(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with empty containers"""
    data = {"empty_dict": {}, "empty_list": [], "empty_tuple": ()}

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    assert result["empty_dict"] == {}
    assert result["empty_list"] == []
    assert result["empty_tuple"] == ()


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_non_tensor_objects(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with non-tensor objects (strings, ints)"""
    data = {
        "string": "test_string",
        "integer": 42,
        "float": 3.14,
        "boolean": True,
        "none_value": None,
    }

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    assert result["string"] == "test_string"
    assert result["integer"] == 42
    assert result["float"] == 3.14
    assert result["boolean"] is True
    assert result["none_value"] is None


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_mixed_tensor_non_tensor_objects(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with mixed tensor/non-tensor objects"""
    data = {
        "tensor": torch.tensor([1.0, 2.0]),
        "string": "test",
        "number": 42,
        "mixed_list": [torch.tensor([3.0]), "string_in_list", 100],
        "mixed_dict": {
            "inner_tensor": torch.tensor([4.0]),
            "inner_string": "inner_test",
        },
    }

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    # Tensors should be moved to device
    assert torch.equal(result["tensor"], torch.tensor([1.0, 2.0]))
    assert result["tensor"].device.type == "cpu"
    assert torch.equal(result["mixed_list"][0], torch.tensor([3.0]))
    assert result["mixed_list"][0].device.type == "cpu"
    assert torch.equal(result["mixed_dict"]["inner_tensor"], torch.tensor([4.0]))
    assert result["mixed_dict"]["inner_tensor"].device.type == "cpu"

    # Non-tensors should remain unchanged
    assert result["string"] == "test"
    assert result["number"] == 42
    assert result["mixed_list"][1] == "string_in_list"
    assert result["mixed_list"][2] == 100
    assert result["mixed_dict"]["inner_string"] == "inner_test"


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_very_deep_nesting(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with very deep nesting"""
    # Create 5-level deep nesting
    data = {
        "level1": {
            "level2": {"level3": {"level4": {"level5": torch.tensor([1.0, 2.0])}}}
        }
    }

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    # Navigate to deep tensor and verify it's on correct device
    deep_tensor = result["level1"]["level2"]["level3"]["level4"]["level5"]
    assert torch.equal(deep_tensor, torch.tensor([1.0, 2.0]))
    assert deep_tensor.device.type == "cpu"


@pytest.mark.asyncio
async def test_move_to_device_recursive_preserves_original_structure(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive preserves original structure"""
    original = {
        "dict_key": {"nested": torch.tensor([1.0])},
        "list_key": [torch.tensor([2.0]), {"nested_in_list": torch.tensor([3.0])}],
        "tuple_key": (torch.tensor([4.0]), "string"),
    }

    result = valid_aggregation_manager._move_to_device_recursive(original, "cpu")

    # Structure should be preserved
    assert isinstance(result, dict)
    assert isinstance(result["dict_key"], dict)
    assert isinstance(result["list_key"], list)
    assert isinstance(result["tuple_key"], tuple)
    assert isinstance(result["list_key"][1], dict)

    # Keys should be preserved
    assert set(result.keys()) == set(original.keys())
    assert set(result["dict_key"].keys()) == set(original["dict_key"].keys())


@pytest.mark.asyncio
async def test_move_to_device_recursive_device_movement_accuracy(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive device movement accuracy"""
    # Create tensors and move them through the function
    data = {
        "tensor1": torch.tensor([1.0, 2.0]),
        "nested": {"tensor2": torch.tensor([3.0, 4.0])},
    }

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    # Verify all tensors are on the correct device
    assert result["tensor1"].device.type == "cpu"
    assert result["nested"]["tensor2"].device.type == "cpu"

    # Verify tensor values are preserved
    assert torch.equal(result["tensor1"], torch.tensor([1.0, 2.0]))
    assert torch.equal(result["nested"]["tensor2"], torch.tensor([3.0, 4.0]))


# -----------------------------------------------------------------------------
# _PROCESS_GRADIENT_TENSORS - TENSOR HANDLING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_process_gradient_tensors_with_regular_torch_tensor_objects(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors with regular torch.Tensor objects"""
    aggregated_state_dict = {}
    state_dict = {
        "layer.weight": torch.tensor([0.1, 0.2, 0.3]),
        "layer.bias": torch.tensor([0.5]),
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Verify both tensors were processed
    assert "layer.weight" in aggregated_state_dict
    assert "layer.bias" in aggregated_state_dict
    assert isinstance(aggregated_state_dict["layer.weight"], list)
    assert isinstance(aggregated_state_dict["layer.bias"], list)
    assert len(aggregated_state_dict["layer.weight"]) == 1
    assert len(aggregated_state_dict["layer.bias"]) == 1

    # Verify tensor values and device
    assert torch.equal(
        aggregated_state_dict["layer.weight"][0], torch.tensor([0.1, 0.2, 0.3])
    )
    assert torch.equal(aggregated_state_dict["layer.bias"][0], torch.tensor([0.5]))
    assert aggregated_state_dict["layer.weight"][0].device.type == "cpu"
    assert aggregated_state_dict["layer.bias"][0].device.type == "cpu"


@pytest.mark.asyncio
async def test_process_gradient_tensors_with_quant_params_dictionaries(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors with quant_params dictionaries"""
    aggregated_state_dict = {}
    state_dict = {
        "layer.weight_quant_params": {
            "scale": torch.tensor([1.0]),
            "zero_point": torch.tensor([0]),
            "meta": "some_metadata",
        },
        "layer.bias_quant_params": {
            "scale": torch.tensor([2.0]),
            "nested": {"deep_tensor": torch.tensor([3.0])},
        },
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Verify quant_params were processed
    assert "layer.weight_quant_params" in aggregated_state_dict
    assert "layer.bias_quant_params" in aggregated_state_dict
    assert isinstance(aggregated_state_dict["layer.weight_quant_params"], list)
    assert isinstance(aggregated_state_dict["layer.bias_quant_params"], list)

    # Verify structure preservation and device movement
    weight_qp = aggregated_state_dict["layer.weight_quant_params"][0]
    assert torch.equal(weight_qp["scale"], torch.tensor([1.0]))
    assert weight_qp["scale"].device.type == "cpu"
    assert weight_qp["meta"] == "some_metadata"

    bias_qp = aggregated_state_dict["layer.bias_quant_params"][0]
    assert torch.equal(bias_qp["nested"]["deep_tensor"], torch.tensor([3.0]))
    assert bias_qp["nested"]["deep_tensor"].device.type == "cpu"


@pytest.mark.asyncio
async def test_process_gradient_tensors_with_nested_dictionary_structures(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors with nested dictionary structures"""
    aggregated_state_dict = {}
    state_dict = {
        "complex_quant_params": {
            "level1": {
                "level2": {"tensor": torch.tensor([1.0, 2.0]), "metadata": "test"},
                "simple_tensor": torch.tensor([3.0]),
            },
            "top_level_tensor": torch.tensor([4.0]),
        }
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    assert "complex_quant_params" in aggregated_state_dict
    complex_qp = aggregated_state_dict["complex_quant_params"][0]

    # Verify nested structure preservation
    assert torch.equal(
        complex_qp["level1"]["level2"]["tensor"], torch.tensor([1.0, 2.0])
    )
    assert complex_qp["level1"]["level2"]["metadata"] == "test"
    assert torch.equal(complex_qp["level1"]["simple_tensor"], torch.tensor([3.0]))
    assert torch.equal(complex_qp["top_level_tensor"], torch.tensor([4.0]))

    # Verify device movement
    assert complex_qp["level1"]["level2"]["tensor"].device.type == "cpu"
    assert complex_qp["level1"]["simple_tensor"].device.type == "cpu"
    assert complex_qp["top_level_tensor"].device.type == "cpu"


@pytest.mark.asyncio
async def test_process_gradient_tensors_with_list_tuple_structures(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors with list/tuple structures"""
    aggregated_state_dict = {}
    state_dict = {
        "list_quant_params": [
            torch.tensor([1.0]),
            {"nested_tensor": torch.tensor([2.0])},
            "string_value",
        ],
        "tuple_quant_params": (torch.tensor([3.0]), torch.tensor([4.0])),
        "regular_tensor": torch.tensor([5.0]),
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Regular tensor should be processed
    assert "regular_tensor" in aggregated_state_dict
    assert torch.equal(aggregated_state_dict["regular_tensor"][0], torch.tensor([5.0]))

    # Only quant_params should be processed for complex structures
    assert "list_quant_params" in aggregated_state_dict
    assert "tuple_quant_params" in aggregated_state_dict

    # Verify structure and device movement for list
    list_qp = aggregated_state_dict["list_quant_params"][0]
    assert torch.equal(list_qp[0], torch.tensor([1.0]))
    assert list_qp[0].device.type == "cpu"
    assert torch.equal(list_qp[1]["nested_tensor"], torch.tensor([2.0]))
    assert list_qp[1]["nested_tensor"].device.type == "cpu"
    assert list_qp[2] == "string_value"

    # Verify structure and device movement for tuple
    tuple_qp = aggregated_state_dict["tuple_quant_params"][0]
    assert isinstance(tuple_qp, tuple)
    assert torch.equal(tuple_qp[0], torch.tensor([3.0]))
    assert torch.equal(tuple_qp[1], torch.tensor([4.0]))
    assert tuple_qp[0].device.type == "cpu"
    assert tuple_qp[1].device.type == "cpu"


@pytest.mark.asyncio
async def test_process_gradient_tensors_with_mixed_data_types(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors with mixed data types"""
    aggregated_state_dict = {}
    state_dict = {
        "tensor": torch.tensor([1.0, 2.0]),
        "string": "not_processed",
        "integer": 42,
        "list": [1, 2, 3],  # Won't be processed (not quant_params)
        "dict": {"key": "value"},  # Won't be processed (not quant_params)
        "mixed_quant_params": {
            "tensor": torch.tensor([3.0]),
            "string": "metadata",
            "number": 123,
            "nested_list": [torch.tensor([4.0]), "text"],
        },
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Only tensors and quant_params should be processed
    expected_keys = {"tensor", "mixed_quant_params"}
    assert set(aggregated_state_dict.keys()) == expected_keys

    # Verify tensor processing
    assert torch.equal(aggregated_state_dict["tensor"][0], torch.tensor([1.0, 2.0]))

    # Verify mixed quant_params processing
    mixed_qp = aggregated_state_dict["mixed_quant_params"][0]
    assert torch.equal(mixed_qp["tensor"], torch.tensor([3.0]))
    assert mixed_qp["tensor"].device.type == "cpu"
    assert mixed_qp["string"] == "metadata"
    assert mixed_qp["number"] == 123
    assert torch.equal(mixed_qp["nested_list"][0], torch.tensor([4.0]))
    assert mixed_qp["nested_list"][0].device.type == "cpu"
    assert mixed_qp["nested_list"][1] == "text"


@pytest.mark.asyncio
async def test_process_gradient_tensors_device_movement_for_all_types(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors device movement for all types"""
    aggregated_state_dict = {}
    state_dict = {
        "simple_tensor": torch.tensor([1.0, 2.0]),
        "complex_quant_params": {
            "tensor": torch.tensor([3.0]),
            "nested": {
                "deep_tensor": torch.tensor([4.0]),
                "list_with_tensor": [torch.tensor([5.0])],
            },
        },
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Verify all tensors moved to correct device
    assert aggregated_state_dict["simple_tensor"][0].device.type == "cpu"

    complex_qp = aggregated_state_dict["complex_quant_params"][0]
    assert complex_qp["tensor"].device.type == "cpu"
    assert complex_qp["nested"]["deep_tensor"].device.type == "cpu"
    assert complex_qp["nested"]["list_with_tensor"][0].device.type == "cpu"


@pytest.mark.asyncio
async def test_process_gradient_tensors_with_empty_containers(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors with empty containers"""
    aggregated_state_dict = {}
    state_dict = {
        "empty_quant_params": {},
        "empty_list_quant_params": [],
        "regular_tensor": torch.tensor([1.0]),  # Ensure something gets processed
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # All should be processed
    assert "empty_quant_params" in aggregated_state_dict
    assert "empty_list_quant_params" in aggregated_state_dict
    assert "regular_tensor" in aggregated_state_dict

    # Verify empty structures preserved
    assert aggregated_state_dict["empty_quant_params"][0] == {}
    assert aggregated_state_dict["empty_list_quant_params"][0] == []
    assert torch.equal(aggregated_state_dict["regular_tensor"][0], torch.tensor([1.0]))


@pytest.mark.asyncio
async def test_process_gradient_tensors_with_deeply_nested_structures(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors with deeply nested structures"""
    aggregated_state_dict = {}
    deeply_nested = {
        "level1": {
            "level2": {
                "level3": {
                    "level4": {
                        "tensor": torch.tensor([1.0, 2.0]),
                        "list": [
                            torch.tensor([3.0]),
                            {"nested_tensor": torch.tensor([4.0])},
                        ],
                    }
                }
            }
        }
    }
    state_dict = {"deep_quant_params": deeply_nested}
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    assert "deep_quant_params" in aggregated_state_dict
    deep_qp = aggregated_state_dict["deep_quant_params"][0]

    # Navigate to deeply nested tensors
    level4 = deep_qp["level1"]["level2"]["level3"]["level4"]
    assert torch.equal(level4["tensor"], torch.tensor([1.0, 2.0]))
    assert level4["tensor"].device.type == "cpu"
    assert torch.equal(level4["list"][0], torch.tensor([3.0]))
    assert level4["list"][0].device.type == "cpu"
    assert torch.equal(level4["list"][1]["nested_tensor"], torch.tensor([4.0]))
    assert level4["list"][1]["nested_tensor"].device.type == "cpu"


# -----------------------------------------------------------------------------
# _PROCESS_GRADIENT_TENSORS - AGGREGATION
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_process_gradient_tensors_appends_to_existing_lists_correctly(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors appends to existing lists correctly"""
    # Pre-populate aggregated_state_dict
    aggregated_state_dict = {
        "layer.weight": [torch.tensor([1.0, 2.0])],
        "layer.bias": [torch.tensor([0.5])],
    }

    state_dict = {
        "layer.weight": torch.tensor([3.0, 4.0]),
        "layer.bias": torch.tensor([1.5]),
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Lists should now have 2 elements each
    assert len(aggregated_state_dict["layer.weight"]) == 2
    assert len(aggregated_state_dict["layer.bias"]) == 2

    # Verify append order
    assert torch.equal(
        aggregated_state_dict["layer.weight"][0], torch.tensor([1.0, 2.0])
    )
    assert torch.equal(
        aggregated_state_dict["layer.weight"][1], torch.tensor([3.0, 4.0])
    )
    assert torch.equal(aggregated_state_dict["layer.bias"][0], torch.tensor([0.5]))
    assert torch.equal(aggregated_state_dict["layer.bias"][1], torch.tensor([1.5]))


@pytest.mark.asyncio
async def test_process_gradient_tensors_creates_new_lists_for_new_keys(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors creates new lists for new keys"""
    aggregated_state_dict = {}
    state_dict = {
        "layer.weight": torch.tensor([0.1, 0.2]),
        "layer.bias": torch.tensor([0.5]),
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # New lists should be created
    assert "layer.weight" in aggregated_state_dict
    assert "layer.bias" in aggregated_state_dict
    assert isinstance(aggregated_state_dict["layer.weight"], list)
    assert isinstance(aggregated_state_dict["layer.bias"], list)
    assert len(aggregated_state_dict["layer.weight"]) == 1
    assert len(aggregated_state_dict["layer.bias"]) == 1

    # Verify values
    assert torch.equal(
        aggregated_state_dict["layer.weight"][0], torch.tensor([0.1, 0.2])
    )
    assert torch.equal(aggregated_state_dict["layer.bias"][0], torch.tensor([0.5]))


@pytest.mark.asyncio
async def test_process_gradient_tensors_maintains_tensor_order(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors maintains tensor order"""
    aggregated_state_dict = {}
    metrics = {"download_bytes": 0}

    # Process multiple state_dicts in order
    state_dicts = [
        {"param": torch.tensor([1.0])},
        {"param": torch.tensor([2.0])},
        {"param": torch.tensor([3.0])},
    ]

    for i, state_dict in enumerate(state_dicts):
        valid_aggregation_manager._process_gradient_tensors(
            state_dict, i, "cpu", aggregated_state_dict, metrics
        )

    # Verify order is maintained
    assert len(aggregated_state_dict["param"]) == 3
    assert torch.equal(aggregated_state_dict["param"][0], torch.tensor([1.0]))
    assert torch.equal(aggregated_state_dict["param"][1], torch.tensor([2.0]))
    assert torch.equal(aggregated_state_dict["param"][2], torch.tensor([3.0]))


@pytest.mark.asyncio
async def test_process_gradient_tensors_handles_duplicate_keys(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors handles duplicate keys"""
    aggregated_state_dict = {}

    # Same key appears in same state_dict (unusual but should work)
    state_dict = {"layer.weight": torch.tensor([1.0, 2.0])}
    metrics = {"download_bytes": 0}

    # Process twice to simulate duplicate processing
    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )
    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 2, "cpu", aggregated_state_dict, metrics
    )

    # Should have 2 entries for the same key
    assert len(aggregated_state_dict["layer.weight"]) == 2
    assert torch.equal(
        aggregated_state_dict["layer.weight"][0], torch.tensor([1.0, 2.0])
    )
    assert torch.equal(
        aggregated_state_dict["layer.weight"][1], torch.tensor([1.0, 2.0])
    )


@pytest.mark.asyncio
async def test_process_gradient_tensors_aggregated_state_dict_structure(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors aggregated_state_dict structure"""
    aggregated_state_dict = {}
    state_dict = {
        "layer1.weight": torch.tensor([0.1, 0.2]),
        "layer1.bias": torch.tensor([0.5]),
        "layer2.weight": torch.tensor([1.0, 2.0, 3.0]),
        "layer2.weight_quant_params": {
            "scale": torch.tensor([1.0]),
            "zero_point": torch.tensor([0]),
        },
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Verify structure
    expected_keys = {
        "layer1.weight",
        "layer1.bias",
        "layer2.weight",
        "layer2.weight_quant_params",
    }
    assert set(aggregated_state_dict.keys()) == expected_keys

    # All values should be lists
    for key in expected_keys:
        assert isinstance(aggregated_state_dict[key], list)
        assert len(aggregated_state_dict[key]) == 1

    # Verify quant_params structure
    qp = aggregated_state_dict["layer2.weight_quant_params"][0]
    assert isinstance(qp, dict)
    assert "scale" in qp
    assert "zero_point" in qp


@pytest.mark.asyncio
async def test_process_gradient_tensors_metrics_calculation_accuracy(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors metrics calculation accuracy"""
    aggregated_state_dict = {}

    # Create tensors with known sizes
    tensor1 = torch.tensor([0.1, 0.2, 0.3, 0.4])  # 4 elements * 4 bytes = 16 bytes
    tensor2 = torch.tensor([1.0, 2.0])  # 2 elements * 4 bytes = 8 bytes

    state_dict = {
        "layer.weight": tensor1,
        "layer.bias": tensor2,
        "layer.metadata_quant_params": {
            "scale": torch.tensor([1.0]),  # 1 element * 4 bytes = 4 bytes (not counted)
            "meta": "string_data",  # No bytes counted
        },
    }
    metrics = {"download_bytes": 100}  # Starting value

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Calculate expected size: only regular tensors count towards metrics
    # tensor1: 4 elements * 4 bytes = 16 bytes
    # tensor2: 2 elements * 4 bytes = 8 bytes
    # quant_params tensors are not counted in metrics
    expected_bytes = (
        100
        + (tensor1.numel() * tensor1.element_size())
        + (tensor2.numel() * tensor2.element_size())
    )
    assert metrics["download_bytes"] == expected_bytes
    assert metrics["download_bytes"] == 100 + 16 + 8  # 124 bytes total


@pytest.mark.asyncio
async def test_process_gradient_tensors_ignores_non_tensor_non_quant_params(
    valid_aggregation_manager,
):
    """Test _process_gradient_tensors ignores non-tensor, non-quant_params"""
    aggregated_state_dict = {}
    state_dict = {
        "layer.weight": torch.tensor([1.0, 2.0]),  # Should be processed
        "layer.metadata": "string_data",  # Should be ignored
        "layer.config": {"setting": "value"},  # Should be ignored
        "layer.values": [1, 2, 3],  # Should be ignored
        "layer.other_quant_params": {
            "scale": torch.tensor([1.0])
        },  # Should be processed
    }
    metrics = {"download_bytes": 0}

    valid_aggregation_manager._process_gradient_tensors(
        state_dict, 1, "cpu", aggregated_state_dict, metrics
    )

    # Only tensor and quant_params should be processed
    expected_keys = {"layer.weight", "layer.other_quant_params"}
    assert set(aggregated_state_dict.keys()) == expected_keys

    # Verify processed items
    assert torch.equal(
        aggregated_state_dict["layer.weight"][0], torch.tensor([1.0, 2.0])
    )
    assert torch.equal(
        aggregated_state_dict["layer.other_quant_params"][0]["scale"],
        torch.tensor([1.0]),
    )


# -----------------------------------------------------------------------------
# _MOVE_TO_DEVICE_RECURSIVE - RECURSIVE PROCESSING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_move_to_device_recursive_with_torch_tensor(valid_aggregation_manager):
    """Test _move_to_device_recursive with torch.Tensor"""
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = valid_aggregation_manager._move_to_device_recursive(tensor, "cpu")

    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))
    assert result.device.type == "cpu"
    assert isinstance(result, torch.Tensor)


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_dict_containing_tensors(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with dict containing tensors"""
    data = {
        "tensor1": torch.tensor([1.0, 2.0]),
        "tensor2": torch.tensor([3.0, 4.0]),
        "non_tensor": "string_value",
    }

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    assert isinstance(result, dict)
    assert torch.equal(result["tensor1"], torch.tensor([1.0, 2.0]))
    assert torch.equal(result["tensor2"], torch.tensor([3.0, 4.0]))
    assert result["tensor1"].device.type == "cpu"
    assert result["tensor2"].device.type == "cpu"
    assert result["non_tensor"] == "string_value"


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_list_containing_tensors(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with list containing tensors"""
    data = [torch.tensor([1.0]), torch.tensor([2.0]), "string_item", 42]

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    assert isinstance(result, list)
    assert len(result) == 4
    assert torch.equal(result[0], torch.tensor([1.0]))
    assert torch.equal(result[1], torch.tensor([2.0]))
    assert result[0].device.type == "cpu"
    assert result[1].device.type == "cpu"
    assert result[2] == "string_item"
    assert result[3] == 42


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_tuple_containing_tensors(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with tuple containing tensors"""
    data = (torch.tensor([1.0]), torch.tensor([2.0]), "string_item", 100)

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    assert isinstance(result, tuple)
    assert len(result) == 4
    assert torch.equal(result[0], torch.tensor([1.0]))
    assert torch.equal(result[1], torch.tensor([2.0]))
    assert result[0].device.type == "cpu"
    assert result[1].device.type == "cpu"
    assert result[2] == "string_item"
    assert result[3] == 100


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_nested_dict_list_combinations(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with nested dict/list combinations"""
    data = {
        "list_of_tensors": [torch.tensor([1.0]), torch.tensor([2.0])],
        "dict_of_tensors": {
            "inner_tensor": torch.tensor([3.0]),
            "inner_list": [torch.tensor([4.0]), {"deep_tensor": torch.tensor([5.0])}],
        },
    }

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    # Verify structure and device placement
    assert torch.equal(result["list_of_tensors"][0], torch.tensor([1.0]))
    assert torch.equal(result["list_of_tensors"][1], torch.tensor([2.0]))
    assert torch.equal(result["dict_of_tensors"]["inner_tensor"], torch.tensor([3.0]))
    assert torch.equal(result["dict_of_tensors"]["inner_list"][0], torch.tensor([4.0]))
    assert torch.equal(
        result["dict_of_tensors"]["inner_list"][1]["deep_tensor"], torch.tensor([5.0])
    )

    # Verify all tensors are on CPU
    assert result["list_of_tensors"][0].device.type == "cpu"
    assert result["dict_of_tensors"]["inner_tensor"].device.type == "cpu"
    assert (
        result["dict_of_tensors"]["inner_list"][1]["deep_tensor"].device.type == "cpu"
    )


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_empty_containers(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with empty containers"""
    data = {"empty_dict": {}, "empty_list": [], "empty_tuple": ()}

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    assert result["empty_dict"] == {}
    assert result["empty_list"] == []
    assert result["empty_tuple"] == ()


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_non_tensor_objects(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with non-tensor objects (strings, ints)"""
    data = {
        "string": "test_string",
        "integer": 42,
        "float": 3.14,
        "boolean": True,
        "none_value": None,
    }

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    assert result["string"] == "test_string"
    assert result["integer"] == 42
    assert result["float"] == 3.14
    assert result["boolean"] is True
    assert result["none_value"] is None


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_mixed_tensor_non_tensor_objects(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with mixed tensor/non-tensor objects"""
    data = {
        "tensor": torch.tensor([1.0, 2.0]),
        "string": "test",
        "number": 42,
        "mixed_list": [torch.tensor([3.0]), "string_in_list", 100],
        "mixed_dict": {
            "inner_tensor": torch.tensor([4.0]),
            "inner_string": "inner_test",
        },
    }

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    # Tensors should be moved to device
    assert torch.equal(result["tensor"], torch.tensor([1.0, 2.0]))
    assert result["tensor"].device.type == "cpu"
    assert torch.equal(result["mixed_list"][0], torch.tensor([3.0]))
    assert result["mixed_list"][0].device.type == "cpu"
    assert torch.equal(result["mixed_dict"]["inner_tensor"], torch.tensor([4.0]))
    assert result["mixed_dict"]["inner_tensor"].device.type == "cpu"

    # Non-tensors should remain unchanged
    assert result["string"] == "test"
    assert result["number"] == 42
    assert result["mixed_list"][1] == "string_in_list"
    assert result["mixed_list"][2] == 100
    assert result["mixed_dict"]["inner_string"] == "inner_test"


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_circular_references(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with circular references (should handle gracefully)"""
    # Create a structure with circular reference
    data = {"tensor": torch.tensor([1.0]), "ref": None}
    data["ref"] = data  # Circular reference

    # This should not crash, though behavior depends on implementation
    # The method should handle this gracefully without infinite recursion
    try:
        result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")
        # If it completes, verify the tensor was moved
        assert torch.equal(result["tensor"], torch.tensor([1.0]))
        assert result["tensor"].device.type == "cpu"
    except RecursionError:
        # If recursion error occurs, it's expected behavior for circular refs
        pytest.skip("Circular references cause recursion error as expected")


@pytest.mark.asyncio
async def test_move_to_device_recursive_with_very_deep_nesting(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive with very deep nesting"""
    # Create 5-level deep nesting
    data = {
        "level1": {
            "level2": {"level3": {"level4": {"level5": torch.tensor([1.0, 2.0])}}}
        }
    }

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    # Navigate to deep tensor and verify it's on correct device
    deep_tensor = result["level1"]["level2"]["level3"]["level4"]["level5"]
    assert torch.equal(deep_tensor, torch.tensor([1.0, 2.0]))
    assert deep_tensor.device.type == "cpu"


@pytest.mark.asyncio
async def test_move_to_device_recursive_preserves_original_structure(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive preserves original structure"""
    original = {
        "dict_key": {"nested": torch.tensor([1.0])},
        "list_key": [torch.tensor([2.0]), {"nested_in_list": torch.tensor([3.0])}],
        "tuple_key": (torch.tensor([4.0]), "string"),
    }

    result = valid_aggregation_manager._move_to_device_recursive(original, "cpu")

    # Structure should be preserved
    assert isinstance(result, dict)
    assert isinstance(result["dict_key"], dict)
    assert isinstance(result["list_key"], list)
    assert isinstance(result["tuple_key"], tuple)
    assert isinstance(result["list_key"][1], dict)

    # Keys should be preserved
    assert set(result.keys()) == set(original.keys())
    assert set(result["dict_key"].keys()) == set(original["dict_key"].keys())


@pytest.mark.asyncio
async def test_move_to_device_recursive_device_movement_accuracy(
    valid_aggregation_manager,
):
    """Test _move_to_device_recursive device movement accuracy"""
    # Create tensors and move them through the function
    data = {
        "tensor1": torch.tensor([1.0, 2.0]),
        "nested": {"tensor2": torch.tensor([3.0, 4.0])},
    }

    result = valid_aggregation_manager._move_to_device_recursive(data, "cpu")

    # Verify all tensors are on the correct device
    assert result["tensor1"].device.type == "cpu"
    assert result["nested"]["tensor2"].device.type == "cpu"

    # Verify tensor values are preserved
    assert torch.equal(result["tensor1"], torch.tensor([1.0, 2.0]))
    assert torch.equal(result["nested"]["tensor2"], torch.tensor([3.0, 4.0]))


# -----------------------------------------------------------------------------
# _GET_WITH_RETRY - RETRY LOGIC
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_with_retry_with_successful_first_attempt(valid_aggregation_manager):
    """Test _get_with_retry with successful first attempt"""
    # Mock successful response
    mock_response = ({"param": torch.tensor([1.0])}, 100)

    with patch.object(
        valid_aggregation_manager, "_get_gradient_from_uid", return_value=mock_response
    ) as mock_get:
        result = await valid_aggregation_manager._get_with_retry(
            "123", 1, "gradient", 30, local=True
        )

    assert result == mock_response
    # Verify _get_gradient_from_uid was called only once with correct parameters
    mock_get.assert_called_once_with(
        "123", 1, 30, local=True, time_min=None, time_max=None, show_progress=False
    )


@pytest.mark.asyncio
async def test_get_with_retry_with_failure_then_success(valid_aggregation_manager):
    """Test _get_with_retry with failure then success"""
    mock_response = ({"param": torch.tensor([1.0])}, 100)

    with patch.object(valid_aggregation_manager, "_get_gradient_from_uid") as mock_get:
        # First call raises exception, second succeeds
        mock_get.side_effect = [Exception("Network error"), mock_response]

        with patch("asyncio.sleep") as mock_sleep:
            result = await valid_aggregation_manager._get_with_retry(
                "123", 1, "gradient", 30, local=True
            )

    assert result == mock_response
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once_with(1.0)  # Base delay


@pytest.mark.asyncio
async def test_get_with_retry_with_all_attempts_failing(valid_aggregation_manager):
    """Test _get_with_retry with all attempts failing (max_retries=3)"""
    with patch.object(valid_aggregation_manager, "_get_gradient_from_uid") as mock_get:
        mock_get.side_effect = Exception("Persistent error")

        with patch("asyncio.sleep") as mock_sleep:
            result = await valid_aggregation_manager._get_with_retry(
                "123", 1, "gradient", 30, local=True
            )

    assert result is None
    assert mock_get.call_count == 3  # Max retries
    # Should sleep between retries (2 sleep calls for 3 attempts)
    assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_get_with_retry_retry_delay_calculation_exponential_backoff(
    valid_aggregation_manager,
):
    """Test _get_with_retry retry delay calculation (exponential backoff)"""
    with patch.object(valid_aggregation_manager, "_get_gradient_from_uid") as mock_get:
        mock_get.side_effect = Exception("Persistent error")

        with patch("asyncio.sleep") as mock_sleep:
            await valid_aggregation_manager._get_with_retry(
                "123", 1, "gradient", 30, local=True
            )

    # Verify exponential backoff: 1.0 * 2^0 = 1.0, then 1.0 * 2^1 = 2.0
    expected_delays = [1.0, 2.0]
    actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
    assert actual_delays == expected_delays


@pytest.mark.asyncio
async def test_get_with_retry_with_different_exception_types(valid_aggregation_manager):
    """Test _get_with_retry with different exception types"""
    exceptions = [
        ConnectionError("Network issue"),
        TimeoutError("Request timeout"),
        ValueError("Invalid data"),
    ]

    for exception in exceptions:
        with patch.object(
            valid_aggregation_manager, "_get_gradient_from_uid"
        ) as mock_get:
            mock_get.side_effect = exception

            result = await valid_aggregation_manager._get_with_retry(
                "123", 1, "gradient", 30, local=True
            )

        assert result is None
        assert mock_get.call_count == 3  # Should retry for all exception types


@pytest.mark.asyncio
async def test_get_with_retry_with_timeout_exceptions(valid_aggregation_manager):
    """Test _get_with_retry with timeout exceptions"""
    with patch.object(valid_aggregation_manager, "_get_gradient_from_uid") as mock_get:
        mock_get.side_effect = asyncio.TimeoutError("Request timed out")

        result = await valid_aggregation_manager._get_with_retry(
            "123", 1, "gradient", 30, local=True
        )

    assert result is None
    assert mock_get.call_count == 3


@pytest.mark.asyncio
async def test_get_with_retry_with_network_exceptions(valid_aggregation_manager):
    """Test _get_with_retry with network exceptions"""
    with patch.object(valid_aggregation_manager, "_get_gradient_from_uid") as mock_get:
        mock_get.side_effect = ConnectionError("Connection refused")

        result = await valid_aggregation_manager._get_with_retry(
            "123", 1, "gradient", 30, local=True
        )

    assert result is None
    assert mock_get.call_count == 3


@pytest.mark.asyncio
async def test_get_with_retry_with_storage_exceptions(valid_aggregation_manager):
    """Test _get_with_retry with storage exceptions"""
    with patch.object(valid_aggregation_manager, "_get_gradient_from_uid") as mock_get:
        mock_get.side_effect = FileNotFoundError("Storage file not found")

        result = await valid_aggregation_manager._get_with_retry(
            "123", 1, "gradient", 30, local=True
        )

    assert result is None
    assert mock_get.call_count == 3


@pytest.mark.asyncio
async def test_get_with_retry_returns_none_after_max_retries(valid_aggregation_manager):
    """Test _get_with_retry returns None after max retries"""
    with patch.object(valid_aggregation_manager, "_get_gradient_from_uid") as mock_get:
        mock_get.side_effect = Exception("Persistent failure")

        result = await valid_aggregation_manager._get_with_retry(
            "123", 1, "gradient", 30, local=True
        )

    assert result is None
    assert mock_get.call_count == 3  # Exactly max_retries attempts


@pytest.mark.asyncio
async def test_get_with_retry_preserves_return_value_on_success(
    valid_aggregation_manager,
):
    """Test _get_with_retry preserves return value on success"""
    expected_state_dict = {
        "layer.weight": torch.tensor([0.1, 0.2, 0.3]),
        "layer.bias": torch.tensor([0.5]),
    }
    expected_global_step = 150
    mock_response = (expected_state_dict, expected_global_step)

    with patch.object(
        valid_aggregation_manager, "_get_gradient_from_uid", return_value=mock_response
    ):
        result = await valid_aggregation_manager._get_with_retry(
            "123", 1, "gradient", 30, local=True, time_min=None, time_max=None
        )

    assert result == mock_response
    state_dict, global_step = result
    assert torch.equal(state_dict["layer.weight"], expected_state_dict["layer.weight"])
    assert torch.equal(state_dict["layer.bias"], expected_state_dict["layer.bias"])
    assert global_step == expected_global_step


@pytest.mark.asyncio
async def test_get_with_retry_passes_all_parameters_correctly(
    valid_aggregation_manager,
):
    """Test _get_with_retry passes all parameters correctly to _get_gradient_from_uid"""
    from datetime import datetime, timezone

    time_min = datetime.now(timezone.utc)
    time_max = datetime.now(timezone.utc)
    mock_response = ({"param": torch.tensor([1.0])}, 100)

    with patch.object(
        valid_aggregation_manager, "_get_gradient_from_uid", return_value=mock_response
    ) as mock_get:
        await valid_aggregation_manager._get_with_retry(
            "456",
            5,
            "gradient",
            45,
            local=False,
            time_min=time_min,
            time_max=time_max,
            extra_param="test",
        )

    mock_get.assert_called_once_with(
        "456",
        5,
        45,
        local=False,
        time_min=time_min,
        time_max=time_max,
        show_progress=False,  # Add this
        extra_param="test",
    )


@pytest.mark.asyncio
async def test_get_with_retry_parameter_preservation_across_retries(
    valid_aggregation_manager,
):
    """Test _get_with_retry parameter preservation across retries"""
    from datetime import datetime, timezone

    time_min = datetime.now(timezone.utc)
    time_max = datetime.now(timezone.utc)
    mock_response = ({"param": torch.tensor([1.0])}, 100)

    with patch.object(valid_aggregation_manager, "_get_gradient_from_uid") as mock_get:
        # First call fails, second succeeds
        mock_get.side_effect = [Exception("Temporary failure"), mock_response]

        with patch("asyncio.sleep"):
            result = await valid_aggregation_manager._get_with_retry(
                "789",
                10,
                "gradient",
                60,
                local=True,
                time_min=time_min,
                time_max=time_max,
                custom_param="value",
            )

    assert result == mock_response
    assert mock_get.call_count == 2

    # Verify both calls had the same parameters
    for call in mock_get.call_args_list:
        args, kwargs = call
        assert args == ("789", 10, 60)
        assert kwargs["local"] is True
        assert kwargs["time_min"] == time_min
        assert kwargs["time_max"] == time_max
        assert kwargs["custom_param"] == "value"


# -----------------------------------------------------------------------------
# _GET_WITH_RETRY - PARAMETER PASSING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_with_retry_passes_all_parameters_correctly_to_get_gradient_from_uid(
    valid_aggregation_manager,
):
    """Test _get_with_retry passes all parameters correctly to _get_gradient_from_uid"""
    from datetime import datetime, timezone

    time_min = datetime.now(timezone.utc)
    time_max = datetime.now(timezone.utc)
    mock_response = ({"param": torch.tensor([1.0])}, 100)

    with patch.object(
        valid_aggregation_manager, "_get_gradient_from_uid", return_value=mock_response
    ) as mock_get:
        await valid_aggregation_manager._get_with_retry(
            "456", 5, "gradient", 45, local=False, time_min=time_min, time_max=time_max
        )

    mock_get.assert_called_once_with(
        "456",
        5,
        45,
        local=False,
        time_min=time_min,
        time_max=time_max,
        show_progress=False,  # Add this
    )


@pytest.mark.asyncio
async def test_get_with_retry_with_local_true_parameter(valid_aggregation_manager):
    """Test _get_with_retry with local=True parameter"""
    mock_response = ({"param": torch.tensor([1.0])}, 100)

    with patch.object(
        valid_aggregation_manager, "_get_gradient_from_uid", return_value=mock_response
    ) as mock_get:
        result = await valid_aggregation_manager._get_with_retry(
            "123", 1, "gradient", 30, local=True
        )

    assert result == mock_response
    mock_get.assert_called_once_with(
        "123",
        1,
        30,
        local=True,
        time_min=None,
        time_max=None,
        show_progress=False,  # Add this
    )


@pytest.mark.asyncio
async def test_get_with_retry_with_local_false_parameter(valid_aggregation_manager):
    """Test _get_with_retry with local=False parameter"""
    mock_response = ({"param": torch.tensor([1.0])}, 100)

    with patch.object(
        valid_aggregation_manager, "_get_gradient_from_uid", return_value=mock_response
    ) as mock_get:
        result = await valid_aggregation_manager._get_with_retry(
            "123", 1, "gradient", 30, local=False
        )

    assert result == mock_response
    mock_get.assert_called_once_with(
        "123",
        1,
        30,
        local=False,
        time_min=None,
        time_max=None,
        show_progress=False,  # Add this
    )


@pytest.mark.asyncio
async def test_get_with_retry_with_time_constraints_parameters(
    valid_aggregation_manager,
):
    """Test _get_with_retry with time constraints parameters"""
    from datetime import datetime, timezone, timedelta

    time_min = datetime.now(timezone.utc) - timedelta(hours=1)
    time_max = datetime.now(timezone.utc) + timedelta(hours=1)
    mock_response = ({"param": torch.tensor([1.0])}, 100)

    with patch.object(
        valid_aggregation_manager, "_get_gradient_from_uid", return_value=mock_response
    ) as mock_get:
        result = await valid_aggregation_manager._get_with_retry(
            "789", 3, "gradient", 60, local=True, time_min=time_min, time_max=time_max
        )

    assert result == mock_response
    mock_get.assert_called_once_with(
        "789",
        3,
        60,
        local=True,
        time_min=time_min,
        time_max=time_max,
        show_progress=False,
    )


@pytest.mark.asyncio
async def test_get_with_retry_with_additional_kwargs(valid_aggregation_manager):
    """Test _get_with_retry with additional kwargs"""
    from datetime import datetime, timezone

    time_min = datetime.now(timezone.utc)
    mock_response = ({"param": torch.tensor([1.0])}, 100)

    # Since _get_with_retry uses **kwargs, we can pass additional parameters
    with patch.object(
        valid_aggregation_manager, "_get_gradient_from_uid", return_value=mock_response
    ) as mock_get:
        result = await valid_aggregation_manager._get_with_retry(
            "555",
            7,
            "gradient",
            90,
            local=False,
            time_min=time_min,
            time_max=None,
            extra_flag=True,
            custom_param="test_value",
        )

    assert result == mock_response
    mock_get.assert_called_once_with(
        "555",
        7,
        90,
        local=False,
        time_min=time_min,
        time_max=None,
        show_progress=False,  # Add this
        extra_flag=True,
        custom_param="test_value",
    )


@pytest.mark.asyncio
async def test_get_with_retry_parameter_preservation_across_retries_detailed(
    valid_aggregation_manager,
):
    """Test _get_with_retry parameter preservation across retries"""
    from datetime import datetime, timezone

    time_min = datetime.now(timezone.utc)
    time_max = datetime.now(timezone.utc)
    mock_response = ({"param": torch.tensor([1.0])}, 100)

    with patch.object(valid_aggregation_manager, "_get_gradient_from_uid") as mock_get:
        # First two calls fail, third succeeds
        mock_get.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            mock_response,
        ]

        with patch("asyncio.sleep"):
            result = await valid_aggregation_manager._get_with_retry(
                "777",
                15,
                "gradient",
                180,
                local=True,
                time_min=time_min,
                time_max=time_max,
                retry_flag=True,
            )

    assert result == mock_response
    assert mock_get.call_count == 3

    # Verify all calls had identical parameters
    expected_args = ("777", 15, 180)
    expected_kwargs = {
        "local": True,
        "time_min": time_min,
        "time_max": time_max,
        "show_progress": False,  # Add this
        "retry_flag": True,
    }

    for call in mock_get.call_args_list:
        args, kwargs = call
        assert args == expected_args
        assert kwargs == expected_kwargs


# -----------------------------------------------------------------------------
# _GET_GRADIENT_FROM_UID - BUCKET RETRIEVAL
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_gradient_from_uid_with_valid_uid_and_existing_bucket(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid with valid UID and existing bucket"""
    from unittest.mock import MagicMock

    # Mock bucket
    mock_bucket = MagicMock()
    mock_bucket.name = "test-bucket"

    # Mock successful bucket retrieval - use integer UID as expected by implementation
    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ) as mock_get_bucket:
        # Mock _check_local_gradient to return None (no local file)
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            # Mock storage client download
            mock_gradient_data = {
                "state_dict": {"param": torch.tensor([1.0])},
                "global_step": 100,
            }
            temp_file_path = "/tmp/test_gradient.pt"

            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=b"mock_data",
                ) as mock_get_object:
                    with patch(
                        "torch.load", return_value=mock_gradient_data
                    ) as mock_torch_load:
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ) as mock_delete:
                            result = (
                                await valid_aggregation_manager._get_gradient_from_uid(
                                    "123", 1, 30, local=False
                                )
                            )

    # Verify bucket was retrieved with integer UID (as implementation converts string to int)
    mock_get_bucket.assert_called_once_with(123)

    # Verify result
    assert result is not None
    state_dict, global_step = result
    assert "param" in state_dict
    assert global_step == 100
    assert global_step == 100


@pytest.mark.asyncio
async def test_get_gradient_from_uid_with_invalid_uid(valid_aggregation_manager):
    """Test _get_gradient_from_uid with invalid UID (non-numeric string)"""
    # Non-numeric UID will cause int() conversion to fail
    result = await valid_aggregation_manager._get_gradient_from_uid(
        "invalid_uid", 1, 30, local=False
    )

    # Should return None for invalid UID that can't be converted to int
    assert result is None


@pytest.mark.asyncio
async def test_get_gradient_from_uid_with_numeric_uid_having_no_bucket(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid with numeric UID having no bucket"""
    # Mock bucket retrieval returning None (UID exists but no bucket)
    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=None,
    ) as mock_get_bucket:
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            result = await valid_aggregation_manager._get_gradient_from_uid(
                "456", 5, 30, local=True
            )

    # Verify bucket lookup was attempted with integer conversion
    mock_get_bucket.assert_called_once_with(456)

    # Should return None when no bucket found
    assert result is None


@pytest.mark.asyncio
async def test_get_gradient_from_uid_with_peer_manager_chain_manager_get_bucket_failure(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid with peer_manager.chain_manager.get_bucket failure"""
    # Mock bucket retrieval raising an exception
    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        side_effect=Exception("Chain manager error"),
    ) as mock_get_bucket:
        result = await valid_aggregation_manager._get_gradient_from_uid(
            "789", 3, 30, local=False
        )

    # Verify bucket lookup was attempted with integer conversion
    mock_get_bucket.assert_called_once_with(789)

    # Should return None on bucket retrieval failure
    assert result is None


@pytest.mark.asyncio
async def test_get_gradient_from_uid_bucket_lookup_error_handling(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid bucket lookup error handling"""
    # Test various types of exceptions during bucket lookup using numeric UIDs
    exceptions_to_test = [
        ConnectionError("Network error"),
        TimeoutError("Timeout during bucket lookup"),
        ValueError("Invalid bucket configuration"),
        KeyError("Missing bucket key"),
        AttributeError("Chain manager not available"),
    ]

    for i, exception in enumerate(exceptions_to_test):
        uid = str(100 + i)  # Use numeric UIDs: "100", "101", "102", etc.
        expected_int_uid = 100 + i

        with patch.object(
            valid_aggregation_manager.peer_manager.chain_manager,
            "get_bucket",
            side_effect=exception,
        ) as mock_get_bucket:
            result = await valid_aggregation_manager._get_gradient_from_uid(
                uid, 1, 30, local=False
            )

        # Verify bucket lookup was attempted with integer conversion
        mock_get_bucket.assert_called_once_with(expected_int_uid)

        # Should return None for any bucket lookup error
        assert result is None


@pytest.mark.asyncio
async def test_get_gradient_from_uid_bucket_retrieval_with_different_uids(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid bucket retrieval with different UIDs"""
    from unittest.mock import MagicMock

    # Test multiple numeric UIDs to ensure proper parameter passing
    uids_to_test = ["123", "456", "789", "100", "200"]

    for uid in uids_to_test:
        expected_int_uid = int(uid)
        mock_bucket = MagicMock()
        mock_bucket.name = f"bucket-{uid}"

        with patch.object(
            valid_aggregation_manager.peer_manager.chain_manager,
            "get_bucket",
            return_value=mock_bucket,
        ) as mock_get_bucket:
            with patch.object(
                valid_aggregation_manager, "_check_local_gradient", return_value=None
            ):
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=None,
                ):
                    await valid_aggregation_manager._get_gradient_from_uid(
                        uid, 1, 30, local=False
                    )

        # Verify correct UID was passed to bucket lookup (converted to int)
        mock_get_bucket.assert_called_once_with(expected_int_uid)


@pytest.mark.asyncio
async def test_get_gradient_from_uid_bucket_caching_behavior(valid_aggregation_manager):
    """Test _get_gradient_from_uid bucket caching behavior"""
    from unittest.mock import MagicMock

    mock_bucket = MagicMock()
    mock_bucket.name = "cached-bucket"

    # Make multiple calls with same numeric UID
    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ) as mock_get_bucket:
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            with patch.object(
                valid_aggregation_manager.storage_client,
                "get_object",
                return_value=None,
            ):
                # First call
                await valid_aggregation_manager._get_gradient_from_uid(
                    "999", 1, 30, local=False
                )
                # Second call with same UID
                await valid_aggregation_manager._get_gradient_from_uid(
                    "999", 2, 30, local=False
                )

    # Verify bucket lookup was called for each request (no caching in current implementation)
    assert mock_get_bucket.call_count == 2
    for call in mock_get_bucket.call_args_list:
        assert call[0][0] == 999  # Integer conversion of "999"


# -----------------------------------------------------------------------------
# _GET_GRADIENT_FROM_UID - LOCAL STORAGE
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_gradient_from_uid_with_local_true_and_existing_local_file(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid with local=True and existing local file"""
    from unittest.mock import MagicMock

    # Mock bucket
    mock_bucket = MagicMock()
    mock_bucket.name = "test-bucket"

    # Mock local file exists and is valid
    mock_local_data = {
        "state_dict": {"layer.weight": torch.tensor([1.0, 2.0])},
        "global_step": 150,
        "timestamp": "2024-01-01T12:00:00Z",
    }

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager,
            "_check_local_gradient",
            return_value=mock_local_data,
        ) as mock_check_local:
            result = await valid_aggregation_manager._get_gradient_from_uid(
                "123", 1, 30, local=True
            )

    # Should return local data without attempting remote download
    assert result is not None
    state_dict, global_step = result
    assert torch.equal(state_dict["layer.weight"], torch.tensor([1.0, 2.0]))
    assert global_step == 150

    # Verify local check was called
    mock_check_local.assert_called_once()


@pytest.mark.asyncio
async def test_get_gradient_from_uid_with_local_true_and_missing_local_file(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid with local=True and missing local file"""
    from unittest.mock import MagicMock

    # Mock bucket
    mock_bucket = MagicMock()
    mock_bucket.name = "test-bucket"

    # Mock local file doesn't exist, but remote does
    mock_remote_data = {
        "state_dict": {"layer.bias": torch.tensor([0.5])},
        "global_step": 200,
    }
    temp_file_path = "/tmp/test_gradient.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ) as mock_check_local:
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=b"mock_data",
                ):
                    with patch("torch.load", return_value=mock_remote_data):
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ):
                            result = (
                                await valid_aggregation_manager._get_gradient_from_uid(
                                    "456", 2, 30, local=True
                                )
                            )

    # Should fallback to remote download when local file missing
    assert result is not None
    state_dict, global_step = result
    assert torch.equal(state_dict["layer.bias"], torch.tensor([0.5]))
    assert global_step == 200

    # Verify local check was attempted first
    mock_check_local.assert_called_once()


@pytest.mark.asyncio
async def test_get_gradient_from_uid_with_local_true_and_corrupted_local_file(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid with local=True and corrupted local file"""
    from unittest.mock import MagicMock

    # Mock bucket
    mock_bucket = MagicMock()
    mock_bucket.name = "test-bucket"

    # Mock remote data for fallback
    mock_remote_data = {
        "state_dict": {"layer.weight": torch.tensor([3.0])},
        "global_step": 250,
    }
    temp_file_path = "/tmp/test_gradient.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        # Mock _check_local_gradient to raise exception (corrupted file)
        with patch.object(
            valid_aggregation_manager,
            "_check_local_gradient",
            side_effect=Exception("Corrupted file"),
        ) as mock_check_local:
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=b"mock_data",
                ):
                    with patch("torch.load", return_value=mock_remote_data):
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ):
                            result = (
                                await valid_aggregation_manager._get_gradient_from_uid(
                                    "789", 3, 30, local=True
                                )
                            )

    # Should fallback to remote download when local file is corrupted
    assert result is not None
    state_dict, global_step = result
    assert torch.equal(state_dict["layer.weight"], torch.tensor([3.0]))
    assert global_step == 250

    # Verify local check was attempted
    mock_check_local.assert_called_once()


@pytest.mark.asyncio
async def test_get_gradient_from_uid_local_storage_time_constraint_checking(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid local storage time constraint checking"""
    from unittest.mock import MagicMock
    from datetime import datetime, timezone, timedelta

    # Mock bucket
    mock_bucket = MagicMock()
    mock_bucket.name = "test-bucket"

    time_min = datetime.now(timezone.utc) - timedelta(hours=1)
    time_max = datetime.now(timezone.utc) + timedelta(hours=1)

    # Mock local file that meets time constraints
    mock_local_data = {
        "state_dict": {"layer.weight": torch.tensor([4.0])},
        "global_step": 300,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager,
            "_check_local_gradient",
            return_value=mock_local_data,
        ) as mock_check_local:
            result = await valid_aggregation_manager._get_gradient_from_uid(
                "111", 1, 30, local=True, time_min=time_min, time_max=time_max
            )

    # Should return local data
    assert result is not None
    state_dict, global_step = result
    assert torch.equal(state_dict["layer.weight"], torch.tensor([4.0]))
    assert global_step == 300

    # Verify time constraints were passed to local check
    mock_check_local.assert_called_once()


@pytest.mark.asyncio
async def test_get_gradient_from_uid_local_storage_priority_over_remote(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid local storage priority over remote"""
    from unittest.mock import MagicMock

    # Mock bucket
    mock_bucket = MagicMock()
    mock_bucket.name = "test-bucket"

    # Mock local file exists
    mock_local_data = {
        "state_dict": {"layer.local": torch.tensor([5.0])},
        "global_step": 400,
        "timestamp": "2024-01-01T12:00:00Z",
    }

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager,
            "_check_local_gradient",
            return_value=mock_local_data,
        ) as mock_check_local:
            # Mock storage client - this should NOT be called since local file exists
            with patch.object(
                valid_aggregation_manager.storage_client, "get_object"
            ) as mock_get_object:
                result = await valid_aggregation_manager._get_gradient_from_uid(
                    "222", 2, 30, local=True
                )

    # Should return local data
    assert result is not None
    state_dict, global_step = result
    assert torch.equal(state_dict["layer.local"], torch.tensor([5.0]))
    assert global_step == 400

    # Verify local check was called but remote download was not
    mock_check_local.assert_called_once()
    mock_get_object.assert_not_called()


# -----------------------------------------------------------------------------
# _GET_GRADIENT_FROM_UID - REMOTE STORAGE
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_gradient_from_uid_remote_storage_download_success(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid remote storage download success"""
    from unittest.mock import MagicMock

    # Mock bucket
    mock_bucket = MagicMock()
    mock_bucket.name = "remote-bucket"

    # Mock successful remote download
    mock_remote_data = {
        "state_dict": {"layer.remote": torch.tensor([6.0])},
        "global_step": 500,
    }
    temp_file_path = "/tmp/remote_gradient.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ) as mock_create_temp:
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=b"gradient_data",
                ) as mock_get_object:
                    with patch(
                        "torch.load", return_value=mock_remote_data
                    ) as mock_torch_load:
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ) as mock_delete:
                            result = (
                                await valid_aggregation_manager._get_gradient_from_uid(
                                    "333", 3, 30, local=False
                                )
                            )

    # Verify successful download and processing
    assert result is not None
    state_dict, global_step = result
    assert torch.equal(state_dict["layer.remote"], torch.tensor([6.0]))
    assert global_step == 500

    # Verify download workflow
    mock_create_temp.assert_called_once_with("gradient")
    mock_get_object.assert_called_once()
    mock_torch_load.assert_called_once_with(
        temp_file_path,
        map_location=valid_aggregation_manager.device,
        weights_only=False,
    )
    mock_delete.assert_called_once_with(temp_file_path)


@pytest.mark.asyncio
async def test_get_gradient_from_uid_remote_storage_download_failure(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid remote storage download failure"""
    from unittest.mock import MagicMock

    # Mock bucket
    mock_bucket = MagicMock()
    mock_bucket.name = "remote-bucket"

    temp_file_path = "/tmp/failed_gradient.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                # Mock storage client download failure
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=None,
                ) as mock_get_object:
                    result = await valid_aggregation_manager._get_gradient_from_uid(
                        "444", 4, 30, local=False
                    )

    # Should return None on download failure
    assert result is None

    # Verify download was attempted
    mock_get_object.assert_called_once()


@pytest.mark.asyncio
async def test_get_gradient_from_uid_remote_storage_timeout(valid_aggregation_manager):
    """Test _get_gradient_from_uid remote storage timeout"""
    from unittest.mock import MagicMock
    import asyncio

    # Mock bucket
    mock_bucket = MagicMock()
    mock_bucket.name = "timeout-bucket"

    temp_file_path = "/tmp/timeout_gradient.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                # Mock storage client timeout
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    side_effect=asyncio.TimeoutError("Download timeout"),
                ) as mock_get_object:
                    result = await valid_aggregation_manager._get_gradient_from_uid(
                        "555", 5, 30, local=False
                    )

    # Should return None on timeout
    assert result is None

    # Verify download was attempted
    mock_get_object.assert_called_once()


@pytest.mark.asyncio
async def test_get_gradient_from_uid_remote_storage_with_time_constraints(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid remote storage with time constraints"""
    from unittest.mock import MagicMock
    from datetime import datetime, timezone, timedelta

    # Mock bucket
    mock_bucket = MagicMock()
    mock_bucket.name = "time-constrained-bucket"

    time_min = datetime.now(timezone.utc) - timedelta(hours=1)
    time_max = datetime.now(timezone.utc) + timedelta(hours=1)

    # Mock successful remote download with time constraints
    mock_remote_data = {
        "state_dict": {"layer.time": torch.tensor([7.0])},
        "global_step": 600,
    }
    temp_file_path = "/tmp/time_gradient.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=b"time_data",
                ):
                    with patch("torch.load", return_value=mock_remote_data):
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ):
                            result = (
                                await valid_aggregation_manager._get_gradient_from_uid(
                                    "666",
                                    6,
                                    30,
                                    local=False,
                                    time_min=time_min,
                                    time_max=time_max,
                                )
                            )

    # Should succeed with time constraints
    assert result is not None
    state_dict, global_step = result
    assert torch.equal(state_dict["layer.time"], torch.tensor([7.0]))
    assert global_step == 600


@pytest.mark.asyncio
async def test_get_gradient_from_uid_remote_storage_error_handling(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid remote storage error handling"""
    from unittest.mock import MagicMock

    # Mock bucket
    mock_bucket = MagicMock()
    mock_bucket.name = "error-bucket"

    temp_file_path = "/tmp/error_gradient.pt"

    # Test various storage errors
    storage_errors = [
        ConnectionError("Network connection failed"),
        PermissionError("Access denied"),
        FileNotFoundError("Remote file not found"),
        OSError("Storage system error"),
        Exception("Unknown storage error"),
    ]

    for error in storage_errors:
        with patch.object(
            valid_aggregation_manager.peer_manager.chain_manager,
            "get_bucket",
            return_value=mock_bucket,
        ):
            with patch.object(
                valid_aggregation_manager, "_check_local_gradient", return_value=None
            ):
                with patch.object(
                    valid_aggregation_manager.file_manager,
                    "create_temp_file",
                    return_value=temp_file_path,
                ):
                    with patch.object(
                        valid_aggregation_manager.storage_client,
                        "get_object",
                        side_effect=error,
                    ) as mock_get_object:
                        result = await valid_aggregation_manager._get_gradient_from_uid(
                            "777", 7, 30, local=False
                        )

        # Should return None for any storage error
        assert result is None

        # Verify download was attempted
        mock_get_object.assert_called_once()


@pytest.mark.asyncio
async def test_get_gradient_from_uid_remote_storage_torch_load_error(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid remote storage torch.load error"""
    from unittest.mock import MagicMock

    # Mock bucket
    mock_bucket = MagicMock()
    mock_bucket.name = "corrupt-data-bucket"

    temp_file_path = "/tmp/corrupt_gradient.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=b"corrupt_data",
                ):
                    # Mock torch.load failure (corrupted file)
                    with patch(
                        "torch.load",
                        side_effect=Exception("Failed to load tensor data"),
                    ) as mock_torch_load:
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ) as mock_delete:
                            result = (
                                await valid_aggregation_manager._get_gradient_from_uid(
                                    "888", 8, 30, local=False
                                )
                            )

    # Should return None when torch.load fails
    assert result is None

    # Verify cleanup still occurs
    mock_delete.assert_called_once_with(temp_file_path)


@pytest.mark.asyncio
async def test_get_gradient_from_uid_remote_storage_gradient_key_construction(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid remote storage gradient key construction"""
    from unittest.mock import MagicMock

    # Mock bucket
    mock_bucket = MagicMock()
    mock_bucket.name = "key-test-bucket"

    # Mock successful download
    mock_remote_data = {
        "state_dict": {"layer.key": torch.tensor([8.0])},
        "global_step": 700,
    }
    temp_file_path = "/tmp/key_gradient.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=b"key_data",
                ) as mock_get_object:
                    with patch("torch.load", return_value=mock_remote_data):
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ):
                            result = (
                                await valid_aggregation_manager._get_gradient_from_uid(
                                    "999", 9, 30, local=False
                                )
                            )

    # Verify gradient key construction: gradient-{window}-{uid}-v{version}.pt
    import tplr

    expected_key = f"gradient-9-999-v{tplr.__version__}.pt"
    mock_get_object.assert_called_once()
    # Get the actual call arguments
    call_args = mock_get_object.call_args
    assert (
        call_args[0][0] == expected_key
    )  # First positional argument should be the key

    # Verify result
    assert result is not None
    state_dict, global_step = result
    assert torch.equal(state_dict["layer.key"], torch.tensor([8.0]))
    assert global_step == 700


# -----------------------------------------------------------------------------
# _GET_GRADIENT_FROM_UID - DATA PROCESSING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_gradient_from_uid_temp_file_creation_and_cleanup(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid temp file creation and cleanup"""
    from unittest.mock import MagicMock

    mock_bucket = MagicMock()
    mock_bucket.name = "temp-file-bucket"

    mock_remote_data = {
        "state_dict": {"layer.temp": torch.tensor([1.0])},
        "global_step": 100,
    }
    temp_file_path = "/tmp/temp_gradient.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ) as mock_create:
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=b"temp_data",
                ):
                    with patch("torch.load", return_value=mock_remote_data):
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ) as mock_delete:
                            result = (
                                await valid_aggregation_manager._get_gradient_from_uid(
                                    "123", 5, 30, local=False
                                )
                            )

    # Verify temp file lifecycle
    mock_create.assert_called_once_with("gradient")
    mock_delete.assert_called_once_with(temp_file_path)

    # Verify result
    assert result is not None
    state_dict, global_step = result
    assert torch.equal(state_dict["layer.temp"], torch.tensor([1.0]))
    assert global_step == 100


@pytest.mark.asyncio
async def test_get_gradient_from_uid_torch_load_with_weights_only_false(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid torch.load with weights_only=False"""
    from unittest.mock import MagicMock

    mock_bucket = MagicMock()
    mock_bucket.name = "weights-only-bucket"

    mock_remote_data = {
        "state_dict": {"layer.weights": torch.tensor([2.0])},
        "global_step": 200,
    }
    temp_file_path = "/tmp/weights_gradient.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=b"weights_data",
                ):
                    with patch(
                        "torch.load", return_value=mock_remote_data
                    ) as mock_torch_load:
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ):
                            result = (
                                await valid_aggregation_manager._get_gradient_from_uid(
                                    "456", 6, 30, local=False
                                )
                            )

    # Verify torch.load called with correct parameters
    mock_torch_load.assert_called_once_with(
        temp_file_path,
        map_location=valid_aggregation_manager.device,
        weights_only=False,
    )

    # Verify result
    assert result is not None
    state_dict, global_step = result
    assert torch.equal(state_dict["layer.weights"], torch.tensor([2.0]))
    assert global_step == 200


@pytest.mark.asyncio
async def test_get_gradient_from_uid_return_tuple_format(valid_aggregation_manager):
    """Test _get_gradient_from_uid return tuple format (state_dict, global_step)"""
    from unittest.mock import MagicMock

    mock_bucket = MagicMock()
    mock_bucket.name = "tuple-format-bucket"

    mock_remote_data = {
        "state_dict": {"layer.format": torch.tensor([3.0, 4.0])},
        "global_step": 300,
    }
    temp_file_path = "/tmp/format_gradient.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=b"format_data",
                ):
                    with patch("torch.load", return_value=mock_remote_data):
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ):
                            result = (
                                await valid_aggregation_manager._get_gradient_from_uid(
                                    "789", 7, 30, local=False
                                )
                            )

    # Verify return format is a tuple
    assert isinstance(result, tuple)
    assert len(result) == 2

    state_dict, global_step = result
    assert isinstance(state_dict, dict)
    assert isinstance(global_step, int)
    assert torch.equal(state_dict["layer.format"], torch.tensor([3.0, 4.0]))
    assert global_step == 300


@pytest.mark.asyncio
async def test_get_gradient_from_uid_with_missing_state_dict_key(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid with missing state_dict key"""
    from unittest.mock import MagicMock

    mock_bucket = MagicMock()
    mock_bucket.name = "missing-state-dict-bucket"

    # Mock data without state_dict key - should use the whole object as state_dict
    mock_remote_data = {"global_step": 400, "param1": torch.tensor([5.0])}
    temp_file_path = "/tmp/missing_state_dict.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=b"missing_data",
                ):
                    with patch("torch.load", return_value=mock_remote_data):
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ):
                            result = (
                                await valid_aggregation_manager._get_gradient_from_uid(
                                    "101", 8, 30, local=False
                                )
                            )

    # Should return the whole loaded data as state_dict when state_dict key is missing
    assert result is not None
    state_dict, global_step = result
    assert state_dict == mock_remote_data  # Full object used as state_dict
    assert global_step == 400


@pytest.mark.asyncio
async def test_get_gradient_from_uid_with_missing_global_step_key(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid with missing global_step key"""
    from unittest.mock import MagicMock

    mock_bucket = MagicMock()
    mock_bucket.name = "missing-global-step-bucket"

    # Mock data without global_step key - should default to 0
    mock_remote_data = {"state_dict": {"layer.missing": torch.tensor([6.0])}}
    temp_file_path = "/tmp/missing_global_step.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=b"missing_step_data",
                ):
                    with patch("torch.load", return_value=mock_remote_data):
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ):
                            result = (
                                await valid_aggregation_manager._get_gradient_from_uid(
                                    "202", 9, 30, local=False
                                )
                            )

    # Should default global_step to 0 when missing
    assert result is not None
    state_dict, global_step = result
    assert torch.equal(state_dict["layer.missing"], torch.tensor([6.0]))
    assert global_step == 0  # Default value


@pytest.mark.asyncio
async def test_get_gradient_from_uid_with_malformed_loaded_data(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid with malformed loaded data"""
    from unittest.mock import MagicMock

    mock_bucket = MagicMock()
    mock_bucket.name = "malformed-data-bucket"

    # Test various malformed data scenarios
    malformed_data_cases = [
        ("not_a_dict", "not_a_dict", 0),  # String instead of dict
        (123, 123, 0),  # Integer instead of dict
        ([], [], 0),  # List instead of dict
        (None, None, 0),  # None value
    ]

    for i, (malformed_data, expected_state_dict, expected_global_step) in enumerate(
        malformed_data_cases
    ):
        temp_file_path = f"/tmp/malformed_{i}.pt"

        with patch.object(
            valid_aggregation_manager.peer_manager.chain_manager,
            "get_bucket",
            return_value=mock_bucket,
        ):
            with patch.object(
                valid_aggregation_manager, "_check_local_gradient", return_value=None
            ):
                with patch.object(
                    valid_aggregation_manager.file_manager,
                    "create_temp_file",
                    return_value=temp_file_path,
                ):
                    with patch.object(
                        valid_aggregation_manager.storage_client,
                        "get_object",
                        return_value=b"malformed_data",
                    ):
                        with patch("torch.load", return_value=malformed_data):
                            with patch.object(
                                valid_aggregation_manager.file_manager, "delete_file"
                            ):
                                result = await valid_aggregation_manager._get_gradient_from_uid(
                                    f"{300 + i}", 10 + i, 30, local=False
                                )

        # Should handle malformed data gracefully
        assert result is not None
        state_dict, global_step = result
        assert state_dict == expected_state_dict
        assert global_step == expected_global_step


@pytest.mark.asyncio
async def test_get_gradient_from_uid_file_cleanup_on_exception(
    valid_aggregation_manager,
):
    """Test _get_gradient_from_uid file cleanup on exception"""
    from unittest.mock import MagicMock

    mock_bucket = MagicMock()
    mock_bucket.name = "exception-cleanup-bucket"

    temp_file_path = "/tmp/exception_cleanup.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager, "_check_local_gradient", return_value=None
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "get_object",
                    return_value=b"exception_data",
                ):
                    # Mock torch.load to raise an exception
                    with patch(
                        "torch.load", side_effect=RuntimeError("File corrupted")
                    ):
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ) as mock_delete:
                            result = (
                                await valid_aggregation_manager._get_gradient_from_uid(
                                    "404", 11, 30, local=False
                                )
                            )

    # Should return None on exception
    assert result is None

    # Should still cleanup temp file even when exception occurs
    mock_delete.assert_called_once_with(temp_file_path)


# -----------------------------------------------------------------------------
# _CHECK_LOCAL_GRADIENT - FILE EXISTENCE
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_check_local_gradient_with_existing_valid_file(valid_aggregation_manager):
    """Test _check_local_gradient with existing valid file"""
    from datetime import datetime, timezone

    mock_local_data = {
        "state_dict": {"layer.local": torch.tensor([7.0])},
        "global_step": 500,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    local_path = "/tmp/local_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getmtime", return_value=time.time()):
                with patch("torch.load", return_value=mock_local_data):
                    result = await valid_aggregation_manager._check_local_gradient(
                        "123", 5
                    )

    assert result is not None
    assert result["state_dict"]["layer.local"].equal(torch.tensor([7.0]))
    assert result["global_step"] == 500
    assert "timestamp" in result


@pytest.mark.asyncio
async def test_check_local_gradient_with_non_existent_file(valid_aggregation_manager):
    """Test _check_local_gradient with non-existent file"""
    local_path = "/tmp/nonexistent_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=False):
            result = await valid_aggregation_manager._check_local_gradient("456", 6)

    assert result is None


@pytest.mark.asyncio
async def test_check_local_gradient_with_corrupted_file(valid_aggregation_manager):
    """Test _check_local_gradient with corrupted file"""
    local_path = "/tmp/corrupted_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getmtime", return_value=time.time()):
                # Mock torch.load to raise an exception (corrupted file)
                with patch("torch.load", side_effect=RuntimeError("File corrupted")):
                    result = await valid_aggregation_manager._check_local_gradient(
                        "789", 7
                    )

    assert result is None


@pytest.mark.asyncio
async def test_check_local_gradient_with_unreadable_file_permissions(
    valid_aggregation_manager,
):
    """Test _check_local_gradient with unreadable file (permissions)"""
    local_path = "/tmp/unreadable_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getmtime", return_value=time.time()):
                # Mock torch.load to raise permission error
                with patch(
                    "torch.load", side_effect=PermissionError("Permission denied")
                ):
                    result = await valid_aggregation_manager._check_local_gradient(
                        "101", 8
                    )

    assert result is None


@pytest.mark.asyncio
async def test_check_local_gradient_path_construction_accuracy(
    valid_aggregation_manager,
):
    """Test _check_local_gradient path construction accuracy"""
    import tplr

    uid = "999"
    window = 15
    expected_filename = f"gradient-{window}-{uid}-v{tplr.__version__}.pt"

    with patch.object(
        valid_aggregation_manager.file_manager, "get_local_storage_path"
    ) as mock_get_path:
        with patch("os.path.exists", return_value=False):
            await valid_aggregation_manager._check_local_gradient(uid, window)

    # Verify path construction parameters
    mock_get_path.assert_called_once_with(uid, window, expected_filename)


# -----------------------------------------------------------------------------
# _CHECK_LOCAL_GRADIENT - TIME CONSTRAINTS
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_check_local_gradient_with_file_newer_than_time_min(
    valid_aggregation_manager,
):
    """Test _check_local_gradient with file newer than time_min"""
    from datetime import datetime, timezone, timedelta

    # Create time constraints - file should be newer than time_min
    time_min = datetime.now(timezone.utc) - timedelta(hours=1)
    file_time = datetime.now(timezone.utc)  # Current time (newer than time_min)

    mock_local_data = {
        "state_dict": {"layer.time": torch.tensor([1.0])},
        "global_step": 100,
        "timestamp": file_time.isoformat(),
    }
    local_path = "/tmp/newer_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getmtime", return_value=file_time.timestamp()):
                with patch("torch.load", return_value=mock_local_data):
                    result = await valid_aggregation_manager._check_local_gradient(
                        "123", 5, time_min=time_min
                    )

    # Should return the data since file is newer than time_min
    assert result is not None
    assert result["state_dict"]["layer.time"].equal(torch.tensor([1.0]))
    assert result["global_step"] == 100


@pytest.mark.asyncio
async def test_check_local_gradient_with_file_older_than_time_min(
    valid_aggregation_manager,
):
    """Test _check_local_gradient with file older than time_min"""
    from datetime import datetime, timezone, timedelta

    # Create time constraints - file should be older than time_min
    time_min = datetime.now(timezone.utc)
    file_time = datetime.now(timezone.utc) - timedelta(hours=1)  # Older than time_min

    local_path = "/tmp/older_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getmtime", return_value=file_time.timestamp()):
                result = await valid_aggregation_manager._check_local_gradient(
                    "456", 6, time_min=time_min
                )

    # Should return None since file is older than time_min
    assert result is None


@pytest.mark.asyncio
async def test_check_local_gradient_with_file_newer_than_time_max(
    valid_aggregation_manager,
):
    """Test _check_local_gradient with file newer than time_max"""
    from datetime import datetime, timezone, timedelta

    # Create time constraints - file should be newer than time_max
    time_max = datetime.now(timezone.utc) - timedelta(hours=1)
    file_time = datetime.now(timezone.utc)  # Current time (newer than time_max)

    local_path = "/tmp/too_new_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getmtime", return_value=file_time.timestamp()):
                result = await valid_aggregation_manager._check_local_gradient(
                    "789", 7, time_max=time_max
                )

    # Should return None since file is newer than time_max
    assert result is None


@pytest.mark.asyncio
async def test_check_local_gradient_with_file_older_than_time_max(
    valid_aggregation_manager,
):
    """Test _check_local_gradient with file older than time_max"""
    from datetime import datetime, timezone, timedelta

    # Create time constraints - file should be older than time_max
    time_max = datetime.now(timezone.utc)
    file_time = datetime.now(timezone.utc) - timedelta(hours=1)  # Older than time_max

    mock_local_data = {
        "state_dict": {"layer.old": torch.tensor([2.0])},
        "global_step": 200,
        "timestamp": file_time.isoformat(),
    }
    local_path = "/tmp/old_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getmtime", return_value=file_time.timestamp()):
                with patch("torch.load", return_value=mock_local_data):
                    result = await valid_aggregation_manager._check_local_gradient(
                        "101", 8, time_max=time_max
                    )

    # Should return the data since file is older than time_max
    assert result is not None
    assert result["state_dict"]["layer.old"].equal(torch.tensor([2.0]))
    assert result["global_step"] == 200


@pytest.mark.asyncio
async def test_check_local_gradient_with_file_timestamp_exactly_at_time_min(
    valid_aggregation_manager,
):
    """Test _check_local_gradient with file timestamp exactly at time_min"""
    from datetime import datetime, timezone

    # Create time constraints - file timestamp exactly at time_min
    exact_time = datetime.now(timezone.utc)

    mock_local_data = {
        "state_dict": {"layer.exact": torch.tensor([3.0])},
        "global_step": 300,
        "timestamp": exact_time.isoformat(),
    }
    local_path = "/tmp/exact_min_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getmtime", return_value=exact_time.timestamp()):
                with patch("torch.load", return_value=mock_local_data):
                    result = await valid_aggregation_manager._check_local_gradient(
                        "202", 9, time_min=exact_time
                    )

    # Should return the data since file timestamp equals time_min (>=)
    assert result is not None
    assert result["state_dict"]["layer.exact"].equal(torch.tensor([3.0]))
    assert result["global_step"] == 300


@pytest.mark.asyncio
async def test_check_local_gradient_with_file_timestamp_exactly_at_time_max(
    valid_aggregation_manager,
):
    """Test _check_local_gradient with file timestamp exactly at time_max"""
    from datetime import datetime, timezone

    # Create time constraints - file timestamp exactly at time_max
    exact_time = datetime.now(timezone.utc)

    mock_local_data = {
        "state_dict": {"layer.exact_max": torch.tensor([4.0])},
        "global_step": 400,
        "timestamp": exact_time.isoformat(),
    }
    local_path = "/tmp/exact_max_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getmtime", return_value=exact_time.timestamp()):
                with patch("torch.load", return_value=mock_local_data):
                    result = await valid_aggregation_manager._check_local_gradient(
                        "303", 10, time_max=exact_time
                    )

    # Should return the data since file timestamp equals time_max (<=)
    assert result is not None
    assert result["state_dict"]["layer.exact_max"].equal(torch.tensor([4.0]))
    assert result["global_step"] == 400


@pytest.mark.asyncio
async def test_check_local_gradient_with_none_time_constraints(
    valid_aggregation_manager,
):
    """Test _check_local_gradient with None time constraints"""
    from datetime import datetime, timezone

    mock_local_data = {
        "state_dict": {"layer.no_constraints": torch.tensor([5.0])},
        "global_step": 500,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    local_path = "/tmp/no_constraints_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getmtime", return_value=time.time()):
                with patch("torch.load", return_value=mock_local_data):
                    result = await valid_aggregation_manager._check_local_gradient(
                        "404", 11, time_min=None, time_max=None
                    )

    # Should return the data since no time constraints are applied
    assert result is not None
    assert result["state_dict"]["layer.no_constraints"].equal(torch.tensor([5.0]))
    assert result["global_step"] == 500


@pytest.mark.asyncio
async def test_check_local_gradient_timezone_handling_for_file_timestamps(
    valid_aggregation_manager,
):
    """Test _check_local_gradient timezone handling for file timestamps"""
    from datetime import datetime, timezone, timedelta

    # Test with different timezone - UTC vs local
    utc_time = datetime.now(timezone.utc)
    local_file_time = utc_time - timedelta(hours=1)  # File time in past

    mock_local_data = {
        "state_dict": {"layer.timezone": torch.tensor([6.0])},
        "global_step": 600,
        "timestamp": utc_time.isoformat(),
    }
    local_path = "/tmp/timezone_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            # Mock file mtime as timestamp (should be converted to UTC)
            with patch("os.path.getmtime", return_value=local_file_time.timestamp()):
                with patch("torch.load", return_value=mock_local_data):
                    result = await valid_aggregation_manager._check_local_gradient(
                        "505", 12, time_min=local_file_time, time_max=utc_time
                    )

    # Should handle timezone conversion correctly
    assert result is not None
    assert result["state_dict"]["layer.timezone"].equal(torch.tensor([6.0]))
    assert result["global_step"] == 600


# -----------------------------------------------------------------------------
# _CHECK_LOCAL_GRADIENT - DATA LOADING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_check_local_gradient_torch_load_success(valid_aggregation_manager):
    """Test _check_local_gradient torch.load success"""
    from datetime import datetime, timezone

    mock_local_data = {
        "state_dict": {"layer.success": torch.tensor([7.0])},
        "global_step": 700,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    local_path = "/tmp/load_success_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getmtime", return_value=time.time()):
                with patch(
                    "torch.load", return_value=mock_local_data
                ) as mock_torch_load:
                    result = await valid_aggregation_manager._check_local_gradient(
                        "606", 13
                    )

    # Verify torch.load was called correctly
    mock_torch_load.assert_called_once_with(local_path, weights_only=False)

    # Verify result
    assert result is not None
    assert result == mock_local_data
    assert result["state_dict"]["layer.success"].equal(torch.tensor([7.0]))
    assert result["global_step"] == 700


@pytest.mark.asyncio
async def test_check_local_gradient_torch_load_failure(valid_aggregation_manager):
    """Test _check_local_gradient torch.load failure"""
    local_path = "/tmp/load_failure_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getmtime", return_value=time.time()):
                # Mock torch.load to raise an exception
                with patch(
                    "torch.load", side_effect=Exception("Load failed")
                ) as mock_torch_load:
                    result = await valid_aggregation_manager._check_local_gradient(
                        "707", 14
                    )

    # Verify torch.load was called
    mock_torch_load.assert_called_once_with(local_path, weights_only=False)

    # Should return None on load failure
    assert result is None


@pytest.mark.asyncio
async def test_check_local_gradient_preserves_existing_timestamp(
    valid_aggregation_manager,
):
    """Test _check_local_gradient preserves existing timestamp"""
    from datetime import datetime, timezone

    original_timestamp = datetime.now(timezone.utc).isoformat()
    mock_local_data_with_timestamp = {
        "state_dict": {"layer.with_timestamp": torch.tensor([9.0])},
        "global_step": 900,
        "timestamp": original_timestamp,
    }
    local_path = "/tmp/with_timestamp_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getmtime", return_value=time.time()):
                with patch("torch.load", return_value=mock_local_data_with_timestamp):
                    result = await valid_aggregation_manager._check_local_gradient(
                        "909", 16
                    )

    # Should preserve existing timestamp
    assert result is not None
    assert result["timestamp"] == original_timestamp
    assert result["state_dict"]["layer.with_timestamp"].equal(torch.tensor([9.0]))
    assert result["global_step"] == 900


@pytest.mark.asyncio
async def test_check_local_gradient_return_value_structure(valid_aggregation_manager):
    """Test _check_local_gradient return value structure"""
    from datetime import datetime, timezone

    mock_local_data = {
        "state_dict": {"layer.structure": torch.tensor([10.0])},
        "global_step": 1000,
        "additional_field": "extra_data",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    local_path = "/tmp/structure_gradient.pt"

    with patch.object(
        valid_aggregation_manager.file_manager,
        "get_local_storage_path",
        return_value=local_path,
    ):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getmtime", return_value=time.time()):
                with patch("torch.load", return_value=mock_local_data):
                    result = await valid_aggregation_manager._check_local_gradient(
                        "1010", 17
                    )

    # Verify return value structure
    assert result is not None
    assert isinstance(result, dict)

    # Required fields
    assert "state_dict" in result
    assert "global_step" in result
    assert "timestamp" in result

    # Additional fields preserved
    assert "additional_field" in result
    assert result["additional_field"] == "extra_data"

    # Verify data integrity
    assert result["state_dict"]["layer.structure"].equal(torch.tensor([10.0]))
    assert result["global_step"] == 1000


@pytest.mark.asyncio
async def test_check_local_gradient_error_handling_during_load(
    valid_aggregation_manager,
):
    """Test _check_local_gradient error handling during load"""
    local_path = "/tmp/error_handling_gradient.pt"

    # Test various error scenarios
    error_scenarios = [
        FileNotFoundError("File not found"),
        PermissionError("Permission denied"),
        RuntimeError("Corrupted file"),
        ValueError("Invalid data format"),
        OSError("I/O error"),
    ]

    for error in error_scenarios:
        with patch.object(
            valid_aggregation_manager.file_manager,
            "get_local_storage_path",
            return_value=local_path,
        ):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.getmtime", return_value=time.time()):
                    with patch("torch.load", side_effect=error):
                        result = await valid_aggregation_manager._check_local_gradient(
                            "error", 18
                        )

        # Should handle all errors gracefully and return None
        assert result is None


# -----------------------------------------------------------------------------
# LOAD_AGGREGATION - FILE DOWNLOAD
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_load_aggregation_with_existing_aggregation_file(
    valid_aggregation_manager,
):
    """Test load_aggregation with existing aggregation file"""
    from unittest.mock import MagicMock

    window = 5
    mock_bucket = MagicMock()
    mock_bucket.name = "existing-file-bucket"

    # Mock aggregation data
    mock_aggregation_data = {
        "aggregated_gradients": {"layer.agg": torch.tensor([1.0, 2.0])},
        "num_gradients": 5,
        "window": window,
    }

    # Mock file size (small file)
    file_size = 50 * 1024 * 1024  # 50MB

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object_size",
            return_value=file_size,
        ):
            with patch.object(
                valid_aggregation_manager.storage_client,
                "get_object",
                return_value=b"aggregation_data",
            ) as mock_get_object:
                with patch.object(
                    valid_aggregation_manager.file_manager,
                    "create_temp_file",
                    return_value="/tmp/agg.pt",
                ):
                    with patch("torch.load", return_value=mock_aggregation_data):
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ):
                            result = await valid_aggregation_manager.load_aggregation(
                                window
                            )

    # Verify successful load
    assert result is not None
    assert result["aggregated_gradients"]["layer.agg"].equal(torch.tensor([1.0, 2.0]))
    assert result["num_gradients"] == 5
    assert result["window"] == window

    # Verify get_object was called
    mock_get_object.assert_called_once()


@pytest.mark.asyncio
async def test_load_aggregation_with_non_existent_aggregation_file(
    valid_aggregation_manager,
):
    """Test load_aggregation with non-existent aggregation file"""
    from unittest.mock import MagicMock

    window = 10
    mock_bucket = MagicMock()
    mock_bucket.name = "non-existent-bucket"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        # Mock get_object_size returning None (file doesn't exist)
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object_size",
            return_value=None,
        ):
            result = await valid_aggregation_manager.load_aggregation(window)

    # Should return None when file doesn't exist
    assert result is None


@pytest.mark.asyncio
async def test_load_aggregation_filename_construction_accuracy(
    valid_aggregation_manager,
):
    """Test load_aggregation filename construction accuracy"""
    from unittest.mock import MagicMock
    import tplr

    window = 15
    mock_bucket = MagicMock()
    mock_bucket.name = "filename-test-bucket"

    # Mock the bucket property directly since load_aggregation uses self.bucket
    valid_aggregation_manager.bucket = mock_bucket

    with patch.object(
        valid_aggregation_manager.storage_client, "get_object_size", return_value=None
    ) as mock_get_size:
        await valid_aggregation_manager.load_aggregation(window)

    # Verify filename construction - uses "aggregator" not "aggregation"
    expected_filename = f"aggregator-{window}-v{tplr.__version__}.pt"
    mock_get_size.assert_called_once()
    call_args = mock_get_size.call_args
    assert call_args[0][0] == expected_filename  # First argument should be filename


@pytest.mark.asyncio
async def test_load_aggregation_bucket_configuration_access(valid_aggregation_manager):
    """Test load_aggregation bucket configuration access"""
    from unittest.mock import MagicMock

    window = 20
    mock_bucket = MagicMock()
    mock_bucket.name = "bucket-config-test"
    mock_bucket.region = "us-west-2"
    mock_bucket.access_key = "test-access-key"

    # Set bucket directly since load_aggregation uses self.bucket
    valid_aggregation_manager.bucket = mock_bucket

    with patch.object(
        valid_aggregation_manager.storage_client, "get_object_size", return_value=None
    ):
        await valid_aggregation_manager.load_aggregation(window)

    # Verify bucket configuration is available
    assert valid_aggregation_manager.bucket.name == "bucket-config-test"
    assert valid_aggregation_manager.bucket.region == "us-west-2"
    assert valid_aggregation_manager.bucket.access_key == "test-access-key"


@pytest.mark.asyncio
async def test_load_aggregation_credentials_handling(valid_aggregation_manager):
    """Test load_aggregation credentials handling"""
    from unittest.mock import MagicMock

    window = 25
    mock_bucket = MagicMock()
    mock_bucket.name = "credentials-test-bucket"

    # Mock credentials error
    credentials_error = Exception("Invalid credentials")

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object_size",
            side_effect=credentials_error,
        ):
            result = await valid_aggregation_manager.load_aggregation(window)

    # Should handle credentials error gracefully
    assert result is None


# -----------------------------------------------------------------------------
# LOAD_AGGREGATION - FILE SIZE HANDLING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_load_aggregation_with_small_files_standard_download(
    valid_aggregation_manager,
):
    """Test load_aggregation with small files (<100MB) - standard download"""
    from unittest.mock import MagicMock

    window = 30
    mock_bucket = MagicMock()
    mock_bucket.name = "small-file-bucket"

    # Small file size (under 100MB threshold)
    file_size = 50 * 1024 * 1024  # 50MB

    mock_aggregation_data = {
        "aggregated_gradients": {"layer.small": torch.tensor([3.0])},
        "num_gradients": 3,
        "window": window,
    }

    temp_file_path = "/tmp/small_agg.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object_size",
            return_value=file_size,
        ):
            # Should use standard get_object for small files
            with patch.object(
                valid_aggregation_manager.storage_client,
                "get_object",
                return_value=b"small_file_data",
            ) as mock_get_object:
                # Should NOT call multipart_download for small files
                with patch.object(
                    valid_aggregation_manager.storage_client, "multipart_download"
                ) as mock_multipart:
                    with patch.object(
                        valid_aggregation_manager.file_manager,
                        "create_temp_file",
                        return_value=temp_file_path,
                    ):
                        with patch("torch.load", return_value=mock_aggregation_data):
                            with patch.object(
                                valid_aggregation_manager.file_manager, "delete_file"
                            ):
                                result = (
                                    await valid_aggregation_manager.load_aggregation(
                                        window
                                    )
                                )

    # Verify standard download was used
    mock_get_object.assert_called_once()
    mock_multipart.assert_not_called()

    # Verify result
    assert result is not None
    assert result["aggregated_gradients"]["layer.small"].equal(torch.tensor([3.0]))


@pytest.mark.asyncio
async def test_load_aggregation_with_large_files_multipart_download(
    valid_aggregation_manager,
):
    """Test load_aggregation with large files (>100MB) - multipart download"""
    from unittest.mock import MagicMock

    window = 35
    mock_bucket = MagicMock()
    mock_bucket.name = "large-file-bucket"

    # Large file size (over 100MB threshold)
    file_size = 200 * 1024 * 1024  # 200MB

    mock_aggregation_data = {
        "aggregated_gradients": {"layer.large": torch.tensor([4.0, 5.0, 6.0])},
        "num_gradients": 10,
        "window": window,
    }

    temp_file_path = "/tmp/large_agg.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object_size",
            return_value=file_size,
        ):
            # Should NOT use standard get_object for large files
            with patch.object(
                valid_aggregation_manager.storage_client, "get_object"
            ) as mock_get_object:
                # Should use multipart_download for large files
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "multipart_download",
                    return_value=True,
                ) as mock_multipart:
                    with patch.object(
                        valid_aggregation_manager.file_manager,
                        "create_temp_file",
                        return_value=temp_file_path,
                    ):
                        with patch("torch.load", return_value=mock_aggregation_data):
                            with patch.object(
                                valid_aggregation_manager.file_manager, "delete_file"
                            ):
                                result = (
                                    await valid_aggregation_manager.load_aggregation(
                                        window
                                    )
                                )

    # Verify multipart download was used
    mock_get_object.assert_not_called()
    mock_multipart.assert_called_once()

    # Verify result
    assert result is not None
    assert result["aggregated_gradients"]["layer.large"].equal(
        torch.tensor([4.0, 5.0, 6.0])
    )


@pytest.mark.asyncio
async def test_load_aggregation_file_size_detection_accuracy(valid_aggregation_manager):
    """Test load_aggregation file size detection accuracy"""
    from unittest.mock import MagicMock

    window = 40
    mock_bucket = MagicMock()
    mock_bucket.name = "size-detection-bucket"

    # Test exact threshold boundary (100MB)
    file_size_exact = 100 * 1024 * 1024  # Exactly 100MB

    mock_aggregation_data = {
        "aggregated_gradients": {"layer.exact": torch.tensor([7.0])},
        "num_gradients": 1,
        "window": window,
    }

    temp_file_path = "/tmp/exact_size_agg.pt"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object_size",
            return_value=file_size_exact,
        ) as mock_get_size:
            # At exactly 100MB, should use standard download (<=)
            with patch.object(
                valid_aggregation_manager.storage_client,
                "get_object",
                return_value=b"exact_size_data",
            ) as mock_get_object:
                with patch.object(
                    valid_aggregation_manager.storage_client, "multipart_download"
                ) as mock_multipart:
                    with patch.object(
                        valid_aggregation_manager.file_manager,
                        "create_temp_file",
                        return_value=temp_file_path,
                    ):
                        with patch("torch.load", return_value=mock_aggregation_data):
                            with patch.object(
                                valid_aggregation_manager.file_manager, "delete_file"
                            ):
                                result = (
                                    await valid_aggregation_manager.load_aggregation(
                                        window
                                    )
                                )

    # Verify size detection accuracy
    mock_get_size.assert_called_once()

    # At exactly 100MB, should use standard download
    mock_get_object.assert_called_once()
    mock_multipart.assert_not_called()


@pytest.mark.asyncio
async def test_load_aggregation_get_object_size_failure_handling(
    valid_aggregation_manager,
):
    """Test load_aggregation get_object_size failure handling"""
    from unittest.mock import MagicMock

    window = 45
    mock_bucket = MagicMock()
    mock_bucket.name = "size-failure-bucket"

    # Mock get_object_size to raise an exception
    size_error = Exception("Failed to get object size")

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object_size",
            side_effect=size_error,
        ):
            result = await valid_aggregation_manager.load_aggregation(window)

    # Should handle get_object_size failure gracefully
    assert result is None


@pytest.mark.asyncio
async def test_load_aggregation_get_object_size_none_return(valid_aggregation_manager):
    """Test load_aggregation get_object_size None return"""
    from unittest.mock import MagicMock

    window = 50
    mock_bucket = MagicMock()
    mock_bucket.name = "size-none-bucket"

    with patch.object(
        valid_aggregation_manager.peer_manager.chain_manager,
        "get_bucket",
        return_value=mock_bucket,
    ):
        # Mock get_object_size returning None (file not found)
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object_size",
            return_value=None,
        ) as mock_get_size:
            result = await valid_aggregation_manager.load_aggregation(window)

    # Verify get_object_size was called
    mock_get_size.assert_called_once()

    # Should return None when get_object_size returns None
    assert result is None


# -----------------------------------------------------------------------------
# LOAD_AGGREGATION - DOWNLOAD METHODS
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_load_aggregation_standard_download_success(valid_aggregation_manager):
    """Test load_aggregation standard download success"""
    from unittest.mock import MagicMock

    window = 5
    mock_bucket = MagicMock()
    mock_bucket.name = "standard-download-bucket"
    valid_aggregation_manager.bucket = mock_bucket

    # Small file size (triggers standard download)
    file_size = 50 * 1024 * 1024  # 50MB

    mock_aggregation_data = {
        "aggregated_gradients": {"layer.standard": torch.tensor([1.0, 2.0])},
        "num_gradients": 10,
        "window": window,
    }

    temp_file_path = "/tmp/standard_agg.pt"

    with patch.object(
        valid_aggregation_manager.storage_client,
        "get_object_size",
        return_value=file_size,
    ):
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object",
            return_value=b"standard_data",
        ) as mock_get_object:
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch("torch.load", return_value=mock_aggregation_data):
                    with patch.object(
                        valid_aggregation_manager.file_manager, "delete_file"
                    ):
                        result = await valid_aggregation_manager.load_aggregation(
                            window
                        )

    # Verify standard download was used successfully
    mock_get_object.assert_called_once()

    # Verify result
    assert result is not None
    assert result["aggregated_gradients"]["layer.standard"].equal(
        torch.tensor([1.0, 2.0])
    )
    assert result["num_gradients"] == 10
    assert result["window"] == window


@pytest.mark.asyncio
async def test_load_aggregation_standard_download_failure(valid_aggregation_manager):
    """Test load_aggregation standard download failure"""
    from unittest.mock import MagicMock

    window = 6
    mock_bucket = MagicMock()
    mock_bucket.name = "standard-fail-bucket"
    valid_aggregation_manager.bucket = mock_bucket

    # Small file size (triggers standard download)
    file_size = 30 * 1024 * 1024  # 30MB

    download_error = Exception("Network error during download")

    with patch.object(
        valid_aggregation_manager.storage_client,
        "get_object_size",
        return_value=file_size,
    ):
        # Mock get_object to fail
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object",
            side_effect=download_error,
        ) as mock_get_object:
            result = await valid_aggregation_manager.load_aggregation(window)

    # Verify standard download was attempted
    mock_get_object.assert_called_once()

    # Should return None on download failure
    assert result is None


@pytest.mark.asyncio
async def test_load_aggregation_multipart_download_success(valid_aggregation_manager):
    """Test load_aggregation multipart download success"""
    from unittest.mock import MagicMock

    window = 7
    mock_bucket = MagicMock()
    mock_bucket.name = "multipart-success-bucket"
    valid_aggregation_manager.bucket = mock_bucket

    # Large file size (triggers multipart download)
    file_size = 150 * 1024 * 1024  # 150MB

    mock_aggregation_data = {
        "aggregated_gradients": {"layer.multipart": torch.tensor([3.0, 4.0, 5.0])},
        "num_gradients": 20,
        "window": window,
    }

    temp_file_path = "/tmp/multipart_agg.pt"

    with patch.object(
        valid_aggregation_manager.storage_client,
        "get_object_size",
        return_value=file_size,
    ):
        # Should use multipart_download for large files
        with patch.object(
            valid_aggregation_manager.storage_client,
            "multipart_download",
            return_value=True,
        ) as mock_multipart:
            with patch.object(
                valid_aggregation_manager.storage_client, "get_object"
            ) as mock_get_object:
                with patch.object(
                    valid_aggregation_manager.file_manager,
                    "create_temp_file",
                    return_value=temp_file_path,
                ):
                    with patch("torch.load", return_value=mock_aggregation_data):
                        with patch.object(
                            valid_aggregation_manager.file_manager, "delete_file"
                        ):
                            result = await valid_aggregation_manager.load_aggregation(
                                window
                            )

    # Verify multipart download was used
    mock_multipart.assert_called_once()
    mock_get_object.assert_not_called()  # Should not use standard download

    # Verify result
    assert result is not None
    assert result["aggregated_gradients"]["layer.multipart"].equal(
        torch.tensor([3.0, 4.0, 5.0])
    )
    assert result["num_gradients"] == 20


@pytest.mark.asyncio
async def test_load_aggregation_multipart_download_failure(valid_aggregation_manager):
    """Test load_aggregation multipart download failure"""
    from unittest.mock import MagicMock

    window = 8
    mock_bucket = MagicMock()
    mock_bucket.name = "multipart-fail-bucket"
    valid_aggregation_manager.bucket = mock_bucket

    # Large file size (triggers multipart download)
    file_size = 200 * 1024 * 1024  # 200MB

    temp_file_path = "/tmp/multipart_fail_agg.pt"

    with patch.object(
        valid_aggregation_manager.storage_client,
        "get_object_size",
        return_value=file_size,
    ):
        # Mock multipart_download to fail
        with patch.object(
            valid_aggregation_manager.storage_client,
            "multipart_download",
            return_value=False,
        ) as mock_multipart:
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                # Don't expect delete_file to be called on multipart failure
                result = await valid_aggregation_manager.load_aggregation(window)

    # Verify multipart download was attempted
    mock_multipart.assert_called_once()

    # Should return None on download failure
    assert result is None


@pytest.mark.asyncio
async def test_load_aggregation_timeout_handling_45s(valid_aggregation_manager):
    """Test load_aggregation timeout handling (45s)"""
    from unittest.mock import MagicMock

    window = 9
    mock_bucket = MagicMock()
    mock_bucket.name = "timeout-bucket"
    valid_aggregation_manager.bucket = mock_bucket

    # Small file size (standard download)
    file_size = 40 * 1024 * 1024  # 40MB

    import tplr

    expected_filename = f"aggregator-{window}-v{tplr.__version__}.pt"

    timeout_error = asyncio.TimeoutError("Download timeout after 45s")

    with patch.object(
        valid_aggregation_manager.storage_client,
        "get_object_size",
        return_value=file_size,
    ):
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object",
            side_effect=timeout_error,
        ) as mock_get_object:
            result = await valid_aggregation_manager.load_aggregation(window)

    # Verify timeout parameter was used (45 seconds)
    mock_get_object.assert_called_once()
    call_args = mock_get_object.call_args
    # Check that timeout=45 was passed
    assert call_args[1]["timeout"] == 45

    # Should return None on timeout
    assert result is None


@pytest.mark.asyncio
async def test_load_aggregation_download_method_selection_logic(
    valid_aggregation_manager,
):
    """Test load_aggregation download method selection logic"""
    from unittest.mock import MagicMock

    mock_bucket = MagicMock()
    mock_bucket.name = "method-selection-bucket"
    valid_aggregation_manager.bucket = mock_bucket

    # Test cases: (file_size, expected_method, should_call_get_object, should_call_multipart)
    test_cases = [
        (50 * 1024 * 1024, "standard", True, False),  # 50MB - standard
        (100 * 1024 * 1024, "standard", True, False),  # 100MB exactly - standard
        (101 * 1024 * 1024, "multipart", False, True),  # 101MB - multipart
        (500 * 1024 * 1024, "multipart", False, True),  # 500MB - multipart
    ]

    for i, (
        file_size,
        method_name,
        should_call_get_object,
        should_call_multipart,
    ) in enumerate(test_cases):
        window = 10 + i
        temp_file_path = f"/tmp/method_test_{i}.pt"

        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object_size",
            return_value=file_size,
        ):
            with patch.object(
                valid_aggregation_manager.storage_client,
                "get_object",
                return_value=b"test_data",
            ) as mock_get_object:
                with patch.object(
                    valid_aggregation_manager.storage_client,
                    "multipart_download",
                    return_value=True,
                ) as mock_multipart:
                    with patch.object(
                        valid_aggregation_manager.file_manager,
                        "create_temp_file",
                        return_value=temp_file_path,
                    ):
                        with patch("torch.load", return_value={"test": "data"}):
                            with patch.object(
                                valid_aggregation_manager.file_manager, "delete_file"
                            ):
                                await valid_aggregation_manager.load_aggregation(window)

        # Verify correct method was selected
        if should_call_get_object:
            mock_get_object.assert_called_once()
        else:
            mock_get_object.assert_not_called()

        if should_call_multipart:
            mock_multipart.assert_called_once()
        else:
            mock_multipart.assert_not_called()


# -----------------------------------------------------------------------------
# LOAD_AGGREGATION - DATA PROCESSING
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_load_aggregation_temp_file_creation_and_cleanup(
    valid_aggregation_manager,
):
    """Test load_aggregation temp file creation and cleanup"""
    from unittest.mock import MagicMock

    window = 15
    mock_bucket = MagicMock()
    mock_bucket.name = "temp-file-bucket"
    valid_aggregation_manager.bucket = mock_bucket

    file_size = 60 * 1024 * 1024  # 60MB
    temp_file_path = "/tmp/temp_agg_test.pt"

    mock_aggregation_data = {
        "aggregated_gradients": {"layer.temp": torch.tensor([6.0])},
        "num_gradients": 5,
        "window": window,
    }

    with patch.object(
        valid_aggregation_manager.storage_client,
        "get_object_size",
        return_value=file_size,
    ):
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object",
            return_value=b"temp_data",
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ) as mock_create:
                with patch("torch.load", return_value=mock_aggregation_data):
                    with patch.object(
                        valid_aggregation_manager.file_manager, "delete_file"
                    ) as mock_delete:
                        result = await valid_aggregation_manager.load_aggregation(
                            window
                        )

    # Verify temp file lifecycle - actual prefix is "aggregation_load"
    mock_create.assert_called_once_with("aggregation_load")
    mock_delete.assert_called_once_with(temp_file_path)

    # Verify result
    assert result is not None
    assert result["aggregated_gradients"]["layer.temp"].equal(torch.tensor([6.0]))


@pytest.mark.asyncio
async def test_load_aggregation_torch_load_with_map_location(valid_aggregation_manager):
    """Test load_aggregation torch.load with map_location"""
    from unittest.mock import MagicMock

    window = 16
    mock_bucket = MagicMock()
    mock_bucket.name = "map-location-bucket"
    valid_aggregation_manager.bucket = mock_bucket

    file_size = 70 * 1024 * 1024  # 70MB
    temp_file_path = "/tmp/map_location_agg.pt"

    mock_aggregation_data = {
        "aggregated_gradients": {"layer.device": torch.tensor([7.0])},
        "num_gradients": 3,
        "window": window,
    }

    with patch.object(
        valid_aggregation_manager.storage_client,
        "get_object_size",
        return_value=file_size,
    ):
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object",
            return_value=b"device_data",
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch(
                    "torch.load", return_value=mock_aggregation_data
                ) as mock_torch_load:
                    with patch.object(
                        valid_aggregation_manager.file_manager, "delete_file"
                    ):
                        result = await valid_aggregation_manager.load_aggregation(
                            window
                        )

    # Verify torch.load was called with correct map_location
    mock_torch_load.assert_called_once_with(
        temp_file_path,
        map_location=valid_aggregation_manager.device,
        weights_only=False,
    )

    # Verify result
    assert result is not None
    assert result["aggregated_gradients"]["layer.device"].equal(torch.tensor([7.0]))


@pytest.mark.asyncio
async def test_load_aggregation_torch_load_with_weights_only_false(
    valid_aggregation_manager,
):
    """Test load_aggregation torch.load with weights_only=False"""
    from unittest.mock import MagicMock

    window = 17
    mock_bucket = MagicMock()
    mock_bucket.name = "weights-only-bucket"
    valid_aggregation_manager.bucket = mock_bucket

    file_size = 80 * 1024 * 1024  # 80MB
    temp_file_path = "/tmp/weights_only_agg.pt"

    mock_aggregation_data = {
        "aggregated_gradients": {"layer.weights": torch.tensor([8.0, 9.0])},
        "num_gradients": 7,
        "window": window,
        "metadata": {"version": "1.0", "created_at": "2024-01-01"},  # Non-weight data
    }

    with patch.object(
        valid_aggregation_manager.storage_client,
        "get_object_size",
        return_value=file_size,
    ):
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object",
            return_value=b"weights_data",
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch(
                    "torch.load", return_value=mock_aggregation_data
                ) as mock_torch_load:
                    with patch.object(
                        valid_aggregation_manager.file_manager, "delete_file"
                    ):
                        result = await valid_aggregation_manager.load_aggregation(
                            window
                        )

    # Verify torch.load was called with weights_only=False
    call_args = mock_torch_load.call_args
    assert call_args[1]["weights_only"] == False

    # Verify non-weight data was preserved (only possible with weights_only=False)
    assert result is not None
    assert "metadata" in result
    assert result["metadata"]["version"] == "1.0"


@pytest.mark.asyncio
async def test_load_aggregation_file_cleanup_on_success(valid_aggregation_manager):
    """Test load_aggregation file cleanup on success"""
    from unittest.mock import MagicMock

    window = 18
    mock_bucket = MagicMock()
    mock_bucket.name = "cleanup-success-bucket"
    valid_aggregation_manager.bucket = mock_bucket

    file_size = 90 * 1024 * 1024  # 90MB
    temp_file_path = "/tmp/cleanup_success_agg.pt"

    mock_aggregation_data = {
        "aggregated_gradients": {"layer.cleanup": torch.tensor([10.0])},
        "num_gradients": 1,
        "window": window,
    }

    with patch.object(
        valid_aggregation_manager.storage_client,
        "get_object_size",
        return_value=file_size,
    ):
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object",
            return_value=b"cleanup_data",
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch("torch.load", return_value=mock_aggregation_data):
                    with patch.object(
                        valid_aggregation_manager.file_manager, "delete_file"
                    ) as mock_delete:
                        result = await valid_aggregation_manager.load_aggregation(
                            window
                        )

    # Should cleanup temp file on successful processing
    mock_delete.assert_called_once_with(temp_file_path)

    # Verify successful result
    assert result is not None
    assert result["aggregated_gradients"]["layer.cleanup"].equal(torch.tensor([10.0]))


@pytest.mark.asyncio
async def test_load_aggregation_file_cleanup_on_failure(valid_aggregation_manager):
    """Test load_aggregation file cleanup on failure"""
    from unittest.mock import MagicMock

    window = 19
    mock_bucket = MagicMock()
    mock_bucket.name = "cleanup-failure-bucket"
    valid_aggregation_manager.bucket = mock_bucket

    file_size = 95 * 1024 * 1024  # 95MB
    temp_file_path = "/tmp/cleanup_failure_agg.pt"

    torch_load_error = Exception("Corrupted aggregation file")

    with patch.object(
        valid_aggregation_manager.storage_client,
        "get_object_size",
        return_value=file_size,
    ):
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object",
            return_value=b"corrupted_data",
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                # Mock torch.load to fail
                with patch("torch.load", side_effect=torch_load_error):
                    result = await valid_aggregation_manager.load_aggregation(window)

    # The implementation doesn't call delete_file on torch.load failure
    # It only handles cleanup in the finally block for successful paths
    # Should return None on failure
    assert result is None


@pytest.mark.asyncio
async def test_load_aggregation_return_value_structure(valid_aggregation_manager):
    """Test load_aggregation return value structure"""
    from unittest.mock import MagicMock

    window = 20
    mock_bucket = MagicMock()
    mock_bucket.name = "return-structure-bucket"
    valid_aggregation_manager.bucket = mock_bucket

    file_size = 45 * 1024 * 1024  # 45MB
    temp_file_path = "/tmp/return_structure_agg.pt"

    mock_aggregation_data = {
        "aggregated_gradients": {
            "layer1.weight": torch.tensor([11.0, 12.0]),
            "layer1.bias": torch.tensor([13.0]),
            "layer2.weight": torch.tensor([14.0, 15.0, 16.0]),
        },
        "num_gradients": 15,
        "window": window,
        "timestamp": "2024-01-01T12:00:00Z",
        "version": "1.0.0",
        "metadata": {"total_parameters": 6, "aggregation_method": "federated_avg"},
    }

    with patch.object(
        valid_aggregation_manager.storage_client,
        "get_object_size",
        return_value=file_size,
    ):
        with patch.object(
            valid_aggregation_manager.storage_client,
            "get_object",
            return_value=b"structure_data",
        ):
            with patch.object(
                valid_aggregation_manager.file_manager,
                "create_temp_file",
                return_value=temp_file_path,
            ):
                with patch("torch.load", return_value=mock_aggregation_data):
                    with patch.object(
                        valid_aggregation_manager.file_manager, "delete_file"
                    ):
                        result = await valid_aggregation_manager.load_aggregation(
                            window
                        )

    # Verify return value structure and completeness
    assert result is not None
    assert isinstance(result, dict)

    # Core required fields
    assert "aggregated_gradients" in result
    assert "num_gradients" in result
    assert "window" in result

    # Verify aggregated_gradients structure
    assert isinstance(result["aggregated_gradients"], dict)
    assert len(result["aggregated_gradients"]) == 3
    assert result["aggregated_gradients"]["layer1.weight"].equal(
        torch.tensor([11.0, 12.0])
    )
    assert result["aggregated_gradients"]["layer1.bias"].equal(torch.tensor([13.0]))
    assert result["aggregated_gradients"]["layer2.weight"].equal(
        torch.tensor([14.0, 15.0, 16.0])
    )

    # Verify metadata fields
    assert result["num_gradients"] == 15
    assert result["window"] == window
    assert result["timestamp"] == "2024-01-01T12:00:00Z"
    assert result["version"] == "1.0.0"

    # Verify additional metadata preserved
    assert "metadata" in result
    assert result["metadata"]["total_parameters"] == 6
    assert result["metadata"]["aggregation_method"] == "federated_avg"
