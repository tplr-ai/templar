# The MIT License (MIT)
# © 2025 tplr.ai

import pytest
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock, call, AsyncMock
import numpy as np
import torch
import asyncio
from pathlib import Path

# Import the classes from the actual module location
from src.tplr.sharded_dataset import SharedShardedDataset, ShardedDatasetManager


class TestSharedShardedDataset:
    """Comprehensive tests for SharedShardedDataset class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data_files(self, temp_dir):
        """Create sample data files for testing."""
        # Create sample tokens file
        tokens_file = os.path.join(temp_dir, "train_000001.npy")
        sample_tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint16)
        np.save(tokens_file, sample_tokens)
        
        # Create sample IDs file
        ids_file = os.path.join(temp_dir, "sample_ids_000001.bin")
        sample_ids = np.array([100, 101], dtype=np.uint64)
        sample_ids.tofile(ids_file)
        
        return tokens_file, ids_file
    
    @pytest.fixture
    def mock_distributed(self):
        """Mock torch.distributed for testing."""
        with patch('torch.distributed.barrier') as mock_barrier:
            yield mock_barrier
    
    def test_locate_shards_with_custom_path(self, temp_dir):
        """Test locate_shards with custom path."""
        tokens_file, ids_file = SharedShardedDataset.locate_shards(
            shard_index=5, custom_path=temp_dir
        )
        
        expected_tokens = os.path.join(temp_dir, "train_000005.npy")
        expected_ids = os.path.join(temp_dir, "sample_ids_000005.bin")
        
        assert tokens_file == expected_tokens
        assert ids_file == expected_ids
    
    def test_locate_shards_with_env_variable(self, temp_dir):
        """Test locate_shards using environment variable."""
        with patch.dict(os.environ, {'DATASET_BINS_PATH': temp_dir}):
            tokens_file, ids_file = SharedShardedDataset.locate_shards(shard_index=10)
            
            expected_tokens = os.path.join(temp_dir, "train_000010.npy")
            expected_ids = os.path.join(temp_dir, "sample_ids_000010.bin")
            
            assert tokens_file == expected_tokens
            assert ids_file == expected_ids
    
    def test_locate_shards_no_path_configured(self):
        """Test locate_shards raises ValueError when no path is configured."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Dataset path not configured"):
                SharedShardedDataset.locate_shards(shard_index=1)
    
    def test_locate_shards_formatting(self, temp_dir):
        """Test that shard index is properly formatted with leading zeros."""
        tokens_file, ids_file = SharedShardedDataset.locate_shards(
            shard_index=42, custom_path=temp_dir
        )
        
        assert "train_000042.npy" in tokens_file
        assert "sample_ids_000042.bin" in ids_file
    
    def test_locate_shards_large_index(self, temp_dir):
        """Test locate_shards with large shard index."""
        tokens_file, ids_file = SharedShardedDataset.locate_shards(
            shard_index=999999, custom_path=temp_dir
        )
        
        assert "train_999999.npy" in tokens_file
        assert "sample_ids_999999.bin" in ids_file
    
    def test_locate_shards_zero_index(self, temp_dir):
        """Test locate_shards with zero index."""
        tokens_file, ids_file = SharedShardedDataset.locate_shards(
            shard_index=0, custom_path=temp_dir
        )
        
        assert "train_000000.npy" in tokens_file
        assert "sample_ids_000000.bin" in ids_file
    
    def test_check_paths_existing_files(self, sample_data_files):
        """Test check_paths with existing files."""
        tokens_file, ids_file = sample_data_files
        # Should not raise any exception
        SharedShardedDataset.check_paths([tokens_file, ids_file])
    
    def test_check_paths_missing_files(self, temp_dir):
        """Test check_paths raises FileNotFoundError for missing files."""
        missing_file = os.path.join(temp_dir, "nonexistent.npy")
        
        with pytest.raises(FileNotFoundError, match="Pre-processed file.*not found"):
            SharedShardedDataset.check_paths([missing_file])
    
    def test_check_paths_empty_list(self):
        """Test check_paths with empty list."""
        # Should not raise any exception
        SharedShardedDataset.check_paths([])
    
    def test_check_paths_multiple_missing_files(self, temp_dir):
        """Test check_paths fails on first missing file."""
        file1 = os.path.join(temp_dir, "missing1.npy")
        file2 = os.path.join(temp_dir, "missing2.npy")
        
        with pytest.raises(FileNotFoundError):
            SharedShardedDataset.check_paths([file1, file2])
    
    def test_check_paths_mixed_existing_missing(self, temp_dir, sample_data_files):
        """Test check_paths with mix of existing and missing files."""
        tokens_file, _ = sample_data_files
        missing_file = os.path.join(temp_dir, "missing.npy")
        
        with pytest.raises(FileNotFoundError):
            SharedShardedDataset.check_paths([tokens_file, missing_file])
    
    def test_check_paths_error_message_format(self, temp_dir):
        """Test check_paths error message format."""
        os.makedirs(os.path.join(temp_dir, "subdir"), exist_ok=True)
        missing_file = os.path.join(temp_dir, "subdir", "missing.npy")
        
        with pytest.raises(FileNotFoundError) as exc_info:
            SharedShardedDataset.check_paths([missing_file])
        
        error_msg = str(exc_info.value)
        assert "missing.npy" in error_msg
        assert "Run the preprocessing script first" in error_msg
    
    @patch('tplr.logger.info')
    def test_init_successful(self, mock_logger, sample_data_files, mock_distributed):
        """Test successful initialization of SharedShardedDataset."""
        tokens_file, ids_file = sample_data_files
        
        with patch.object(SharedShardedDataset, 'locate_shards', return_value=(tokens_file, ids_file)):
            dataset = SharedShardedDataset(
                shard_index=1,
                sequence_length=5,
                rank=0,
                world_size=1,
                token_dtype=np.uint16
            )
            
            assert dataset.seqlen == 5
            assert dataset.rank == 0
            assert dataset.world == 1
            assert dataset.total_samples == 2  # Based on sample data
            mock_logger.assert_called_once()
            # Check that the log message contains expected information
            log_call = mock_logger.call_args
            assert "rank 0" in log_call[0][0]
            assert "2 samples" in log_call[0][0]
    
    @patch('tplr.logger.info')
    def test_init_distributed_mode(self, mock_logger, sample_data_files, mock_distributed):
        """Test initialization in distributed mode."""
        tokens_file, ids_file = sample_data_files
        
        with patch.object(SharedShardedDataset, 'locate_shards', return_value=(tokens_file, ids_file)):
            dataset = SharedShardedDataset(
                shard_index=1,
                sequence_length=5,
                rank=1,
                world_size=4,
                token_dtype=np.uint16
            )
            
            assert dataset.rank == 1
            assert dataset.world == 4
            mock_distributed.assert_called_once_with(device_ids=[1])
    
    @patch('tplr.logger.info')
    def test_init_single_process_mode(self, mock_logger, sample_data_files, mock_distributed):
        """Test initialization in single process mode (world_size=1)."""
        tokens_file, ids_file = sample_data_files
        
        with patch.object(SharedShardedDataset, 'locate_shards', return_value=(tokens_file, ids_file)):
            dataset = SharedShardedDataset(
                shard_index=1,
                sequence_length=5,
                rank=0,
                world_size=1,
                token_dtype=np.uint16
            )
            
            assert dataset.world == 1
            # Should not call distributed barrier in single process mode
            mock_distributed.assert_not_called()
    
    def test_init_missing_files(self, temp_dir):
        """Test initialization fails with missing files."""
        with patch.dict(os.environ, {'DATASET_BINS_PATH': temp_dir}):
            with pytest.raises(FileNotFoundError):
                SharedShardedDataset(
                    shard_index=999,
                    sequence_length=5,
                    rank=0,
                    world_size=1
                )
    
    def test_mmap_tokens_and_ids(self, sample_data_files):
        """Test memory mapping of tokens and IDs."""
        tokens_file, ids_file = sample_data_files
        
        with patch.object(SharedShardedDataset, 'locate_shards', return_value=(tokens_file, ids_file)):
            with patch('tplr.logger.info'):
                dataset = SharedShardedDataset(
                    shard_index=1,
                    sequence_length=5,
                    rank=0,
                    world_size=1,
                    token_dtype=np.uint16
                )
                
                # Verify tokens are loaded correctly
                assert isinstance(dataset.tokens, torch.Tensor)
                assert len(dataset.tokens) == 10
                
                # Verify sample IDs are loaded correctly
                assert isinstance(dataset.sample_ids, torch.Tensor)
                assert dataset.sample_ids.dtype == torch.uint64
                assert len(dataset.sample_ids) == 2
    
    def test_mmap_different_dtypes(self, temp_dir):
        """Test memory mapping with different token dtypes."""
        # Create tokens file with uint32 dtype
        tokens_file = os.path.join(temp_dir, "train_000001.npy")
        sample_tokens = np.array([1, 2, 3, 4], dtype=np.uint32)
        np.save(tokens_file, sample_tokens)
        
        # Create IDs file
        ids_file = os.path.join(temp_dir, "sample_ids_000001.bin")
        sample_ids = np.array([100], dtype=np.uint64)
        sample_ids.tofile(ids_file)
        
        with patch.object(SharedShardedDataset, 'locate_shards', return_value=(tokens_file, ids_file)):
            with patch('tplr.logger.info'):
                dataset = SharedShardedDataset(
                    shard_index=1,
                    sequence_length=2,
                    rank=0,
                    world_size=1,
                    token_dtype=np.uint32
                )
                
                assert isinstance(dataset.tokens, torch.Tensor)
                assert len(dataset.tokens) == 4
    
    def test_len_method(self, sample_data_files):
        """Test __len__ method."""
        tokens_file, ids_file = sample_data_files
        
        with patch.object(SharedShardedDataset, 'locate_shards', return_value=(tokens_file, ids_file)):
            with patch('tplr.logger.info'):
                dataset = SharedShardedDataset(
                    shard_index=1,
                    sequence_length=5,
                    rank=0,
                    world_size=1
                )
                
                assert len(dataset) == 2  # Based on sample data
    
    def test_getitem_valid_index(self, sample_data_files):
        """Test __getitem__ with valid index."""
        tokens_file, ids_file = sample_data_files
        
        with patch.object(SharedShardedDataset, 'locate_shards', return_value=(tokens_file, ids_file)):
            with patch('tplr.logger.info'):
                dataset = SharedShardedDataset(
                    shard_index=1,
                    sequence_length=5,
                    rank=0,
                    world_size=1
                )
                
                # Test first sequence
                sequence = dataset[0]
                assert isinstance(sequence, torch.Tensor)
                assert len(sequence) == 5
                
                # Test second sequence
                sequence = dataset[1]
                assert isinstance(sequence, torch.Tensor)
                assert len(sequence) == 5
    
    def test_getitem_invalid_index(self, sample_data_files):
        """Test __getitem__ with invalid index."""
        tokens_file, ids_file = sample_data_files
        
        with patch.object(SharedShardedDataset, 'locate_shards', return_value=(tokens_file, ids_file)):
            with patch('tplr.logger.info'):
                dataset = SharedShardedDataset(
                    shard_index=1,
                    sequence_length=5,
                    rank=0,
                    world_size=1
                )
                
                with pytest.raises(IndexError):
                    dataset[2]  # Index beyond available samples
                
                with pytest.raises(IndexError):
                    dataset[100]  # Much larger index
    
    def test_getitem_boundary_cases(self, sample_data_files):
        """Test __getitem__ with boundary cases."""
        tokens_file, ids_file = sample_data_files
        
        with patch.object(SharedShardedDataset, 'locate_shards', return_value=(tokens_file, ids_file)):
            with patch('tplr.logger.info'):
                dataset = SharedShardedDataset(
                    shard_index=1,
                    sequence_length=5,
                    rank=0,
                    world_size=1
                )
                
                # Test at boundary (last valid index)
                sequence = dataset[1]
                assert isinstance(sequence, torch.Tensor)
                
                # Test just beyond boundary
                with pytest.raises(IndexError):
                    dataset[2]
    
    def test_getitem_sequence_slicing(self, sample_data_files):
        """Test that __getitem__ returns correct sequence slices."""
        tokens_file, ids_file = sample_data_files
        
        with patch.object(SharedShardedDataset, 'locate_shards', return_value=(tokens_file, ids_file)):
            with patch('tplr.logger.info'):
                dataset = SharedShardedDataset(
                    shard_index=1,
                    sequence_length=3,  # Smaller sequence length for testing
                    rank=0,
                    world_size=1
                )
                
                # Manually check that slicing works correctly
                # First sequence should be tokens[0:3], second should be tokens[3:6]
                sequence_0 = dataset[0]
                sequence_1 = dataset[1]
                
                assert len(sequence_0) == 3
                assert len(sequence_1) == 3
                
                # Verify they're different slices
                assert not torch.equal(sequence_0, sequence_1)
    
    def test_sample_id_valid_index(self, sample_data_files):
        """Test sample_id method with valid index."""
        tokens_file, ids_file = sample_data_files
        
        with patch.object(SharedShardedDataset, 'locate_shards', return_value=(tokens_file, ids_file)):
            with patch('tplr.logger.info'):
                dataset = SharedShardedDataset(
                    shard_index=1,
                    sequence_length=5,
                    rank=0,
                    world_size=1
                )
                
                # Test getting sample IDs
                sample_id_0 = dataset.sample_id(0)
                sample_id_1 = dataset.sample_id(1)
                
                assert isinstance(sample_id_0, int)
                assert isinstance(sample_id_1, int)
                assert sample_id_0 == 100  # Based on sample data
                assert sample_id_1 == 101  # Based on sample data
    
    def test_sample_id_invalid_index(self, sample_data_files):
        """Test sample_id method with invalid index."""
        tokens_file, ids_file = sample_data_files
        
        with patch.object(SharedShardedDataset, 'locate_shards', return_value=(tokens_file, ids_file)):
            with patch('tplr.logger.info'):
                dataset = SharedShardedDataset(
                    shard_index=1,
                    sequence_length=5,
                    rank=0,
                    world_size=1
                )
                
                # Test invalid indices (PyTorch will raise IndexError)
                with pytest.raises(IndexError):
                    dataset.sample_id(2)
                
                with pytest.raises(IndexError):
                    dataset.sample_id(100)
    
    def test_sample_id_return_type(self, sample_data_files):
        """Test sample_id returns proper integer type."""
        tokens_file, ids_file = sample_data_files
        
        with patch.object(SharedShardedDataset, 'locate_shards', return_value=(tokens_file, ids_file)):
            with patch('tplr.logger.info'):
                dataset = SharedShardedDataset(
                    shard_index=1,
                    sequence_length=5,
                    rank=0,
                    world_size=1
                )
                
                sample_id = dataset.sample_id(0)
                assert isinstance(sample_id, int)
                assert sample_id >= 0  # Should be a valid ID


class TestShardedDatasetManager:
    """Comprehensive tests for ShardedDatasetManager class."""
    
    @pytest.fixture
    def mock_comms(self):
        """Mock tplr.comms.Comms instance."""
        comms = Mock()
        comms.get_own_bucket.return_value = Mock()
        comms.s3_get_object = AsyncMock()
        return comms
    
    @pytest.fixture
    def manager(self, mock_comms):
        """Create a ShardedDatasetManager instance for testing."""
        return ShardedDatasetManager(
            sequence_length=512,
            rank=0,
            world_size=1,
            comms=mock_comms,
            token_dtype=np.uint16
        )
    
    def test_init(self, manager, mock_comms):
        """Test ShardedDatasetManager initialization."""
        assert manager.sequence_length == 512
        assert manager.rank == 0
        assert manager.world_size == 1
        assert manager.token_dtype == np.uint16
        assert manager.shard_index == 0
        assert manager.active_dataset is None
        assert manager.upcoming_dataset is None
        assert manager.comms == mock_comms
        assert manager.max_dataset_idx == 10
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        mock_comms = Mock()
        manager = ShardedDatasetManager(
            sequence_length=1024,
            rank=2,
            world_size=4,
            comms=mock_comms,
            token_dtype=np.uint32
        )
        
        assert manager.sequence_length == 1024
        assert manager.rank == 2
        assert manager.world_size == 4
        assert manager.token_dtype == np.uint32
    
    def test_init_default_token_dtype(self):
        """Test initialization uses default token dtype."""
        mock_comms = Mock()
        manager = ShardedDatasetManager(
            sequence_length=512,
            rank=0,
            world_size=1,
            comms=mock_comms
        )
        
        assert manager.token_dtype == np.uint16  # Default value
    
    @patch('os.path.exists')
    @patch('tplr.logger.info')
    def test_prepare_shard_files_exist(self, mock_logger, mock_exists, manager):
        """Test prepare_shard when files already exist."""
        mock_exists.return_value = True
        
        with patch.object(SharedShardedDataset, 'locate_shards', 
                         return_value=('/path/tokens.npy', '/path/ids.bin')):
            task = manager.prepare_shard(5)
            
            assert isinstance(task, asyncio.Task)
            mock_logger.assert_called_once_with("Preparing shard 5 at /path/tokens.npy")
    
    @patch('os.path.exists')
    @patch('tplr.logger.info')
    @patch('builtins.print')
    def test_prepare_shard_files_exist_with_print(self, mock_print, mock_logger, mock_exists, manager):
        """Test prepare_shard prints message when files exist."""
        mock_exists.return_value = True
        
        with patch.object(SharedShardedDataset, 'locate_shards', 
                         return_value=('/path/tokens.npy', '/path/ids.bin')):
            manager.prepare_shard(5)
            
            mock_print.assert_called_once_with("Shard 5 already exists on disk. Loading...")
    
    @patch('os.path.exists')
    @patch('tplr.logger.info')
    def test_prepare_shard_files_not_exist(self, mock_logger, mock_exists, manager):
        """Test prepare_shard when files don't exist."""
        mock_exists.return_value = False
        mock_bucket = Mock()
        manager.comms.get_own_bucket.return_value = mock_bucket
        
        with patch.object(SharedShardedDataset, 'locate_shards', 
                         return_value=('/path/tokens.npy', '/path/ids.bin')):
            with patch.object(manager, 'download_files', new_callable=AsyncMock) as mock_download:
                task = manager.prepare_shard(5)
                
                assert isinstance(task, asyncio.Task)
                manager.comms.get_own_bucket.assert_called_once_with("dataset", "read")
    
    @patch('os.path.exists')
    @patch('tplr.logger.info')
    def test_prepare_shard_partial_files_exist(self, mock_logger, mock_exists, manager):
        """Test prepare_shard when only some files exist."""
        # Mock exists to return True for tokens but False for IDs
        def mock_exists_side_effect(path):
            return 'tokens' in path
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_bucket = Mock()
        manager.comms.get_own_bucket.return_value = mock_bucket
        
        with patch.object(SharedShardedDataset, 'locate_shards', 
                         return_value=('/path/tokens.npy', '/path/ids.bin')):
            with patch.object(manager, 'download_files', new_callable=AsyncMock):
                task = manager.prepare_shard(5)
                
                assert isinstance(task, asyncio.Task)
                # Should trigger download since not all files exist
                manager.comms.get_own_bucket.assert_called_once_with("dataset", "read")
    
    @pytest.mark.asyncio
    async def test_download_files(self, manager):
        """Test download_files method."""
        mock_bucket = Mock()
        tokens_file = '/path/tokens.npy'
        ids_file = '/path/ids.bin'
        
        # Mock the s3_get_object calls
        manager.comms.s3_get_object = AsyncMock()
        
        result = await manager.download_files(mock_bucket, tokens_file, ids_file)
        
        # Verify both files were requested for download
        assert manager.comms.s3_get_object.call_count == 2
        expected_calls = [
            call(tokens_file, mock_bucket, load_data=False),
            call(ids_file, mock_bucket, load_data=False)
        ]
        manager.comms.s3_get_object.assert_has_calls(expected_calls, any_order=True)
    
    @pytest.mark.asyncio
    async def test_download_files_with_pathlike_objects(self, manager):
        """Test download_files with Path-like objects."""
        mock_bucket = Mock()
        tokens_file = Path('/path/tokens.npy')
        ids_file = Path('/path/ids.bin')
        
        manager.comms.s3_get_object = AsyncMock()
        
        await manager.download_files(mock_bucket, tokens_file, ids_file)
        
        # Should handle Path objects correctly  
        assert manager.comms.s3_get_object.call_count == 2
    
    @pytest.mark.asyncio
    async def test_create_dataset(self, manager):
        """Test create_dataset method."""
        with patch.object(manager, 'prepare_shard') as mock_prepare:
            mock_task = AsyncMock()
            mock_prepare.return_value = mock_task
            
            with patch('src.tplr.sharded_dataset.SharedShardedDataset') as mock_dataset_class:
                mock_dataset = Mock()
                mock_dataset_class.return_value = mock_dataset
                
                result = await manager.create_dataset(3)
                
                mock_prepare.assert_called_once_with(3)
                mock_task.assert_awaited_once()
                mock_dataset_class.assert_called_once_with(
                    shard_index=3,
                    sequence_length=512,
                    rank=0,
                    world_size=1,
                    token_dtype=np.uint16
                )
                assert result == mock_dataset
    
    @pytest.mark.asyncio
    async def test_create_dataset_with_custom_params(self):
        """Test create_dataset with custom parameters."""
        mock_comms = Mock()
        manager = ShardedDatasetManager(
            sequence_length=256,
            rank=1,
            world_size=2,
            comms=mock_comms,
            token_dtype=np.uint32
        )
        
        with patch.object(manager, 'prepare_shard') as mock_prepare:
            mock_task = AsyncMock()
            mock_prepare.return_value = mock_task
            
            with patch('src.tplr.sharded_dataset.SharedShardedDataset') as mock_dataset_class:
                mock_dataset = Mock()
                mock_dataset_class.return_value = mock_dataset
                
                result = await manager.create_dataset(5)
                
                mock_dataset_class.assert_called_once_with(
                    shard_index=5,
                    sequence_length=256,
                    rank=1,
                    world_size=2,
                    token_dtype=np.uint32
                )
    
    @pytest.mark.asyncio
    async def test_initialize_datasets(self, manager):
        """Test initialize_datasets method."""
        with patch.object(manager, 'create_dataset') as mock_create:
            with patch.object(manager, 'prepare_shard') as mock_prepare:
                mock_dataset = Mock()
                mock_create.return_value = mock_dataset
                mock_task = Mock()
                mock_prepare.return_value = mock_task
                
                await manager.initialize_datasets(5)
                
                mock_create.assert_called_once_with(5)
                mock_prepare.assert_called_once_with(6)  # next shard
                assert manager.active_dataset == mock_dataset
                assert manager.upcoming_dataset == mock_task
    
    @pytest.mark.asyncio
    async def test_initialize_datasets_wrap_around(self, manager):
        """Test initialize_datasets wraps around at max_dataset_idx."""
        with patch.object(manager, 'create_dataset') as mock_create:
            with patch.object(manager, 'prepare_shard') as mock_prepare:
                mock_dataset = Mock()
                mock_create.return_value = mock_dataset
                mock_task = Mock()
                mock_prepare.return_value = mock_task
                
                await manager.initialize_datasets(9)  # max_dataset_idx - 1
                
                mock_create.assert_called_once_with(9)
                mock_prepare.assert_called_once_with(0)  # wraps to 0
    
    @pytest.mark.asyncio
    async def test_initialize_datasets_at_max_boundary(self, manager):
        """Test initialize_datasets at max_dataset_idx boundary."""
        # Set max_dataset_idx to test edge case
        manager.max_dataset_idx = 5
        
        with patch.object(manager, 'create_dataset') as mock_create:
            with patch.object(manager, 'prepare_shard') as mock_prepare:
                mock_dataset = Mock()
                mock_create.return_value = mock_dataset
                mock_task = Mock()
                mock_prepare.return_value = mock_task
                
                await manager.initialize_datasets(4)  # max_dataset_idx - 1
                
                mock_create.assert_called_once_with(4)
                mock_prepare.assert_called_once_with(0)  # wraps to 0
    
    @pytest.mark.asyncio
    @patch('tplr.logger.info')
    @patch('tplr.logger.error')
    @patch('os.remove')
    async def test_swap_datasets_successful_deletion(self, mock_remove, mock_error, mock_info, manager):
        """Test swap_datasets with successful file deletion."""
        # Set up initial state
        old_dataset = Mock()
        old_dataset.tokens_file = '/old/tokens.npy'
        old_dataset.ids_file = '/old/ids.bin'
        manager.active_dataset = old_dataset
        manager.shard_index = 2
        
        # Mock upcoming dataset
        upcoming_task = AsyncMock()
        manager.upcoming_dataset = upcoming_task
        
        with patch.object(manager, 'initialize_datasets') as mock_init:
            result = await manager.swap_datasets()
            
            # Verify shard index was incremented and wrapped
            assert manager.shard_index == 3
            assert result == 3
            
            # Verify upcoming dataset was awaited
            upcoming_task.assert_awaited_once()
            
            # Verify initialization was called with new shard index
            mock_init.assert_called_once_with(3)
            
            # Verify logging
            mock_info.assert_called_once_with("successfully swapped datasets.")
            
            # Verify file deletion (rank 0)
            assert mock_remove.call_count == 2
            mock_remove.assert_any_call('/old/tokens.npy')
            mock_remove.assert_any_call('/old/ids.bin')
            
            # Verify no error logging
            mock_error.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('tplr.logger.info')
    @patch('tplr.logger.error')
    @patch('os.remove')
    async def test_swap_datasets_deletion_failure(self, mock_remove, mock_error, mock_info, manager):
        """Test swap_datasets with file deletion failure."""
        # Set up initial state
        old_dataset = Mock()
        old_dataset.tokens_file = '/old/tokens.npy'
        old_dataset.ids_file = '/old/ids.bin'
        manager.active_dataset = old_dataset
        
        # Mock file deletion failure
        mock_remove.side_effect = FileNotFoundError("File not found")
        
        with patch.object(manager, 'initialize_datasets') as mock_init:
            await manager.swap_datasets()
            
            # Verify error logging
            assert mock_error.call_count == 2
            mock_error.assert_any_call("tokens_file file not available for deletion")
            mock_error.assert_any_call("ids_file file not available for deletion")
    
    @pytest.mark.asyncio
    @patch('os.remove')
    async def test_swap_datasets_non_rank_zero(self, mock_remove, manager):
        """Test swap_datasets on non-rank-zero process doesn't delete files."""
        manager.rank = 1  # Not rank 0
        old_dataset = Mock()
        manager.active_dataset = old_dataset
        
        with patch.object(manager, 'initialize_datasets'):
            await manager.swap_datasets()
            
            # Verify no file deletion attempted
            mock_remove.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_swap_datasets_no_upcoming_dataset(self, manager):
        """Test swap_datasets when there's no upcoming dataset."""
        manager.upcoming_dataset = None
        old_dataset = Mock()
        manager.active_dataset = old_dataset
        
        with patch.object(manager, 'initialize_datasets'):
            # Should not raise an exception
            await manager.swap_datasets()
    
    @pytest.mark.asyncio
    async def test_swap_datasets_no_old_dataset(self, manager):
        """Test swap_datasets when there's no old dataset."""
        manager.active_dataset = None
        
        with patch.object(manager, 'initialize_datasets'):
            with patch('os.remove') as mock_remove:
                await manager.swap_datasets()
                
                # Should not attempt file deletion
                mock_remove.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_swap_datasets_shard_index_wraparound(self, manager):
        """Test swap_datasets properly wraps shard index."""
        manager.shard_index = 9  # max_dataset_idx - 1
        
        with patch.object(manager, 'initialize_datasets'):
            result = await manager.swap_datasets()
            
            assert manager.shard_index == 0  # Should wrap to 0
            assert result == 0
    
    @pytest.mark.asyncio
    @patch('tplr.logger.error')
    @patch('os.remove')
    async def test_swap_datasets_mixed_deletion_results(self, mock_remove, mock_error, manager):
        """Test swap_datasets when some files delete successfully and others fail."""
        old_dataset = Mock()
        old_dataset.tokens_file = '/old/tokens.npy'
        old_dataset.ids_file = '/old/ids.bin'
        manager.active_dataset = old_dataset
        
        # Mock deletion to succeed for first file, fail for second
        def mock_remove_side_effect(path):
            if 'ids' in path:
                raise FileNotFoundError("File not found")
        
        mock_remove.side_effect = mock_remove_side_effect
        
        with patch.object(manager, 'initialize_datasets'):
            await manager.swap_datasets()
            
            # Should have attempted both deletions
            assert mock_remove.call_count == 2
            # Should have logged error for the failed deletion
            mock_error.assert_called_once_with("ids_file file not available for deletion")


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_shared_sharded_dataset_locate_shards_edge_cases(self):
        """Test locate_shards edge cases."""
        # Test with negative shard index (should still format correctly)
        with pytest.raises(ValueError):
            SharedShardedDataset.locate_shards(-1, custom_path="")
    
    def test_dataset_manager_max_dataset_idx_boundary(self):
        """Test dataset manager behavior at max_dataset_idx boundary."""
        mock_comms = Mock()
        manager = ShardedDatasetManager(
            sequence_length=512,
            rank=0,
            world_size=1,
            comms=mock_comms
        )
        
        # Test boundary conditions
        manager.shard_index = 9  # max_dataset_idx - 1
        next_shard = (manager.shard_index + 1) % manager.max_dataset_idx
        assert next_shard == 0
        
        manager.shard_index = 10  # At max_dataset_idx
        next_shard = (manager.shard_index + 1) % manager.max_dataset_idx
        assert next_shard == 1
    
    def test_dataset_manager_zero_max_dataset_idx(self):
        """Test dataset manager with zero max_dataset_idx."""
        mock_comms = Mock()
        manager = ShardedDatasetManager(
            sequence_length=512,
            rank=0,
            world_size=1,
            comms=mock_comms
        )
        
        # Artificially set max_dataset_idx to 0 to test division by zero
        manager.max_dataset_idx = 0
        
        # This should raise ZeroDivisionError in modulo operations
        with pytest.raises(ZeroDivisionError):
            next_shard = (manager.shard_index + 1) % manager.max_dataset_idx
    
    def test_shared_sharded_dataset_empty_path_handling(self):
        """Test SharedShardedDataset handles empty paths gracefully."""
        # Test with empty custom path
        with pytest.raises(ValueError):
            SharedShardedDataset.locate_shards(0, custom_path="")
        
        # Test with None environment variable
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Dataset path not configured"):
                SharedShardedDataset.locate_shards(0)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.mark.asyncio
    async def test_full_dataset_lifecycle(self, temp_dir):
        """Test complete dataset lifecycle from creation to swap."""
        # Create sample data files
        for i in range(3):
            tokens_file = os.path.join(temp_dir, f"train_{i:06d}.npy")
            sample_tokens = np.array([i*10 + j for j in range(10)], dtype=np.uint16)
            np.save(tokens_file, sample_tokens)
            
            ids_file = os.path.join(temp_dir, f"sample_ids_{i:06d}.bin")
            sample_ids = np.array([i*100, i*100 + 1], dtype=np.uint64)
            sample_ids.tofile(ids_file)
        
        # Mock comms
        mock_comms = Mock()
        mock_comms.get_own_bucket.return_value = Mock()
        mock_comms.s3_get_object = AsyncMock()
        
        # Create manager
        manager = ShardedDatasetManager(
            sequence_length=5,
            rank=0,
            world_size=1,
            comms=mock_comms,
            token_dtype=np.uint16
        )
        manager.max_dataset_idx = 3  # Limit for testing
        
        with patch.dict(os.environ, {'DATASET_BINS_PATH': temp_dir}):
            with patch('tplr.logger.info'):
                # Initialize with first dataset
                await manager.initialize_datasets(0)
                
                assert manager.active_dataset is not None
                assert len(manager.active_dataset) == 2  # 2 samples per shard
                
                # Swap to next dataset
                new_shard_idx = await manager.swap_datasets()
                assert new_shard_idx == 1
                assert manager.shard_index == 1
                
                # Verify new dataset is loaded
                assert manager.active_dataset is not None
                assert len(manager.active_dataset) == 2
    
    @pytest.mark.asyncio
    async def test_dataset_manager_multiple_swaps(self, temp_dir):
        """Test multiple dataset swaps in sequence."""
        # Create sample data files for multiple shards
        for i in range(5):
            tokens_file = os.path.join(temp_dir, f"train_{i:06d}.npy")
            sample_tokens = np.array([i*10 + j for j in range(8)], dtype=np.uint16)
            np.save(tokens_file, sample_tokens)
            
            ids_file = os.path.join(temp_dir, f"sample_ids_{i:06d}.bin")
            sample_ids = np.array([i*100, i*100 + 1], dtype=np.uint64)
            sample_ids.tofile(ids_file)
        
        # Mock comms
        mock_comms = Mock()
        mock_comms.get_own_bucket.return_value = Mock()
        mock_comms.s3_get_object = AsyncMock()
        
        # Create manager
        manager = ShardedDatasetManager(
            sequence_length=4,
            rank=0,
            world_size=1,
            comms=mock_comms,
            token_dtype=np.uint16
        )
        manager.max_dataset_idx = 5
        
        with patch.dict(os.environ, {'DATASET_BINS_PATH': temp_dir}):
            with patch('tplr.logger.info'):
                with patch('tplr.logger.error'):
                    # Initialize with first dataset
                    await manager.initialize_datasets(0)
                    assert manager.shard_index == 0
                    
                    # Perform multiple swaps
                    for expected_idx in [1, 2, 3, 4, 0]:  # Should wrap around
                        new_idx = await manager.swap_datasets()
                        assert new_idx == expected_idx
                        assert manager.shard_index == expected_idx