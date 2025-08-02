# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import asyncio
import os
import sys
import time
import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock, call
from datetime import datetime, timedelta, timezone
from contextlib import nullcontext

import torch
import torch.distributed as dist
import numpy as np
import pytest

# Mock external dependencies before importing the module under test
sys.modules['bittensor'] = Mock()
sys.modules['uvloop'] = Mock()
sys.modules['tplr'] = Mock()
sys.modules['tplr.compress'] = Mock()
sys.modules['tplr.comms'] = Mock()
sys.modules['tplr.sharded_dataset'] = Mock()
sys.modules['tplr.metrics'] = Mock()
sys.modules['tplr.neurons'] = Mock()
sys.modules['transformers'] = Mock()
sys.modules['transformers.models'] = Mock()
sys.modules['transformers.models.llama'] = Mock()
sys.modules['neurons'] = Mock()

# Import after mocking
from tests.test_miner import Miner


class TestMinerConfiguration(unittest.TestCase):
    """Test suite for Miner configuration and argument parsing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_argv = sys.argv.copy()
        
    def tearDown(self):
        """Clean up after tests."""
        sys.argv = self.original_argv
    
    def test_miner_config_default_values(self):
        """Test that miner_config returns correct default values."""
        sys.argv = ['test']
        with patch('argparse.ArgumentParser.parse_args') as mock_parse:
            mock_parse.return_value = Mock(
                netuid=268,
                project='templar',
                actual_batch_size=None,
                device='cuda',
                local_rank=0,
                debug=False,
                trace=False,
                store_gathers=False,
                test=False,
                local=False
            )
            with patch('tplr.debug') as mock_debug, \
                 patch('tplr.trace') as mock_trace, \
                 patch('bittensor.config') as mock_bt_config:
                
                config = Miner.miner_config()
                
                self.assertEqual(config.netuid, 268)
                self.assertEqual(config.project, 'templar')
                self.assertEqual(config.device, 'cuda')
                self.assertFalse(config.debug)
                self.assertFalse(config.trace)
                mock_debug.assert_not_called()
                mock_trace.assert_not_called()

    def test_miner_config_debug_enabled(self):
        """Test that debug mode is properly enabled."""
        sys.argv = ['test', '--debug']
        with patch('argparse.ArgumentParser.parse_args') as mock_parse:
            mock_parse.return_value = Mock(debug=True, trace=False)
            with patch('tplr.debug') as mock_debug, \
                 patch('tplr.trace') as mock_trace, \
                 patch('bittensor.config') as mock_bt_config:
                
                Miner.miner_config()
                mock_debug.assert_called_once()
                mock_trace.assert_not_called()

    def test_miner_config_trace_enabled(self):
        """Test that trace mode is properly enabled."""
        sys.argv = ['test', '--trace']
        with patch('argparse.ArgumentParser.parse_args') as mock_parse:
            mock_parse.return_value = Mock(debug=False, trace=True)
            with patch('tplr.debug') as mock_debug, \
                 patch('tplr.trace') as mock_trace, \
                 patch('bittensor.config') as mock_bt_config:
                
                Miner.miner_config()
                mock_debug.assert_not_called()
                mock_trace.assert_called_once()

    def test_miner_config_custom_netuid(self):
        """Test custom netuid configuration."""
        sys.argv = ['test', '--netuid', '123']
        with patch('argparse.ArgumentParser.parse_args') as mock_parse:
            mock_parse.return_value = Mock(netuid=123, debug=False, trace=False)
            with patch('bittensor.config') as mock_bt_config:
                config = Miner.miner_config()
                self.assertEqual(config.netuid, 123)

    def test_miner_config_local_rank_from_env(self):
        """Test local_rank takes value from environment variable."""
        with patch.dict(os.environ, {'LOCAL_RANK': '2'}):
            sys.argv = ['test']
            with patch('argparse.ArgumentParser.parse_args') as mock_parse:
                mock_parse.return_value = Mock(local_rank=2, debug=False, trace=False)
                with patch('bittensor.config') as mock_bt_config:
                    config = Miner.miner_config()
                    self.assertEqual(config.local_rank, 2)


class TestMinerDistributedUtils(unittest.TestCase):
    """Test suite for distributed utility methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
    
    def test_should_continue_all_ranks_have_batch(self):
        """Test should_continue when all ranks have batches."""
        with patch('torch.distributed.all_reduce') as mock_all_reduce:
            mock_all_reduce.side_effect = lambda tensor, op: tensor.fill_(1)
            
            result = Miner.should_continue(True, self.device)
            self.assertTrue(result)
            mock_all_reduce.assert_called_once()

    def test_should_continue_some_rank_exhausted(self):
        """Test should_continue when at least one rank is exhausted."""
        with patch('torch.distributed.all_reduce') as mock_all_reduce:
            mock_all_reduce.side_effect = lambda tensor, op: tensor.fill_(0)
            
            result = Miner.should_continue(True, self.device)
            self.assertFalse(result)
            mock_all_reduce.assert_called_once()

    def test_should_continue_local_rank_exhausted(self):
        """Test should_continue when local rank is exhausted."""
        with patch('torch.distributed.all_reduce') as mock_all_reduce:
            mock_all_reduce.side_effect = lambda tensor, op: tensor.fill_(0)
            
            result = Miner.should_continue(False, self.device)
            self.assertFalse(result)

    def test_is_distributed_not_available(self):
        """Test _is_distributed when torch.distributed is not available."""
        miner = Mock()
        miner.world_size = 1
        
        with patch('torch.distributed.is_available', return_value=False):
            result = Miner._is_distributed(miner)
            self.assertFalse(result)

    def test_is_distributed_not_initialized(self):
        """Test _is_distributed when torch.distributed is not initialized."""
        miner = Mock()
        miner.world_size = 2
        
        with patch('torch.distributed.is_available', return_value=True), \
             patch('torch.distributed.is_initialized', return_value=False):
            result = Miner._is_distributed(miner)
            self.assertFalse(result)

    def test_is_distributed_single_gpu(self):
        """Test _is_distributed with single GPU setup."""
        miner = Mock()
        miner.world_size = 1
        
        with patch('torch.distributed.is_available', return_value=True), \
             patch('torch.distributed.is_initialized', return_value=True):
            result = Miner._is_distributed(miner)
            self.assertFalse(result)

    def test_is_distributed_multi_gpu(self):
        """Test _is_distributed with multi-GPU setup."""
        miner = Mock()
        miner.world_size = 4
        
        with patch('torch.distributed.is_available', return_value=True), \
             patch('torch.distributed.is_initialized', return_value=True):
            result = Miner._is_distributed(miner)
            self.assertTrue(result)

    def test_ddp_reduce_single_gpu_int(self):
        """Test _ddp_reduce with single GPU and integer value."""
        miner = Mock()
        miner.device = torch.device('cpu')
        
        with patch.object(Miner, '_is_distributed', return_value=False):
            result = Miner._ddp_reduce(miner, 42)
            self.assertEqual(result, 42.0)

    def test_ddp_reduce_single_gpu_tensor(self):
        """Test _ddp_reduce with single GPU and tensor value."""
        miner = Mock()
        miner.device = torch.device('cpu')
        tensor_value = torch.tensor(5.5)
        
        with patch.object(Miner, '_is_distributed', return_value=False):
            result = Miner._ddp_reduce(miner, tensor_value)
            self.assertEqual(result, 5.5)

    def test_ddp_reduce_multi_gpu_sum(self):
        """Test _ddp_reduce with multi-GPU setup using SUM operation."""
        miner = Mock()
        miner.device = torch.device('cpu')
        
        with patch.object(Miner, '_is_distributed', return_value=True), \
             patch('torch.tensor') as mock_tensor, \
             patch('torch.distributed.all_reduce') as mock_all_reduce:
            
            mock_tensor_obj = Mock()
            mock_tensor_obj.item.return_value = 10.0
            mock_tensor.return_value = mock_tensor_obj
            
            result = Miner._ddp_reduce(miner, 5, dist.ReduceOp.SUM)
            self.assertEqual(result, 10.0)
            mock_all_reduce.assert_called_once()

    def test_ddp_reduce_multi_gpu_avg(self):
        """Test _ddp_reduce with multi-GPU setup using AVG operation."""
        miner = Mock()
        miner.device = torch.device('cpu')
        
        with patch.object(Miner, '_is_distributed', return_value=True), \
             patch('torch.tensor') as mock_tensor, \
             patch('torch.distributed.all_reduce') as mock_all_reduce:
            
            mock_tensor_obj = Mock()
            mock_tensor_obj.item.return_value = 2.5
            mock_tensor.return_value = mock_tensor_obj
            
            result = Miner._ddp_reduce(miner, 5, dist.ReduceOp.AVG)
            self.assertEqual(result, 2.5)
            mock_all_reduce.assert_called_once()


class TestMinerInitialization(unittest.TestCase):
    """Test suite for Miner initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_env = os.environ.copy()
        
    def tearDown(self):
        """Clean up after tests."""
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch('torch.distributed.init_process_group')
    @patch('torch.cuda.set_device')
    @patch('tplr.load_hparams')
    @patch('bittensor.wallet')
    @patch('bittensor.subtensor')
    @patch('transformers.models.llama.LlamaForCausalLM')
    @patch('torch.compile')
    @patch('tplr.compress.TransformDCT')
    @patch('tplr.compress.CompressDCT')
    @patch('tplr.comms.Comms')
    @patch('tplr.sharded_dataset.ShardedDatasetManager')
    @patch('tplr.initialize_wandb')
    @patch('tplr.metrics.MetricsLogger')
    def test_miner_initialization_single_gpu(self, mock_metrics, mock_wandb, mock_dataset, 
                                           mock_comms, mock_compress_dct, mock_transform_dct,
                                           mock_compile, mock_llama, mock_subtensor, mock_wallet,
                                           mock_hparams, mock_set_device, mock_init_pg):
        """Test Miner initialization with single GPU setup."""
        
        # Set up environment for single GPU
        os.environ.update({
            'RANK': '0',
            'WORLD_SIZE': '1',
            'LOCAL_RANK': '0'
        })
        
        # Mock configuration
        mock_config = Mock()
        mock_config.local = False
        mock_config.actual_batch_size = None
        mock_config.netuid = 268
        mock_config.device = 'cuda'
        mock_config.peers = None
        
        # Mock hyperparameters
        mock_hparams_obj = Mock()
        mock_hparams_obj.batch_size = 32
        mock_hparams_obj.model_config = Mock()
        mock_hparams_obj.target_chunk = 1024
        mock_hparams_obj.quantization_bins = 256
        mock_hparams_obj.quantization_range = (-1, 1)
        mock_hparams_obj.outer_learning_rate = 0.01
        mock_hparams_obj.inner_learning_rate = 0.001
        mock_hparams_obj.weight_decay = 0.01
        mock_hparams_obj.inner_steps = 10
        mock_hparams_obj.validator_offset = 1
        mock_hparams_obj.peer_list_window_margin = 2
        mock_hparams_obj.warmup_steps = 100
        mock_hparams_obj.t_max = 1000
        mock_hparams_obj.topk_compression = 0.1
        mock_hparams_obj.use_dct = True
        mock_hparams_obj.blocks_per_window = 50
        mock_hparams_obj.sequence_length = 512
        mock_hparams_obj.checkpoint_frequency = 100
        mock_hparams_obj.windows_per_shard = 100
        mock_hparams_obj.tokenizer = Mock()
        mock_hparams_obj.tokenizer.pad_token_id = 0
        mock_hparams.return_value = mock_hparams_obj
        
        # Mock wallet and metagraph
        mock_wallet_obj = Mock()
        mock_wallet_obj.hotkey.ss58_address = 'test_hotkey'
        mock_wallet.return_value = mock_wallet_obj
        
        mock_subtensor_obj = Mock()
        mock_metagraph = Mock()
        mock_metagraph.hotkeys = ['test_hotkey', 'other_hotkey']
        mock_metagraph.netuid = 268
        mock_subtensor_obj.metagraph.return_value = mock_metagraph
        mock_subtensor_obj.block = 1000
        mock_subtensor.return_value = mock_subtensor_obj
        
        # Mock model
        mock_model = Mock()
        mock_llama.return_value = mock_model
        mock_compile.return_value = mock_model
        
        with patch.object(Miner, 'miner_config', return_value=mock_config), \
             patch('neurons.BaseNode.__init__') as mock_base_init, \
             patch('torch.manual_seed'), \
             patch('torch.cuda.manual_seed_all'), \
             patch('numpy.random.seed'), \
             patch('random.seed'):
            
            miner = Miner()
            
            # Verify initialization
            self.assertEqual(miner.rank, 0)
            self.assertEqual(miner.world_size, 1)
            self.assertEqual(miner.local_rank, 0)
            self.assertTrue(miner.is_master)
            self.assertEqual(miner.uid, 0)
            
            # Verify distributed setup was not called for single GPU
            mock_init_pg.assert_not_called()
            mock_set_device.assert_not_called()

    @patch('torch.distributed.init_process_group')
    @patch('torch.cuda.set_device')
    @patch('tplr.load_hparams')
    @patch('bittensor.wallet')
    @patch('bittensor.subtensor')
    @patch('transformers.models.llama.LlamaForCausalLM')
    @patch('torch.compile')
    @patch('torch.nn.parallel.DistributedDataParallel')
    @patch('tplr.compress.TransformDCT')
    @patch('tplr.compress.CompressDCT')
    @patch('torch.optim.SGD')
    @patch('torch.distributed.optim.ZeroRedundancyOptimizer')
    @patch('tplr.comms.Comms')
    @patch('tplr.sharded_dataset.ShardedDatasetManager')
    def test_miner_initialization_multi_gpu(self, mock_dataset, mock_comms, mock_zero_optimizer,
                                          mock_sgd, mock_compress_dct, mock_transform_dct,
                                          mock_ddp, mock_compile, mock_llama, mock_subtensor,
                                          mock_wallet, mock_hparams, mock_set_device, mock_init_pg):
        """Test Miner initialization with multi-GPU setup."""
        
        # Set up environment for multi-GPU
        os.environ.update({
            'RANK': '1',
            'WORLD_SIZE': '4',
            'LOCAL_RANK': '1'
        })
        
        # Mock configuration
        mock_config = Mock()
        mock_config.local = False
        mock_config.actual_batch_size = None
        mock_config.netuid = 268
        mock_config.device = 'cuda'
        mock_config.peers = None
        
        # Mock hyperparameters with all required attributes
        mock_hparams_obj = Mock()
        mock_hparams_obj.batch_size = 32
        mock_hparams_obj.model_config = Mock()
        mock_hparams_obj.target_chunk = 1024
        mock_hparams_obj.quantization_bins = 256
        mock_hparams_obj.quantization_range = (-1, 1)
        mock_hparams_obj.outer_learning_rate = 0.01
        mock_hparams_obj.inner_learning_rate = 0.001
        mock_hparams_obj.weight_decay = 0.01
        mock_hparams_obj.inner_steps = 10
        mock_hparams_obj.validator_offset = 1
        mock_hparams_obj.peer_list_window_margin = 2
        mock_hparams_obj.warmup_steps = 100
        mock_hparams_obj.t_max = 1000
        mock_hparams_obj.topk_compression = 0.1
        mock_hparams_obj.use_dct = True
        mock_hparams_obj.blocks_per_window = 50
        mock_hparams_obj.sequence_length = 512
        mock_hparams_obj.checkpoint_frequency = 100
        mock_hparams_obj.windows_per_shard = 100
        mock_hparams_obj.tokenizer = Mock()
        mock_hparams_obj.tokenizer.pad_token_id = 0
        mock_hparams.return_value = mock_hparams_obj
        
        # Mock wallet and metagraph
        mock_wallet_obj = Mock()
        mock_wallet_obj.hotkey.ss58_address = 'test_hotkey'
        mock_wallet.return_value = mock_wallet_obj
        
        mock_subtensor_obj = Mock()
        mock_metagraph = Mock()
        mock_metagraph.hotkeys = ['test_hotkey', 'other_hotkey']
        mock_metagraph.netuid = 268
        mock_subtensor_obj.metagraph.return_value = mock_metagraph
        mock_subtensor_obj.block = 1000
        mock_subtensor.return_value = mock_subtensor_obj
        
        # Mock model
        mock_model = Mock()
        mock_model.named_parameters.return_value = [
            ('layer1.weight', torch.randn(10, 10)),
            ('layer2.weight', torch.randn(20, 20))
        ]
        mock_llama.return_value = mock_model
        mock_compile.return_value = mock_model
        mock_ddp.return_value = mock_model
        
        with patch.object(Miner, 'miner_config', return_value=mock_config), \
             patch('neurons.BaseNode.__init__') as mock_base_init, \
             patch('torch.manual_seed'), \
             patch('torch.cuda.manual_seed_all'), \
             patch('numpy.random.seed'), \
             patch('random.seed'), \
             patch('torch.distributed.barrier'):
            
            miner = Miner()
            
            # Verify multi-GPU initialization
            self.assertEqual(miner.rank, 1)
            self.assertEqual(miner.world_size, 4)
            self.assertEqual(miner.local_rank, 1)
            self.assertFalse(miner.is_master)
            self.assertEqual(miner.uid, 0)
            
            # Verify distributed setup was called
            mock_init_pg.assert_called_once()
            mock_set_device.assert_called_once_with(1)
            mock_ddp.assert_called_once()

    def test_miner_initialization_unregistered_wallet(self):
        """Test that Miner exits when wallet is not registered."""
        os.environ.update({
            'RANK': '0',
            'WORLD_SIZE': '1',
            'LOCAL_RANK': '0'
        })
        
        mock_config = Mock()
        mock_config.local = False
        mock_config.actual_batch_size = None
        mock_config.netuid = 268
        mock_config.device = 'cuda'
        
        mock_hparams_obj = Mock()
        mock_hparams_obj.batch_size = 32
        mock_hparams_obj.model_config = Mock()
        
        mock_wallet_obj = Mock()
        mock_wallet_obj.hotkey.ss58_address = 'unregistered_hotkey'
        
        mock_subtensor_obj = Mock()
        mock_metagraph = Mock()
        mock_metagraph.hotkeys = ['registered_hotkey1', 'registered_hotkey2']
        mock_metagraph.netuid = 268
        mock_subtensor_obj.metagraph.return_value = mock_metagraph
        mock_subtensor_obj.block = 1000
        
        with patch.object(Miner, 'miner_config', return_value=mock_config), \
             patch('tplr.load_hparams', return_value=mock_hparams_obj), \
             patch('bittensor.wallet', return_value=mock_wallet_obj), \
             patch('bittensor.subtensor', return_value=mock_subtensor_obj), \
             patch('sys.exit') as mock_exit, \
             patch('torch.manual_seed'), \
             patch('torch.cuda.manual_seed_all'), \
             patch('numpy.random.seed'), \
             patch('random.seed'):
            
            miner = Miner()
            mock_exit.assert_called_once()

    def test_batch_size_override(self):
        """Test that actual_batch_size overrides hparams batch_size."""
        os.environ.update({
            'RANK': '0',
            'WORLD_SIZE': '1',
            'LOCAL_RANK': '0'
        })
        
        mock_config = Mock()
        mock_config.local = False
        mock_config.actual_batch_size = 64  # Override value
        mock_config.netuid = 268
        mock_config.device = 'cuda'
        mock_config.peers = None
        
        mock_hparams_obj = Mock()
        mock_hparams_obj.batch_size = 32  # Original value
        
        # Add all required attributes
        for attr in ['model_config', 'target_chunk', 'quantization_bins', 'quantization_range',
                    'outer_learning_rate', 'inner_learning_rate', 'weight_decay', 'inner_steps',
                    'validator_offset', 'peer_list_window_margin', 'warmup_steps', 't_max',
                    'topk_compression', 'use_dct', 'blocks_per_window', 'sequence_length',
                    'checkpoint_frequency', 'windows_per_shard']:
            setattr(mock_hparams_obj, attr, Mock())
        
        mock_hparams_obj.tokenizer = Mock()
        mock_hparams_obj.tokenizer.pad_token_id = 0
        
        with patch.object(Miner, 'miner_config', return_value=mock_config), \
             patch('tplr.load_hparams', return_value=mock_hparams_obj), \
             patch('bittensor.wallet') as mock_wallet, \
             patch('bittensor.subtensor') as mock_subtensor, \
             patch('transformers.models.llama.LlamaForCausalLM'), \
             patch('torch.compile'), \
             patch('tplr.compress.TransformDCT'), \
             patch('tplr.compress.CompressDCT'), \
             patch('torch.optim.SGD'), \
             patch('torch.distributed.optim.ZeroRedundancyOptimizer'), \
             patch('tplr.comms.Comms'), \
             patch('tplr.sharded_dataset.ShardedDatasetManager'), \
             patch('neurons.BaseNode.__init__'), \
             patch('torch.manual_seed'), \
             patch('torch.cuda.manual_seed_all'), \
             patch('numpy.random.seed'), \
             patch('random.seed'):
            
            # Setup wallet and metagraph mocks
            mock_wallet_obj = Mock()
            mock_wallet_obj.hotkey.ss58_address = 'test_hotkey'
            mock_wallet.return_value = mock_wallet_obj
            
            mock_subtensor_obj = Mock()
            mock_metagraph = Mock()
            mock_metagraph.hotkeys = ['test_hotkey']
            mock_metagraph.netuid = 268
            mock_subtensor_obj.metagraph.return_value = mock_metagraph
            mock_subtensor_obj.block = 1000
            mock_subtensor.return_value = mock_subtensor_obj
            
            miner = Miner()
            
            # Verify batch size was overridden
            self.assertEqual(miner.hparams.batch_size, 64)


class TestMinerInnerSteps(unittest.IsolatedAsyncioTestCase):
    """Test suite for inner_steps method."""
    
    async def setUp(self):
        """Set up async test fixtures."""
        self.miner = Mock()
        self.miner.stop_event = Mock()
        self.miner.stop_event.is_set.return_value = False
        self.miner.loop = Mock()
        self.miner.world_size = 1
        self.miner.is_master = True
        self.miner.device = torch.device('cpu')
        self.miner.tokenizer = Mock()
        self.miner.tokenizer.pad_token_id = 0
        self.miner.model = Mock()
        self.miner.sampler = Mock()
        self.miner.sampler.grad_accum_steps = 2
        self.miner.hparams = Mock()
        self.miner.hparams.inner_steps = 5
        self.miner.inner_optimizer = Mock()
        self.miner.inner_scheduler = Mock()
        self.miner.bare_model = Mock()
        self.miner.bare_model.parameters.return_value = [torch.randn(10, 10)]
        
        # Mock methods
        self.miner._get_offloaded_param = Mock(return_value=[torch.randn(10, 10)])
        self.miner._ddp_reduce = Mock(side_effect=lambda x, op=None: float(x) if isinstance(x, (int, float)) else float(x.item()) if hasattr(x, 'item') else float(x))

    async def test_inner_steps_single_batch(self):
        """Test inner_steps with a single batch."""
        # Mock data loader
        mock_loader = Mock()
        mock_batch = torch.randint(0, 1000, (4, 10))  # batch_size=4, seq_len=10
        
        async def mock_run_in_executor(executor, func, *args):
            if func == next:
                return mock_batch
        
        self.miner.loop.run_in_executor = mock_run_in_executor
        
        # Mock model output
        mock_output = Mock()
        mock_output.loss = torch.tensor(2.0)
        self.miner.model.return_value = mock_output
        
        # Mock autocast context
        with patch('torch.autocast', return_value=nullcontext()), \
             patch('torch.nn.utils.clip_grad_norm_'), \
             patch('asyncio.sleep'):
            
            result = await Miner.inner_steps(
                self.miner, 
                loader=mock_loader, 
                step_window=100
            )
            
            # Verify result structure
            self.assertIn('total_loss', result)
            self.assertIn('window_entry_loss', result)
            self.assertIn('batch_count', result)
            self.assertIn('batch_tokens', result)
            
            # Verify optimizer was called
            self.miner.inner_optimizer.step.assert_called()
            self.miner.inner_scheduler.step.assert_called()

    async def test_inner_steps_empty_loader(self):
        """Test inner_steps with empty data loader."""
        mock_loader = Mock()
        
        async def mock_run_in_executor(executor, func, *args):
            if func == next:
                raise StopIteration()
        
        self.miner.loop.run_in_executor = mock_run_in_executor
        
        with patch('asyncio.sleep'):
            result = await Miner.inner_steps(
                self.miner, 
                loader=mock_loader, 
                step_window=100
            )
            
            # Should return zero values for empty loader
            self.assertEqual(result['batch_count'], 0)
            self.assertEqual(result['batch_tokens'], 0)

    async def test_inner_steps_window_change(self):
        """Test inner_steps behavior when window changes."""
        self.miner.current_window = 101  # Different from step_window
        
        mock_loader = Mock()
        mock_batch = torch.randint(0, 1000, (2, 10))
        
        async def mock_run_in_executor(executor, func, *args):
            if func == next:
                return mock_batch
        
        self.miner.loop.run_in_executor = mock_run_in_executor
        
        mock_output = Mock()
        mock_output.loss = torch.tensor(1.5)
        self.miner.model.return_value = mock_output
        
        with patch('torch.autocast', return_value=nullcontext()), \
             patch('torch.nn.utils.clip_grad_norm_'), \
             patch('asyncio.sleep'):
            
            result = await Miner.inner_steps(
                self.miner, 
                loader=mock_loader, 
                step_window=100
            )
            
            # Should exit early due to window change
            self.assertGreaterEqual(result['batch_count'], 0)

    async def test_inner_steps_gradient_accumulation(self):
        """Test gradient accumulation logic in inner_steps."""
        self.miner.sampler.grad_accum_steps = 4  # Accumulate over 4 micro-batches
        
        mock_loader = Mock()
        mock_batch = torch.randint(0, 1000, (1, 10))
        
        call_count = 0
        async def mock_run_in_executor(executor, func, *args):
            nonlocal call_count
            if func == next:
                call_count += 1
                if call_count <= 8:  # Return 8 batches total
                    return mock_batch
                else:
                    raise StopIteration()
        
        self.miner.loop.run_in_executor = mock_run_in_executor
        
        mock_output = Mock()
        mock_output.loss = torch.tensor(1.0)
        self.miner.model.return_value = mock_output
        
        with patch('torch.autocast', return_value=nullcontext()), \
             patch('torch.nn.utils.clip_grad_norm_'), \
             patch('asyncio.sleep'):
            
            result = await Miner.inner_steps(
                self.miner, 
                loader=mock_loader, 
                step_window=100
            )
            
            # Should have processed 8 batches
            self.assertEqual(result['batch_count'], 8)
            # Should have taken 2 optimizer steps (8 batches / 4 accumulation steps)
            self.assertEqual(self.miner.inner_optimizer.step.call_count, 2)


class TestMinerUtilityMethods(unittest.TestCase):
    """Test suite for utility methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.miner = Mock()
        self.miner.bare_model = Mock()
        
    def test_get_offloaded_param(self):
        """Test _get_offloaded_param method."""
        # Mock model parameters
        param1 = torch.randn(10, 10)
        param2 = torch.randn(5, 5)
        self.miner.bare_model.parameters.return_value = [param1, param2]
        
        result = Miner._get_offloaded_param(self.miner)
        
        # Should return CPU copies of parameters
        self.assertEqual(len(result), 2)
        for copied_param, original_param in zip(result, [param1, param2]):
            self.assertTrue(torch.equal(copied_param, original_param.cpu()))
            self.assertEqual(copied_param.device, torch.device('cpu'))

    def test_get_offloaded_param_empty_model(self):
        """Test _get_offloaded_param with model having no parameters."""
        self.miner.bare_model.parameters.return_value = []
        
        result = Miner._get_offloaded_param(self.miner)
        
        self.assertEqual(len(result), 0)

    def test_get_offloaded_param_large_tensors(self):
        """Test _get_offloaded_param with large tensors."""
        # Create large tensors to test memory handling
        large_param = torch.randn(1000, 1000)
        self.miner.bare_model.parameters.return_value = [large_param]
        
        result = Miner._get_offloaded_param(self.miner)
        
        self.assertEqual(len(result), 1)
        self.assertTrue(torch.equal(result[0], large_param.cpu()))
        self.assertEqual(result[0].device, torch.device('cpu'))


class TestMinerAsyncMethods(unittest.IsolatedAsyncioTestCase):
    """Test suite for async methods."""
    
    async def setUp(self):
        """Set up async test fixtures."""
        self.miner = Mock()
        self.miner.model = Mock()
        self.miner.inner_optimizer = Mock()
        self.miner.device = torch.device('cpu')
        
    async def test_cleanup_window(self):
        """Test cleanup_window method."""
        with patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('torch.clear_autocast_cache') as mock_clear_cache, \
             patch('torch.cuda.memory_allocated', return_value=1024**3), \
             patch('torch.cuda.memory_reserved', return_value=2*1024**3):
            
            await Miner.cleanup_window(self.miner)
            
            # Verify cleanup methods were called
            self.miner.model.zero_grad.assert_called_once_with(set_to_none=True)
            self.miner.inner_optimizer.zero_grad.assert_called_once_with(set_to_none=True)
            mock_empty_cache.assert_called_once()
            mock_clear_cache.assert_called_once()

    def test_set_dataloader_miner_mode(self):
        """Test set_dataloader in miner mode."""
        self.miner.dataset_manager = Mock()
        self.miner.dataset_manager.active_dataset = Mock()
        self.miner.uid = 1
        self.miner.current_window = 50
        self.miner.rank = 0
        self.miner.world_size = 2
        self.miner.hparams = Mock()
        self.miner.hparams.inner_steps = 10
        self.miner.hparams.micro_batch_size = 4
        self.miner.hparams.batch_size = 32
        self.miner.hparams.target_batch_size = 128
        
        mock_sampler = Mock()
        
        with patch('tplr.MinerSampler', return_value=mock_sampler) as mock_miner_sampler, \
             patch('torch.utils.data.DataLoader') as mock_dataloader:
            
            Miner.set_dataloader(self.miner, validator=False)
            
            # Verify MinerSampler was used
            mock_miner_sampler.assert_called_once()
            mock_dataloader.assert_called_once()
            self.assertEqual(self.miner.sampler, mock_sampler)

    def test_set_dataloader_validator_mode(self):
        """Test set_dataloader in validator mode."""
        self.miner.dataset_manager = Mock()
        self.miner.dataset_manager.active_dataset = Mock()
        self.miner.uid = 1
        self.miner.current_window = 50
        self.miner.rank = 0
        self.miner.world_size = 2
        self.miner.hparams = Mock()
        self.miner.hparams.inner_steps = 10
        self.miner.hparams.micro_batch_size = 4
        self.miner.hparams.target_batch_size = 128
        self.miner.hparams.validator_sample_micro_bs = 2
        
        mock_sampler = Mock()
        
        with patch('tplr.EvalSampler', return_value=mock_sampler) as mock_eval_sampler, \
             patch('torch.utils.data.DataLoader') as mock_dataloader:
            
            Miner.set_dataloader(self.miner, validator=True)
            
            # Verify EvalSampler was used
            mock_eval_sampler.assert_called_once()
            mock_dataloader.assert_called_once()
            self.assertEqual(self.miner.sampler, mock_sampler)


class TestMinerEdgeCases(unittest.TestCase):
    """Test suite for edge cases and error conditions."""
    
    def test_miner_config_missing_environment_vars(self):
        """Test miner_config when environment variables are missing."""
        # Clear relevant environment variables
        env_backup = os.environ.copy()
        try:
            if 'LOCAL_RANK' in os.environ:
                del os.environ['LOCAL_RANK']
            
            sys.argv = ['test']
            with patch('argparse.ArgumentParser.parse_args') as mock_parse:
                mock_parse.return_value = Mock(
                    local_rank=0,  # Should default to 0
                    debug=False,
                    trace=False
                )
                with patch('bittensor.config') as mock_bt_config:
                    config = Miner.miner_config()
                    self.assertEqual(config.local_rank, 0)
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

    def test_should_continue_with_invalid_device(self):
        """Test should_continue with invalid device type."""
        invalid_device = "invalid_device"
        
        with patch('torch.distributed.all_reduce') as mock_all_reduce:
            # Should handle gracefully even with invalid device
            mock_all_reduce.side_effect = lambda tensor, op: tensor.fill_(1)
            
            result = Miner.should_continue(True, invalid_device)
            self.assertTrue(result)

    def test_ddp_reduce_with_nan_values(self):
        """Test _ddp_reduce with NaN values."""
        miner = Mock()
        miner.device = torch.device('cpu')
        
        with patch.object(Miner, '_is_distributed', return_value=False):
            result = Miner._ddp_reduce(miner, float('nan'))
            self.assertTrue(np.isnan(result))

    def test_ddp_reduce_with_inf_values(self):
        """Test _ddp_reduce with infinite values."""
        miner = Mock()
        miner.device = torch.device('cpu')
        
        with patch.object(Miner, '_is_distributed', return_value=False):
            result = Miner._ddp_reduce(miner, float('inf'))
            self.assertTrue(np.isinf(result))

    def test_ddp_reduce_with_zero_tensor(self):
        """Test _ddp_reduce with zero tensor."""
        miner = Mock()
        miner.device = torch.device('cpu')
        zero_tensor = torch.zeros(1)
        
        with patch.object(Miner, '_is_distributed', return_value=False):
            result = Miner._ddp_reduce(miner, zero_tensor)
            self.assertEqual(result, 0.0)

    def test_get_offloaded_param_with_cuda_tensors(self):
        """Test _get_offloaded_param with CUDA tensors (if available)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        miner = Mock()
        cuda_param = torch.randn(5, 5).cuda()
        miner.bare_model.parameters.return_value = [cuda_param]
        
        result = Miner._get_offloaded_param(miner)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].device, torch.device('cpu'))
        self.assertTrue(torch.equal(result[0], cuda_param.cpu()))


class TestMinerIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for Miner class combining multiple components."""
    
    async def test_miner_initialization_and_cleanup_integration(self):
        """Test integration between initialization and cleanup methods."""
        os.environ.update({
            'RANK': '0',
            'WORLD_SIZE': '1',
            'LOCAL_RANK': '0'
        })
        
        mock_config = Mock()
        mock_config.local = False
        mock_config.actual_batch_size = None
        mock_config.netuid = 268
        mock_config.device = 'cuda'
        mock_config.peers = None
        
        # Minimal hparams for integration test
        mock_hparams_obj = Mock()
        for attr in ['batch_size', 'model_config', 'target_chunk', 'quantization_bins', 
                    'quantization_range', 'outer_learning_rate', 'inner_learning_rate', 
                    'weight_decay', 'inner_steps', 'validator_offset', 'peer_list_window_margin',
                    'warmup_steps', 't_max', 'topk_compression', 'use_dct', 'blocks_per_window',
                    'sequence_length', 'checkpoint_frequency', 'windows_per_shard']:
            setattr(mock_hparams_obj, attr, 1 if 'steps' in attr else Mock())
        
        mock_hparams_obj.tokenizer = Mock()
        mock_hparams_obj.tokenizer.pad_token_id = 0
        
        with patch.object(Miner, 'miner_config', return_value=mock_config), \
             patch('tplr.load_hparams', return_value=mock_hparams_obj), \
             patch('bittensor.wallet') as mock_wallet, \
             patch('bittensor.subtensor') as mock_subtensor, \
             patch('transformers.models.llama.LlamaForCausalLM'), \
             patch('torch.compile'), \
             patch('tplr.compress.TransformDCT'), \
             patch('tplr.compress.CompressDCT'), \
             patch('torch.optim.SGD'), \
             patch('torch.distributed.optim.ZeroRedundancyOptimizer'), \
             patch('tplr.comms.Comms'), \
             patch('tplr.sharded_dataset.ShardedDatasetManager'), \
             patch('neurons.BaseNode.__init__'), \
             patch('torch.manual_seed'), \
             patch('torch.cuda.manual_seed_all'), \
             patch('numpy.random.seed'), \
             patch('random.seed'), \
             patch('torch.cuda.empty_cache'), \
             patch('torch.clear_autocast_cache'):
            
            # Setup mocks
            mock_wallet_obj = Mock()
            mock_wallet_obj.hotkey.ss58_address = 'test_hotkey'
            mock_wallet.return_value = mock_wallet_obj
            
            mock_subtensor_obj = Mock()
            mock_metagraph = Mock()
            mock_metagraph.hotkeys = ['test_hotkey']
            mock_metagraph.netuid = 268
            mock_subtensor_obj.metagraph.return_value = mock_metagraph
            mock_subtensor_obj.block = 1000
            mock_subtensor.return_value = mock_subtensor_obj
            
            # Initialize miner
            miner = Miner()
            
            # Test cleanup integration
            await miner.cleanup_window()
            
            # Verify both initialization and cleanup worked
            self.assertEqual(miner.uid, 0)
            self.assertTrue(miner.is_master)


if __name__ == '__main__':
    # Run tests with increased verbosity
    unittest.main(verbosity=2)