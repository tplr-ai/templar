
# Additional comprehensive unit tests for Validator class

import asyncio
import copy
import hashlib
import os
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from io import StringIO
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

import numpy as np
import pytest
import torch
from openskill.models import PlackettLuce
from transformers.models.llama import LlamaForCausalLM

import tplr
from neurons import BaseNode


class TestValidatorConfiguration:
    """Test validator configuration and initialization."""
    
    def test_validator_config_default_values(self):
        """Test that validator_config returns expected default values."""
        with patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(
            netuid=268, project="templar", device="cuda", debug=False, trace=False,
            store_gathers=False, test=False, local=False
        )):
            with patch('bt.config') as mock_bt_config:
                mock_bt_config.return_value = argparse.Namespace(
                    netuid=268, project="templar", device="cuda", debug=False, trace=False
                )
                config = Validator.validator_config()
                assert config.netuid == 268
                assert config.project == "templar" 
                assert config.device == "cuda"
                assert not config.debug
                assert not config.trace

    def test_validator_config_debug_enables_tracing(self):
        """Test that debug flag enables debug logging."""
        with patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(
            debug=True, trace=False, netuid=268, project="templar", device="cuda",
            store_gathers=False, test=False, local=False
        )):
            with patch('bt.config') as mock_bt_config:
                mock_bt_config.return_value = argparse.Namespace(debug=True, trace=False)
                with patch('tplr.debug') as mock_debug:
                    Validator.validator_config()
                    mock_debug.assert_called_once()

    def test_validator_config_trace_enables_tracing(self):
        """Test that trace flag enables trace logging.""" 
        with patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(
            debug=False, trace=True, netuid=268, project="templar", device="cuda",
            store_gathers=False, test=False, local=False
        )):
            with patch('bt.config') as mock_bt_config:
                mock_bt_config.return_value = argparse.Namespace(debug=False, trace=True)
                with patch('tplr.trace') as mock_trace:
                    Validator.validator_config()
                    mock_trace.assert_called_once()


class TestValidatorInitialization:
    """Test validator initialization and setup."""

    @patch('tplr.load_hparams')
    @patch('bt.wallet')
    @patch('bt.subtensor')
    @patch('tplr.setup_loki_logger')
    @patch.object(Validator, 'validator_config')
    def test_validator_init_success(self, mock_config, mock_loki, mock_subtensor, mock_wallet, mock_hparams):
        """Test successful validator initialization."""
        # Setup mocks
        mock_config.return_value = MagicMock(local=False, netuid=268, device="cuda")
        mock_hparams.return_value = MagicMock()
        mock_wallet_instance = MagicMock()
        mock_wallet_instance.hotkey.ss58_address = "test_address"
        mock_wallet.return_value = mock_wallet_instance
        
        mock_subtensor_instance = MagicMock()
        mock_metagraph = MagicMock()
        mock_metagraph.hotkeys = ["test_address", "other_address"]
        mock_metagraph.netuid = 268
        mock_subtensor_instance.metagraph.return_value = mock_metagraph
        mock_subtensor.return_value = mock_subtensor_instance

        with patch.object(BaseNode, '__init__'), \
             patch('LlamaForCausalLM'), \
             patch('tplr.compress.TransformDCT'), \
             patch('tplr.compress.CompressDCT'), \
             patch('torch.optim.SGD'), \
             patch('torch.optim.lr_scheduler.SequentialLR'), \
             patch('tplr.comms.Comms'), \
             patch('tplr.initialize_wandb'), \
             patch('tplr.metrics.MetricsLogger'), \
             patch('tplr.sharded_dataset.ShardedDatasetManager'), \
             patch('os.path.isfile', return_value=False):
            
            validator = Validator()
            assert validator.uid == 0  # Index of test_address in hotkeys

    @patch('tplr.load_hparams')
    @patch('bt.wallet') 
    @patch('bt.subtensor')
    @patch.object(Validator, 'validator_config')
    def test_validator_init_wallet_not_registered_exits(self, mock_config, mock_subtensor, mock_wallet, mock_hparams):
        """Test that validator exits if wallet is not registered."""
        mock_config.return_value = MagicMock(local=False, netuid=268, device="cuda")
        mock_hparams.return_value = MagicMock()
        
        mock_wallet_instance = MagicMock()
        mock_wallet_instance.hotkey.ss58_address = "unregistered_address"
        mock_wallet.return_value = mock_wallet_instance
        
        mock_subtensor_instance = MagicMock()
        mock_metagraph = MagicMock()
        mock_metagraph.hotkeys = ["other_address_1", "other_address_2"]
        mock_metagraph.netuid = 268
        mock_subtensor_instance.metagraph.return_value = mock_metagraph
        mock_subtensor.return_value = mock_subtensor_instance

        with patch.object(BaseNode, '__init__'), \
             patch('sys.exit') as mock_exit:
            
            Validator()
            mock_exit.assert_called_once()


class TestTimerContextManager:
    """Test the timer context manager utility."""

    def test_timer_basic_functionality(self):
        """Test timer measures duration correctly."""
        with patch('time.perf_counter', side_effect=[0.0, 1.5]):
            with patch('tplr.logger.debug') as mock_debug:
                with timer("test_operation"):
                    pass
                mock_debug.assert_called_once_with("test_operation took 1.50s")

    def test_timer_with_wandb_logging(self):
        """Test timer logs to wandb when provided."""
        mock_wandb = MagicMock()
        with patch('time.perf_counter', side_effect=[0.0, 2.5]):
            with patch('tplr.logger.debug'):
                with timer("test_op", wandb_obj=mock_wandb, step=10):
                    pass
                mock_wandb.log.assert_called_once_with({"validator/test_op": 2.5}, step=10)

    def test_timer_with_metrics_logger(self):
        """Test timer logs to metrics logger when provided."""
        mock_metrics = MagicMock()
        with patch('time.perf_counter', side_effect=[0.0, 3.0]):
            with patch('tplr.logger.debug'):
                with timer("test_metric", metrics_logger=mock_metrics, step=5):
                    pass
                mock_metrics.log.assert_called_once_with(
                    measurement="timing",
                    tags={"window": 5},
                    fields={"test_metric": 3.0}
                )


class TestValidatorScoring:
    """Test validator scoring mechanisms."""

    def setup_method(self):
        """Setup test validator with mocked dependencies."""
        with patch.object(Validator, '__init__', return_value=None):
            self.validator = Validator()
            self.validator.config = MagicMock(device="cuda")
            self.validator.hparams = MagicMock(
                reset_inactivity_windows=10,
                binary_score_ma_alpha=0.1,
                topk_compression=0.1,
                use_dct=True,
                eval_lr_factor=1.0
            )
            self.validator.current_window = 100
            self.validator.sync_window = 100
            self.validator.final_scores = torch.zeros(256)
            self.validator.weights = torch.zeros(256) 
            self.validator.gradient_scores = torch.zeros(256)
            self.validator.binary_moving_averages = torch.zeros(256)
            self.validator.binary_indicator_scores = torch.zeros(256)
            self.validator.sync_scores = torch.zeros(256)
            self.validator.openskill_ratings = {}
            self.validator.eval_peers = {}
            self.validator.inactive_scores = {}
            self.validator.lr = 0.001

    def test_reset_peer_after_inactivity(self):
        """Test peer reset after extended inactivity."""
        uid = 5
        inactive_since = 85  # 15 windows ago
        
        # Set some initial scores
        self.validator.final_scores[uid] = 0.5
        self.validator.weights[uid] = 0.3
        self.validator.openskill_ratings[uid] = MagicMock()
        self.validator.eval_peers[uid] = 2
        self.validator.inactive_scores[uid] = (inactive_since, 0.5)
        
        result = self.validator.reset_peer(inactive_since, uid)
        
        assert result is True
        assert self.validator.final_scores[uid] == 0.0
        assert self.validator.weights[uid] == 0.0
        assert uid not in self.validator.openskill_ratings
        assert uid not in self.validator.eval_peers
        assert uid not in self.validator.inactive_scores

    def test_reset_peer_not_inactive_long_enough(self):
        """Test peer not reset if not inactive long enough."""
        uid = 5
        inactive_since = 95  # Only 5 windows ago
        
        initial_score = 0.5
        self.validator.final_scores[uid] = initial_score
        self.validator.inactive_scores[uid] = (inactive_since, initial_score)
        
        result = self.validator.reset_peer(inactive_since, uid)
        
        assert result is False
        assert self.validator.final_scores[uid] == initial_score

    def test_log_sync_score_with_all_metrics(self):
        """Test logging sync scores with all metrics."""
        self.validator.wandb = MagicMock()
        self.validator.metrics_logger = MagicMock()
        self.validator.global_step = 50
        
        eval_uid = 10
        sync_result = {
            "l2_norm": 0.5,
            "avg_l2_norm": 0.3,
            "avg_abs_diff": 0.2,
            "max_diff": 1.0,
            "avg_steps_behind": 2.5,
            "max_steps_behind": 5.0
        }
        
        with patch('tplr.log_with_context'):
            self.validator.log_sync_score(eval_uid, sync_result)
        
        # Verify wandb logging
        expected_wandb_data = {
            f"validator/sync/l2_norm/{eval_uid}": 0.5,
            f"validator/sync/avg_l2_norm/{eval_uid}": 0.3,
            f"validator/sync/avg_abs_diff/{eval_uid}": 0.2,
            f"validator/sync/sync_max_diff/{eval_uid}": 1.0,
            f"validator/sync/avg_steps_behind/{eval_uid}": 2.5,
            f"validator/sync/max_steps_behind/{eval_uid}": 5.0,
        }
        self.validator.wandb.log.assert_called_once_with(expected_wandb_data, step=50)
        
        # Verify metrics logger
        self.validator.metrics_logger.log.assert_called_once()

    def test_log_sync_score_with_missing_metrics_uses_defaults(self):
        """Test sync score logging uses default values for missing metrics.""" 
        self.validator.wandb = MagicMock()
        self.validator.metrics_logger = MagicMock()
        self.validator.global_step = 25
        
        eval_uid = 15
        sync_result = {}  # Empty result should use defaults
        
        with patch('tplr.log_with_context'):
            self.validator.log_sync_score(eval_uid, sync_result)
        
        # Verify default values were used (99.0 for most metrics)
        expected_wandb_data = {
            f"validator/sync/l2_norm/{eval_uid}": 99.0,
            f"validator/sync/avg_l2_norm/{eval_uid}": 99.0,
            f"validator/sync/avg_abs_diff/{eval_uid}": 99.0,
            f"validator/sync/sync_max_diff/{eval_uid}": 99.0,
            f"validator/sync/avg_steps_behind/{eval_uid}": 99.0,
            f"validator/sync/max_steps_behind/{eval_uid}": 99.0,
        }
        self.validator.wandb.log.assert_called_once_with(expected_wandb_data, step=25)


class TestOpenSkillRatings:
    """Test OpenSkill rating system integration."""

    def setup_method(self):
        """Setup test validator for OpenSkill tests."""
        with patch.object(Validator, '__init__', return_value=None):
            self.validator = Validator()
            self.validator.openskill_model = PlackettLuce()
            self.validator.openskill_ratings = {}
            self.validator.final_scores = torch.zeros(256)
            self.validator.binary_moving_averages = torch.zeros(256)
            self.validator.sync_scores = torch.zeros(256)
            self.validator.evaluated_uids = set()
            self.validator.wandb = MagicMock()
            self.validator.metrics_logger = MagicMock()
            self.validator.global_step = 100
            self.validator.sync_window = 50

    def test_update_openskill_ratings_with_single_peer(self):
        """Test OpenSkill rating update with single peer."""
        uid = 10
        self.validator.current_window_scores = {uid: 0.8}
        self.validator.openskill_ratings[uid] = self.validator.openskill_model.rating(name=str(uid))
        self.validator.evaluated_uids.add(uid)
        
        with patch('tplr.log_with_context'), \
             patch('os.get_terminal_size', return_value=MagicMock(columns=100)), \
             patch('os.environ.__setitem__'):
            
            self.validator.update_openskill_ratings()
        
        # Verify rating was updated
        assert uid in self.validator.openskill_ratings
        assert self.validator.final_scores[uid] > 0  # Should be positive with good score
        
        # Verify logging calls
        self.validator.wandb.log.assert_called()
        self.validator.metrics_logger.log.assert_called()

    def test_update_openskill_ratings_with_multiple_peers(self):
        """Test OpenSkill rating update with multiple competing peers."""
        uids = [10, 20, 30]
        scores = [0.9, 0.7, 0.5]  # Different performance levels
        
        self.validator.current_window_scores = dict(zip(uids, scores))
        for uid in uids:
            self.validator.openskill_ratings[uid] = self.validator.openskill_model.rating(name=str(uid))
            self.validator.evaluated_uids.add(uid)
        
        with patch('tplr.log_with_context'), \
             patch('os.get_terminal_size', return_value=MagicMock(columns=100)), \
             patch('os.environ.__setitem__'):
            
            self.validator.update_openskill_ratings()
        
        # Verify all ratings were updated
        for uid in uids:
            assert uid in self.validator.openskill_ratings
        
        # Verify final scores reflect performance differences
        assert self.validator.final_scores[10] >= self.validator.final_scores[20]
        assert self.validator.final_scores[20] >= self.validator.final_scores[30]

    def test_update_openskill_ratings_no_current_scores(self):
        """Test OpenSkill update does nothing when no current scores exist."""
        self.validator.current_window_scores = {}
        
        original_ratings = self.validator.openskill_ratings.copy()
        
        self.validator.update_openskill_ratings()
        
        # Verify no changes were made
        assert self.validator.openskill_ratings == original_ratings

    def test_update_openskill_ratings_handles_table_creation_error(self):
        """Test OpenSkill update gracefully handles table creation errors."""
        uid = 15
        self.validator.current_window_scores = {uid: 0.6}
        self.validator.openskill_ratings[uid] = self.validator.openskill_model.rating(name=str(uid))
        self.validator.evaluated_uids.add(uid)
        
        with patch('tplr.log_with_context') as mock_log, \
             patch('os.get_terminal_size', side_effect=Exception("Terminal error")):
            
            # Should not raise exception
            self.validator.update_openskill_ratings()
        
        # Should log warning about table creation failure
        warning_calls = [call for call in mock_log.call_args_list 
                        if len(call[1]) > 0 and call[1].get('level') == 'warning']
        assert len(warning_calls) > 0


class TestWeightUpdate:
    """Test weight update and emission distribution."""

    def setup_method(self):
        """Setup validator for weight testing."""
        with patch.object(Validator, '__init__', return_value=None):
            self.validator = Validator()
            self.validator.weights = torch.zeros(256)
            self.validator.final_scores = torch.zeros(256)
            self.validator.evaluated_uids = set()
            self.validator.burn_uid = 1
            self.validator.hparams = MagicMock(
                burn_rate=0.1,
                gather_share=0.75,
                gather_peer_count=15,
                reserve_peer_count=10,
                gather_top_ratio=2.0,
                reserve_decay_ratio=0.75
            )

    def test_update_weights_with_positive_scores(self):
        """Test weight update with peers having positive scores."""
        # Setup peers with positive scores
        uids = [10, 20, 30, 40, 50]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        
        for uid, score in zip(uids, scores):
            self.validator.final_scores[uid] = score
            self.validator.evaluated_uids.add(uid)
        
        self.validator.update_weights()
        
        # Verify burn weight is set
        assert self.validator.weights[self.validator.burn_uid] == 0.1
        
        # Verify gather peers get higher weights
        gather_weights = []
        for uid in uids[:3]:  # First 3 should be gather peers
            gather_weights.append(self.validator.weights[uid].item())
        
        # Weights should be positive and generally decreasing
        assert all(w > 0 for w in gather_weights)
        assert gather_weights[0] >= gather_weights[1] >= gather_weights[2]
        
        # Total weights should sum to approximately 1
        total_weight = self.validator.weights.sum().item()
        assert abs(total_weight - 1.0) < 1e-6

    def test_update_weights_no_positive_scores(self):
        """Test weight update when no peers have positive scores."""
        # Setup peers with zero/negative scores
        uids = [10, 20, 30]
        for uid in uids:
            self.validator.final_scores[uid] = 0.0
            self.validator.evaluated_uids.add(uid)
        
        self.validator.update_weights()
        
        # Only burn weight should be set
        assert self.validator.weights[self.validator.burn_uid] == 1.0
        for uid in uids:
            assert self.validator.weights[uid] == 0.0

    def test_update_weights_burn_rate_zero(self):
        """Test weight update with zero burn rate."""
        self.validator.hparams.burn_rate = 0.0
        
        uids = [10, 20, 30]
        for uid in uids:
            self.validator.final_scores[uid] = 0.5
            self.validator.evaluated_uids.add(uid)
        
        self.validator.update_weights()
        
        # No burn weight
        assert self.validator.weights[self.validator.burn_uid] == 0.0
        
        # All weight should go to peers
        peer_weight_sum = sum(self.validator.weights[uid].item() for uid in uids)
        assert abs(peer_weight_sum - 1.0) < 1e-6

    def test_update_weights_reserve_capping(self):
        """Test that reserve weights are capped below minimum gather weight."""
        # Setup many peers to trigger reserve capping
        uids = list(range(10, 40))  # 30 peers
        for uid in uids:
            self.validator.final_scores[uid] = 0.5
            self.validator.evaluated_uids.add(uid)
        
        self.validator.update_weights()
        
        gather_uids = sorted(uids, key=lambda u: self.validator.final_scores[u].item(), 
                           reverse=True)[:self.validator.hparams.gather_peer_count]
        reserve_uids = sorted(uids, key=lambda u: self.validator.final_scores[u].item(),
                           reverse=True)[self.validator.hparams.gather_peer_count:
                           self.validator.hparams.gather_peer_count + self.validator.hparams.reserve_peer_count]
        
        if gather_uids and reserve_uids:
            min_gather_weight = min(self.validator.weights[uid].item() for uid in gather_uids)
            max_reserve_weight = max(self.validator.weights[uid].item() for uid in reserve_uids)
            
            # Reserve weights should be less than gather weights
            assert max_reserve_weight <= min_gather_weight


class TestModelEvaluation:
    """Test model evaluation functionality."""

    def setup_method(self):
        """Setup validator for model evaluation tests."""
        with patch.object(Validator, '__init__', return_value=None):
            self.validator = Validator()
            self.validator.config = MagicMock(device="cuda")
            self.validator.sync_window = 100
            self.validator.current_window = 100
            self.validator.tokenizer = MagicMock()
            self.validator.tokenizer.pad_token_id = 0

    @pytest.mark.asyncio
    async def test_evaluate_model_basic(self):
        """Test basic model evaluation functionality."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([1.0], device="cuda")]
        mock_model.eval.return_value = None
        
        # Mock model output
        mock_output = MagicMock()
        mock_output.loss.item.return_value = 2.5
        mock_model.return_value = mock_output
        
        # Create test data loader
        test_batches = [
            torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        ]
        
        with patch('torch.no_grad'), \
             patch('torch.autocast'), \
             patch('torch.cuda.empty_cache'):
            
            total_loss, n_batches = await self.validator.evaluate_model(mock_model, test_batches)
        
        assert total_loss == 5.0  # 2.5 * 2 batches
        assert n_batches == 2

    @pytest.mark.asyncio 
    async def test_evaluate_model_empty_batch(self):
        """Test model evaluation with empty batches."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([1.0], device="cuda")]
        
        # Mix of empty and valid batches
        test_batches = [
            None,  # Empty batch
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.tensor([]),  # Empty tensor
            torch.tensor([[4, 5, 6]], dtype=torch.long)
        ]
        
        mock_output = MagicMock()
        mock_output.loss.item.return_value = 1.5
        mock_model.return_value = mock_output
        
        with patch('torch.no_grad'), \
             patch('torch.autocast'), \
             patch('torch.cuda.empty_cache'), \
             patch('tplr.log_with_context'):
            
            total_loss, n_batches = await self.validator.evaluate_model(mock_model, test_batches)
        
        assert n_batches == 2  # Only 2 valid batches processed
        assert total_loss == 3.0  # 1.5 * 2

    @pytest.mark.asyncio
    async def test_evaluate_model_no_valid_batches(self):
        """Test model evaluation when no valid batches exist."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([1.0], device="cuda")]
        
        test_batches = [None, torch.tensor([]), None]
        
        with patch('torch.no_grad'), \
             patch('torch.autocast'), \
             patch('tplr.log_with_context'):
            
            total_loss, n_batches = await self.validator.evaluate_model(mock_model, test_batches)
        
        assert total_loss == 0.0
        assert n_batches == 0


class TestPeerSelection:
    """Test peer selection algorithms."""

    def setup_method(self):
        """Setup validator for peer selection tests."""
        with patch.object(Validator, '__init__', return_value=None):
            self.validator = Validator()
            self.validator.metagraph = MagicMock()
            self.validator.comms = MagicMock()
            self.validator.weights = torch.zeros(256)
            self.validator.sync_window = 100
            self.validator.current_window = 100
            self.validator.hparams = MagicMock(
                gather_peer_count=5,
                reserve_peer_count=3,
                minimum_peers=3
            )

    def test_select_initial_peers_by_incentive(self):
        """Test initial peer selection prioritizes by incentive."""
        # Setup metagraph data
        self.validator.metagraph.uids.tolist.return_value = [10, 20, 30, 40, 50]
        self.validator.metagraph.I.tolist.return_value = [0.8, 0.9, 0.1, 0.7, 0.6]
        
        # All peers are active
        self.validator.comms.active_peers = {10, 20, 30, 40, 50}
        
        with patch('tplr.log_with_context'):
            result = self.validator.select_initial_peers()
        
        assert result is not None
        gather_peers, reserve_peers = result
        
        # Should select highest incentive peers first
        expected_order = [20, 10, 40, 50, 30]  # Sorted by incentive desc
        all_selected = gather_peers + reserve_peers
        
        assert len(gather_peers) == 5
        assert len(reserve_peers) == 3
        assert all_selected == expected_order

    def test_select_initial_peers_insufficient_peers(self):
        """Test initial peer selection with insufficient active peers."""
        self.validator.metagraph.uids.tolist.return_value = [10, 20]
        self.validator.metagraph.I.tolist.return_value = [0.5, 0.6]
        self.validator.comms.active_peers = {10, 20}  # Only 2 active peers
        
        with patch('tplr.log_with_context'):
            result = self.validator.select_initial_peers()
        
        assert result is None  # Should return None when insufficient peers

    def test_select_initial_peers_zero_incentive_peers(self):
        """Test initial peer selection includes zero-incentive active peers."""
        self.validator.metagraph.uids.tolist.return_value = [10, 20, 30, 40, 50, 60]
        self.validator.metagraph.I.tolist.return_value = [0.8, 0.0, 0.0, 0.7, 0.0, 0.9]
        self.validator.comms.active_peers = {10, 20, 30, 40, 50, 60}
        
        with patch('tplr.log_with_context'), \
             patch('random.sample', return_value=[20, 30]):  # Mock random selection
            
            result = self.validator.select_initial_peers()
        
        assert result is not None
        gather_peers, reserve_peers = result
        
        # Should include both incentive and zero-incentive peers
        all_selected = gather_peers + reserve_peers
        assert 60 in all_selected  # Highest incentive
        assert 10 in all_selected  # Second highest incentive
        assert 40 in all_selected  # Third highest incentive

    def test_select_next_peers_by_weight(self):
        """Test next peer selection prioritizes by weight."""
        # Setup weights for peers
        peer_weights = {10: 0.3, 20: 0.5, 30: 0.1, 40: 0.4, 50: 0.2}
        for uid, weight in peer_weights.items():
            self.validator.weights[uid] = weight
        
        self.validator.comms.active_peers = set(peer_weights.keys())
        
        with patch('tplr.log_with_context'):
            result = self.validator.select_next_peers()
        
        assert result is not None
        gather_peers, reserve_peers = result
        
        # Should select by highest weight first
        expected_order = [20, 40, 10, 50, 30]  # Sorted by weight desc
        all_selected = gather_peers + reserve_peers
        
        assert all_selected == expected_order

    def test_select_next_peers_insufficient_active(self):
        """Test next peer selection with insufficient active peers."""
        self.validator.comms.active_peers = {10, 20}  # Only 2 peers
        
        with patch('tplr.log_with_context'):
            result = self.validator.select_next_peers()
        
        assert result is None


class TestBinning:
    """Test peer binning for evaluation."""

    def setup_method(self):
        """Setup validator for binning tests."""
        with patch.object(Validator, '__init__', return_value=None):
            self.validator = Validator()
            self.validator.eval_peers = {}
            self.validator.openskill_ratings = {}
            self.validator.sync_window = 100
            self.validator.current_window = 100
            self.validator.hparams = MagicMock(uids_per_window=3)
            self.validator.comms = MagicMock()

    def test_bin_evaluation_peers_sufficient_peers(self):
        """Test binning with sufficient peers for multiple bins."""
        # Setup peers with different OpenSkill ratings
        uids = list(range(10, 25))  # 15 peers
        for i, uid in enumerate(uids):
            self.validator.eval_peers[uid] = 1
            # Create mock rating with different ordinal values
            mock_rating = MagicMock()
            mock_rating.ordinal.return_value = 1.0 - (i * 0.1)  # Decreasing ratings
            self.validator.openskill_ratings[uid] = mock_rating
        
        with patch('tplr.log_with_context'):
            bins = self.validator.bin_evaluation_peers(num_bins=5)
        
        assert len(bins) == 5
        total_peers_in_bins = sum(len(peer_list) for peer_list in bins.values())
        assert total_peers_in_bins == 15
        
        # Verify peers are distributed across bins
        for bin_idx, peer_list in bins.items():
            assert len(peer_list) >= 2  # Should have at least 2 peers per bin

    def test_bin_evaluation_peers_insufficient_peers(self):
        """Test binning with insufficient peers returns single bin."""
        # Setup only a few peers
        uids = [10, 20, 30]
        for uid in uids:
            self.validator.eval_peers[uid] = 1
            mock_rating = MagicMock()
            mock_rating.ordinal.return_value = 1.0
            self.validator.openskill_ratings[uid] = mock_rating
        
        with patch('tplr.log_with_context'):
            bins = self.validator.bin_evaluation_peers(num_bins=5)
        
        assert len(bins) == 1
        assert 0 in bins
        assert set(bins[0]) == set(uids)

    def test_bin_evaluation_peers_no_openskill_ratings(self):
        """Test binning with peers having no OpenSkill ratings."""
        uids = list(range(10, 20))
        for uid in uids:
            self.validator.eval_peers[uid] = 1
            # No OpenSkill ratings - should use default 0.0
        
        with patch('tplr.log_with_context'):
            bins = self.validator.bin_evaluation_peers(num_bins=3)
        
        # Should still create bins even with zero ratings
        total_peers_in_bins = sum(len(peer_list) for peer_list in bins.values())
        assert total_peers_in_bins == len(uids)

    def test_select_next_bin_for_evaluation(self):
        """Test random bin selection."""
        with patch('random.randint', return_value=2), \
             patch('tplr.log_with_context'):
            
            selected_bin = self.validator.select_next_bin_for_evaluation(num_bins=5)
        
        assert selected_bin == 2

    def test_select_evaluation_uids_from_bin(self):
        """Test UID selection from specific bin."""
        bins = {
            0: [10, 20, 30],
            1: [40, 50, 60, 70],
            2: [80, 90]
        }
        
        # Setup eval_peers weights
        for uid_list in bins.values():
            for uid in uid_list:
                self.validator.eval_peers[uid] = 1
        
        self.validator.comms.weighted_random_sample_no_replacement = MagicMock(
            return_value=[40, 50, 60]
        )
        
        with patch('tplr.log_with_context'):
            selected_uids = self.validator.select_evaluation_uids_from_bin(bins, 1)
        
        assert selected_uids == [40, 50, 60]
        self.validator.comms.weighted_random_sample_no_replacement.assert_called_once()

    def test_select_evaluation_uids_from_nonexistent_bin(self):
        """Test UID selection from non-existent bin falls back to bin 0."""
        bins = {0: [10, 20, 30]}
        
        for uid in bins[0]:
            self.validator.eval_peers[uid] = 1
        
        self.validator.comms.weighted_random_sample_no_replacement = MagicMock(
            return_value=[10, 20]
        )
        
        with patch('tplr.log_with_context'):
            selected_uids = self.validator.select_evaluation_uids_from_bin(bins, 5)  # Non-existent bin
        
        # Should fall back to bin 0
        assert selected_uids == [10, 20]


class TestGradientProcessing:
    """Test gradient processing and validation."""

    def setup_method(self):
        """Setup validator for gradient tests."""
        with patch.object(Validator, '__init__', return_value=None):
            self.validator = Validator()
            self.validator.config = MagicMock(device="cuda")
            self.validator.hparams = MagicMock(
                topk_compression=0.1,
                use_dct=True,
                eval_lr_factor=1.0
            )
            self.validator.sync_window = 100
            self.validator.current_window = 100
            self.validator.lr = 0.001
            self.validator.totalks = {"layer.weight": 1000}
            self.validator.xshapes = {"layer.weight": (10, 10)}
            self.validator.transformer = MagicMock()
            self.validator.compressor = MagicMock()
            self.validator.comms = MagicMock()

    def test_update_model_with_gradient_valid_data(self):
        """Test model update with valid gradient data."""
        # Create mock model
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.to.return_value = mock_param
        mock_model.zero_grad.return_value = None
        mock_model.named_parameters.return_value = [("layer.weight", mock_param)]
        
        # Create valid state dict
        eval_state_dict = {
            "layer.weightidxs": torch.tensor([1, 2, 3]),
            "layer.weightvals": torch.tensor([0.1, 0.2, 0.3]),
            "layer.weightquant_params": {"scale": 1.0}
        }
        
        # Mock decompression
        mock_grad = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        self.validator.compressor.decompress.return_value = mock_grad
        self.validator.transformer.decode.return_value = mock_grad
        
        # Should not raise exception
        self.validator.update_model_with_gradient(mock_model, 10, eval_state_dict)
        
        # Verify model was updated
        mock_model.zero_grad.assert_called_once()
        self.validator.transformer.decode.assert_called_once()

    def test_update_model_with_gradient_invalid_indices(self):
        """Test model update with invalid gradient indices."""
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_model.named_parameters.return_value = [("layer.weight", mock_param)]
        
        # Invalid indices (beyond totalk bounds)
        eval_state_dict = {
            "layer.weightidxs": torch.tensor([999999]),  # Way beyond bounds
            "layer.weightvals": torch.tensor([0.1]),
            "layer.weightquant_params": {"scale": 1.0}
        }
        
        self.validator.comms.check_compressed_indices.side_effect = ValueError("Invalid indices")
        
        with pytest.raises(ValueError, match="Invalid gradient data"):
            self.validator.update_model_with_gradient(mock_model, 10, eval_state_dict)

    def test_update_model_with_gradient_nan_values(self):
        """Test model update with NaN gradient values."""
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_model.named_parameters.return_value = [("layer.weight", mock_param)]
        
        # NaN values in gradient
        eval_state_dict = {
            "layer.weightidxs": torch.tensor([1, 2, 3]),
            "layer.weightvals": torch.tensor([0.1, float('nan'), 0.3]),
            "layer.weightquant_params": {"scale": 1.0}
        }
        
        with patch('tplr.log_with_context'):
            with pytest.raises(ValueError, match="NaN or Inf values"):
                self.validator.update_model_with_gradient(mock_model, 10, eval_state_dict)

    def test_update_model_with_gradient_missing_totalk(self):
        """Test model update with missing totalk information."""
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_model.named_parameters.return_value = [("unknown.weight", mock_param)]
        
        eval_state_dict = {
            "unknown.weightidxs": torch.tensor([1, 2, 3]),
            "unknown.weightvals": torch.tensor([0.1, 0.2, 0.3]),
            "unknown.weightquant_params": {"scale": 1.0}
        }
        
        with patch('tplr.log_with_context'):
            with pytest.raises(ValueError, match="Missing totalk"):
                self.validator.update_model_with_gradient(mock_model, 10, eval_state_dict)

    def test_update_model_with_gradient_nan_in_decompressed(self):
        """Test model update with NaN in decompressed gradient."""
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.to.return_value = mock_param
        mock_model.named_parameters.return_value = [("layer.weight", mock_param)]
        
        eval_state_dict = {
            "layer.weightidxs": torch.tensor([1, 2, 3]),
            "layer.weightvals": torch.tensor([0.1, 0.2, 0.3]),
            "layer.weightquant_params": {"scale": 1.0}
        }
        
        # Mock decompressed gradient with NaN
        nan_grad = torch.tensor([[0.1, float('nan')], [0.3, 0.4]])
        self.validator.compressor.decompress.return_value = nan_grad
        self.validator.transformer.decode.return_value = nan_grad
        
        with patch('tplr.log_with_context'):
            with pytest.raises(ValueError, match="NaN or Inf in decompressed gradient"):
                self.validator.update_model_with_gradient(mock_model, 10, eval_state_dict)


class TestStateManagement:
    """Test validator state save/load functionality."""

    def setup_method(self):
        """Setup validator for state tests."""
        with patch.object(Validator, '__init__', return_value=None):
            self.validator = Validator()
            self.validator.global_step = 100
            self.validator.gradient_scores = torch.tensor([0.1, 0.2, 0.3])
            self.validator.sync_scores = torch.tensor([0.8, 0.9, 0.7])
            self.validator.binary_indicator_scores = torch.tensor([1.0, -1.0, 1.0])
            self.validator.final_scores = torch.tensor([0.5, 0.6, 0.4])
            self.validator.binary_moving_averages = torch.tensor([0.2, 0.3, 0.1])
            self.validator.weights = torch.tensor([0.3, 0.4, 0.3])
            
            # Mock OpenSkill ratings
            mock_model = MagicMock()
            mock_rating1 = MagicMock()
            mock_rating1.mu = 25.0
            mock_rating1.sigma = 8.0
            mock_rating1.ordinal.return_value = 1.0
            mock_rating2 = MagicMock()
            mock_rating2.mu = 20.0
            mock_rating2.sigma = 7.0
            mock_rating2.ordinal.return_value = 0.5
            
            self.validator.openskill_ratings = {10: mock_rating1, 20: mock_rating2}
            self.validator.state_path = "test_state.pt"

    def test_state_dict_creation(self):
        """Test state dictionary creation with all components."""
        state_dict = self.validator._state_dict()
        
        # Verify all required keys are present
        expected_keys = {
            "global_step", "gradient_scores", "sync_scores", 
            "binary_indicator_scores", "final_scores", 
            "binary_moving_averages", "weights", "openskill_ratings"
        }
        assert set(state_dict.keys()) == expected_keys
        
        # Verify global step
        assert state_dict["global_step"] == 100
        
        # Verify tensor data types (should be CPU tensors)
        for key in ["gradient_scores", "sync_scores", "binary_indicator_scores", 
                   "final_scores", "binary_moving_averages", "weights"]:
            assert isinstance(state_dict[key], torch.Tensor)
            assert state_dict[key].device == torch.device("cpu")
        
        # Verify OpenSkill ratings structure
        openskill_data = state_dict["openskill_ratings"]
        assert 10 in openskill_data
        assert 20 in openskill_data
        assert openskill_data[10]["mu"] == 25.0
        assert openskill_data[10]["sigma"] == 8.0
        assert openskill_data[10]["ordinal"] == 1.0

    @pytest.mark.asyncio
    async def test_save_state_success(self):
        """Test successful state saving."""
        with patch('asyncio.to_thread') as mock_to_thread, \
             patch('tplr.log_with_context'):
            
            mock_to_thread.return_value = None
            
            await self.validator.save_state()
            
            mock_to_thread.assert_called_once()
            # Verify torch.save was called with correct arguments
            args, kwargs = mock_to_thread.call_args
            assert args[0] == torch.save
            assert args[2] == "test_state.pt"

    @pytest.mark.asyncio
    async def test_save_state_failure(self):
        """Test state saving with exception handling."""
        with patch('asyncio.to_thread', side_effect=Exception("Save failed")), \
             patch('tplr.log_with_context') as mock_log:
            
            await self.validator.save_state()
            
            # Should log warning about save failure
            warning_calls = [c for c in mock_log.call_args_list 
                           if len(c[1]) > 0 and c[1].get('level') == 'warning']
            assert len(warning_calls) > 0

    def test_load_state_success(self):
        """Test successful state loading."""
        # Create mock saved state
        mock_state = {
            "global_step": 150,
            "gradient_scores": torch.tensor([0.4, 0.5, 0.6]),
            "sync_scores": torch.tensor([0.7, 0.8, 0.9]),
            "binary_indicator_scores": torch.tensor([-1.0, 1.0, -1.0]),
            "final_scores": torch.tensor([0.2, 0.3, 0.4]),
            "binary_moving_averages": torch.tensor([0.1, 0.2, 0.3]),
            "weights": torch.tensor([0.2, 0.3, 0.5]),
            "openskill_ratings": {
                30: {"mu": 30.0, "sigma": 5.0, "ordinal": 2.0},
                40: {"mu": 25.0, "sigma": 6.0, "ordinal": 1.5}
            }
        }
        
        # Setup validator for loading
        self.validator.config = MagicMock(device="cuda")
        self.validator.openskill_model = MagicMock()
        mock_rating = MagicMock()
        self.validator.openskill_model.rating.return_value = mock_rating
        
        with patch('torch.load', return_value=mock_state), \
             patch('tplr.logger.info'):
            
            self.validator.load_state()
        
        # Verify state was loaded correctly
        assert self.validator.global_step == 150
        assert torch.equal(self.validator.gradient_scores, mock_state["gradient_scores"].float())
        
        # Verify OpenSkill ratings were restored
        assert 30 in self.validator.openskill_ratings
        assert 40 in self.validator.openskill_ratings

    def test_load_state_file_not_found(self):
        """Test state loading when file doesn't exist."""
        with patch('torch.load', side_effect=FileNotFoundError()), \
             patch('tplr.logger.warning') as mock_warning:
            
            self.validator.load_state()
            
            mock_warning.assert_called_once()

    def test_load_state_corrupted_file(self):
        """Test state loading with corrupted state file."""
        with patch('torch.load', side_effect=Exception("Corrupted file")), \
             patch('tplr.logger.warning') as mock_warning:
            
            self.validator.load_state()
            
            mock_warning.assert_called_once()

    def test_load_state_partial_data(self):
        """Test state loading with missing keys (partial data)."""
        # State with some missing keys
        partial_state = {
            "global_step": 75,
            "gradient_scores": torch.tensor([0.1, 0.2]),
            # Missing other tensors
        }
        
        self.validator.config = MagicMock(device="cuda")
        self.validator.global_step = 50  # Original value
        
        with patch('torch.load', return_value=partial_state), \
             patch('tplr.logger.info'):
            
            self.validator.load_state()
        
        # Should update available fields
        assert self.validator.global_step == 75
        # Missing fields should remain unchanged (handled by .get() calls)


class TestTrainingPoolDigest:
    """Test training pool digest calculation."""

    def setup_method(self):
        """Setup validator for digest tests."""
        with patch.object(Validator, '__init__', return_value=None):
            self.validator = Validator()
            self.validator.dataset = MagicMock()
            self.validator.hparams = MagicMock(
                inner_steps=10,
                micro_batch_size=2,
                batch_size=8,
                target_batch_size=16
            )

    def test_training_pool_digest_calculation(self):
        """Test training pool digest calculation produces consistent results."""
        uid = 10
        window = 50
        
        # Mock the MinerSampler
        mock_sampler = MagicMock()
        mock_indices = np.array([1, 2, 3, 4, 5])
        mock_ids = [100, 200, 300, 400, 500]
        mock_sampler._global_indices.return_value = mock_indices
        mock_sampler.ids_for_indices.return_value = mock_ids
        
        with patch('tplr.MinerSampler', return_value=mock_sampler):
            digest, sample_count = self.validator._training_pool_digest(uid, window)
        
        # Verify digest is hex string of correct length (32 chars for 16 bytes)
        assert isinstance(digest, str)
        assert len(digest) == 32
        assert all(c in '0123456789abcdef' for c in digest)
        
        # Verify sample count matches mock data
        assert sample_count == len(mock_ids)

    def test_training_pool_digest_consistency(self):
        """Test that same inputs produce same digest."""
        uid = 15
        window = 75
        
        mock_sampler = MagicMock()
        mock_indices = np.array([10, 20, 30])
        mock_ids = [1000, 2000, 3000]
        mock_sampler._global_indices.return_value = mock_indices
        mock_sampler.ids_for_indices.return_value = mock_ids
        
        with patch('tplr.MinerSampler', return_value=mock_sampler):
            digest1, count1 = self.validator._training_pool_digest(uid, window)
            digest2, count2 = self.validator._training_pool_digest(uid, window)
        
        # Should be identical
        assert digest1 == digest2
        assert count1 == count2

    def test_log_digest_match_success(self):
        """Test digest match logging for successful match."""
        uid = 20
        expected_digest = "abcd1234efgh5678"
        expected_count = 100
        
        self.validator.sync_window = 80
        
        # Mock our digest calculation
        with patch.object(self.validator, '_training_pool_digest', 
                         return_value=(expected_digest, expected_count)), \
             patch('tplr.log_with_context') as mock_log:
            
            meta = {
                "sample_digest": expected_digest,
                "sample_count": expected_count
            }
            
            result = self.validator.log_digest_match(uid, meta)
        
        assert result is True
        # Should log success message
        mock_log.assert_called_once()
        call_args = mock_log.call_args[1]
        assert call_args['level'] == 'info'
        assert '✅' in call_args['message']

    def test_log_digest_match_failure(self):
        """Test digest match logging for mismatch."""
        uid = 25
        expected_digest = "correct_digest"
        expected_count = 100
        
        self.validator.sync_window = 85
        
        with patch.object(self.validator, '_training_pool_digest', 
                         return_value=(expected_digest, expected_count)), \
             patch('tplr.log_with_context') as mock_log:
            
            meta = {
                "sample_digest": "wrong_digest",
                "sample_count": 50  # Also wrong count
            }
            
            result = self.validator.log_digest_match(uid, meta)
        
        assert result is False
        # Should log warning message
        mock_log.assert_called_once()
        call_args = mock_log.call_args[1]
        assert call_args['level'] == 'warning'
        assert '❌' in call_args['message']


class TestMinPowerNormalization:
    """Test min_power_normalization utility function."""

    def test_min_power_normalization_basic(self):
        """Test basic power normalization functionality."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = min_power_normalization(logits, power=2.0)
        
        # Should normalize to sum to 1
        assert torch.allclose(result.sum(), torch.tensor(1.0))
        
        # Higher logits should have higher probabilities
        assert result[2] > result[1] > result[0]

    def test_min_power_normalization_scalar_input(self):
        """Test normalization with scalar input."""
        logits = torch.tensor(5.0)  # Scalar
        result = min_power_normalization(logits, power=2.0)
        
        # Should return single probability of 1.0
        assert result.shape == (1,)
        assert torch.allclose(result, torch.tensor([1.0]))

    def test_min_power_normalization_zero_sum(self):
        """Test normalization when powered sum is below epsilon."""
        logits = torch.tensor([0.0, 0.0, 0.0])
        result = min_power_normalization(logits, power=2.0, epsilon=1e-8)
        
        # Should return zeros when sum is too small
        assert torch.allclose(result, torch.zeros_like(logits))

    def test_min_power_normalization_custom_power(self):
        """Test normalization with different power values."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        
        result_p1 = min_power_normalization(logits, power=1.0)
        result_p3 = min_power_normalization(logits, power=3.0)
        
        # Both should sum to 1 but have different distributions
        assert torch.allclose(result_p1.sum(), torch.tensor(1.0))
        assert torch.allclose(result_p3.sum(), torch.tensor(1.0))
        
        # Higher power should make the distribution more peaked
        assert result_p3[2] > result_p1[2]  # Largest element more emphasized

    def test_min_power_normalization_negative_logits(self):
        """Test normalization with negative logits."""
        logits = torch.tensor([-1.0, 0.0, 1.0])
        result = min_power_normalization(logits, power=2.0)
        
        # Should still work and sum to 1
        assert torch.allclose(result.sum(), torch.tensor(1.0))
        assert all(result >= 0)  # All probabilities should be non-negative

    def test_min_power_normalization_custom_epsilon(self):
        """Test normalization with custom epsilon value."""
        logits = torch.tensor([1e-10, 1e-10])
        
        # With large epsilon, should return zeros
        result_large_eps = min_power_normalization(logits, power=2.0, epsilon=1e-5)
        assert torch.allclose(result_large_eps, torch.zeros_like(logits))
        
        # With small epsilon, should normalize
        result_small_eps = min_power_normalization(logits, power=2.0, epsilon=1e-25)
        assert torch.allclose(result_small_eps.sum(), torch.tensor(1.0))


class TestAsyncFunctionality:  
    """Test async methods in validator."""

    def setup_method(self):
        """Setup validator for async tests."""
        with patch.object(Validator, '__init__', return_value=None):
            self.validator = Validator()
            self.validator.comms = MagicMock()
            self.validator.config = MagicMock(device="cuda")
            self.validator.sync_window = 100
            self.validator.current_window = 100
            self.validator.lr = 0.001
            self.validator.param_avg_change = {}

    @pytest.mark.asyncio
    async def test_evaluate_miner_sync_success(self):
        """Test successful miner sync evaluation."""
        eval_uid = 10
        mock_debug_dict = {"param1_debug": [0.1, 0.2]}
        
        self.validator.comms.get.return_value = (mock_debug_dict, None)
        
        mock_comparison = {
            "success": True,
            "avg_steps_behind": 2.0,
            "l2_norm": 0.5
        }
        
        with patch('tplr.neurons.compare_model_with_debug_dict', 
                  return_value=mock_comparison):
            
            result = await self.validator.evaluate_miner_sync(eval_uid)
        
        assert result["success"] is True
        assert "sync_score" in result
        assert result["avg_steps_behind"] == 2.0
        
        # Verify sync score calculation: (1 - 2.0/5.0)^2.5 = 0.6^2.5 ≈ 0.373
        expected_sync_score = (1.0 - min(2.0, 5.0) / 5.0) ** 2.5
        assert abs(result["sync_score"] - expected_sync_score) < 0.001

    @pytest.mark.asyncio
    async def test_evaluate_miner_sync_no_debug_data(self):
        """Test miner sync evaluation when no debug data available."""
        eval_uid = 15
        self.validator.comms.get.return_value = None
        
        result = await self.validator.evaluate_miner_sync(eval_uid)
        
        assert result["success"] is False
        assert result["sync_score"] == 0.0
        assert "Failed to retrieve debug dictionary" in result["error"]

    @pytest.mark.asyncio
    async def test_evaluate_miner_sync_invalid_debug_format(self):
        """Test miner sync evaluation with invalid debug format."""
        eval_uid = 20
        self.validator.comms.get.return_value = ("invalid_format", None)
        
        result = await self.validator.evaluate_miner_sync(eval_uid)
        
        assert result["success"] is False
        assert result["sync_score"] == 0.0
        assert "Invalid debug dictionary format" in result["error"]

    @pytest.mark.asyncio
    async def test_evaluate_miner_sync_comparison_failure(self):
        """Test miner sync evaluation when comparison fails."""
        eval_uid = 25
        mock_debug_dict = {"param1_debug": [0.1, 0.2]}
        
        self.validator.comms.get.return_value = (mock_debug_dict, None)
        
        mock_comparison = {"success": False}
        
        with patch('tplr.neurons.compare_model_with_debug_dict', 
                  return_value=mock_comparison):
            
            result = await self.validator.evaluate_miner_sync(eval_uid)
        
        assert result["success"] is False
        assert result["sync_score"] == 0.0

    @pytest.mark.asyncio
    async def test_create_checkpoint_async(self):
        """Test async checkpoint creation."""
        # Setup model state
        mock_model = MagicMock()
        mock_state_dict = {
            "layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "layer.bias": torch.tensor([0.1, 0.2])
        }
        mock_model.state_dict.return_value = mock_state_dict
        
        self.validator.model = mock_model
        self.validator.start_window = 10
        self.validator.current_window = 50
        self.validator.sync_window = 45
        self.validator.executor = MagicMock()  # ThreadPoolExecutor mock
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_in_executor.return_value = {
                "model_state_dict": {k: v.cpu().clone() for k, v in mock_state_dict.items()},
                "start_window": 10,
                "current_window": 50,
                "sync_window": 45
            }
            
            result = await self.validator.create_checkpoint_async()
        
        assert "model_state_dict" in result
        assert "start_window" in result
        assert "current_window" in result
        assert "sync_window" in result
        
        assert result["start_window"] == 10
        assert result["current_window"] == 50
        assert result["sync_window"] == 45

    @pytest.mark.asyncio
    async def test_upload_gather_results_highest_stake(self):
        """Test gather results upload for highest stake validator."""
        # Setup as highest stake validator
        self.validator.uid = 5
        mock_metagraph = MagicMock()
        mock_metagraph.S.argmax.return_value.item.return_value = 5
        self.validator.metagraph = mock_metagraph
        self.validator.sync_window = 100
        self.validator.current_window = 105
        
        # Create mock gather result
        mock_state_dict = {
            "param1": torch.tensor([1.0, 2.0]),
            "param2": torch.tensor([[3.0, 4.0], [5.0, 6.0]])
        }
        gather_result = SimpleNamespace(
            state_dict=mock_state_dict,  
            uids=[10, 20, 30],
            skipped_uids=[40],
            success_rate=0.75
        )
        
        self.validator.comms.put = AsyncMock()
        
        with patch('tplr.log_with_context'):
            await self.validator.upload_gather_results(gather_result)
        
        # Verify upload was attempted
        self.validator.comms.put.assert_called_once()
        call_args = self.validator.comms.put.call_args
        
        # Verify payload structure
        payload = call_args[1]["state_dict"]
        assert "state_dict" in payload
        assert "uids" in payload
        assert "skipped_uids" in payload
        assert "success_rate" in payload
        
        assert payload["uids"] == [10, 20, 30]
        assert payload["skipped_uids"] == [40]
        assert payload["success_rate"] == 0.75

    @pytest.mark.asyncio
    async def test_upload_gather_results_not_highest_stake(self):
        """Test gather results upload skipped for non-highest stake validator."""
        # Setup as non-highest stake validator
        self.validator.uid = 10
        mock_metagraph = MagicMock()
        mock_metagraph.S.argmax.return_value.item.return_value = 5  # Different UID
        self.validator.metagraph = mock_metagraph
        
        gather_result = SimpleNamespace(
            state_dict={},
            uids=[],
            skipped_uids=[],
            success_rate=1.0
        )
        
        self.validator.comms.put = AsyncMock()
        
        await self.validator.upload_gather_results(gather_result)
        
        # Should not upload
        self.validator.comms.put.assert_not_called()

    @pytest.mark.asyncio 
    async def test_upload_gather_results_with_exception(self):
        """Test gather results upload handles exceptions gracefully."""
        # Setup as highest stake validator
        self.validator.uid = 1
        mock_metagraph = MagicMock()
        mock_metagraph.S.argmax.return_value.item.return_value = 1
        self.validator.metagraph = mock_metagraph
        self.validator.sync_window = 50
        self.validator.current_window = 55
        
        gather_result = SimpleNamespace(
            state_dict={},
            uids=[],
            skipped_uids=[],
            success_rate=1.0
        )
        
        self.validator.comms.put = AsyncMock(side_effect=Exception("Upload failed"))
        
        with patch('tplr.log_with_context') as mock_log:
            # Should not raise exception
            await self.validator.upload_gather_results(gather_result)
        
        # Should log warning
        warning_calls = [c for c in mock_log.call_args_list 
                        if len(c[1]) > 0 and c[1].get('level') == 'warning']
        assert len(warning_calls) > 0