"""Unit tests for validator functionality"""
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch
from ..utils.assertions import assert_tensor_equal, assert_gradient_valid
from ..mocks.bittensor import mock_bt
from ..mocks.model import MockLlamaForCausalLM

# Mark all tests as async
pytestmark = pytest.mark.asyncio

@pytest.fixture(autouse=True)
def mock_llama(monkeypatch):
    """Mock LlamaForCausalLM"""
    monkeypatch.setattr(
        "neurons.validator.LlamaForCausalLM",
        MockLlamaForCausalLM
    )

class TestValidatorBasicEvaluation:
    """Test basic evaluation flow"""
    
    @pytest.fixture
    async def validator_instance(self, mock_model, mock_transformer, mock_compressor):
        """Create validator instance with mocked config"""
        from neurons.validator import Validator
        
        # Create mock config with all required attributes
        mock_config = SimpleNamespace(
            netuid=1,
            device='cpu',
            debug=False,
            trace=False,
            project='test_project',
            peers=[],
            store_gathers=False
        )
        
        # Mock all dependencies
        with patch.dict('sys.modules', {'bittensor': mock_bt, 'bt': mock_bt}), \
             patch.object(Validator, 'config', return_value=mock_config), \
             patch('tplr.load_hparams', return_value=SimpleNamespace(
                blocks_per_window=100,
                target_chunk=512,
                topk_compression=0.1,
                catch_up_threshold=5,
                catch_up_min_peers=1,
                catch_up_batch_size=10,
                catch_up_timeout=300,
                active_check_interval=60,
                recent_windows=5,
                validator_sample_rate=0.5,
                ma_alpha=0.9,
                windows_per_weights=10,
                model_config={},
                tokenizer=mock_model.tokenizer,
                learning_rate=0.01,
                power_normalisation=2.0,
                checkpoint_frequency=100
             )), \
             patch('tplr.compress.TransformDCT', return_value=mock_transformer), \
             patch('tplr.compress.CompressDCT', return_value=mock_compressor), \
             patch('tplr.initialize_wandb'), \
             patch('tplr.comms.Comms'), \
             patch('transformers.LlamaForCausalLM', return_value=mock_model):
            
            validator = Validator()
            return validator

    async def test_basic_evaluation_flow(self, validator_instance):
        """Test basic evaluation with both own and random data"""
        # Setup test data
        own_data = torch.randn(2, 128)
        random_data = torch.randn(2, 128)
        
        validator_instance.own_dataset = SimpleNamespace(
            __iter__=lambda self: iter([own_data])
        )
        validator_instance.random_dataset = SimpleNamespace(
            __iter__=lambda self: iter([random_data])
        )
        
        result = await validator_instance.evaluate_batch()
        
        assert result is not None
        assert hasattr(result, 'own_improvement')
        assert hasattr(result, 'random_improvement')
        assert -1 <= validator_instance.binary_moving_averages[0] <= 1
        assert validator_instance.moving_avg_scores[0] >= 0

    async def test_sampling_rate_consistency(self, validator_instance):
        """Test sampling rate is consistently applied"""
        sample_rate = validator_instance.hparams.validator_sample_rate
        
        # Run multiple evaluations
        results = []
        for _ in range(10):
            result = await validator_instance.evaluate_batch()
            results.append(result)
            
        # Verify sampling rate
        sampled_count = sum(1 for r in results if r is not None)
        expected_count = int(10 * sample_rate)
        assert abs(sampled_count - expected_count) <= 1  # Allow small variance

    async def test_moving_averages_computation(self, validator_instance):
        """Test moving average calculations"""
        alpha = validator_instance.hparams.ma_alpha
        
        # Initial values
        initial_binary = validator_instance.binary_moving_avg
        initial_score = validator_instance.score_moving_avg
        
        # Simulate evaluation with known improvements
        result = SimpleNamespace(
            own_improvement=0.5,
            random_improvement=0.3
        )
        
        validator_instance.update_moving_averages(result)
        
        # Verify binary indicator
        binary_indicator = 1 if result.own_improvement > result.random_improvement else -1
        expected_binary = alpha * initial_binary + (1 - alpha) * binary_indicator
        assert_tensor_equal(
            validator_instance.binary_moving_avg,
            expected_binary,
            "Binary moving average incorrect"
        )
        
        # Verify score average
        score = max(0, result.own_improvement - result.random_improvement)
        expected_score = alpha * initial_score + (1 - alpha) * score
        assert_tensor_equal(
            validator_instance.score_moving_avg,
            expected_score,
            "Score moving average incorrect"
        )

class TestValidatorEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @pytest.fixture
    async def validator_instance(self, mock_model, mock_transformer, mock_compressor):
        """Create validator instance with standard mocks"""
        from neurons.validator import Validator
        
        mock_config = SimpleNamespace(
            netuid=1,
            device='cpu',
            debug=False,
            trace=False,
            project='test_project',
            peers=[],
            store_gathers=False
        )
        
        with patch.object(Validator, 'config', return_value=mock_config), \
             patch('tplr.load_hparams', return_value=SimpleNamespace(
                blocks_per_window=100,
                target_chunk=512,
                topk_compression=0.1,
                catch_up_threshold=5,
                catch_up_min_peers=1,
                catch_up_batch_size=10,
                catch_up_timeout=300,
                active_check_interval=60,
                recent_windows=5,
                validator_sample_rate=0.5,
                ma_alpha=0.9,
                windows_per_weights=10,
                model_config={},
                tokenizer=mock_model.tokenizer,
                learning_rate=0.01,
                power_normalisation=2.0,
                checkpoint_frequency=100
             )), \
             patch('tplr.compress.TransformDCT', return_value=mock_transformer), \
             patch('tplr.compress.CompressDCT', return_value=mock_compressor), \
             patch('tplr.initialize_wandb'), \
             patch('tplr.comms.Comms'), \
             patch('transformers.LlamaForCausalLM', return_value=mock_model):
            
            validator = Validator()
            return validator

    async def test_zero_gradient_handling(self, validator_instance):
        """Test handling of zero/near-zero gradients"""
        # Setup test data with zero gradients
        own_data = torch.zeros(2, 128)
        random_data = torch.zeros(2, 128)
        
        validator_instance.own_dataset = SimpleNamespace(
            __iter__=lambda self: iter([own_data])
        )
        validator_instance.random_dataset = SimpleNamespace(
            __iter__=lambda self: iter([random_data])
        )
        
        result = await validator_instance.evaluate_batch()
        
        # Verify zero gradient handling
        assert result is not None
        assert result.own_improvement >= 0
        assert result.random_improvement >= 0
        assert_gradient_valid(validator_instance.binary_moving_avg)
        assert_gradient_valid(validator_instance.score_moving_avg)

    async def test_large_gradient_handling(self, validator_instance):
        """Test handling of unusually large gradients"""
        # Setup test data with large values
        own_data = torch.randn(2, 128) * 1e6
        random_data = torch.randn(2, 128) * 1e6
        
        validator_instance.own_dataset = SimpleNamespace(
            __iter__=lambda self: iter([own_data])
        )
        validator_instance.random_dataset = SimpleNamespace(
            __iter__=lambda self: iter([random_data])
        )
        
        result = await validator_instance.evaluate_batch()
        
        # Verify large gradient handling
        assert result is not None
        assert torch.isfinite(torch.tensor(result.own_improvement))
        assert torch.isfinite(torch.tensor(result.random_improvement))
        assert_gradient_valid(validator_instance.binary_moving_avg)
        assert_gradient_valid(validator_instance.score_moving_avg)

    async def test_moving_average_edge_cases(self, validator_instance):
        """Test moving average behavior in edge cases"""
        # Test initial state
        assert validator_instance.binary_moving_avg == 0
        assert validator_instance.score_moving_avg == 0
        
        # Test extreme binary indicators
        for improvement in [(1.0, 0.0), (0.0, 1.0)]:  # (own, random)
            result = SimpleNamespace(
                own_improvement=improvement[0],
                random_improvement=improvement[1]
            )
            validator_instance.update_moving_averages(result)
            assert -1 <= validator_instance.binary_moving_avg <= 1
            assert validator_instance.score_moving_avg >= 0

class TestValidatorMemoryManagement:
    """Test memory cleanup and efficiency"""
    
    @pytest.fixture
    async def validator_instance(self, mock_model, mock_transformer, mock_compressor):
        """Create validator instance with memory tracking"""
        from neurons.validator import Validator
        import gc
        import psutil
        
        # Record initial memory state
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss
        
        mock_config = SimpleNamespace(
            netuid=1,
            device='cpu',
            debug=False,
            trace=False,
            project='test_project',
            peers=[],
            store_gathers=False
        )
        
        with patch.object(Validator, 'config', return_value=mock_config), \
             patch('tplr.load_hparams', return_value=SimpleNamespace(
                blocks_per_window=100,
                target_chunk=512,
                topk_compression=0.1,
                catch_up_threshold=5,
                catch_up_min_peers=1,
                catch_up_batch_size=10,
                catch_up_timeout=300,
                active_check_interval=60,
                recent_windows=5,
                validator_sample_rate=0.5,
                ma_alpha=0.9,
                windows_per_weights=10,
                model_config={},
                tokenizer=mock_model.tokenizer,
                learning_rate=0.01,
                power_normalisation=2.0,
                checkpoint_frequency=100
             )), \
             patch('tplr.compress.TransformDCT', return_value=mock_transformer), \
             patch('tplr.compress.CompressDCT', return_value=mock_compressor), \
             patch('tplr.initialize_wandb'), \
             patch('tplr.comms.Comms'), \
             patch('transformers.LlamaForCausalLM', return_value=mock_model):
            
            validator = Validator()
            yield validator
            
            # Cleanup
            gc.collect()
            final_memory = psutil.Process().memory_info().rss
            assert final_memory <= initial_memory * 1.1  # Allow 10% overhead

    async def test_memory_cleanup(self, validator_instance):
        """Test proper cleanup of temporary resources"""
        import gc
        import psutil
        
        # Record memory before operations
        gc.collect()
        start_memory = psutil.Process().memory_info().rss
        
        # Perform multiple evaluations
        for _ in range(5):
            result = await validator_instance.evaluate_batch()
            assert result is not None
        
        # Force cleanup
        gc.collect()
        end_memory = psutil.Process().memory_info().rss
        
        # Verify no significant memory growth
        assert end_memory <= start_memory * 1.1  # Allow 10% overhead

    async def test_large_batch_memory(self, validator_instance):
        """Test memory efficiency with large batches"""
        # Setup large test data
        batch_size = 32
        seq_length = 512
        own_data = torch.randn(batch_size, seq_length)
        random_data = torch.randn(batch_size, seq_length)
        
        validator_instance.own_dataset = SimpleNamespace(
            __iter__=lambda self: iter([own_data])
        )
        validator_instance.random_dataset = SimpleNamespace(
            __iter__=lambda self: iter([random_data])
        )
        
        import gc
        import psutil
        
        # Record memory before large batch
        gc.collect()
        start_memory = psutil.Process().memory_info().rss
        
        result = await validator_instance.evaluate_batch()
        
        # Force cleanup
        gc.collect()
        end_memory = psutil.Process().memory_info().rss
        
        # Verify memory usage
        assert result is not None
        assert end_memory <= start_memory * 1.5  # Allow 50% overhead for large batch

