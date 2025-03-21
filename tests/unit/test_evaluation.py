import torch
from unittest.mock import MagicMock

# Import evaluation functions
from tplr import evaluation

# Import relevant mocks from tests/mocks
from tests.mocks.model import MockModel, MockTransformer, MockCompressor


# ----------------------------
# Test Class: TestApplyCompressedGradient
# ----------------------------
class TestApplyCompressedGradient:
    def test_valid_state_dict_updates_gradients(self):
        """
        Test that when a valid state_dict containing expected keys is provided,
        apply_compressed_gradient decodes, decompresses, and applies a sign-based update.

        Steps:
          - Create a MockModel instance.
          - Create dummy state_dict with keys like "layer1.weightidxs" and "layer1.weightvals".
          - Use MockTransformer (for decoding) and MockCompressor (for decompression).
          - Set dummy xshapes and totalks for each parameter.
          - Call apply_compressed_gradient with a fixed learning rate.
          - Verify that each parameter is updated by subtracting lr * sign(dummy_gradient).
        """
        model = MockModel()
        device = torch.device("cpu")
        lr = 0.01
        transformer = MockTransformer()
        compressor = MockCompressor()
        xshapes = {"layer1.weight": (10, 10), "layer1.bias": (10,)}
        totalks = {"layer1.weight": 50, "layer1.bias": 5}
        state_dict = {
            "layer1.weightidxs": torch.tensor([0, 1]),
            "layer1.weightvals": torch.tensor([0.1, -0.1]),
            "layer1.biasidxs": torch.tensor([0]),
            "layer1.biasvals": torch.tensor([0.2]),
        }
        original_params = {n: p.data.clone() for n, p in model.named_parameters()}

        updated_model = evaluation.apply_compressed_gradient(
            model, state_dict, transformer, compressor, xshapes, totalks, device, lr
        )

        # We assume the mocks return a tensor of ones. Thus, the update subtracts lr * ones.
        for n, p in updated_model.named_parameters():
            expected = original_params[n] - lr * torch.ones_like(p)
            assert torch.allclose(p.data, expected, atol=1e-6)

    def test_missing_gradient_keys_skips_parameter(self):
        """
        Test that parameters missing the required gradient keys in state_dict remain unchanged.

        Steps:
          - Create a model with at least two parameters.
          - Provide a state_dict that includes gradient data for only one parameter.
          - Call apply_compressed_gradient.
          - Verify that the parameter(s) missing gradient keys are not updated.
        """
        model = MockModel()
        device = torch.device("cpu")
        lr = 0.01
        transformer = MockTransformer()
        compressor = MockCompressor()
        xshapes = {"layer1.weight": (10, 10), "layer1.bias": (10,)}
        totalks = {"layer1.weight": 50, "layer1.bias": 5}
        state_dict = {
            "layer1.weightidxs": torch.tensor([0, 1]),
            "layer1.weightvals": torch.tensor([0.1, -0.1]),
            # Missing keys for "layer1.bias"
        }
        original_params = {n: p.data.clone() for n, p in model.named_parameters()}

        updated_model = evaluation.apply_compressed_gradient(
            model, state_dict, transformer, compressor, xshapes, totalks, device, lr
        )
        for n, p in updated_model.named_parameters():
            if "bias" in n:
                assert torch.allclose(p.data, original_params[n], atol=1e-6)
            else:
                expected = original_params[n] - lr * torch.ones_like(p)
                assert torch.allclose(p.data, expected, atol=1e-6)


# ----------------------------
# Test Class: TestComputeAverageLoss
# ----------------------------
class TestComputeAverageLoss:
    def test_average_loss_computation(self):
        """
        Validate that compute_average_loss returns the correct average loss.

        Steps:
          - Create a dummy model with a forward method returning constant loss (e.g. 3.0).
          - Define dummy batches (e.g. lists of token IDs).
          - Use a dummy tokenizer with a set pad_token_id.
          - Set sample_rate to 1 so that all batches are sampled.
          - Verify that the average loss computed equals the expected constant.
        """
        model = MockModel()
        batches = [[1, 2, 3, 0] for _ in range(3)]
        tokenizer = type("DummyTokenizer", (), {"pad_token_id": 0})
        device = torch.device("cpu")
        sample_rate = 1.0

        avg_loss, count, sampled_indices, total_batches = (
            evaluation.compute_average_loss(
                model, batches, tokenizer, device, sample_rate
            )
        )
        expected_loss = 3.0
        assert count == len(sampled_indices)
        assert total_batches == len(batches)
        assert abs(avg_loss - expected_loss) < 1e-6

    def test_sample_rate_edge_case(self):
        """
        Ensure that even with an extremely low sample_rate, at least one batch is sampled.
        """
        model = MockModel()
        batches = [[1, 2, 3]]
        tokenizer = type("DummyTokenizer", (), {"pad_token_id": 0})
        device = torch.device("cpu")
        sample_rate = 0.01

        avg_loss, count, sampled_indices, total_batches = (
            evaluation.compute_average_loss(
                model, batches, tokenizer, device, sample_rate
            )
        )
        assert count >= 1
        assert len(sampled_indices) >= 1


# ----------------------------
# Test Class: TestEvaluateLossChange
# ----------------------------
class TestEvaluateLossChange:
    def test_loss_change_computation(self):
        """
        Validate that evaluate_loss_change returns loss_before and loss_after that differ
        when a gradient update is applied.

        Steps:
          - Set up a dummy model, batches, tokenizer, and a dummy scheduler (with get_last_lr).
          - Use a valid state_dict, MockTransformer, and MockCompressor.
          - Call evaluate_loss_change and verify the outputs are numeric.
        """
        model = MockModel()
        batches = [[1, 2, 3, 0] for _ in range(3)]
        tokenizer = type("DummyTokenizer", (), {"pad_token_id": 0})
        device = torch.device("cpu")
        sample_rate = 1.0
        dummy_scheduler = MagicMock()
        dummy_scheduler.get_last_lr.return_value = [0.01]

        transformer = MockTransformer()
        compressor = MockCompressor()
        xshapes = {"layer1.weight": (10, 10), "layer1.bias": (10,)}
        totalks = {"layer1.weight": 50, "layer1.bias": 5}
        state_dict = {
            "layer1.weightidxs": torch.tensor([0, 1]),
            "layer1.weightvals": torch.tensor([0.1, -0.1]),
            "layer1.biasidxs": torch.tensor([0]),
            "layer1.biasvals": torch.tensor([0.2]),
        }

        (
            loss_before,
            loss_after,
            count_before,
            count_after,
            sampled_indices,
            total_batches,
        ) = evaluation.evaluate_loss_change(
            model,
            batches,
            tokenizer,
            device,
            sample_rate,
            state_dict,
            transformer,
            compressor,
            xshapes,
            totalks,
            dummy_scheduler,
        )
        assert isinstance(loss_before, float)
        assert isinstance(loss_after, float)

    def test_improvement_metrics_with_zero_loss(self):
        """
        Edge case: If loss_before is zero, the relative improvement should be set to 0.0.
        """
        loss_before_own = 0.0
        loss_after_own = 0.0
        loss_before_random = 10.0
        loss_after_random = 9.0
        rel_own, rel_random, grad_score, binary_indicator = (
            evaluation.compute_improvement_metrics(
                loss_before_own, loss_after_own, loss_before_random, loss_after_random
            )
        )
        assert rel_own == 0.0
        assert binary_indicator == -1


# ----------------------------
# Test Class: TestComputeImprovementMetrics
# ----------------------------
class TestComputeImprovementMetrics:
    def test_improvement_metrics_normal(self):
        """
        Validate that with nonzero loss improvement values, the relative improvements and
        binary indicator are computed correctly.

        Steps:
          - Provide loss_before and loss_after for both own and random evaluations.
          - Verify that the binary indicator is 1 if own improvement is greater.
        """
        loss_before_own = 10.0
        loss_after_own = 8.0  # Improvement of 2
        loss_before_random = 10.0
        loss_after_random = 9.0  # Improvement of 1
        rel_own, rel_random, grad_score, binary_indicator = (
            evaluation.compute_improvement_metrics(
                loss_before_own, loss_after_own, loss_before_random, loss_after_random
            )
        )
        assert binary_indicator == 1
        assert rel_own == 0.2
        assert rel_random == 0.1


# ----------------------------
# Test Class: TestComputeAvgLoss
# ----------------------------
class TestComputeAvgLoss:
    def test_avg_loss_correct_when_batches_selected(self):
        """
        Verify that compute_avg_loss computes the average loss over specified batches correctly.

        Steps:
          - Provide a list of batches and pre-determined sampled_indices.
          - Assert that the returned average loss equals the constant dummy loss.
        """
        model = MockModel()
        batches = [[1, 2, 3, 0], [1, 2, 3, 0]]
        sampled_indices = [0, 1]
        tokenizer = type("DummyTokenizer", (), {"pad_token_id": 0})
        device = torch.device("cpu")
        avg_loss, count = evaluation.compute_avg_loss(
            model, batches, sampled_indices, tokenizer, device
        )
        assert count == len(sampled_indices)
        assert abs(avg_loss - 3.0) < 1e-6

    def test_avg_loss_empty_sampled_indices(self):
        """
        Edge case: If no batches are selected (empty sampled_indices), the function should return 0.0 and count 0.
        """
        model = MockModel()
        batches = [[1, 2, 3, 0]]
        sampled_indices = []
        tokenizer = type("DummyTokenizer", (), {"pad_token_id": 0})
        device = torch.device("cpu")
        avg_loss, count = evaluation.compute_avg_loss(
            model, batches, sampled_indices, tokenizer, device
        )
        assert count == 0
        assert avg_loss == 0.0


# ----------------------------
# Test Class: TestApplyGradientUpdate
# ----------------------------
class TestApplyGradientUpdate:
    def test_valid_state_dict_applies_update(self):
        """
        Validate that apply_gradient_update applies a sign-based update to model parameters.

        Steps:
          - Create a dummy model.
          - Provide a state_dict with all required keys.
          - Before calling apply_gradient_update, store a copy of the model parameters.
          - After the update, verify that each parameter's data has been updated correctly
            (expected: original_value - lr * sign(dummy_gradient)).
        """
        model = MockModel()
        device = torch.device("cpu")
        lr = 0.01
        transformer = MockTransformer()
        compressor = MockCompressor()
        xshapes = {"layer1.weight": (10, 10), "layer1.bias": (10,)}
        totalks = {"layer1.weight": 50, "layer1.bias": 5}
        state_dict = {
            "layer1.weightidxs": torch.tensor([0, 1]),
            "layer1.weightvals": torch.tensor([0.1, -0.1]),
            "layer1.biasidxs": torch.tensor([0]),
            "layer1.biasvals": torch.tensor([0.2]),
        }
        original_params = {n: p.data.clone() for n, p in model.named_parameters()}

        evaluation.apply_gradient_update(
            model, state_dict, transformer, compressor, xshapes, totalks, device, lr
        )
        # Assuming that the mocks yield a tensor of ones, the update subtracts lr * ones from each parameter.
        for n, p in model.named_parameters():
            expected = original_params[n] - lr * torch.ones_like(original_params[n])
            assert torch.allclose(p.data, expected, atol=1e-6)

    def test_missing_keys_in_state_dict(self):
        """
        Edge case: If state_dict lacks keys for certain parameters, those parameters should remain unchanged.

        Steps:
          - Create a state_dict that is empty.
          - Store the original parameter values.
          - Call apply_gradient_update and verify that parameter data is unchanged.
        """
        model = MockModel()
        device = torch.device("cpu")
        lr = 0.01
        transformer = MockTransformer()
        compressor = MockCompressor()
        xshapes = {"layer1.weight": (10, 10), "layer1.bias": (10,)}
        totalks = {"layer1.weight": 50, "layer1.bias": 5}
        state_dict = {}
        original_params = {n: p.data.clone() for n, p in model.named_parameters()}

        evaluation.apply_gradient_update(
            model, state_dict, transformer, compressor, xshapes, totalks, device, lr
        )
        # No update should be applied, so the parameter data should remain the same.
        for n, p in model.named_parameters():
            assert torch.allclose(p.data, original_params[n], atol=1e-6)


# ----------------------------
# Test Class: TestWeightedRandomSampleNoReplacement
# ----------------------------
class TestWeightedRandomSampleNoReplacement:
    def test_returns_all_candidates_if_k_exceeds_length(self):
        """
        Verify that if the number of candidates is less than or equal to k, all candidates are returned.
        """
        candidates = ["a", "b", "c"]
        weights = [1, 1, 1]
        k = 5
        selected = evaluation.weighted_random_sample_no_replacement(
            candidates, weights, k
        )
        assert set(selected) == set(candidates)

    def test_selects_random_sample_based_on_weights(self):
        """
        Validate that given a candidate list with weights, exactly k unique candidates are returned.

        Note: This test should ideally be run multiple times to statistically observe
        that items with higher weights are selected more frequently.
        """
        candidates = ["a", "b", "c", "d", "e"]
        weights = [10, 5, 1, 3, 7]
        k = 3
        selected = evaluation.weighted_random_sample_no_replacement(
            candidates, weights, k
        )
        assert len(set(selected)) == k

    def test_invalid_inputs_return_empty_list(self):
        """
        Edge case: If candidates/weights are empty or k <= 0, an empty list is returned.
        """
        selected = evaluation.weighted_random_sample_no_replacement([], [], 0)
        assert selected == []

    def test_total_weight_zero_returns_random_sample(self):
        """
        Edge case: When the sum of weights is zero, the function should fall back to random sampling.
        """
        candidates = ["a", "b", "c"]
        weights = [0, 0, 0]
        k = 2
        selected = evaluation.weighted_random_sample_no_replacement(
            candidates, weights, k
        )
        assert len(selected) == k


# ----------------------------
# Test Class: TestSafeLast
# ----------------------------
class TestSafeLast:
    def test_returns_last_element(self):
        """
        Verify that safe_last returns the last element of a non-empty list.
        """
        metric_list = [1, 2, 3, 4]
        result = evaluation.safe_last(metric_list)
        assert result == 4

    def test_empty_list_returns_zero(self):
        """
        Edge case: When provided with an empty list, safe_last returns 0.0.
        """
        result = evaluation.safe_last([])
        assert result == 0.0
