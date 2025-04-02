import math
import pytest
import torch
import torch.nn as nn

from tplr.neurons import compare_model_with_debug_dict


class SimpleModel(nn.Module):
    """A simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


@pytest.fixture
def setup_model():
    """Create a model with deterministic weights for testing."""
    model = SimpleModel()

    # Set deterministic weights for reproducible tests
    with torch.no_grad():
        # Set specific values for the first layer
        model.linear1.weight.fill_(0.1)
        model.linear1.bias.fill_(0.01)

        # Set specific values for the second layer
        model.linear2.weight.fill_(0.2)
        model.linear2.bias.fill_(0.02)

    return model


@pytest.mark.asyncio
async def test_exact_match(setup_model):
    """Test when model parameters exactly match debug dict values."""
    model = setup_model

    # Create debug dict that exactly matches the model parameters
    debug_dict = {}
    for name, param in model.named_parameters():
        debug_dict[name + "_debug"] = param.flatten()[:2].detach().cpu().tolist()

    learning_rate = 0.01

    # Compare model with debug dict
    result = await compare_model_with_debug_dict(model, debug_dict, learning_rate)

    # Verify the results
    assert result["success"] is True
    assert result["l2_norm"] == pytest.approx(0.0, abs=1e-6)
    assert result["avg_l2_norm"] == pytest.approx(0.0, abs=1e-6)
    assert result["avg_abs_diff"] == pytest.approx(0.0, abs=1e-6)
    assert result["max_diff"] == pytest.approx(0.0, abs=1e-6)
    assert result["avg_steps_behind"] == pytest.approx(0.0, abs=1e-6)
    assert result["max_steps_behind"] == pytest.approx(0.0, abs=1e-6)
    assert result["param_count"] > 0
    assert result["learning_rate"] == 0.01


@pytest.mark.asyncio
async def test_one_step_behind(setup_model):
    """Test when model parameters are one step behind debug dict values."""
    model = setup_model
    learning_rate = 0.01

    # Create debug dict with values that are one step ahead
    debug_dict = {}
    for name, param in model.named_parameters():
        # Make debug values one learning_rate step ahead
        values = param.flatten()[:2].detach().cpu()
        ahead_values = values + learning_rate
        debug_dict[name + "_debug"] = ahead_values.tolist()

    # Compare model with debug dict
    result = await compare_model_with_debug_dict(model, debug_dict, learning_rate)

    # Verify the results
    assert result["success"] is True
    # Each parameter should be exactly one step behind
    assert result["avg_steps_behind"] == pytest.approx(1.0, abs=1e-2)
    assert result["max_steps_behind"] == pytest.approx(1.0, abs=1e-2)


@pytest.mark.asyncio
async def test_multiple_steps_behind(setup_model):
    """Test when model parameters are multiple steps behind debug dict values."""
    model = setup_model
    learning_rate = 0.01
    steps_behind = 5.0

    # Create debug dict with values that are multiple steps ahead
    debug_dict = {}
    for name, param in model.named_parameters():
        # Make debug values multiple learning_rate steps ahead
        values = param.flatten()[:2].detach().cpu()
        ahead_values = values + (learning_rate * steps_behind)
        debug_dict[name + "_debug"] = ahead_values.tolist()

    # Compare model with debug dict
    result = await compare_model_with_debug_dict(model, debug_dict, learning_rate)

    # Verify the results
    assert result["success"] is True
    assert result["avg_steps_behind"] == pytest.approx(steps_behind, abs=1e-2)
    assert result["max_steps_behind"] == pytest.approx(steps_behind, abs=1e-2)


@pytest.mark.asyncio
async def test_missing_parameters(setup_model):
    """Test with a debug dict missing some parameters."""
    model = setup_model

    # Create debug dict with only one parameter
    debug_dict = {}
    first_param = next(iter(model.named_parameters()))
    name, param = first_param
    debug_dict[name + "_debug"] = param.flatten()[:2].detach().cpu().tolist()

    learning_rate = 0.01

    # Compare model with debug dict
    result = await compare_model_with_debug_dict(model, debug_dict, learning_rate)

    # Verify the results
    assert result["success"] is True
    # Should only count the parameters that were found in debug_dict
    assert result["param_count"] == 2  # Two values from the first parameter
    assert result["l2_norm"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_empty_debug_dict(setup_model):
    """Test with an empty debug dict."""
    model = setup_model
    debug_dict = {}
    learning_rate = 0.01

    # Compare model with debug dict
    result = await compare_model_with_debug_dict(model, debug_dict, learning_rate)

    # Verify the results
    assert result["success"] is True
    assert result["param_count"] == 0
    assert result["l2_norm"] == pytest.approx(0.0, abs=1e-6)
    assert math.isinf(result["avg_l2_norm"])
    assert math.isinf(result["avg_steps_behind"])


@pytest.mark.asyncio
async def test_different_devices():
    """Test comparing model on different devices if CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping test")

    # Create model on CPU
    cpu_model = SimpleModel()

    # Create debug dict from CPU model
    debug_dict = {}
    for name, param in cpu_model.named_parameters():
        debug_dict[name + "_debug"] = param.flatten()[:2].detach().cpu().tolist()

    # Move model to CUDA
    cuda_model = SimpleModel().to("cuda")

    # Copy weights from CPU model to ensure they match
    with torch.no_grad():
        for (_, cpu_param), (_, cuda_param) in zip(
            cpu_model.named_parameters(), cuda_model.named_parameters()
        ):
            cuda_param.copy_(cpu_param)

    learning_rate = 0.01

    # Compare CUDA model with CPU debug dict
    result = await compare_model_with_debug_dict(cuda_model, debug_dict, learning_rate)

    # Verify the results
    assert result["success"] is True
    assert result["l2_norm"] == pytest.approx(0.0, abs=1e-6)
    assert result["avg_steps_behind"] == pytest.approx(0.0, abs=1e-6)
