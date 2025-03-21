"""Mock model and related components"""

from .base import BaseMock
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from unittest.mock import MagicMock
from transformers import PretrainedConfig


# DummyLoss/DummyOutput define a constant loss output (3.0) to be used for testing.
class DummyLoss:
    def __init__(self, value=3.0):
        self.value = value

    def item(self):
        return self.value


class DummyOutput:
    def __init__(self, loss_value=3.0):
        self._loss = DummyLoss(loss_value)

    @property
    def loss(self):
        return self._loss


class MockModelConfig(PretrainedConfig):
    """Mock config that inherits from PretrainedConfig"""

    model_type = "llama"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = 32000
        self.hidden_size = 4096
        self.num_attention_heads = 32
        self.num_hidden_layers = 32


class MockLlamaForCausalLM(MagicMock):
    """Mock LLaMA model"""

    def __init__(self, config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config is None:
            config = MockModelConfig()
        self.config = config
        self.tokenizer = MagicMock()

        # Add mock parameters
        self._parameters = {
            "layer.weight": torch.nn.Parameter(torch.randn(10, 10)),
            "layer.bias": torch.nn.Parameter(torch.randn(10)),
        }

    def parameters(self):
        return self._parameters.values()


class MockModel(BaseMock):
    """Mock model with basic parameter operations and constant loss output for testing."""

    def __init__(self):
        super().__init__()
        self.params = {
            "layer1.weight": torch.nn.Parameter(torch.randn(10, 10)),
            "layer1.bias": torch.nn.Parameter(torch.randn(10)),
        }
        # Use a consistent DummyOutput that always returns a loss of 3.0.
        self.forward = MagicMock(return_value=DummyOutput())

    def named_parameters(self):
        return self.params.items()

    def parameters(self):
        return self.params.values()

    def state_dict(self):
        # Return a dictionary with clones of the parameters on the CPU
        return {
            name: param.detach().cpu().clone() for name, param in self.params.items()
        }

    def eval(self):
        """
        Simulates switching the model to evaluation mode.
        Returns self for chaining.
        """
        return self

    def train(self):
        """
        Simulates switching the model to training mode.
        Returns self for chaining.
        """
        return self

    def __call__(self, *args, **kwargs):
        """
        Allows the model to be callable, mimicking torch.nn.Module behavior.
        """
        return self.forward(*args, **kwargs)

    def zero_grad(self):
        """
        Simulates zeroing the gradients of all parameters.
        """
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def clone(self):
        # Instead of using deep copy, create a new instance and copy only essential attributes.
        new_instance = MockModel()
        new_instance.forward = self.forward
        # If there are additional attributes (e.g., parameters) needed by evaluate_peer, copy them here.
        if hasattr(self, "params"):
            # If params is a dict, perform a shallow copy.
            new_instance.params = (
                self.params.copy() if isinstance(self.params, dict) else self.params
            )
        return new_instance


class MockOptimizer(SGD):
    """Mock optimizer with basic operations"""

    def __init__(self, params, lr=0.01):
        super().__init__(params, lr=lr)
        self.step = MagicMock()
        self.zero_grad = MagicMock()


class MockScheduler(StepLR):
    """Mock learning rate scheduler"""

    def __init__(self, optimizer, step_size=10):
        super().__init__(optimizer, step_size=step_size)
        self.step = MagicMock()
        self.get_last_lr = MagicMock(return_value=[0.01])


class MockTransformer(BaseMock):
    """Mock transformer for gradient processing"""

    def __init__(self):
        super().__init__()
        self.shapes = {"layer1.weight": (10, 10), "layer1.bias": (10,)}
        self.totalks = {"layer1.weight": 50, "layer1.bias": 5}

    def encode(self, tensor):
        # Match actual transformer behavior
        return tensor.clone()

    def decode(self, tensor):
        return tensor.clone()


class MockCompressor(BaseMock):
    """Mock compressor for gradient compression"""

    def compress(self, tensor, topk):
        # More realistic compression simulation
        n = min(topk, tensor.numel())
        return (torch.arange(n), torch.ones(n) * 0.1, tensor.shape, n)

    def decompress(self, p, idxs, vals, xshape, totalk):
        return torch.ones_like(p) * 0.1

    def batch_decompress(self, p, idxs, vals, xshape, totalk):
        return torch.ones_like(p) * 0.1
