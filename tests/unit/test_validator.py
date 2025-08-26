from types import SimpleNamespace

import pytest
import torch

import tplr
from neurons.validator import Validator


class MockCompressor:
    def _dequantize_values(self, v, qp):
        return v.float()

    def maybe_dequantize_values(self, vals, quant_params, device):
        if isinstance(vals, list):
            return [v.float() for v in vals]
        return vals.float()


@pytest.fixture
def validator():
    class MockValidator:
        def __init__(self):
            self.hparams = SimpleNamespace(gradient_clip_val=1.0)
            self.compressor = MockCompressor()
            self.model = torch.nn.Linear(2, 2)

        def compute_peer_val_norms(self, gather_result):
            return Validator.compute_peer_val_norms(self, gather_result)

    return MockValidator()


def test_compute_peer_val_norms_simple(validator):
    gather_result = SimpleNamespace(
        contribs={},
        state_dict=SimpleNamespace(
            weightvals=[torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
            biasvals=[torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0])],
        ),
    )
    val_norms = validator.compute_peer_val_norms(gather_result)

    expected_weight_norm = torch.median(
        torch.tensor(
            [torch.norm(torch.tensor([1.0, 2.0])), torch.norm(torch.tensor([3.0, 4.0]))]
        )
    )
    expected_bias_norm = torch.median(
        torch.tensor(
            [torch.norm(torch.tensor([5.0, 6.0])), torch.norm(torch.tensor([7.0, 8.0]))]
        )
    )

    assert torch.allclose(val_norms["weightvals"], expected_weight_norm)
    assert torch.allclose(val_norms["biasvals"], expected_bias_norm)


def test_compute_peer_val_norms_none_state_dict(validator):
    gather_result = SimpleNamespace(contribs={}, state_dict=None)
    with pytest.raises(
        ValueError, match="Must have gather_result.state_dict to compute norms"
    ):
        validator.compute_peer_val_norms(gather_result)


def test_compute_peer_val_norms_zero_peer_norm(validator):
    gather_result = SimpleNamespace(
        contribs={},
        state_dict=SimpleNamespace(
            weightvals=[torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])],
            biasvals=[torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0])],
        ),
    )
    val_norms = validator.compute_peer_val_norms(gather_result)

    expected_weight_norm = torch.median(
        torch.tensor(
            [torch.norm(torch.tensor([0.0, 0.0])), torch.norm(torch.tensor([0.0, 0.0]))]
        )
    )
    expected_bias_norm = torch.median(
        torch.tensor(
            [torch.norm(torch.tensor([5.0, 6.0])), torch.norm(torch.tensor([7.0, 8.0]))]
        )
    )

    assert torch.allclose(val_norms["weightvals"], expected_weight_norm)
    assert torch.allclose(val_norms["biasvals"], expected_bias_norm)


def test_compute_peer_val_norms_zero_avg_norm(validator):
    gather_result = SimpleNamespace(
        contribs={},
        state_dict=SimpleNamespace(
            weightvals=[torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])],
            biasvals=[torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])],
        ),
    )
    val_norms = validator.compute_peer_val_norms(gather_result)

    expected_weight_norm = torch.median(
        torch.tensor(
            [torch.norm(torch.tensor([0.0, 0.0])), torch.norm(torch.tensor([0.0, 0.0]))]
        )
    )
    expected_bias_norm = torch.median(
        torch.tensor(
            [torch.norm(torch.tensor([0.0, 0.0])), torch.norm(torch.tensor([0.0, 0.0]))]
        )
    )

    assert torch.allclose(val_norms["weightvals"], expected_weight_norm)
    assert torch.allclose(val_norms["biasvals"], expected_bias_norm)
