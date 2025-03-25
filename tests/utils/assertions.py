"""Common test assertions"""
import torch

def assert_tensor_equal(a, b, msg=None, rtol=1e-5, atol=1e-8):
    """Assert two tensors are equal within tolerance"""
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(f"{msg or ''}\nExpected:\n{a}\nGot:\n{b}")

def assert_gradient_valid(gradient, msg=None):
    """Assert gradient tensor is valid"""
    if not torch.isfinite(gradient).all():
        raise AssertionError(f"{msg or ''}\nGradient contains invalid values:\n{gradient}") 