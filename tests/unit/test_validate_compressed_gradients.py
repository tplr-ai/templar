"""
Unit tests for validate_compressed_gradients.

Test Cases:

1. test_valid_input:
   - Create a state_dict with keys ending in "idxs" and "vals" containing proper torch.Tensor values.
   - totalks for each parameter is provided (e.g., {"param": 10}).
   - Provide allowed_topk such that the number of indices does not exceed the maximum.
   - Expect: (True, None).

2. test_missing_totalk:
   - Provide a state_dict with a parameter key (e.g., "param_idxs") that is missing in totalks.
   - Expect: (False, "Missing totalk for parameter param").

3. test_invalid_indexes_type:
   - Provide a state_dict where a key ending with "idxs" is not a torch.Tensor (e.g., a list).
   - Expect: (False, "Invalid indices for param_idxs: expected tensor, got <type>").

4. test_empty_indices:
   - Use an empty tensor for an "idxs" key.
   - Expect: (True, None) because an empty tensor is considered valid.

5. test_indices_out_of_bounds:
   - Create a state_dict with "param_idxs" containing indices less than 0 or greater than or equal to totalk.
   - For example, totalk = 10 but the tensor includes a value of 10 or -1.
   - Expect: (False, "Indices out of bounds for param_idxs: ...").

6. test_excessive_topk:
   - Set totalk = 100 and allowed_topk such that the maximum allowed count is 5 (5% of 100).
   - Provide a tensor with 6 indices.
   - Expect: (False, "Too many indices for param_idxs: got 6, max allowed is 5 (5% of 100)").

7. test_vals_contains_nan:
   - Create a state_dict with "param_vals" that contains a NaN value.
   - Expect: (False, "Values contain NaN or Inf for parameter param_vals").

8. test_vals_contains_inf:
   - Similar to test_vals_contains_nan but with an Inf value.
   - Expect: (False, "Values contain NaN or Inf for parameter param_vals").

9. test_valid_device_handling:
   - Run the function with a non-default device (e.g., "cuda" if available).
   - Ensure that all tensor.to(device) calls work without error, and valid input still returns (True, None).

The tests below are written as stubs with comments; you can implement assertions using your preferred
test framework (e.g. pytest).
"""

import torch
from tplr.neurons import validate_compressed_gradients


class TestValidateCompressedGradients:
    def test_valid_input(self):
        # totalks for parameter "param" is 10.
        # allowed_topk = 50 means max allowed indices = int(10 * 50/100) = 5.
        totalks = {"param": 10}
        allowed_topk = 50
        state_dict = {
            "param_idxs": torch.tensor([0, 1, 2]),  # valid indices within [0,9]
            "param_vals": torch.tensor([0.1, 0.2, 0.3]),
        }
        is_valid, err = validate_compressed_gradients(
            state_dict, totalks, allowed_topk, device="cpu"
        )
        assert is_valid, f"Expected valid input, but got error: {err}"

    def test_missing_totalk(self):
        # totalks is missing the key for "param".
        totalks = {}
        state_dict = {
            "param_idxs": torch.tensor([0, 1]),
            "param_vals": torch.tensor([0.1, 0.2]),
        }
        is_valid, err = validate_compressed_gradients(
            state_dict, totalks, allowed_topk=50, device="cpu"
        )
        assert not is_valid and "Missing totalk for parameter param" in err

    def test_invalid_indexes_type(self):
        # "param_idxs" is not a torch.Tensor but a list.
        totalks = {"param": 10}
        state_dict = {
            "param_idxs": [0, 1, 2],  # invalid type
            "param_vals": torch.tensor([0.1, 0.2, 0.3]),
        }
        is_valid, err = validate_compressed_gradients(
            state_dict, totalks, allowed_topk=50, device="cpu"
        )
        assert not is_valid and "Invalid indices for param_idxs" in err

    def test_empty_indices(self):
        # Empty tensor for "param_idxs" should be considered valid.
        totalks = {"param": 10}
        state_dict = {
            "param_idxs": torch.tensor([]),  # empty tensor
            "param_vals": torch.tensor([0.1, 0.2, 0.3]),
        }
        is_valid, err = validate_compressed_gradients(
            state_dict, totalks, allowed_topk=50, device="cpu"
        )
        assert is_valid, (
            f"Expected valid input with empty indices, but got error: {err}"
        )

    def test_indices_out_of_bounds(self):
        # Indices include a value (10) which is out-of-bounds (valid: 0-9 for totalk=10).
        totalks = {"param": 10}
        state_dict = {
            "param_idxs": torch.tensor([1, 10]),
            "param_vals": torch.tensor([0.1, 0.2]),
        }
        is_valid, err = validate_compressed_gradients(
            state_dict, totalks, allowed_topk=50, device="cpu"
        )
        assert not is_valid and "Indices out of bounds" in err

    def test_excessive_topk(self):
        # totalk = 100, allowed_topk = 5 means max allowed indices = int(100 * 5/100) = 5.
        # Here we provide 6 indices.
        totalks = {"param": 100}
        state_dict = {
            "param_idxs": torch.tensor([0, 1, 2, 3, 4, 5]),
            "param_vals": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        }
        is_valid, err = validate_compressed_gradients(
            state_dict, totalks, allowed_topk=5, device="cpu"
        )
        assert not is_valid and "Too many indices for param_idxs" in err

    def test_vals_contains_nan(self):
        # "param_vals" contains a NaN.
        totalks = {"param": 10}
        state_dict = {
            "param_idxs": torch.tensor([0, 1, 2]),
            "param_vals": torch.tensor([0.1, float("nan"), 0.3]),
        }
        is_valid, err = validate_compressed_gradients(
            state_dict, totalks, allowed_topk=50, device="cpu"
        )
        assert (
            not is_valid and "Values contain NaN or Inf for parameter param_vals" in err
        )

    def test_vals_contains_inf(self):
        # "param_vals" contains an Inf.
        totalks = {"param": 10}
        state_dict = {
            "param_idxs": torch.tensor([0, 1, 2]),
            "param_vals": torch.tensor([0.1, float("inf"), 0.3]),
        }
        is_valid, err = validate_compressed_gradients(
            state_dict, totalks, allowed_topk=50, device="cpu"
        )
        assert (
            not is_valid and "Values contain NaN or Inf for parameter param_vals" in err
        )

    def test_valid_device_handling(self):
        # Test using a non-default device.
        totalks = {"param": 10}
        state_dict = {
            "param_idxs": torch.tensor([0, 1, 2]),
            "param_vals": torch.tensor([0.1, 0.2, 0.3]),
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_valid, err = validate_compressed_gradients(
            state_dict, totalks, allowed_topk=50, device=device
        )
        assert is_valid, (
            f"Expected valid input on device {device}, but got error: {err}"
        )
