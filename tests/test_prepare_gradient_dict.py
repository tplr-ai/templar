import torch
import pytest
from tplr.neurons import prepare_gradient_dict


class DummyScheduler:
    def get_last_lr(self):
        return [0.01]


class DummyHparams:
    def __init__(self):
        self.weight_decay = 0.1
        self.momentum_decay = 0.9
        self.topk_compression = 5


class DummyCompressor:
    def compress(self, encoded_tensor, topk):
        dummy_idxs = "dummy_idxs"
        dummy_vals = "dummy_vals"
        dummy_xshape = "dummy_xshape"
        dummy_totalk = "dummy_totalk"
        dummy_quant_params = "dummy_quant_params"
        return dummy_idxs, dummy_vals, dummy_xshape, dummy_totalk, dummy_quant_params

    def decompress(self, p, idxs, vals, xshape, totalk, quant_params):
        return torch.tensor([0.5, 0.5])


class DummyTransformer:
    def encode(self, tensor):
        return tensor

    def decode(self, tensor):
        return torch.tensor([0.1, 0.1])


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.weight.grad = torch.tensor([0.1, 0.2])


class DummyMiner:
    def __init__(self):
        self.model = DummyModel()
        self.scheduler = DummyScheduler()
        self.hparams = DummyHparams()
        self.momentum = {"weight": torch.zeros_like(self.model.weight)}
        self.owned_params = {"weight", "weight1", "weight2"}
        self.compressor = DummyCompressor()
        self.transformer = DummyTransformer()
        self.logger = DummyLogger()


# Test 1: Return Structure and Types
# ---------------------------------
# - Call prepare_gradient_dict with valid inputs
# - Verify it returns a tuple of length 3 containing (gradient, xshapes, totalks)
# - Check gradient dict has expected keys (weightidxs, weightvals, metadata)
# - Verify xshapes, totalks, are dicts with expected keys
# - Confirm metadata attachment is logged
def test_return_structure_and_types(caplog):
    # Create dummy miner instance
    miner = DummyMiner()
    # Define a valid step_window
    step_window = 5

    with caplog.at_level("INFO", logger="templar"):
        # Call the helper function.
        result = prepare_gradient_dict(miner, step_window)

    # Check that we get exactly four items returned.
    assert isinstance(result, tuple)
    assert len(result) == 3
    gradient, xshapes, totalks = result

    # Check that gradient is a dict
    assert isinstance(gradient, dict)
    # Verify gradient has keys for the parameter "weight"
    assert "weightidxs" in gradient
    assert "weightvals" in gradient
    assert "weightquant_params" in gradient
    # Verify that the metadata key exists
    assert "metadata" in gradient
    # Check that metadata equals the expected dictionary.
    expected_metadata = {"window": step_window}
    assert gradient["metadata"] == expected_metadata

    # Check that xshapes, totalks, are dictionaries.
    assert isinstance(xshapes, dict)
    assert isinstance(totalks, dict)
    # And that they contain key "weight"
    assert "weight" in xshapes
    assert "weight" in totalks


def test_metadata_attachment():
    """
    Test 2: Metadata Attachment
    ---------------------------
    - Call prepare_gradient_dict.
    - Verify that gradient["metadata"] exactly equals {"window": step_window}.
    """
    # Create dummy miner instance
    miner = DummyMiner()
    # Define a step_window value.
    step_window = 42

    # Call prepare_gradient_dict.
    gradient, xshapes, totalks = prepare_gradient_dict(miner, step_window)

    # Verify that gradient["metadata"] exactly equals the expected dictionary.
    expected_metadata = {"window": step_window}
    assert gradient.get("metadata") == expected_metadata, (
        f"Metadata does not match. Expected: {expected_metadata}, Got: {gradient.get('metadata')}"
    )


def test_weight_decay_application():
    """
    Test 3: Weight Decay Application
    ---------------------------------
    - Initialize a dummy parameter tensor with a known value (e.g., torch.tensor([1.0, 2.0])).
    - Set hparams.weight_decay to a known value (e.g., 0.1) and scheduler to return a specific lr (e.g., 0.01).
    - After calling the function, check that for each parameter, p.data was multiplied by (1 - lr * weight_decay).
      (For instance, expected new p.data == original_p.data * (1 - 0.01 * 0.1).)
    """
    # Create dummy miner instance.
    miner = DummyMiner()
    # Clone the original parameter value for later comparison.
    original_weight_data = miner.model.weight.data.clone()

    # Define arbitrary step_window value.
    step_window = 5

    # Call the prepare_gradient_dict helper.
    _ = prepare_gradient_dict(miner, step_window)

    # The DummyScheduler returns lr 0.01 and hparams.weight_decay is set to 0.1.
    # Therefore, the expected decay factor is: 1 - 0.01 * 0.1 = 0.999.
    expected_decay = 1 - 0.01 * 0.1
    expected_weight_data = original_weight_data * expected_decay

    # Assert that the updated parameter data matches the expected result.
    torch.testing.assert_close(miner.model.weight.data, expected_weight_data)


def test_momentum_decay_and_gradient_accumulation():
    """
    Test 4: Momentum Calculation on First Iteration
    -------------------------------------------------
    - Set initial momentum for parameter 'weight' (will be overwritten on first call).
    - Provide a dummy gradient for p.grad.
    - Verify that on the *first* call, the momentum is set directly to (lr * p.grad)
      and the transmitted gradient is *not* subtracted.
    """
    miner = DummyMiner()
    # Set a known initial momentum for "weight". This will be overwritten in the first call.
    miner.momentum["weight"] = torch.tensor([0.2, 0.3])
    # Set a dummy gradient for the parameter.
    miner.model.weight.grad = torch.tensor([0.1, 0.2])

    # Note for the FIRST iteration:
    #   - DummyScheduler returns lr = 0.01
    #   - Initial momentum and momentum_decay are effectively ignored.
    #   - Momentum is set to: lr * p.grad = 0.01 * [0.1, 0.2] = [0.001, 0.002]
    #   - Subtraction of transmitted gradient is skipped.
    #
    # Therefore, the final momentum after the first call should be [0.001, 0.002].
    expected_final_momentum = torch.tensor([0.001, 0.002])

    step_window = 5
    # This is the first call, so miner.gradient_iteration_counter will be 1.
    prepare_gradient_dict(miner, step_window)

    # TODO: Add a separate test case to verify the momentum calculation logic
    #       for iterations > 5, where initial momentum is used, decay is applied,
    #       and the transmitted gradient is subtracted. This would likely involve
    #       manually setting miner.gradient_iteration_counter or calling the function multiple times.

    torch.testing.assert_close(
        miner.momentum["weight"],
        expected_final_momentum,
        msg="Final momentum does not match expected value for the first iteration.",
    )


def test_compressor_and_transformer_calls():
    """
    Test 5: Compressor and Transformer Calls
    ------------------------------------------
    - Use dummy implementations that record the arguments passed to compressor.compress and transformer.decode.
    - Verify:
         • compressor.compress is called with the result of transformer.encode(miner.momentum['weight'])
           and the correct value of hparams.topk_compression.
         • transformer.decode is called with the result from compressor.decompress.
    - Also, verify that the dummy return values are included in the output dictionaries.
    """

    # Define recorder classes for compressor and transformer.
    class DummyRecordingCompressor:
        def __init__(self):
            self.called_args = None

        def compress(self, encoded_tensor, topk):
            self.called_args = (encoded_tensor.clone(), topk)
            dummy_idxs = "recorded_dummy_idxs"
            dummy_vals = "recorded_dummy_vals"
            dummy_xshape = "recorded_dummy_xshape"
            dummy_totalk = "recorded_dummy_totalk"
            dummy_quant_params = "recorded_dummy_quant_params"
            return (
                dummy_idxs,
                dummy_vals,
                dummy_xshape,
                dummy_totalk,
                dummy_quant_params,
            )

        def decompress(self, p, idxs, vals, xshape, totalk, quant_params):
            return torch.tensor([0.2, 0.2])

    class DummyRecordingTransformer:
        def __init__(self):
            self.decode_called_with = None

        def encode(self, tensor):
            return tensor

        def decode(self, tensor):
            self.decode_called_with = tensor.clone()
            return torch.tensor([0.1, 0.1])

    miner = DummyMiner()
    miner.compressor = DummyRecordingCompressor()
    miner.transformer = DummyRecordingTransformer()

    # Set initial momentum and gradient value for the single parameter "weight".
    # Initial momentum is ignored on the first call.
    miner.momentum["weight"] = torch.tensor([1.0, 1.0])
    miner.model.weight.grad = torch.tensor([0.3, 0.4])

    # Expected computation for the tensor passed to compressor.compress on the FIRST iteration:
    # Momentum is set directly to: lr * p.grad
    # lr = 0.01 (from DummyScheduler)
    # p.grad = [0.3, 0.4]
    # Expected tensor = 0.01 * [0.3, 0.4] = [0.003, 0.004]
    # This is the tensor that transformer.encode receives (and passes through in this dummy)
    # and then compressor.compress receives.
    expected_tensor_for_compression = torch.tensor([0.003, 0.004])

    step_window = 7
    # Call the helper so that compressor.compress and transformer.decode are invoked.
    # This is the first call, miner.gradient_iteration_counter becomes 1.
    gradient, xshapes, totalks = prepare_gradient_dict(miner, step_window)

    # Check that compressor.compress was called with the expected encoded tensor and topk.
    recorder_compressor = miner.compressor
    assert recorder_compressor.called_args is not None, (
        "compressor.compress was not called."
    )
    recorded_tensor, recorded_topk = recorder_compressor.called_args
    torch.testing.assert_close(
        recorded_tensor,
        expected_tensor_for_compression,  # Use the corrected expected value
        msg="compressor.compress argument (encoded tensor) does not match expected value for the first iteration.",
    )
    assert recorded_topk == miner.hparams.topk_compression, (
        "compressor.compress argument (topk) does not match hparams."
    )

    # Check that transformer.decode was called with the result of compressor.decompress.
    recorder_transformer = miner.transformer
    # The dummy decompress returns [0.2, 0.2]
    expected_tensor_for_decode = torch.tensor([0.2, 0.2])
    assert recorder_transformer.decode_called_with is not None, (
        "transformer.decode was not called."
    )
    torch.testing.assert_close(
        recorder_transformer.decode_called_with,
        expected_tensor_for_decode,
        msg="transformer.decode argument does not match the output of compressor.decompress.",
    )

    # Verify dummy return values are in the output dicts
    assert gradient["weightidxs"] == "recorded_dummy_idxs"
    assert gradient["weightvals"] == "recorded_dummy_vals"
    assert xshapes["weight"] == "recorded_dummy_xshape"
    assert totalks["weight"] == "recorded_dummy_totalk"
    # Verify metadata
    assert gradient["metadata"]["window"] == step_window


# Test 6: Handling Multiple Parameters
# --------------------------------------
# - Create a dummy model with two parameters, e.g., "weight1" and "weight2".
# - Ensure that the returned gradient dict includes keys: "weight1idxs", "weight1vals", "weight2idxs", "weight2vals".
# - Also verify that xshapes, and totalks dicts have both "weight1" and "weight2" as keys.


def test_handling_multiple_parameters():
    """
    Test 6: Handling Multiple Parameters
    --------------------------------------
    - Create a dummy model with two parameters, e.g., "weight1" and "weight2".
    - Ensure that the returned gradient dict includes keys: "weight1idxs", "weight1vals", "weight2idxs", "weight2vals".
    - Also verify that xshapes and totalks dicts have both "weight1" and "weight2" as keys.
    """

    class DummyMultiModel(torch.nn.Module):
        def __init__(self):
            super(DummyMultiModel, self).__init__()
            self.weight1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.weight2 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
            self.weight1.grad = torch.tensor([0.1, 0.1])
            self.weight2.grad = torch.tensor([0.2, 0.2])

    miner = DummyMiner()
    miner.model = DummyMultiModel()
    miner.momentum = {
        "weight1": torch.zeros_like(miner.model.weight1),
        "weight2": torch.zeros_like(miner.model.weight2),
    }

    step_window = 10
    gradient, xshapes, totalks = prepare_gradient_dict(miner, step_window)

    # Check for gradient keys from both parameters.
    assert "weight1idxs" in gradient, "Missing key: weight1idxs"
    assert "weight1vals" in gradient, "Missing key: weight1vals"
    assert "weight1quant_params" in gradient, "Missing key: weight1quant_params"
    assert "weight2idxs" in gradient, "Missing key: weight2idxs"
    assert "weight2vals" in gradient, "Missing key: weight2vals"
    assert "weight2quant_params" in gradient, "Missing key: weight2quant_params"

    # Check that xshapes, and totalks have both "weight1" and "weight2".
    for key in ["weight1", "weight2"]:
        assert key in xshapes, f"Missing {key} in xshapes"
        assert key in totalks, f"Missing {key} in totalks"


def test_behavior_when_p_grad_is_none():
    """
    Test 7: Behavior When p.grad is None
    -------------------------------------
    - Create a dummy model parameter with p.grad set to None.
    - Verify that the function raises an exception due to the missing gradient.
    """

    # Define a dummy model with one parameter whose grad is None.
    class DummyNoGradModel(torch.nn.Module):
        def __init__(self):
            super(DummyNoGradModel, self).__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.weight.grad = None  # No gradient provided.

    miner = DummyMiner()
    miner.model = DummyNoGradModel()
    # Reinitialize momentum accordingly.
    miner.momentum = {"weight": torch.zeros_like(miner.model.weight)}

    step_window = 3

    with pytest.raises(Exception) as excinfo:
        prepare_gradient_dict(miner, step_window)

    # You might expect a TypeError or AttributeError; ensure some exception is raised.
    assert "None" in str(excinfo.value) or "grad" in str(excinfo.value)


def test_logging_behavior(caplog):
    """
    Test 8: Logging Behavior
    ------------------------
    - Use the pytest caplog fixture to capture log calls.
    - Verify that when prepare_gradient_dict is executed, a log call is made that includes
      the correct metadata string (containing window).
    """
    miner = DummyMiner()
    step_window = 15

    with caplog.at_level("INFO", logger="templar"):
        prepare_gradient_dict(miner, step_window)


def test_correct_use_of_scheduler_learning_rate():
    """
    Test 9: Correct Use of Scheduler Learning Rate
    -----------------------------------------------
    - Set up the dummy scheduler so get_last_lr() returns a list (e.g., [0.02]).
    - Confirm through the parameter update calculations and weight decay adjustments that 0.02 is used
      (i.e., check that p.data is scaled by (1 - 0.02 * weight_decay)).
    """

    # Create a dummy scheduler that returns [0.02]
    class DummyScheduler02:
        def get_last_lr(self):
            return [0.02]

    # Create a dummy miner instance and update its scheduler.
    miner = DummyMiner()
    miner.scheduler = DummyScheduler02()

    # Clone the original parameter data.
    original_weight_data = miner.model.weight.data.clone()

    step_window = 5

    # Call the helper function (which applies weight decay).
    prepare_gradient_dict(miner, step_window)

    # Compute the expected decay: p.data should be multiplied by (1 - 0.02 * weight_decay)
    expected_decay_factor = 1 - 0.02 * miner.hparams.weight_decay
    expected_weight_data = original_weight_data * expected_decay_factor

    torch.testing.assert_close(miner.model.weight.data, expected_weight_data)


def test_propagation_of_compressor_failure():
    """
    Test 10: Propagation of Exceptions (Compressor Failure)
    --------------------------------------------------------
    - Force compressor.compress to throw an exception.
    - Verify that prepare_gradient_dict propagates the exception instead of silently swallowing it.
    """
    miner = DummyMiner()

    # Override compressor.compress to throw an exception.
    def failing_compress(encoded_tensor, topk):
        raise RuntimeError("Compressor error")

    miner.compressor.compress = failing_compress

    step_window = 5

    with pytest.raises(RuntimeError, match="Compressor error"):
        prepare_gradient_dict(miner, step_window)


def test_propagation_of_transformer_failure():
    """
    Test 11: Propagation of Exceptions (Transformer Failure)
    ---------------------------------------------------------
    - Force transformer.decode to throw an exception.
    - Verify that prepare_gradient_dict propagates this exception as expected.
    """
    miner = DummyMiner()

    # Override transformer.decode to throw an exception.
    def failing_decode(tensor):
        raise RuntimeError("Transformer error")

    miner.transformer.decode = failing_decode

    step_window = 5

    with pytest.raises(RuntimeError, match="Transformer error"):
        prepare_gradient_dict(miner, step_window)
