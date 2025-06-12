from transformers import LlamaForCausalLM

import tplr

# Maximum allowed number of parameters (number of parameters using the local
MAX_PARAMS = 2_064_480


def test_toy_model_creation():
    """Test that we can create a minimal working model configuration"""
    # Load default hparams
    hparams = tplr.load_hparams(use_local_run_hparams=True)

    # Create the model
    model = LlamaForCausalLM(hparams.model_config)

    # Count total parameters and assert it's under 3M
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params <= MAX_PARAMS, (
        f"Model has {total_params:,} parameters, which exceeds the {MAX_PARAMS:,} limit"
    )
