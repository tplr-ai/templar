# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Global imports
import json
import copy  # For deep copying the model args
from types import SimpleNamespace
from transformers import AutoTokenizer

# Local imports
from .logging import logger
from .models.llama3.model import TransformerModelArgs

# --- Define Model Flavor Configurations using TransformerModelArgs ---
MODEL_FLAVORS: dict[str, TransformerModelArgs] = {
    "debugmodel": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=8,
        n_kv_heads=8,
        rope_theta=10000.0,
        multiple_of=128,
    ),
    "debugmodel_flex_attn": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=8,
        n_kv_heads=8,
        rope_theta=10000.0,
        multiple_of=128,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000.0,
    ),
    "70B": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000.0,
    ),
    "405B": TransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000.0,
    ),
    # Mapping old names to new TT names for this specific config
    "custom_2k_16l_8h": TransformerModelArgs(
        dim=2048,  # from hidden_size
        n_layers=16,  # from num_hidden_layers
        n_heads=8,  # from num_attention_heads
        n_kv_heads=8,  # from num_key_value_heads
        # intermediate_size 8192. Calculate multiplier:
        # TT hidden_dim = int(2 * (4 * dim) / 3) = int(2 * (4 * 2048)/3) = int(5461.33) = 5461
        # TT multiple_of (default 256): 256 * ((5461 + 255) // 256) = 256 * (5716 // 256) = 256 * 22 = 5632
        # ffn_dim_multiplier = intermediate_size / hidden_dim = 8192 / 5632 = 1.4545... Let's use 1.45
        ffn_dim_multiplier=1.45,  # Calculated based on intermediate_size
        multiple_of=256,  # TT default, seems appropriate
        rope_theta=500000.0,  # Using Llama 3 default, adjust if needed
        norm_eps=1e-5,  # TT default
        # max_position_embeddings -> sequence_length -> model_args.max_seq_len
        # activation_function -> Ignored, TT uses SiLU
    ),
}

# --- Define keys that can override flavor defaults, using TT names ---
MODEL_OVERRIDE_KEYS = [
    "dim",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "multiple_of",
    "ffn_dim_multiplier",
    "norm_eps",
    "rope_theta",
    "depth_init",
    "use_flex_attn",
    "attn_mask_type",
]

# --- Updated DEFAULT_HPARAMS - Minimal model architecture params ---
DEFAULT_HPARAMS = {
    # Run configuration
    "spec_version": 5,
    "project": "templar",  # Default project name
    "model_flavor": "debugmodel",  # Default flavor
    "tokenizer_name": "meta-llama/Meta-Llama-3-8B",  # Default Tokenizer
    # Templar Training/Sequence parameters (sequence_length used by model)
    "sequence_length": 1024,
    "pages_per_window": 2,
    "batch_size": 8,
    "learning_rate": 0.001,
    # Templar Window and Sync parameters
    "blocks_per_window": 2,
    "windows_per_sync": 100,
    "windows_per_weights": 10,
    # Templar Optimization parameters
    "momentum_decay": 0.999,
    "topk_compression": 32,
    "target_chunk": 64,
    "scores_alpha": 0.001,  # Old score param, might need review
    "weight_decay": 0.0,  # Often 0.1 for AdamW
    # Templar Scheduler parameters
    "warmup_steps": 250,
    "alpha_f": 0.1,
    "t_max": 20000,
    "use_flex_attn": None,  # Use None to signal "use flavor default unless overridden"
    "attn_mask_type": None,
    # Loss compilation flag
    "compile_loss": False,
    # --- Validator specific hparams ---
    "validator_offset": 1,
    "uids_per_window": 1,
    "binary_score_ma_alpha": 0.05,
    "eval_lr_factor": 0.1,
    "validator_sample_rate": 0.1,
    "sync_max_steps_behind": 5.0,
    "openskill_mu": 10.0,
    "openskill_sigma": 10.0 / 3.0,
    "openskill_beta": (10.0 / 3.0) / 2.0,
    "openskill_tau": (10.0 / 3.0) / 100.0,
    "peer_replacement_frequency": 10,
    "peer_list_window_margin": 2,
    "peers_to_replace": 1,
    "minimum_peers": 5,
    "max_topk_peers": 8,
    "reset_inactivity_windows": 50,
    "gradient_score_ma_alpha": 0.6,
    "final_score_ma_alpha": 0.75,
    "moving_average_window": 5,
    "power_normalisation": 2.0,
    # --- Comms specific hparams ---
    "active_check_interval": 300,
    "recent_windows": 3,
    "time_window_delta_seconds": 120,
    "checkpoint_frequency": 100,
    "catch_up_threshold": 15,
    "catch_up_batch_size": 5,
    "catch_up_timeout": 300,
    "topk_peers": 20,
    "checkpoint_init_version": None,
}


def create_namespace(hparams: dict) -> SimpleNamespace:
    """
    Create a SimpleNamespace from the hyperparameters, loading the base model
    configuration directly from MODEL_FLAVORS based on model_flavor,
    then applying overrides (using TT names) and runtime updates.

    Args:
        hparams (dict): Hyperparameters dictionary loaded from JSON.

    Returns:
        SimpleNamespace: Namespace containing hyperparameters and model configuration.
    """
    # Merge with defaults to get model_flavor etc.
    full_hparams = DEFAULT_HPARAMS.copy()
    full_hparams.update(hparams)

    hparams_ns = SimpleNamespace(**full_hparams)

    # --- Get Base Model Args based on Flavor ---
    model_flavor = getattr(hparams_ns, "model_flavor", "debugmodel")
    if model_flavor not in MODEL_FLAVORS:
        raise ValueError(
            f"Unknown model_flavor '{model_flavor}'. Available flavors: {list(MODEL_FLAVORS.keys())}"
        )

    # Get a *copy* of the base model args to modify
    model_args_instance = copy.deepcopy(MODEL_FLAVORS[model_flavor])
    logger.info(f"Loading base TransformerModelArgs for flavor: '{model_flavor}'")

    # --- Load Tokenizer (Needed for vocab_size, eos_id) ---
    try:
        trust_remote_code = False  # Adjust if needed
        tokenizer_name = getattr(
            hparams_ns, "tokenizer_name", DEFAULT_HPARAMS["tokenizer_name"]
        )
        hparams_ns.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            verbose=False,
            clean_up_tokenization_spaces=True,
            trust_remote_code=trust_remote_code,
        )
        if hparams_ns.tokenizer.pad_token is None:
            hparams_ns.tokenizer.pad_token = hparams_ns.tokenizer.eos_token
            logger.info(
                f"Set pad_token to eos_token ({hparams_ns.tokenizer.eos_token_id})"
            )

        hparams_ns.eos_id = hparams_ns.tokenizer.eos_token_id
        if hparams_ns.eos_id is None:
            raise ValueError(
                f"Tokenizer {tokenizer_name} does not have an EOS token defined."
            )

    except Exception as e:
        logger.error(f"Failed to load tokenizer '{tokenizer_name}': {e}")
        raise

    # --- Apply Runtime Updates to Model Args ---
    model_args_instance.vocab_size = hparams_ns.tokenizer.vocab_size
    model_args_instance.max_seq_len = hparams_ns.sequence_length
    model_args_instance.eos_id = hparams_ns.eos_id

    # --- Apply Overrides from hparams.json (using TT names) ---
    overrides_applied = {}
    for key in MODEL_OVERRIDE_KEYS:
        # Check if the key exists in the *original* JSON hparams
        if key in hparams:
            override_value = hparams[key]
            if hasattr(model_args_instance, key):
                # Only override if the value in JSON is different from the flavor's default
                # This prevents logging overrides when the JSON just restates the default
                current_flavor_value = getattr(model_args_instance, key)
                if current_flavor_value != override_value:
                    setattr(model_args_instance, key, override_value)
                    overrides_applied[key] = override_value
            else:
                logger.warning(
                    f"Override key '{key}' from hparams.json not found in TransformerModelArgs."
                )

    # Apply flags like use_flex_attn if explicitly set in hparams JSON
    # (overriding both DEFAULT_HPARAMS and the flavor's setting)
    for flag_key in ["use_flex_attn", "attn_mask_type"]:
        if flag_key in hparams and hparams[flag_key] is not None:
            if hasattr(model_args_instance, flag_key):
                current_flavor_value = getattr(model_args_instance, flag_key)
                if current_flavor_value != hparams[flag_key]:
                    setattr(model_args_instance, flag_key, hparams[flag_key])
                    overrides_applied[flag_key] = hparams[flag_key]
            else:
                logger.warning(
                    f"Override flag key '{flag_key}' from hparams.json not found in TransformerModelArgs."
                )

    if overrides_applied:
        logger.info(
            f"Applied overrides to flavor '{model_flavor}': {overrides_applied}"
        )

    # --- Store Final Model Args ---
    hparams_ns.model_args = model_args_instance
    logger.info(f"Final TorchTitan Model Args: {hparams_ns.model_args}")

    # Cleanup old attribute if present
    if hasattr(hparams_ns, "model_config"):
        delattr(hparams_ns, "model_config")

    # Add a check for weight_decay compatibility if needed (AdamW uses it, SGD doesn't by default)
    # Example:
    # if hparams_ns.optimizer_name == "SGD" and getattr(hparams_ns, 'weight_decay', 0.0) > 0.0:
    #    logger.warning("weight_decay > 0 specified but default SGD optimizer does not use it directly.")

    return hparams_ns


# load_hparams function remains the same
def load_hparams(
    hparams_file: str = "hparams.json", use_local_run_hparams: bool = False
) -> SimpleNamespace:
    """
    Load hyperparameters from a JSON file, using model_flavor.

    Args:
        hparams_file (str): Path to the hyperparameters JSON file.
        use_local_run_hparams (bool): Whether to load and merge hparams-local-run.json.

    Returns:
        SimpleNamespace: A namespace containing the hyperparameters and model configuration.
    """
    try:
        with open(hparams_file, "r") as f:
            hparams = json.load(f)
        logger.info(f"Loaded base hyperparameters from {hparams_file}")

        if use_local_run_hparams:
            try:
                with open("hparams-local-run.json", "r") as f:
                    hparams_local_run = json.load(f)
                # Merge local overrides INTO the base hparams
                hparams.update(hparams_local_run)
                logger.info(
                    f"Using updated hparams for a local run from hparams-local-run.json. Effective overrides: {hparams_local_run}"
                )
            except FileNotFoundError:
                logger.warning(
                    "Local run specified (--local) but hparams-local-run.json not found. Using base hparams."
                )
            except json.JSONDecodeError as e:
                logger.error(
                    f"Invalid JSON in hparams-local-run.json: {e}. Using base hparams."
                )
                raise
        else:
            logger.info("Using hparams for a normal run (not local)")

        # create_namespace now handles the flavor logic using the merged hparams
        return create_namespace(hparams)
    except FileNotFoundError:
        logger.warning(
            f"Base config file {hparams_file} not found, using default hyperparameters"
        )
        return create_namespace({})
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {hparams_file}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading hyperparameters: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise
