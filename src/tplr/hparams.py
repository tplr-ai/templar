# The MIT License (MIT)
# © 2024 templar.tech

import json
from types import SimpleNamespace
from transformers import AutoTokenizer, LlamaConfig

from .logging import logger

DEFAULT_HPARAMS = {
    # Run configuration
    "spec_version": 5,
    "project": "templar",
    
    # Model parameters
    "sequence_length": 1024,
    "pages_per_window": 2,
    "batch_size": 8,
    "learning_rate": 0.001,
    
    # Window and sync parameters
    "blocks_per_window": 2,
    "windows_per_sync": 100,
    "windows_per_weights": 10,
    
    # Optimization parameters
    "momentum_decay": 0.999,
    "topk_compression": 32,
    "target_chunk": 64,
    "scores_alpha": 0.001,
    
    # Model architecture (these should be in your hparams.json)
    "tokenizer_name": "huggyllama/llama-7b",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "intermediate_size": 11008,
    "num_key_value_heads": 32,
    "activation_function": "silu",
    "max_position_embeddings": 2048,
}

def create_namespace(hparams: dict) -> SimpleNamespace:
    """
    Create a SimpleNamespace from the hyperparameters and add model configuration.

    Args:
        hparams (dict): Hyperparameters dictionary.

    Returns:
        SimpleNamespace: Namespace containing hyperparameters and model configuration.
    """
    # Merge with defaults
    full_hparams = DEFAULT_HPARAMS.copy()
    full_hparams.update(hparams)
    
    hparams_ns = SimpleNamespace(**full_hparams)

    # Initialize tokenizer
    try:
        hparams_ns.tokenizer = AutoTokenizer.from_pretrained(
            hparams_ns.tokenizer_name, verbose=False, clean_up_tokenization_spaces=True
        )
        hparams_ns.tokenizer.pad_token = hparams_ns.tokenizer.eos_token
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    # Initialize model config
    try:
        hparams_ns.model_config = LlamaConfig(
            vocab_size=hparams_ns.tokenizer.vocab_size,
            hidden_size=hparams_ns.hidden_size,
            num_hidden_layers=hparams_ns.num_hidden_layers,
            num_attention_heads=hparams_ns.num_attention_heads,
            intermediate_size=hparams_ns.intermediate_size,
            num_key_value_heads=hparams_ns.num_key_value_heads,
            activation_function=hparams_ns.activation_function,
            max_position_embeddings=hparams_ns.max_position_embeddings,
        )
    except Exception as e:
        logger.error(f"Failed to create model config: {e}")
        raise

    return hparams_ns

def load_hparams(hparams_file: str = "hparams.json") -> SimpleNamespace:
    """
    Load hyperparameters from a JSON file.

    Args:
        hparams_file (str): Path to the hyperparameters JSON file.

    Returns:
        SimpleNamespace: A namespace containing the hyperparameters and model configuration.

    Example:
        hparams = load_hparams()
        print(hparams.hidden_size)
        print(hparams.model_config)
    """
    try:
        with open(hparams_file, "r") as f:
            hparams = json.load(f)
        return create_namespace(hparams)
    except FileNotFoundError:
        logger.warning(f"No {hparams_file} found, using default hyperparameters")
        return create_namespace({})
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {hparams_file}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading hyperparameters: {e}")
        raise 