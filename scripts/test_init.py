#!/usr/bin/env python3

# Script to verify model initialization is deterministic

import torch
import numpy as np
import random
from transformers import LlamaForCausalLM
from transformers import LlamaConfig
import hashlib


def get_model_hash(model):
    """Calculate a hash of all model parameters combined."""
    hasher = hashlib.md5()
    for name, param in model.named_parameters():
        # Convert tensor to bytes and update hash
        param_bytes = param.detach().cpu().numpy().tobytes()
        hasher.update(param_bytes)
    return hasher.hexdigest()


def compare_models(model1, model2):
    """Compare two models parameter by parameter and return if identical."""
    identical = True
    differing_params = []

    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        # Check if parameter names match
        if name1 != name2:
            print(f"Parameter name mismatch: {name1} vs {name2}")
            identical = False
            differing_params.append((name1, name2))
            continue

        # Check if parameters are identical
        if not torch.allclose(param1, param2):
            print(f"Parameter values different for {name1}")
            identical = False
            differing_params.append(name1)

            # Show differences in first non-matching parameter
            if len(differing_params) == 1:
                print(f"Sample difference in {name1}:")
                diff = (param1 - param2).abs()
                max_diff_idx = diff.argmax().item()
                flat_p1 = param1.flatten()
                flat_p2 = param2.flatten()
                print(f"  Model1: {flat_p1[max_diff_idx].item()}")
                print(f"  Model2: {flat_p2[max_diff_idx].item()}")
                print(f"  Max diff: {diff.max().item()}")

    return identical, differing_params


# Set common seeds
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Creating model configurations...")

# Use a smaller LLaMA model for testing purposes
config = LlamaConfig(
    vocab_size=32000,
    hidden_size=512,  # Smaller for faster testing
    intermediate_size=1024,
    num_hidden_layers=4,  # Fewer layers for faster testing
    num_attention_heads=8,
    max_position_embeddings=2048,
)

print("Initializing models...")

# Create 5 models with explicit seed reset before each
models_with_seed = []
for i in range(5):
    print(f"Creating model {i + 1} with seed {SEED}")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    models_with_seed.append(LlamaForCausalLM(config))

# Create 1 model without setting the seed again
print("Creating model 6 without resetting seed")
model_no_seed = LlamaForCausalLM(config)

# Calculate hashes for each model
hashes = [get_model_hash(model) for model in models_with_seed]
hash_no_seed = get_model_hash(model_no_seed)

print("\nHash values for the 5 models with the same seed:")
for i, hash_val in enumerate(hashes):
    print(f"Model {i + 1} hash: {hash_val}")
print(f"\nModel without seed reset hash: {hash_no_seed}")

print("\nComparing all models with Model 1:")
for i in range(1, 5):
    print(f"\nComparing Model 1 and Model {i + 1} (both with explicit seed {SEED}):")
    identical, params_diff = compare_models(models_with_seed[0], models_with_seed[i])
    if identical:
        print(f"✅ Models 1 and {i + 1} are IDENTICAL")
    else:
        print(f"❌ Models 1 and {i + 1} DIFFER in {len(params_diff)} parameters")

print("\nComparing Model 1 and Model without seed reset:")
identical, params_diff = compare_models(models_with_seed[0], model_no_seed)
if identical:
    print("✅ Models 1 and the model without seed reset are IDENTICAL")
else:
    print(
        f"❌ Models 1 and the model without seed reset DIFFER in {len(params_diff)} parameters"
    )

print("\nConclusion:")
all_seeded_identical = all(hash_val == hashes[0] for hash_val in hashes)
if all_seeded_identical:
    print("✅ All 5 models initialized with the same seed are IDENTICAL")
else:
    print("❌ Even with the same seed, the models have DIFFERENCES")

if hash_no_seed in hashes:
    print("✅ The model without seed reset MATCHES the seeded models")
else:
    print("❌ The model without seed reset is DIFFERENT from the seeded models")

print(
    "\nTo ensure deterministic initialization across nodes, use the same seed before EACH model instantiation."
)
