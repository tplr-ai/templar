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

"""TorchTitan Model Factory

Unified model creation and initialization for TorchTitan models across
evaluator, validator, and miner components.
"""

from types import SimpleNamespace
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torchtitan.config import (
    ActivationCheckpoint,
    Float8,
    JobConfig,
    Model,
    Parallelism,
    Training,
)
from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3 import (
    Llama3StateDictAdapter,
    TransformerModelArgs,
    llama3_configs,
)
from torchtitan.models.llama3 import Transformer as TitanLlama
from torchtitan.models.llama3.infra.parallelize import parallelize_llama
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

import tplr


def get_titan_model_args(hparams: SimpleNamespace) -> TransformerModelArgs:
    """Get TorchTitan TransformerModelArgs from predefined configs or custom hparams.

    Args:
        hparams: Hyperparameters object containing model configuration

    Returns:
        TransformerModelArgs configured for TorchTitan

    Raises:
        ValueError: If model_size is not valid
    """
    model_size = hparams.model_size
    sequence_length = hparams.sequence_length
    vocab_size = (
        hparams.tokenizer.vocab_size
        if hasattr(hparams, "tokenizer")
        else hparams.model_config.vocab_size
    )

    if model_size in llama3_configs:
        tplr.logger.info(f"Using predefined TorchTitan config for {model_size}")
        # Create a copy of the config to avoid modifying the original
        args = TransformerModelArgs(**vars(llama3_configs[model_size]))
        # Update max_seq_len with our sequence length
        args.max_seq_len = sequence_length
        # Update vocab_size if provided (from tokenizer)
        args.vocab_size = vocab_size
        tplr.logger.info(f"Using vocab_size: {vocab_size}")
        return args

    # Fall back to creating custom config from hparams for non-standard sizes (150M, 1B, etc.)
    tplr.logger.info(f"Creating custom TorchTitan config for {model_size} from hparams")

    # Create custom TransformerModelArgs from hparams
    args = TransformerModelArgs(
        dim=hparams.model_config.hidden_size,
        n_layers=hparams.model_config.num_hidden_layers,
        n_heads=hparams.model_config.num_attention_heads,
        n_kv_heads=getattr(
            hparams.model_config,
            "num_key_value_heads",
            hparams.model_config.num_attention_heads,
        ),
        vocab_size=vocab_size,
        ffn_dim_multiplier=None,  # Will calculate from intermediate_size
        multiple_of=256,
        norm_eps=getattr(hparams.model_config, "rms_norm_eps", 1e-5),
        rope_theta=getattr(hparams.model_config, "rope_theta", 10000.0),
        max_seq_len=sequence_length,
    )

    # Calculate ffn_dim_multiplier to match the intermediate_size from hparams
    # The formula in TorchTitan is: ffn_dim = int(8 * dim / 3 * ffn_dim_multiplier)
    # We need to reverse engineer ffn_dim_multiplier from intermediate_size
    target_intermediate_size = hparams.model_config.intermediate_size
    base_ffn = int(8 * args.dim / 3)
    args.ffn_dim_multiplier = target_intermediate_size / base_ffn

    tplr.logger.info(
        f"Custom config: dim={args.dim}, n_layers={args.n_layers}, "
        f"n_heads={args.n_heads}, n_kv_heads={args.n_kv_heads}, "
        f"intermediate_size={target_intermediate_size}"
    )

    return args


def create_job_config(
    hparams: SimpleNamespace, role: Literal["miner", "validator", "evaluator"] = "miner"
) -> JobConfig:
    """Create JobConfig from hparams with role-specific defaults.

    Args:
        hparams: Hyperparameters object containing torchtitan config
        role: Component role (miner, validator, or evaluator)

    Returns:
        JobConfig configured for the specific role
    """
    tt = getattr(hparams, "torchtitan", SimpleNamespace())

    # Role-specific defaults
    mixed_precision_param_default: Literal["bfloat16", "float32"]
    mixed_precision_reduce_default: Literal["float32"] = "float32"  # Only valid value
    compile_default = getattr(tt, "compile", False)

    if role == "evaluator":
        # Evaluator uses minimal settings for inference
        mixed_precision_param_default = "float32"
        enable_cpu_offload_default = False
    elif role == "validator":
        # Validator uses conservative settings
        # Ensure we only get valid values
        mp_param = getattr(tt, "mixed_precision_param", "float32")
        mixed_precision_param_default = (
            "bfloat16" if mp_param == "bfloat16" else "float32"
        )
        enable_cpu_offload_default = False
    else:  # miner
        # Miner uses full settings from config
        # Ensure we only get valid values
        mp_param = getattr(tt, "mixed_precision_param", "float32")
        mixed_precision_param_default = (
            "bfloat16" if mp_param == "bfloat16" else "float32"
        )
        enable_cpu_offload_default = getattr(tt, "enable_cpu_offload", False)

    # Build Training config
    training = Training(
        seq_len=hparams.sequence_length,
        compile=compile_default,
        enable_cpu_offload=enable_cpu_offload_default,
        mixed_precision_param=mixed_precision_param_default,
        mixed_precision_reduce=mixed_precision_reduce_default,
    )

    # Build Parallelism config
    parallelism = Parallelism(
        enable_async_tensor_parallel=getattr(tt, "enable_async_tensor_parallel", False),
        disable_loss_parallel=getattr(tt, "disable_loss_parallel", True),
        fsdp_reshard_after_forward=getattr(tt, "fsdp_reshard_after_forward", "default"),
        enable_compiled_autograd=getattr(tt, "enable_compiled_autograd", False),
    )

    # Build ActivationCheckpoint config
    activation_checkpoint = ActivationCheckpoint(
        mode=tt.activation_checkpoint.get("mode", "selective")
        if hasattr(tt, "activation_checkpoint")
        and isinstance(tt.activation_checkpoint, dict)
        else "selective",
        selective_ac_option=tt.activation_checkpoint.get("option", "op")
        if hasattr(tt, "activation_checkpoint")
        and isinstance(tt.activation_checkpoint, dict)
        else "op",
    )

    return JobConfig(
        training=training,
        parallelism=parallelism,
        model=Model(converters=getattr(tt, "converters", [])),
        float8=Float8(recipe_name=getattr(tt, "float8_recipe_name", None)),
        activation_checkpoint=activation_checkpoint,
    )


def create_parallel_dims(
    world_size: int,
    hparams: SimpleNamespace,
    role: Literal["miner", "validator", "evaluator"] = "miner",
) -> ParallelDims:
    """Create ParallelDims based on role and configuration.

    Args:
        world_size: Number of distributed processes
        hparams: Hyperparameters object containing torchtitan config
        role: Component role (miner, validator, or evaluator)

    Returns:
        ParallelDims configured for the specific role

    Raises:
        ValueError: If parallelization parameters are invalid
    """
    if role == "evaluator":
        # Evaluator: support both single and multi-GPU configurations
        dp_shard = min(4, world_size)  # Use up to 4 GPUs for TP
        if world_size % dp_shard != 0:
            raise ValueError(
                f"World size ({world_size}) must be divisible by "
                f"dp_shard degree ({dp_shard})"
            )
        return ParallelDims(
            dp_replicate=world_size // dp_shard,
            dp_shard=dp_shard,
            tp=1,
            pp=1,
            cp=1,
            ep=1,
            world_size=world_size,
        )
    elif role == "validator":
        # Validator: pipeline parallelism with data parallel replication
        dp_shard = 4
        if world_size % dp_shard != 0:
            raise ValueError(
                f"World size ({world_size}) must be divisible by "
                f"dp_shard degree ({dp_shard})"
            )
        return ParallelDims(
            dp_replicate=world_size // dp_shard,
            dp_shard=dp_shard,
            tp=1,
            pp=1,
            cp=1,
            ep=1,
            world_size=world_size,
        )
    else:  # miner
        # Miner: flexible configuration from hparams
        tt = getattr(hparams, "torchtitan", SimpleNamespace())

        tp_degree = int(getattr(tt, "tp_degree", 1))
        pp_degree = int(getattr(tt, "pp_degree", 1))
        cp_degree = int(getattr(tt, "cp_degree", 1))
        dp_replicate = int(getattr(tt, "dp_replicate", 1))
        dp_shard = int(getattr(tt, "dp_shard", 1))

        # Validation
        if dp_replicate > 1 and dp_shard > 1:
            raise ValueError(
                "Specify either torchtitan.dp_replicate or torchtitan.dp_shard, "
                "but not both."
            )

        if dp_replicate > 1 and (tp_degree > 1 or pp_degree > 1 or cp_degree > 1):
            raise ValueError("dp_replicate can only be used when tp/pp/cp are all 1.")

        dp_replicate = int(dp_replicate or 1)
        dp_shard = int(dp_shard or 1)

        if world_size % (dp_replicate * dp_shard) != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by "
                f"dp_replicate × dp_shard ({dp_replicate}×{dp_shard})."
            )

        if world_size % tp_degree != 0:
            raise ValueError(
                f"World size ({world_size}) must be divisible by "
                f"tensor-parallel degree ({tp_degree})"
            )

        return ParallelDims(
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
            tp=tp_degree,
            pp=pp_degree,
            cp=cp_degree,
            ep=1,
            world_size=world_size,
        )


def initialize_torchtitan_model(
    hparams: SimpleNamespace,
    role: Literal["miner", "validator", "evaluator"] = "miner",
    device: str = "cuda",
    world_size: int = 1,
) -> nn.Module:
    """Complete model initialization pipeline for TorchTitan models.

    Args:
        hparams: Hyperparameters object from tplr.load_hparams()
        role: Component role (miner, validator, or evaluator)
        device: Device to initialize model on
        world_size: Number of distributed processes

    Returns:
        Initialized and parallelized TorchTitan model
    """
    # Get model arguments using predefined TorchTitan configs or custom hparams
    titan_args = get_titan_model_args(hparams)

    # Create parallelization dimensions
    pdims = create_parallel_dims(world_size, hparams, role)

    # Create job config
    job_config = create_job_config(hparams, role)

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        # Only build a mesh when distributed is initialized (avoids single-proc errors)
        _ = pdims.build_mesh()

    # Create model on meta device
    with torch.device("meta"):
        model = TitanLlama(titan_args)

    # Parallelize the model
    model = parallelize_llama(
        model,
        parallel_dims=pdims,
        job_config=job_config,
    )

    # Initialize weights via rank‑0 broadcast of a single full FP32 state (FSDP2/DCP)
    target_device = "cpu" if role == "evaluator" else device
    model.to_empty(device=target_device)
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        rank = dist.get_rank()
        if rank == 0:
            with torch.device("meta"):
                ref = TitanLlama(titan_args)  # unsharded reference
            ref.to_empty(device="cpu")
            torch.manual_seed(42)
            with torch.no_grad():
                ref.init_weights()
            full_sd = get_model_state_dict(
                ref, options=StateDictOptions(full_state_dict=True)
            )
        else:
            full_sd = {}
        set_model_state_dict(
            model,
            full_sd,
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, strict=True
            ),
        )
        dist.barrier()
    else:
        # Single‑process path (e.g., 1‑GPU validator / CPU evaluator)
        with torch.device("meta"):
            ref = TitanLlama(titan_args)
        ref.to_empty(device="cpu")
        torch.manual_seed(42)
        with torch.no_grad():
            ref.init_weights()
        full_sd = get_model_state_dict(
            ref, options=StateDictOptions(full_state_dict=True)
        )
        missing, unexpected = set_model_state_dict(
            model,
            full_sd,
            options=StateDictOptions(full_state_dict=True, strict=True),
        )
        assert not missing and not unexpected, (
            f"Missing {missing}, unexpected {unexpected}"
        )

    # Log initialization details
    if role == "miner":
        tt = getattr(hparams, "torchtitan", SimpleNamespace())
        tp_degree = int(getattr(tt, "tp_degree", 1))
        dp_shard = int(getattr(tt, "dp_shard", 1))
        tplr.logger.info(
            f"[Model Factory] Initialized {role} model with TP={tp_degree}, "
            f"DP_shard={dp_shard}, world_size={world_size}"
        )
    else:
        tplr.logger.info(f"[Model Factory] Initialized {role} model on {device}")

    return model


def _get_actual_intermediate_size(
    titan_state_dict: dict, hparams: SimpleNamespace
) -> int:
    """Extract actual intermediate size from model state dict.

    Args:
        titan_state_dict: TorchTitan model state dict
        hparams: Hyperparameters object

    Returns:
        Actual intermediate size from model or hparams fallback
    """
    for key in [
        "layers.0._orig_mod.feed_forward.w1.weight",
        "layers.0.feed_forward.w1.weight",
    ]:
        if key in titan_state_dict:
            return titan_state_dict[key].shape[0]

    # Fallback to hparams
    tplr.logger.warning(
        f"Could not determine intermediate_size from model, using hparams value: {hparams.model_config.intermediate_size}"
    )
    return hparams.model_config.intermediate_size


def convert_titan_to_hf(
    titan_model: nn.Module,
    hparams: SimpleNamespace,
    save_path: str | None = None,
) -> LlamaForCausalLM:
    """Convert TorchTitan model to HuggingFace format.

    This function converts a TorchTitan Llama model to HuggingFace format,
    handling state dict conversion and proper configuration mapping.

    Args:
        titan_model: TorchTitan model to convert
        hparams: Hyperparameters object containing model configuration
        save_path: Optional path to save the converted HuggingFace model

    Returns:
        LlamaForCausalLM: Converted HuggingFace model

    Raises:
        ValueError: If conversion fails
    """
    try:
        # Get TorchTitan state dict
        titan_state_dict = titan_model.state_dict()

        # Determine actual intermediate size from model weights
        actual_intermediate_size = _get_actual_intermediate_size(
            titan_state_dict, hparams
        )
        tplr.logger.info(f"Using intermediate_size: {actual_intermediate_size}")

        # Create HuggingFace config
        hf_config = LlamaConfig(
            vocab_size=hparams.model_config.vocab_size,
            hidden_size=hparams.model_config.hidden_size,
            intermediate_size=actual_intermediate_size,
            num_hidden_layers=hparams.model_config.num_hidden_layers,
            num_attention_heads=hparams.model_config.num_attention_heads,
            num_key_value_heads=getattr(
                hparams.model_config,
                "num_key_value_heads",
                hparams.model_config.num_attention_heads,
            ),
            hidden_act=getattr(hparams.model_config, "hidden_act", "silu"),
            max_position_embeddings=hparams.sequence_length,
            initializer_range=getattr(hparams.model_config, "initializer_range", 0.02),
            rms_norm_eps=getattr(hparams.model_config, "rms_norm_eps", 1e-5),
            use_cache=False,
            rope_theta=getattr(hparams.model_config, "rope_theta", 10000.0),
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            pretraining_tp=1,
            tie_word_embeddings=False,
        )

        # Create HuggingFace model
        hf_model = LlamaForCausalLM(hf_config)

        # Clean state dict (remove prefixes and special keys)
        cleaned = {}
        for k, v in titan_state_dict.items():
            if "freqs_cis" in k:
                continue
            nk = k.replace("_orig_mod.", "").replace("module.", "")
            cleaned[nk] = v

        # Get TorchTitan args from predefined configs or custom hparams
        titan_args = get_titan_model_args(hparams)

        # Use official adapter for state dict conversion
        adapter = Llama3StateDictAdapter(titan_args)
        hf_state_dict = adapter.to_hf(cleaned)

        # Load converted state dict into HuggingFace model
        hf_model.load_state_dict(hf_state_dict, strict=True)

        # Save if path provided
        if save_path:
            hf_model.save_pretrained(save_path)
            tplr.logger.info(
                f"Successfully saved TorchTitan model in HuggingFace format to {save_path}"
            )

        return hf_model

    except Exception as e:
        tplr.logger.error(f"HF conversion failed: {e}", exc_info=True)
        raise ValueError(
            f"Failed to convert TorchTitan model to HuggingFace format: {e}"
        )
