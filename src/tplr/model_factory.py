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
import torch.nn as nn
from torchtitan.config_manager import (
    ActivationCheckpoint,
    Float8,
    JobConfig,
    Model,
    Parallelism,
    Training,
)
from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3 import Transformer as TitanLlama
from torchtitan.models.llama3 import TransformerModelArgs
from torchtitan.models.llama3.infra.parallelize import parallelize_llama

import tplr


def create_titan_model_args(hf_config, sequence_length: int) -> TransformerModelArgs:
    """Convert HuggingFace LlamaConfig to TorchTitan TransformerModelArgs.

    Args:
        hf_config: HuggingFace LlamaConfig object
        sequence_length: Maximum sequence length for the model

    Returns:
        TransformerModelArgs configured for TorchTitan
    """
    return TransformerModelArgs(
        dim=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=getattr(hf_config, "num_key_value_heads", None),
        vocab_size=hf_config.vocab_size,
        multiple_of=256,
        max_seq_len=sequence_length,
        rope_theta=getattr(hf_config, "rope_theta", 1e4),
        depth_init=True,
    )


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

    if role == "evaluator":
        # Evaluator uses minimal settings for inference
        compile_default = False
        mixed_precision_param_default = "float32"
        enable_cpu_offload_default = False
    elif role == "validator":
        # Validator uses conservative settings
        compile_default = False
        # Ensure we only get valid values
        mp_param = getattr(tt, "mixed_precision_param", "float32")
        mixed_precision_param_default = (
            "bfloat16" if mp_param == "bfloat16" else "float32"
        )
        enable_cpu_offload_default = False
    else:  # miner
        # Miner uses full settings from config
        compile_default = getattr(tt, "compile", False)
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
        # Evaluator: single GPU, no sharding
        return ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            tp=1,
            pp=1,
            cp=1,
            ep=1,
            world_size=1,
        )
    elif role == "validator":
        # Validator: data parallel replication, no sharding
        return ParallelDims(
            dp_replicate=world_size,
            dp_shard=1,
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
    # Create model arguments from HuggingFace config
    hf_cfg = hparams.model_config
    titan_args = create_titan_model_args(hf_cfg, hparams.sequence_length)

    # Create model on meta device
    with torch.device("meta"):
        model = TitanLlama(titan_args)

    # Create parallelization dimensions
    pdims = create_parallel_dims(world_size, hparams, role)

    # Create job config
    job_config = create_job_config(hparams, role)

    # Build mesh if needed (for miner)
    if role == "miner" and world_size > 1:
        _ = pdims.build_mesh()

    # Parallelize the model
    model = parallelize_llama(
        model,
        parallel_dims=pdims,
        job_config=job_config,
    )

    # Initialize model weights on the target device
    if role == "evaluator":
        # Evaluator initializes on CPU
        model.to_empty(device="cpu")
        model.init_weights()  # type: ignore
    else:
        # Validator and miner initialize on GPU
        model.to_empty(device=device)
        model.init_weights()  # type: ignore

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
