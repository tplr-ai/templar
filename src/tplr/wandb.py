# The MIT License (MIT)
# © 2024 templar.tech

import os
import wandb
from wandb.sdk.wandb_run import Run
from . import __version__
from .logging import logger


def initialize_wandb(
    run_prefix: str, uid: str, config: any, group: str, job_type: str
) -> Run:
    """Initialize WandB run with version tracking for unified workspace management."""
    wandb_dir = os.path.join(os.getcwd(), "wandb")
    os.makedirs(wandb_dir, exist_ok=True)

    # Modified run ID file to not include version
    run_id_file = os.path.join(wandb_dir, f"wandb_run_id_{run_prefix}{uid}.txt")

    # Check for existing run
    run_id = None
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()

        try:
            api = wandb.Api(timeout=60)
            api.run(f"tplr/{config.project}/{run_id}")
            logger.info(f"Found existing run ID: {run_id}")
        except Exception:
            logger.info(f"Previous run {run_id} not found in WandB, starting new run")
            run_id = None
            os.remove(run_id_file)

    # Initialize WandB with version as a tag
    run = wandb.init(
        project=config.project,
        entity="tplr",
        id=run_id,
        resume="must" if run_id else "never",
        name=f"{run_prefix}{uid}",
        config=config,
        group=group,
        job_type=job_type,
        dir=wandb_dir,
        tags=[f"v{__version__}"],
        settings=wandb.Settings(
            init_timeout=300,
            _disable_stats=True,
        ),
    )

    # Add version history to run config
    if "version_history" not in run.config:
        run.config.update({"version_history": [__version__]}, allow_val_change=True)
    elif __version__ not in run.config.version_history:
        version_history = run.config.version_history + [__version__]
        run.config.update({"version_history": version_history}, allow_val_change=True)

    # Keep current version in config
    run.config.update({"current_version": __version__}, allow_val_change=True)

    # Track the last step seen for each version
    version_steps = {}

    # Get the current global step from WandB if resuming
    if run_id:
        try:
            api = wandb.Api(timeout=60)
            run_data = api.run(f"tplr/{config.project}/{run_id}")
            history = run_data.scan_history()
            global_step = max((row.get("_step", 0) for row in history), default=0)
            version_steps["global"] = global_step
        except Exception:
            version_steps["global"] = 0
    else:
        version_steps["global"] = 0

    # Create a wrapper for wandb.log that automatically adds version
    original_log = run.log

    def log_with_version(metrics, **kwargs):
        # Only increment if step not provided
        if "step" not in kwargs:
            version_steps["global"] += 1
            current_step = version_steps["global"]
        else:
            current_step = kwargs["step"]
            # Update global step if provided step is higher
            if current_step > version_steps["global"]:
                version_steps["global"] = current_step

        # Initialize version step if needed
        if __version__ not in version_steps:
            version_steps[__version__] = current_step

        # Use version-specific step counter
        versioned_metrics = {}
        for k, v in metrics.items():
            versioned_metrics[f"v{__version__}/{k}"] = v
            versioned_metrics[f"latest/{k}"] = v

        # Add version-specific step counter
        versioned_metrics[f"v{__version__}/step"] = current_step

        original_log(versioned_metrics, **kwargs)

    run.log = log_with_version

    # Save run ID for future resumption
    if not run_id:
        with open(run_id_file, "w") as f:
            f.write(run.id)

    return run
