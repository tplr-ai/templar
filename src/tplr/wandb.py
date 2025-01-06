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
            api = wandb.Api()
            api.run(f"tplr/{config.project}/{run_id}")
            logger.info(f"Found existing run ID: {run_id}")
        except Exception:
            logger.info(f"Previous run {run_id} not found in WandB, starting new run")
            run_id = None
            os.remove(run_id_file)

    # Initialize WandB with version as a tag
    run = wandb.init(
        project=config.project,  # Remove version from project name
        entity="tplr",
        id=run_id,
        resume="must" if run_id else "never",
        name=f"{run_prefix}{uid}",
        config=config,
        group=group,
        job_type=job_type,
        dir=wandb_dir,
        tags=[f"v{__version__}"],  # Add version as a tag
        settings=wandb.Settings(
            init_timeout=300,
            _disable_stats=True,
        ),
    )

    # Add version to run config
    run.config.update({"version": __version__})

    # Create a wrapper for wandb.log that automatically adds version
    original_log = run.log

    def log_with_version(metrics, **kwargs):
        # Add version to each metric name
        versioned_metrics = {f"v{__version__}/{k}": v for k, v in metrics.items()}
        return original_log(versioned_metrics, **kwargs)

    run.log = log_with_version

    # Save run ID for future resumption
    if not run_id:
        with open(run_id_file, "w") as f:
            f.write(run.id)

    return run

