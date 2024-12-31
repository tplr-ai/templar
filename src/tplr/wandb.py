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
    """Initialize WandB run with persistence and resumption capabilities.

    Args:
        run_prefix (str): Prefix for the run name (e.g., 'V' for validator, 'M' for miner)
        uid (str): Unique identifier for the run
        config (any): Configuration object containing project and other settings
        group (str): Group name for organizing runs
        job_type (str): Type of job (e.g., 'validation', 'training')

    Returns:
        Run: Initialized WandB run object
    """
    # Ensure the wandb directory exists
    wandb_dir = os.path.join(os.getcwd(), "wandb")
    os.makedirs(wandb_dir, exist_ok=True)

    # Define the run ID file path inside the wandb directory
    run_id_file = os.path.join(
        wandb_dir, f"wandb_run_id_{run_prefix}{uid}_{__version__}.txt"
    )

    # Check for existing run and verify it still exists in wandb
    run_id = None
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()

        # Verify if run still exists in wandb
        try:
            api = wandb.Api()
            api.run(f"tplr/{config.project}-v{__version__}/{run_id}")
            logger.info(f"Found existing run ID: {run_id}")
        except Exception:
            logger.info(f"Previous run {run_id} not found in WandB, starting new run")
            run_id = None
            os.remove(run_id_file)

    # Initialize WandB
    run = wandb.init(
        project=f"{config.project}-v{__version__}",
        entity="tplr",
        id=run_id,
        resume="must" if run_id else "never",
        name=f"{run_prefix}{uid}",
        config=config,
        group=group,
        job_type=job_type,
        dir=wandb_dir,
        settings=wandb.Settings(
            init_timeout=300,
            _disable_stats=True,
        ),
    )

    # Special handling for evaluator
    if run_prefix == "E":
        tasks = config.tasks.split(",")
        for task in tasks:
            metric_name = f"eval/{task}"
            wandb.define_metric(
                name=metric_name, step_metric="global_step", plot=True, summary="max"
            )

    # Save run ID for future resumption
    if not run_id:
        with open(run_id_file, "w") as f:
            f.write(run.id)

    return run


# TODO: Add error handling for network issues
# TODO: Add retry mechanism for wandb initialization
# TODO: Add cleanup mechanism for old run ID files
# TODO: Add support for custom wandb settings
