# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# fmt: off


# Global imports
import os
import wandb

# Local imports
from templar import __version__, logger

def initialize_wandb(run_prefix, uid, config, group, job_type):
    # Ensure the wandb directory exists
    wandb_dir = os.path.join(os.getcwd(), 'wandb')
    os.makedirs(wandb_dir, exist_ok=True)

    # Define the run ID file path inside the wandb directory
    run_id_file = os.path.join(
        wandb_dir, f"wandb_run_id_{run_prefix}{uid}_{__version__}.txt"
    )

    # Check for existing run and verify it still exists in wandb
    run_id = None
    if os.path.exists(run_id_file):
        with open(run_id_file, 'r') as f:
            run_id = f.read().strip()
        
        # Verify if run still exists in wandb
        try:
            api = wandb.Api()
            api.run(f"tplr/{config.project}-v{__version__}/{run_id}")
            logger.info(f"Found existing run ID: {run_id}")
        except Exception:
            # Run doesn't exist anymore, clear the run_id
            logger.info(f"Previous run {run_id} not found in WandB, starting new run")
            run_id = None
            os.remove(run_id_file)

    # Initialize WandB
    run = wandb.init(
        project=f"{config.project}-v{__version__}",
        entity='tplr',
        id=run_id,
        resume='must' if run_id else 'never',
        name=f'{run_prefix}{uid}',
        config=config,
        group=group,
        job_type=job_type,
        dir=wandb_dir,
        settings=wandb.Settings(
            init_timeout=300,
            _disable_stats=True,
        )
    )

    # Special handling for evaluator
    if run_prefix == "E":
        tasks = config.tasks.split(',')
        for task in tasks:
            metric_name = f"eval/{task}"
            # Set up x/y plot configuration
            wandb.define_metric(
                name=metric_name,
                step_metric="global_step",  # This sets global_step as x-axis
                plot=True,  # Ensure it creates a line plot
                summary="max"
            )

    # Save run ID for future resumption
    if not run_id:
        with open(run_id_file, 'w') as f:
            f.write(run.id)

    return run
