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

    # Attempt to read the existing run ID
    if os.path.exists(run_id_file):
        with open(run_id_file, 'r') as f:
            run_id = f.read().strip()
        logger.info(f"Resuming WandB run with id {run_id}")
    else:
        run_id = None
        logger.info("Starting a new WandB run.")

    # Initialize WandB
    wandb.init(
        project=f"{config.project}-v{__version__}",
        entity='tplr',
        resume='allow',
        id=run_id,
        name=f'{run_prefix}{uid}',
        config=config,
        group=group,
        job_type=job_type,
        dir=wandb_dir,
        anonymous='allow',
    )

    # Save the run ID if starting a new run
    if run_id is None:
        run_id = wandb.run.id
        with open(run_id_file, 'w') as f:
            f.write(run_id)
        logger.info(f"Started new WandB run with id {run_id}")

    return wandb
