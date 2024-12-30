# The MIT License (MIT)
# © 2024 templar.tech

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
# fmt: off

# Global imports
import os
import wandb as wandbm

# Local imports
from . import __version__, logger

class WandbManager:
    def __init__(self, run_prefix=None, uid=None, config=None, group=None, job_type=None, is_validator=False):
        """Initialize WandB manager with proper run management and resumability.
        
        Args:
            run_prefix: Optional prefix for the run name (if None, determined by is_validator)
            uid: User ID
            config: Config object containing wandb settings
            group: Group name (if None, determined by is_validator)
            job_type: Job type (if None, determined by is_validator)
            is_validator: Boolean indicating if this is a validator run
        """
        self.wandb_dir = os.path.join(os.getcwd(), 'wandb')
        os.makedirs(self.wandb_dir, exist_ok=True)
        self.run = None

        if all(x is not None for x in [uid, config]):
            # Set defaults based on validator status if not provided
            if run_prefix is None:
                run_prefix = 'V' if is_validator else 'M'
            if group is None:
                group = 'validator' if is_validator else 'miner'
            if job_type is None:
                job_type = 'validation' if is_validator else 'training'

            # Define the run ID file path inside the wandb directory
            run_id_file = os.path.join(
                self.wandb_dir, f"wandb_run_id_{run_prefix}{uid}_{__version__}.txt"
            )

            # Check for existing run and verify it still exists in wandb
            run_id = None
            if os.path.exists(run_id_file):
                with open(run_id_file, 'r') as f:
                    run_id = f.read().strip()
                
                # Verify if run still exists in wandb
                try:
                    api = wandbm.Api()
                    api.run(f"tplr/{config.project}-v{__version__}/{run_id}")
                    logger.info(f"Found existing run ID: {run_id}")
                except Exception:
                    logger.info(f"Previous run {run_id} not found in WandB, starting new run")
                    run_id = None
                    os.remove(run_id_file)

            # Initialize WandB
            self.run = wandbm.init(
                project=f"{config.project}-v{__version__}",
                entity='tplr',
                id=run_id,
                resume='allow',
                name=f'{run_prefix}{uid}',
                config=config,
                group=group,
                job_type=job_type,
                dir=self.wandb_dir,
                settings=wandbm.Settings(
                    init_timeout=300,
                    _disable_stats=True,
                )
            )

            # Save run ID for future resumption
            if not run_id:
                with open(run_id_file, 'w') as f:
                    f.write(self.run.id)
