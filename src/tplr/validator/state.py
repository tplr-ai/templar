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

import os
from typing import TYPE_CHECKING

import torch
import tplr

if TYPE_CHECKING:
    from .validator_core import ValidatorCore


class ValidatorStateManager:
    """
    Manages all persistent state for the Validator, including scores, ratings, and checkpoint data.
    """

    def __init__(self, validator_instance: "ValidatorCore"):
        """
        Initialize the state manager.
        
        Args:
            validator_instance: Reference to the ValidatorCore instance
        """
        self.validator_instance = validator_instance
        self.state_path = f"validator-state-{tplr.__version__}.pt"

    def load_validator_state(self) -> None:
        """
        Load validator state from disk and populate attributes on validator_instance.
        Handle default initialization if no state file exists.
        """
        if os.path.isfile(self.state_path):
            self._load_existing_state()
        else:
            self._initialize_default_state()

    def _load_existing_state(self) -> None:
        """Load state from existing file."""
        tplr.logger.info("Loading validator state")
        
        try:
            state = torch.load(self.state_path, map_location=self.validator_instance.config.device)
        except FileNotFoundError:
            tplr.logger.warning(f"No validator state found at {self.state_path}")
            self._initialize_default_state()
            return
        except Exception as e:
            tplr.logger.warning(f"Failed to deserialize validator state: {e}")
            self._initialize_default_state()
            return

        # Load global step
        self.validator_instance.global_step = int(
            state.get("global_step", getattr(self.validator_instance, "global_step", 0))
        )

        # Load tensor states
        tensor_names = [
            "gradient_scores",
            "sync_scores", 
            "binary_indicator_scores",
            "final_scores",
            "binary_moving_averages",
            "weights",
        ]
        
        for tensor_name in tensor_names:
            if tensor_name in state:
                setattr(
                    self.validator_instance,
                    tensor_name,
                    state[tensor_name].float().to(self.validator_instance.config.device),
                )

        # Load inactive_scores tracking dict
        self.validator_instance.inactive_scores = state.get("inactive_scores", {})

        # Load OpenSkill ratings
        try:
            saved_openskill = state.get("openskill_ratings", {})
            self.validator_instance.openskill_ratings = {
                int(uid): self.validator_instance.openskill_model.rating(
                    mu=float(rating_data["mu"]), 
                    sigma=float(rating_data["sigma"]), 
                    name=str(uid)
                )
                for uid, rating_data in saved_openskill.items()
            }
            tplr.logger.info(
                f"Restored OpenSkill ratings for {len(self.validator_instance.openskill_ratings)} peers"
            )
        except Exception as e:
            tplr.logger.warning(f"Failed to restore OpenSkill ratings: {e}")
            self.validator_instance.openskill_ratings = {}

    def _initialize_default_state(self) -> None:
        """Initialize default state when no existing state file is found."""
        device = self.validator_instance.config.device
        
        self.validator_instance.gradient_scores = torch.zeros(256, dtype=torch.float32, device=device)
        self.validator_instance.sync_scores = torch.zeros(256, dtype=torch.float32, device=device)
        self.validator_instance.binary_indicator_scores = torch.zeros(256, dtype=torch.float32, device=device)
        self.validator_instance.final_scores = torch.zeros(256, dtype=torch.float32, device=device)
        self.validator_instance.binary_moving_averages = torch.zeros(256, dtype=torch.float32, device=device)
        self.validator_instance.weights = torch.zeros(256, dtype=torch.float32, device=device)
        
        # Initialize empty OpenSkill ratings
        self.validator_instance.openskill_ratings = {}

    def save_validator_state(self) -> None:
        """Save current scores and ratings to disk."""
        try:
            tplr.logger.info("Saving validator state")
            state_dict = self._get_validator_state_dict_for_save()
            torch.save(state_dict, self.state_path)
        except Exception as e:
            tplr.logger.warning(f"Failed to save validator state: {e}")

    def _get_validator_state_dict_for_save(self) -> dict:
        """Collect all relevant tensors (moved to CPU) and ratings into a dict for torch.save."""
        return {
            "global_step": self.validator_instance.global_step,
            "gradient_scores": self.validator_instance.gradient_scores.cpu(),
            "sync_scores": self.validator_instance.sync_scores.cpu(),
            "binary_indicator_scores": self.validator_instance.binary_indicator_scores.cpu(),
            "final_scores": self.validator_instance.final_scores.cpu(),
            "binary_moving_averages": self.validator_instance.binary_moving_averages.cpu(),
            "weights": self.validator_instance.weights.cpu(),
            # Store inactive_scores tracking dict
            "inactive_scores": self.validator_instance.inactive_scores,
            # Store OpenSkill statistics per-uid for full restoration
            "openskill_ratings": {
                int(uid): {
                    "mu": float(rating.mu),
                    "sigma": float(rating.sigma),
                    "ordinal": float(rating.ordinal()),
                }
                for uid, rating in self.validator_instance.openskill_ratings.items()
            },
        }

    async def save_validator_checkpoint_data(self) -> None:
        """Save model, optimizer, scheduler state for the validator."""
        if (
            self.validator_instance.global_step % self.validator_instance.hparams.checkpoint_frequency == 0
            and self.validator_instance.global_step != 0
        ):
            tplr.logger.info(
                f"Creating checkpoint at global_step {self.validator_instance.global_step}"
            )
            
            checkpoint_data = {
                "model_state_dict": {
                    k: v.cpu().clone() 
                    for k, v in self.validator_instance.model.state_dict().items()
                },
                "optimizer_state_dict": {
                    k: v.cpu().clone() if torch.is_tensor(v) else v
                    for k, v in self.validator_instance.optimizer.state_dict().items()
                },
                "scheduler_state_dict": self.validator_instance.scheduler.state_dict(),
                "start_window": self.validator_instance.start_window,
                "current_window": self.validator_instance.current_window,
                "sync_window": self.validator_instance.sync_window,
            }
            
            # Use asyncio.create_task to avoid blocking
            import asyncio
            asyncio.create_task(
                self.validator_instance.comms.put(
                    state_dict=checkpoint_data,
                    uid=str(self.validator_instance.uid),
                    window=self.validator_instance.sync_window,
                    key="checkpoint",
                    global_step=self.validator_instance.global_step,
                    local=False,
                )
            ) 