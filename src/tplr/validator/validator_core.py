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

import asyncio
from collections import defaultdict
from datetime import datetime
from contextlib import contextmanager
from time import perf_counter

import torch
from openskill.models import PlackettLuce

from ..neurons.base_neuron import BaseNeuron
from ..utils.logging import log_with_context
from . import data_utils, evaluation, peer_selection, scoring
from .state import ValidatorStateManager


@contextmanager
def timer(name: str, wandb_obj=None, step=None, metrics_logger=None):
    """Timer context manager for measuring operation durations like the original validator."""
    start = perf_counter()
    yield
    duration = perf_counter() - start
    import tplr
    tplr.logger.debug(f"{name} took {duration:.2f}s")
    if wandb_obj and step is not None:
        wandb_obj.log({f"validator/{name}": duration}, step=step)
    if metrics_logger and step is not None:
        metrics_logger.log(
            measurement="timing", tags={"window": step}, fields={name: duration}
        )


class ValidatorCore(BaseNeuron):
    """
    Validator implementation that inherits from BaseNeuron.
    Orchestrates all validator-specific operations by calling functions from specialized modules.
    """

    @staticmethod  
    def config():
        """Static config method for backward compatibility with original validator."""
        import argparse
        import bittensor as bt
        import tplr
        
        parser = argparse.ArgumentParser(description="Validator script")
        parser.add_argument(
            "--netuid", type=int, default=268, help="Bittensor network UID."
        )
        parser.add_argument(
            "--project", type=str, default="templar", help="Wandb project."
        )
        parser.add_argument(
            "--device", type=str, default="cuda", help="Device to use for training"
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument("--trace", action="store_true", help="Enable trace logging")
        parser.add_argument(
            "--store-gathers",
            action="store_true",
            help="Store gathered gradients in R2",
        )
        parser.add_argument(
            "--test",
            action="store_true",
            help="Test mode - use all peers without filtering",
        )
        parser.add_argument(
            "--local",
            action="store_true",
            help="Local run - use toy model, small enough for a laptop.",
        )
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        config = bt.config(parser)
        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()

        return config

    def __init__(self):
        """Initialize ValidatorCore."""
        super().__init__(neuron_type="validator")
        self._validator_specific_init()
        
        # Initialize state manager and load state
        self.state_manager = ValidatorStateManager(self)
        self.state_manager.load_validator_state()

    @property
    def current_window(self) -> int:
        """Get current window from chain manager."""
        return self.chain_manager.current_window

    @current_window.setter
    def current_window(self, value: int) -> None:
        """Set current window on chain manager."""
        self.chain_manager.current_window = value

    def _validator_specific_init(self) -> None:
        """Initialize validator-only attributes and default score tensors."""
        # OpenSkill model for peer rating
        self.openskill_model = PlackettLuce(
            beta=self.hparams.openskill_beta, tau=self.hparams.openskill_tau
        )
        
        # Initialize empty OpenSkill ratings (loaded by state manager)
        self.openskill_ratings = {}
        
        # Validator step counter
        self.eval_count = 0
        self.sync_window = self.current_window
        
        # Window tracking
        self.window_step = 0
        
        # Peer tracking
        self.eval_peers_rank_counters = defaultdict(int)
        self.peers_last_eval_window = {}
        self.last_peer_update_window = None
        
        # Weighted selection counters for fair picking of eval peers (from old validator)
        self.eval_peers = defaultdict(lambda: 1)
        
        # Current window tracking
        self.current_window_gradient_scores = {}
        self.inactive_peer_tracker = {}
        self.evaluated_uids = set()
        
        # Score tracking variables (per-batch instance variables)
        self.loss_before_per_batch_own = 0.0
        self.loss_after_per_batch_own = 0.0
        self.loss_before_per_batch_random = 0.0
        self.loss_after_per_batch_random = 0.0
        self.loss_improvement_own = 0.0
        self.loss_improvement_random = 0.0
        self.relative_improvement_own = 0.0
        self.relative_improvement_random = 0.0
        
        # For better looking graphs if no eval peers could be evaluated
        self.previous_avg_loss_before_own = 0.0
        self.previous_avg_loss_after_own = 0.0
        self.previous_avg_loss_before_random = 0.0
        self.previous_avg_loss_after_random = 0.0
        
        # For logging continuity if no evals happen (legacy version)
        self.previous_avg_losses = {
            "loss_before_own": 0.0,
            "loss_after_own": 0.0,
            "loss_before_random": 0.0,
            "loss_after_random": 0.0,
        }
        
        # Valid score tracking
        self.valid_score_indices = []
        
        # Track inactive peer scores
        self.inactive_scores = {}  # {uid: (last_active_window, last_score)}
        
        # Validator-specific configuration attributes that scoring module expects
        # Add these as instance attributes instead of trying to modify hparams
        self.missing_gradient_slash_rate = 0.75
        self.sync_score_slash_rate = 0.75  
        self.inactivity_slash_rate = 0.25
        self.reset_inactivity_windows = 10
        self.max_allowed_steps_behind = 5.0
        self.binary_moving_average_alpha = 0.1
        self.power_normalisation = 2.0
        self.weight_setting_frequency = 10

    async def main_loop(self) -> None:
        """Implements the main operational loop for validators."""
        # No initial sleep - use proper offset logic per window instead
        
        while True:
            try:
                # 1. Wait for the validator window offset (from old validator logic)
                while self.sync_window >= (
                    self.current_window - self.hparams.validator_offset
                ):
                    log_with_context(
                        "info",
                        f"Waiting for validator window offset, synced: {self.sync_window}, current:{self.current_window}, offset:{self.hparams.validator_offset}",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                    await asyncio.sleep(12)

                # 2. Start timing the entire window
                import tplr
                window_start = tplr.T()
                
                # Increment sync_window and clear per-window trackers
                self.sync_window += 1
                self.current_window_gradient_scores.clear()
                self.evaluated_uids.clear()
                
                log_with_context(
                    "info",
                    f"Starting validator window {self.sync_window}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )

                # 3. Save state
                self.state_manager.save_validator_state()

                # 4. Handle peer list posting with timing
                peer_start = tplr.T()
                await self._handle_peer_list_posting()
                
                # 5. Update local peer lists with timing
                await self._update_local_peer_lists()
                
                peer_update_time = tplr.T() - peer_start
                log_with_context(
                    "info", 
                    f"Peer update completed in {peer_update_time:.2f}s",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )

                # 6. Handle inactive peers and penalties
                await self._handle_inactive_peers()

                # 7. Get time constraints for gathering with timing
                time_min, time_max = await self._get_gather_time_constraints()

                # 8. Gather gradients with comprehensive timing
                gather_start = tplr.T()
                
                # Get gather peers from comms (not peer_manager directly)
                gather_peers = self.comms.peers
                if not gather_peers:
                    log_with_context(
                        "warning",
                        "No gather peers available, skipping window",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                    continue

                log_with_context(
                    "info",
                    f"Starting gradient gather from {len(gather_peers)} peers",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )

                # Use timer context manager for gradient gathering
                with timer("gradient_gather", self.wandb, self.global_step, self.metrics_logger):
                    gather_result = await self.comms.gather(
                        my_uid=self.uid,
                        uids=gather_peers,
                        window=self.sync_window,  # Use current sync window like old validator
                        timeout=45,
                        device="cpu",
                        local=False,
                        totalks=self.totalks,
                        time_min=time_min,
                        time_max=time_max,
                    )
                
                gather_time = tplr.T() - gather_start
                log_with_context(
                    "info",
                    f"Gradient gather completed in {gather_time:.2f}s",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )

                # 9. Apply gathered gradients if available with timing
                if gather_result and gather_result.uids:
                    with timer("apply_gradients", self.wandb, self.global_step, self.metrics_logger):
                        await self._apply_gathered_gradients(gather_result)

                # 10. Sync evaluation with comprehensive timing
                eval_start = tplr.T()
                
                # Get source details for evaluation
                source_details = {
                    "gather_result": gather_result,
                    "gather_peers": gather_peers,
                    "skipped_uids": gather_result.skipped_uids if gather_result else [],
                }

                # Evaluate sync status
                sync_results = await self._evaluate_sync_status(source_details)

                # Main evaluation phase with timing
                with timer("peer_evaluation", self.wandb, self.global_step, self.metrics_logger):
                    avg_losses, evaluated_count = await self._main_evaluation_phase(time_min, time_max)
                
                evaluation_time = tplr.T() - eval_start
                log_with_context(
                    "info",
                    f"Evaluation phase completed in {evaluation_time:.2f}s, evaluated {evaluated_count} peers",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )

                # 11. Update model and weights with timing
                update_start = tplr.T()
                
                with timer("model_update", self.wandb, self.global_step, self.metrics_logger):
                    # Update weights
                    self.update_weights()
                    
                    # Update OpenSkill ratings
                    self.update_openskill_ratings()
                
                model_update_time = tplr.T() - update_start
                log_with_context(
                    "info",
                    f"Model update completed in {model_update_time:.2f}s",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )

                # 12. Comprehensive metrics logging with all timing data
                window_total_time = tplr.T() - window_start
                
                await self._log_aggregated_window_metrics(
                    avg_losses, 
                    evaluated_count, 
                    source_details,
                    timing_data={
                        "window_total": window_total_time,
                        "peer_update": peer_update_time,
                        "gather": gather_time,
                        "evaluation": evaluation_time,
                        "model_update": model_update_time,
                        "window_start": window_start,
                        "peer_start": peer_start,
                        "eval_start": eval_start,
                        "update_start": update_start,
                        "gather_start": gather_start,
                    }
                )

                log_with_context(
                    "info",
                    f"Window {self.sync_window} completed in {window_total_time:.2f}s (peer: {peer_update_time:.2f}s, gather: {gather_time:.2f}s, eval: {evaluation_time:.2f}s, update: {model_update_time:.2f}s)",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )

                # 13. Checkpoint creation
                if (
                    self.global_step % self.hparams.checkpoint_frequency == 0
                    and self.global_step != 0
                ):
                    with timer("checkpoint_creation", self.wandb, self.global_step, self.metrics_logger):
                        await self.chain_manager.save_checkpoint(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            self.global_step,
                            self.current_window,
                            self.start_window,
                            self.sync_window,
                        )

                # 14. Profiling summary (every 10 windows)
                if self.sync_window % 10 == 0:
                    with timer("profiling_summary", self.wandb, self.global_step, self.metrics_logger):
                        import tplr.r2_dataset
                        tplr.logger.info("Logging performance profiling summary...")
                        tplr.r2_dataset.R2DatasetLoader.log_profiling_summary()

                # 15. Increment global step
                self.global_step += 1
                
                # Clear CUDA cache
                torch.cuda.empty_cache()

            except Exception as e:
                log_with_context(
                    "error",
                    f"Error in main loop: {e}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                await asyncio.sleep(5)  # Brief pause before retry

    async def _handle_peer_list_posting(self) -> None:
        """Orchestrate peer list posting logic."""
        if peer_selection.should_update_gather_peer_list(self):
            peers, is_initial = await peer_selection.select_gather_peers_for_posting(self)
            
            if peers is not None:
                await self.comms.post_peer_list(
                    peers=peers,
                    first_effective_window=self.current_window + self.hparams.peer_list_window_margin,
                    sync_window=self.sync_window,
                    weights=self.weights,
                    initial_selection=is_initial,
                )
                self.last_peer_update_window = self.sync_window
                
                log_with_context(
                    "info",
                    f"Posted {'initial' if is_initial else 'updated'} peer list with {len(peers)} peers",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                    peer_count=len(peers),
                    is_initial=is_initial,
                )
            else:
                log_with_context(
                    "warning",
                    "Failed to select peers for posting",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )

    #TODO: move to peer manager
    async def _update_local_peer_lists(self) -> None:
        """Update local peer lists using new Comms orchestration."""
        # Update peers with buckets
        self.comms.peer_manager.update_peers_with_buckets()
        
        # Use new Comms method to refresh gather targets
        await self.comms.refresh_gather_targets()
        
        # Access updated gather target peers from PeerManager
        self.comms.peers = self.comms.peer_manager.gather_target_peers
        
        # Initialize eval_peers from comms.eval_peers if available, else use active peers
        if hasattr(self.comms, 'eval_peers') and self.comms.eval_peers:
            self.eval_peers = self.comms.eval_peers
        else:
            # Initialize eval_peers with all active peers having equal weight (1)
            active_peers = list(self.comms.peer_manager.active_peers)
            self.eval_peers = defaultdict(lambda: 1)
            for uid in active_peers:
                self.eval_peers[uid] = 1
        
        log_with_context(
            "info", 
            f"Updated peers - eval:{len(self.eval_peers)}",
            sync_window=self.sync_window,
            current_window=self.current_window,
        )
        
        log_with_context(
            "info",
            f"Current gather peers: {self.comms.peers}",
            sync_window=self.sync_window,
            current_window=self.current_window,
        )
        
        log_with_context(
            "info",
            f"Current evaluation peers: {list(self.eval_peers.keys())}",
            sync_window=self.sync_window,
            current_window=self.current_window,
        )

    # TODO: move to base neuron 
    async def _get_gather_time_constraints(self) -> tuple[datetime, datetime]:
        """Query blockchain for timestamp to define time_min, time_max."""
        # Use new Comms orchestration method for time window calculation
        # Fix: Add +1 to sync_window to match old validator logic: (sync_window + 1) * blocks_per_window
        time_window = await self.comms.get_activity_time_window(self.sync_window + 1)
        
        if time_window is None:
            raise RuntimeError(f"Failed to calculate time window for sync window {self.sync_window}")
            
        time_min, time_max = time_window
        
        log_with_context(
            "info",
            f"Using time window for gather: {time_min} to {time_max}",
            sync_window=self.sync_window,
            current_window=self.current_window,
        )
        return time_min, time_max

    async def _evaluate_sync_status(self, source_details: dict) -> list:
        """Evaluate sync status for gather peers."""
        gather_result = source_details.get("gather_result")
        skipped_uids = gather_result.skipped_uids if gather_result else []
        
        successful_peers = list(set(self.comms.peers) - set(skipped_uids))
        
        if not successful_peers:
            return []
            
        # Evaluate sync status for successful peers
        sync_eval_tasks = [
            evaluation.evaluate_miner_sync_status(self, uid, self.sync_window - 1)
            for uid in successful_peers
        ]
        
        sync_eval_results = await asyncio.gather(*sync_eval_tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to list
        valid_results = []
        for result in sync_eval_results:
            if isinstance(result, Exception):
                log_with_context(
                    "error",
                    f"Sync evaluation failed: {result}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                valid_results.append(None)
            else:
                valid_results.append(result)
        
        return valid_results

    async def _main_evaluation_phase(self, time_min: datetime, time_max: datetime) -> tuple[dict, int]:
        """
        Main evaluation phase that selects peers and evaluates them.
        
        Returns:
            Tuple of (average_losses_dict, evaluated_count)
        """
        # Bin evaluation peers by performance
        bins = peer_selection.bin_evaluation_peers_by_performance(self)
        
        # Select evaluation bin
        bin_idx = peer_selection.select_evaluation_bin(self, len(bins))
        
        # Select UIDs from bin
        selected_uids = peer_selection.select_uids_from_bin(self, bins, bin_idx)
        
        if not selected_uids:
            log_with_context(
                "warning",
                "No UIDs selected for evaluation",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            return self.previous_avg_losses, 0
        
        # Update eval peer rank counters
        peer_selection.update_eval_peer_rank_counters(self, selected_uids)
        
        # Preload common random dataloader
        with timer("preload_common_random_data", self.wandb, self.global_step, self.metrics_logger):
            random_seed = 256 + (self.sync_window % 100)  # Seeds > 255 are "random"
            common_data_result = await data_utils.preload_dataloader(self, random_seed)
            
        if common_data_result["loader"] is None:
            log_with_context(
                "error",
                "Failed to preload common random dataloader",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )
            return self.previous_avg_losses, 0
        
        common_random_loader = common_data_result["loader"]
        
        # Evaluate selected UIDs
        loss_metrics = []
        evaluated_count = 0
        
        # Preload first UID's data
        next_uid_data = None
        if len(selected_uids) > 0:
            next_uid_data = await data_utils.preload_dataloader(self, selected_uids[0])
        
        for i, eval_uid in enumerate(selected_uids):
            # Use preloaded data for current UID
            current_uid_data = next_uid_data
            
            # Preload next UID's data while processing current
            if i + 1 < len(selected_uids):
                next_uid_data = await data_utils.preload_dataloader(self, selected_uids[i + 1])
            
            if current_uid_data and current_uid_data["loader"] is not None:
                with timer(f"evaluate_uid_{eval_uid}", self.wandb, self.global_step, self.metrics_logger):
                    result = await evaluation.evaluate_individual_miner(
                        self, eval_uid, current_uid_data["loader"], common_random_loader, time_min, time_max
                    )
                
                if result is not None:
                    loss_metrics.append(result)
                    evaluated_count += 1
            else:
                log_with_context(
                    "warning",
                    f"Failed to load data for UID {eval_uid}, skipping evaluation",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                    eval_uid=eval_uid,
                )
        
        # Calculate average losses
        if loss_metrics:
            avg_losses = {
                "loss_before_own": sum(m["loss_own_before"] for m in loss_metrics) / len(loss_metrics),
                "loss_after_own": sum(m["loss_own_after"] for m in loss_metrics) / len(loss_metrics),
                "loss_before_random": sum(m["loss_random_before"] for m in loss_metrics) / len(loss_metrics),
                "loss_after_random": sum(m["loss_random_after"] for m in loss_metrics) / len(loss_metrics),
            }
            self.previous_avg_losses = avg_losses
        else:
            avg_losses = self.previous_avg_losses
        
        log_with_context(
            "info",
            f"Completed evaluation phase: {evaluated_count}/{len(selected_uids)} peers evaluated successfully",
            sync_window=self.sync_window,
            current_window=self.current_window,
            evaluated_count=evaluated_count,
            selected_count=len(selected_uids),
        )
        
        return avg_losses, evaluated_count

    async def _log_aggregated_window_metrics(self, avg_losses: dict, evaluated_count: int, source_details: dict, timing_data: dict) -> None:
        """Log aggregated metrics for the window with comprehensive timing data."""
        improvement_own = avg_losses["loss_before_own"] - avg_losses["loss_after_own"]
        improvement_random = avg_losses["loss_before_random"] - avg_losses["loss_after_random"]

        # Calculate relative improvements for continuity
        relative_improvement_own = (
            improvement_own / avg_losses["loss_before_own"]
            if avg_losses["loss_before_own"] > 0
            else 0
        )
        relative_improvement_random = (
            improvement_random / avg_losses["loss_before_random"]
            if avg_losses["loss_before_random"] > 0
            else 0
        )

        # Store improvements in instance variables for other modules
        self.loss_improvement_own = improvement_own
        self.loss_improvement_random = improvement_random
        self.relative_improvement_own = relative_improvement_own
        self.relative_improvement_random = relative_improvement_random

        # Get gather metrics if available
        gather_result = source_details.get("gather_result")
        success_rate = gather_result.success_rate if gather_result else 0.0
        gather_peers = source_details.get("gather_peers", {})

        # Comprehensive metrics logging to WandB (like original validator)
        evaluation_metrics = {
            "validator/window": self.sync_window,
            "validator/current_block": self.current_block,
            "validator/loss_before_own": avg_losses["loss_before_own"],
            "validator/loss_after_own": avg_losses["loss_after_own"], 
            "validator/loss_before_random": avg_losses["loss_before_random"],
            "validator/loss_after_random": avg_losses["loss_after_random"],
            "validator/loss_improvement_own": improvement_own,
            "validator/loss_improvement_random": improvement_random,
            "validator/relative_improvement_own": relative_improvement_own,
            "validator/relative_improvement_random": relative_improvement_random,
            "validator/network/step": self.global_step,
            "validator/network/evaluated_uids": len(self.evaluated_uids),
            "validator/optimizer/learning_rate": self.scheduler.get_last_lr()[0],
            "validator/network/active_miners": len(self.valid_score_indices),
            "validator/gather/success_rate": success_rate * 100,
            # Comprehensive timing metrics like original validator
            "validator/timing/window_total": timing_data["window_total"],
            "validator/timing/peer_update": timing_data["peer_update"],
            "validator/timing/gather": timing_data["gather"],
            "validator/timing/evaluation": timing_data["evaluation"],
            "validator/timing/model_update": timing_data["model_update"],
        }
        self.wandb.log(evaluation_metrics, step=self.global_step)

        # Log comprehensive metrics to InfluxDB (like original validator)
        gather_success_rate = float(success_rate * 100)
        skipped_uids = source_details.get("skipped_uids", [])
        total_skipped = len(skipped_uids)

        self.metrics_logger.log(
            measurement="validator_window_v2",
            tags={
                "window": int(self.sync_window),
                "global_step": int(self.global_step),
            },
            fields={
                "loss_own_before": float(avg_losses["loss_before_own"]),
                "loss_own_after": float(avg_losses["loss_after_own"]),
                "loss_random_before": float(avg_losses["loss_before_random"]),
                "loss_random_after": float(avg_losses["loss_after_random"]),
                "loss_own_improvement": float(relative_improvement_own),
                "loss_random_improvement": float(relative_improvement_random),
                "current_block": int(self.current_block),
                "evaluated_uids_count": int(len(self.evaluated_uids)),
                "learning_rate": float(self.scheduler.get_last_lr()[0]),
                "active_miners_count": int(len(self.valid_score_indices)),
                "gather_success_rate": gather_success_rate,
                # Comprehensive timing fields like original validator
                "window_total_time": float(timing_data["window_total"]),
                "peer_update_time": float(timing_data["peer_update"]),
                "gather_time": float(timing_data["gather"]),
                "evaluation_time": float(timing_data["evaluation"]),
                "model_update_time": float(timing_data["model_update"]),
                "total_peers": int(len(gather_peers)),
                "total_skipped": int(total_skipped),
            },
            with_system_metrics=True,
            with_gpu_metrics=True,
        )

        log_with_context(
            "info",
            "Finished comprehensive metrics logging for validator",
            sync_window=self.sync_window,
            current_window=self.current_window,
        )

        # Store previous averages for continuity if no evaluations happen
        self.previous_avg_losses = avg_losses

    async def _handle_inactive_peers(self) -> None:
        """Handle inactive peers and apply penalties."""
        scoring.process_inactive_peers(self)

    async def _apply_gathered_gradients(self, gather_result) -> None:
        """Apply gathered gradients to the validator model."""
        evaluation.apply_gradients_to_validator_model(self, gather_result)

    def min_power_normalization(self, logits, power=2.0, epsilon=1e-8):
        """Normalizes logits using a minimum power normalization approach.

        This function applies power normalization to the input logits, raising them to a power
        and normalizing to create a probability distribution. If the sum is too small (below epsilon),
        returns zeros to avoid division by very small numbers.

        Args:
            logits (torch.Tensor): Input tensor to be normalized
            power (float, optional): Power to raise the logits to. Defaults to 2.0.
            epsilon (float, optional): Small value to prevent division by zero. Defaults to 1e-8.

        Returns:
            torch.Tensor: Normalized probabilities
        """
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)

        powered_logits = logits**power
        sum_powered = torch.sum(powered_logits)
        if sum_powered > epsilon:
            probabilities = powered_logits / sum_powered
        else:
            probabilities = torch.zeros_like(powered_logits)

        return probabilities

    def update_weights(self) -> None:
        """
        Update the weights for all evaluated peers using min power normalization.
        This method:
        1. Creates a mask for peers that have been evaluated
        2. Creates a mask for evaluated peers with positive scores
        3. Applies power normalization to only the positive scores
        4. Verifies that weights sum to approximately 1.0
        This approach only assigns weights to peers with positive scores.
        """
        self.weights = torch.zeros_like(self.final_scores)
        evaluated_mask = torch.zeros_like(self.final_scores, dtype=torch.bool)
        evaluated_mask[list(self.evaluated_uids)] = True

        # Create a mask for positive scores among evaluated peers
        positive_mask = evaluated_mask.clone()
        positive_mask[evaluated_mask] = self.final_scores[evaluated_mask] > 0

        # Only consider peers with positive scores
        positive_scores = self.final_scores[positive_mask]

        if len(positive_scores) > 0:
            # Apply power normalization to only the positive scores
            normalized_weights = self.min_power_normalization(
                positive_scores,
                power=self.power_normalisation,
            )

            # Assign weights only to peers with positive scores
            self.weights[positive_mask] = normalized_weights

            weight_sum = self.weights.sum().item()
            log_with_context(
                "debug",
                f"Weight sum: {weight_sum}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            if abs(weight_sum - 1.0) > 1e-6:
                log_with_context(
                    "warning",
                    f"Weights sum to {weight_sum}, expected close to 1.0",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
        else:
            log_with_context(
                "warning",
                "No positive scores found among evaluated peers. All weights set to zero.",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

    def update_openskill_ratings(self):
        """
        Update OpenSkill ratings based on gradient scores and recalculate final scores.

        This method:
        1. Processes all peers evaluated in the current window
        2. Updates their OpenSkill ratings based on gradient performance
        3. Recalculates final scores using OpenSkill mu value combined with binary and sync scores
        4. Logs the updated ratings to monitoring systems

        The OpenSkill rating system provides a probabilistic skill rating that accounts for
        uncertainty and relative performance between peers. Ratings are updated using the
        PlackettLuce model where higher gradient scores indicate better performance.

        The final score calculation combines:
        - OpenSkill mu (mean skill estimate)
        - Binary moving average (filtered to non-negative values)
        - Sync score (model synchronization quality)
        """
        if (
            hasattr(self, "current_window_gradient_scores")
            and len(self.current_window_gradient_scores) > 1
        ):
            # Get UIDs and scores
            window_uids = list(self.current_window_gradient_scores.keys())

            # Store original ordinal values to calculate diff after update
            original_ordinals = {}
            for uid in window_uids:
                if uid in self.openskill_ratings:
                    original_ordinals[uid] = float(
                        self.openskill_ratings[uid].ordinal()
                    )
                else:
                    # For new peers without previous ratings
                    original_ordinals[uid] = 0.0

            # Calculate ranks based on gradient scores (lower rank = better performance)
            # In OpenSkill, ranks start at 1 (best) and increase for worse performers
            scores = [self.current_window_gradient_scores[uid] for uid in window_uids]

            # Create teams list for OpenSkill
            teams = [[self.openskill_ratings[uid]] for uid in window_uids]

            # Rate the teams using scores (higher score is better in OpenSkill)
            rated_teams = self.openskill_model.rate(teams, scores=scores)

            # Store updated ratings
            for i, uid in enumerate(window_uids):
                self.openskill_ratings[uid] = rated_teams[i][0]

                # Log updated OpenSkill values
                openskill_mu = float(self.openskill_ratings[uid].mu)
                openskill_sigma = float(self.openskill_ratings[uid].sigma)
                openskill_ordinal = float(self.openskill_ratings[uid].ordinal())

                sync_score = float(
                    self.sync_scores[uid].item() if uid in self.evaluated_uids else 0.0
                )

                self.final_scores[uid] = (
                    openskill_ordinal
                    * max(0, self.binary_moving_averages[uid].item())
                    * sync_score
                )
                log_with_context(
                    "info",
                    f"Computed Final Score for UID {uid}: {self.final_scores[uid]}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                    eval_uid=uid,
                )

                # Log to WandB
                self.wandb.log(
                    {
                        f"validator/openskill/mu/{uid}": openskill_mu,
                        f"validator/openskill/sigma/{uid}": openskill_sigma,
                        f"validator/openskill/ordinal/{uid}": openskill_ordinal,
                    },
                    step=self.global_step,
                )

                # Log to InfluxDB
                self.metrics_logger.log(
                    measurement="validator_openskill",
                    tags={
                        "eval_uid": str(uid),
                        "window": int(self.sync_window),
                        "global_step": int(self.global_step),
                    },
                    fields={
                        "mu": openskill_mu,
                        "sigma": openskill_sigma,
                        "ordinal": openskill_ordinal,
                    },
                )

            log_with_context(
                "info",
                f"Updated OpenSkill ratings for {len(window_uids)} peers based on gradient scores",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            # Clear the current window scores
            self.current_window_gradient_scores = {}

    def reset_peer(self, inactive_since: int, uid: int) -> bool:
        """Backwards compatibility method for reset_peer - delegates to scoring module."""
        return scoring._reset_peer(self, inactive_since, uid)

    def log_sync_score(self, eval_uid: int, sync_result: dict[str, bool | float | int | str]) -> None:
        """Backwards compatibility method for log_sync_score - delegates to evaluation module."""
        evaluation._log_sync_score(self, eval_uid, sync_result) 