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
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import cast

import torch
from bittensor.core.subtensor import ScaleObj
from openskill.models import PlackettLuce

import tplr
from tplr.neurons import neuron_utils
from ..neurons.base_neuron import BaseNeuron
from ..utils.logging import log_with_context, timer
from . import data_utils, evaluation, peer_selection, scoring
from .state import ValidatorStateManager


class ValidatorCore(BaseNeuron):
    """
    Validator implementation that inherits from BaseNeuron.
    Orchestrates all validator-specific operations by calling functions from specialized modules.
    """

    def __init__(self):
        """Initialize ValidatorCore."""
        super().__init__(neuron_type="validator")
        self._validator_specific_init()
        
        # Initialize state manager and load state
        self.state_manager = ValidatorStateManager(self)
        self.state_manager.load_validator_state()

    def _validator_specific_init(self) -> None:
        """Initialize validator-only attributes and default score tensors."""
        # OpenSkill model for peer rating
        self.openskill_model = PlackettLuce(
            beta=self.hparams.openskill_beta, tau=self.hparams.openskill_tau
        )
        
        # Validator step counter
        self.eval_count = 0
        self.sync_window = self.current_window
        
        # Peer tracking
        self.eval_peers_rank_counters = defaultdict(int)
        self.peers_last_eval_window = {}
        self.last_peer_update_window = None
        
        # Current window tracking
        self.current_window_gradient_scores = {}
        self.inactive_peer_tracker = {}
        self.evaluated_uids = set()
        
        # For logging continuity if no evals happen
        self.previous_avg_losses = {
            "loss_before_own": 0.0,
            "loss_after_own": 0.0,
            "loss_before_random": 0.0,
            "loss_after_random": 0.0,
        }

    async def main_loop(self) -> None:
        """Implements the main operational loop for validators."""
        # Wait for validator offset
        await asyncio.sleep(self.hparams.validator_offset)
        
        while True:
            # 1. Increment sync_window and clear per-window trackers
            self.sync_window += 1
            self.current_window_gradient_scores.clear()
            self.evaluated_uids.clear()
            
            log_with_context(
                "info",
                f"Starting validator window {self.sync_window}",
                sync_window=self.sync_window,
                current_window=self.current_window,
            )

            # 2. Save validator state
            self.state_manager.save_validator_state()

            # 3. Handle peer list posting
            await self._handle_peer_list_posting()

            # 4. Update local peer lists
            await self._update_local_peer_lists()

            # 5. Process inactive peers
            scoring.process_inactive_peers(self)

            # 6. Get gather time constraints
            time_min, time_max = await self._get_gather_time_constraints()

            # 7. Get gradients for window
            processed_gradient_data, source_details = await evaluation.get_gradients_for_window(
                self, time_min, time_max
            )

            if processed_gradient_data is not None:
                # 8. Evaluate sync status for gather peers
                sync_eval_results = await self._evaluate_sync_status(source_details)

                # 9. Process sync evaluations and slash peers missing gradients
                scoring.process_sync_evaluations_and_slash(
                    self, 
                    list(set(self.comms.peers) - set(source_details["skipped_uids"])),
                    sync_eval_results
                )
                scoring.slash_peers_missing_gradients(
                    self, source_details["skipped_uids"], source_details["success_rate"]
                )

                # 10. Main evaluation phase
                avg_losses, evaluated_count = await self._main_evaluation_phase(time_min, time_max)

                # 11. Update OpenSkill ratings and final scores
                scoring.update_openskill_ratings_and_final_scores(self)

                # 12. Update final validator weights
                scoring.update_final_validator_weights(self)

                # 13. Log formatted score table
                scoring.log_formatted_score_table(self)

                # 14. Log all peer detailed metrics
                scoring.log_all_peer_detailed_metrics(self)

                # 15. Set weights on subnet (if applicable this window)
                await scoring.set_weights_on_subnet(self)

                # 16. Apply gradients to validator model
                evaluation.apply_gradients_to_validator_model(self, processed_gradient_data)

                # 17. Store validator debug info
                await evaluation.store_validator_debug_info(self, source_details)

                # 18. Log aggregated window metrics
                await self._log_aggregated_window_metrics(avg_losses, evaluated_count, source_details)

            else:
                log_with_context(
                    "warning",
                    "No gradients obtained this window, skipping evaluation and model update",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )

            # 19. Save validator checkpoint data (if applicable)
            await self.state_manager.save_validator_checkpoint_data()

            # 20. Increment counters and cleanup
            self.global_step += 1
            self.eval_count += 1
            torch.cuda.empty_cache()

            # 21. Wait for next window
            await self._wait_for_next_validator_window()

    async def _handle_peer_list_posting(self) -> None:
        """Orchestrate peer list posting logic."""
        if peer_selection.should_update_gather_peer_list(self):
            peers, is_initial = await peer_selection.select_gather_peers_for_posting(self)
            
            if peers is not None:
                await self.comms.post_peer_list(peers)
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
        """Update local peer lists."""
        # Update peers with buckets
        self.comms.update_peers_with_buckets()
        
        # Update peers using neuron_utils module function
        await neuron_utils.update_peers(instance=self, window=self.sync_window, peer_start=tplr.T())
        await neuron_utils.update_peers(instance=self, window=self.sync_window)

    # TODO: move to base neuron 
    async def _get_gather_time_constraints(self) -> tuple[datetime, datetime]:
        """Query blockchain for timestamp to define time_min, time_max."""
        sync_block = self.sync_window * self.hparams.blocks_per_window
        retries = 0
        delay = 1
        max_retries = 5
        max_delay = 60
        
        while True:
            try:
                response = self.subtensor.query_module("Timestamp", "Now", block=sync_block)
                if response is None or not isinstance(response, ScaleObj):
                    raise ValueError(f"Could not query timestamp for {sync_block}")
                ts_value = cast(int, response.value) / 1000  # convert ms to seconds
                break
            except Exception as e:
                log_with_context(
                    "error",
                    f"Failed to query timestamp for block {sync_block}: {str(e)}. Retry {retries + 1}/{max_retries}",
                    sync_window=self.sync_window,
                    current_window=self.current_window,
                )
                retries += 1
                if retries > max_retries:
                    log_with_context(
                        "error",
                        "Exceeded maximum retries for timestamp query.",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                    raise e
                time.sleep(delay)
                delay = min(delay * 2, max_delay)

        time_min = datetime.fromtimestamp(ts_value, tz=timezone.utc)
        time_max = time_min + timedelta(seconds=self.hparams.time_window_delta_seconds)
        
        log_with_context(
            "info",
            f"Using time window for gather: {time_min} to {time_max}",
            sync_window=self.sync_window,
            current_window=self.current_window,
        )
        return time_min, time_max

    async def _evaluate_sync_status(self, source_details: dict) -> list:
        """Evaluate sync status for gather peers."""
        successful_peers = list(set(self.comms.peers) - set(source_details["skipped_uids"]))
        
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

    async def _log_aggregated_window_metrics(self, avg_losses: dict, evaluated_count: int, source_details: dict) -> None:
        """Log aggregated window metrics."""
        # Calculate improvements
        improvement_own = avg_losses["loss_before_own"] - avg_losses["loss_after_own"]
        improvement_random = avg_losses["loss_before_random"] - avg_losses["loss_after_random"]
        
        relative_improvement_own = improvement_own / max(avg_losses["loss_before_own"], 1e-8)
        relative_improvement_random = improvement_random / max(avg_losses["loss_before_random"], 1e-8)
        
        # Log to WandB
        self.wandb.log(
            {
                "validator/window/sync_window": self.sync_window,
                "validator/window/evaluated_peers": evaluated_count,
                "validator/window/gather_success_rate": source_details["success_rate"],
                "validator/window/from_aggregator": source_details["from_aggregator"],
                "validator/loss/avg_loss_before_own": avg_losses["loss_before_own"],
                "validator/loss/avg_loss_after_own": avg_losses["loss_after_own"],
                "validator/loss/avg_loss_before_random": avg_losses["loss_before_random"],
                "validator/loss/avg_loss_after_random": avg_losses["loss_after_random"],
                "validator/improvement/own": improvement_own,
                "validator/improvement/random": improvement_random,
                "validator/improvement/relative_own": relative_improvement_own,
                "validator/improvement/relative_random": relative_improvement_random,
                "validator/timing/gather_time": source_details["gather_time"],
            },
            step=self.global_step,
        )
        
        # Log to InfluxDB
        self.metrics_logger.log(
            measurement="validator_window_summary",
            tags={
                "window": int(self.sync_window),
                "global_step": int(self.global_step),
            },
            fields={
                "evaluated_peers": int(evaluated_count),
                "gather_success_rate": float(source_details["success_rate"]),
                "from_aggregator": bool(source_details["from_aggregator"]),
                "avg_loss_before_own": float(avg_losses["loss_before_own"]),
                "avg_loss_after_own": float(avg_losses["loss_after_own"]),
                "avg_loss_before_random": float(avg_losses["loss_before_random"]),
                "avg_loss_after_random": float(avg_losses["loss_after_random"]),
                "improvement_own": float(improvement_own),
                "improvement_random": float(improvement_random),
                "relative_improvement_own": float(relative_improvement_own),
                "relative_improvement_random": float(relative_improvement_random),
                "gather_time": float(source_details["gather_time"]),
            },
            with_system_metrics=True,
            with_gpu_metrics=True,
        )

    async def _wait_for_next_validator_window(self) -> None:
        """Wait for the next validator window."""
        target_window = self.sync_window + 1
        
        log_with_context(
            "info",
            f"Waiting for validator window {target_window}",
            sync_window=self.sync_window,
            current_window=self.current_window,
        )
        
        while self.current_window < target_window:
            await asyncio.sleep(0.1)
        
        log_with_context(
            "info",
            f"Advanced to validator window {target_window}",
            sync_window=target_window,
            current_window=self.current_window,
        ) 