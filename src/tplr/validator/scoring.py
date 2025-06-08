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
from io import StringIO
from typing import TYPE_CHECKING, Dict, List

import torch
from rich.console import Console
from rich.table import Table

from ..common.math_utils import min_power_normalization
from ..utils.logging import log_with_context

if TYPE_CHECKING:
    from .validator_core import ValidatorCore


def process_inactive_peers(validator_instance: "ValidatorCore") -> None:
    """
    Iterate validator_instance.inactive_peer_tracker, apply penalties, reset fully inactive peers.
    
    Args:
        validator_instance: ValidatorCore instance
    """
    to_remove = []
    
    for uid, (inactive_since, last_score) in validator_instance.inactive_peer_tracker.items():
        windows_inactive = validator_instance.sync_window - inactive_since
        
        if windows_inactive > 0:
            # Apply inactivity penalty
            penalty = validator_instance.inactivity_slash_rate ** windows_inactive
            new_score = last_score * penalty
            
            # Update final score
            validator_instance.final_scores[uid] = new_score
            
            log_with_context(
                "info",
                f"UID {uid} inactive for {windows_inactive} windows, score: {last_score:.4f} -> {new_score:.4f}",
                sync_window=validator_instance.sync_window,
                current_window=validator_instance.current_window,
            )
            
            # Check if peer should be fully reset
            if _reset_peer(validator_instance, inactive_since, uid):
                to_remove.append(uid)
                
    # Remove fully reset peers from tracker
    for uid in to_remove:
        del validator_instance.inactive_peer_tracker[uid]


def _reset_peer(validator_instance: "ValidatorCore", inactive_since: int, uid: int) -> bool:
    """Check if peer should be fully reset and reset if needed."""
    if validator_instance.sync_window - inactive_since > validator_instance.reset_inactivity_windows:
        validator_instance.final_scores[uid] = 0.0
        validator_instance.weights[uid] = 0.0
        validator_instance.gradient_scores[uid] = 0.0
        validator_instance.binary_moving_averages[uid] = 0.0
        validator_instance.binary_indicator_scores[uid] = 0.0
        validator_instance.sync_scores[uid] = 0.0
        if uid in validator_instance.openskill_ratings:
            del validator_instance.openskill_ratings[uid]
        if uid in validator_instance.eval_peers_rank_counters:
            del validator_instance.eval_peers_rank_counters[uid]
        
        log_with_context(
            "info",
            f"UID {uid} fully reset after extended inactivity",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
        return True
    return False


def process_sync_evaluations_and_slash(
    validator_instance: "ValidatorCore", 
    evaluated_peer_uids: List[int], 
    sync_eval_results: List[Dict]
) -> None:
    """
    Update validator_instance.sync_scores and apply penalties if avg_steps_behind is too high.
    
    Args:
        validator_instance: ValidatorCore instance
        evaluated_peer_uids: UIDs that were evaluated for sync
        sync_eval_results: Results from sync evaluations
    """
    for uid, sync_result in zip(evaluated_peer_uids, sync_eval_results):
        # Handle None results from failed evaluations
        if sync_result is None:
            validator_instance.sync_scores[uid] = 0.1
            log_with_context(
                "info",
                f"UID {uid} failed sync evaluation",
                sync_window=validator_instance.sync_window,
                current_window=validator_instance.current_window,
            )
            continue
            
        if sync_result and sync_result.get("success", False):
            sync_score = float(sync_result.get("sync_score", 0.0))
            avg_steps_behind = float(sync_result.get("avg_steps_behind", 0.0))
            
            # Update sync score
            validator_instance.sync_scores[uid] = sync_score
            
            # Apply penalty if too far behind
            if avg_steps_behind > validator_instance.max_allowed_steps_behind:
                penalty_factor = validator_instance.sync_score_slash_rate
                validator_instance.final_scores[uid] *= penalty_factor
                
                log_with_context(
                    "warning",
                    f"UID {uid} slashed for being {avg_steps_behind:.1f} steps behind (penalty: {penalty_factor})",
                    sync_window=validator_instance.sync_window,
                    current_window=validator_instance.current_window,
                )
        else:
            # Failed sync evaluation - set low sync score
            validator_instance.sync_scores[uid] = 0.1
            
            log_with_context(
                "info",
                f"UID {uid} failed sync evaluation",
                sync_window=validator_instance.sync_window,
                current_window=validator_instance.current_window,
            )


def slash_peers_missing_gradients(
    validator_instance: "ValidatorCore", 
    skipped_uids_from_gather: List[int], 
    gather_success_rate: float
) -> None:
    """
    Apply penalties to UIDs in skipped_uids_from_gather.
    
    Args:
        validator_instance: ValidatorCore instance
        skipped_uids_from_gather: UIDs that were skipped during gather
        gather_success_rate: Success rate of the gather operation
    """
    if not skipped_uids_from_gather:
        return
        
    penalty_factor = validator_instance.missing_gradient_slash_rate
    
    for uid in skipped_uids_from_gather:
        # Apply penalty to final score
        validator_instance.final_scores[uid] *= penalty_factor
        
        # Track in inactive peers if not already tracked
        if uid not in validator_instance.inactive_peer_tracker:
            validator_instance.inactive_peer_tracker[uid] = (
                validator_instance.sync_window, 
                float(validator_instance.final_scores[uid])
            )
    
    log_with_context(
        "warning",
        f"Slashed {len(skipped_uids_from_gather)} peers for missing gradients (success_rate: {gather_success_rate:.2%})",
        sync_window=validator_instance.sync_window,
        current_window=validator_instance.current_window,
        skipped_uids=skipped_uids_from_gather,
    )


def slash_peer_for_eval_failure(validator_instance: "ValidatorCore", eval_uid: int, reason_code: str) -> None:
    """
    Specific slashing for peers failing evaluation steps.
    
    Args:
        validator_instance: ValidatorCore instance
        eval_uid: UID that failed evaluation
        reason_code: Reason for the failure
    """
    penalty_factor = 0.5  # TODO: Make this configurable in hparams
    
    # Apply penalty to final score
    validator_instance.final_scores[eval_uid] *= penalty_factor
    
    # Set low scores for this evaluation
    validator_instance.gradient_scores[eval_uid] = 0.0
    validator_instance.binary_indicator_scores[eval_uid] = 0.0
    
    log_with_context(
        "warning",
        f"UID {eval_uid} slashed for evaluation failure: {reason_code} (penalty: {penalty_factor})",
        sync_window=validator_instance.sync_window,
        current_window=validator_instance.current_window,
        eval_uid=eval_uid,
        reason=reason_code,
    )


def update_scores_after_evaluation(
    validator_instance: "ValidatorCore",
    eval_uid: int,
    loss_own_before: float,
    loss_own_after: float,
    loss_random_before: float,
    loss_random_after: float
) -> None:
    """
    Update gradient score, binary indicator, and binary moving average for evaluated peer.
    Store relative improvements in validator instance for logging.
    
    Args:
        validator_instance: ValidatorCore instance
        eval_uid: UID of evaluated peer
        loss_own_before: Loss before applying gradient on UID-specific data
        loss_own_after: Loss after applying gradient on UID-specific data
        loss_random_before: Loss before applying gradient on random data
        loss_random_after: Loss after applying gradient on random data
    """
    # Initialize or update OpenSkill rating for this peer (like original validator)
    if eval_uid not in validator_instance.openskill_ratings:
        validator_instance.openskill_ratings[eval_uid] = validator_instance.openskill_model.rating(
            name=str(eval_uid)
        )

    # Calculate improvements
    improvement_own = loss_own_before - loss_own_after
    improvement_random = loss_random_before - loss_random_after
    
    # Calculate relative improvements
    relative_improvement_own = improvement_own / max(loss_own_before, 1e-8)
    relative_improvement_random = improvement_random / max(loss_random_before, 1e-8)
    
    # Store in validator instance for logging compatibility (these will be overwritten per UID)
    validator_instance.relative_improvement_own = relative_improvement_own
    validator_instance.relative_improvement_random = relative_improvement_random
    
    # Calculate gradient score as relative improvement on random data (like original validator)
    gradient_score = improvement_random / max(loss_random_before, 1e-8)
    
    # Update gradient score
    validator_instance.gradient_scores[eval_uid] = gradient_score
    
    # Update binary indicator using LOCAL relative improvement comparison (like original validator)
    # Compare relative improvements: own vs random data performance
    binary_indicator = 1.0 if relative_improvement_own > relative_improvement_random else -1.0
    validator_instance.binary_indicator_scores[eval_uid] = binary_indicator
    
    # Update binary moving average with decay (using hparams parameter name)
    alpha = validator_instance.hparams.binary_score_ma_alpha
    current_avg = float(validator_instance.binary_moving_averages[eval_uid])
    new_avg = (1 - alpha) * current_avg + alpha * binary_indicator
    validator_instance.binary_moving_averages[eval_uid] = new_avg
    
    # Add to current window scores for OpenSkill update
    validator_instance.current_window_gradient_scores[eval_uid] = gradient_score
    
    log_with_context(
        "info",
        f"Updated scores for UID {eval_uid}: gradient={gradient_score:.6f}, binary_avg={new_avg:.4f}, "
        f"rel_improve_own={relative_improvement_own:.4f}, "
        f"rel_improve_random={relative_improvement_random:.4f}",
        sync_window=validator_instance.sync_window,
        current_window=validator_instance.current_window,
        eval_uid=eval_uid,
        gradient_score=gradient_score,
        binary_indicator=binary_indicator,
        improvement_own=improvement_own,
        improvement_random=improvement_random,
        relative_improvement_own=relative_improvement_own,
        relative_improvement_random=relative_improvement_random,
    )


def update_openskill_ratings_and_final_scores(validator_instance: "ValidatorCore") -> None:
    """
    Use current_window_gradient_scores to update openskill_ratings.
    Recalculate final_scores using OpenSkill ordinal, binary moving average, and sync score.
    
    Args:
        validator_instance: ValidatorCore instance
    """
    if (
        hasattr(validator_instance, "current_window_gradient_scores")
        and len(validator_instance.current_window_gradient_scores) > 1
    ):
        # Get UIDs and scores
        window_uids = list(validator_instance.current_window_gradient_scores.keys())

        # Store original ordinal values to calculate diff after update
        original_ordinals = {}
        for uid in window_uids:
            if uid in validator_instance.openskill_ratings:
                original_ordinals[uid] = float(
                    validator_instance.openskill_ratings[uid].ordinal()
                )
            else:
                # For new peers without previous ratings
                original_ordinals[uid] = 0.0

        # Calculate scores for OpenSkill rating
        scores = [validator_instance.current_window_gradient_scores[uid] for uid in window_uids]

        # Create teams list for OpenSkill
        teams = [[validator_instance.openskill_ratings[uid]] for uid in window_uids]

        # Rate the teams using scores (higher score is better in OpenSkill)
        rated_teams = validator_instance.openskill_model.rate(teams, scores=scores)

        # Store updated ratings and recalculate final scores
        for i, uid in enumerate(window_uids):
            validator_instance.openskill_ratings[uid] = rated_teams[i][0]

            # Calculate final score components
            openskill_ordinal = float(validator_instance.openskill_ratings[uid].ordinal())
            binary_avg = max(0, validator_instance.binary_moving_averages[uid].item())
            sync_score = float(
                validator_instance.sync_scores[uid].item() if uid in validator_instance.evaluated_uids else 0.0
            )

            # Calculate final score
            final_score = openskill_ordinal * max(0, binary_avg) * sync_score
            validator_instance.final_scores[uid] = final_score

            log_with_context(
                "info",
                f"Updated final score for UID {uid}: {final_score:.6f} (OS: {openskill_ordinal:.4f}, binary: {binary_avg:.4f}, sync: {sync_score:.4f})",
                sync_window=validator_instance.sync_window,
                current_window=validator_instance.current_window,
                eval_uid=uid,
                final_score=final_score,
                openskill_ordinal=openskill_ordinal,
                binary_avg=binary_avg,
                sync_score=sync_score,
            )

            # Log to WandB
            validator_instance.wandb.log(
                {
                    f"validator/openskill/mu/{uid}": float(validator_instance.openskill_ratings[uid].mu),
                    f"validator/openskill/sigma/{uid}": float(validator_instance.openskill_ratings[uid].sigma),
                    f"validator/openskill/ordinal/{uid}": openskill_ordinal,
                },
                step=validator_instance.global_step,
            )

            # Log to InfluxDB
            validator_instance.metrics_logger.log(
                measurement="validator_openskill",
                tags={
                    "eval_uid": str(uid),
                    "window": int(validator_instance.sync_window),
                    "global_step": int(validator_instance.global_step),
                },
                fields={
                    "mu": float(validator_instance.openskill_ratings[uid].mu),
                    "sigma": float(validator_instance.openskill_ratings[uid].sigma),
                    "ordinal": openskill_ordinal,
                },
            )

        # Log OpenSkill rankings table
        _log_openskill_rankings_table(validator_instance, window_uids, original_ordinals)

        log_with_context(
            "info",
            f"Updated OpenSkill ratings for {len(window_uids)} peers based on gradient scores",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )


def _log_openskill_rankings_table(
    validator_instance: "ValidatorCore", window_uids: List[int], original_ordinals: Dict[int, float]
) -> None:
    """Helper function to log OpenSkill rankings table."""
    try:
        # Sort UIDs by current window gradient scores (descending)
        sorted_uids = sorted(
            window_uids,
            key=lambda uid: validator_instance.current_window_gradient_scores[uid],
            reverse=True,
        )

        try:
            width = os.get_terminal_size().columns
        except Exception:
            width = 0
        os.environ["COLUMNS"] = str(max(200, width))

        rich_table = Table(
            title=f"Current Match Rankings (Window {validator_instance.sync_window})"
        )
        rich_table.add_column("Match Rank")
        rich_table.add_column("UID")
        rich_table.add_column("Match Score")
        rich_table.add_column("OpenSkill μ (After)")
        rich_table.add_column("OpenSkill σ (After)")
        rich_table.add_column("Ordinal (After)")
        rich_table.add_column("Ordinal Δ")

        # Add rows to table
        for rank, uid in enumerate(sorted_uids, 1):
            rating = validator_instance.openskill_ratings[uid]
            ordinal_before = original_ordinals[uid]
            ordinal_after = rating.ordinal()
            ordinal_diff = ordinal_after - ordinal_before

            # Format the diff with color indicators
            diff_str = f"{ordinal_diff:+.4f}"

            rich_table.add_row(
                str(rank),
                str(uid),
                f"{validator_instance.current_window_gradient_scores[uid]:.6f}",
                f"{rating.mu:.4f}",
                f"{rating.sigma:.4f}",
                f"{ordinal_after:.4f}",
                diff_str,
            )

        # Render table to string
        sio = StringIO()
        console = Console(file=sio, width=int(os.environ["COLUMNS"]))
        console.print(rich_table)
        table_str = sio.getvalue()

        log_with_context(
            "info",
            f"Current Match Rankings (Window {validator_instance.sync_window}):\n{table_str}",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
    except Exception as e:
        log_with_context(
            "warning",
            f"Failed to create OpenSkill rankings table: {str(e)}",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )


def update_final_validator_weights(validator_instance: "ValidatorCore") -> None:
    """
    Normalize final_scores (positive scores only) into weights using min_power_normalization.
    
    Args:
        validator_instance: ValidatorCore instance
    """
    validator_instance.weights = torch.zeros_like(validator_instance.final_scores)
    evaluated_mask = torch.zeros_like(validator_instance.final_scores, dtype=torch.bool)
    evaluated_mask[list(validator_instance.evaluated_uids)] = True

    # Create a mask for positive scores among evaluated peers
    positive_mask = evaluated_mask.clone()
    positive_mask[evaluated_mask] = validator_instance.final_scores[evaluated_mask] > 0

    # Only consider peers with positive scores
    positive_scores = validator_instance.final_scores[positive_mask]

    if len(positive_scores) > 0:
        # Apply power normalization to only the positive scores
        normalized_weights = min_power_normalization(
            positive_scores,
            power=validator_instance.power_normalisation,
        )

        # Assign weights only to peers with positive scores
        validator_instance.weights[positive_mask] = normalized_weights

        weight_sum = validator_instance.weights.sum().item()
        log_with_context(
            "debug",
            f"Weight sum: {weight_sum}",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )

        if abs(weight_sum - 1.0) > 1e-6:
            log_with_context(
                "warning",
                f"Weights sum to {weight_sum}, expected close to 1.0",
                sync_window=validator_instance.sync_window,
                current_window=validator_instance.current_window,
            )
    else:
        log_with_context(
            "warning",
            "No positive scores found among evaluated peers. All weights set to zero.",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )


def log_formatted_score_table(validator_instance: "ValidatorCore") -> None:
    """
    Generate and log the rich table of current scores for evaluated_uids.
    
    Args:
        validator_instance: ValidatorCore instance
    """
    if not validator_instance.evaluated_uids:
        log_with_context(
            "info",
            "No peers were evaluated this window",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
        return

    try:
        # Get terminal width for table formatting
        try:
            width = os.get_terminal_size().columns
        except Exception:
            width = 0
        os.environ["COLUMNS"] = str(max(200, width))

        # Create rich table
        table = Table(title=f"Validator Scores (Window {validator_instance.sync_window})")
        table.add_column("UID")
        table.add_column("Gradient Score")
        table.add_column("Binary Avg")
        table.add_column("Sync Score")
        table.add_column("Final Score")
        table.add_column("Weight")

        # Sort evaluated UIDs by final score (descending)
        sorted_uids = sorted(
            validator_instance.evaluated_uids,
            key=lambda uid: float(validator_instance.final_scores[uid]),
            reverse=True,
        )

        # Add rows to table
        for uid in sorted_uids:
            gradient_score = float(validator_instance.gradient_scores[uid])
            binary_avg = float(validator_instance.binary_moving_averages[uid])
            sync_score = float(validator_instance.sync_scores[uid])
            final_score = float(validator_instance.final_scores[uid])
            weight = float(validator_instance.weights[uid])

            table.add_row(
                str(uid),
                f"{gradient_score:.6f}",
                f"{binary_avg:.4f}",
                f"{sync_score:.4f}",
                f"{final_score:.6f}",
                f"{weight:.6f}",
            )

        # Render table to string
        sio = StringIO()
        console = Console(file=sio, width=int(os.environ["COLUMNS"]))
        console.print(table)
        table_str = sio.getvalue()

        log_with_context(
            "info",
            f"Validator Scores (Window {validator_instance.sync_window}):\n{table_str}",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )

    except Exception as e:
        log_with_context(
            "warning",
            f"Failed to create score table: {str(e)}",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )


def log_all_peer_detailed_metrics(validator_instance: "ValidatorCore") -> None:
    """
    Log individual scores for each evaluated UID to WandB and InfluxDB.
    
    Args:
        validator_instance: ValidatorCore instance
    """
    for uid in validator_instance.evaluated_uids:
        gradient_score = float(validator_instance.gradient_scores[uid])
        binary_avg = float(validator_instance.binary_moving_averages[uid])
        sync_score = float(validator_instance.sync_scores[uid])
        final_score = float(validator_instance.final_scores[uid])
        weight = float(validator_instance.weights[uid])

        # Log to WandB
        validator_instance.wandb.log(
            {
                f"validator/scores/gradient/{uid}": gradient_score,
                f"validator/scores/binary_avg/{uid}": binary_avg,
                f"validator/scores/sync/{uid}": sync_score,
                f"validator/scores/final/{uid}": final_score,
                f"validator/weights/{uid}": weight,
            },
            step=validator_instance.global_step,
        )

        # Log to InfluxDB
        validator_instance.metrics_logger.log(
            measurement="validator_peer_scores",
            tags={
                "eval_uid": str(uid),
                "window": int(validator_instance.sync_window),
                "global_step": int(validator_instance.global_step),
            },
            fields={
                "gradient_score": gradient_score,
                "binary_avg": binary_avg,
                "sync_score": sync_score,
                "final_score": final_score,
                "weight": weight,
            },
            with_system_metrics=True,
            with_gpu_metrics=True,
        )


async def set_weights_on_subnet(validator_instance: "ValidatorCore") -> None:
    """
    Call validator_instance.subtensor.set_weights with positive weights.
    
    Args:
        validator_instance: ValidatorCore instance
    """
    # Only set weights if this is the appropriate window
    if validator_instance.sync_window % validator_instance.weight_setting_frequency != 0:
        return

    # Get UIDs and weights for peers with positive weights
    positive_weight_mask = validator_instance.weights > 0
    positive_uids = torch.nonzero(positive_weight_mask).squeeze().tolist()
    
    if not isinstance(positive_uids, list):
        positive_uids = [positive_uids] if isinstance(positive_uids, int) else []
    
    if not positive_uids:
        log_with_context(
            "warning",
            "No positive weights to set on subnet",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
        return

    positive_weights = validator_instance.weights[positive_weight_mask]

    try:
        # Set weights on subnet
        success, message = await validator_instance.loop.run_in_executor(
            validator_instance.executor,
            validator_instance.subtensor.set_weights,
            validator_instance.wallet,
            validator_instance.config.netuid,
            positive_uids,
            positive_weights,
            0,  # version_key
        )

        if success:
            log_with_context(
                "info",
                f"Successfully set weights for {len(positive_uids)} peers",
                sync_window=validator_instance.sync_window,
                current_window=validator_instance.current_window,
                num_weights=len(positive_uids),
                total_weight=float(positive_weights.sum()),
            )
        else:
            log_with_context(
                "error",
                f"Failed to set weights: {message}",
                sync_window=validator_instance.sync_window,
                current_window=validator_instance.current_window,
            )

    except Exception as e:
        log_with_context(
            "error",
            f"Exception while setting weights: {e}",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        ) 