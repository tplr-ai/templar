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

import random
from typing import TYPE_CHECKING, Dict, List

from ..utils.logging import log_with_context

if TYPE_CHECKING:
    from .validator_core import ValidatorCore


def should_update_gather_peer_list(validator_instance: "ValidatorCore") -> bool:
    """
    Returns bool based on hparams.peer_replacement_frequency.
    
    Args:
        validator_instance: ValidatorCore instance
        
    Returns:
        True if peer list should be updated
    """
    return (
        validator_instance.last_peer_update_window is None
        or validator_instance.sync_window - validator_instance.last_peer_update_window
        >= validator_instance.hparams.peer_replacement_frequency
    )


async def select_gather_peers_for_posting(validator_instance: "ValidatorCore") -> tuple[List[int] | None, bool]:
    """
    Implements select_initial_peers and select_next_peers logic for choosing the list of peers 
    that miners should gather from.
    
    Args:
        validator_instance: ValidatorCore instance
        
    Returns:
        Tuple of (peer_list, is_initial_selection)
    """
    if validator_instance.last_peer_update_window is None:
        peers = select_initial_peers(validator_instance)
        return peers, True
    else:
        peers = select_next_peers(validator_instance)
        return peers, False


def select_initial_peers(validator_instance: "ValidatorCore") -> List[int] | None:
    """
    Simple initial peer selection based on incentive.
    1) Select peers with highest incentive
    2) If needed, fill remaining slots with active peers
    3) Ensure we have minimum number of peers
    
    Args:
        validator_instance: ValidatorCore instance
        
    Returns:
        List of selected peer UIDs or None if insufficient peers
    """
    try:
        log_with_context(
            "info",
            "Starting selection of initial gather peers",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )

        # 1. Create a dictionary of active peers with non-zero incentive
        uid_to_incentive = {}
        for uid, incentive in zip(
            validator_instance.metagraph.uids.tolist(), 
            validator_instance.metagraph.I.tolist()
        ):
            if incentive > 0 and uid in validator_instance.comms.active_peers:
                uid_to_incentive[uid] = float(incentive)

        # Sort by incentive (highest first) and take top peers
        top_incentive_peers = sorted(
            uid_to_incentive.keys(),
            key=lambda uid: uid_to_incentive[uid],
            reverse=True,
        )[: validator_instance.hparams.max_topk_peers]

        # If we have enough peers with incentive, return them
        if len(top_incentive_peers) == validator_instance.hparams.max_topk_peers:
            log_with_context(
                "info",
                f"Selected {len(top_incentive_peers)} initial peers purely based on incentive",
                sync_window=validator_instance.sync_window,
                current_window=validator_instance.current_window,
            )
            return top_incentive_peers

        # 2. If needed, fill up with active peers that don't have incentive
        remaining_active_peers = [
            int(peer)
            for peer in validator_instance.comms.active_peers
            if peer not in top_incentive_peers
        ]

        # Calculate how many more peers we need
        needed_peers = validator_instance.hparams.max_topk_peers - len(top_incentive_peers)

        # Randomly select from remaining active peers
        additional_peers = random.sample(
            remaining_active_peers, min(len(remaining_active_peers), needed_peers)
        )

        # Combine the lists
        selected_peers = top_incentive_peers + additional_peers

        # Ensure we have enough peers
        if len(selected_peers) >= validator_instance.hparams.minimum_peers:
            log_with_context(
                "info",
                f"Selected {len(selected_peers)} initial peers: "
                f"{len(top_incentive_peers)} with incentive and "
                f"{len(additional_peers)} without incentive",
                sync_window=validator_instance.sync_window,
                current_window=validator_instance.current_window,
            )
            return selected_peers

        # 3. If we don't have enough peers, give up
        log_with_context(
            "info",
            f"Failed to select at least {validator_instance.hparams.minimum_peers} initial gather "
            f"peers. Found only {len(selected_peers)} active peers, of which "
            f"{len(top_incentive_peers)} had incentive and "
            f"{len(additional_peers)} were incentiveless active peers.",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
        return None

    except Exception as e:
        log_with_context(
            "error",
            f"Failed to create new peer list: {e}",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
        return None


def select_next_peers(validator_instance: "ValidatorCore") -> List[int] | None:
    """
    Simple peer selection that prioritizes the highest weights.
    1) Get all active peers
    2) Sort them by weight (highest first)
    3) Select up to max_topk_peers
    4) If not enough high-weight peers, fill remaining with random active peers
    
    Args:
        validator_instance: ValidatorCore instance
        
    Returns:
        List of selected peer UIDs or None if insufficient peers
    """
    # Get all active peers as a list
    active_peers = [int(peer) for peer in validator_instance.comms.active_peers]

    # Check if we have enough active peers
    if len(active_peers) < validator_instance.hparams.minimum_peers:
        log_with_context(
            "info",
            f"Not enough active peers ({len(active_peers)}) to meet minimum requirement ({validator_instance.hparams.minimum_peers})",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
        return None

    # Create list of (peer_id, weight) tuples
    peer_weights = []
    for peer_id in active_peers:
        weight = float(validator_instance.weights.cpu()[peer_id])
        peer_weights.append((peer_id, weight))

    # Sort by weight (highest first)
    peer_weights.sort(key=lambda x: x[1], reverse=True)

    # Take peers with highest weights first
    highest_weight_count = min(len(peer_weights), validator_instance.hparams.max_topk_peers)
    selected_peers = [peer_id for peer_id, _ in peer_weights[:highest_weight_count]]

    log_with_context(
        "info",
        f"Selected {len(selected_peers)} peers based on highest weights",
        sync_window=validator_instance.sync_window,
        current_window=validator_instance.current_window,
    )

    return selected_peers


def bin_evaluation_peers_by_performance(validator_instance: "ValidatorCore") -> Dict[int, List[int]]:
    """
    Bins evaluation peers based on their performance metrics using OpenSkill ordinal.
    
    Args:
        validator_instance: ValidatorCore instance
        
    Returns:
        Dictionary mapping bin indices to lists of peer UIDs
    """
    num_bins = validator_instance.hparams.num_evaluation_bins
    
    # Get all active peers
    active_peers = list(validator_instance.eval_peers.keys())

    # If we don't have enough peers, return a single bin with all peers
    if len(active_peers) < num_bins * validator_instance.hparams.uids_per_window:
        log_with_context(
            "info",
            f"Not enough active peers ({len(active_peers)}) for {num_bins} bins. Using single bin.",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
        return {0: active_peers}

    # Collect performance metrics for binning
    peer_metrics = []
    for uid in active_peers:
        metric = (
            float(validator_instance.openskill_ratings[uid].ordinal())
            if uid in validator_instance.openskill_ratings
            else 0.0
        )
        peer_metrics.append((uid, metric))

    # Sort peers by metric (highest first)
    peer_metrics.sort(key=lambda x: x[1], reverse=True)
    sorted_peers = [uid for uid, _ in peer_metrics]

    total_peers = len(sorted_peers)
    peers_per_bin = total_peers // num_bins
    remainder = total_peers % num_bins

    # Create bins with all peers distributed
    bins = {}
    start_idx = 0

    for i in range(num_bins):
        # Add one extra peer to the first 'remainder' bins
        bin_size = peers_per_bin + (1 if i < remainder else 0)
        end_idx = start_idx + bin_size

        # Skip empty bins
        if start_idx >= len(sorted_peers):
            continue

        # Ensure we don't go beyond the array bounds
        end_idx = min(end_idx, len(sorted_peers))

        bins[i] = sorted_peers[start_idx:end_idx]
        start_idx = end_idx

    # Log the bins for debugging
    for bin_idx, peer_list in bins.items():
        log_with_context(
            "info",
            f"Bin {bin_idx} (size {len(peer_list)}): {peer_list}",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )

    return bins


def select_evaluation_bin(validator_instance: "ValidatorCore", num_bins: int) -> int:
    """
    Randomly selects a bin index for evaluation.
    
    Args:
        validator_instance: ValidatorCore instance  
        num_bins: Total number of bins
        
    Returns:
        The bin index to evaluate in this window
    """
    next_bin = random.randint(0, num_bins - 1)

    log_with_context(
        "info",
        f"Randomly selected bin {next_bin} for evaluation in window {validator_instance.sync_window}",
        sync_window=validator_instance.sync_window,
        current_window=validator_instance.current_window,
    )

    return next_bin


def select_uids_from_bin(
    validator_instance: "ValidatorCore", bins: Dict[int, List[int]], bin_idx: int
) -> List[int]:
    """
    Selects UIDs from a specific bin using weighted random sampling.
    
    Args:
        validator_instance: ValidatorCore instance
        bins: Dictionary mapping bin indices to lists of peer UIDs
        bin_idx: The specific bin to select peers from
        
    Returns:
        Selected UIDs for evaluation
    """
    # Ensure the bin exists
    if bin_idx not in bins:
        log_with_context(
            "warning",
            f"Bin {bin_idx} not found. Using bin 0 instead.",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
        )
        bin_idx = 0

    # Get peers in the selected bin
    bin_peers = bins[bin_idx]

    # Get weights for weighted sampling
    candidate_uids = list(bin_peers)
    candidate_weights = [validator_instance.eval_peers[uid] for uid in candidate_uids]

    # Determine how many peers to select (either all in the bin or uids_per_window)
    k = min(validator_instance.hparams.uids_per_window, len(candidate_uids))

    # Use weighted random sampling
    selected_uids = validator_instance.comms.weighted_random_sample_no_replacement(
        candidate_uids, candidate_weights, k
    )

    log_with_context(
        "info",
        f"Selected {len(selected_uids)} evaluation UIDs from bin {bin_idx}",
        sync_window=validator_instance.sync_window,
        current_window=validator_instance.current_window,
    )

    return selected_uids


def update_eval_peer_rank_counters(validator_instance: "ValidatorCore", evaluated_uids_this_round: List[int]) -> None:
    """
    Updates counters for fairness - reset counters for chosen peers, increment for not chosen.
    
    Args:
        validator_instance: ValidatorCore instance
        evaluated_uids_this_round: UIDs that were evaluated this round
    """
    # Reset counters for chosen peers
    for uid in evaluated_uids_this_round:
        validator_instance.eval_peers[uid] = 1

    # Increment counters for not chosen peers
    for uid in validator_instance.eval_peers.keys():
        if uid not in evaluated_uids_this_round:
            validator_instance.eval_peers[uid] += 1
            
    # Update comms eval_peers reference
    validator_instance.comms.eval_peers = validator_instance.eval_peers 