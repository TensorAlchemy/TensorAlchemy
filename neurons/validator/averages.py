"""
moving_averages.py

Handles calculation and updating of moving averages for miner scores in a
distributed network. Includes functionality for score decay based on
iterations, and miner inactivity handling.

Key functions:
- update_moving_averages: Main function for updating moving averages
- track_miner_responses: Tracks miner response history with hotkey mapping
- should_apply_decay: Determines if decay should be applied based on iterations
- apply_decay: Applies decay to inactive miners' scores
- log_moving_averages: Logs moving averages for monitoring
- apply_blacklist: Applies blacklist to moving averages

Usage:
    new_ma_scores = await update_moving_averages(previous_ma_scores, scoring_results)
"""

from collections import deque
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass

import bittensor as bt
import torch
from loguru import logger

from neurons.constants import MOVING_AVERAGE_ALPHA
from neurons.validator.backend.exceptions import PostMovingAveragesError
from neurons.config import (
    get_config,
    get_device,
    get_metagraph,
    get_backend_client,
    get_blacklist,
)
from scoring.types import ScoringResults


@dataclass
class MinerResponse:
    uid: int
    hotkey: str
    timestamp: int


# Global variables
miner_response_history: deque = deque(maxlen=200)
iteration_count: int = 0
previous_hotkeys: List[str] = []
hotkey_to_uid_map: Dict[str, int] = {}


async def save_moving_averages(
    hotkeys: List[str],
    scores: torch.FloatTensor,
) -> None:
    """Save moving average scores to the backend."""
    try:
        await get_backend_client().post_moving_averages(hotkeys, scores)
    except PostMovingAveragesError as e:
        logger.error(f"Failed to post moving averages: {e}")


def log_moving_averages(moving_average_scores: torch.FloatTensor) -> None:
    """Log moving averages for Grafana monitoring."""
    for uid, score in enumerate(moving_average_scores.tolist()):
        if score > 0:
            logger.info(f"miner_uid={uid}, miner_score={score:.4f}")


async def apply_blacklist(
    scores: torch.FloatTensor, metagraph: bt.metagraph
) -> torch.FloatTensor:
    """Apply blacklist to moving average scores."""
    try:
        hotkey_blacklist, coldkey_blacklist = await get_blacklist()

        def is_blacklisted(hotkey: str, coldkey: str) -> bool:
            return hotkey in hotkey_blacklist or coldkey in coldkey_blacklist

        return torch.tensor(
            [
                0 if is_blacklisted(hotkey, coldkey) else score
                for score, hotkey, coldkey in zip(
                    scores, metagraph.hotkeys, metagraph.coldkeys
                )
            ]
        )
    except Exception as e:
        logger.error(f"Error applying blacklist: {e}")
        return scores


def update_hotkey_mapping(metagraph: bt.metagraph) -> None:
    """Update the mapping of hotkeys to UIDs based on current metagraph state."""
    global hotkey_to_uid_map
    hotkey_to_uid_map = {
        hotkey: uid for uid, hotkey in enumerate(metagraph.hotkeys)
    }


def get_uid_for_hotkey(hotkey: str) -> Optional[int]:
    """Get current UID for a hotkey, returning None if not found."""
    return hotkey_to_uid_map.get(hotkey)


def track_miner_responses(
    responding_uids: Set[int], metagraph: bt.metagraph
) -> None:
    """Update the miner response history with hotkey tracking."""
    global miner_response_history

    # Create response records with both UID and hotkey
    responses = [
        MinerResponse(
            uid=uid, hotkey=metagraph.hotkeys[uid], timestamp=iteration_count
        )
        for uid in responding_uids
    ]
    miner_response_history.appendleft(responses)


def find_last_response(hotkey: str) -> Optional[int]:
    """Find the last response iteration for a given hotkey."""
    for i, responses in enumerate(miner_response_history):
        if any(resp.hotkey == hotkey for resp in responses):
            return i
    return None


def should_apply_decay(uid: int, metagraph: bt.metagraph) -> bool:
    """Determine if decay should be applied based on hotkey history."""
    global miner_response_history, iteration_count

    if not miner_response_history:
        return False

    decay_cycles: int = get_config().alchemy.ma_decay_cycles
    min_history_needed = 255 // decay_cycles

    if iteration_count < min_history_needed:
        return False

    hotkey = metagraph.hotkeys[uid]
    last_seen = find_last_response(hotkey)

    if last_seen is None:
        return True

    return last_seen >= min_history_needed


def adjust_for_miner_count(
    previous_ma_scores: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, bool]:
    """
    Adjust previous moving average scores for changes in miner count by tracking
    hotkey changes at each UID position.

    Args:
        previous_ma_scores: Previous moving average scores tensor
        rewards: Current rewards tensor

    Returns:
        Tuple of (adjusted scores tensor, whether any adjustment was made)
    """
    global previous_hotkeys

    device = get_device()
    metagraph = get_metagraph()

    current_hotkeys: List[str] = metagraph.hotkeys
    adjusted_scores: torch.FloatTensor = torch.zeros(
        len(current_hotkeys),
        device=device,
    )

    was_adjusted: bool = False

    # Handle first run case
    if not previous_hotkeys:
        previous_hotkeys = current_hotkeys
        return previous_ma_scores, False

    # Map previous hotkeys to their scores
    previous_hotkey_to_score: Dict[str, float] = {
        hotkey: score.item()
        for hotkey, score in zip(previous_hotkeys, previous_ma_scores)
    }

    # For each current position, either keep the previous score or use zero
    for i, hotkey in enumerate(current_hotkeys):
        if hotkey in previous_hotkey_to_score:
            # Existing miner - maintain their previous score
            adjusted_scores[i] = previous_hotkey_to_score[hotkey]
        else:
            # New miner - start with zero
            adjusted_scores[i] = 0.0
            was_adjusted = True
            logger.info(
                f"New miner detected at position {i} with hotkey {hotkey}"
            )

    if was_adjusted:
        logger.info(
            "Miners changed. Moving averages adjusted "
            + "to maintain hotkey mapping."
        )

    # Update previous_hotkeys for next iteration
    previous_hotkeys = current_hotkeys

    return adjusted_scores, was_adjusted


async def update_moving_averages(
    previous_ma_scores: torch.FloatTensor,
    scoring_results: ScoringResults,
    alpha: float = MOVING_AVERAGE_ALPHA,
) -> torch.FloatTensor:
    global iteration_count
    iteration_count += 1

    # Section 1: Prepare data and adjust for miner count changes
    metagraph: bt.metagraph = get_metagraph()
    rewards = torch.nan_to_num(
        scoring_results.combined_scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).to(get_device())

    # Update hotkey mapping and adjust scores
    update_hotkey_mapping(metagraph)
    previous_ma_scores, _was_adjusted = adjust_for_miner_count(
        previous_ma_scores,
    )
    previous_ma_scores = previous_ma_scores.to(get_device())

    # Section 2: Calculate new moving averages
    new_ma_scores = alpha * rewards + (1 - alpha) * previous_ma_scores

    # Section 3: Track miner responses and apply decay
    responding_uids = scoring_results.combined_uids
    track_miner_responses(responding_uids, metagraph)
    decay_rate = get_config().alchemy.ma_decay

    # Start with previous scores
    updated_ma_scores = previous_ma_scores.clone()

    # Log responding UIDs and their score changes
    for uid in responding_uids:
        logger.debug(
            f"UID {uid} (hotkey: {metagraph.hotkeys[uid]}) responded - "
            + f"Score updated from {previous_ma_scores[uid]:.4f}"
            + f" to {new_ma_scores[uid]:.4f}"
        )

    # Update scores for users who responded
    updated_ma_scores[responding_uids] = new_ma_scores[responding_uids]

    # Get list of UIDs that should decay
    decaying_uids = torch.tensor(
        [
            uid
            for uid, _ in enumerate(previous_ma_scores)
            if should_apply_decay(uid, metagraph)
        ]
    )

    # Apply decay to those UIDs
    if len(decaying_uids) > 0:
        logger.debug(f"Decay rate: {decay_rate}")
        for uid in decaying_uids:
            old_score = previous_ma_scores[uid]
            new_score = old_score * (1.0 - decay_rate)
            if old_score < 1e-6:
                continue

            logger.debug(
                f"UID {uid} (hotkey: {metagraph.hotkeys[uid]}) decaying. "
                + f"Score updated from {old_score:.4f}"
                + f" to {new_score:.4f}"
            )

        updated_ma_scores[decaying_uids] = previous_ma_scores[decaying_uids] * (
            1.0 - decay_rate
        )

    # Section 4: Log, save, and apply blacklist
    log_moving_averages(updated_ma_scores)
    await save_moving_averages(metagraph.hotkeys, updated_ma_scores)
    final_scores = await apply_blacklist(updated_ma_scores, metagraph)

    return final_scores
