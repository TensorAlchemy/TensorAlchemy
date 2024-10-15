"""
moving_averages.py

Handles calculation and updating of moving averages for miner scores in a
distributed network. Includes functionality for score decay based on
iterations, and miner inactivity handling.

Key functions:
- update_moving_averages: Main function for updating moving averages
- track_miner_responses: Tracks miner response history
- should_apply_decay: Determines if decay should be applied based on iterations
- apply_decay: Applies decay to inactive miners' scores
- log_moving_averages: Logs moving averages for monitoring
- apply_blacklist: Applies blacklist to moving averages

Usage:
    new_ma_scores = await update_moving_averages(previous_ma_scores, scoring_results)
"""

from collections import deque
from typing import List, Set, Dict

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

# Global variables
miner_response_history: deque = deque(maxlen=256)
iteration_count: int = 0


def track_miner_responses(responding_uids: Set[int]) -> None:
    """Update the miner response history."""
    global miner_response_history
    miner_response_history.appendleft(responding_uids)


def should_apply_decay(uid: int) -> bool:
    """Determine if decay should be applied to a miner's score based on iterations."""
    global miner_response_history, iteration_count
    if not miner_response_history:
        return False

    last_seen = next(
        (i for i, uids in enumerate(miner_response_history) if uid in uids),
        None,
    )
    if last_seen is None:
        return iteration_count >= len(miner_response_history)

    delay_cycles: int = get_config().alchemy.ma_delay_cycles
    return last_seen >= (255 // delay_cycles)


def apply_decay(
    scores: torch.FloatTensor, decay_rate: float
) -> torch.FloatTensor:
    """Apply decay to inactive miners' scores."""
    return torch.tensor(
        [
            score * (1.0 - decay_rate) if should_apply_decay(uid) else score
            for uid, score in enumerate(scores)
        ]
    )


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


def adjust_for_miner_count(
    previous_ma_scores: torch.FloatTensor, rewards: torch.FloatTensor
) -> torch.FloatTensor:
    """Adjust previous moving average scores for changes in miner count."""
    if rewards.size(0) > previous_ma_scores.size(0):
        logger.info("New miners detected. Adjusting moving averages.")
        new_miners_count = rewards.size(0) - previous_ma_scores.size(0)
        return torch.cat(
            [
                previous_ma_scores,
                torch.zeros(new_miners_count, device=get_device()),
            ]
        )
    elif rewards.size(0) < previous_ma_scores.size(0):
        logger.warning(
            "Fewer miners than expected. Truncating moving averages."
        )
        return previous_ma_scores[: rewards.size(0)]
    return previous_ma_scores


async def save_moving_averages(
    hotkeys: List[str], scores: torch.FloatTensor
) -> None:
    """Save moving average scores to the backend."""
    try:
        await get_backend_client().post_moving_averages(hotkeys, scores)
    except PostMovingAveragesError as e:
        logger.error(f"Failed to post moving averages: {e}")


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

    previous_ma_scores = adjust_for_miner_count(previous_ma_scores, rewards).to(
        get_device()
    )

    # Section 2: Calculate new moving averages
    new_ma_scores = alpha * rewards + (1 - alpha) * previous_ma_scores

    # Section 3: Track miner responses and apply decay
    responding_uids = set(
        scoring_results.combined_uids.to(torch.long).tolist(),
    )

    track_miner_responses(responding_uids)
    decay_rate = get_config().alchemy.ma_decay
    updated_ma_scores = torch.tensor(
        [
            new_ma_scores[uid]
            if uid in responding_uids
            else score * (1.0 - decay_rate)
            if should_apply_decay(uid)
            else score
            for uid, score in enumerate(previous_ma_scores)
        ]
    )

    # Section 4: Log, save, and apply blacklist
    log_moving_averages(updated_ma_scores)
    await save_moving_averages(metagraph.hotkeys, updated_ma_scores)
    final_scores = await apply_blacklist(updated_ma_scores, metagraph)

    return final_scores
