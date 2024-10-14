from collections import deque
from typing import List, Optional


import bittensor as bt
import torch
from loguru import logger

from neurons.constants import MOVING_AVERAGE_ALPHA


from neurons.validator.backend.exceptions import PostMovingAveragesError
from neurons.validator.utils import ttl_get_block
from neurons.config import (
    get_config,
    get_device,
    get_metagraph,
    get_backend_client,
    get_blacklist,
)
from scoring.types import (
    ScoringResults,
)

block_last_ma_decay: int = -1

# Store the last 256 iterations of miner responses
miner_response_history = deque(maxlen=256)
# Start decay after ~21 iterations of inactivity
DECAY_THRESHOLD: int = 255 // 12
# Start decay for never-seen miners after 255 iterations
GLOBAL_INACTIVITY_THRESHOLD: int = 255


def log_moving_averages_for_grafana(
    moving_average_scores: torch.FloatTensor,
) -> None:
    list_ma: List[float] = moving_average_scores.tolist()

    for uid in range(moving_average_scores.numel()):
        try:
            score = float(list_ma[uid])
            if score > 0:
                logger.info(
                    f"miner_uid={uid}, miner_score={score:.4f}",
                )
        except IndexError:
            continue

        except Exception as e:
            logger.error(str(e))


def should_apply_decay(uid: int, current_block: int) -> bool:
    global miner_response_history
    if not miner_response_history:
        # Initialize the history with the current block number and an empty set
        miner_response_history.appendleft((current_block, set()))
        logger.info(
            f"Initialized miner response history at block {current_block}"
        )
        return False  # Don't apply decay immediately after initialization

    last_seen = next(
        (block for block, uids in miner_response_history if uid in uids), None
    )

    if last_seen is None:
        # Miner has never been seen
        oldest_block = (
            miner_response_history[-1][0]
            if miner_response_history
            else current_block
        )
        return (current_block - oldest_block) >= GLOBAL_INACTIVITY_THRESHOLD

    return (current_block - last_seen) > DECAY_THRESHOLD


def adjust_alpha_with_time(base_alpha: float, block_delta: int) -> float:
    """
    Adjust the alpha value based on the time elapsed since the last update.

    :param base_alpha: The base alpha value for moving average calculations.
    :param block_delta: The number of blocks that have passed since the last update.
    :return: An adjusted alpha value.
    """
    # Ensure we don't apply a massive change if it's the first update
    if block_delta <= 1:
        return base_alpha

    delta_factor: float = get_config().alchemy.delta_factor

    # Adjust alpha based on time elapsed
    # This increases the weight of new scores when more time has passed
    # Use the DELTA_ADJUSTMENT_FACTOR to control the
    # aggressiveness of the adjustment
    return 1 - (1 - base_alpha) ** (block_delta * delta_factor)


async def update_moving_averages(
    previous_ma_scores: torch.FloatTensor,
    scoring_results: ScoringResults,
    alpha: Optional[float] = MOVING_AVERAGE_ALPHA,
) -> torch.FloatTensor:
    global block_last_ma_decay, miner_response_history
    metagraph: bt.metagraph = get_metagraph()

    rewards = torch.nan_to_num(
        scoring_results.combined_scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).to(get_device())

    # Handle changes in the number of miners
    if rewards.size(0) > previous_ma_scores.size(0):
        logger.info("New miners detected. Adjusting moving averages.")
        new_miners_count = rewards.size(0) - previous_ma_scores.size(0)
        new_miner_scores = torch.zeros(new_miners_count, device=get_device())
        previous_ma_scores = torch.cat([previous_ma_scores, new_miner_scores])
    elif rewards.size(0) < previous_ma_scores.size(0):
        logger.warning(
            "Fewer miners than expected. Truncating moving averages."
        )
        previous_ma_scores = previous_ma_scores[: rewards.size(0)]

    # Calculate the time elapsed since last update
    block_now: int = ttl_get_block()
    block_delta: int = max(1, block_now - block_last_ma_decay)
    block_last_ma_decay = block_now

    # Adjust alpha based on time elapsed
    adjusted_alpha = adjust_alpha_with_time(alpha, block_delta)

    # Apply the moving average update with adjusted alpha
    new_moving_average_scores = adjusted_alpha * rewards + (
        1 - adjusted_alpha
    ) * previous_ma_scores.to(get_device())

    # Prepare to update scores
    updated_ma_scores = previous_ma_scores.clone()
    uids_to_scatter: torch.Tensor = scoring_results.combined_uids.to(torch.long)
    logger.info(f"Scattering MA deltas over UIDS {uids_to_scatter}")

    # Update miner response history
    miner_response_history.appendleft(set(uids_to_scatter.tolist()))

    # Apply decay to all scores
    ma_decay = get_config().alchemy.ma_decay
    current_block = ttl_get_block()

    for uid in range(len(updated_ma_scores)):
        if uid in uids_to_scatter:
            # Update scores for miners who responded
            updated_ma_scores[uid] = new_moving_average_scores[uid]
            continue

        if should_apply_decay(uid, current_block):
            # decay for inactive miners
            updated_ma_scores[uid] *= 1.0 - ma_decay

    # Log moving averages for monitoring
    log_moving_averages_for_grafana(updated_ma_scores)

    # Save moving averages scores on backend
    try:
        await get_backend_client().post_moving_averages(
            metagraph.hotkeys,
            updated_ma_scores,
        )
    except PostMovingAveragesError as e:
        logger.error(f"Failed to post moving averages: {e}")

    # Apply blacklist
    try:
        hotkey_blacklist, coldkey_blacklist = await get_blacklist()
        for i, (hotkey, coldkey) in enumerate(
            zip(metagraph.hotkeys, metagraph.coldkeys)
        ):
            if hotkey in hotkey_blacklist or coldkey in coldkey_blacklist:
                updated_ma_scores[i] = 0
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while applying blacklist: {e}"
        )

    return updated_ma_scores
