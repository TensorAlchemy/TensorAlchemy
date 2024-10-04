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

    # Adjust alpha based on time elapsed
    # This increases the weight of new scores when more time has passed
    return 1 - (1 - base_alpha) ** block_delta


async def update_moving_averages(
    previous_ma_scores: torch.FloatTensor,
    scoring_results: ScoringResults,
    alpha: Optional[float] = MOVING_AVERAGE_ALPHA,
) -> torch.FloatTensor:
    global block_last_ma_decay
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

    # Apply decay to all scores
    ma_decay = get_config().alchemy.ma_decay
    updated_ma_scores *= 1.0 - ma_decay

    # Update scores for miners who responded
    updated_ma_scores[uids_to_scatter] = new_moving_average_scores[
        uids_to_scatter
    ]

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
