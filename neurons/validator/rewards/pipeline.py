from typing import Callable, Dict, List, Optional, Tuple

import torch
import bittensor as bt
from loguru import logger

from neurons.protocol import ModelType
from neurons.utils.log import summarize_rewards
from neurons.validator.config import get_device, get_metagraph

from neurons.validator.rewards.models import (
    PackedRewardModel,
    get_reward_functions,
    get_masking_functions,
)
from neurons.validator.rewards.types import (
    MaskedRewards,
    AutomatedRewards,
)


async def apply_function(
    function: PackedRewardModel,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
    event: Optional[Dict] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Apply a single reward or masking function and log its results.

    This function serves as a standardized way to apply
    any reward or masking function in our pipeline.

    It encapsulates the common logic of applying a function,
    weighting its results,
    and logging the outcomes.

    This standardization simplifies the process of adding new reward
    or masking functions and ensures consistent handling
    and logging across all functions.
    """
    if event is None:
        event = {}

    result_i, result_i_normalized = await function.apply(
        synapse,
        responses,
    )
    result = function.weight * result_i_normalized

    event[function.name] = {}
    event[function.name]["score"] = result_i.tolist()
    event[function.name]["normalized"] = result_i_normalized.tolist()

    logger.info(
        #
        f"{function.name} - {summarize_rewards(result_i_normalized)}"
    )

    return result, event


ResultCombiner = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


async def apply_functions(
    functions: List[PackedRewardModel],
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
    initial_seed: torch.Tensor,
    combine: ResultCombiner,
) -> Tuple[torch.Tensor, Dict]:
    """
    Apply a list of reward or masking functions sequentially.

    This function orchestrates the application of multiple reward or masking
    functions.

    It's designed to handle both reward and masking scenarios.

    The sequential application enables complex reward schemes where
    the final reward is a product of multiple factors.
    """
    event: Dict = {}
    results = initial_seed

    for function in functions:
        rewards, event = await apply_function(
            function,
            synapse,
            responses,
            event,
        )

        # Use our passed function to combine results
        # this allows us different types of combination
        # depending on if it's a mask or reward
        results = combine(results, rewards)

    return results, event


async def apply_reward_functions(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> Tuple[torch.Tensor, Dict]:
    """
    Apply all relevant reward functions for a given model type.
    """
    return await apply_functions(
        get_reward_functions(model_type),
        synapse,
        responses,
        combine=lambda results, rewards: results * rewards,
        initial_seed=torch.ones(get_metagraph().n).to(get_device()),
    )


async def apply_masking_functions(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> Tuple[torch.Tensor, Dict]:
    """
    Apply all relevant masking functions for a given model type.
    """
    return await apply_functions(
        get_masking_functions(model_type),
        synapse,
        responses,
        combine=torch.maximum,
        initial_seed=torch.zeros(get_metagraph().n).to(get_device()),
    )


async def get_automated_rewards(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
    task_type: str,
) -> AutomatedRewards:
    """
    Calculate the final automated rewards for a set of responses.

    This function is the core of the automated reward calculation process.

    It combines the results of both reward and masking functions
    to produce a final reward for each response.

    This approach allows for a comprehensive evaluation of responses,
    taking into account both their quality (via reward functions) and
    their appropriateness (via masking functions).
    """
    event = {"task_type": task_type}

    # Apply reward functions (including human voting)
    rewards, reward_event = await apply_reward_functions(
        model_type,
        synapse,
        responses,
    )
    event.update(reward_event)

    # Apply masking functions
    mask, masking_event = await apply_masking_functions(
        model_type,
        synapse,
        responses,
    )
    event.update(masking_event)

    # Apply mask to rewards
    # NOTE: If mask is (1) that means we had a trigger
    #       so we want to reduce score by the effect
    #       of the mask.
    #
    #       A mask value of 1 means "failed a check",
    #       so we multiply by (1 - mask)
    rewards *= 1.0 - mask

    return AutomatedRewards(
        event=event,
        rewards=rewards,
    )


async def get_masked_rewards(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> MaskedRewards:
    """
    Apply only the masking functions to a set of responses.

    This function is used when we need to quickly filter responses
    without calculating full rewards.

    It's particularly useful for scenarios where we need to exclude
    inappropriate content before further processing or when we want to
    separate the filtering step from the reward calculation step for more
    fine-grained control over the process.
    """
    rewards, event = await apply_masking_functions(
        model_type,
        synapse,
        responses,
    )

    return MaskedRewards(rewards=rewards, event=event)


def filter_rewards(
    isalive_dict: Dict[int, int],
    isalive_threshold: int,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """
    Adjust rewards based on the 'isalive' status of miners.

    This function is crucial for maintaining the health and fairness of
    the network.

    By zeroing out rewards for miners that have exceeded the
    'isalive' threshold, we prevent overactive or potentially
    malicious miners from dominating the reward distribution.

    This helps in maintaining a balanced and diverse
    set of active miners in the network.
    """
    for uid, count in isalive_dict.items():
        if count >= isalive_threshold:
            rewards[uid] = 0.0

    return rewards
