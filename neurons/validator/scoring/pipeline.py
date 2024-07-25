from typing import Callable, Dict, List

import torch
import bittensor as bt
from loguru import logger

from neurons.protocol import ModelType
from neurons.utils.log import summarize_rewards
from neurons.validator.config import get_device, get_metagraph

from neurons.validator.scoring.models.types import PackedRewardModel
from neurons.validator.scoring.models import (
    get_reward_functions,
    get_masking_functions,
)
from neurons.validator.scoring.types import (
    ScoringResult,
    ScoringResults,
)

ResultCombiner = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


async def apply_function(
    initial_seed: torch.Tensor,
    function: PackedRewardModel,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> ScoringResult:
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
    result: ScoringResult = await function.apply(
        synapse,
        responses,
    )

    logger.info(
        #
        function.name
        + f" - {summarize_rewards(result.scores)}"
    )

    # Build up a new score instead of re-using the one above
    return ScoringResult(
        type=result.type,
        # Normalization of scores
        # [ 0.0, 0.2, 0.4, 1.0 ]
        normalized=result.normalized,
        # Apply weighting to final score
        # We also apply initial seed here
        # This prevents values from dropping to zero
        # if one reward function fails when doing multiply combine
        #
        # human = 0.0 (because no votes)
        # image = 1.6 (because good image)
        # So we need to keep the scores of previous
        # pipeline items around 1.0 to retain actual score
        #
        # reward model -> [1.0, 1.2, 1.4, 2.0]
        # mask model -> [0.0, 0.2, 0.4, 1.0]
        scores=initial_seed + function.weight * result.normalized,
    )


async def apply_functions(
    initial_seed: torch.Tensor,
    functions: List[PackedRewardModel],
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
    combine: ResultCombiner,
) -> ScoringResults:
    """
    Apply a list of reward or masking functions sequentially.

    This function orchestrates the application of multiple reward or masking
    functions.

    It's designed to handle both reward and masking scenarios.

    The sequential application enables complex reward schemes where
    the final reward is a product of multiple factors.
    """
    results: ScoringResults = ScoringResults(combined_scores=initial_seed)

    for function in functions:
        if not function.should_apply(synapse, responses):
            continue

        reward: ScoringResult = await apply_function(
            initial_seed,
            function,
            synapse,
            responses,
        )

        # Use our passed function to combine results
        # this allows us different types of combination
        # depending on if it's a mask or reward
        results.combined_scores = combine(
            results.combined_scores,
            reward.scores,
        )

        # And add it to the list for later
        results.add_score(reward)

    return results


async def apply_reward_functions(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> ScoringResults:
    """
    Apply all relevant reward functions for a given model type.
    """
    initial_seed: torch.Tensor = torch.ones(
        get_metagraph().n,
    ).to(get_device())

    return await apply_functions(
        initial_seed,
        get_reward_functions(model_type),
        synapse,
        responses,
        combine=lambda results, rewards: results * rewards,
    )


async def apply_masking_functions(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> ScoringResults:
    """
    Apply all relevant masking functions for a given model type.
    """
    initial_seed: torch.Tensor = torch.zeros(
        get_metagraph().n,
    ).to(get_device())

    return await apply_functions(
        initial_seed,
        get_masking_functions(model_type),
        synapse,
        responses,
        combine=torch.maximum,
    )


async def get_scoring_results(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> ScoringResults:
    """
    Calculate the final automated rewards for a set of responses.

    This function is the core of the automated reward calculation process.

    It combines the results of both reward and masking functions
    to produce a final reward for each response.

    This approach allows for a comprehensive evaluation of responses,
    taking into account both their quality (via reward functions) and
    their appropriateness (via masking functions).
    """
    # Apply reward functions (including human voting)
    rewards: ScoringResults = await apply_reward_functions(
        model_type,
        synapse,
        responses,
    )

    # Apply masking functions
    masks: ScoringResults = await apply_masking_functions(
        model_type,
        synapse,
        responses,
    )

    combined_scores: torch.Tensor = rewards.combined_scores

    # Reset to be zero-centric
    combined_scores -= 1.0

    # Apply mask to rewards
    # NOTE: If mask is (1) that means we had a trigger
    #       so we want to reduce score by the effect
    #       of the mask.
    #
    #       A mask value of 1 means "failed a check",
    #       so we'll set those scores to 0.0.
    #
    #       0.0 here means "don't change the weights"
    combined_scores[masks.combined_scores != 0] = 0.0

    return ScoringResults(
        # Simple list concatenation
        scores=rewards.scores + masks.scores,
        # And the actual result scores
        combined_scores=combined_scores,
    )


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
