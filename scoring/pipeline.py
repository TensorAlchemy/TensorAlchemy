from typing import Callable, List

import torch
from loguru import logger

from neurons.config import get_device, get_metagraph
from neurons.protocol import BaseTask
from neurons.utils.log import summarize_rewards
from scoring.models import get_masking_functions, get_reward_functions
from scoring.models.types import PackedRewardModel
from scoring.types import ScoringResult, ScoringResults, combine_uids

ResultCombiner = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


async def apply_function(
    initial_seed: torch.Tensor,
    function: PackedRewardModel,
    synapse: BaseTask,
    responses: List[BaseTask],
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
        + f": {summarize_rewards(result.scores)}"
    )

    # Build up a new score instead of re-using the one above
    return ScoringResult(
        type=result.type,
        # We'll keep track of these to be able
        # to scatter the rewards across moving_averages later
        uids=result.uids,
        # Raw scores
        raw=result.scores,
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
    synapse: BaseTask,
    responses: List[BaseTask],
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
    synapse: BaseTask,
    responses: List[BaseTask],
) -> ScoringResults:
    """
    Apply all relevant reward functions for a given model type.
    """
    initial_seed: torch.Tensor = torch.ones(
        get_metagraph().n,
    ).to(get_device())

    return await apply_functions(
        initial_seed,
        get_reward_functions(synapse.task_type),
        synapse,
        responses,
        combine=lambda results, rewards: results * rewards,
    )


async def apply_masking_functions(
    synapse: BaseTask,
    responses: List[BaseTask],
) -> ScoringResults:
    """
    Apply all relevant masking functions for a given model type.
    """
    initial_seed: torch.Tensor = torch.zeros(
        get_metagraph().n,
    ).to(get_device())

    return await apply_functions(
        initial_seed,
        get_masking_functions(synapse.task_type),
        synapse,
        responses,
        combine=torch.maximum,
    )


async def get_scoring_results(
    synapse: BaseTask,
    responses: List[BaseTask],
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
        synapse,
        responses,
    )

    # Apply masking functions
    masks: ScoringResults = await apply_masking_functions(
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
    combined_scores[masks.combined_scores >= 1e-6] = 0.0

    return ScoringResults(
        # Simple list concatenation
        scores=rewards.scores + masks.scores,
        # And the actual result scores
        combined_scores=combined_scores,
        # And the combined UIDs
        combined_uids=combine_uids(rewards.combined_uids, masks.combined_uids),
    )
