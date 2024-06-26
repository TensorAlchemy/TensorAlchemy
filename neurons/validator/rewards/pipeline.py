from typing import Dict, List, Optional, Tuple

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


def get_uids(responses: List[bt.Synapse]) -> torch.Tensor:
    metagraph: bt.metagraph = get_metagraph()

    return torch.tensor(
        [
            #
            metagraph.hotkeys.index(response.dendrite.hotkey)
            for response in responses
        ],
        dtype=torch.long,
    ).to(get_device())


async def apply_function(
    function: PackedRewardModel,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
    event: Optional[Dict] = None,
) -> Tuple[torch.Tensor, dict]:
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
        f"{function.name}, "
        + f"{summarize_rewards(result_i_normalized)}"
    )

    return result, event


async def apply_functions(
    reward_functions: List[PackedRewardModel],
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> Tuple[torch.Tensor, Dict]:
    event: Dict = {}
    rewards = torch.zeros(get_metagraph().n).to(get_device())

    for function in reward_functions:
        new_rewards, event = await apply_function(
            function,
            synapse,
            responses,
            event,
        )

        rewards *= new_rewards

    return rewards, event


async def apply_reward_functions(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> Tuple[torch.Tensor, Dict]:
    return await apply_functions(
        get_reward_functions(model_type),
        synapse,
        responses,
    )


async def apply_masking_functions(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> Tuple[torch.Tensor, Dict]:
    return await apply_functions(
        get_masking_functions(model_type),
        synapse,
        responses,
    )


async def get_automated_rewards(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
    task_type: str,
) -> AutomatedRewards:
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
    rewards *= mask

    return AutomatedRewards(
        event=event,
        rewards=rewards,
    )


async def get_masked_rewards(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> MaskedRewards:
    """Apply masking functions (NSFW, Blacklist etc.) and return rewards

    Return 0 score if response didn't pass check
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
    for uid, count in isalive_dict.items():
        if count >= isalive_threshold:
            rewards[uid] = 0.0

    return rewards
