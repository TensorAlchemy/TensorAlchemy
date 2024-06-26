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


async def apply_masking_functions(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> Tuple[torch.Tensor, dict]:
    event = {}
    masking_functions: List[PackedRewardModel] = get_masking_functions(model_type)

    mask = torch.ones(get_metagraph().n).to(get_device())

    for mask_function in masking_functions:
        mask_i, mask_i_normalized = await mask_function.apply(
            synapse,
            responses,
        )
        mask *= mask_i_normalized

        event[mask_function.name] = {}
        event[mask_function.name]["score"] = mask_i.tolist()
        event[mask_function.name]["normalized"] = mask_i_normalized.tolist()

        logger.info(
            #
            f"{mask_function.name}, "
            + f"{summarize_rewards(mask_i_normalized)}"
        )

    return mask, event


async def apply_reward_function(
    reward_function: PackedRewardModel,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
    rewards: torch.Tensor,
    event: Optional[Dict] = None,
) -> Tuple[torch.Tensor, dict]:
    if event is None:
        event = {}

    reward_i, reward_i_normalized = await reward_function.apply(
        synapse,
        responses,
    )
    rewards += reward_function.weight * reward_i_normalized

    event[reward_function.name] = {}
    event[reward_function.name]["score"] = reward_i.tolist()
    event[reward_function.name]["normalized"] = reward_i_normalized.tolist()

    logger.info(
        #
        f"{reward_function.name}, "
        + f"{summarize_rewards(reward_i_normalized)}"
    )

    return rewards, event


async def apply_reward_functions(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> tuple[torch.Tensor, dict]:
    reward_functions: List[PackedRewardModel] = get_reward_functions(model_type)

    rewards = torch.zeros(get_metagraph().n).to(get_device())
    event: Dict = {}
    for function in reward_functions:
        rewards, event = await apply_reward_function(
            function,
            synapse,
            responses,
            rewards,
            event,
        )

    return rewards, event


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
