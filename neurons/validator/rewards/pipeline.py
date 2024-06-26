from typing import Dict, List, Optional

import torch
import bittensor as bt
from loguru import logger

from neurons.protocol import ModelType
from neurons.validator.config import get_device
from neurons.validator.rewards.models.empty import EmptyScoreRewardModel
from neurons.validator.rewards.models.blacklist import BlacklistFilter
from neurons.validator.rewards.models.similarity import ModelSimilarityRewardModel
from neurons.validator.rewards.models.human import HumanValidationRewardModel
from neurons.validator.rewards.models.image_reward import ImageRewardModel
from neurons.validator.rewards.models.nsfw import NSFWRewardModel
from neurons.validator.rewards.types import (
    MaskedRewards,
    PackedRewardModel,
    RewardModelType,
    AutomatedRewards,
)

ModelStorage = Dict[RewardModelType, PackedRewardModel]

# Init Reward Models
REWARD_MODELS: ModelStorage = {
    RewardModelType.EMPTY: PackedRewardModel(
        weight=0.0,
        model=EmptyScoreRewardModel(),
    ),
    RewardModelType.IMAGE: PackedRewardModel(
        weight=0.8,
        model=ImageRewardModel(),
    ),
    RewardModelType.SIMILARITY: PackedRewardModel(
        weight=0.2,
        model=ModelSimilarityRewardModel(),
    ),
    RewardModelType.HUMAN: PackedRewardModel(
        weight=0.1 / 32,
        model=HumanValidationRewardModel(),
    ),
}

MASKING_MODELS: ModelStorage = {
    RewardModelType.NSFW: PackedRewardModel(
        weight=1.0,
        model=NSFWRewardModel(),
    ),
    RewardModelType.BLACKLIST: PackedRewardModel(
        weight=1.0,
        model=BlacklistFilter(),
    ),
}


def get_function(
    models: ModelStorage,
    reward_type: RewardModelType,
) -> PackedRewardModel:
    if reward_type not in models:
        raise ValueError(f"PackedRewardModel {reward_type} not found")

    return models[reward_type]


def get_reward_functions(model_type: ModelType) -> List[PackedRewardModel]:
    if model_type != ModelType.ALCHEMY:
        return [
            get_function(REWARD_MODELS, RewardModelType.IMAGE),
            get_function(REWARD_MODELS, RewardModelType.HUMAN),
        ]

    return [
        get_function(REWARD_MODELS, RewardModelType.IMAGE),
        get_function(REWARD_MODELS, RewardModelType.SIMILARITY),
        get_function(REWARD_MODELS, RewardModelType.HUMAN),
    ]


def get_masking_functions(_model_type: ModelType) -> List[PackedRewardModel]:
    return [
        get_function(MASKING_MODELS, RewardModelType.NSFW),
        get_function(MASKING_MODELS, RewardModelType.BLACKLIST),
    ]


async def apply_masking_functions(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: list,
) -> tuple[Dict[int, float], dict]:
    event = {}
    masking_functions: List[PackedRewardModel] = get_masking_functions(model_type)

    mask = {response.dendrite.uid: 1.0 for response in responses}

    for function in masking_functions:
        mask_i, mask_i_normalized = await function.apply(synapse, responses)

        for uid in mask:
            if uid in mask_i:
                mask[uid] *= mask_i_normalized[uid]

        event[function.name] = mask_i
        event[function.name + "_normalized"] = mask_i_normalized
        logger.info(f"{function.name} {mask_i_normalized}")

    return mask, event


async def apply_reward_function(
    reward_function: PackedRewardModel,
    synapse: bt.Synapse,
    responses: list,
    rewards: Dict[int, float],
    event: Optional[Dict] = None,
) -> tuple[Dict[int, float], dict]:
    if event is None:
        event = {}

    reward_i, reward_i_normalized = await reward_function.apply(synapse, responses)

    for uid in rewards:
        if uid in reward_i_normalized:
            rewards[uid] += reward_function.weight * reward_i_normalized[uid]

    event[reward_function.name] = reward_i
    event[reward_function.name + "_normalized"] = reward_i_normalized
    logger.info(f"{reward_function.name}, {reward_i_normalized}")

    return rewards, event


async def apply_reward_functions(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: list,
) -> tuple[Dict[int, float], dict]:
    reward_functions: List[PackedRewardModel] = get_reward_functions(model_type)

    rewards = {response.dendrite.uid: 0.0 for response in responses}
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
    for uid in rewards:
        rewards[uid] *= mask.get(uid, 0.0)

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
    rewards: Dict[int, float],
) -> Dict[int, float]:
    for uid, count in isalive_dict.items():
        if count >= isalive_threshold:
            rewards[uid] = 0.0

    return rewards
