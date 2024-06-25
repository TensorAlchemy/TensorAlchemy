from typing import Dict, List, Optional

import torch
import bittensor as bt
from loguru import logger

from neurons.protocol import ModelType
from neurons.validator.config import get_device
from neurons.validator.rewards.models.blacklist import BlacklistFilter
from neurons.validator.rewards.models.diversity import ModelDiversityRewardModel
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
    RewardModelType.IMAGE: PackedRewardModel(
        weight=0.8,
        model=ImageRewardModel(),
    ),
    RewardModelType.DIVERSITY: PackedRewardModel(
        weight=0.2,
        model=ModelDiversityRewardModel(),
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


def get_model(
    models: ModelStorage,
    reward_type: RewardModelType,
) -> PackedRewardModel:
    if reward_type not in models:
        raise ValueError(f"PackedRewardModel {reward_type} not found")

    return models[reward_type]


def get_reward_functions(model_type: ModelType) -> List[PackedRewardModel]:
    if model_type != ModelType.ALCHEMY:
        return [
            get_model(REWARD_MODELS, RewardModelType.IMAGE),
            get_model(REWARD_MODELS, RewardModelType.HUMAN),
        ]

    return [
        get_model(REWARD_MODELS, RewardModelType.IMAGE),
        get_model(REWARD_MODELS, RewardModelType.DIVERSITY),
        get_model(REWARD_MODELS, RewardModelType.HUMAN),
    ]


def get_masking_functions(_model_type: ModelType) -> List[PackedRewardModel]:
    return [
        get_model(MASKING_MODELS, RewardModelType.NSFW),
        get_model(MASKING_MODELS, RewardModelType.BLACKLIST),
    ]


async def apply_masking_functions(
    model_type: ModelType,
    synapse: bt.synapse,
    responses: list,
    rewards: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    event = {}
    masking_functions: List[PackedRewardModel] = get_masking_functions(
        model_type,
    )

    for function in masking_functions:
        mask_i, mask_i_normalized = await function.apply(
            synapse,
            responses,
            rewards,
        )

        rewards *= mask_i_normalized.to(get_device())
        event[function.name] = mask_i.tolist()
        event[function.name + "_normalized"] = mask_i_normalized.tolist()
        logger.info(f"{function.name} {mask_i_normalized.tolist()}")

    return rewards, event


async def apply_reward_function(
    reward_function: PackedRewardModel,
    synapse: bt.synapse,
    responses: list,
    rewards: torch.Tensor,
    event: Optional[Dict] = None,
) -> tuple[torch.Tensor, dict]:
    if event is None:
        event = {}

    reward_i, reward_i_normalized = await reward_function.apply(
        synapse,
        responses,
        rewards,
    )

    rewards += reward_function.weight * reward_i_normalized.to(get_device())
    event[reward_function.name] = reward_i.tolist()
    event[reward_function.name + "_normalized"] = reward_i_normalized.tolist()
    logger.info(f"{reward_function.name}, {reward_i_normalized.tolist()}")

    return rewards, event


async def apply_reward_functions(
    model_type: ModelType,
    synapse: bt.synapse,
    responses: list,
    rewards: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    reward_functions: List[PackedRewardModel] = get_reward_functions(model_type)

    event: Dict = {}
    for function in reward_functions:
        rewards, event = apply_reward_function(
            function,
            synapse,
            responses,
            rewards,
        )

    return rewards, event


async def get_automated_rewards(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
    task_type: str,
) -> AutomatedRewards:
    event = {"task_type": task_type}

    # Initialize rewards tensor
    rewards: torch.Tensor = torch.zeros(
        len(responses),
        dtype=torch.float32,
    ).to(get_device())

    # Apply reward functions (including human voting)
    rewards, reward_event = await apply_reward_functions(
        model_type,
        synapse,
        responses,
        rewards,
    )
    event.update(reward_event)

    # Apply masking functions
    rewards, masking_event = await apply_masking_functions(
        model_type,
        synapse,
        responses,
        rewards,
    )
    event.update(masking_event)

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
        torch.ones(len(responses)).to(get_device()),
    )

    return MaskedRewards(rewards=rewards, event=event)


def filter_rewards(
    isalive_dict: Dict[int, int],
    isalive_threshold: int,
    rewards: torch.Tensor,
) -> torch.Tensor:
    for uid, count in isalive_dict.items():
        if count >= isalive_threshold:
            rewards[uid] = 0

    return rewards
