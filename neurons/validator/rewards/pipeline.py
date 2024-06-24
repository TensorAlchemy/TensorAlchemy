from typing import Dict, List

import torch
import bittensor as bt
from loguru import logger

from neurons.protocol import ModelType
from neurons.validator.config import get_device
from neurons.validator.backend.client import TensorAlchemyBackendClient
from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.models.blacklist import BlacklistFilter
from neurons.validator.rewards.models.diversity import ModelDiversityRewardModel
from neurons.validator.rewards.models.human import HumanValidationRewardModel
from neurons.validator.rewards.models.image_reward import ImageRewardModel
from neurons.validator.rewards.models.nsfw import NSFWRewardModel
from neurons.validator.rewards.types import (
    PackedRewardModel,
    RewardModelType,
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
        ]

    return [
        get_model(REWARD_MODELS, RewardModelType.IMAGE),
        get_model(REWARD_MODELS, RewardModelType.DIVERSITY),
    ]


def get_masking_functions(_model_type: ModelType) -> List[PackedRewardModel]:
    return [
        get_model(MASKING_MODELS, RewardModelType.NSFW),
        get_model(MASKING_MODELS, RewardModelType.BLACKLIST),
    ]


async def apply_masking_functions(
    model_type: ModelType,
    responses: list,
    rewards: torch.Tensor,
    synapse: bt.synapse,
) -> tuple[torch.Tensor, dict]:
    event = {}

    masking_functions: List[PackedRewardModel] = get_masking_functions(
        model_type,
    )

    for function in masking_functions:
        mask_i, mask_i_normalized = await function.model.apply(
            responses,
            rewards,
        )

        rewards *= mask_i_normalized.to(get_device())

        event[function.name] = mask_i.tolist()
        event[function.name + "_normalized"] = mask_i_normalized.tolist()
        logger.info(f"{function.name} {mask_i_normalized.tolist()}")

    return rewards, event


async def apply_reward_functions(
    model_type: ModelType,
    responses: list,
    rewards: torch.Tensor,
    synapse: bt.synapse,
) -> tuple[torch.Tensor, dict]:
    event = {}
    reward_functions: List[PackedRewardModel] = get_reward_functions(
        model_type,
    )

    for function in reward_functions:
        reward_i, reward_i_normalized = await function.model.apply(
            responses,
            rewards,
            synapse,
        )

        rewards += function.weight * reward_i_normalized.to(get_device())
        event[function.name] = reward_i.tolist()
        event[function.name + "_normalized"] = reward_i_normalized.tolist()

        logger.info(f"{function.name}, {reward_i_normalized.tolist()}")

    return rewards, event
