from typing import Dict, List, Optional, Tuple

import torch
import bittensor as bt
from loguru import logger

from neurons.protocol import ModelType
from neurons.validator.config import get_device, get_metagraph
from neurons.validator.rewards.models import (
    EmptyScoreRewardModel,
    BlacklistFilter,
    ModelSimilarityRewardModel,
    HumanValidationRewardModel,
    ImageRewardModel,
    NSFWRewardModel,
)
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
    models: ModelStorage, reward_type: RewardModelType
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
) -> Tuple[torch.Tensor, dict]:
    event = {}
    masking_functions: List[PackedRewardModel] = get_masking_functions(model_type)

    mask = torch.ones(len(get_metagraph().n)).to(get_device())
    uids = torch.tensor(
        [response.dendrite.uid for response in responses], dtype=torch.long
    ).to(get_device())

    for function in masking_functions:
        mask_i, mask_i_normalized = await function.apply(synapse, responses)
        mask[uids] *= mask_i_normalized

        event[function.name] = mask_i.tolist()
        event[function.name + "_normalized"] = mask_i_normalized.tolist()
        logger.info(f"{function.name} {mask_i_normalized.tolist()}")

    return mask, event


async def apply_reward_function(
    reward_function: PackedRewardModel,
    synapse: bt.Synapse,
    responses: list,
    rewards: torch.Tensor,
    event: Optional[Dict] = None,
) -> Tuple[torch.Tensor, dict]:
    if event is None:
        event = {}

    uids = torch.tensor(
        [response.dendrite.uid for response in responses], dtype=torch.long
    ).to(get_device())

    reward_i, reward_i_normalized = await reward_function.apply(synapse, responses)
    rewards[uids] += reward_function.weight * reward_i_normalized

    event[reward_function.name] = reward_i.tolist()
    event[reward_function.name + "_normalized"] = reward_i_normalized.tolist()
    logger.info(f"{reward_function.name}, {reward_i_normalized.tolist()}")

    return rewards, event


async def apply_reward_functions(
    model_type: ModelType,
    synapse: bt.Synapse,
    responses: list,
) -> tuple[torch.Tensor, dict]:
    reward_functions: List[PackedRewardModel] = get_reward_functions(model_type)

    rewards = torch.zeros(len(get_metagraph().n)).to(get_device())
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
