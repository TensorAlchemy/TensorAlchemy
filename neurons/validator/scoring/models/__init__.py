from typing import List

import torch
import bittensor as bt

from neurons.protocol import ModelType

from neurons.validator.scoring.models.empty import EmptyScoreRewardModel

from neurons.validator.scoring.models.masks.nsfw import NSFWRewardModel
from neurons.validator.scoring.models.masks.duplicate import DuplicateFilter
from neurons.validator.scoring.models.masks.blacklist import BlacklistFilter

from neurons.validator.scoring.models.rewards.human import (
    HumanValidationRewardModel,
)
from neurons.validator.scoring.models.rewards.image_reward import (
    ImageRewardModel,
)

from neurons.validator.scoring.models.types import (
    RewardModelType,
    ModelStorage,
    PackedRewardModel,
)


# Init Reward Models
REWARD_MODELS: ModelStorage = None
MASKING_MODELS: ModelStorage = None


def get_reward_models() -> ModelStorage:
    global REWARD_MODELS
    if not REWARD_MODELS:
        REWARD_MODELS = {
            RewardModelType.EMPTY: PackedRewardModel(
                weight=0.0,
                model=EmptyScoreRewardModel(),
            ),
            RewardModelType.IMAGE: PackedRewardModel(
                weight=0.8,
                model=ImageRewardModel(),
            ),
            RewardModelType.HUMAN: PackedRewardModel(
                weight=0.2,
                model=HumanValidationRewardModel(),
            ),
        }

    return REWARD_MODELS


def more_than_one_response(
    _synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> bool:
    return len(responses) > 0


def get_masking_models() -> ModelStorage:
    global MASKING_MODELS
    if not MASKING_MODELS:
        MASKING_MODELS = {
            RewardModelType.NSFW: PackedRewardModel(
                weight=1.0,
                model=NSFWRewardModel(),
            ),
            RewardModelType.BLACKLIST: PackedRewardModel(
                weight=1.0,
                model=BlacklistFilter(),
            ),
            RewardModelType.DUPLICATE: PackedRewardModel(
                weight=1.0,
                model=DuplicateFilter(),
                should_apply=more_than_one_response,
            ),
        }

    return MASKING_MODELS


def get_function(
    models: ModelStorage, reward_type: RewardModelType
) -> PackedRewardModel:
    if reward_type not in models:
        raise ValueError(f"PackedRewardModel {reward_type} not found")
    return models[reward_type]


def get_reward_functions(model_type: ModelType) -> List[PackedRewardModel]:
    if model_type != ModelType.ALCHEMY:
        return [
            get_function(get_reward_models(), RewardModelType.IMAGE),
            get_function(get_reward_models(), RewardModelType.HUMAN),
        ]

    raise NotImplementedError("Alchemy model not yet imlepmented")

    return [
        get_function(get_reward_models(), RewardModelType.IMAGE),
        get_function(get_reward_models(), RewardModelType.HUMAN),
    ]


def get_masking_functions(_model_type: ModelType) -> List[PackedRewardModel]:
    return [
        get_function(get_masking_models(), RewardModelType.NSFW),
        get_function(get_masking_models(), RewardModelType.BLACKLIST),
    ]
