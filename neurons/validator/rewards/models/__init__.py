from typing import Dict, List, Tuple

import torch
from pydantic import BaseModel

from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.models.empty import EmptyScoreRewardModel
from neurons.validator.rewards.models.blacklist import BlacklistFilter
from neurons.validator.rewards.models.similarity import ModelSimilarityRewardModel
from neurons.validator.rewards.models.human import HumanValidationRewardModel
from neurons.validator.rewards.models.image_reward import ImageRewardModel
from neurons.validator.rewards.models.nsfw import NSFWRewardModel

from neurons.validator.rewards.types import RewardModelType, ModelType


class PackedRewardModel(BaseModel):
    weight: float
    model: BaseRewardModel

    class Config:
        arbitrary_types_allowed = True

    @property
    def name(self) -> RewardModelType:
        return self.model.name

    def apply(
        self, *args, **kwargs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor,]:
        return self.model.apply(*args, **kwargs)


ModelStorage = Dict[RewardModelType, PackedRewardModel]

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
            RewardModelType.SIMILARITY: PackedRewardModel(
                weight=0.2,
                model=ModelSimilarityRewardModel(),
            ),
            RewardModelType.HUMAN: PackedRewardModel(
                weight=0.1 / 32,
                model=HumanValidationRewardModel(),
            ),
        }

    return REWARD_MODELS


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
    return [
        get_function(get_reward_models(), RewardModelType.IMAGE),
        get_function(get_reward_models(), RewardModelType.SIMILARITY),
        get_function(get_reward_models(), RewardModelType.HUMAN),
    ]


def get_masking_functions(_model_type: ModelType) -> List[PackedRewardModel]:
    return [
        get_function(get_masking_models(), RewardModelType.NSFW),
        get_function(get_masking_models(), RewardModelType.BLACKLIST),
    ]
