from typing import List

from neurons.protocol import ModelType
from scoring.models.types import (
    ModelStorage,
    PackedRewardModel,
    RewardModelType,
)

import bittensor as bt

# Init Reward Models
REWARD_MODELS: ModelStorage = None
MASKING_MODELS: ModelStorage = None


def get_reward_models() -> ModelStorage:
    from scoring.models.empty import EmptyScoreRewardModel
    from scoring.models.rewards.enhanced_clip import EnhancedClipRewardModel
    from scoring.models.rewards.human import HumanValidationRewardModel
    from scoring.models.rewards.image_reward import ImageRewardModel

    global REWARD_MODELS
    if not REWARD_MODELS:
        REWARD_MODELS = {
            RewardModelType.EMPTY: PackedRewardModel(
                weight=0.0,
                model=EmptyScoreRewardModel(),
            ),
            RewardModelType.ENHANCED_CLIP: PackedRewardModel(
                weight=0.12,
                model=EnhancedClipRewardModel(),
            ),
            RewardModelType.HUMAN: PackedRewardModel(
                weight=0.20,
                model=HumanValidationRewardModel(),
            ),
            RewardModelType.IMAGE: PackedRewardModel(
                weight=0.68,
                model=ImageRewardModel(),
            ),
        }

    return REWARD_MODELS


def should_check_duplicates(
    synapse: bt.Synapse,
    responses: List[bt.Synapse],
) -> bool:
    if synapse.seed > -1:
        return False

    return len(responses) > 1


def get_masking_models() -> ModelStorage:
    from scoring.models.masks.blacklist import BlacklistFilter
    from scoring.models.masks.duplicate import DuplicateFilter
    from scoring.models.masks.nsfw import NSFWRewardModel

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
                should_apply=should_check_duplicates,
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
    if model_type == ModelType.ALCHEMY:
        raise NotImplementedError("Alchemy model not yet imlepmented")

    if model_type == ModelType.SCORING:
        return [
            get_function(get_reward_models(), RewardModelType.ENHANCED_CLIP),
            get_function(get_reward_models(), RewardModelType.IMAGE),
        ]

    return [
        get_function(get_reward_models(), RewardModelType.ENHANCED_CLIP),
        get_function(get_reward_models(), RewardModelType.IMAGE),
        get_function(get_reward_models(), RewardModelType.HUMAN),
    ]


def get_masking_functions(model_type: ModelType) -> List[PackedRewardModel]:
    if model_type == ModelType.ALCHEMY:
        raise NotImplementedError("Alchemy model not yet imlepmented")

    if model_type == ModelType.SCORING:
        return []

    return [
        get_function(get_masking_models(), RewardModelType.NSFW),
        get_function(get_masking_models(), RewardModelType.BLACKLIST),
        get_function(get_masking_models(), RewardModelType.DUPLICATE),
    ]
