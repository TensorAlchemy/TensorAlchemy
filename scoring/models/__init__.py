from typing import Callable, List

import bittensor as bt

from neurons.protocol import BaseTask, ModelType, TaskType
from scoring.models.rewards.image_reward import ImageRewardModel
from scoring.models.types import (
    ModelStorage,
    PackedRewardModel,
    RewardModelType,
)

# Init Reward Models
REWARD_MODELS: ModelStorage = None
MASKING_MODELS: ModelStorage = None


def get_inpainting_models() -> ModelStorage:
    from scoring.models.rewards.inpainting.boundary_coherence import (
        BoundaryCoherenceModel,
    )
    from scoring.models.rewards.inpainting.mask_adherence import (
        MaskAdherenceModel,
    )
    from scoring.models.rewards.inpainting.structure_consistency import (
        StructureConsistencyModel,
    )
    from scoring.models.rewards.inpainting.semantic_consistency import (
        SemanticConsistencyModel,
    )

    global INPAINTING_MODELS
    if not INPAINTING_MODELS:
        INPAINTING_MODELS = {
            RewardModelType.BOUNDARY_COHERENCE: PackedRewardModel(
                weight=0.25,
                model=BoundaryCoherenceModel(),
            ),
            RewardModelType.MASK_ADHERENCE: PackedRewardModel(
                weight=0.25,
                model=MaskAdherenceModel(),
            ),
            RewardModelType.STRUCTURE_CONSISTENCY: PackedRewardModel(
                weight=0.25,
                model=StructureConsistencyModel(),
            ),
            RewardModelType.SEMANTIC_CONSISTENCY: PackedRewardModel(
                weight=0.15,
                model=SemanticConsistencyModel(),
            ),
            RewardModelType.IMAGE: PackedRewardModel(
                weight=0.10,
                model=ImageRewardModel(),
            ),
        }
    return INPAINTING_MODELS


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
    synapse: BaseTask,
    responses: List[BaseTask],
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


def get_reward_functions(
    task_type: TaskType = TaskType.TEXT_TO_IMAGE,
) -> List[PackedRewardModel]:
    if task_type == TaskType.INPAINT_IMAGE:
        get_func: Callable = lambda x: get_function(get_inpainting_models(), x)

        return [
            get_func(RewardModelType.BOUNDARY_COHERENCE),
            get_func(RewardModelType.MASK_ADHERENCE),
            get_func(RewardModelType.STRUCTURE_CONSISTENCY),
            get_func(RewardModelType.SEMANTIC_CONSISTENCY),
            get_func(RewardModelType.IMAGE),
        ]

    get_func: Callable = lambda x: get_function(get_reward_models(), x)
    return [
        get_func(RewardModelType.ENHANCED_CLIP),
        get_func(RewardModelType.IMAGE),
        get_func(RewardModelType.HUMAN),
    ]


def get_masking_functions(_task_type: TaskType) -> List[PackedRewardModel]:
    return [
        get_function(get_masking_models(), RewardModelType.NSFW),
        get_function(get_masking_models(), RewardModelType.BLACKLIST),
        get_function(get_masking_models(), RewardModelType.DUPLICATE),
    ]
