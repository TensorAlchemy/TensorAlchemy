from enum import Enum
from typing import Callable, Dict, List, Tuple

import torch
from pydantic import BaseModel, ConfigDict, Field

from neurons.protocol import BaseTask
from scoring.models.base import BaseRewardModel


class RewardModelType(str, Enum):
    NSFW = "NSFW"
    DUPLICATE = "DUPLICATE"
    BLACKLIST = "BLACKLIST"
    EMPTY = "EMPTY"
    HUMAN = "HUMAN"
    IMAGE = "IMAGE"
    ENHANCED_CLIP = "ENHANCED_CLIP"

    # Inpainting-specific models
    BOUNDARY_COHERENCE = "BOUNDARY_COHERENCE"
    MASK_ADHERENCE = "MASK_ADHERENCE"
    STRUCTURE_CONSISTENCY = "STRUCTURE_CONSISTENCY"
    SEMANTIC_CONSISTENCY = "SEMANTIC_CONSISTENCY"


def default_should_apply(
    _synapse: BaseTask,
    _responses: List[BaseTask],
) -> bool:
    return True


class PackedRewardModel(BaseModel):
    weight: float
    model: BaseRewardModel
    model_config = ConfigDict(arbitrary_types_allowed=True)

    should_apply: Callable[[BaseTask, List[BaseTask]], bool] = Field(
        default=default_should_apply
    )

    @property
    def name(self) -> RewardModelType:
        return self.model.name

    def apply(
        self,
        *args,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return self.model.apply(*args, **kwargs)


ModelStorage = Dict[RewardModelType, PackedRewardModel]
