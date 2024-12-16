from enum import Enum
from typing import Callable, Dict, List, Tuple

import torch
from pydantic import BaseModel, ConfigDict, Field
from scoring.models.base import BaseRewardModel

import bittensor as bt


class RewardModelType(str, Enum):
    # Masking models
    # TODO: Maybe move these out
    NSFW = "NSFW"
    DUPLICATE = "DUPLICATE"
    BLACKLIST = "BLACKLIST"

    # Reward models
    EMPTY = "EMPTY"
    HUMAN = "HUMAN"
    IMAGE = "IMAGE"
    ENHANCED_CLIP = "ENHANCED_CLIP"


def default_should_apply(
    _synapse: bt.Synapse,
    _responses: List[bt.Synapse],
) -> bool:
    return True


class PackedRewardModel(BaseModel):
    weight: float
    model: BaseRewardModel
    model_config = ConfigDict(arbitrary_types_allowed=True)

    should_apply: Callable[[bt.Synapse, List[bt.Synapse]], bool] = Field(
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
