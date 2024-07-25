from pydantic import BaseModel
from enum import Enum
from typing import Dict, Tuple
from pydantic import ConfigDict, BaseModel

import torch

from neurons.validator.scoring.models.base import BaseRewardModel


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


class PackedRewardModel(BaseModel):
    weight: float
    model: BaseRewardModel
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
