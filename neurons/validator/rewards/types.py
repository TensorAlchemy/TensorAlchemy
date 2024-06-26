from enum import Enum
from typing import Tuple

import torch
from pydantic import BaseModel
from neurons.validator.rewards.models.base import BaseRewardModel


class PackedRewardModel(BaseModel):
    weight: float
    model: BaseRewardModel

    class Config:
        arbitrary_types_allowed = True

    @property
    def name(self) -> str:
        return str(self.model.name)

    def apply(
        self, *args, **kwargs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor,]:
        return self.model.apply(*args, **kwargs)


class RewardModelType(str, Enum):
    EMPTY = "EMPTY"
    HUMAN = "HUMAN"
    IMAGE = "IMAGE"
    NSFW = "NSFW_FILTER"
    DIVERSITY = "DIVERSITY"
    BLACKLIST = "BLACKLIST_FILTER"


class AutomatedRewards(BaseModel):
    event: dict
    rewards: torch.Tensor

    class Config:
        arbitrary_types_allowed = True


class MaskedRewards(BaseModel):
    event: dict
    rewards: torch.Tensor

    class Config:
        arbitrary_types_allowed = True
