from enum import Enum

import torch
from pydantic import BaseModel


class RewardModelType(str, Enum):
    # Masking models
    # TODO: Maybe move these out
    NSFW = "NSFW"
    BLACKLIST = "BLACKLIST"

    # Reward models
    EMPTY = "EMPTY"
    HUMAN = "HUMAN"
    IMAGE = "IMAGE"
    SIMILARITY = "SIMILARITY"


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
