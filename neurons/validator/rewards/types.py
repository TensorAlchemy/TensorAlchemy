from enum import Enum

import torch
from pydantic import BaseModel


class RewardModelType(Enum):
    DIVERSITY = "DIVERSITY_REWARD_MODEL"
    IMAGE = "IMAGE_REWARD_MODEL"
    HUMAN = "HUMAN_REWARD_MODEL"
    BLACKLIST = "BLACKLIST_FILTER"
    NSFW = "NSFW_FILTER"


class AutomatedRewards(BaseModel):
    scattered_rewards: torch.Tensor
    rewards: torch.Tensor
    event: dict

    class Config:
        arbitrary_types_allowed = True


class MaskedRewards(BaseModel):
    rewards: torch.Tensor
    event: dict

    class Config:
        arbitrary_types_allowed = True
