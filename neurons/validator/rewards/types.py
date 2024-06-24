from enum import Enum

import torch
from pydantic import BaseModel


class RewardModelType(Enum):
    diversity = "diversity_reward_model"
    image = "image_reward_model"
    human = "human_reward_model"
    blacklist = "blacklist_filter"
    nsfw = "nsfw_filter"
    model_diversity = "model_diversity_reward_model"


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
