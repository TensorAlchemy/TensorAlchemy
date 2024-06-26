from typing import Dict, List
import bittensor as bt
from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import RewardModelType


class EmptyScoreRewardModel(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.EMPTY

    def reward(self, _response: bt.Synapse) -> float:
        return 0.0
