from scoring.models.base import BaseRewardModel
from scoring.models.types import RewardModelType

import bittensor as bt


class EmptyScoreRewardModel(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.EMPTY

    def get_reward(self, _response: bt.Synapse) -> float:
        return 0.0
