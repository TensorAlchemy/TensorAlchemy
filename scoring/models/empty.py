from neurons.protocol import BaseTask
from scoring.models.base import BaseRewardModel
from scoring.models.types import RewardModelType


class EmptyScoreRewardModel(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.EMPTY

    async def get_reward(self, _response: BaseTask) -> float:
        return 0.0
