from typing import Dict, List
import bittensor as bt
from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import RewardModelType


class EmptyScoreRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return str(RewardModelType.EMPTY)

    async def get_rewards(
        self, _synapse: bt.Synapse, responses: List[bt.Synapse]
    ) -> Dict[int, float]:
        return {response.dendrite.uuid: 0.0 for response in responses}
