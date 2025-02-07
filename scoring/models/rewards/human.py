from typing import Dict, List

import torch
from loguru import logger

from neurons.config import get_backend_client
from neurons.protocol import BaseTask
from scoring.models.base import BaseRewardModel
from scoring.models.types import RewardModelType

HumanVotingResults = Dict[str, Dict[str, float]]


def process_voting_scores(
    human_voting_scores: HumanVotingResults,
) -> Dict[str, float]:
    to_return: Dict[str, float] = {}

    for inner_dict in human_voting_scores.values():
        for hotkey, value in inner_dict.items():
            to_return[hotkey] = to_return.get(hotkey, 0) + value

    return to_return


class HumanValidationRewardModel(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.HUMAN

    async def get_rewards(
        self,
        synapse: BaseTask,
        responses: List[BaseTask],
    ) -> torch.Tensor:
        logger.info("Extracting human votes...")

        try:
            voting_scores: Dict[str, float] = process_voting_scores(
                await get_backend_client().get_votes()
            )

        except Exception as e:
            logger.error(f"Error while getting votes: {e}")
            return super().zeros()

        async def get_reward(response: BaseTask) -> float:
            if not (response.axon and response.axon.hotkey):
                return 0.0

            return voting_scores.get(
                response.axon.hotkey,
                0.0,
            )

        return await super().build_rewards_tensor(
            get_reward,
            synapse,
            responses,
        )
