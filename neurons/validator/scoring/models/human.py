# In neurons/validator/rewards/models/human.py

from typing import Dict, List
from loguru import logger
import bittensor as bt
import torch

from neurons.validator.scoring.models.base import BaseRewardModel
from neurons.validator.scoring.models.types import RewardModelType
from neurons.validator.config import get_backend_client, get_device, get_metagraph


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
        synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> torch.Tensor:
        logger.info("Extracting human votes...")

        try:
            voting_scores: Dict[str, float] = process_voting_scores(
                await get_backend_client().get_votes()
            )

        except Exception as e:
            logger.error(f"Error while getting votes: {e}")
            return torch.zeros(get_metagraph().n).to(get_device())

        def get_reward(response: bt.Synapse) -> float:
            return voting_scores.get(
                response.axon.hotkey,
                0.0,
            )

        return await super().build_rewards_tensor(
            get_reward,
            synapse,
            responses,
        )
