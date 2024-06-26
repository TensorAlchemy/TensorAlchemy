# In neurons/validator/rewards/models/human.py

from typing import Dict, List

from loguru import logger
import bittensor as bt
import torch

from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import RewardModelType
from neurons.validator.config import get_backend_client, get_device


class HumanValidationRewardModel(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.HUMAN

    async def get_rewards(
        self,
        _synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> Dict[str, float]:
        logger.info("Extracting human votes...")

        human_voting_scores_dict = {}

        try:
            self.human_voting_scores = await get_backend_client().get_votes()
        except Exception as e:
            logger.error(f"Error while getting votes: {e}")
            return {response.dendrite.hotkey: 0.0 for response in responses}

        if self.human_voting_scores:
            for inner_dict in self.human_voting_scores.values():
                for hotkey, value in inner_dict.items():
                    human_voting_scores_dict[hotkey] = (
                        human_voting_scores_dict.get(hotkey, 0) + value
                    )

        rewards = torch.zeros(self.metagraph.n).to(get_device())

        for response in responses:
            uid = self.metagraph.hotkeys.index(response.dendrite.hotkey)
            rewards[uid] = human_voting_scores_dict.get(
                response.dendrite.hotkey,
                0.0,
            )

        return rewards

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if rewards.sum() == 0:
            return rewards

        return (
            #
            rewards
            - rewards.min()
        ) / (rewards.max() - rewards.min() + 1e-8)
