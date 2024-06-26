# In neurons/validator/rewards/models/human.py

from typing import Dict, List
import bittensor as bt
from loguru import logger
import torch

from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import RewardModelType
from neurons.validator.config import get_backend_client


class HumanValidationRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return str(RewardModelType.HUMAN)

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

        rewards = {
            response.dendrite.hotkey: human_voting_scores_dict.get(
                response.dendrite.hotkey, 0.0
            )
            for response in responses
        }
        return rewards

    def normalize_rewards(self, rewards: Dict[str, float]) -> Dict[str, float]:
        if not rewards:
            return rewards

        values = torch.tensor(list(rewards.values()))
        if values.numel() > 1:
            normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
        else:
            normalized = values

        return {
            #
            hotkey: float(norm)
            for hotkey, norm in zip(
                rewards.keys(),
                normalized,
            )
        }
