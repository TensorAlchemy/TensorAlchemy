from typing import Any, List

import torch
from torch import Tensor
import bittensor as bt
from loguru import logger
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_fixed

from neurons.validator.config import get_backend_client, get_metagraph
from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import RewardModelType


class HumanValidationRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return str(RewardModelType.HUMAN)

    async def get_rewards(
        self,
        _synapse: bt.Synapse,
        responses: torch.FloatTensor,
        rewards: torch.FloatTensor,
    ) -> tuple[Tensor, Tensor | Any]:
        logger.info("Extracting human votes...")

        hotkeys: List[str] = get_metagraph().hotkeys
        human_voting_scores = None
        human_voting_scores_dict = {}

        max_retries = 3
        backoff = 2
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_retries), wait=wait_fixed(backoff)
            ):
                with attempt:
                    human_voting_scores = await get_backend_client().get_votes()

        except RetryError as e:
            logger.error(f"error while getting votes: {e}")
            # Return empty results
            return self.human_voting_scores, self.human_voting_scores

        if human_voting_scores:
            for inner_dict in human_voting_scores.values():
                for key, value in inner_dict.items():
                    if key in human_voting_scores_dict:
                        human_voting_scores_dict[key] += value
                    else:
                        human_voting_scores_dict[key] = value

        if human_voting_scores_dict != {}:
            for index, hotkey in enumerate(hotkeys):
                if hotkey in human_voting_scores_dict.keys():
                    self.human_voting_scores[index] = human_voting_scores_dict[hotkey]

        return self.human_voting_scores

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        if self.human_voting_scores.sum() == 0:
            human_voting_scores_normalised = rewards
        else:
            human_voting_scores_normalised = rewards / rewards.sum()

        return human_voting_scores_normalised
