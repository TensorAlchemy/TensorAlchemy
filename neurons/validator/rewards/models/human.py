from typing import Any, List

import bittensor as bt
import torch
from loguru import logger
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_fixed
from torch import Tensor

from neurons.validator.config import get_backend_client
from neurons.validator.backend.client import TensorAlchemyBackendClient
from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import RewardModelType


class HumanValidationRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return str(RewardModelType.HUMAN)

    async def get_rewards(
        self,
        _synapse: bt.Synapse,
        hotkeys: List[str],
        # mock=False, mock_winner=None, mock_loser=None
    ) -> tuple[Tensor, Tensor | Any]:
        logger.info("Extracting human votes...")

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

        if self.human_voting_scores.sum() == 0:
            human_voting_scores_normalised = self.human_voting_scores
        else:
            human_voting_scores_normalised = (
                self.human_voting_scores / self.human_voting_scores.sum()
            )

        return self.human_voting_scores, human_voting_scores_normalised
