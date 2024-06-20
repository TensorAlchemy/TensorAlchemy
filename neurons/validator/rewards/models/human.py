from typing import Any, List

import bittensor as bt
import torch
from loguru import logger
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_fixed
from torch import Tensor

from neurons.validator import config as validator_config
from neurons.validator.backend.client import TensorAlchemyBackendClient
from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import RewardModelType


class HumanValidationRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.human.value

    def __init__(
        self,
        metagraph: "bt.metagraph.Metagraph",
        backend_client: TensorAlchemyBackendClient,
    ):
        super().__init__()
        self.device = validator_config.get_default_device()
        self.human_voting_scores = torch.zeros((metagraph.n)).to(self.device)
        self.backend_client = backend_client

    async def get_rewards(
        self,
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
                    human_voting_scores = await self.backend_client.get_votes()
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

        # TODO: move out mock code
        # else:
        #     human_voting_scores_dict = {hotkey: 50 for hotkey in hotkeys}
        #     if (mock_winner is not None) and (
        #         mock_winner in human_voting_scores_dict.keys()
        #     ):
        #         human_voting_scores_dict[mock_winner] = 100
        #     if (mock_loser is not None) and (
        #         mock_loser in human_voting_scores_dict.keys()
        #     ):
        #         human_voting_scores_dict[mock_loser] = 1

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
