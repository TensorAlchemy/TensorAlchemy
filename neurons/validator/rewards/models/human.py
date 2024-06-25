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

    def get_successful_indices(
        self,
        rewards: torch.FloatTensor,
        _responses: List[Any],
    ) -> List[int]:
        return [i for i, reward in enumerate(rewards) if reward > 0]

    async def get_rewards(
        self,
        _synapse: bt.Synapse,
        _responses: torch.FloatTensor,
        rewards: torch.FloatTensor,
    ) -> tuple[Tensor, Tensor | Any]:
        logger.info("Extracting human votes...")

        hotkeys: List[str] = get_metagraph().hotkeys
        to_return = torch.zeros((get_metagraph().n)).to(self.device)
        human_voting_scores_dict = {}

        max_retries = 3
        backoff = 2
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_retries), wait=wait_fixed(backoff)
            ):
                with attempt:
                    self.human_voting_scores = await get_backend_client().get_votes()

        except RetryError as e:
            logger.error(f"error while getting votes: {e}")
            # Return empty results
            return to_return

        if human_voting_scores:
            for inner_dict in human_voting_scores.values():
                for hotkey, value in inner_dict.items():
                    if hotkey in human_voting_scores_dict:
                        human_voting_scores_dict[hotkey] += value
                    else:
                        human_voting_scores_dict[hotkey] = value

        if human_voting_scores_dict != {}:
            for index, hotkey in enumerate(hotkeys):
                if hotkey in human_voting_scores_dict.keys():
                    to_return[index] = human_voting_scores_dict[hotkey]

        return to_return

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        if rewards.sum() == 0:
            return rewards

        return rewards / rewards.sum()
