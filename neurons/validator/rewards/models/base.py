from abc import abstractmethod
from typing import List, Tuple, Dict

import bittensor as bt
import torch
from loguru import logger

from neurons.validator.config import get_device, get_metagraph
from neurons.validator.rewards.types import RewardModelType


class BaseRewardModel:
    @property
    @abstractmethod
    def name(self) -> RewardModelType:
        ...

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return str(self.name)

    def __init__(self) -> None:
        self.metagraph = get_metagraph()

    async def get_rewards(
        self,
        _synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> torch.Tensor:
        rewards = torch.zeros(self.metagraph.n).to(get_device())

        for response in responses:
            score = self.reward(response)
            hotkey = response.dendrite.hotkey

            try:
                index = self.metagraph.hotkeys.index(hotkey)
                rewards[index] = score
                logger.info(
                    f"Assigned score {score} to index {index} for hotkey {hotkey}"
                )
            except ValueError:
                logger.error(f"Hotkey {hotkey} not found in metagraph")

        return rewards

    @abstractmethod
    def reward(self, response) -> float:
        return 0.0

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        return rewards

    async def apply(
        self,
        synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        # Get rewards for the responses
        rewards = await self.get_rewards(synapse, responses)

        # Normalize rewards
        normalized_rewards = self.normalize_rewards(rewards)

        return rewards, normalized_rewards
