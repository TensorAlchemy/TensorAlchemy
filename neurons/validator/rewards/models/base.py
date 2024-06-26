from abc import abstractmethod
from typing import List, Tuple, Dict

import bittensor as bt
import torch
from loguru import logger

from neurons.validator.config import get_device, get_metagraph


class BaseRewardModel:
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return str(self.name)

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.var = 0.0
        self.count_limit = 3000
        self.metagraph = get_metagraph()

    async def get_rewards(
        self,
        _synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> torch.Tensor:
        rewards = torch.zeros(len(bt.metagraph.n)).to(get_device())

        for response in responses:
            uid = bt.metagraph.hotkeys.index(response.dendrite.hotkey)
            rewards[uid] = self.reward(response)

        return rewards

    @abstractmethod
    def reward(self, response) -> float:
        return 0.0

    def normalize_rewards(self, rewards: Dict[int, float]) -> torch.Tensor:
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
