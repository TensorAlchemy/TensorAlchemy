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

    @abstractmethod
    async def get_rewards(
        self, synapse: bt.Synapse, responses: List[bt.Synapse]
    ) -> Dict[int, float]:
        ...

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.var = 0.0
        self.count_limit = 3000
        self.metagraph = get_metagraph()

    def normalize_rewards(self, rewards: Dict[int, float]) -> Dict[int, float]:
        if not rewards:
            return rewards

        values = torch.tensor(list(rewards.values()))
        normalized = (values - values.mean()) / (values.std() + 1e-8)
        return {uid: float(norm) for uid, norm in zip(rewards.keys(), normalized)}

    def was_success(self, response: bt.Synapse) -> bool:
        return response.dendrite.is_success

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
