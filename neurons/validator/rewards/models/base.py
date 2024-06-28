from abc import abstractmethod
from typing import Callable, List, Tuple, Dict

import bittensor as bt
import torch
from loguru import logger

from neurons.validator.config import get_device, get_metagraph
from neurons.validator.rewards.types import RewardModelType, ScoringResult


class BaseRewardModel:
    @property
    @abstractmethod
    def name(self) -> RewardModelType:
        ...

    def is_strict_uid_scoring(self) -> float:
        return True

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return str(self.name)

    def __init__(self):
        if not hasattr(self, "get_reward"):
            raise TypeError(
                #
                f"Subclasse {self.__class__.__name__} "
                + "must implement reward method"
            )

    async def build_rewards_tensor(
        self,
        method: Callable,
        _synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> torch.Tensor:
        if not callable(method):
            raise NotImplementedError(f"{method.__name__} is not callable!")

        rewards = torch.zeros(get_metagraph().n).to(get_device())
        for response in responses:
            score = method(response)
            hotkey = response.axon.hotkey
            try:
                index = get_metagraph().hotkeys.index(hotkey)
                rewards[index] = score
                logger.info(
                    f"Assigned {score}" + f" to index {index}" + f" for hotkey {hotkey}"
                )
            except ValueError:
                logger.error(f"Hotkey {hotkey} not found in metagraph")

        return rewards

    async def get_rewards(
        self,
        synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> torch.Tensor:
        return await self.build_rewards_tensor(
            self.get_reward,
            synapse,
            responses,
        )

    def get_reward(self, _response: bt.Synapse) -> float:
        return 0.0

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if rewards.sum() == 0:
            return rewards

        y_range: float = rewards.max() - rewards.min() + 1e-8
        to_return: torch.Tensor = (rewards - rewards.min()) / y_range

        if self.is_strict_uid_scoring():
            to_return[rewards == 0] = 0

        return to_return

    async def apply(
        self,
        synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> ScoringResult:
        # Get rewards for the responses
        rewards = await self.get_rewards(synapse, responses)

        # Normalize rewards
        normalized_rewards = self.normalize_rewards(rewards)

        return ScoringResult(
            scores=rewards,
            type=self.name,
            normalized=normalized_rewards,
        )
