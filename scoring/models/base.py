import inspect
from abc import abstractmethod
from typing import TYPE_CHECKING, Callable, List

import torch
from loguru import logger
from neurons.config import get_device, get_metagraph

import bittensor as bt

if TYPE_CHECKING:
    from scoring.models.types import RewardModelType
    from scoring.types import ScoringResult


class BaseRewardModel:
    @property
    @abstractmethod
    def name(self) -> "RewardModelType": ...

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

    def zeros(self) -> torch.Tensor:
        return torch.zeros(get_metagraph().n).to(get_device())

    def ones(self) -> torch.Tensor:
        return torch.ones(get_metagraph().n).to(get_device())

    async def build_rewards_tensor(
        self,
        method: Callable,
        _synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> torch.Tensor:
        if not callable(method):
            raise NotImplementedError(f"{method.__name__} is not callable!")

        rewards = self.zeros()
        for response in responses:
            if inspect.iscoroutinefunction(method):
                score = await method(response)
            else:
                score = method(response)

            hotkey = response.axon.hotkey
            try:
                index = get_metagraph().hotkeys.index(hotkey)
                rewards[index] = score
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
    ) -> "ScoringResult":
        # Get rewards for the responses
        if inspect.iscoroutinefunction(self.get_rewards):
            rewards = await self.get_rewards(synapse, responses)
        else:
            rewards = self.get_rewards(synapse, responses)

        # Normalize rewards
        normalized_rewards = rewards
        if get_metagraph().n > 1:
            normalized_rewards = self.normalize_rewards(rewards)

        from scoring.types import ScoringResult

        # Get all UIDs from the responses
        response_uids = get_uids(responses)

        # Get non-zero UIDs from rewards
        non_zero_uids = torch.nonzero(rewards).squeeze()
        # Ensure non_zero_uids is always 1D, even if there's only one element
        if non_zero_uids.dim() == 0:
            non_zero_uids = non_zero_uids.unsqueeze(0)

        # Combine response_uids and non_zero_uids, removing duplicates
        all_uids = torch.unique(torch.cat([response_uids, non_zero_uids]))

        return ScoringResult(
            scores=rewards,
            type=self.name,
            uids=all_uids,
            normalized=normalized_rewards,
        )


def get_uids(responses: List[bt.Synapse]) -> torch.Tensor:
    metagraph: bt.metagraph = get_metagraph()

    return torch.tensor(
        [
            #
            metagraph.hotkeys.index(response.axon.hotkey)
            for response in responses
        ],
        dtype=torch.long,
    ).to(get_device())
