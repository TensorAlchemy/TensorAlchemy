from abc import ABC, abstractmethod
from typing import List, Dict
import bittensor as bt
from bittensor.utils import torch

from neurons.protocol import ModelType
from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import AutomatedRewards, MaskedRewards


class AbstractRewardProcessor(ABC):

    @abstractmethod
    async def get_automated_rewards(
        self,
        validator: "StableValidator",
        model_type: ModelType,
        responses: List[bt.Synapse],
        uids: List[int],
        task_type,
        synapse,
        device: torch.device = None,
    ) -> AutomatedRewards:
        pass

    @abstractmethod
    async def get_masked_rewards(
        self, responses: List[bt.Synapse], models: List[BaseRewardModel]
    ) -> MaskedRewards:
        pass

    @abstractmethod
    async def get_human_rewards(
        self,
        hotkeys: List[str],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def filter_rewards(
        self,
        isalive_dict: Dict[int, int],
        isalive_threshold: int,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        pass
