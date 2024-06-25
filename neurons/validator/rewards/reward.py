from typing import Dict, List

import bittensor as bt
import torch

from neurons.protocol import ModelType
from neurons.validator.config import (
    get_device,
    get_metagraph,
)
from neurons.validator.rewards.interface import AbstractRewardProcessor
from neurons.validator.rewards.types import (
    AutomatedRewards,
    MaskedRewards,
    RewardModelType,
)
from neurons.validator.rewards.pipeline import (
    apply_reward_functions,
    apply_masking_functions,
)


class RewardProcessor(AbstractRewardProcessor):
    def __init__(self):
        self.reward_names = [
            RewardModelType.IMAGE,
            RewardModelType.DIVERSITY,
        ]

    async def get_automated_rewards(
        self,
        model_type: ModelType,
        responses: List[bt.Synapse],
        task_type,
        synapse,
    ) -> AutomatedRewards:
        event = {"task_type": task_type}

        # Initialise rewards tensor
        rewards: torch.Tensor = torch.zeros(
            len(responses),
            dtype=torch.float32,
        ).to(self.device)

        rewards, reward_event = await apply_reward_functions(
            model_type,
            responses,
            rewards,
            synapse,
        )

        event.update(reward_event)

        rewards, masking_event = await apply_masking_functions(
            model_type,
            responses,
            rewards,
        )

        event.update(masking_event)

        return AutomatedRewards(
            rewards=rewards,
            event=event,
        )

    async def get_masked_rewards(
        self,
        model_type: ModelType,
        responses: List[bt.Synapse],
    ) -> MaskedRewards:
        """Apply masking functions (NSFW, Blacklist etc.) and return rewards

        Return 0 score if response didn't pass check
        """
        rewards, event = await apply_masking_functions(
            model_type,
            responses,
            torch.ones(len(responses)).to(get_device()),
        )

        return MaskedRewards(rewards=rewards, event=event)

    def filter_rewards(
        self,
        isalive_dict: Dict[int, int],
        isalive_threshold: int,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        for uid, count in isalive_dict.items():
            if count >= isalive_threshold:
                rewards[uid] = 0

        return rewards
