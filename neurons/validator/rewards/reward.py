from typing import Dict, List, Tuple

import bittensor as bt
import torch
from loguru import logger

from neurons.protocol import ModelType
from neurons.validator.backend.client import TensorAlchemyBackendClient
from neurons.validator.rewards.interface import AbstractRewardProcessor
from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.models.blacklist import BlacklistFilter
from neurons.validator.rewards.models.diversity import ModelDiversityRewardModel
from neurons.validator.rewards.models.human import HumanValidationRewardModel
from neurons.validator.rewards.models.image_reward import ImageRewardModel
from neurons.validator.rewards.models.nsfw import NSFWRewardModel
from neurons.validator.rewards.types import (
    AutomatedRewards,
    RewardModelType,
    MaskedRewards,
)
from neurons.validator.utils import measure_time


class RewardProcessor(AbstractRewardProcessor):

    def __init__(
        self,
        metagraph: "bt.metagraph.Metagraph",
        device: torch.device,
        backend_client: TensorAlchemyBackendClient,
    ):
        self.metagraph = metagraph
        self.device = device
        self.backend_client = backend_client

        # Init Reward Models
        self.reward_models: Dict[str, BaseRewardModel] = {
            "ImageRewardModel": ImageRewardModel(),
            "ModelDiversityRewardModel": ModelDiversityRewardModel(),
        }

        # Init Reward Weights
        self.reward_weights: Dict[str, float] = {
            "ImageRewardModel": 0.8,
            "ModelDiversityRewardModel": 0.2,
        }

        self.reward_names = [
            RewardModelType.image.value,
            RewardModelType.model_diversity.value,
        ]

        self.human_voting_scores = torch.zeros(self.metagraph.n).to(self.device)
        self.human_voting_weight = 0.10 / 32
        self.human_voting_reward_model = HumanValidationRewardModel(
            self.metagraph, self.backend_client
        )

        # Init masking function
        self.masking_functions = [BlacklistFilter(), NSFWRewardModel()]

    def _get_reward_function(
        self,
        reward_function: str,
        reward_models: Dict[str, BaseRewardModel],
        reward_weights: Dict[str, float],
    ) -> Tuple[BaseRewardModel, float]:
        if not reward_function in reward_models:
            raise ValueError()

        if not reward_function in reward_weights:
            raise ValueError()

        return reward_models[reward_function], reward_weights[reward_function]

    @measure_time
    def _get_reward_functions(
        self,
        model_type: ModelType,
        reward_models: Dict[str, BaseRewardModel],
        reward_weights: Dict[str, float],
    ) -> List[Tuple[BaseRewardModel, float]]:
        if model_type == ModelType.ALCHEMY.value.lower():
            return [
                self._get_reward_function(
                    "ImageRewardModel", reward_models, reward_weights
                ),
                self._get_reward_function(
                    "ModelDiversityRewardModel", reward_models, reward_weights
                ),
            ]

        return [
            self._get_reward_function(
                "ImageRewardModel", reward_models, reward_weights
            ),
        ]

    async def _apply_reward_functions(
        self,
        model_type: ModelType,
        reward_models: Dict[str, BaseRewardModel],
        reward_weights: Dict[str, float],
        responses: list,
        rewards: torch.Tensor,
        device: torch.device,
        synapse: bt.synapse,
    ) -> tuple[torch.Tensor, dict]:
        event = {}
        reward_functions = self._get_reward_functions(
            model_type, reward_models, reward_weights
        )
        for reward_fn_i, weight_i in reward_functions:
            reward_i, reward_i_normalized = await reward_fn_i.apply(
                responses, rewards, synapse
            )
            rewards += weight_i * reward_i_normalized.to(device)
            event[reward_fn_i.name] = reward_i.tolist()
            event[reward_fn_i.name + "_normalized"] = reward_i_normalized.tolist()
            logger.info(f"{reward_fn_i.name}, {reward_i_normalized.tolist()}")
        return rewards, event

    async def apply_masking_functions(
        self,
        masking_functions: list,
        responses: list,
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        event = {}
        for masking_fn_i in masking_functions:
            mask_i, mask_i_normalized = await masking_fn_i.apply(responses, rewards)
            rewards *= mask_i_normalized.to(self.device)
            event[masking_fn_i.name] = mask_i.tolist()
            event[masking_fn_i.name + "_normalized"] = mask_i_normalized.tolist()
            logger.info(f"{masking_fn_i.name} {mask_i_normalized.tolist()}")
        return rewards, event

    @measure_time
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
        if not device:
            device = self.device

        event = {"task_type": task_type}

        # Initialise rewards tensor
        rewards: torch.Tensor = torch.zeros(len(responses), dtype=torch.float32).to(
            device
        )

        rewards, reward_event = await self._apply_reward_functions(
            model_type,
            self.reward_models,
            self.reward_weights,
            responses,
            rewards,
            device,
            synapse,
        )
        event.update(reward_event)

        rewards, masking_event = await self.apply_masking_functions(
            self.masking_functions, responses, rewards
        )

        event.update(masking_event)

        uids_tensor = torch.tensor(uids).to(device)
        scattered_rewards: torch.Tensor = validator.moving_average_scores.scatter(
            0, uids_tensor, rewards
        ).to(validator.device)

        return AutomatedRewards(
            scattered_rewards=scattered_rewards, rewards=rewards, event=event
        )

    async def get_masked_rewards(
        self,
        responses: List[bt.Synapse],
        models: List[BaseRewardModel],
        device: torch.device = None,
    ) -> MaskedRewards:
        """Apply masking functions (NSFW, Blacklist etc.) and return rewards

        Return 0 score if response didn't pass check
        """
        if not device:
            device = self.device

        rewards, event = await self.apply_masking_functions(
            models, responses, torch.ones(len(responses)).to(device)
        )
        return MaskedRewards(rewards=rewards, event=event)

    async def get_human_voting_scores(
        self,
        hotkeys: List[str],
    ) -> torch.Tensor:
        _, human_voting_scores_normalised = (
            await self.human_voting_reward_model.get_rewards(
                hotkeys,
            )
        )
        return human_voting_scores_normalised

    def apply_human_voting_weight(
        self,
        rewards: torch.Tensor,
        human_voting_scores: torch.Tensor,
        human_voting_weight: float,
    ) -> torch.Tensor:
        scattered_rewards_adjusted = rewards + (
            human_voting_weight * human_voting_scores
        )
        return scattered_rewards_adjusted

    async def get_human_rewards(
        self,
        hotkeys: List[str],
        rewards: torch.Tensor,
        # mock: bool = False,
        # mock_winner: str = None,
        # mock_loser: str = None,
    ) -> torch.Tensor:
        human_voting_scores = await self.get_human_voting_scores(
            hotkeys,
            # mock,
            # mock_winner,
            # mock_loser,
        )
        scattered_rewards_adjusted = self.apply_human_voting_weight(
            rewards, human_voting_scores, self.human_voting_weight
        )
        return scattered_rewards_adjusted

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
