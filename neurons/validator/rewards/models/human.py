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
        # Return the enum value for the HUMAN reward model type
        return RewardModelType.HUMAN.value

    def __init__(
        self,
        metagraph: "bt.metagraph.Metagraph",
        backend_client: TensorAlchemyBackendClient,
    ):
        # Initialize the HumanValidationRewardModel
        super().__init__()
        # Set the device (CPU or GPU) based on the validator configuration
        self.device = validator_config.get_default_device()
        # Initialize a tensor to store human voting
        # scores for each neuron in the metagraph
        self.human_voting_scores = torch.zeros((metagraph.n)).to(self.device)
        # Store the backend client for API calls
        self.backend_client = backend_client

    async def get_rewards(
        self,
        hotkeys: List[str],
    ) -> tuple[Tensor, Tensor | Any]:
        logger.info("Extracting human votes...")
        human_voting_scores = None
        human_voting_scores_dict = {}

        # Set up retry parameters for API calls
        max_retries = 3
        backoff = 2

        # Attempt to get votes from the backend with retries
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_retries), wait=wait_fixed(backoff)
            ):
                with attempt:
                    human_voting_scores = await self.backend_client.get_votes()
        except RetryError as e:
            logger.error(f"error while getting votes: {e}")
            # Return empty results if all retries fail
            return self.human_voting_scores, self.human_voting_scores

        # Process the received voting scores
        if human_voting_scores:
            for inner_dict in human_voting_scores.values():
                for key, value in inner_dict.items():
                    # Aggregate scores for each hotkey
                    if key in human_voting_scores_dict:
                        human_voting_scores_dict[key] += value
                    else:
                        human_voting_scores_dict[key] = value

        # Update the human_voting_scores tensor with aggregated scores
        if human_voting_scores_dict != {}:
            for index, hotkey in enumerate(hotkeys):
                if hotkey in human_voting_scores_dict.keys():
                    self.human_voting_scores[index] = human_voting_scores_dict[hotkey]

        # Normalize the scores if they're not all zero
        if self.human_voting_scores.sum() == 0:
            human_voting_scores_normalised = self.human_voting_scores
        else:
            human_voting_scores_normalised = (
                self.human_voting_scores / self.human_voting_scores.sum()
            )

        # Return both raw and normalized scores
        return self.human_voting_scores, human_voting_scores_normalised
