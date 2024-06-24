from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import bittensor as bt
import numpy as np
import torch
import torchvision.transforms as T
from loguru import logger

from neurons.validator.utils import cosine_distance

transform = T.Compose([T.PILToTensor()])


@dataclass(frozen=True)
class DefaultRewardFrameworkConfig:
    """Reward framework default configuration.
    Note: All the weights should add up to 1.0.
    """

    diversity_model_weight: float = 0.05
    image_model_weight: float = 0.95
    human_model_weight: float = 0

    @torch.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        special_cos_dist = (
            cosine_distance(image_embeds, self.special_care_embeds)
            .cpu()
            .float()
            .numpy()
        )
        cos_dist = (
            cosine_distance(image_embeds, self.concept_embeds).cpu().float().numpy()
        )

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {
                "special_scores": {},
                "special_care": [],
                "concept_scores": {},
                "bad_concepts": [],
                "bad_score": 0.0,
            }

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 1.0

            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(
                    concept_cos - (concept_threshold * adjustment), 3
                )
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append(
                        {concept_idx, result_img["special_scores"][concept_idx]}
                    )

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(
                    concept_cos - (concept_threshold * adjustment), 3
                )
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)
                    result_img["bad_score"] += result_img["concept_scores"][concept_idx]

            result.append(result_img)

        has_nsfw_concepts = [
            len(res["bad_concepts"]) > 0 and res["bad_score"] > 0.01 for res in result
        ]

        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                if torch.is_tensor(images) or torch.is_tensor(images[0]):
                    images[idx] = torch.zeros_like(images[idx])  # black image
                else:
                    try:
                        images[idx] = np.zeros(
                            transform(images[idx]).shape
                        )  # black image
                    except Exception:
                        images[idx] = np.zeros((1024, 1024, 3))

        if any(has_nsfw_concepts):
            logger.warning(
                "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        return images, has_nsfw_concepts


class BaseRewardModel:
    @property
    @abstractmethod
    def name(self) -> str: ...

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return str(self.name)

    @abstractmethod
    async def get_rewards(self, responses: List, rewards) -> torch.FloatTensor: ...

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.var = 0.0
        self.count_limit = 3000

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        """
        This method normalizes the given rewards by updating the moving mean and variance statistics. The rewards are first standardized, and then scaled to the 0-1 range using a cumulative distribution function (CDF) to ensure they're in a comparable range across different environments.

        Args:
        rewards (torch.FloatTensor): The reward values to be normalized.

        Returns:
        torch.FloatTensor: The normalized reward values.

        Note:
        - This function uses Welford's online algorithm to update the mean and variance.
        - It standardizes the reward values using the updated mean and variance.
        - It then scales the standardized values to the 0-1 range using the error function (erf) as a CDF.
        """
        # Get the number of rewards (successful responses).
        new_count = rewards.numel()

        # Update stats only if there are new rewards.
        if 0 < new_count and 0 < self.count + new_count:
            # Calculate the mean and standard deviation of the new rewards.
            new_mean = rewards.mean()
            new_var = rewards.var(dim=0)

            # Compute the weights for the new and old rewards.
            new_weight = new_count / (self.count + new_count)
            old_weight = self.count / (self.count + new_count)

            # Save the difference in means before updating the old mean.
            diff = new_mean - self.mean

            # Update the old mean with the new mean and weights.
            self.mean = new_weight * new_mean + old_weight * self.mean
            # Update the old variance with the new variance and weights, and adjusting for the difference in means.
            self.var = (
                (new_weight * new_var)
                + (old_weight * self.var)
                + (new_weight * old_weight) * diff * diff
            )
            # Update the old count with the new count, but don't exceed the limit.
            self.count = min(self.count_limit, self.count + new_count)

        # Standardize the rewards using the updated mean and variance.
        rewards = rewards - self.mean
        if self.var > 0:
            rewards /= torch.sqrt(self.var)
        # Scale the standardized rewards to the range [0, 1] using the error function as a cumulative distribution function (CDF).
        rewards = 0.5 * (
            1 + torch.erf(rewards / torch.sqrt(torch.tensor([2.0])).to(rewards.device))
        )

        return rewards

    async def apply(
        self, responses: List[bt.Synapse], rewards, synapse=None
    ) -> torch.FloatTensor:
        """Applies the reward model across each call. Unsuccessful responses are zeroed."""
        # Get indices of correctly responding calls.

        successful_generations_indices: List[int] = [
            idx
            for idx, resp in enumerate(responses)
            if resp.dendrite.status_code == 200
        ]

        # Get all completions from responding calls.
        successful_generations: List[str] = [
            responses[idx] for idx in successful_generations_indices
        ]

        # Reward each completion.
        successful_rewards = await self.get_rewards(
            successful_generations, rewards, synapse
        )

        # Softmax rewards across samples.
        successful_rewards_normalized = self.normalize_rewards(successful_rewards)

        # Init zero rewards for all calls.
        filled_rewards = torch.zeros(len(responses), dtype=torch.float32)
        filled_rewards_normalized = torch.zeros(len(responses), dtype=torch.float32)

        # Fill reward tensor.
        for idx, reward, reward_normalized in zip(
            successful_generations_indices,
            successful_rewards,
            successful_rewards_normalized,
        ):
            filled_rewards[idx] = reward
            filled_rewards_normalized[idx] = reward_normalized
        # Return the filled rewards.
        return filled_rewards, filled_rewards_normalized
