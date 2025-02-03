"""
Enhanced CLIP Reward Model

This module implements an enhanced version of the CLIP
(Contrastive Language-Image Pre-Training) model for image-text similarity scoring.

It breaks down prompts into individual elements and computes similarity scores for each element,
providing a more granular and accurate assessment of image-text alignment.

The model uses a combination of CLIP and a Language Model (LLM) such as OpenAI or Corcel to achieve
higher resolution in prompt evaluation. This approach allows for a more precise analysis of each
"pixel" of the prompt, substantially increasing CLIP's effectiveness.

Note: Weights for this model and other related models can be found in the
TensorAlchemy/scoring/models/__init__.py file.

Author: TensorAlchemy developers
Date: July 30, 2024
"""

import traceback
from typing import Dict, List

import bittensor as bt
import torch
from loguru import logger
from transformers import CLIPModel, CLIPProcessor

from neurons.config import get_device
from neurons.protocol import BaseTask
from neurons.utils.image import synapse_to_image
from scoring.models.base import BaseRewardModel
from scoring.models.rewards.enhanced_clip.utils import (
    PromptBreakdown,
    break_down_prompt,
)
from scoring.models.types import RewardModelType


class EnhancedClipRewardModel(BaseRewardModel):
    """
    An enhanced CLIP-based reward model for image-text similarity scoring.

    This model extends the base CLIP model by breaking down
    prompts into individual elements and computing similarity scores
    for each element.

    This approach provides a more detailed and accurate assessment
    of how well an image matches a given text prompt.
    """

    @property
    def name(self) -> RewardModelType:
        return RewardModelType.ENHANCED_CLIP

    def __init__(self):
        """
        Initialize the EnhancedClipRewardModel with CLIP model and processor.
        """
        super().__init__()
        self.device: torch.device = get_device()
        self.model: CLIPModel = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.threshold_min: float = 0.0

    def compute_clip_score(
        self,
        prompt_elements: PromptBreakdown,
        response: BaseTask,
    ) -> float:
        """
        Compute the enhanced CLIP score for a given prompt and image response.

        Args:
            prompt_elements (PromptBreakdown): Breakdown of the prompt
                                               into individual elements.

            response (BaseTask): The response containing
                                   the image to be evaluated.

        Returns:
            float: The computed similarity score
                   between the prompt elements and the image.
        """
        if not response.images or any(
            image is None for image in response.images
        ):
            return 0.0

        try:
            image = synapse_to_image(response)
            descriptions: List[str] = [
                element["description"]
                for element in prompt_elements["elements"]
            ]

            inputs: Dict[str, torch.Tensor] = self.processor(
                text=descriptions,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits_per_image: torch.Tensor = outputs.logits_per_image.squeeze()
            similarities: torch.Tensor = (
                logits_per_image - logits_per_image.min()
            ) / (logits_per_image.max() - logits_per_image.min())

            adjusted_similarities: torch.Tensor = torch.where(
                similarities > self.threshold_min,
                similarities,
                torch.zeros_like(similarities),
            )

            final_result: float = (adjusted_similarities + 1).prod().item() - 1

            for i, desc in enumerate(descriptions):
                logger.info(
                    f"Element: {desc}, "
                    f"Similarity: {similarities[i].item():.4f}, "
                    f"Adjusted: {adjusted_similarities[i].item():.4f}"
                )

            logger.info(f"Enhanced CLIP similarity score: {final_result:.4f}")

            final_result /= len(descriptions)

            return final_result

        except Exception:
            logger.error(traceback.format_exc())
            return 0.0

    async def get_rewards(
        self,
        synapse: BaseTask,
        responses: List[BaseTask],
    ) -> torch.Tensor:
        """
        Compute rewards for a list of responses based on their similarity to the given prompt.

        Args:
            synapse (BaseTask): The original synapse containing the prompt.
            responses (List[BaseTask]): List of responses to be evaluated.

        Returns:
            torch.Tensor: A tensor of computed rewards for each response.
        """
        prompt_elements: PromptBreakdown = await break_down_prompt(
            synapse.prompt
        )

        def get_reward(response: BaseTask) -> float:
            return self.compute_clip_score(prompt_elements, response)

        rewards: torch.Tensor = await super().build_rewards_tensor(
            get_reward,
            synapse,
            responses,
        )

        return rewards
