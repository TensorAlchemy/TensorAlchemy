from typing import List

import bittensor as bt
import torch
from loguru import logger
from transformers import CLIPProcessor, CLIPModel

from neurons.utils.image import synapse_to_tensors

from neurons.validator.config import get_device
from neurons.validator.scoring.models.base import BaseRewardModel
from neurons.validator.scoring.models.types import RewardModelType


class ClipRewardModel(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.CLIP

    def __init__(self):
        super().__init__()
        self.device = get_device()
        self.scoring_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def get_reward(self, response: bt.Synapse) -> float:
        if not response.images:
            return 0.0

        if any(image is None for image in response.images):
            return 0.0

        logger.info("Running CLIP scoring...")

        try:
            # Clip expects RGB values in range (0, 255)
            scaled_tensors = [
                tensor * 255 for tensor in synapse_to_tensors(response)
            ]

            # Ensure the prompt is a list of strings
            prompt = [response.prompt]

            inputs = self.processor(
                text=prompt,
                images=scaled_tensors[0],
                return_tensors="pt",
                padding=True,
            )

            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Ensure the model is on the correct device
                self.scoring_model = self.scoring_model.to(self.device)
                outputs = self.scoring_model(**inputs)

            # Get the similarity score
            image_features = outputs.image_embeds.to(self.device)
            text_features = outputs.text_embeds.to(self.device)

            # Normalize features
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

            # Compute similarity
            similarity_score = (
                (100.0 * image_features @ text_features.T).squeeze().item()
            )

            logger.info(f"CLIP similarity score: {similarity_score:.4f}")

            return similarity_score

        except Exception as e:
            logger.error(f"Error in CLIP scoring: {e}")
            return 0.0
