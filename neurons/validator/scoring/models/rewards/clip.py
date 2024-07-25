from typing import List

import bittensor as bt
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
        self.scoring_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def get_reward(self, response: bt.Synapse) -> float:
        if not response.images:
            return 0.0

        if any(image is None for image in response.images):
            return 0.0

        logger.info("Running clip scoring...")

        try:
            # Clip expects RGB int values in range (0, 255)
            scaled_tensors = [
                #
                tensor * 255
                for tensor in synapse_to_tensors(response)
            ]

            clip_input = self.processor(
                scaled_tensors[0],
                return_tensors="pt",
            ).to(get_device())

            results = self.scoring_model(
                images=response.images[0],
                text=response.prompt,
                clip_input=clip_input.pixel_values.to(get_device()),
            )

            print(results)

            return results.logits_per_image

        except Exception as e:
            logger.error(f"Error in CLIP scoring: {e}")
            return 0.0
