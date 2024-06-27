from typing import Dict, List
import bittensor as bt
from loguru import logger
from transformers import CLIPImageProcessor
from neurons.safety import StableDiffusionSafetyChecker
from neurons.validator.config import get_device
from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import RewardModelType


class NSFWRewardModel(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.NSFW

    def __init__(self):
        super().__init__()
        self.safetychecker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(get_device())
        self.processor = CLIPImageProcessor()

    def get_reward(self, response: bt.Synapse) -> float:
        if not response.images:
            return 1.0

        if any(image is None for image in response.images):
            return 1.0

        try:
            clip_input = self.processor(
                [bt.Tensor.deserialize(image) for image in response.images],
                return_tensors="pt",
            ).to(get_device())

            _, has_nsfw_concept = self.safetychecker.forward(
                images=response.images,
                clip_input=clip_input.pixel_values.to(get_device()),
            )

            return 1.0 if any(has_nsfw_concept) else 0.0

        except Exception as e:
            logger.error(f"Error in NSFW detection: {e}")
            logger.error(f"images={response.images}")
            return 0.0
