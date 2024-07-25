import bittensor as bt
from loguru import logger
from transformers import CLIPImageProcessor

from neurons.utils.image import synapse_to_tensors
from neurons.safety import StableDiffusionSafetyChecker

from neurons.validator.config import get_device
from neurons.validator.scoring.models.base import BaseRewardModel
from neurons.validator.scoring.models.types import RewardModelType


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
            # Clip expects RGB int values in range (0, 255)
            scaled_tensors = [
                tensor * 255 for tensor in synapse_to_tensors(response)
            ]

            clip_input = self.processor(
                scaled_tensors,
                return_tensors="pt",
            ).to(get_device())

            _, has_nsfw_concept = self.safetychecker.forward(
                images=response.images,
                clip_input=clip_input.pixel_values.to(get_device()),
            )

            return 1.0 if any(has_nsfw_concept) else 0.0

        except Exception as e:
            logger.error(f"Error in NSFW detection: {e}")
            return 0.0
