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
    def name(self) -> str:
        return str(RewardModelType.NSFW)

    def __init__(self):
        super().__init__()
        self.safetychecker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(get_device())
        self.processor = CLIPImageProcessor()

    def reward(self, response: bt.Synapse) -> float:
        if not response.images or any(image is None for image in response.images):
            return 0.0

        try:
            clip_input = self.processor(
                [bt.Tensor.deserialize(image) for image in response.images],
                return_tensors="pt",
            ).to(get_device())

            _, has_nsfw_concept = self.safetychecker.forward(
                images=response.images,
                clip_input=clip_input.pixel_values.to(get_device()),
            )

            return 0.0 if any(has_nsfw_concept) else 1.0

        except Exception as e:
            logger.error(f"Error in NSFW detection: {e}")
            logger.error(f"images={response.images}")
            return 1.0

    async def get_rewards(
        self,
        _synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> Dict[str, float]:
        return {
            response.dendrite.hotkey: self.reward(response) for response in responses
        }

    def normalize_rewards(self, rewards: Dict[str, float]) -> Dict[str, float]:
        return rewards
