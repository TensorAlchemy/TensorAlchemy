from typing import List

import bittensor as bt
import ImageReward as RM
import torch
from PIL.Image import Image as ImageType

from neurons.config import get_device
from neurons.protocol import BaseTask
from neurons.utils.image import synapse_to_images
from scoring.models.base import BaseRewardModel
from scoring.models.types import RewardModelType


class ImageRewardModel(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.IMAGE

    def __init__(self):
        super().__init__()
        self.scoring_model = RM.load("ImageReward-v1.0", device=get_device())

    async def get_reward(self, response: BaseTask) -> float:
        with torch.no_grad():
            try:
                images: List[ImageType] = synapse_to_images(response)

                if not images:
                    raise ValueError("No images")

            except Exception:
                return 0.0

            _, scores = self.scoring_model.inference_rank(
                response.prompt,
                images,
            )

            image_scores = torch.tensor(scores)
            mean_image_score = torch.mean(image_scores)

            return mean_image_score.item()
