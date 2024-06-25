import ImageReward as RM
import bittensor as bt
import torch
import torchvision.transforms as transforms
from loguru import logger

from neurons.validator.config import get_device
from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import RewardModelType


class ImageRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return str(RewardModelType.IMAGE)

    def __init__(self):
        super().__init__()
        self.scoring_model = RM.load("ImageReward-v1.0", device=get_device())

    def reward(self, response: torch.FloatTensor) -> float:
        try:
            with torch.no_grad():
                images = [
                    transforms.ToPILImage()(bt.Tensor.deserialize(image))
                    for image in response.images
                ]
                _, scores = self.scoring_model.inference_rank(response.prompt, images)

                image_scores = torch.tensor(scores)
                mean_image_scores = torch.mean(image_scores)

                return mean_image_scores

        except Exception:
            logger.error("ImageReward score is 0. No image in response.")
            return 0.0

    async def get_rewards(
        self,
        _synapse: bt.Synapse,
        responses: torch.FloatTensor,
        rewards: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return torch.tensor(
            [self.reward(response) for response in responses],
            dtype=torch.float32,
        )
