import bittensor as bt

import torch
from loguru import logger

from neurons.validator.config import get_device
from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import RewardModelType


class BlacklistFilter(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.BLACKLIST

    def __init__(self):
        super().__init__()
        self.question_blacklist = []
        self.answer_blacklist = []

    def get_reward(self, response: bt.Synapse) -> float:
        # Check the number of returned images in the response
        if len(response.images) != response.num_images_per_prompt:
            return 1.0

        # If any images in the response fail the reward for that response is
        # 0.0
        for image in response.images:
            # Check if the image can be serialized
            try:
                img = bt.Tensor.deserialize(image)
            except Exception:
                logger.warning("Could not deserialise image")
                return 1.0

            # Check if the image is black image
            if img.sum() == 0:
                return 1.0

            # Check if the image has the type bt.tensor
            if not isinstance(image, bt.Tensor):
                return 1.0

            if image.shape[1] != response.width:
                return 1.0

            # check image size
            if image.shape[2] != response.height:
                return 1.0

        return 0.0
