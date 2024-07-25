import numpy as np
import bittensor as bt
from loguru import logger

from neurons.utils.log import image_to_str
from neurons.utils.image import synapse_to_images

from neurons.validator.scoring.models.base import BaseRewardModel
from neurons.validator.scoring.models.types import RewardModelType


class BlacklistFilter(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.BLACKLIST

    def __init__(self):
        super().__init__()
        self.answer_blacklist = []
        self.question_blacklist = []

    def get_reward(self, response: bt.Synapse) -> float:
        # Check the number of returned images in the response
        if len(response.images) != response.num_images_per_prompt:
            return 1.0

        # If any images in the response fail
        # the reward for that response is 0.0
        for image in synapse_to_images(response):
            logger.info(image_to_str(image))

            # Check if the image is black image
            if np.array(image).sum() < 1:
                return 1.0

            if image.width != response.width:
                return 1.0

            # check image size
            if image.height != response.height:
                return 1.0

        return 0.0
