import bittensor as bt
from typing import Dict, List
from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import RewardModelType


class BlacklistFilter(BaseRewardModel):
    @property
    def name(self) -> str:
        return str(RewardModelType.BLACKLIST)

    def __init__(self):
        super().__init__()
        self.question_blacklist = []
        self.answer_blacklist = []

    def reward(self, response: bt.Synapse) -> float:
        # Check the number of returned images in the response
        if len(response.images) != response.num_images_per_prompt:
            return 0.0

        # If any images in the response fail the reward for that response is 0.0
        for image in response.images:
            # Check if the image can be serialized
            try:
                img = bt.Tensor.deserialize(image)
            except Exception:
                return 0.0

            # Check if the image is black image
            if img.sum() == 0:
                return 0.0

            # Check if the image has the type bt.tensor
            if not isinstance(image, bt.Tensor):
                return 0.0

            # check image size
            if not (
                (image.shape[1] == response.height)
                and (image.shape[2] == response.width)
            ):
                return 0.0
        return 1.0

    async def get_rewards(
        self,
        _synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> Dict[int, float]:
        return {
            response.dendrite.hotkey: self.reward(response) for response in responses
        }

    def normalize_rewards(self, rewards: Dict[int, float]) -> Dict[int, float]:
        return rewards
