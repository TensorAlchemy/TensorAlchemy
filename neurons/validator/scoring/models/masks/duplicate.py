from typing import List, Tuple
import random

import torch
import bittensor as bt
from loguru import logger

from neurons.utils.image import synapse_to_tensors
from neurons.validator.config import get_device
from neurons.validator.scoring.models.base import BaseRewardModel
from neurons.validator.scoring.models.types import RewardModelType


class DuplicateFilter(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.DUPLICATE

    def __init__(self, pixel_check_percentage: float = 0.15):
        super().__init__()
        self.pixel_check_percentage = pixel_check_percentage

    def extract_check_pixels(
        self, images: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        check_pixels = []

        for image in images:
            height, width = image.shape[1:3]
            num_pixels = int(height * width * self.pixel_check_percentage)

            pixels_to_check = [
                (random.randint(0, height - 1), random.randint(0, width - 1))
                for _ in range(num_pixels)
            ]

            extracted = torch.stack(
                [image[:, y, x] for y, x in pixels_to_check]
            )
            check_pixels.append(extracted)

        return check_pixels

    def is_duplicate(
        self,
        pixels1: torch.Tensor,
        pixels2: torch.Tensor,
        tolerance: float = 1e-6,
    ) -> bool:
        return torch.allclose(pixels1, pixels2, atol=tolerance)

    def get_rewards(
        self,
        _synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> torch.Tensor:
        logger.info("Checking for duplicate images...")

        rewards = torch.ones(len(responses)).to(get_device())

        # Extract check pixels for all images
        all_check_pixels = []
        for response in responses:
            if not response.images or any(
                image is None for image in response.images
            ):
                all_check_pixels.append(None)
            else:
                images = synapse_to_tensors(response)
                all_check_pixels.append(self.extract_check_pixels(images))

        # Check for duplicates
        for i in range(len(responses)):
            if all_check_pixels[i] is None:
                rewards[i] = 0.0
                continue

            for j in range(i + 1, len(responses)):
                if all_check_pixels[j] is None:
                    continue

                if len(all_check_pixels[i]) != len(all_check_pixels[j]):
                    continue

                all_duplicates = all(
                    self.is_duplicate(pixels1, pixels2)
                    for pixels1, pixels2 in zip(
                        all_check_pixels[i], all_check_pixels[j]
                    )
                )

                if all_duplicates:
                    rewards[i] = 0.0
                    rewards[j] = 0.0
                    break

        return rewards
