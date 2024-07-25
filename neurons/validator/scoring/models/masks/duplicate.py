from typing import List
import random

import torch
import numpy as np
import bittensor as bt
from loguru import logger

from neurons.utils.image import synapse_to_tensors
from neurons.validator.config import get_device, get_metagraph
from neurons.validator.scoring.models.base import BaseRewardModel
from neurons.validator.scoring.models.types import RewardModelType


class DuplicateFilter(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.DUPLICATE

    def __init__(self, pixel_check_percentage: float = 0.15):
        super().__init__()
        self.pixel_check_percentage = pixel_check_percentage

    def extract_check_pixels(self, images: List[torch.Tensor]) -> np.ndarray:
        all_check_pixels = []
        for image in images:
            height, width = image.shape[1:3]
            num_pixels = int(height * width * self.pixel_check_percentage)

            pixels_to_check = np.random.randint(
                0, (height, width), size=(num_pixels, 2)
            )

            extracted = (
                image[:, pixels_to_check[:, 0], pixels_to_check[:, 1]]
                .cpu()
                .numpy()
            )
            all_check_pixels.append(extracted.flatten())

        return np.concatenate(all_check_pixels)

    async def get_rewards(
        self,
        synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> torch.Tensor:
        logger.info("Checking for duplicate images...")

        metagraph = get_metagraph()
        mask = torch.zeros(metagraph.n).to(get_device())

        valid_responses = []
        all_check_pixels = []

        for response in responses:
            if not response.images:
                continue
            if any(image is None for image in response.images):
                continue

            images = synapse_to_tensors(response)
            all_check_pixels.append(self.extract_check_pixels(images))
            valid_responses.append(response)

        if not valid_responses:
            return mask

        all_check_pixels = np.array(all_check_pixels)

        n = len(valid_responses)
        duplicate_mask = np.zeros(n, dtype=bool)

        for i in range(n):
            if duplicate_mask[i]:
                continue
            duplicates = np.all(
                np.isclose(all_check_pixels, all_check_pixels[i], atol=1e-6),
                axis=1,
            )
            duplicate_mask |= duplicates

            # The image is not a duplicate of itself
            duplicate_mask[i] = False

        for idx, is_duplicate in enumerate(duplicate_mask):
            if is_duplicate:
                continue

            hotkey = valid_responses[idx].axon.hotkey
            try:
                metagraph_idx = metagraph.hotkeys.index(hotkey)
                mask[metagraph_idx] = 1.0
            except ValueError:
                logger.error(f"Hotkey {hotkey} not found in metagraph")

        return mask
