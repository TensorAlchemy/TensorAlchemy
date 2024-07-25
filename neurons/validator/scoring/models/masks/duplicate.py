from typing import List

import torch
import numpy as np
import imagehash
from PIL import Image
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

    def __init__(self, hash_size: int = 8, threshold_ratio: float = 0.2):
        super().__init__()
        self.hash_size = hash_size
        self.threshold_ratio = threshold_ratio

    def compute_phash(self, image: torch.Tensor) -> imagehash.ImageHash:
        # Convert torch tensor to PIL Image
        img = Image.fromarray(
            (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        return imagehash.phash(img, hash_size=self.hash_size)

    def are_images_similar(
        self, hash1: imagehash.ImageHash, hash2: imagehash.ImageHash
    ) -> bool:
        max_diff = int(self.hash_size * self.hash_size * self.threshold_ratio)
        return hash1 - hash2 <= max_diff

    async def get_rewards(
        self,
        _synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> torch.Tensor:
        logger.info(f"Checking {len(responses)} images for duplicates...")

        metagraph = get_metagraph()
        mask = torch.zeros(metagraph.n).to(get_device())

        valid_responses = []
        all_hashes = []

        for response in responses:
            if not response.images:
                continue

            if any(image is None for image in response.images):
                continue

            images = synapse_to_tensors(response)
            response_hashes = [self.compute_phash(img) for img in images]
            all_hashes.append(response_hashes)
            valid_responses.append(response)

        if not valid_responses:
            return mask

        n = len(valid_responses)
        duplicate_mask = np.zeros(n, dtype=bool)

        for i in range(n):
            if duplicate_mask[i]:
                continue
            for j in range(i + 1, n):
                if len(all_hashes[i]) != len(all_hashes[j]):
                    continue
                if all(
                    self.are_images_similar(hash1, hash2)
                    for hash1, hash2 in zip(all_hashes[i], all_hashes[j])
                ):
                    duplicate_mask[i] = True
                    duplicate_mask[j] = True
                    break

        for idx, is_duplicate in enumerate(duplicate_mask):
            if is_duplicate:
                hotkey = valid_responses[idx].axon.hotkey
                if hotkey in metagraph.hotkeys:
                    mask[metagraph.hotkeys.index(hotkey)] = 1.0

        return mask
