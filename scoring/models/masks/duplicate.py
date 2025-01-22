from typing import List

import bittensor as bt
import imagehash
import numpy as np
import torch
from loguru import logger
from PIL import Image

from neurons.config import get_metagraph
from neurons.utils.image import synapse_to_tensors
from scoring.models.base import BaseRewardModel
from scoring.models.types import RewardModelType


class DuplicateFilter(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.DUPLICATE

    def __init__(
        self,
        # How many bytes to use for the entire image
        # More bytes is a higher resolution hash
        hash_size: int = 12,
        # How similar should they be to trigger
        # Less than this threshold = trigger
        # HIGHER is more strict
        # LOWER  is less strict
        threshold_ratio: float = 0.09,
    ):
        super().__init__()
        self.hash_size = hash_size
        self.threshold_ratio = threshold_ratio

    def compute_phash(self, image: torch.Tensor) -> imagehash.ImageHash:
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
        logger.info(f"Checking {len(responses)} responses for duplicates...")

        mask = super().zeros()

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
                similar_images = sum(
                    self.are_images_similar(hash1, hash2)
                    for hash1, hash2 in zip(all_hashes[i], all_hashes[j])
                )

                if similar_images > 0:
                    duplicate_mask[i] = True
                    duplicate_mask[j] = True
                    break

        metagraph = get_metagraph()

        for idx, is_duplicate in enumerate(duplicate_mask):
            if not is_duplicate:
                continue

            hotkey = valid_responses[idx].axon.hotkey
            if hotkey not in metagraph.hotkeys:
                continue

            logger.warning(f"Duplicate detected! {hotkey=}")
            mask[metagraph.hotkeys.index(hotkey)] = 1.0

        return mask
