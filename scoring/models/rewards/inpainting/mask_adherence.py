import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from neurons.protocol import BaseTask
from neurons.utils.image import synapse_to_images
from scoring.models.base import BaseRewardModel
from scoring.models.rewards.inpainting.base import BaseInpaintingModel
from scoring.models.types import RewardModelType


class MaskAdherenceModel(BaseInpaintingModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.MASK_ADHERENCE

    def __init__(self):
        super().__init__()
        self.ssim_weight = 0.7
        self.pixel_weight = 0.3

    async def compute_score(
        self,
        input_image: torch.Tensor,
        mask_image: torch.Tensor,
        generated_image: torch.Tensor,
    ) -> float:
        # Convert tensors to numpy arrays for SSIM
        original = input_image.numpy()
        generated = generated_image.numpy()
        mask = mask_image.numpy()

        # Invert mask to focus on unchanged regions
        inv_mask = 1 - mask

        # Calculate SSIM for unchanged regions
        ssim_score = ssim(
            original * inv_mask, generated * inv_mask, multichannel=True
        )

        # Calculate pixel-wise difference in unchanged regions
        pixel_diff = np.mean(np.abs((original - generated) * inv_mask))
        pixel_score = 1.0 - (pixel_diff / 255.0)

        # Combine scores
        return self.ssim_weight * ssim_score + self.pixel_weight * pixel_score
