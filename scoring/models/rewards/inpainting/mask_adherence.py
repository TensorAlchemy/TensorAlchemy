import torch
import torch.nn.functional as F

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

    def compute_ssim(
        self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
    ) -> float:
        """Compute SSIM between two images using PyTorch."""
        # Create a Gaussian window
        window = torch.ones(window_size, window_size)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(img1.size(1), 1, window_size, window_size)
        window = window.to(img1.device) / window.numel()

        # Calculate means
        mu1 = F.conv2d(
            img1, window, padding=window_size // 2, groups=img1.size(1)
        )
        mu2 = F.conv2d(
            img2, window, padding=window_size // 2, groups=img2.size(1)
        )

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Calculate variances and covariance
        sigma1_sq = (
            F.conv2d(
                img1 * img1,
                window,
                padding=window_size // 2,
                groups=img1.size(1),
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(
                img2 * img2,
                window,
                padding=window_size // 2,
                groups=img2.size(1),
            )
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(
                img1 * img2,
                window,
                padding=window_size // 2,
                groups=img1.size(1),
            )
            - mu1_mu2
        )

        # Constants for stability
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim_map.mean().item()

    async def compute_score(
        self,
        input_image: torch.Tensor,
        mask_image: torch.Tensor,
        generated_image: torch.Tensor,
    ) -> float:
        # Invert mask to focus on unchanged regions
        inv_mask = 1 - mask_image

        # Ensure inputs have batch dimension
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
            generated_image = generated_image.unsqueeze(0)
            inv_mask = inv_mask.unsqueeze(0)

        # Calculate SSIM for unchanged regions
        masked_input = input_image * inv_mask
        masked_generated = generated_image * inv_mask

        ssim_score = self.compute_ssim(masked_input, masked_generated)

        # Calculate pixel-wise difference in unchanged regions
        pixel_diff = torch.mean(
            torch.abs(masked_input - masked_generated)
        ).item()
        pixel_score = 1.0 - (pixel_diff / 255.0)

        # Combine scores
        return self.ssim_weight * ssim_score + self.pixel_weight * pixel_score
