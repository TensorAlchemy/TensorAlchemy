import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from loguru import logger

from scoring.models.rewards.inpainting.base import BaseInpaintingModel
from scoring.models.types import RewardModelType


class BoundaryCoherenceModel(BaseInpaintingModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.BOUNDARY_COHERENCE

    def __init__(self):
        super().__init__()
        # Create horizontal and vertical Sobel kernels
        self.kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        # Kernel for finding boundaries
        self.boundary_kernel = torch.ones(1, 1, 5, 5)

    def detect_edges(self, image: torch.Tensor) -> torch.Tensor:
        # Convert to grayscale if needed
        if image.shape[0] == 3:
            image = TF.rgb_to_grayscale(image)

        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        image = image.unsqueeze(0)  # Add batch dimension

        # Apply Sobel filters
        grad_x = F.conv2d(image, self.kernel_x.to(image.device), padding=1)
        grad_y = F.conv2d(image, self.kernel_y.to(image.device), padding=1)

        # Calculate gradient magnitude
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        return magnitude.squeeze(0)

    def get_boundary_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Creates a mask that only highlights the boundary regions
        between masked and unmasked areas.
        """
        # Dilate mask
        dilated = F.conv2d(
            mask.unsqueeze(0), self.boundary_kernel.to(mask.device), padding=2
        )

        # Erode mask
        eroded = (
            F.conv2d(
                mask.unsqueeze(0),
                self.boundary_kernel.to(mask.device),
                padding=2,
            )
            >= 25
        )  # All 25 pixels must be 1 for erosion

        # Boundary is the difference between dilated and eroded
        # This gives us just the transition region
        boundary = (dilated > 0) & ~eroded

        # Log boundary region size for debugging
        boundary_size = torch.sum(boundary).item()
        logger.debug(f"Boundary region size: {boundary_size} pixels")

        return boundary.squeeze(0)

    async def compute_score(
        self,
        input_image: torch.Tensor,
        mask_image: torch.Tensor,
        generated_image: torch.Tensor,
    ) -> float:

        try:
            # Detect edges in both images
            original_edges = self.detect_edges(input_image)
            generated_edges = self.detect_edges(generated_image)

            # Get just the boundary region mask
            boundary_mask = self.get_boundary_mask(mask_image)

            # Compare edge consistency only in boundary regions
            edge_diff = torch.abs(original_edges - generated_edges)

            # Only consider differences in the boundary regions
            masked_diff = edge_diff * boundary_mask

            # Calculate score based on boundary regions only
            # Add small epsilon to avoid division by zero
            total_boundary_pixels = torch.sum(boundary_mask) + 1e-6
            boundary_score = (
                1.0 - (torch.sum(masked_diff) / total_boundary_pixels).item()
            )

            logger.info(f"Boundary coherence score: {boundary_score:.4f}")
            return boundary_score

        except Exception as e:
            logger.error(f"Error in boundary coherence scoring: {str(e)}")
            return 0.0
