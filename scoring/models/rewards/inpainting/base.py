from abc import abstractmethod

import httpx
import torch
from loguru import logger

from neurons.protocol import BaseTask
from neurons.utils.image import base64_to_tensor
from scoring.models.base import BaseRewardModel


class BaseInpaintingModel(BaseRewardModel):
    """Base class for inpainting models that handles image downloading and preprocessing"""

    async def download_image(self, url: str) -> torch.Tensor:
        """Download and convert image to tensor"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()

                # Convert response to tensor
                return base64_to_tensor(response.content)

        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}")
            raise

    async def get_inpainting_images(
        self, response: BaseTask
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all required images for inpainting evaluation"""
        try:
            # Download original input image
            input_image = await self.download_image(response.input_image)

            # Download mask image - ensure it's binary
            mask_image = await self.download_image(response.mask_image)
            mask_image = (mask_image > 0.5).float()

            # Convert generated image from base64
            generated_image = base64_to_tensor(response.images[0])

            return input_image, mask_image, generated_image

        except Exception as e:
            logger.error(f"Error preparing inpainting images: {str(e)}")
            raise

    @abstractmethod
    async def compute_score(
        self,
        input_image: torch.Tensor,
        mask_image: torch.Tensor,
        generated_image: torch.Tensor,
    ) -> float:
        """Compute the actual score for the model"""

    async def get_reward(self, response: BaseTask) -> float:
        if not response.images:
            return 0.0

        try:
            input_image, mask_image, generated_image = (
                await self.get_inpainting_images(response)
            )
            return await self.compute_score(
                input_image, mask_image, generated_image
            )
        except Exception as e:
            logger.error(f"Error in {self.name} scoring: {str(e)}")
            return 0.0
