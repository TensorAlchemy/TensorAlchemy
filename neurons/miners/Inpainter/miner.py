import asyncio
import time
from io import BytesIO
from typing import Optional, Tuple

import httpx
import torch
from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler
from loguru import logger
from PIL import Image
from PIL.Image import Image as ImageType

from neurons.config import get_config, get_device
from neurons.miners.base.miner import BaseMiner
from neurons.protocol import ImageGeneration, ImageInpainting, IsAlive
from neurons.utils.image import image_to_base64
from neurons.utils.nsfw import clean_nsfw_from_prompt


def empty_image(
    width: int,
    height: int,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> ImageType:
    return Image.new("RGB", (width, height), color)


async def download_image_from_url(url: str) -> ImageType:
    """Download an image from a URL and convert to PIL Image asynchronously using httpx"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))


class InpaintMiner(BaseMiner):
    model: Optional[AutoPipelineForInpainting] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def create_attachments(self) -> None:
        """Return list of forward function tuples for axon"""

        logger.info("Attaching synapses")

        # IsAlive synapse (default bound)
        self.attach_synapse(IsAlive)

        # ImageGeneration synapse
        self.attach_synapse(ImageGeneration, self.generate_image)

        # ImageInpainting synapse
        self.attach_synapse(ImageInpainting, self.inpaint_image)

    def initialize_implementation(self) -> None:
        """Initialize SDXL model"""
        try:
            config = get_config().miner
            self.model = AutoPipelineForInpainting.from_pretrained(
                config.alchemy_model,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to(get_config().miner.device)

            self.model.scheduler = DEISMultistepScheduler.from_config(
                self.model.scheduler.config
            )

            logger.info("Generate image to warm up model")
            asyncio.run(
                self.generate(
                    "Warming up the pipes",
                    steps=1,
                    image=empty_image(64, 64, (0, 0, 0)),
                    mask=empty_image(64, 64, (255, 255, 255)),
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        image: ImageType,
        mask: ImageType,
        steps: int = 32,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = "",
        max_retries: int = 3,
    ) -> Optional[ImageType]:
        """Shared generation logic used by both generate and inpaint methods.

        Args:
            prompt: The text prompt for generation
            image: The input image to modify
            mask: The mask indicating areas to modify
            steps: Number of inference steps
            width: Output image width
            height: Output image height
            seed: Random seed for generation
            negative_prompt: Text prompt for what to avoid
            max_retries: Maximum number of retry attempts on failure

        Returns:
            The generated image or None if all retries failed
        """
        assert self.model is not None, "Model not loaded, cannot continue"

        if seed is None or seed < 0:
            seed = int(time.time())

        generator = torch.Generator(device=get_device()).manual_seed(seed)

        for attempt in range(max_retries):
            try:
                with torch.inference_mode():
                    result = self.model(
                        prompt=clean_nsfw_from_prompt(prompt),
                        width=width,
                        height=height,
                        image=image,
                        mask_image=mask,
                        generator=generator,
                        num_inference_steps=steps,
                        negative_prompt=negative_prompt,
                    )
                    return result.images[0]
            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed all {max_retries} generation attempts"
                    )
                    return None
                continue

    async def inpaint_image(self, synapse: ImageInpainting) -> ImageInpainting:
        """Handle image inpainting requests"""
        try:
            # Download and convert input images from URLs asynchronously
            input_image, mask_image = await asyncio.gather(
                download_image_from_url(synapse.input_image),
                download_image_from_url(synapse.mask_image),
            )

            result_image: Optional[ImageType] = await self.generate(
                image=input_image,
                mask=mask_image,
                negative_prompt=synapse.negative_prompt,
                prompt=synapse.prompt,
                seed=synapse.seed,
                steps=synapse.steps,
            )

            if result_image:
                synapse.images = [image_to_base64(result_image)]

        except Exception as e:
            logger.error(f"Error in image inpainting: {e}")

        return synapse

    async def generate_image(self, synapse: ImageGeneration) -> ImageGeneration:
        """Main image generation entrypoint that maintains Synapse protocol"""
        # Create blank image and full white mask for generation
        try:
            init_image: ImageType = empty_image(
                synapse.width,
                synapse.height,
                color=(0, 0, 0),
            )
            mask: ImageType = empty_image(
                synapse.width,
                synapse.height,
                color=(255, 255, 255),
            )

            result_image: Optional[ImageType] = await self.generate(
                height=synapse.height,
                image=init_image,
                mask=mask,
                negative_prompt=synapse.negative_prompt,
                prompt=synapse.prompt,
                seed=synapse.seed,
                steps=synapse.steps,
                width=synapse.width,
            )

            if result_image:
                synapse.images = [image_to_base64(result_image)]

        except Exception as e:
            logger.error(f"Error in image generation: {e}")

        return synapse
