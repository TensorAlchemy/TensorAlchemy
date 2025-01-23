import asyncio
import time
from typing import Optional
import httpx
from io import BytesIO

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
            asyncio.run(self.generate("Warming up the pipes", steps=1))

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
    ) -> ImageType:
        """Shared generation logic used by both generate and inpaint methods."""
        assert self.model is not None, "Model not loaded, cannot continue"

        if seed is None or seed < 0:
            seed = int(time.time())

        generator = torch.Generator(device=get_device()).manual_seed(seed)

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

    async def inpaint(
        self,
        image: ImageType,
        mask: ImageType,
        prompt: str,
        steps: int = 32,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = "",
        **_kwargs,
    ) -> ImageType:
        """Inpaint an existing image using a mask."""
        return await self.generate(
            prompt=prompt,
            image=image,
            mask=mask,
            steps=steps,
            seed=seed,
            negative_prompt=negative_prompt,
        )

    async def inpaint_image(self, synapse: ImageInpainting) -> ImageInpainting:
        """Handle image inpainting requests"""
        try:
            # Download and convert input images from URLs asynchronously
            input_image, mask_image = await asyncio.gather(
                download_image_from_url(synapse.input_image),
                download_image_from_url(synapse.mask_image),
            )

            result_image: ImageType = await self.inpaint(
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
        try:
            # Create blank image and full white mask for generation
            init_image: ImageType = Image.new(
                "RGB",
                (synapse.width, synapse.height),
                (0, 0, 0),
            )
            mask: ImageType = Image.new(
                "RGB",
                (synapse.width, synapse.height),
                (255, 255, 255),
            )

            result_image = None
            for attempt in range(3):
                try:
                    result_image = await self.generate(
                        prompt=synapse.prompt,
                        steps=synapse.steps,
                        seed=synapse.seed,
                        negative_prompt=synapse.negative_prompt,
                        height=synapse.height,
                        width=synapse.width,
                        init_image=init_image,
                        mask=mask,
                    )
                    break
                except Exception as e:
                    logger.error(
                        f"Generation attempt {attempt + 1} failed: {e}"
                    )

            if result_image:
                synapse.images = [image_to_base64(result_image)]
            else:
                logger.info(f"Failed to generate any images after 3 attempts.")

        except Exception as e:
            logger.error(f"Error in image generation: {e}")

        return synapse
