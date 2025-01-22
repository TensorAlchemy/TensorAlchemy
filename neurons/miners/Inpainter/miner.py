import asyncio
import time
from typing import Optional, Tuple

import torch
from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler
from loguru import logger
from PIL import Image

from neurons.config import get_config, get_device
from neurons.miners.base.miner import BaseMiner
from neurons.protocol import ImageGeneration, IsAlive
from neurons.utils.image import image_to_base64
from neurons.utils.nsfw import clean_nsfw_from_prompt


class InpaintMiner(BaseMiner):
    model: AutoPipelineForInpainting = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def create_attachments(self) -> None:
        """Return list of forward function tuples for axon"""

        async def isalive(synapse: IsAlive) -> IsAlive:
            logger.info("Received IsAlive synapse")
            return synapse

        async def forward(synapse: ImageGeneration) -> ImageGeneration:
            logger.info("Received ImageGeneration synapse")
            return await self.generate_image(synapse)

        async def blacklist(synapse: ImageGeneration) -> Tuple[bool, str]:
            logger.info("Received ImageGeneration blacklist")
            return await self._base_blacklist(synapse)

        async def blacklist_isalive(synapse: IsAlive) -> Tuple[bool, str]:
            logger.info("Received ImageGeneration blacklist")
            return await self._base_blacklist(synapse)

        async def priority(synapse: ImageGeneration) -> float:
            logger.info("Received ImageGeneration priority")
            return await self._base_priority(synapse)

        async def priority_isalive(synapse: IsAlive) -> float:
            logger.info("Received IsAlive priority")
            return await self._base_priority(synapse)

        logger.info("Setting up miner Bittensor attachments...")

        # IsAlive synapse
        self.axon.attach(
            forward_fn=isalive,
            priority_fn=priority_isalive,
            blacklist_fn=blacklist_isalive,
        )

        # Generate synapse
        self.axon.attach(
            forward_fn=forward,
            priority_fn=priority,
            blacklist_fn=blacklist,
        )

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
        steps: int = 32,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = "",
        height: int = 1024,
        width: int = 1024,
    ) -> Image.Image:
        """Generate a new image from scratch using inpainting model."""
        if seed is None:
            seed = int(time.time())

        # Create blank image and full white mask for generation
        init_image = Image.new("RGB", (width, height), (0, 0, 0))
        mask = Image.new("RGB", (width, height), (255, 255, 255))
        generator = torch.Generator(device=get_device()).manual_seed(seed)

        with torch.inference_mode():
            result = self.model(
                prompt=clean_nsfw_from_prompt(prompt),
                image=init_image,
                mask_image=mask,
                generator=generator,
                num_inference_steps=steps,
                negative_prompt=negative_prompt,
            )
            return result.images[0]

    async def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        steps: int = 32,
        seed: Optional[int] = None,
        negative_prompt: str = "",
        **_kwargs,
    ) -> ImageGeneration:
        """Inpaint an existing image using a mask."""
        if seed is None:
            seed = int(time.time())

        generator = torch.Generator(device=get_device()).manual_seed(seed)

        with torch.inference_mode():
            result = self.model(
                prompt=clean_nsfw_from_prompt(prompt),
                image=image,
                mask_image=mask,
                generator=generator,
                num_inference_steps=steps,
                negative_prompt=negative_prompt,
            )
            return result.images[0]

    async def generate_image(self, synapse: ImageGeneration) -> ImageGeneration:
        """Main image generation entrypoint that maintains Synapse protocol"""
        try:
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
