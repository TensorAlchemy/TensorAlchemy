import torch
import time
from typing import List, Optional, Tuple
from loguru import logger
from PIL import Image
from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler

from neurons.protocol import ImageGeneration, IsAlive
from neurons.utils.nsfw import clean_nsfw_from_prompt
from neurons.utils.image import image_to_base64
from neurons.config import get_config, get_device

from neurons.miners.base.miner import BaseMiner
from neurons.miners.InpaintMiner.models import MinerState


class InpaintMiner(BaseMiner):
    def __init__(self, **kwargs) -> None:
        self.state = MinerState()

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

        # IsAlive synapse
        self.axon.attach(forward_fn=isalive)

        # Generate synapse
        self.axon.attach(
            forward_fn=forward,
            blacklist_fn=blacklist,
        )

    def initialize_implementation(self) -> None:
        """Initialize SDXL model"""
        try:
            config = get_config().miner
            pipe = AutoPipelineForInpainting.from_pretrained(
                config.alchemy_model,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to(get_config().miner.device)

            pipe.scheduler = DEISMultistepScheduler.from_config(
                pipe.scheduler.config
            )

            self.state.config.model = pipe
            self.generate("Warming up the pipes")

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        """Run a quick inference pass to warm up the model"""

        init_image = Image.new("RGB", (1024, 1024), (0, 0, 0))
        mask = Image.new("RGB", (1024, 1024), (255, 255, 255))

        if seed is None:
            seed = int(time.time())

        generator = torch.Generator(device=get_device()).manual_seed(seed)

        with torch.inference_mode():
            return self.state.config.model(
                prompt=prompt,
                image=init_image,
                mask_image=mask,
                generator=generator,
                num_inference_steps=steps,
                negative_prompt=negative_prompt,
            )

    async def generate_image(self, request: ImageGeneration) -> List[str]:
        """Main image generation entrypoint"""
        try:
            images = []
            for attempt in range(3):
                try:
                    images = self.generate(
                        prompt=clean_nsfw_from_prompt(request.prompt),
                        negative_prompt=request.negative_prompt,
                        steps=request.steps,
                        seed=request.seed,
                    ).images
                    break
                except Exception as e:
                    logger.error(
                        f"Generation attempt {attempt + 1} failed: {e}"
                    )

            return [image_to_base64(img) for img in images] if images else []

        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            return []
