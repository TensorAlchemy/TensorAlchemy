import torch
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from PIL import Image
import torchvision.transforms as transforms
from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler

from neurons.protocol import ImageGeneration, IsAlive
from neurons.utils.nsfw import clean_nsfw_from_prompt
from neurons.utils.image import image_to_base64
from neurons.config import get_config

from neurons.miners.base.miner import BaseMiner
from neurons.miners.StableMiner.models import TaskType, MinerState


class StableMiner(BaseMiner):
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
            self.state.config.model = self._load_pipeline(config.alchemy_model)

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def _load_pipeline(self, model_path: str) -> AutoPipelineForInpainting:
        """Load a SDXL pipeline with standard settings"""
        pipe = AutoPipelineForInpainting.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(get_config().miner.device)

        pipe.scheduler = DEISMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        self._warm_up_pipeline(pipe)

        return pipe

    def _warm_up_pipeline(self, pipeline: AutoPipelineForInpainting) -> None:
        """Run a quick inference pass to warm up the model"""

        init_image = Image.new("RGB", (CELL_SIZE, CELL_SIZE), (0, 0, 0))
        mask = Image.new("RGB", (CELL_SIZE, CELL_SIZE), (255, 255, 255))

        with torch.no_grad():
            pipeline(
                prompt="warmup",
                image=init_image,
                mask=mask,
                num_inference_steps=1,
                width=512,
                height=512,
                num_images_per_prompt=1,
            )

    async def generate_image(self, request: Dict[str, Any]) -> List[str]:
        """Main image generation entrypoint"""
        try:
            images = []
            for attempt in range(3):
                try:
                    model_args = self._prepare_generation_args(request)
                    images = self.state.config.model(**model_args).images
                    break
                except Exception as e:
                    logger.error(
                        f"Generation attempt {attempt + 1} failed: {e}"
                    )

            return [image_to_base64(img) for img in images] if images else []

        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            return []

    def _prepare_generation_args(
        self,
        request: ImageGeneration,
    ) -> Dict[str, Any]:
        """Prepare arguments for model inference"""
        args = {
            "prompt": [clean_nsfw_from_prompt(request.prompt)],
            "width": request.width or self.state.config.width,
            "height": request.height or self.state.config.height,
            "num_images_per_prompt": request.num_images_per_prompt,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.steps,
            "generator": self._create_generator(request.seed),
            "denoising_end": 0.8,
            "output_type": "latent",
        }

        if request.negative_prompt:
            args["negative_prompt"] = [request.negative_prompt]

        if request.generation_type == TaskType.IMAGE_TO_IMAGE:
            args["image"] = self._prepare_input_image(request.prompt_image)

        return args

    def _create_generator(
        self, seed: Optional[int] = None
    ) -> List[torch.Generator]:
        """Create deterministic generator if seed provided"""
        generator = torch.Generator(device=get_config().miner.device)
        if seed is not None:
            generator.manual_seed(seed)
        return [generator]

    def _prepare_input_image(self, image_data: bytes) -> Any:
        """Convert input image data for img2img"""
        return transforms.ToPILImage()(image_data)
