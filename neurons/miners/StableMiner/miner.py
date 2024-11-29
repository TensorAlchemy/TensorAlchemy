import torch
import asyncio
from typing import List, Dict, Any, Tuple
from loguru import logger

import bittensor as bt
from diffusers import StableDiffusionXLPipeline
from diffusers.callbacks import SDXLCFGCutoffCallback
import torchvision.transforms as transforms

from neurons.protocol import ImageGeneration, IsAlive
from neurons.miners.base.miner import BaseMiner
from neurons.miners.StableMiner.models import TaskType, StableMinerState
from neurons.utils.nsfw import clean_nsfw_from_prompt
from neurons.utils.image import image_to_base64
from neurons.config import get_config


class StableMiner(BaseMiner):
    def __init__(self) -> None:
        self.stable_state = StableMinerState()
        super().__init__()

    def initialize_implementation(self) -> None:
        """Initialize SDXL model"""
        try:
            logger.info("Loading SDXL model...")
            self.stable_state.model_config.model = (
                StableDiffusionXLPipeline.from_pretrained(
                    get_config().miner.model_name,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                ).to(get_config().miner.device)
            )

            if get_config().refiner.enable:
                logger.info("Loading SDXL refiner...")
                self.stable_state.model_config.refiner = (
                    StableDiffusionXLPipeline.from_pretrained(
                        get_config().refiner.model_name,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True,
                    ).to(get_config().miner.device)
                )

            # Optimize if enabled
            if get_config().miner.optimize:
                self._optimize_models()

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def _optimize_models(self) -> None:
        """Compile and optimize loaded models"""
        try:
            # Optimize base model
            if self.stable_state.model_config.model:
                logger.info("Optimizing base model")
                self.stable_state.model_config.model.unet = torch.compile(
                    self.stable_state.model_config.model.unet,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                self._warm_up_model(self.stable_state.model_config.model)

            # Optimize refiner if present
            if self.stable_state.model_config.refiner:
                logger.info("Optimizing refiner")
                self.stable_state.model_config.refiner.unet = torch.compile(
                    self.stable_state.model_config.refiner.unet,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                self._warm_up_model(self.stable_state.model_config.refiner)

        except Exception as e:
            logger.error(f"Failed to optimize models: {e}")
            raise

    async def generate_image(self, synapse: ImageGeneration) -> ImageGeneration:
        """Handle image generation request"""
        try:
            generated_images = await self._attempt_generate_images(synapse)

            if generated_images:
                synapse.images = generated_images
                synapse.completion = "success"
            else:
                synapse.completion = "failed"

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            synapse.completion = "failed"

        return synapse

    async def _attempt_generate_images(
        self,
        synapse: ImageGeneration,
    ) -> List[str]:
        """Attempt to generate images with retries"""
        images = []
        for attempt in range(3):
            try:
                # Setup generation arguments
                model_args = self._setup_model_args(synapse)

                # Set generator for reproducibility
                model_args["generator"] = [
                    torch.Generator(
                        device=get_config().miner.device
                    ).manual_seed(synapse.seed)
                ]

                # Generate images
                images = self.generate_with_refiner(model_args)
                logger.info(
                    f"Image generation successful after {attempt + 1} attempt(s)"
                )
                break

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(5)

        if not images:
            logger.error("Failed to generate images after all attempts")
            return []

        return [image_to_base64(image) for image in images]

    def generate_with_refiner(
        self,
        model_args: Dict[str, Any],
    ) -> List[Any]:
        """Generate images with optional refinement step"""
        if (
            self.stable_state.model_config.refiner
            and get_config().refiner.enable
        ):
            # First pass with base model
            images = self.stable_state.model_config.model(**model_args).images

            # Setup refiner args
            refiner_args = {
                "denoising_start": model_args["denoising_end"],
                "prompt": model_args["prompt"],
                "num_inference_steps": int(
                    model_args["num_inference_steps"] * 0.2
                ),
                "image": images,
            }

            # Refine images
            images = self.stable_state.model_config.refiner(
                **refiner_args
            ).images
        else:
            # Single pass without refiner
            args = {
                k: v
                for k, v in model_args.items()
                if k not in ["denoising_end", "output_type"]
            }
            images = self.stable_state.model_config.model(**args).images

        return images

    def _setup_model_args(
        self,
        synapse: ImageGeneration,
    ) -> Dict[str, Any]:
        """Setup model arguments from synapse request"""
        args = {
            "prompt": [clean_nsfw_from_prompt(synapse.prompt)],
            "width": synapse.width or self.stable_state.model_config.width,
            "height": synapse.height or self.stable_state.model_config.height,
            "num_images_per_prompt": synapse.num_images_per_prompt,
            "guidance_scale": synapse.guidance_scale
            or self.stable_state.model_config.guidance_scale,
            "num_inference_steps": getattr(
                synapse,
                "steps",
                self.stable_state.model_config.num_inference_steps,
            ),
            "denoising_end": 0.8,
            "output_type": "latent",
        }

        if synapse.negative_prompt:
            args["negative_prompt"] = [synapse.negative_prompt]

        # Handle image-to-image
        if synapse.generation_type == TaskType.IMAGE_TO_IMAGE:
            args["image"] = transforms.ToPILImage()(
                bt.Tensor.deserialize(synapse.prompt_image)
            )

        return args

    def _warm_up_model(self, model: StableDiffusionXLPipeline) -> None:
        """Run a warm-up pass through the model"""
        try:
            logger.info("Running model warm-up...")

            args = {
                "prompt": ["warm up pass"],
                "num_inference_steps": 1,
                "width": 512,
                "height": 512,
                "num_images_per_prompt": 1,
            }

            with torch.no_grad():
                _ = model(**args)

            logger.info("Model warm-up complete")

        except Exception as e:
            logger.error(f"Error during model warm-up: {e}")
            raise
