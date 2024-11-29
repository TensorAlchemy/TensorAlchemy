import torch
import asyncio
import traceback
from typing import List, Dict, Any, Tuple

import bittensor as bt
from loguru import logger
from diffusers import StableDiffusionXLPipeline
from diffusers.callbacks import SDXLCFGCutoffCallback
import torchvision.transforms as transforms

from neurons.protocol import ImageGeneration, IsAlive
from neurons.utils.nsfw import clean_nsfw_from_prompt
from neurons.utils.image import image_to_base64
from neurons.config import get_config

from neurons.miners.base.miner import BaseMiner
from neurons.miners.StableMiner.models import TaskType, MinerState, ModelConfig


class StableMiner(BaseMiner):
    def __init__(self) -> None:
        self.state = MinerState()
        # Add this before axon.attach in create_axon
        super().__init__()

    def create_attachments(self) -> None:
        """Return list of forward function tuples for axon"""

        async def isalive(synapse: IsAlive) -> IsAlive:
            return synapse

        async def forward(synapse: ImageGeneration) -> ImageGeneration:
            return await self.generate_image(synapse)

        async def blacklist(synapse: ImageGeneration) -> Tuple[bool, str]:
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
        # Defensive check for config
        if not hasattr(self.state, "config"):
            logger.warning("Config not found, creating new ModelConfig")
            self.state.config = ModelConfig()

        try:
            logger.info("Loading SDXL model...")
            self.state.config.model = StableDiffusionXLPipeline.from_pretrained(
                get_config().miner.alchemy_model,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to(get_config().miner.device)

            if get_config().refiner.enable:
                logger.info("Loading SDXL refiner...")
                self.state.config.refiner = (
                    StableDiffusionXLPipeline.from_pretrained(
                        get_config().miner.alchemy_refiner,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True,
                    ).to(get_config().miner.device)
                )

            # Optimize if enabled
            if get_config().miner.optimize:
                self._optimize_models()

        except Exception:
            logger.error(
                #
                "Failed to initialize models: "
                + traceback.format_exc()
            )
            raise

    def _optimize_models(self) -> None:
        """Compile and optimize loaded models"""
        try:
            # Optimize base model
            if self.state.config.model:
                logger.info("Optimizing base model")
                self.state.config.model.unet = torch.compile(
                    self.state.config.model.unet,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                self._warm_up_model(self.state.config.model)

            # Optimize refiner if present
            if self.state.config.refiner:
                logger.info("Optimizing refiner")
                self.state.config.refiner.unet = torch.compile(
                    self.state.config.refiner.unet,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                self._warm_up_model(self.state.config.refiner)

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
        if self.state.config.refiner and get_config().refiner.enable:
            # First pass with base model
            images = self.state.config.model(**model_args).images

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
            images = self.state.config.refiner(**refiner_args).images
        else:
            # Single pass without refiner
            args = {
                k: v
                for k, v in model_args.items()
                if k not in ["denoising_end", "output_type"]
            }
            images = self.state.config.model(**args).images

        return images

    def _setup_model_args(
        self,
        synapse: ImageGeneration,
    ) -> Dict[str, Any]:
        """Setup model arguments from synapse request"""
        args = {
            "prompt": [clean_nsfw_from_prompt(synapse.prompt)],
            "width": synapse.width or self.state.config.width,
            "height": synapse.height or self.state.config.height,
            "num_images_per_prompt": synapse.num_images_per_prompt,
            "guidance_scale": synapse.guidance_scale
            or self.state.config.guidance_scale,
            "num_inference_steps": getattr(
                synapse,
                "steps",
                self.state.config.num_inference_steps,
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
