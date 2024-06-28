import torch
from typing import Any, Dict, Optional

import torch
from base import BaseMiner, ModelConfig, TaskType
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
)

from loguru import logger
from base import BaseMiner
from transformers import CLIPImageProcessor
from utils import warm_up

from neurons.protocol import ModelType
from neurons.safety import StableDiffusionSafetyChecker
from neurons.miners.StableMiner.schema import ModelConfig, TaskType


class StableMiner(BaseMiner):
    def __init__(self) -> None:
        self.t2i_model_custom: Optional[DiffusionPipeline] = None
        self.t2i_model_alchemy: Optional[DiffusionPipeline] = None
        self.t2i_refiner_custom: Optional[DiffusionPipeline] = None
        self.t2i_refiner_alchemy: Optional[DiffusionPipeline] = None
        self.i2i_model_custom: Optional[DiffusionPipeline] = None
        self.i2i_model_alchemy: Optional[DiffusionPipeline] = None

        super().__init__()

        self.safety_checker: Optional[StableDiffusionSafetyChecker] = None
        self.processor: Optional[CLIPImageProcessor] = None
        self.model_configs: Dict[str, Dict[str, ModelConfig]] = {}

        super().__init__()

        try:
            # Load the models
            self.load_models()

            # Optimize model
            self.optimize_models()

            # Serve the axon
            self.start_axon()

            # Start the miner loop
            self.loop()
        except Exception as e:
            logger.error(f"Error in StableMiner initialization: {e}")
            raise

    def load_models(self) -> None:
        try:
            # Text-to-image
            (
                self.t2i_model_custom,
                self.t2i_refiner_custom,
            ) = self.load_t2i_model_and_refiner(
                self.config.miner.custom_model, self.config.miner.custom_refiner
            )
            # Image-to-image
            # self.i2i_model_custom = self.load_i2i_model(self.t2i_model_custom)

            # TODO: Alchemy model
            # self.t2i_model_alchemy = None
            # self.i2i_model_alchemy = None
            self.t2i_model_alchemy = None
            self.i2i_model_alchemy = None
            self.t2i_refiner_alchemy = None

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(self.config.miner.device)
            self.processor = CLIPImageProcessor()

            self.setup_model_configs()
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def load_t2i_model_and_refiner(
        self, model_name: str, refiner_model: str
    ) -> tuple[AutoPipelineForText2Image, DiffusionPipeline]:
        try:
            model = AutoPipelineForText2Image.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to(self.config.miner.device)

            model.set_progress_bar_config(disable=True)
            model.scheduler = DPMSolverMultistepScheduler.from_config(
                model.scheduler.config
            )

            refiner = DiffusionPipeline.from_pretrained(
                refiner_model,
                text_encoder_2=model.text_encoder_2,
                vae=model.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to(self.config.miner.device)

            refiner.set_progress_bar_config(disable=True)
            refiner.scheduler = DPMSolverMultistepScheduler.from_config(
                refiner.scheduler.config
            )

            return model, refiner

        except Exception as e:
            logger.error(f"Error loading text-to-image model: {e}")
            raise

    def load_i2i_model(self, t2i_model: DiffusionPipeline) -> DiffusionPipeline:
        try:
            model = DiffusionPipeline.from_pipe(t2i_model).to(self.config.miner.device)

            model.set_progress_bar_config(disable=True)
            model.scheduler = DPMSolverMultistepScheduler.from_config(
                model.scheduler.config
            )

            return model
        except Exception as e:
            logger.error(f"Error loading image-to-image model: {e}")
            raise

    def get_model_config(
        self,
        model_type: ModelType,
        task_type: TaskType,
    ) -> ModelConfig:
        if model_type not in self.model_configs:
            raise ValueError(f"{model_type} was not found in model_configs!")

        if task_type not in self.model_configs[model_type]:
            raise ValueError(
                f"{task_type} was not found in model_configs {model_type}!"
            )

        return self.model_configs[model_type][task_type]

    def setup_model_configs(self) -> None:
        self.model_configs = {
            # TODO: Implement Alchemy Model
            # ModelType.ALCHEMY: {
            #     # Text-to-image
            #     TaskType.TEXT_TO_IMAGE: ModelConfig(
            #         args=self.t2i_args,
            #         model=self.t2i_model_alchemy,
            #         refiner=self.t2i_refiner_alchemy,
            #     ),
            #     TaskType.IMAGE_TO_IMAGE: ModelConfig(
            #         args=self.i2i_args,
            #         model=self.i2i_model_alchemy,
            #     ),
            # },
            ModelType.CUSTOM: {
                TaskType.TEXT_TO_IMAGE: ModelConfig(
                    args=self.t2i_args,
                    model=self.t2i_model_custom,
                    refiner=self.t2i_refiner_custom,
                ),
                # TODO: implement i2i
                # TaskType.IMAGE_TO_IMAGE: ModelConfig(
                #     args=self.i2i_args,
                #     model=self.i2i_model_custom,
                # ),
            },
        }

    def optimize_models(self) -> None:
        # TODO: Alchemy model
        return

        if self.config.miner.optimize:
            try:
                self.t2i_model_alchemy.unet = torch.compile(
                    self.t2i_model_alchemy.unet,
                    mode="reduce-overhead",
                    fullgraph=True,
                )

                # Warm up model
                logger.info(
                    ">>> Warming up model with compile... "
                    "this takes roughly two minutes...",
                    color="yellow",
                )
                warm_up(self.t2i_model_alchemy, self.t2i_args)
            except Exception as e:
                logger.error(f"Error optimizing models: {e}")
                raise
