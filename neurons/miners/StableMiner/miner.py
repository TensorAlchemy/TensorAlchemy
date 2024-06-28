import torch
from typing import Dict, Any
from enum import Enum
from base import BaseMiner
from pydantic import BaseModel


from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
)
from neurons.protocol import ModelType
from neurons.safety import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from utils import warm_up
from utils.log import colored_log


class TaskType(str, Enum):
    TEXT_TO_IMAGE = "TEXT_TO_IMAGE"
    IMAGE_TO_IMAGE = "IMAGE_TO_IMAGE"


class ModelConfig(BaseModel):
    args: Dict[str, Any]
    model: AutoPipelineForText2Image


class StableMiner(BaseMiner):
    def __init__(self) -> None:
        super().__init__()

        self.t2i_model_custom: AutoPipelineForText2Image
        self.t2i_model_alchemy: AutoPipelineForText2Image
        self.i2i_model_custom: AutoPipelineForImage2Image
        self.i2i_model_alchemy: AutoPipelineForImage2Image
        self.safety_checker: StableDiffusionSafetyChecker
        self.processor: CLIPImageProcessor
        self.model_configs: Dict[str, Dict[str, Any]]

        # Load the models
        self.load_models()

        # Optimize model
        self.optimize_models()

        # Serve the axon
        self.start_axon()

        # Start the miner loop
        self.loop()

    def load_models(self) -> None:
        # Text-to-image
        self.t2i_model_custom = self.load_t2i_model(self.config.miner.model)

        # Image-to-image
        self.i2i_model_custom = self.load_i2i_model(self.t2i_model_custom)

        # TODO: Alchemy model
        self.t2i_model_alchemy = (
            None  # self.load_t2i_model(self.config.miner.alchemy_model)
        )
        self.i2i_model_alchemy = None  # self.load_i2i_model(self.t2i_model_alchemy)

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.config.miner.device)
        self.processor = CLIPImageProcessor()

        self.setup_model_configs()

    def load_t2i_model(self, model_name: str) -> AutoPipelineForText2Image:
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

        return model

    def load_i2i_model(
        self, t2i_model: AutoPipelineForText2Image
    ) -> AutoPipelineForImage2Image:
        model = AutoPipelineForImage2Image.from_pipe(t2i_model).to(
            self.config.miner.device
        )

        model.set_progress_bar_config(disable=True)
        model.scheduler = DPMSolverMultistepScheduler.from_config(
            model.scheduler.config
        )

        return model

    def get_model_config(
        self,
        model_type: ModelType,
        task_type: TaskType,
    ) -> ModelConfig:
        if model_type not in self.model_configs:
            raise ValueError(f"{model_type} was not found in model_configs!")

        if task_type not in self.model_configs[model_type]:
            raise ValueError(
                #
                task_type
                + f"was not found in model_configs {model_type}!"
            )

        return self.model_configs[model_type][task_type]

    def setup_model_configs(self) -> None:
        self.model_configs = {
            ModelType.ALCHEMY: {
                # Text-to-image
                TaskType.TEXT_TO_IMAGE: ModelConfig(
                    args=self.t2i_args,
                    model=self.t2i_model_alchemy,
                ),
                TaskType.IMAGE_TO_IMAGE: ModelConfig(
                    args=self.i2i_args,
                    model=self.i2i_model_alchemy,
                ),
            },
            ModelType.CUSTOM: {
                TaskType.TEXT_TO_IMAGE: ModelConfig(
                    args=self.t2i_args,
                    model=self.t2i_model_custom,
                ),
                TaskType.IMAGE_TO_IMAGE: ModelConfig(
                    args=self.i2i_args,
                    model=self.i2i_model_custom,
                ),
            },
        }

    def optimize_models(self) -> None:
        # TODO: Alchemy model
        return

        if self.config.miner.optimize:
            self.t2i_model_alchemy.unet = torch.compile(
                self.t2i_model_alchemy.unet, mode="reduce-overhead", fullgraph=True
            )

            # Warm up model
            colored_log(
                ">>> Warming up model with compile... "
                "this takes roughly two minutes...",
                color="yellow",
            )
            warm_up(self.t2i_model_alchemy, self.t2i_args)
