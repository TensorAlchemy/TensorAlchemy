from loguru import logger
import torch
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    DPMSolverMultistepScheduler,
    DiffusionPipeline,
)
from transformers import CLIPImageProcessor
from neurons.protocol import ModelType
from neurons.miners.StableMiner.models import TaskType, TaskConfig
from neurons.miners.StableMiner.stable_miner import StableMiner
from scoring.models.safety import StableDiffusionSafetyChecker
from neurons.utils.log import configure_logging


def run_miner():
    configure_logging()

    task_configs = [
        TaskConfig(
            model_type=ModelType.CUSTOM,
            task_type=TaskType.TEXT_TO_IMAGE,
            pipeline=AutoPipelineForText2Image,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            scheduler=DPMSolverMultistepScheduler,
            safety_checker=StableDiffusionSafetyChecker,
            safety_checker_model_name="CompVis/stable-diffusion-safety-checker",
            processor=CLIPImageProcessor,
            refiner_class=DiffusionPipeline,
            refiner_model_name="stabilityai/stable-diffusion-xl-refiner-1.0",
        ),
        TaskConfig(
            model_type=ModelType.CUSTOM,
            task_type=TaskType.IMAGE_TO_IMAGE,
            pipeline=AutoPipelineForImage2Image,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            scheduler=DPMSolverMultistepScheduler,
            safety_checker=StableDiffusionSafetyChecker,
            safety_checker_model_name="CompVis/stable-diffusion-safety-checker",
            processor=CLIPImageProcessor,
            refiner_class=DiffusionPipeline,
            refiner_model_name="stabilityai/stable-diffusion-xl-refiner-1.0",
        ),
    ]
    logger.info("Outputting miner config:")
    logger.info(f"Task Config: {task_configs}")
    StableMiner(task_configs)
