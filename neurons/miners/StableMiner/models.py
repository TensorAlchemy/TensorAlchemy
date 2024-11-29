from enum import Enum
from typing import Dict, Optional
import torch
from pydantic import BaseModel, Field
from diffusers import StableDiffusionXLPipeline


class TaskType(str, Enum):
    TEXT_TO_IMAGE = "TEXT_TO_IMAGE"
    IMAGE_TO_IMAGE = "IMAGE_TO_IMAGE"


class StableModelConfig(BaseModel):
    """Configuration for a Stable Diffusion model"""

    model: Optional[StableDiffusionXLPipeline] = None
    refiner: Optional[StableDiffusionXLPipeline] = None

    # Default generation params
    guidance_scale: float = Field(default=7.5)
    num_inference_steps: int = Field(default=20)
    width: int = Field(default=1024)
    height: int = Field(default=1024)

    class Config:
        arbitrary_types_allowed = True


class StableMinerState(BaseModel):
    """State for the stable diffusion miner"""

    model_config: StableModelConfig = Field(default_factory=StableModelConfig)
    nsfw_count: int = Field(default=0)

    class Config:
        arbitrary_types_allowed = True
