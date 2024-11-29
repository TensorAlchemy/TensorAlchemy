from enum import Enum
from typing import Dict, Optional
import torch
from pydantic import BaseModel, Field, ConfigDict
from diffusers import StableDiffusionXLPipeline

from neurons.miners.base.models import MinerState as BaseMinerState


class TaskType(str, Enum):
    TEXT_TO_IMAGE = "TEXT_TO_IMAGE"
    IMAGE_TO_IMAGE = "IMAGE_TO_IMAGE"


class ModelConfig(BaseModel):
    """Configuration for a Stable Diffusion model"""

    model: Optional[StableDiffusionXLPipeline] = None
    refiner: Optional[StableDiffusionXLPipeline] = None

    # Default generation params
    guidance_scale: float = Field(default=7.5)
    num_inference_steps: int = Field(default=20)
    width: int = Field(default=1024)
    height: int = Field(default=1024)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MinerState(BaseMinerState):
    """State for the stable diffusion miner"""

    nsfw_count: int = Field(default=0)
    config: ModelConfig = ModelConfig()

    model_config = ConfigDict(arbitrary_types_allowed=True)
