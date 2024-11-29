from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from diffusers import AutoPipelineForInpainting

from neurons.miners.base.models import MinerState as BaseMinerState


class TaskType(str, Enum):
    """Enum defining supported generation tasks"""

    TEXT_TO_IMAGE = "TEXT_TO_IMAGE"
    IMAGE_TO_IMAGE = "IMAGE_TO_IMAGE"


class ModelConfig(BaseModel):
    """Configuration for Inpaint Diffusion pipeline"""

    # Core model components
    model: Optional[AutoPipelineForInpainting] = None

    # Generation parameters
    guidance_scale: float = Field(default=7.5)
    num_inference_steps: int = Field(default=20)
    width: int = Field(default=1024)
    height: int = Field(default=1024)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MinerState(BaseMinerState):
    """Global miner state tracking"""

    nsfw_count: int = Field(default=0)
    config: ModelConfig = ModelConfig()

    model_config = ConfigDict(arbitrary_types_allowed=True)
