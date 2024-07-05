from enum import Enum
from typing import Any, Dict

from pydantic import ConfigDict, BaseModel

from diffusers import DiffusionPipeline


class TaskType(str, Enum):
    TEXT_TO_IMAGE = "TEXT_TO_IMAGE"
    IMAGE_TO_IMAGE = "IMAGE_TO_IMAGE"


class ModelConfig(BaseModel):
    args: Dict[str, Any]
    model: DiffusionPipeline
    model_config = ConfigDict(arbitrary_types_allowed=True)
