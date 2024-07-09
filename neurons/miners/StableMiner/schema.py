from enum import Enum
from typing import Any, Dict, Type

from pydantic import ConfigDict, BaseModel

from diffusers import DiffusionPipeline
import torch


class TaskType(str, Enum):
    TEXT_TO_IMAGE = "TEXT_TO_IMAGE"
    IMAGE_TO_IMAGE = "IMAGE_TO_IMAGE"


class ModelConfig(BaseModel):
    args: Dict[str, Any]
    model: DiffusionPipeline
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TaskConfig(BaseModel):
    pipeline: Type
    torch_dtype: torch.dtype
    use_safetensors: bool
    variant: str

    class Config:
        arbitrary_types_allowed = True
