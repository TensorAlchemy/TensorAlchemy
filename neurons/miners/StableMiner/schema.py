from enum import Enum
from typing import Any, Dict, Type, Optional

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
    task_type: TaskType
    pipeline: Type
    torch_dtype: torch.dtype
    use_safetensors: bool
    variant: str
    scheduler: Optional[Type] = None
    safety_checker: Optional[Type] = None
    processor: Optional[Type] = None

    class Config:
        arbitrary_types_allowed = True
