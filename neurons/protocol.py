from enum import Enum
from typing import Any, List, Optional, Union

import bittensor as bt
import numpy as np
import torch
from pydantic import ConfigDict, Field, field_validator


class TaskType(str, Enum):
    TEXT_TO_IMAGE = "TEXT_TO_IMAGE"
    INPAINT_IMAGE = "INPAINT_IMAGE"

    def __str__(self):
        return self.value

    def to_json(self):
        return self.value


class ModelType(str, Enum):
    SCORING = "SCORING"
    ALCHEMY = "ALCHEMY"
    CUSTOM = "CUSTOM"
    
    def __str__(self):
        return self.value
        
    def to_json(self):
        return self.value


class IsAlive(bt.Synapse):
    answer: Optional[str] = None
    completion: str = Field(
        "",
        title="Completion",
        description="Completion status of the current ImageGeneration object."
        + " This attribute is mutable and can be updated.",
    )


SupportedImageTypes = Union[str, np.ndarray, torch.tensor, bt.Tensor]


def deserialize_incoming_image(inbound_image: Any):
    """Inbound image type is different across different miner versions."""
    from neurons.utils.image import image_to_base64, tensor_to_image

    if isinstance(inbound_image, str):
        # Newest miners already send image as base64 string
        return inbound_image

    if isinstance(inbound_image, dict) and "buffer" in inbound_image:
        # Older miners serializing image as bt.Tensor which is sent as dict
        # { "buffer": "...", "dtype": "torch.uint8", "shape": [3, 1, 1] }
        try:
            # Handle torch dtype strings by converting to numpy dtype first
            if inbound_image["dtype"] == "torch.uint8":
                inbound_image["dtype"] = "uint8"
            inbound = bt.Tensor(**inbound_image).deserialize()
            return image_to_base64(tensor_to_image(tensor=inbound))
        except TypeError as e:
            # Log error and return empty base64 string if deserialization fails
            bt.logging.warning(f"Failed to deserialize image: {str(e)}")
            return ""

    return inbound_image


class BaseImageModel(bt.Synapse):
    """
    Base protocol for image-related requests between miners and validators.
    Contains common fields used across different image manipulation tasks.
    """

    task_id: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Base64 encoded output images
    images: List[str] = []

    # Common parameters
    prompt: str = Field("Bird in the sky")
    negative_prompt: Optional[str] = Field(None)
    num_images_per_prompt: int = Field(1)
    height: int = Field(1024)
    width: int = Field(1024)
    guidance_scale: float = Field(7.5)
    seed: int = Field(-1)
    steps: int = Field(20)
    model_type: str = Field(ModelType.CUSTOM)
    task_type: TaskType
    compute_count: int = 12

    @field_validator("images", mode="before")
    def images_value(cls, inbound_images_list: List[Any]) -> List[str]:
        return [
            deserialize_incoming_image(
                image,
            )
            for image in inbound_images_list
        ]


class ImageGeneration(BaseImageModel):
    """
    Protocol for text-to-image generation requests.
    Extends BaseImageModel with generation-specific fields.
    """

    # Optional input reference image
    prompt_image: Optional[bt.Tensor] = Field(None)
    generation_type: TaskType = Field(TaskType.TEXT_TO_IMAGE)


class ImageInpainting(BaseImageModel):
    """
    Protocol for image inpainting requests.
    Extends BaseImageModel with inpainting-specific fields.
    """

    # Required input image that needs inpainting
    input_image: str = Field(...)
    # Required mask indicating areas to inpaint
    mask_image: str = Field(...)

    generation_type: TaskType = Field(TaskType.INPAINT_IMAGE)


# Combined type for image generation tasks
ImageGenerationTask = Union[ImageGeneration, ImageInpainting]


def denormalize_task(
    id: str,
    image_count: int,
    task_type: TaskType,
    **kwargs,
) -> ImageGenerationTask:
    if type == TaskType.INPAINT_IMAGE:
        return ImageInpainting(
            task_id=id,
            task_type=task_type,
            num_images_per_prompt=image_count,
            **kwargs,
        )

    return ImageGeneration(
        task_id=id,
        task_type=task_type,
        num_images_per_prompt=image_count,
        **kwargs,
    )
