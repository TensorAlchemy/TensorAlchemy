from enum import Enum
from typing import Optional, List, Union, Any

import bittensor as bt
import numpy as np
import torch
from pydantic import BaseModel, Field, ConfigDict, field_validator


class ModelType(str, Enum):
    ALCHEMY = "ALCHEMY"
    CUSTOM = "CUSTOM"


class ImageGenerationTaskModel(BaseModel):
    task_id: str
    prompt: str
    negative_prompt: Optional[str] = None
    prompt_image: Optional[bt.Tensor] = None
    images: Optional[List[bt.Tensor]] = None
    num_images_per_prompt: int
    height: int
    width: int
    guidance_scale: float
    seed: int
    steps: int
    task_type: str
    model_type: Optional[str] = None


def denormalize_image_model(
    id: str, image_count: int, **kwargs
) -> ImageGenerationTaskModel:
    return ImageGenerationTaskModel(
        task_id=id,
        num_images_per_prompt=image_count,
        **kwargs,
    )


def deserialize_incoming_image(inbound_image: Any):
    """Inbound image type is different across different miner versions."""
    from neurons.utils.image import tensor_to_image, image_to_base64

    if isinstance(inbound_image, str):
        # Newest miners already send image as base64 string
        return inbound_image

    if isinstance(inbound_image, dict) and "buffer" in inbound_image:
        # Older miners serializing image as bt.Tensor which is sent as dict
        # { "buffer": "...", "dtype": "torch.uint8", "shape": [3, 1, 1] }
        inbound = bt.Tensor(**inbound_image).deserialize()
        return image_to_base64(tensor_to_image(tensor=inbound))

    return inbound_image


class IsAlive(bt.Synapse):
    computed_body_hash: str = Field("")
    answer: Optional[str] = None
    completion: str = Field(
        "",
        title="Completion",
        description="Completion status of the current ImageGeneration object."
        + " This attribute is mutable and can be updated.",
    )


SupportedImageTypes = Union[str, np.ndarray, torch.tensor, bt.Tensor]


class ImageGeneration(bt.Synapse):
    """
    A simple dummy protocol representation which uses bt.Synapse
    as its base.

    This protocol helps in handling dummy request and response
    communication between the miner and the validator.

    Attributes:
    - dummy_input: An integer value representing the input request
                   sent by the validator.

    - dummy_output: An optional integer value which, when filled,
                    represents the response from the miner.
    """

    # Needed because bittensor does not support np.ndarray
    # as part of their synapse, but they forcefully send it anyway 🤦
    model_config = ConfigDict(arbitrary_types_allowed=True)

    computed_body_hash: str = Field("")

    # Each image is base64 encoded image data
    images: List[str] = []

    prompt_image: Optional[bt.Tensor] = Field(
        None,
    )
    # Required request input, filled by sending dendrite caller.
    prompt: str = Field(
        "Bird in the sky",
    )
    negative_prompt: Optional[str] = Field(
        None,
    )
    num_images_per_prompt: int = Field(
        1,
    )
    height: int = Field(
        1024,
    )
    width: int = Field(
        1024,
    )
    generation_type: str = Field(
        "TEXT_TO_IMAGE",
    )
    guidance_scale: float = Field(
        7.5,
    )
    seed: int = Field(
        -1,
    )
    steps: int = Field(
        50,
    )
    model_type: str = Field(
        ModelType.CUSTOM,
    )

    @field_validator("images", mode="before")
    def images_value(cls, inbound_images_list: List[Any]) -> List[str]:
        return [deserialize_incoming_image(image) for image in inbound_images_list]
