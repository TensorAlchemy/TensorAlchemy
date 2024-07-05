import base64

from enum import Enum
from typing import Optional, List

import numpy as np
import bittensor as bt
from pydantic import BaseModel, Field, ConfigDict


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


class IsAlive(bt.Synapse):
    computed_body_hash: str = Field("")
    answer: Optional[str] = None
    completion: str = Field(
        "",
        title="Completion",
        description="Completion status of the current ImageGeneration object."
        + " This attribute is mutable and can be updated.",
    )


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

    images: List[str | np.ndarray | bt.Tensor] = []

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
