# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 TensorAlchemy

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import typing
from typing import Dict, Optional

import pydantic
from pydantic import BaseModel, Field

import bittensor as bt


class ImageGenerationTaskModel(BaseModel):
    task_id: str
    prompt: str
    negative_prompt: Optional[str]
    prompt_image: Optional[bt.Tensor]
    images: Optional[typing.List[bt.Tensor]]
    num_images_per_prompt: int
    height: int
    width: int
    guidance_scale: float
    seed: int
    steps: int
    task_type: str


def denormalize_image_model(
    id: str, image_count: int, **kwargs
) -> ImageGenerationTaskModel:
    return ImageGenerationTaskModel(
        task_id=id,
        num_images_per_prompt=image_count,
        **kwargs,
    )


class IsAlive(bt.Synapse):
    answer: typing.Optional[str] = None
    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current ImageGeneration object. This attribute is mutable and can be updated.",
    )


class ImageGeneration(bt.Synapse):
    """
        A simple dummy protocol representation which uses bt.Synapse as its base.
        This protocol helps in handling dummy request and response communication between
        the miner and the validator.

        Attributes:
        - dummy_input: An integer value representing the input request sent by the validator.
        - dummy_output: An optional integer value which, when filled, represents the response from the     print(compute)
        print(compute.dump())
        return compute
    miner.
    """

    # Required request input, filled by sending dendrite caller.
    prompt: str = pydantic.Field("Bird in the sky", allow_mutation=False)
    negative_prompt: str = pydantic.Field(None, allow_mutation=False)
    prompt_image: bt.Tensor | None
    images: typing.List[bt.Tensor] = []
    num_images_per_prompt: int = pydantic.Field(1, allow_mutation=False)
    height: int = pydantic.Field(1024, allow_mutation=False)
    width: int = pydantic.Field(1024, allow_mutation=False)
    generation_type: str = pydantic.Field("TEXT_TO_IMAGE", allow_mutation=False)
    guidance_scale: float = pydantic.Field(7.5, allow_mutation=False)
    seed: int = pydantic.Field(1024, allow_mutation=False)
    steps: int = pydantic.Field(50, allow_mutation=False)
