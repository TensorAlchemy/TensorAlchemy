import base64
import binascii
import traceback
from io import BytesIO
from typing import Any, List, Union

import torch
import numpy as np
import bittensor as bt
from loguru import logger

from PIL import Image
from PIL.Image import Image as ImageType

import torchvision.transforms as T


def synapse_to_bytesio(synapse: bt.Synapse, img_index: int = 0) -> BytesIO:
    """
    Convert a Synapse image to BytesIO.

    Args:
        synapse (bt.Synapse): The Synapse response containing images.
        img_index (int): Index of the image to convert.

    Returns:
        BytesIO: The image as a BytesIO object.
    """
    if not synapse.images:
        return BytesIO()

    image: ImageType = synapse_to_image(synapse, img_index)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


def synapse_to_image(synapse: bt.Synapse, img_index: int = 0) -> ImageType:
    """
    Convert a Synapse image to PIL Image.

    Args:
        synapse (bt.Synapse): The Synapse response containing images.
        img_index (int): Index of the image to convert.

    Returns:
        ImageType: The converted PIL Image.
    """
    if not synapse.images:
        return Image.new("RGB", (1, 1))

    inbound: Union[str, bt.Tensor] = synapse.images[img_index]

    if isinstance(inbound, np.ndarray):
        logger.error("Miner sent us a numpy array")
        return Image.new("RGB", (1, 1))

    if isinstance(inbound, str):
        return base64_to_image(inbound)

    return tensor_to_image(inbound)


def synapse_to_images(synapse: bt.Synapse) -> List[ImageType]:
    """
    Convert all Synapse images to PIL Images.

    Args:
        synapse (bt.Synapse): The Synapse response containing images.

    Returns:
        List[ImageType]: List of converted PIL Images.
    """
    return [
        #
        synapse_to_image(synapse, idx)
        for idx in range(len(synapse.images))
    ]


def synapse_to_tensor(synapse: bt.Synapse, img_index: int = 0) -> torch.Tensor:
    """
    Convert a Synapse image to PyTorch Tensor.

    Args:
        synapse (bt.Synapse): The Synapse response containing images.
        img_index (int): Index of the image to convert.

    Returns:
        torch.Tensor: The converted PyTorch Tensor.
    """
    if not synapse.images:
        return torch.zeros((1, 1, 3), dtype=torch.uint8)

    inbound: Any = synapse.images[img_index]

    if isinstance(inbound, np.ndarray):
        return torch.from_numpy(inbound)

    if isinstance(inbound, str):
        return image_to_tensor(base64_to_image(inbound))

    return tensor_to_torch(inbound)


def synapse_to_tensors(synapse: bt.Synapse) -> List[torch.Tensor]:
    """
    Convert all Synapse images to PyTorch Tensors.

    Args:
        synapse (bt.Synapse): The Synapse response containing images.

    Returns:
        List[torch.Tensor]: List of converted PyTorch Tensors.
    """
    return [
        #
        synapse_to_tensor(synapse, idx)
        for idx in range(len(synapse.images))
    ]


def tensor_to_image(tensor: bt.Tensor) -> ImageType:
    """
    Convert a bittensor Tensor to PIL Image.

    Args:
        tensor (bt.Tensor): The bittensor Tensor to convert.

    Returns:
        ImageType: The converted PIL Image.
    """
    try:
        return T.ToPILImage()(tensor_to_torch(tensor))
    except Exception:
        logger.error(f"Error converting tensor to image: {traceback.format_exc()}")
        return Image.new("RGB", (1, 1))


def tensor_to_torch(tensor: bt.Tensor) -> torch.Tensor:
    """
    Convert a bittensor Tensor to PyTorch Tensor.

    Args:
        tensor (bt.Tensor): The bittensor Tensor to convert.

    Returns:
        torch.Tensor: The converted PyTorch Tensor.
    """
    return bt.Tensor.deserialize(tensor)


def image_to_base64(image: ImageType) -> str:
    """
    Convert a PIL Image to base64 string.

    Args:
        image (ImageType): The PIL Image to convert.

    Returns:
        str: The base64 encoded string of the image.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64_image: str) -> ImageType:
    """
    Decode base64 image data and create a PIL Image object.

    Args:
        b64_image (str): Base64 encoded image data.

    Returns:
        ImageType: PIL Image object.
    """
    try:
        image_data = base64.b64decode(b64_image)
        return Image.open(BytesIO(image_data))

    except (binascii.Error, IOError) as e:
        logger.error(f"Error processing base64 image: {str(e)}")
        return Image.new("RGB", (1, 1))


def image_to_tensor(image: ImageType) -> torch.Tensor:
    """
    Convert a PIL Image to PyTorch Tensor.

    Args:
        image (ImageType): The PIL Image to convert.

    Returns:
        torch.Tensor: The converted PyTorch Tensor.
    """
    return T.ToTensor()(image)
