import base64
import binascii
import traceback
from io import BytesIO
from typing import Any, List

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


def multi_to_tensor(inbound: str | np.ndarray | bt.Tensor) -> torch.Tensor:
    """
    Convert a Synapse image to PyTorch Tensor.

    Args:
        synapse (bt.Synapse): The Synapse response containing images.
        img_index (int): Index of the image to convert.

    Returns:
        torch.Tensor: The converted PyTorch Tensor.
    """
    if isinstance(inbound, np.ndarray):
        return torch.from_numpy(inbound)

    if isinstance(inbound, torch.tensor):
        return tensor_to_image(inbound)

    if isinstance(inbound, str):
        return image_to_tensor(base64_to_image(inbound))

    return tensor_to_torch(inbound)


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

    return multi_to_tensor(synapse.images[img_index])


def synapse_to_image(synapse: bt.Synapse, img_index: int = 0) -> Image.Image:
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

    # First, get the tensor using the existing function
    tensor = synapse_to_tensor(synapse, img_index)

    # Convert the tensor to PIL Image
    if tensor.ndim == 2:
        # If it's a 2D tensor (grayscale image)
        return T.ToPILImage()(tensor.unsqueeze(0))

    # If it's a 3D tensor (RGB or RGBA image)
    if tensor.shape[-1] in [1, 3, 4]:
        # Channels are in the last dimension, move them to the first
        tensor = tensor.permute(2, 0, 1)

    return T.ToPILImage()(tensor)


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


def tensor_to_torch(tensor: bt.Tensor) -> torch.Tensor:
    """
    Convert a bittensor Tensor to PyTorch Tensor.

    Args:
        tensor (bt.Tensor): The bittensor Tensor to convert.

    Returns:
        torch.Tensor: The converted PyTorch Tensor.
    """
    return bt.Tensor.deserialize(tensor)


def numpy_to_image(numpy_image: np.ndarray) -> ImageType:
    """
    Convert a numpy array to a PIL Image.

    Args:
    numpy_image (numpy.ndarray): The input numpy array.

    Returns:
    PIL.Image.Image: The numpy array as a PIL Image.
    """
    # Check the number of dimensions
    if numpy_image.ndim == 2:
        # It's a grayscale image
        mode = "L"
    elif numpy_image.ndim == 3:
        if numpy_image.shape[2] == 3:
            # It's an RGB image
            mode = "RGB"
        elif numpy_image.shape[2] == 4:
            # It's an RGBA image
            mode = "RGBA"
        else:
            raise ValueError(
                #
                "Unsupported number of channels: "
                + numpy_image.shape[2]
            )
    else:
        raise ValueError(
            #
            "Unsupported number of dimensions: "
            + numpy_image.ndim
        )

    # Ensure the data type is uint8
    if numpy_image.dtype != np.uint8:
        numpy_image = (numpy_image * 255).astype(np.uint8)

    # Create PIL image
    return Image.fromarray(numpy_image, mode=mode)


def image_to_numpy(pil_image: ImageType) -> np.ndarray:
    """
    Convert a PIL Image to a numpy array.

    Args:
    pil_image (PIL.Image.Image): The input PIL image.

    Returns:
    numpy.ndarray: The image as a numpy array.
    """
    # Convert the image to RGB mode if it's not already
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Convert PIL image to numpy array
    return np.array(pil_image)


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
    buffer.seek(0)
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
