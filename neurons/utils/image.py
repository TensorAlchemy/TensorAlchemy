import base64
import binascii
import traceback
from io import BytesIO
from typing import List

import torch
import numpy as np
import bittensor as bt
from loguru import logger

from PIL import Image
from PIL.Image import Image as ImageType

import torchvision.transforms as T

from neurons.protocol import SupportedImageTypes


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


def synapse_to_base64(synapse: bt.Synapse, img_index: int = 0) -> str:
    """
    Convert a Synapse image to base64 string.

    Args:
        synapse (bt.Synapse): The Synapse response containing images.
        img_index (int): Index of the image to convert.

    Returns:
        str: The image as a base64 encoded string.
    """
    if not synapse.images:
        return ""

    return bytesio_to_base64(synapse_to_bytesio(synapse, img_index))


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
        return empty_image_tensor()

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
        return empty_image()

    return tensor_to_image(synapse_to_tensor(synapse, img_index))


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


def multi_to_tensor(inbound: SupportedImageTypes) -> torch.Tensor:
    """
    Convert a Synapse image to PyTorch Tensor.

    Args:
        synapse (bt.Synapse): The Synapse response containing images.
        img_index (int): Index of the image to convert.

    Returns:
        torch.Tensor: The converted PyTorch Tensor.
    """
    if isinstance(inbound, dict):
        print(inbound.keys())

    if isinstance(inbound, str):
        return image_to_tensor(base64_to_image(inbound))

    if isinstance(inbound, torch.Tensor):
        return inbound

    if isinstance(inbound, np.ndarray):
        return torch.from_numpy(inbound)

    if isinstance(inbound, bt.Tensor):
        return tensor_to_torch(inbound)

    logger.error("Could not transform inbound type")
    return empty_image_tensor()


def tensor_to_torch(tensor: bt.Tensor) -> torch.Tensor:
    """
    Convert a bittensor Tensor to PyTorch Tensor.
    """
    try:
        if isinstance(tensor, bt.Tensor):
            return bt.Tensor.deserialize(tensor)

        if isinstance(tensor, (torch.Tensor, np.ndarray)):
            return torch.as_tensor(tensor)

        if isinstance(tensor, dict) and all(
            key in tensor for key in ["data", "dtype", "shape"]
        ):
            return torch.tensor(
                tensor["data"], dtype=getattr(torch, tensor["dtype"])
            ).reshape(tensor["shape"])

        raise ValueError(f"Unsupported tensor type: {type(tensor)}")
    except Exception:
        logger.error(f"Error tensor to torch: {traceback.format_exc()}")
        return empty_image_tensor()


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
    im_buffer = BytesIO()
    image.save(im_buffer, format="PNG")
    im_buffer.seek(0)

    return base64.b64encode(im_buffer.getvalue()).decode("utf-8")


def bytesio_to_base64(image: BytesIO) -> str:
    """
    Convert a BytesIO object to base64 string.

    Args:
        bytesio (BytesIO): The BytesIO data to convert.

    Returns:
        str: The base64 encoded string of the data.
    """
    return base64.b64encode(image.getvalue())


def base64_to_image(b64_image: str) -> ImageType:
    """
    Decode base64 image data and create a PIL Image object.

    Args:
        b64_image (str): Base64 encoded image data.

    Returns:
        ImageType: PIL Image object.
    """
    try:
        return Image.open(BytesIO(base64.b64decode(b64_image)))

    except Exception:
        logger.error(f"Error converting base64 to image: {traceback.format_exc()}")
        return empty_image()


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
        return empty_image()


def empty_image() -> ImageType:
    """Creates empty image of size (1, 1)"""
    return Image.new("RGB", (1, 1))


def empty_image_tensor() -> torch.Tensor:
    """Creates tensor representation of empty image with size (1, 1)"""
    return torch.zeros((1, 1, 3), dtype=torch.uint8)
