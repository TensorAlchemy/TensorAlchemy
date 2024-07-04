import base64
import binascii
from io import BytesIO

from PIL import Image
from PIL.Image import Image as ImageType


def image_pil_to_base64(image):
    data = BytesIO()
    image.save(data, format="PNG")
    val = data.getvalue()
    b64 = base64.b64encode(val)
    return b64


def image_b64_to_pil(b64_image: str) -> ImageType:
    """
    Decode base64 image data and create a PIL Image object.

    Args:
        b64_image (str): Base64 encoded image data.

    Returns:
        Image: PIL Image object.
    """
    # Decode the base64 image data
    try:
        image_data = base64.b64decode(b64_image)
    except binascii.Error:
        logger.error("Could not process image from Base64")
        return Image.new("RGB", (128, 128), (0, 0, 0))

    # Create a PIL Image object from the decoded data
    return Image.open(BytesIO(image_data))
