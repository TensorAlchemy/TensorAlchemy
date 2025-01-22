from enum import Enum


class TaskType(str, Enum):
    """Enum defining supported generation tasks"""

    TEXT_TO_IMAGE = "TEXT_TO_IMAGE"
    IMAGE_TO_IMAGE = "IMAGE_TO_IMAGE"
