"""
Device management utilities for the Alchemy project.
"""

from typing import Optional

import torch
from loguru import logger

from neurons.constants import is_test


def get_default_device() -> torch.device:
    """
    Get the default device for computation.

    Returns:
        torch.device: The default device (CPU for test environment, CUDA otherwise).
    """
    if is_test():
        logger.info("Using CPU for test environment (CI)")
        return torch.device("cpu:0")
    return torch.device("cuda:0")


def get_device(new_device: Optional[torch.device] = None) -> torch.device:
    """
    Get or set the global device for computation.

    Args:
        new_device (Optional[torch.device]): A new device to set as global.

    Returns:
        torch.device: The current global device.
    """
    global device
    if not device:
        device = new_device or get_default_device()

    return device


device: Optional[torch.device] = None
