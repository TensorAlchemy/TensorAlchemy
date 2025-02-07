import asyncio
import time
from functools import wraps

import torch
from loguru import logger


def measure_time(func):
    """This decorator logs time of function execution"""

    @wraps(func)
    def sync_measure_time_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.warning(
            f"[measure_time] function {func.__name__} took {total_time:.2f} seconds"
        )
        return result

    async def async_measure_time_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.warning(
            f"[measure_time] async function {func.__name__} took {total_time:.2f} seconds"
        )
        return result

    if asyncio.iscoroutinefunction(func):
        return async_measure_time_wrapper
    else:
        return sync_measure_time_wrapper


def get_device_name(device: torch.device):
    """Returns name of GPU model"""
    try:
        if device.type == "cuda":
            device_name = torch.cuda.get_device_name(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            return device_name

        return "CPU"
    except Exception as e:
        logger.error(f"failed to get device name: {e}")
        return "n/a"
