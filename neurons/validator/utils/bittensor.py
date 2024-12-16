import traceback
from ssl import SSLEOFError
from typing import Any, Callable

from loguru import logger
from neurons.config import get_subtensor
from neurons.validator.utils.cache import ttl_cache


def get_subtensor_method(method: str) -> Callable:
    if not hasattr(get_subtensor(), method):
        raise ValueError(f"Subtensor has no method {method}")

    method: Callable = getattr(get_subtensor(), method)

    if not callable(method):
        raise ValueError(f"Subtensor method {method} is not callable")

    return method


def wrap_with_broken_pipe_handler(method: str, *args, **kwargs) -> Any:
    try:
        return get_subtensor_method(method)(*args, **kwargs)

    except (BrokenPipeError, SSLEOFError):
        # Re-init the subtensor
        get_subtensor(nocache=True)
        return wrap_with_broken_pipe_handler(method, *args, **kwargs)

    except Exception:
        logger.error(
            "An unexpected error occurred "
            + "while attempting to get the current block: "
            + traceback.format_exc()
        )

        # Re-init the subtensor
        get_subtensor(nocache=True)
        return get_subtensor_method(method)(*args, **kwargs)


def is_hotkey_registered(**kwargs) -> int:
    return wrap_with_broken_pipe_handler("is_hotkey_registered", **kwargs)


@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block() -> int:
    return wrap_with_broken_pipe_handler("get_current_block")
