import os
from datetime import datetime, timedelta
from typing import List

import torch
from loguru import logger

from neurons.config import get_config, get_device, get_metagraph


def save_ma_scores(moving_average_scores: torch.Tensor) -> None:
    """Save hotkeys, neuron model and moving average scores to filesystem."""
    logger.info("Saving current validator state...")
    try:
        neuron_state_dict = {
            "neuron_weights": moving_average_scores.to("cpu").tolist(),
        }
        torch.save(
            neuron_state_dict,
            f"{get_config().alchemy.full_path}/model.torch",
        )
        logger.info(
            f"Saved model {get_config().alchemy.full_path}/model.torch",
        )
        # empty cache
        torch.cuda.empty_cache()
        logger.info("Saved current validator state.")
    except Exception as e:
        logger.error(f"Failed to save model with error: {e}")


def get_default_ma_scores(
    use_metagraph_incentive: bool = False,
) -> torch.Tensor:
    if use_metagraph_incentive:
        logger.info("Loaded MA scores from metagraph incentive.")
        return get_metagraph().I.to(get_device())

    # NOTE: This was the old format of the way to initialize MA values
    #       this will be removed in the near future, with preference on the above
    return torch.zeros(get_metagraph().n).to(get_device())


def ensure_cache_valid(file_path: str, hours: int = 4) -> None:
    # Get file's last modified time
    last_modified = os.path.getmtime(file_path)
    last_modified_datetime = datetime.fromtimestamp(last_modified)

    # Get current time
    current_time = datetime.now()

    # Check if current time is more than 2 hours ahead of last modified time
    if current_time - last_modified_datetime > timedelta(hours=hours):
        raise ValueError("State file is more than 2 hours old")


def load_ma_scores_from_path(config_path: str) -> torch.Tensor:
    """Load hotkeys and moving average scores from filesystem."""
    file_path = f"{config_path}/model.torch"
    logger.info(
        "Attempting to load previously saved "
        + f"validator state from {config_path}..."
    )

    moving_average_scores: torch.Tensor = get_default_ma_scores()

    # NOTE: Uncomment this to enable sync from weights
    # ensure_cache_valid(file_path, hours=4)

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"State file not found at {file_path}")

    state_dict = torch.load(file_path)
    neuron_weights = torch.tensor(state_dict["neuron_weights"])

    has_nans = torch.isnan(neuron_weights).any()
    has_infs = torch.isinf(neuron_weights).any()

    if has_nans:
        logger.info(f"Nans found in the model state: {has_nans}")

    if has_infs:
        logger.info(f"Infs found in the model state: {has_infs}")

    # Check to ensure that the size of the neruon
    # weights matches the metagraph size.
    if neuron_weights.shape != (get_metagraph().n,):
        logger.warning(
            f"Neuron weights shape {neuron_weights.shape} "
            + f"does not match metagraph n {get_metagraph().n}"
            "Populating new moving_averaged_scores IDs with zeros"
        )
        moving_average_scores[: len(neuron_weights)] = neuron_weights.to(
            get_device()
        )

    # Check for nans in saved state dict
    elif not any([has_nans, has_infs]):
        moving_average_scores = neuron_weights.to(get_device())

    # Zero out any negative scores
    for i, average in enumerate(moving_average_scores):
        if average < 0:
            moving_average_scores[i] = 0

    logger.info(f"Loaded model {file_path}")

    return moving_average_scores


def load_ma_scores() -> torch.Tensor:
    possible_paths: List[str] = [
        get_config().alchemy.full_path,
        get_config().alchemy.full_path.replace(
            "tensor_alchemy", "image_alchemy"
        ),
    ]

    for possible_path in possible_paths:
        try:
            return load_ma_scores_from_path(possible_path)
        except Exception:
            logger.error(
                f"Failed to load previous MA scores from {possible_path}"
            )

    return get_default_ma_scores()
