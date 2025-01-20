import queue
import traceback
from multiprocessing import Event, Queue
from typing import Dict, List, Optional

import bittensor as bt
import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict

from neurons.config import (
    get_backend_client,
    get_config,
    get_metagraph,
    get_subtensor,
    get_wallet,
)
from neurons.validator.backend.exceptions import PostWeightsError
from neurons.validator.utils import ttl_get_block
from neurons.validator.utils.version import get_validator_spec_version


class WeightSettingError(Exception):
    pass


class SetWeightsTask(BaseModel):
    epoch: int
    hotkeys: List[str]
    weights: List[float]  # Changed from torch.Tensor to List[float]
    tries: Optional[int] = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)


def tensor_to_list(tensor: torch.Tensor) -> List[float]:
    return tensor.detach().cpu().tolist()


async def set_weights_loop(
    should_quit: Event,
    set_weights_queue: Queue,
) -> None:
    # Log empty queue each minute
    try:
        weights_event: SetWeightsTask = set_weights_queue.get(block=False)
        if not weights_event:
            return

    except queue.Empty:
        return

    logger.info("Gathered a weights setting task")

    if weights_event.tries > 2:
        logger.error("Weights failed to set 3 times, dropping task...")
        return

    try:
        block: int = ttl_get_block()
        epoch: int = weights_event.epoch

        epoch_length: int = get_config().alchemy.epoch_length

        if block > epoch + epoch_length:
            logger.error("Failed to set weights before next epoch!")
            return

        await set_weights(
            weights_event.hotkeys,
            torch.tensor(weights_event.weights),
        )

    except WeightSettingError:
        logger.info("[set_weights_loop] failed to set weights")
        try:
            set_weights_queue.put_nowait(
                SetWeightsTask(
                    epoch=weights_event.epoch,
                    hotkeys=weights_event.hotkeys,
                    weights=weights_event.weights,
                    tries=weights_event.tries + 1,
                )
            )
        except queue.Full:
            logger.error("Cannot add weights setting task, queue is full!")


async def set_weights(
    hotkeys: List[str],
    moving_average_scores: torch.Tensor,
) -> None:
    logger.info("Going to set weights...")
    config: bt.config = get_config()
    subtensor: bt.subtensor = get_subtensor()
    metagraph: bt.metagraph = get_metagraph()

    # Ensure tensor is on CPU
    moving_average_scores = moving_average_scores.cpu()

    # Calculate the average reward for each uid across non-zero values.
    # Replace any NaN values with 0.
    raw_weights = torch.nn.functional.normalize(
        moving_average_scores,
        p=1,
        dim=0,
    )

    try:
        await get_backend_client().post_weights(hotkeys, raw_weights)
        logger.info("Posted weights to API")
    except PostWeightsError as e:
        logger.error(f"Error logging weights to the weights API: {e}")

    valid_uids: List[int] = []

    # New list to store weights for valid hotkeys
    valid_weights: List[float] = []

    for hotkey, weight in zip(hotkeys, raw_weights):
        try:
            # Only add weight if hotkey is found
            valid_uids.append(metagraph.hotkeys.index(hotkey))
            valid_weights.append(weight)
        except ValueError:
            logger.warning(
                f"Hotkey {hotkey} not found in metagraph,"
                + " no weight will be set"
            )

    try:
        # Now uids and valid_weights have the same length
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            metagraph=metagraph,
            subtensor=subtensor,
            netuid=config.netuid,
            #
            # Which uids should be updated
            uids=torch.tensor(valid_uids).cpu(),
            # Use valid_weights instead of raw_weights
            weights=torch.tensor(valid_weights).cpu(),
        )
    except Exception:
        logger.error(
            #
            "Could not process weights: "
            + traceback.format_exc()
        )
        return

    zipped_weights: Dict[int, float] = dict(
        zip(
            processed_weight_uids.tolist(),
            processed_weights.tolist(),
        )
    )

    logger.info(f"Processed weights: {zipped_weights}")

    try:
        _success, message = subtensor.set_weights(
            wallet=get_wallet(),
            netuid=config.netuid,
            uids=processed_weight_uids,
            weights=processed_weights,
            wait_for_finalization=False,
            version_key=get_validator_spec_version(),
        )

        logger.info(f"set_weights message: {message}")

    except Exception as e:
        logger.error(e)
        raise WeightSettingError from e
