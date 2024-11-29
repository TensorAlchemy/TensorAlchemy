import copy
import time
from typing import Dict, Optional

import bittensor as bt
from loguru import logger

from diffusers import DiffusionPipeline
from neurons.config import get_metagraph


def get_caller_stake(synapse: bt.Synapse) -> Optional[float]:
    """
    Look up the stake of the requesting validator.
    """
    metagraph: bt.metagraph = get_metagraph()

    if synapse.dendrite.hotkey in metagraph.hotkeys:
        index = metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return metagraph.S[index].item()

    return None


def warm_up(model: DiffusionPipeline, local_args: Dict):
    """
    Warm the model up if using optimization.
    """
    start = time.perf_counter()
    c_args = copy.deepcopy(local_args)
    c_args["prompt"] = "An alchemist brewing a vibrant glowing potion."
    model(**c_args)
    logger.info(f"Warm up is complete after {time.perf_counter() - start}")
