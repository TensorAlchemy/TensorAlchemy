import asyncio
import random
import time
import traceback
from functools import wraps
from typing import Any, Dict, List, Tuple

import bittensor as bt
import torch
from loguru import logger

from neurons.config import (
    get_blacklist,
    get_config,
    get_dendrite,
    get_device,
    get_metagraph,
)
from neurons.constants import N_NEURONS, N_NEURONS_TO_QUERY, VPERMIT_TAO
from neurons.protocol import IsAlive


async def check_uid(uid: int) -> Tuple[bool, float]:
    try:
        t1 = time.perf_counter()
        metagraph: bt.metagraph = get_metagraph()

        responses: List[IsAlive] = await get_dendrite().forward(
            synapse=IsAlive(),
            axons=[metagraph.axons[uid]],
            timeout=get_config().alchemy.async_timeout,
        )

        if not responses:
            return uid, False, -1

        response: IsAlive = responses[0]

        if response.is_success:
            return uid, True, time.perf_counter() - t1

        return uid, False, -1
    except Exception:
        logger.error(
            #
            f"Error checking UID {uid}: "
            + traceback.format_exc()
        )
        return uid, False, -1


def memoize_with_expiration(expiration_time: int):
    """
    Decorator to memoize a function with a time-based expiration.

    Args:
        expiration_time (int): Time in seconds before the cached result expires.
    """
    cache: Dict[Any, Tuple[Any, float]] = {}

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < expiration_time:
                    return result

            result = await func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result

        return wrapper

    return decorator


def is_uid_available(uid: int, vpermit_tao_limit: int) -> bool:
    """
    Check if a UID is available based on serving status and stake.

    Args:
        uid (int): The UID to check.
        vpermit_tao_limit (int): The validator permit TAO limit.

    Returns:
        bool: True if the UID is available, False otherwise.
    """
    metagraph = get_metagraph()

    if not metagraph.axons[uid].is_serving:
        return False

    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False

    return True


async def filter_available_uids(
    exclude: List[int] = None,
) -> List[int]:
    """
    Filter available UIDs based on availability and exclusion list.

    Args:
        exclude (List[int], optional): List of UIDs to exclude. Defaults to None.

    Returns:
        List[int]: List of available UIDs.
    """
    exclude = exclude or []
    metagraph = get_metagraph()

    hotkey_blacklist, coldkey_blacklist = await get_blacklist()

    return [
        uid
        for uid in range(metagraph.n.item())
        if is_uid_available(uid, VPERMIT_TAO)
        and metagraph.axons[uid].hotkey not in hotkey_blacklist
        and metagraph.axons[uid].coldkey not in coldkey_blacklist
        and uid not in exclude
    ]


async def check_uids_alive(uids: List[int]) -> Tuple[List[int], List[float]]:
    """
    Check which UIDs are alive.

    Args:
        uids (List[int]): List of UIDs to check.

    Returns:
        Tuple[List[int], List[float]]: A tuple containing the list of alive UIDs and their response times.
    """

    alive_uids = []
    response_times = []

    tasks = [asyncio.create_task(check_uid(uid)) for uid in uids]

    for future in asyncio.as_completed(tasks):
        uid, is_alive, response_time = await future

        if not is_alive:
            continue

        alive_uids.append(uid)
        response_times.append(response_time)

    return alive_uids, response_times


@memoize_with_expiration(20)
async def get_active_uids(limit: int = -1) -> List[int]:
    """
    Fetch all active (alive) UIDs. Results are memoized for 20 seconds.

    Returns:
        List[int]: List of all active UIDs.
    """
    available_uids = await filter_available_uids()

    # Shuffle to avoid always checking the same UIDs first
    random.shuffle(available_uids)

    all_active_uids = []
    for i in range(0, len(available_uids), N_NEURONS_TO_QUERY):
        batch = available_uids[i : i + N_NEURONS_TO_QUERY]
        active_uids, _ = await check_uids_alive(batch)
        all_active_uids.extend(active_uids)

        if limit > 0:
            if len(all_active_uids) >= limit:
                break

    logger.info(f"Found {len(all_active_uids)} active UIDs")
    logger.info(f"Active miners: {all_active_uids}")

    return all_active_uids


async def select_uids(count: int = 12) -> torch.tensor:
    active_uids = await get_active_uids(limit=count)

    logger.info(active_uids)

    if len(active_uids) < 1:
        return torch.tensor([]).to(get_device())

    selected_uids = torch.tensor(
        active_uids[:N_NEURONS],
        dtype=torch.long,
    ).to(get_device())

    logger.info(f"Selected miners: {selected_uids.tolist()}")

    return selected_uids


# Example usage
async def main():
    # Get all active UIDs
    all_active = await get_active_uids()
    logger.info(f"Total active UIDs: {len(all_active)}")

    # Call get_active_uids again (should use cached result)
    all_active_cached = await get_active_uids()
    logger.info(f"Total active UIDs (cached): {len(all_active_cached)}")


if __name__ == "__main__":
    asyncio.run(main())
