import json
import asyncio
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple

from datetime import datetime

import bittensor as bt
import torch
import torchvision.transforms as T
from bittensor import AxonInfo
from loguru import logger

from neurons.constants import MOVING_AVERAGE_ALPHA
from neurons.protocol import ImageGeneration, ImageGenerationTaskModel

from neurons.utils.defaults import Stats
from neurons.utils.log import image_to_str
from neurons.utils.image import (
    synapse_to_base64,
    empty_image_tensor,
)

from neurons.validator.backend.exceptions import PostMovingAveragesError
from neurons.validator.event import EventSchema, convert_enum_keys_to_strings
from neurons.validator.schemas import Batch
from neurons.validator.utils import ttl_get_block
from neurons.validator.scoring.models.types import RewardModelType
from neurons.validator.config import (
    get_config,
    get_device,
    get_metagraph,
    get_backend_client,
)
from neurons.validator.scoring.types import (
    ScoringResult,
    ScoringResults,
)
from neurons.validator.scoring.pipeline import (
    get_scoring_results,
    apply_masking_functions,
)

transform = T.Compose([T.PILToTensor()])


def log_moving_averages(moving_average_scores: torch.FloatTensor) -> None:
    for uid in range(1, 255):
        try:
            score = float(moving_average_scores[uid])
            score_log = f"{score:.4f}"
            if score > 0:
                logger.info(
                    f"miner_uid={uid}, miner_score={score_log}",
                    extra={"miner_uid": uid, "miner_score": score},
                )
        except IndexError:
            continue


async def update_moving_averages(
    previous_ma_scores: torch.FloatTensor,
    rewards: torch.FloatTensor,
    hotkey_blacklist: Optional[List[str]] = None,
    coldkey_blacklist: Optional[List[str]] = None,
    alpha: Optional[float] = MOVING_AVERAGE_ALPHA,
) -> torch.FloatTensor:
    if not hotkey_blacklist:
        hotkey_blacklist = []
    if not coldkey_blacklist:
        coldkey_blacklist = []
    metagraph: bt.metagraph = get_metagraph()

    rewards = torch.nan_to_num(
        rewards,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).to(get_device())

    # Number of miners has increased (a new miner has joined)
    if rewards.size(0) > previous_ma_scores.size(0):
        logger.info("New miners detected. Adjusting moving averages.")
        new_miners_count = rewards.size(0) - previous_ma_scores.size(0)
        new_miner_scores = torch.zeros(new_miners_count, device=get_device())
        previous_ma_scores = torch.cat([previous_ma_scores, new_miner_scores])

    # Number of miners has reduced (less miners online now)
    elif rewards.size(0) < previous_ma_scores.size(0):
        logger.warning(
            "Fewer miners than expected. Truncating moving averages."
        )
        previous_ma_scores = previous_ma_scores[: rewards.size(0)]

    # We merge the new rewards into the moving average using ALPHA
    # Alpha is the rate of integration of a new component into the MA tensor.
    #
    # Example:
    #  - Alpha    = 0.01
    #  - MA_SCORE = 1.00
    #  - Rewards  = 0.50
    #
    # So the result is:
    # (0.01 * 0.5) + (1.0 - 0.01) * 1.00 = 0.995
    moving_average_scores: torch.FloatTensor = alpha * rewards + (
        1 - alpha
    ) * previous_ma_scores.to(get_device())

    try:
        log_moving_averages(moving_average_scores)
    except Exception:
        pass

    # Save moving averages scores on backend
    try:
        await get_backend_client().post_moving_averages(
            metagraph.hotkeys,
            moving_average_scores,
        )
    except PostMovingAveragesError as e:
        logger.error(f"failed to post moving averages: {e}")

    try:
        for i, (hotkey, coldkey) in enumerate(
            zip(metagraph.hotkeys, metagraph.coldkeys)
        ):
            if hotkey in hotkey_blacklist or coldkey in coldkey_blacklist:
                moving_average_scores[i] = 0

    except Exception as e:
        logger.error(f"An unexpected error occurred (E1): {e}")

    return moving_average_scores


async def query_axons_async(
    dendrite: bt.dendrite,
    axons: List[bt.AxonInfo],
    synapse: bt.Synapse,
) -> AsyncIterator[Tuple[int, bt.Synapse]]:
    """
    Asynchronously queries a list of axons and yields the responses.
    Args:
        dendrite (bt.dendrite): The dendrite instance to use for querying.
        axons (List[AxonInfo]): The list of axons to query.
        synapse (bt.Synapse): The synapse object to use for the query.
    Yields:
        Tuple[int, bt.Synapse]: The UID of the axon and the filled Synapse object.
    """
    metagraph: bt.metagraph = get_metagraph()

    async def do_call(inbound_axon: bt.AxonInfo) -> Tuple[int, bt.Synapse]:
        uid: int = metagraph.hotkeys.index(inbound_axon.hotkey)

        # NOTE: Anything except `forward` here causes
        #       weird impure race-conditions.
        #
        #       Please use `forward` for now
        to_return: List[bt.Synapse] = await dendrite.forward(
            synapse=synapse,
            timeout=get_config().alchemy.query_timeout,
            axons=[inbound_axon],
        )

        return uid, to_return[0]

    # Create tasks for all axons
    tasks = [asyncio.create_task(do_call(axon)) for axon in axons]

    # Use asyncio.as_completed to yield results as they complete
    for future in asyncio.as_completed(tasks):
        uid, result = await future
        yield uid, result


async def query_axons_and_process_responses(
    validator: "StableValidator",
    task: ImageGenerationTaskModel,
    axons: List[AxonInfo],
    synapse: bt.Synapse,
) -> List[bt.Synapse]:
    """Request image generation from axons"""
    responses = []
    async for uid, response in query_axons_async(
        validator.dendrite,
        axons,
        synapse,
    ):
        masked_rewards: ScoringResults = await apply_masking_functions(
            validator.model_type,
            synapse,
            responses=[response],
        )

        # Create batch from single response and enqueue uploading
        # Batch will be merged at backend side
        batch_for_upload: Batch = await create_batch_for_upload(
            validator_wallet=validator.wallet,
            metagraph=validator.metagraph,
            batch_id=task.task_id,
            prompt=task.prompt,
            responses=[response],
            masked_rewards=masked_rewards,
        )

        responses.append(response)

        if batch_for_upload:
            try:
                validator.batches_upload_queue.put_nowait(batch_for_upload)
            except Exception as e:
                logger.error(f"Could not add compute to upload queue {e}")

    return responses


def log_query_to_history(validator: "StableValidator", uids: torch.Tensor):
    try:
        for uid in uids:
            validator.miner_query_history_duration[
                validator.metagraph.axons[uid].hotkey
            ] = time.perf_counter()
        for uid in uids:
            validator.miner_query_history_count[
                validator.metagraph.axons[uid].hotkey
            ] += 1
    except Exception as e:
        logger.error(
            f"Failed to log miner counts and histories due to the following error: {e}"
        )

    logger.info(
        f"Miner Counts -> Max: {max(validator.miner_query_history_count.values()):.2f} "
        + f"| Min: {min(validator.miner_query_history_count.values()):.2f} "
        + f"| Mean: {sum(validator.miner_query_history_count.values()) / len(validator.miner_query_history_count.values()):.2f}",
    )


def log_responses(responses: List[ImageGeneration], prompt: str):
    try:
        logger.info(
            #
            f"Received {len(responses)} response(s) for the prompt "
            + prompt
        )

        for response in responses:
            logger.info(
                json.dumps(
                    {
                        "axon_ip": response.axon.ip,
                        "axon_hotkey": response.axon.hotkey,
                        "negative_prompt": response.negative_prompt,
                        "prompt_image": response.prompt_image,
                        "num_images_per_prompt": response.num_images_per_prompt,
                        "height": response.height,
                        "width": response.width,
                        "seed": response.seed,
                        "steps": response.steps,
                        "guidance_scale": response.guidance_scale,
                        "generation_type": response.generation_type,
                        "images": [
                            image_to_str(image) for image in response.images
                        ],
                    },
                    indent=2,
                )
            )

    except Exception as e:
        logger.error(f"Failed to log formatted responses: {e}")


def log_event(event: dict):
    event = EventSchema.from_dict(convert_enum_keys_to_strings(event))
    # Reduce img output
    event.images = [image_to_str(img) for img in event.images]
    logger.info(f"[log_event]: {event}")


async def create_batch_for_upload(
    validator_wallet: bt.wallet,
    metagraph: "bt.metagraph.Metagraph",
    batch_id: str,
    prompt: str,
    responses: List[bt.Synapse],
    masked_rewards: ScoringResults,
) -> Optional[Batch]:
    should_drop_entries = []
    images = []

    uids = get_uids(responses)
    rewards_for_uids = masked_rewards.combined_scores[uids]

    for response, reward in zip(responses, rewards_for_uids):
        if response.images:
            images.append(synapse_to_base64(response))

            if response.is_success and reward.item() == 0:
                should_drop_entries.append(0)
            else:
                # Generated image has non-zero mask,
                # we will dropp it (nsfw, blacklist)
                should_drop_entries.append(1)
        else:
            images.append("NO_IMAGE")
            should_drop_entries.append(1)

            # TODO: Should we just drop it?
            return None

    nsfw_scores: Optional[ScoringResult] = masked_rewards.get_score(
        RewardModelType.NSFW,
    )
    blacklist_scores: Optional[ScoringResult] = masked_rewards.get_score(
        RewardModelType.BLACKLIST,
    )

    if not nsfw_scores:
        return None

    if not blacklist_scores:
        return None

    # Update batches to be sent to the human validation platform
    # if batch_id not in validator.batches.keys():
    return Batch(
        prompt=prompt,
        computes=images,
        batch_id=batch_id,
        should_drop_entries=should_drop_entries,
        validator_hotkey=str(validator_wallet.hotkey.ss58_address),
        miner_hotkeys=[metagraph.hotkeys[uid] for uid in uids],
        miner_coldkeys=[metagraph.coldkeys[uid] for uid in uids],
        # Scores
        nsfw_scores=nsfw_scores.scores[uids].tolist(),
        blacklist_scores=blacklist_scores.scores[uids].tolist(),
    )


def display_run_info(stats: Stats, task_type: str, prompt: str):
    time_elapsed = datetime.now() - stats.start_time

    logger.info(
        "Info"
        + f"-> Date {datetime.strftime(stats.start_time, '%Y/%m/%d %H:%M')}"
        + f" | Elapsed {time_elapsed}"
        + f" | RPM {stats.total_requests / (time_elapsed.total_seconds() / 60):.2f}",
    )
    logger.info(
        "Request"
        + f" -> Type: {task_type}"
        + f" | Total requests sent {stats.total_requests:,}"
        + f" | Timeouts {stats.timeouts:,}",
    )
    logger.info(
        f"Prompt -> {prompt}",
    )


def get_uids(responses: List[bt.Synapse]) -> torch.Tensor:
    metagraph: bt.metagraph = get_metagraph()

    return torch.tensor(
        [
            #
            metagraph.hotkeys.index(response.axon.hotkey)
            for response in responses
        ],
        dtype=torch.long,
    ).to(get_device())


async def run_step(
    validator: "StableValidator",
    task: ImageGenerationTaskModel,
    axons: List[AxonInfo],
    uids: torch.LongTensor,
    model_type: str,
    stats: Stats,
):

    # Get Arguments
    prompt = task.prompt
    task_type = task.task_type

    # Output some information about run
    display_run_info(stats, task_type, prompt)

    # Set seed to -1 so miners will use a random seed by default
    task_type_for_miner = task_type.lower()
    synapse = ImageGeneration(
        prompt=prompt,
        negative_prompt=task.negative_prompt,
        generation_type=task_type_for_miner,
        prompt_image=task.images,
        seed=task.seed,
        guidance_scale=task.guidance_scale,
        steps=task.steps,
        num_images_per_prompt=1,
        width=task.width,
        height=task.height,
        model_type=model_type,
    )

    responses = await query_axons_and_process_responses(
        validator,
        task,
        axons,
        synapse,
    )

    log_query_to_history(validator, uids)

    uids = get_uids(responses)

    logger.info(
        f"UIDs -> {' | '.join([str(uid.item()) for uid in uids])}",
    )

    validator_info = validator.get_validator_info()
    logger.info(
        f"Stats -> Block: {validator_info['block']} "
        f"| Stake: {validator_info['stake']:.4f} "
        f"| Rank: {validator_info['rank']:.4f} "
        f"| VTrust: {validator_info['vtrust']:.4f} "
        f"| Dividends: {validator_info['dividends']:.4f} "
        f"| Emissions: {validator_info['emissions']:.4f}",
    )

    stats.total_requests += 1

    start_time = time.time()

    # Log the results for monitoring purposes.
    if get_config().DEBUG:
        log_responses(responses, prompt)

    # Calculate rewards
    scoring_results: ScoringResults = await get_scoring_results(
        validator.model_type,
        synapse,
        responses,
    )

    # Apply isalive filtering
    rewards_tensor_adjusted = scoring_results.combined_scores

    # TODO: Check and see if miners are getting dropped scores
    #       because the is-alive filter is too strict or broken
    # rewards_tensor_adjusted = filter_rewards(
    #     validator.isalive_dict,
    #     validator.isalive_threshold,
    #     # No need for scattering, directly use the rewards
    #     scoring_results.combined_scores,
    # )

    # Update moving averages
    validator.moving_average_scores = await update_moving_averages(
        validator.moving_average_scores,
        rewards_tensor_adjusted,
        hotkey_blacklist=validator.hotkey_blacklist,
        coldkey_blacklist=validator.coldkey_blacklist,
    )

    # Create event for logging
    event: Dict = {}
    rewards_list = scoring_results.combined_scores[uids].tolist()

    for reward_score in scoring_results.scores:
        event[reward_score.type] = reward_score.scores[uids]

    try:
        # Log the step event.
        event.update(
            {
                "task_type": task_type,
                "block": ttl_get_block(),
                "step_length": time.time() - start_time,
                "prompt": prompt if task_type == "TEXT_TO_IMAGE" else None,
                "uids": uids,
                "hotkeys": [response.axon.hotkey for response in responses],
                "images": [
                    (
                        response.images[0]
                        if (response.images != [])
                        else empty_image_tensor()
                    )
                    for response, reward in zip(responses, rewards_list)
                ],
                "rewards": rewards_list,
                "model_type": model_type,
            }
        )
        event.update(validator_info)
    except Exception as err:
        logger.error(f"Error updating event dict: {err}")

    try:
        log_event(event)
    except Exception as e:
        logger.error(f"Failed while logging event: {e}")

    return event
