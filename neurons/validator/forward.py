import asyncio
import json
import time
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional, Tuple

import bittensor as bt
import torch
import torchvision.transforms as T
from bittensor import AxonInfo
from loguru import logger

from neurons.config import (
    get_backend_client,
    get_config,
    get_device,
    get_metagraph,
    get_wallet,
)
from neurons.protocol import (
    ImageGenerationTask,
    ImageGenerationTask,
    ImageInpainting,
)
from neurons.utils.defaults import Stats
from neurons.utils.image import synapse_to_base64
from neurons.utils.log import image_to_str
from neurons.validator.averages import update_moving_averages
from neurons.validator.event import EventSchema
from neurons.validator.schemas import Batch, ScoresUploadRequest
from neurons.validator.utils import ttl_get_block
from scoring.models.types import RewardModelType
from scoring.pipeline import apply_masking_functions, get_scoring_results
from scoring.types import ScoringResult, ScoringResults

transform = T.Compose([T.PILToTensor()])


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

    print(synapse)

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


async def enqueue_upload_scores(
    validator: "StableValidator",
    task: ImageGenerationTask,
    uids: torch.Tensor,
    scoring_results: ScoringResults,
):
    metagraph = get_metagraph()
    scores: Dict[str, Dict[str, float]] = {}

    for score_result in scoring_results.scores:
        score_type = score_result.type
        if score_type not in [
            RewardModelType.IMAGE,
            RewardModelType.ENHANCED_CLIP,
        ]:
            # Don't store other scores like HUMAN, NSFW, etc.
            continue
        scores[score_type] = {}
        for uid, score in zip(uids, score_result.raw[uids]):
            hotkey = metagraph.hotkeys[uid.item()]
            scores[score_type][hotkey] = score.item()

    try:
        validator.scores_upload_queue.put_nowait(
            ScoresUploadRequest(task_id=task.task_id, scores=scores)
        )
    except Exception as e:
        logger.error(f"Could not add scores to upload queue: {e}")


async def query_axons_and_process_responses(
    validator: "StableValidator",
    task: ImageGenerationTask,
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
        # Create batch from single response and enqueue uploading
        # Batch will be merged at backend side
        try:
            batch_for_upload: Batch = await create_batch_for_upload(
                synapse,
                validator,
                batch_id=task.task_id,
                prompt=task.prompt,
                responses=[response],
            )

            try:
                validator.batches_upload_queue.put_nowait(batch_for_upload)
            except Exception as e:
                logger.error(f"Could not add compute to upload queue {e}")

        except InvalidBatch as e:
            logger.info(f"Batch was invalid: {e}")
            await get_backend_client().fail_task(task.task_id)

        responses.append(response)

    return responses


def log_responses(responses: List[ImageGenerationTask], prompt: str):
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


class InvalidBatch(Exception):
    pass


async def create_batch_for_upload(
    synapse: bt.Synapse,
    validator: "StableValidator",
    batch_id: str,
    prompt: str,
    responses: List[bt.Synapse],
) -> Optional[Batch]:
    should_drop_entries = []
    images = []

    logger.info("Preparing batch for upload...")

    masked_rewards: ScoringResults = await apply_masking_functions(
        validator.model_type,
        synapse,
        responses=responses,
    )

    uids = get_uids(responses)
    masks_for_uids = masked_rewards.combined_scores[uids]

    for response, mask in zip(responses, masks_for_uids):
        if response.images:
            if response.is_success and mask.item() == 0:
                images.append(synapse_to_base64(response))
                should_drop_entries.append(0)
            else:
                # Generated image has non-zero mask,
                # we will dropp it (nsfw, blacklist)
                images.append("NO_IMAGE")
                should_drop_entries.append(1)
        else:
            images.append("NO_IMAGE")
            should_drop_entries.append(1)
            continue

    if all(image == "NO_IMAGE" for image in images):
        raise InvalidBatch("All images dropped")

    logger.info(f"{len(images)} compute responses remain after masking")

    nsfw_scores: Optional[ScoringResult] = masked_rewards.get_score(
        RewardModelType.NSFW,
    )
    blacklist_scores: Optional[ScoringResult] = masked_rewards.get_score(
        RewardModelType.BLACKLIST,
    )

    wallet: bt.wallet = get_wallet()
    metagraph: bt.metagraph = get_metagraph()

    # Update batches to be sent to the human validation platform
    # if batch_id not in validator.batches.keys():
    return Batch(
        prompt=prompt,
        computes=images,
        batch_id=batch_id,
        should_drop_entries=should_drop_entries,
        validator_hotkey=str(wallet.hotkey.ss58_address),
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
    task: ImageGenerationTask | ImageInpainting,
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

    responses = await query_axons_and_process_responses(
        validator,
        task,
        axons,
        task,
    )

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
    # Log CLIP and IMAGE rewards
    clip_rewards = scoring_results.get_score(RewardModelType.ENHANCED_CLIP)
    image_rewards = scoring_results.get_score(RewardModelType.IMAGE)

    if clip_rewards is not None and image_rewards is not None:
        clip_scores = clip_rewards.normalized[uids]
        image_scores = image_rewards.normalized[uids]

        for uid, clip_score, image_score in zip(
            uids, clip_scores, image_scores
        ):
            logger.info(
                f"UID: {uid.item()} - CLIP score: {clip_score.item():.4f}, IMAGE score: {image_score.item():.4f}"
            )

        # Upload scores to backend
        await enqueue_upload_scores(validator, task, uids, scoring_results)
    else:
        logger.warning("CLIP or IMAGE rewards not found in scoring results")

    # Update moving averages
    validator.moving_average_scores = await update_moving_averages(
        validator.moving_average_scores,
        scoring_results,
    )

    validator_info = validator.get_validator_info()

    try:
        event = EventSchema(
            task_type=task_type,
            model_type=model_type,
            compute_count=task.compute_count,
            block=ttl_get_block(),
            uids=uids,
            hotkeys=[response.axon.hotkey for response in responses],
            prompt=prompt if task_type == "TEXT_TO_IMAGE" else None,
            step_length=time.time() - start_time,
            images=[
                (
                    image_to_str(response.images[0])
                    if response.images
                    else "NO IMAGE"
                )
                for response in responses
            ],
            results=scoring_results,
            stake=validator_info["stake"].item(),
            rank=validator_info["rank"].item(),
            vtrust=validator_info["vtrust"].item(),
            dividends=validator_info["dividends"].item(),
            emissions=validator_info["emissions"].item(),
        )

        try:
            log_event(event.to_dict())
        except Exception as e:
            logger.error(f"Failed while logging event: {e}")

        return event.to_dict()

    except Exception as err:
        logger.error(f"Error creating EventSchema: {err}")


def log_event(event: dict):
    logger.info(f"[log_event]: {json.dumps(event)}")
