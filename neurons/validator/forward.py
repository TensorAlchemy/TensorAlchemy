import asyncio
import base64
import copy
import time
from dataclasses import asdict
from datetime import datetime
from io import BytesIO
from typing import Dict, List, AsyncIterator, Optional

import bittensor as bt
import torch
import torchvision.transforms as T
import wandb as wandb_lib
from bittensor import AxonInfo
from loguru import logger
from pydantic import BaseModel

from neurons.constants import MOVING_AVERAGE_ALPHA
from neurons.protocol import ImageGeneration, ImageGenerationTaskModel

from neurons.utils.defaults import Stats
from neurons.utils.log import colored_log, sh

from neurons.validator.backend.exceptions import PostMovingAveragesError
from neurons.validator.event import EventSchema
from neurons.validator.rewards.types import MaskedRewards, AutomatedRewards
from neurons.validator.schemas import Batch
from neurons.validator.utils import ttl_get_block
from neurons.validator.rewards.types import RewardModelType
from neurons.validator.config import (
    get_device,
    get_metagraph,
    get_backend_client,
)
from neurons.validator.rewards.pipeline import (
    filter_rewards,
    get_masked_rewards,
    get_automated_rewards,
)

transform = T.Compose([T.PILToTensor()])


async def update_moving_averages(
    previous_ma_scores: torch.FloatTensor,
    rewards: Dict[str, float],
    hotkey_blacklist: Optional[List[str]] = None,
    coldkey_blacklist: Optional[List[str]] = None,
    alpha: Optional[float] = MOVING_AVERAGE_ALPHA,
) -> torch.FloatTensor:
    if not hotkey_blacklist:
        hotkey_blacklist = []

    if not coldkey_blacklist:
        coldkey_blacklist = []

    metagraph: bt.metagraph = get_metagraph()

    # Convert rewards dict to tensor
    rewards_tensor = torch.zeros_like(previous_ma_scores)
    for hotkey, reward in rewards.items():
        try:
            idx = metagraph.hotkeys.index(hotkey)
            rewards_tensor[idx] = reward
        except ValueError:
            logger.warning(f"Hotkey {hotkey} not found in metagraph")

    rewards_tensor = torch.nan_to_num(
        rewards_tensor,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).to(get_device())

    moving_average_scores: torch.FloatTensor = alpha * rewards_tensor + (
        1 - alpha
    ) * previous_ma_scores.to(get_device())

    # Save moving averages scores on backend
    try:
        await get_backend_client().post_moving_averages(
            metagraph.hotkeys,
            moving_average_scores,
        )
    except PostMovingAveragesError as e:
        logger.error(f"failed to post moving averages: {e}")

    try:
        for i, average in enumerate(moving_average_scores):
            if (metagraph.axons[i].hotkey in hotkey_blacklist) or (
                metagraph.axons[i].coldkey in coldkey_blacklist
            ):
                moving_average_scores[i] = 0

    except Exception as e:
        logger.error(f"An unexpected error occurred (E1): {e}")

    return moving_average_scores


class ImageGenerationResponse(BaseModel):
    axon: AxonInfo
    synapse: ImageGeneration
    time: float
    uid: Optional[int] = None

    def has_images(self) -> bool:
        return len(self.images) > 0

    @property
    def images(self):
        return self.synapse.images


async def query_single_axon(
    dendrite: bt.dendrite,
    axon: AxonInfo,
    synapse: bt.Synapse,
    query_timeout: int,
) -> ImageGenerationResponse:
    start_time = time.time()
    responses = await dendrite(
        [axon],
        synapse,
        timeout=query_timeout,
    )
    total_time = time.time() - start_time
    return ImageGenerationResponse(axon=axon, synapse=responses[0], time=total_time)


async def query_axons_async(
    dendrite: bt.dendrite,
    axons: List[AxonInfo],
    uids: torch.LongTensor,
    synapse: bt.Synapse,
    query_timeout: int,
) -> AsyncIterator[ImageGenerationResponse]:
    routines = [
        query_single_axon(dendrite, axon, synapse, query_timeout) for axon in axons
    ]
    uid_by_axon = {id(axon): uid for uid, axon in zip(uids, axons)}
    for future in asyncio.as_completed(routines):
        result: ImageGenerationResponse = await future
        result.uid = int(uid_by_axon[id(result.axon)])
        yield result


async def query_axons_and_process_responses(
    validator: "StableValidator",
    task: ImageGenerationTaskModel,
    axons: List[AxonInfo],
    uids: torch.LongTensor,
    synapse: bt.Synapse,
    query_timeout: int,
) -> List[ImageGenerationResponse]:
    """Request image generation from axons"""
    responses = []
    async for response in query_axons_async(
        validator.dendrite, axons, uids, synapse, query_timeout
    ):
        logger.info(
            f"axon={response.axon.hotkey} uid={response.uid}"
            f" responded in {response.time:.2f}s"
        )
        masked_rewards = await get_masked_rewards(
            validator.model_type,
            synapse,
            responses=[response.synapse],
        )
        # Create batch from single response and enqueue uploading
        # Batch will be merged at backend side
        batch_for_upload = await create_batch_for_upload(
            validator_wallet=validator.wallet,
            metagraph=validator.metagraph,
            batch_id=task.task_id,
            prompt=task.prompt,
            responses=[response],
            masked_rewards=masked_rewards,
        )
        validator.batches_upload_queue.put_nowait(batch_for_upload)
        responses.append(response)

    # Responses with images should be first in list
    responses.sort(key=lambda r: r.has_images(), reverse=True)

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

    colored_log(
        f"{sh('Miner Counts')} -> Max: {max(validator.miner_query_history_count.values()):.2f} "
        f"| Min: {min(validator.miner_query_history_count.values()):.2f} "
        f"| Mean: {sum(validator.miner_query_history_count.values()) / len(validator.miner_query_history_count.values()):.2f}",
        color="yellow",
    )


def log_responses(responses: List[ImageGeneration], prompt: str):
    try:
        formatted_responses = [
            {
                "negative_prompt": response.negative_prompt,
                "prompt_image": response.prompt_image,
                "num_images_per_prompt": response.num_images_per_prompt,
                "height": response.height,
                "width": response.width,
                "seed": response.seed,
                "steps": response.steps,
                "guidance_scale": response.guidance_scale,
                "generation_type": response.generation_type,
                "images": [image.shape for image in response.images],
            }
            for response in responses
        ]
        logger.info(
            f"Received {len(responses)} response(s) for the prompt '{prompt}': {formatted_responses}"
        )
    except Exception as e:
        logger.error(f"Failed to log formatted responses: {e}")


def log_event_to_wandb(wandb, event: dict, prompt: str):
    logger.info(f"Events: {str(event)}")
    logger.log("EVENTS", "events", **event)

    # Log the event to wandb.
    wandb_event = copy.deepcopy(event)
    file_type = "png"

    def gen_caption(prompt, i):
        return f"{prompt}\n({event['uids'][i]} | {event['hotkeys'][i]})"

    for e, image in enumerate(wandb_event["images"]):
        wandb_img = (
            torch.full([3, 1024, 1024], 255, dtype=torch.float)
            if image == []
            else bt.Tensor.deserialize(image)
        )

        wandb_event["images"][e] = wandb_lib.Image(
            wandb_img,
            caption=gen_caption(prompt, e),
            file_type=file_type,
        )

    wandb_event = EventSchema.from_dict(wandb_event)

    try:
        wandb.log(asdict(wandb_event))
        logger.info("Logged event to wandb.")
    except Exception as e:
        logger.error(f"Unable to log event to wandb due to the following error: {e}")


async def create_batch_for_upload(
    validator_wallet: bt.wallet,
    metagraph: "bt.metagraph.Metagraph",
    batch_id: str,
    prompt: str,
    responses: List[ImageGenerationResponse],
    masked_rewards: MaskedRewards,
):
    uids = [response.uid for response in responses]

    should_drop_entries = []
    images = []
    for response, reward in zip(responses, masked_rewards.rewards):
        if response.has_images() and reward != 0:
            im_file = BytesIO()
            T.transforms.ToPILImage()(bt.Tensor.deserialize(response.images[0])).save(
                im_file, format="PNG"
            )
            # im_bytes: image in binary format.
            im_bytes = im_file.getvalue()
            im_b64 = base64.b64encode(im_bytes)
            images.append(im_b64.decode())
            should_drop_entries.append(0)
        else:
            # Generated image has zero reward, we are dropping it
            im_file = BytesIO()
            T.transforms.ToPILImage()(
                torch.full([3, 1024, 1024], 255, dtype=torch.float)
            ).save(im_file, format="PNG")
            # im_bytes: image in binary format.
            im_bytes = im_file.getvalue()
            im_b64 = base64.b64encode(im_bytes)
            images.append(im_b64.decode())
            should_drop_entries.append(1)

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
        # TODO: Move these to a more abstract dict
        nsfw_scores=masked_rewards.event[RewardModelType.NSFW],
        blacklist_scores=masked_rewards.event[RewardModelType.BLACKLIST],
    )


def display_run_info(stats: Stats, task_type: str, prompt: str):
    time_elapsed = datetime.now() - stats.start_time

    colored_log(
        sh("Info")
        + f"-> Date {datetime.strftime(stats.start_time, '%Y/%m/%d %H:%M')}"
        + f" | Elapsed {time_elapsed}"
        + f" | RPM {stats.total_requests / (time_elapsed.total_seconds() / 60):.2f}",
        color="green",
    )
    colored_log(
        f"{sh('Request')} -> Type: {task_type}"
        + f" | Total requests sent {stats.total_requests:,}"
        + f" | Timeouts {stats.timeouts:,}",
        color="cyan",
    )
    colored_log(
        f"{sh('Prompt')} -> {prompt}",
        color="yellow",
    )


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

    synapse_info = (
        f"Timeout: {synapse.timeout:.2f} "
        f"| Height: {synapse.height} "
        f"| Width: {synapse.width}"
    )

    responses = await query_axons_and_process_responses(
        validator, task, axons, uids, synapse, validator.query_timeout
    )

    log_query_to_history(validator, uids)

    uids = [response.uid for response in responses]
    responses = [r.synapse for r in responses]

    colored_log(f"{sh('Info')} -> {synapse_info}", color="magenta")
    colored_log(
        f"{sh('UIDs')} -> {' | '.join([str(uid) for uid in uids])}",
        color="yellow",
    )

    validator_info = validator.get_validator_info()
    colored_log(
        f"{sh('Stats')} -> Block: {validator_info['block']} "
        f"| Stake: {validator_info['stake']:.4f} "
        f"| Rank: {validator_info['rank']:.4f} "
        f"| VTrust: {validator_info['vtrust']:.4f} "
        f"| Dividends: {validator_info['dividends']:.4f} "
        f"| Emissions: {validator_info['emissions']:.4f}",
        color="cyan",
    )

    stats.total_requests += 1

    start_time = time.time()

    # Log the results for monitoring purposes.
    log_responses(responses, prompt)

    # Calculate rewards
    automated_rewards: AutomatedRewards = await get_automated_rewards(
        validator.model_type,
        synapse,
        responses,
        uids,
        task_type,
    )

    scattered_rewards: torch.Tensor = validator.moving_average_scores.scatter(
        0, uids, automated_rewards.rewards[uids]
    ).to(get_device())

    scattered_rewards_adjusted = filter_rewards(
        validator.isalive_dict,
        validator.isalive_threshold,
        scattered_rewards,
    )

    # Update moving averages
    validator.moving_average_scores = await update_moving_averages(
        validator.moving_average_scores,
        scattered_rewards_adjusted,
        hotkey_blacklist=validator.hotkey_blacklist,
        coldkey_blacklist=validator.coldkey_blacklist,
    )

    # Update event and save it to wandb
    event = automated_rewards.event
    rewards_list = automated_rewards.rewards[uids].tolist()
    try:
        # Log the step event.
        event.update(
            {
                "block": ttl_get_block(validator),
                "step_length": time.time() - start_time,
                "prompt_t2i": prompt if task_type == "TEXT_TO_IMAGE" else None,
                # "prompt_i2i": prompt if task_type == "IMAGE_TO_IMAGE" else None,
                "uids": uids.tolist(),
                "hotkeys": [
                    validator.metagraph.axons[uid.item()].hotkey for uid in uids
                ],
                "images": [
                    (
                        response.images[0]
                        if (response.images != []) and (reward != 0)
                        else []
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

    log_event_to_wandb(validator.wandb, event, prompt)

    return event
