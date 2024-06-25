import base64
import copy
import time
from dataclasses import asdict
from datetime import datetime
from io import BytesIO
from typing import List

import torch
import torchvision.transforms as T
from loguru import logger
from neurons.constants import MOVING_AVERAGE_ALPHA
from neurons.protocol import ImageGeneration, ImageGenerationTaskModel
from neurons.utils import colored_log, sh, upload_batches
from neurons.validator.backend.exceptions import PostMovingAveragesError
from neurons.validator.event import EventSchema
from neurons.validator.reward import (
    filter_rewards,
    get_automated_rewards,
    get_human_rewards,
)
from neurons.validator.utils import ttl_get_block

import bittensor as bt
import wandb as wandb_lib
from bittensor import AxonInfo

transform = T.Compose([T.PILToTensor()])


def update_moving_averages(
    moving_averaged_scores: torch.Tensor,
    rewards: torch.Tensor,
    device: torch.device,
    alpha=MOVING_AVERAGE_ALPHA,
) -> torch.FloatTensor:
    rewards = torch.nan_to_num(
        rewards,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).to(device)
    moving_averaged_scores: torch.FloatTensor = alpha * rewards + (
        1 - alpha
    ) * moving_averaged_scores.to(device)
    return moving_averaged_scores


async def query_axons(
    dendrite: bt.dendrite,
    axons: List[AxonInfo],
    synapse: bt.Synapse,
    query_timeout: int,
) -> List[ImageGeneration]:
    """Request image generation from axons"""
    return await dendrite(
        axons,
        synapse,
        timeout=query_timeout,
    )


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


async def run_step(
    validator,
    task: ImageGenerationTaskModel,
    axons: List[AxonInfo],
    uids: torch.LongTensor,
):
    # Get Arguments
    prompt = task.prompt
    batch_id = task.task_id
    task_type = task.task_type

    time_elapsed = datetime.now() - validator.stats.start_time

    colored_log(
        sh("Info")
        + f"-> Date {datetime.strftime(validator.stats.start_time, '%Y/%m/%d %H:%M')}"
        + f" | Elapsed {time_elapsed}"
        + f" | RPM {validator.stats.total_requests / (time_elapsed.total_seconds() / 60):.2f}",
        color="green",
    )
    colored_log(
        f"{sh('Request')} -> Type: {task_type}"
        + f" | Total requests sent {validator.stats.total_requests:,}"
        + f" | Timeouts {validator.stats.timeouts:,}",
        color="cyan",
    )
    colored_log(
        f"{sh('Prompt')} -> {prompt}",
        color="yellow",
    )

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
    )

    synapse_info = (
        f"Timeout: {synapse.timeout:.2f} "
        f"| Height: {synapse.height} "
        f"| Width: {synapse.width}"
    )

    responses = await query_axons(
        validator.dendrite, axons, synapse, validator.query_timeout
    )

    log_query_to_history(validator, uids)

    # Sort responses
    responses_empty_flag = [
        #
        1 if not response.images else 0
        for response in responses
    ]

    sorted_index = [
        item[0]
        for item in sorted(
            list(zip(range(0, len(responses_empty_flag)), responses_empty_flag)),
            key=lambda x: x[1],
        )
    ]

    uids = torch.tensor([uids[index] for index in sorted_index]).to(validator.device)
    responses = [responses[index] for index in sorted_index]

    colored_log(f"{sh('Info')} -> {synapse_info}", color="magenta")
    colored_log(
        f"{sh('UIDs')} -> {' | '.join([str(uid) for uid in uids.tolist()])}",
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

    validator.stats.total_requests += 1

    start_time = time.time()

    # Log the results for monitoring purposes.
    log_responses(responses, prompt)

    scattered_rewards, event, rewards = await get_automated_rewards(
        validator, responses, uids, task_type
    )

    scattered_rewards_adjusted = await get_human_rewards(validator, scattered_rewards)

    scattered_rewards_adjusted = filter_rewards(
        validator.isalive_dict, validator.isalive_threshold, scattered_rewards_adjusted
    )

    validator.moving_average_scores = update_moving_averages(
        validator.moving_average_scores, scattered_rewards_adjusted, validator.device
    )

    # Save moving averages scores on backend
    try:
        await validator.backend_client.post_moving_averages(
            validator.hotkeys, validator.moving_average_scores
        )
    except PostMovingAveragesError as e:
        logger.error(f"failed to post moving averages: {e}")

    try:
        for i, average in enumerate(validator.moving_average_scores):
            if (validator.metagraph.axons[i].hotkey in validator.hotkey_blacklist) or (
                validator.metagraph.axons[i].coldkey in validator.coldkey_blacklist
            ):
                validator.moving_average_scores[i] = 0

    except Exception as e:
        logger.error(f"An unexpected error occurred (E1): {e}")

    try:
        # Log the step event.
        event.update(
            {
                "block": ttl_get_block(validator),
                "step_length": time.time() - start_time,
                "prompt_t2i": prompt if task_type == "TEXT_TO_IMAGE" else None,
                "prompt_i2i": prompt if task_type == "IMAGE_TO_IMAGE" else None,
                "uids": uids.tolist(),
                "hotkeys": [validator.metagraph.axons[uid].hotkey for uid in uids],
                "images": [
                    (
                        response.images[0]
                        if (response.images != []) and (reward != 0)
                        else []
                    )
                    for response, reward in zip(responses, rewards.tolist())
                ],
                "rewards": rewards.tolist(),
            }
        )
        event.update(validator_info)
    except Exception as err:
        logger.error(f"Error updating event dict: {err}")

    try:
        should_drop_entries = []
        images = []
        for response, reward in zip(responses, rewards.tolist()):
            if (response.images != []) and (reward != 0):
                im_file = BytesIO()
                T.transforms.ToPILImage()(
                    bt.Tensor.deserialize(response.images[0])
                ).save(im_file, format="PNG")
                # im_bytes: image in binary format.
                im_bytes = im_file.getvalue()
                im_b64 = base64.b64encode(im_bytes)
                images.append(im_b64.decode())
                should_drop_entries.append(0)
            else:
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
        if batch_id not in validator.batches.keys():
            validator.batches[batch_id] = {
                "prompt": prompt,
                "computes": images,
                "batch_id": batch_id,
                "nsfw_scores": event["nsfw_filter"],
                "blacklist_scores": event["blacklist_filter"],
                "should_drop_entries": should_drop_entries,
                "validator_hotkey": str(validator.wallet.hotkey.ss58_address),
                "miner_hotkeys": [validator.metagraph.hotkeys[uid] for uid in uids],
                "miner_coldkeys": [validator.metagraph.coldkeys[uid] for uid in uids],
            }

        # Upload the batches to the Human Validation Platform
        validator.batches = await upload_batches(
            validator.backend_client,
            validator.batches,
        )

    except Exception as e:
        logger.error(f"An unexpected error occurred appending the batch: {e}")

    log_event_to_wandb(validator.wandb, event, prompt)

    return event
