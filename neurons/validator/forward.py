import base64
import copy
import time
import uuid
from asyncio import AbstractEventLoop
from dataclasses import asdict
from datetime import datetime
from io import BytesIO
from typing import List

import requests
import torch
import torchvision.transforms as T
from loguru import logger
from neurons.constants import MOVING_AVERAGE_ALPHA
from neurons.protocol import ImageGeneration
from neurons.utils import colored_log, sh, upload_batches
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


def query_axons(
    loop: AbstractEventLoop,
    dendrite: bt.dendrite,
    axons: List[AxonInfo],
    synapse: bt.Synapse,
    query_timeout: int,
) -> List[ImageGeneration]:
    """Request image generation from axons"""
    return loop.run_until_complete(
        dendrite(
            axons,
            synapse,
            timeout=query_timeout,
        )
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
    except Exception:
        logger.error("Failed to log miner counts and histories")

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


def save_images_data_for_manual_validation(
    responses: List[ImageGeneration], prompt: str
):
    logger.info("Saving images...")
    for i, r in enumerate(responses):
        for image in r.images:
            T.transforms.ToPILImage()(bt.Tensor.deserialize(image)).save(
                f"neurons/validator/images/{i}.png"
            )

    logger.info("Saving prompt...")
    with open("neurons/validator/images/prompt.txt", "w") as f:
        f.write(prompt)


def post_moving_averages(
    api_url: str, hotkeys: List[str], moving_average_scores: torch.Tensor
):
    try:
        response = requests.post(
            f"{api_url}/validator/averages",
            json={
                "averages": {
                    hotkey: moving_average.item()
                    for hotkey, moving_average in zip(hotkeys, moving_average_scores)
                }
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if response.status_code != 200:
            logger.info("Error logging moving averages to the Averages API")
            return False
        else:
            logger.info("Successfully logged moving averages to the Averages API")
            return True
    except Exception:
        logger.info("Error logging moving averages to the Averages API")
        return False


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


def run_step(self, task, axons, uids, task_type="text_to_image", image=None):
    # Get task arguments
    batch_id = task.get("Id", str(uuid.uuid))
    prompt = task.get("prompt", "Bird in the sky")
    guidance_scale = task.get("guidance_scale", 7.5)
    seed = task.get("seed", -1)
    steps = task.get("steps", 30)
    num_images_per_prompt = task.get("num_images_per_prompt", 1)
    width = task.get("width", 1024)
    height = task.get("height", 1024)

    time_elapsed = datetime.now() - self.stats.start_time

    colored_log(
        f"{sh('Info')} -> Date {datetime.strftime(self.stats.start_time, '%Y/%m/%d %H:%M')} | Elapsed {time_elapsed} | RPM {self.stats.total_requests/(time_elapsed.total_seconds()/60):.2f}",
        color="green",
    )
    colored_log(
        f"{sh('Request')} -> Type: {task_type} | Total requests sent {self.stats.total_requests:,} | Timeouts {self.stats.timeouts:,}",
        color="cyan",
    )
    colored_log(
        f"{sh('Prompt')} -> {prompt}",
        color="yellow",
    )

    # Set seed to -1 so miners will use a random seed by default
    synapse = ImageGeneration(
        generation_type=task_type,
        prompt=prompt,
        prompt_image=image,
        seed=seed,
        guidance_scale=guidance_scale,
        steps=steps,
        num_images_per_prompt=num_images_per_prompt,
        width=width,
        height=height,
    )

    synapse_info = (
        f"Timeout: {synapse.timeout:.2f} "
        f"| Height: {synapse.height} "
        f"| Width: {synapse.width}"
    )

    responses = query_axons(
        self.loop, self.dendrite, axons, synapse, self.query_timeout
    )

    log_query_to_history(self, uids)

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

    uids = torch.tensor([uids[index] for index in sorted_index]).to(self.device)
    responses = [responses[index] for index in sorted_index]

    colored_log(f"{sh('Info')} -> {synapse_info}", color="magenta")
    colored_log(
        f"{sh('UIDs')} -> {' | '.join([str(uid) for uid in uids.tolist()])}",
        color="yellow",
    )

    validator_info = self.get_validator_info()
    colored_log(
        f"{sh('Stats')} -> Block: {validator_info['block']} "
        f"| Stake: {validator_info['stake']:.4f} "
        f"| Rank: {validator_info['rank']:.4f} "
        f"| VTrust: {validator_info['vtrust']:.4f} "
        f"| Dividends: {validator_info['dividends']:.4f} "
        f"| Emissions: {validator_info['emissions']:.4f}",
        color="cyan",
    )

    self.stats.total_requests += 1

    start_time = time.time()

    # Log the results for monitoring purposes.
    log_responses(responses, prompt)

    # Save images for manual validator
    if not self.config.alchemy.disable_manual_validator:
        save_images_data_for_manual_validation(responses, prompt)
    scattered_rewards, event, rewards = get_automated_rewards(
        self, responses, uids, task_type
    )

    scattered_rewards_adjusted = get_human_rewards(self, scattered_rewards)

    scattered_rewards_adjusted = filter_rewards(
        self.isalive_dict, self.isalive_threshold, scattered_rewards_adjusted
    )

    self.moving_average_scores = update_moving_averages(
        self.moving_average_scores, scattered_rewards_adjusted, self.device
    )

    # Save moving averages scores on backend
    post_moving_averages(self.api_url, self.hotkeys, self.moving_average_scores)

    try:
        for i, average in enumerate(self.moving_average_scores):
            if (self.metagraph.axons[i].hotkey in self.hotkey_blacklist) or (
                self.metagraph.axons[i].coldkey in self.coldkey_blacklist
            ):
                self.moving_average_scores[i] = 0

    except Exception as e:
        logger.error(f"An unexpected error occurred (E1): {e}")

    try:
        # Log the step event.
        event.update(
            {
                "block": ttl_get_block(self),
                "step_length": time.time() - start_time,
                "prompt_t2i": prompt if task_type == "text_to_image" else None,
                "prompt_i2i": prompt if task_type == "image_to_image" else None,
                "uids": uids.tolist(),
                "hotkeys": [self.metagraph.axons[uid].hotkey for uid in uids],
                "images": [
                    (
                        response.images[0]
                        if (response.images != []) and (reward != 0)
                        else []
                    )
                    for response, reward in zip(responses, rewards.tolist())
                ],
                "rewards": rewards.tolist(),
                # "moving_averages": self.moving_average_scores
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
        if batch_id not in self.batches.keys():
            self.batches[batch_id] = {
                "prompt": prompt,
                "computes": images,
                "batch_id": batch_id,
                "nsfw_scores": event["nsfw_filter"],
                "blacklist_scores": event["blacklist_filter"],
                "should_drop_entries": should_drop_entries,
                "validator_hotkey": str(self.wallet.hotkey.ss58_address),
                "miner_hotkeys": [self.metagraph.hotkeys[uid] for uid in uids],
                "miner_coldkeys": [self.metagraph.coldkeys[uid] for uid in uids],
            }

        # Upload the batches to the Human Validation Platform
        upload_batches(self.batches, self.api_url)
    except Exception as e:
        logger.error(f"An unexpected error occurred appending the batch: {e}")

    log_event_to_wandb(self.wandb, event, prompt)

    return event
