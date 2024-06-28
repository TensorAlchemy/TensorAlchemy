import asyncio
import base64
import copy
import time
from typing import (
    AsyncIterator,
    Dict,
    List,
    Optional,
    Tuple,
)

from dataclasses import asdict
from datetime import datetime
from io import BytesIO

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
from neurons.validator.event import EventSchema, convert_enum_keys_to_strings
from neurons.validator.schemas import Batch
from neurons.validator.utils import ttl_get_block
from neurons.validator.rewards.types import RewardModelType
from neurons.validator.config import (
    get_device,
    get_metagraph,
    get_backend_client,
)
from neurons.validator.rewards.types import (
    ScoringResult,
    ScoringResults,
)
from neurons.validator.rewards.pipeline import (
    filter_rewards,
    get_scoring_results,
    apply_masking_functions,
)

transform = T.Compose([T.PILToTensor()])


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
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    ).to(get_device())

    moving_average_scores: torch.FloatTensor = alpha * rewards + (
        1 - alpha
    ) * previous_ma_scores.to(get_device())

    print("**************************************************")
    print(f"{moving_average_scores=}")

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


async def query_axons_async(
    dendrite: bt.dendrite,
    axons: List[bt.AxonInfo],
    synapse: bt.Synapse,
    query_timeout: int,
) -> AsyncIterator[Tuple[int, bt.Synapse]]:
    """
    Asynchronously queries a list of axons and yields the responses.
    Args:
        dendrite (bt.dendrite): The dendrite instance to use for querying.
        axons (List[AxonInfo]): The list of axons to query.
        synapse (bt.Synapse): The synapse object to use for the query.
        query_timeout (int): The timeout duration for the query.
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
            timeout=query_timeout,
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
    query_timeout: int,
) -> List[bt.Synapse]:
    """Request image generation from axons"""
    responses = []
    async for uid, response in query_axons_async(
        validator.dendrite,
        axons,
        synapse,
        query_timeout,
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
    event = convert_enum_keys_to_strings(event)

    logger.info(f"Events: {str(event)}")

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
    masked_rewards: ScoringResults,
) -> Optional[Batch]:
    uids = get_uids(responses)

    should_drop_entries = []
    images = []

    masking_results_for_uids: torch.Tensor = masked_rewards.combined_scores[uids]

    for response, reward in zip(responses, masking_results_for_uids):
        if response.is_success and reward != 0:
            im_file = BytesIO()
            T.transforms.ToPILImage()(
                bt.Tensor.deserialize(response.images[0]),
            ).save(im_file, format="PNG")
            # im_bytes: image in binary format.
            im_bytes = im_file.getvalue()
            im_b64 = base64.b64encode(im_bytes)
            images.append(im_b64.decode())
            should_drop_entries.append(0)
        else:
            # Generated image has zero reward, we are dropping it
            im_file = BytesIO()
            # im_bytes: image in binary format.
            im_bytes = im_file.getvalue()
            im_b64 = base64.b64encode(im_bytes)
            images.append(im_b64.decode())
            should_drop_entries.append(1)

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

    synapse_info = (
        f"Timeout: {synapse.timeout:.2f} "
        f"| Height: {synapse.height} "
        f"| Width: {synapse.width}"
    )

    responses = await query_axons_and_process_responses(
        validator,
        task,
        axons,
        synapse,
        validator.query_timeout,
    )

    log_query_to_history(validator, uids)

    uids = get_uids(responses)

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
    scoring_results: ScoringResults = await get_scoring_results(
        validator.model_type,
        synapse,
        responses,
    )

    print("==================================================")
    print("COMBINED")
    print(scoring_results.combined_scores)

    for score in scoring_results.scores:
        print("--------------------------------------------------")
        print(score.type, score.scores, score.normalized)

    # Apply isalive filtering
    rewards_tensor_adjusted = filter_rewards(
        validator.isalive_dict,
        validator.isalive_threshold,
        # No need for scattering, directly use the rewards
        scoring_results.combined_scores,
    )

    print("--------------------------------------------------")
    print("FILTERED")
    print(scoring_results.combined_scores)

    # Update moving averages
    validator.moving_average_scores = await update_moving_averages(
        validator.moving_average_scores,
        rewards_tensor_adjusted,
        hotkey_blacklist=validator.hotkey_blacklist,
        coldkey_blacklist=validator.coldkey_blacklist,
    )

    # Update event and save it to wandb
    event: Dict = {}
    rewards_list = scoring_results.combined_scores[uids].tolist()

    for reward_score in scoring_results.scores:
        event[reward_score.type] = reward_score.scores[uids]

    try:
        # Log the step event.
        event.update(
            {
                "task_type": task_type,
                "block": ttl_get_block(validator),
                "step_length": time.time() - start_time,
                "prompt": prompt if task_type == "TEXT_TO_IMAGE" else None,
                "uids": uids,
                "hotkeys": [response.axon.hotkey for response in responses],
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

    try:
        log_event_to_wandb(
            validator.wandb,
            event,
            prompt,
        )
    except Exception as e:
        logger.error(f"Failed while logging to wandb: {e}")

    return event
