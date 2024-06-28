# Utils for checkpointing and saving the model.
import asyncio
import random
import time
import traceback
from functools import lru_cache, update_wrapper
from math import floor
from typing import Any, Callable, List, Optional

import bittensor as bt
import numpy as np
import requests
import torch
import torch.nn as nn
from loguru import logger


from neurons.constants import (
    N_NEURONS_TO_QUERY,
    VPERMIT_TAO,
)
from neurons.protocol import IsAlive
from neurons.validator.services.openai.service import (
    get_openai_service,
    OpenAIRequestFailed,
)


def _ttl_hash_gen(seconds: int):
    start_time = time.time()
    while True:
        yield floor((time.time() - start_time) / seconds)


# LRU Cache with TTL
def ttl_cache(maxsize: int = 128, typed: bool = False, ttl: int = -1):
    if ttl <= 0:
        ttl = 65536
    hash_gen = _ttl_hash_gen(ttl)

    def wrapper(func: Callable) -> Callable:
        @lru_cache(maxsize, typed)
        def ttl_func(_ttl_hash, *args, **kwargs):
            return func(*args, **kwargs)

        def wrapped(*args, **kwargs) -> Any:
            th = next(hash_gen)
            return ttl_func(th, *args, **kwargs)

        return update_wrapper(wrapped, func)

    return wrapper


# 12 seconds updating block.
@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block(self) -> int:
    return self.subtensor.get_current_block()


async def check_uid(dendrite, self, uid, response_times):
    try:
        t1 = time.perf_counter()
        response = await dendrite(
            self.metagraph.axons[uid],
            IsAlive(),
            deserialize=False,
            timeout=self.async_timeout,
        )
        if response.is_success:
            response_times.append(time.perf_counter() - t1)
            self.isalive_dict[uid] = 0
            return True
        else:
            try:
                self.isalive_failed_count[uid] += 1
                key = self.metagraph.axons[uid].hotkey
                self.miner_query_history_fail_count[key] += 1
                # If miner doesn't respond for 3 iterations rest it's count to
                # the average to avoid spamming
                if self.miner_query_history_fail_count[key] >= 3:
                    self.miner_query_history_duration[key] = time.perf_counter()
                    self.miner_query_history_count[key] = int(
                        np.array(list(self.miner_query_history_count.values())).mean()
                    )
            except Exception:
                pass
            return False
    except Exception as e:
        logger.error(f"Error checking UID {uid}: {e}\n{traceback.format_exc()}")
        return False


def check_uid_availability(
    dendrite,
    metagraph: "bt.metagraph.Metagraph",
    uid: int,
    vpermit_tao_limit: int,
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # # Filter for miners that are processing other responses
    # if not await check_uid(dendrite, metagraph.axons[uid], uid):
    # return False
    # # Available otherwise.
    return True


async def get_random_uids(
    self, dendrite, k: int, exclude: List[int] = None
) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    # Filter for only serving miners
    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            dendrite, self.metagraph, uid, VPERMIT_TAO
        )
        uid_is_not_excluded = exclude is None or uid not in exclude
        if (
            uid_is_available
            and (self.metagraph.axons[uid].hotkey not in self.hotkey_blacklist)
            and (self.metagraph.axons[uid].coldkey not in self.coldkey_blacklist)
        ):
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    # # Sort candidate UIDs by their count history
    # # This prioritises miners that have been queried less than average
    # candidate_uids = [i for i,_ in sorted(zip(candidate_uids, [self.miner_query_history_count[self.metagraph.axons[uid].hotkey] for uid in candidate_uids]))]

    # Random sort candidate_uids
    random.shuffle(candidate_uids)

    # Find the first K uids that respond with IsAlive
    final_uids = []
    t0 = time.perf_counter()
    attempt_counter = 0
    avg_num_list = []
    for uid in range(0, len(candidate_uids), N_NEURONS_TO_QUERY):
        tasks = []

        logger.info(f"UIDs in pool: {final_uids}")
        logger.info(f"Querying uids: {candidate_uids[uid:uid+N_NEURONS_TO_QUERY]}")

        t1 = time.perf_counter()

        times_list = []

        for u in candidate_uids[uid : uid + N_NEURONS_TO_QUERY]:
            tasks.append(check_uid(dendrite, self, u, times_list))

        responses = await asyncio.gather(*tasks)
        attempt_counter += 1

        logger.info(f"Time to get responses: {time.perf_counter() - t1:.2f}s")

        list_slice = times_list[-25:]
        time_sum = sum(list_slice)

        logger.info(
            f"Number of times stored: {len(times_list)} | Average successful response across {len(list_slice)} samples: {time_sum / len(list_slice) if len(list_slice) > 0 else 0:.2f}"
        )

        if True in responses:
            t2 = time.perf_counter()

            temp_list = []

            for i, response in enumerate(responses):
                if response and (len(final_uids) < k):
                    final_uids.append(candidate_uids[uid + i])

                    temp_list.append(candidate_uids[uid + i])

                elif len(final_uids) >= k:
                    break
                else:
                    continue

            logger.info(f"Added uids: {temp_list} in {time.perf_counter() - t2:.2f}s")

            avg_num_list.append(len(temp_list))

            if len(final_uids) >= k:
                break

        else:
            continue

    sum_avg = 0
    try:
        sum_avg = sum(avg_num_list) / attempt_counter
    except Exception:
        pass

    logger.info(
        f"Time to find all {len(final_uids)} uids: {time.perf_counter() - t0:.2f}s in {attempt_counter} attempts | Avg active UIDs per attempt: {sum_avg:.2f}"
    )

    # print({f"UID_{candidate_uid}": "Active" if candidate_uid in final_uids else "Inactive" for i, candidate_uid in enumerate(candidate_uids)})

    uids = (
        torch.tensor(final_uids)
        if len(final_uids) < k
        else torch.tensor(random.sample(final_uids, k))
    )

    return uids


def calculate_mean_dissimilarity(dissimilarity_matrix):
    num_images = len(dissimilarity_matrix)
    mean_dissimilarities = []

    for i in range(num_images):
        dissimilarity_values = [
            dissimilarity_matrix[i][j] for j in range(num_images) if i != j
        ]
        # error: list index out of range
        if len(dissimilarity_values) == 0 or sum(dissimilarity_values) == 0:
            mean_dissimilarities.append(0)
            continue
        # divide by amount of non zero values
        non_zero_values = [value for value in dissimilarity_values if value != 0]
        mean_dissimilarity = sum(dissimilarity_values) / len(non_zero_values)
        mean_dissimilarities.append(mean_dissimilarity)

    # Min-max normalization
    non_zero_values = [value for value in mean_dissimilarities if value != 0]

    if len(non_zero_values) == 0:
        return [0.5] * num_images

    min_value = min(non_zero_values)
    max_value = max(mean_dissimilarities)
    range_value = max_value - min_value
    if range_value != 0:
        mean_dissimilarities = [
            (value - min_value) / range_value for value in mean_dissimilarities
        ]
    else:
        # All elements are the same (no range), set all values to 0.5
        mean_dissimilarities = [0.5] * num_images
    # clamp to [0,1]
    mean_dissimilarities = [
        #
        min(1, max(0, value))
        for value in mean_dissimilarities
    ]

    return mean_dissimilarities


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


def corcel_parse_response(text):
    split = text.split('"')
    if len(split) == 3:
        # Has quotes
        split = [x for x in split if x]

        if split:
            split = split[0]
        else:
            logger.info(f"Returning (X1) default text: {text}")
            return text
    elif len(split) == 1:
        split = split[0]
    elif len(split) > 3:
        split = [x for x in split if x]
        if len(split) > 0:
            split = split[0]
    else:
        logger.info(f"Split: {split}")
        logger.info(f"Returning (X2) default text: {text}")
        return text

    logger.info(f"Returning parsed text: {split}")
    return split


def call_corcel(self, prompt):
    HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"{self.corcel_api_key}",
    }
    JSON = {
        "miners_to_query": 1,
        "top_k_miners_to_query": 160,
        "ensure_responses": True,
        "miner_uids": [],
        "messages": [
            {
                "role": "system",
                "content": prompt,
            }
        ],
        "model": "cortext-ultra",
        "stream": False,
        "top_p": 1.0,
        "temperature": 1,
        "max_tokens": 250,
        "seed": random.randint(0, 1_000_000),
    }

    logger.info(f"Using args: {JSON}")

    response = None

    try:
        response = requests.post(
            "https://api.corcel.io/cortext/text", json=JSON, headers=HEADERS, timeout=15
        )
        response = response.json()[0]["choices"][0]["delta"]["content"]
    except requests.exceptions.ReadTimeout as e:
        logger.info(
            "Corcel request timed out after 15 seconds..."
            + " falling back to OpenAI..."
        )

    if response:
        logger.info(f"Prompt generated with Corcel: {response}")

    return response


def get_random_creature():
    return random.choice(
        [
            "cat",
            "dog",
            "elephant",
            "lion",
            "butterfly",
            "eagle",
            "dolphin",
            "snake",
            "dragon",
            "unicorn",
            "phoenix",
            "mermaid",
            "centaur",
            "griffin",
            "werewolf",
            "fairy",
            "goblin",
            "minotaur",
            "pegasus",
            "kraken",
            "octopus",
            "panda",
            "giraffe",
            "kangaroo",
            "penguin",
            "parrot",
            "tiger",
            "bear",
            "rabbit",
            "turtle",
            "fox",
            "owl",
            "human",
            "man",
            "woman",
            "girl",
            "boy",
            "robot",
            "drone",
        ]
    )


def get_random_perspective():
    return random.choice(
        [
            "aerial",
            "close-up",
            "panoramic",
            "microscopic",
            "bird's-eye",
            "worm's-eye",
            "fisheye",
            "top-down",
            "side-view",
            "rear-view",
            "isometric",
            "first-person",
            "third-person",
            "macro",
            "wide-angle",
            "telephoto",
            "tilted",
            "skewed",
            "distorted",
            "flipped",
            "mirrored",
            "kaleidoscopic",
            "cross-section",
            "cutaway",
            "translucent",
            "x-ray",
            "thermal",
            "infrared",
            "ultraviolet",
            "low-angle",
            "high-angle",
        ]
    )


def get_random_adjective():
    return random.choice(
        [
            "happy",
            "sad",
            "excited",
            "tired",
            "hungry",
            "playful",
            "curious",
            "brave",
            "majestic",
            "enchanted",
            "mysterious",
            "radiant",
            "ethereal",
            "vibrant",
            "serene",
            "bustling",
            "whimsical",
            "luminous",
            "enigmatic",
            "shiny",
            "colorful",
            "rusty",
            "old",
            "new",
            "large",
            "small",
            "tall",
            "short",
            "wide",
            "narrow",
            "thick",
            "thin",
            "smooth",
        ]
    )


def get_random_object():
    return random.choice(
        [
            "book",
            "pen",
            "phone",
            "camera",
            "guitar",
            "bicycle",
            "car",
            "cup",
            "crystal",
            "tome",
            "amulet",
            "scepter",
            "chalice",
            "orb",
            "mirror",
            "locket",
            "tapestry",
            "sculpture",
            "lamp",
            "chair",
            "table",
            "umbrella",
            "hammer",
            "scissors",
            "knife",
            "spoon",
            "fork",
            "paintbrush",
            "vase",
            "clock",
            "globe",
            "telescope",
            "human",
            "human face",
        ]
    )


def get_random_background():
    return random.choice(
        [
            # Future prompts
            "spaceship",
            "near earth orbit",
            "futuristic city",
            "city of 2033",
            "mars",
            # Standard prompts
            "beach",
            "mountains",
            "city",
            "countryside",
            "park",
            "library",
            "cafe",
            "bedroom",
            "forest",
            "castle",
            "cave",
            "island",
            "desert",
            "underwater",
            "sky",
            "garden",
            "ruins",
            "stadium",
            "mall",
            "factory",
            "farm",
            "school",
            "hospital",
            "airport",
            "train station",
            "bridge",
            "tunnel",
            "highway",
            "river",
            "lake",
            "ocean",
            "space",
        ]
    )


def generate_story_prompt() -> str:
    random.seed(int(time.time()))
    random_creature = get_random_creature()
    random_adjective = get_random_adjective()
    random_object = get_random_object()
    random_background = get_random_background()

    random_perspective = get_random_perspective()

    to_return: str = (
        "You are an image prompt generator. "
        + "Your purpose is to generate a single, "
        + "short story that can be used as a prompt for Dalle-3. "
        + "Please ensure that the story is creative, "
        + "visually descriptive, and coherent. "
        + "The story should be less than 30 words. "
        + "Avoid using any additional elements or deviating from "
        + "the specified creature, adjective, object, and background."
        + "The story **must** incorporate the following elements:\n\n"
        + f"- Background: {random_background}\n\n"
        + f"- Creature: {random_creature}\n"
        + f"- Adjective: {random_adjective}\n"
    )

    if random.random() > 0.85:
        to_return += f"- Object: {random_object}\n"

    if random.random() > 0.85:
        to_return += f"- Perspective: {random_perspective}\n\n"

    return to_return


async def generate_random_prompt_gpt(
    self,
    model: str = "gpt-4",
    prompt: Optional[str] = None,
):
    """Generates random prompt for image generation

    Returns None if there was some error during generation
    (i.e OpenAI request failed etc.)
    """
    response = None
    if not prompt:
        prompt = generate_story_prompt()

    # Generate the prompt from corcel if we have an API key
    if self.corcel_api_key:
        try:
            response = call_corcel(self, prompt)
            if response:
                # Parse response to remove quotes and also adapt
                # the bug with corcel where the output is repeated N times
                response = corcel_parse_response(response)
                if response.startswith("{"):
                    response = None
        except Exception as e:
            logger.error(f"An unexpected error occurred calling corcel: {e}")
            logger.error("Falling back to OpenAI if available...")

    if not response:
        openai_service = get_openai_service()
        try:
            response = await openai_service.create_completion_request(model, prompt)
        except OpenAIRequestFailed as e:
            logger.info(f"error during creation of completion prompt: {e}")

    # Remove any double quotes from the output
    if response:
        response = response.replace('"', "")
        response = response.strip()

    return response


def generate_followup_prompt_gpt(
    self,
    prompt,
    model="gpt-4",
    followup_prompt="An image has now been generated from your first prompt."
    + " What is a second instruction that can be applied to this generated image?",
):
    # Update this for next week. Combine this and the method above.
    messages = [
        {"role": "system", "content": "You are an image prompt generator."},
        {"role": "assistant", "content": f"{prompt}"},
        {
            "role": "user",
            "content": f"{followup_prompt}",
        },
    ]

    for _ in range(2):
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            new_prompt = response.choices[0].message.content
            logger.info(f"I2I prompt is {new_prompt}")
            return new_prompt

        except Exception as e:
            logger.error(f"Error when calling OpenAI: {e}")
            time.sleep(0.5)

    return None


def measure_time(func):
    """This decorator logs time of function execution"""

    @wraps(func)
    def sync_measure_time_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.warning(
            f"[measure_time] function {func.__name__} took {total_time:.2f} seconds"
        )
        return result

    async def async_measure_time_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.warning(
            f"[measure_time] async function {func.__name__} took {total_time:.2f} seconds"
        )
        return result

    if asyncio.iscoroutinefunction(func):
        return async_measure_time_wrapper
    else:
        return sync_measure_time_wrapper


def get_device_name(device: torch.device):
    """Returns name of GPU model"""
    try:
        if device.type == "cuda":
            # Fetch the device index and then get the device name
            device_name = torch.cuda.get_device_name(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            return device_name

        # Return 'CPU' as it does not have a specific name like GPUs do
        return "CPU"
    except Exception as e:
        logger.error(f"failed to get device name: {e}")
        return "n/a"
