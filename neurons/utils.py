import _thread
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from threading import Timer
from typing import Any, Dict, List

import requests
import sentry_sdk
import torch
from google.cloud import storage
from loguru import logger
from neurons.constants import (
    IA_BUCKET_NAME,
    IA_MINER_BLACKLIST,
    IA_MINER_WARNINGLIST,
    IA_MINER_WHITELIST,
    IA_TEST_BUCKET_NAME,
    IA_VALIDATOR_BLACKLIST,
    IA_VALIDATOR_SETTINGS_FILE,
    IA_VALIDATOR_WEIGHT_FILES,
    IA_VALIDATOR_WHITELIST,
    MINIMUM_COMPUTES_FOR_SUBMIT,
    N_NEURONS,
    NSFW_WORDLIST_DEFAULT,
    NSFW_WORDLIST_URL,
    WANDB_MINER_PATH,
    WANDB_VALIDATOR_PATH,
)
from neurons.exceptions import MinimumValidImagesError
from neurons.validator.backend.client import TensorAlchemyBackendClient
from neurons.validator.backend.exceptions import UpdateTaskError
from neurons.validator.backend.models import TaskState
from neurons.validator.utils import init_wandb
from pydantic import BaseModel


@dataclass
class Stats:
    start_time: datetime
    start_dt: datetime
    total_requests: int
    timeouts: int
    nsfw_count: int
    generation_time: int


# Colors to use in the logs
COLORS = {
    "r": "\033[1;31;40m",
    "g": "\033[1;32;40m",
    "b": "\033[1;34;40m",
    "y": "\033[1;33;40m",
    "m": "\033[1;35;40m",
    "c": "\033[1;36;40m",
    "w": "\033[1;37;40m",
}


def load_nsfw_words(url: str) -> List[str]:
    try:
        response = requests.get(url)
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Split the content into lines and strip whitespace
        words = [line.strip() for line in response.text.splitlines()]

        # Remove empty lines
        words = [word for word in words if word]

        return words

    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred while loading NSFW words from {url}: {str(e)}")
        return NSFW_WORDLIST_DEFAULT


NSFW_WORDS: List[str] = load_nsfw_words(NSFW_WORDLIST_URL)


# Utility function for coloring logs
def colored_log(
    message: str,
    color: str = "white",
    level: str = "INFO",
) -> None:
    logger.opt(colors=True).log(level, f"<bold><{color}>{message}</{color}></bold>")


def sh(message: str):
    return f"{message: <12}"


# Get default stats
def get_defaults(self):
    now = datetime.now()
    stats = Stats(
        start_time=now,
        start_dt=datetime.strftime(now, "%Y/%m/%d %H:%M"),
        total_requests=0,
        nsfw_count=0,
        timeouts=0,
        generation_time=0,
    )
    return stats


# Background Loop
class BackgroundTimer(Timer):
    def run(self):
        self.function(*self.args, **self.kwargs)
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


def get_coldkey_for_hotkey(self, hotkey):
    """
    Look up the coldkey of the caller.
    """
    if hotkey in self.metagraph.hotkeys:
        index = self.metagraph.hotkeys.index(hotkey)
        return self.metagraph.coldkeys[index]
    return None


MINIMUM_VALID_IMAGES_ERROR: str = "MINIMUM_VALID_IMAGES_ERROR"


class BatchSubmissionRequest(BaseModel):
    batch_id: str
    # Results
    prompt: str
    computes: List[str]

    # Filtering
    nsfw_scores: List[float]
    blacklist_scores: List[int] = []
    should_drop_entries: List[int] = []

    # Miner
    miner_hotkeys: List[str]
    miner_coldkeys: List[str]

    # Validator
    validator_hotkey: str


def filter_batch_before_submission(batch: Dict[str, Any]) -> Dict[str, Any]:
    to_return: Dict[str, Any] = {
        "batch_id": batch["batch_id"],
        "prompt": batch["prompt"],
        # Compute specific stuff
        "computes": [],
        "miner_hotkeys": [],
        "miner_coldkeys": [],
        "validator_hotkey": batch["validator_hotkey"],
        "nsfw_scores": [],
        "blacklist_scores": [],
        "should_drop_entries": [],
    }

    for idx, compute in enumerate(batch["computes"]):
        should_drop_entry = batch["should_drop_entries"][idx]
        if should_drop_entry > 0:
            logger.info("Dropped one submitted image")
            continue

        blacklist_score = batch["blacklist_scores"][idx]
        if blacklist_score < 1:
            logger.info("Dropped one blacklisted image")
            continue

        nsfw_score = batch["nsfw_scores"][idx]
        if nsfw_score < 1:
            logger.info("Dropped one NSFW image")
            continue

        to_return["computes"].append(compute)
        to_return["blacklist_scores"].append(batch["blacklist_scores"][idx])
        to_return["miner_coldkeys"].append(batch["miner_coldkeys"][idx])
        to_return["nsfw_scores"].append(batch["nsfw_scores"][idx])
        to_return["should_drop_entries"].append(batch["should_drop_entries"][idx])
        to_return["miner_hotkeys"].append(batch["miner_hotkeys"][idx])

    if len(to_return["computes"]) < MINIMUM_COMPUTES_FOR_SUBMIT:
        raise MinimumValidImagesError()

    return dict(BatchSubmissionRequest(**to_return))


async def upload_batches(
    backend_client: TensorAlchemyBackendClient, batches: Dict
) -> Dict:
    logger.info(f"Number of batches in queue: {len(batches)}")
    max_retries = 3
    backoff = 5
    batches_for_deletion = set()

    for batch_id in batches.keys():
        batch = batches[batch_id]
        for attempt in range(0, max_retries):
            try:
                filtered_batch = filter_batch_before_submission(batch)
                response = await backend_client.post_batch(filtered_batch)
                if response.status_code == 200:
                    logger.info(
                        "Successfully posted batch" + f" {filtered_batch['batch_id']}"
                    )
                    batches_for_deletion.add(batch_id)
                    break

                response_data = response.json()
                if "code" in response_data:
                    if response_data["code"] == MINIMUM_VALID_IMAGES_ERROR:
                        batches_for_deletion.add(batch_id)
                        break

                logger.info(f"{response_data=}")
                raise Exception(
                    "Failed to post batch. " + f"Status code: {response.status_code}"
                )
            except MinimumValidImagesError as e:
                batches_for_deletion.add(batch_id)
                try:
                    await backend_client.update_task_state(
                        batch_id,
                        TaskState.FAILED,
                    )
                except UpdateTaskError:
                    logger.error(
                        f"Failed to post {batch_id} to the {TaskState.FAILED.value} endpoint"
                    )

                break
            except Exception as e:
                backoff *= 2  # Double the backoff for the next attempt
                if attempt != max_retries:
                    logger.error(
                        f"Attempt number {attempt+1} failed to"
                        + f" send batch {batch['batch_id']}. "
                        + f"Retrying in {backoff} seconds. Error: {e}"
                    )
                    time.sleep(backoff)
                    continue

                logger.error(
                    f"Attempted to post batch {batch['batch_id']} "
                    + f"{attempt+1} times unsuccessfully. "
                    + f"Skipping this batch and moving to the next batch. Error: {e}"
                )
                batches_for_deletion.add(batch_id)
                break

    # Delete any processed branches
    new_batches: Dict = {}

    for batch_id, batch in batches.items():
        if batch_id not in batches_for_deletion:
            new_batches[batch_id] = batch

    return new_batches


def background_loop(self, is_validator):
    """
    Handles terminating the miner after deregistration and
    updating the blacklist and whitelist.
    """
    neuron_type = "Validator" if is_validator else "Miner"
    whitelist_type = IA_VALIDATOR_WHITELIST if is_validator else IA_MINER_WHITELIST
    blacklist_type = IA_VALIDATOR_BLACKLIST if is_validator else IA_MINER_BLACKLIST
    warninglist_type = IA_MINER_WARNINGLIST

    bucket_name = (
        IA_TEST_BUCKET_NAME if self.subtensor.network == "test" else IA_BUCKET_NAME
    )

    # Terminate the miner / validator after deregistration
    if self.background_steps % 5 == 0 and self.background_steps > 1:
        try:
            self.metagraph.sync(subtensor=self.subtensor)
            if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
                logger.info(f">>> {neuron_type} has deregistered... terminating.")
                try:
                    _thread.interrupt_main()
                except Exception as e:
                    logger.info(
                        f"An error occurred trying to terminate the main thread: {e}."
                    )
                try:
                    os.exit(0)
                except Exception as e:
                    logger.info(f"An error occurred trying to use os._exit(): {e}.")
                sys.exit(0)
        except Exception as e:
            logger.info(f">>> An unexpected error occurred syncing the metagraph: {e}")

    # Send new batches to the Human Validation Bot
    try:
        if (self.background_steps % 1 == 0) and is_validator and (self.batches != {}):
            self.batches = asyncio.run(
                upload_batches(self.backend_client, self.batches)
            )

    except Exception as e:
        logger.info(
            f"An error occurred trying to submit a batch: "
            + f"{e}\n{traceback.format_exc()}"
        )
        sentry_sdk.capture_exception(e)

    # Update the whitelists and blacklists
    if self.background_steps % 5 == 0:
        try:
            # Create client if needed
            if not self.storage_client:
                self.storage_client = storage.Client.create_anonymous_client()
                logger.info("Created anonymous storage client.")

            # Update the blacklists
            blacklist_for_neuron = retrieve_public_file(
                self.storage_client, bucket_name, blacklist_type
            )
            if blacklist_for_neuron:
                self.hotkey_blacklist = set(
                    [
                        k
                        for k, v in blacklist_for_neuron.items()
                        if v["type"] == "hotkey"
                    ]
                )
                self.coldkey_blacklist = set(
                    [
                        k
                        for k, v in blacklist_for_neuron.items()
                        if v["type"] == "coldkey"
                    ]
                )
                logger.info("Retrieved the latest blacklists.")

            # Update the whitelists
            whitelist_for_neuron = retrieve_public_file(
                self.storage_client, bucket_name, whitelist_type
            )
            if whitelist_for_neuron:
                self.hotkey_whitelist = set(
                    [
                        k
                        for k, v in whitelist_for_neuron.items()
                        if v["type"] == "hotkey"
                    ]
                )
                self.coldkey_whitelist = set(
                    [
                        k
                        for k, v in whitelist_for_neuron.items()
                        if v["type"] == "coldkey"
                    ]
                )
                logger.info("Retrieved the latest whitelists.")

            # Update the warning list
            warninglist_for_neuron = retrieve_public_file(
                self.storage_client, bucket_name, warninglist_type
            )
            if warninglist_for_neuron:
                self.hotkey_warninglist = {
                    k: [v["reason"], v["resolve_by"]]
                    for k, v in warninglist_for_neuron.items()
                    if v["type"] == "hotkey"
                }
                self.coldkey_warninglist = {
                    k: [v["reason"], v["resolve_by"]]
                    for k, v in warninglist_for_neuron.items()
                    if v["type"] == "coldkey"
                }
                logger.info("Retrieved the latest warninglists.")
                if self.wallet.hotkey.ss58_address in self.hotkey_warninglist.keys():
                    hotkey_address: str = self.hotkey_warninglist[
                        self.wallet.hotkey.ss58_address
                    ][0]
                    hotkey_warning: str = self.hotkey_warninglist[
                        self.wallet.hotkey.ss58_address
                    ][1]

                    colored_log(
                        f"This hotkey is on the warning list: {hotkey_address}"
                        + f" | Date for rectification: {hotkey_warning}",
                        color="red",
                    )
                coldkey = get_coldkey_for_hotkey(self, self.wallet.hotkey.ss58_address)
                if coldkey in self.coldkey_warninglist.keys():
                    coldkey_address: str = self.coldkey_warninglist[coldkey][0]
                    coldkey_warning: str = self.coldkey_warninglist[coldkey][1]
                    colored_log(
                        f"This coldkey is on the warning list: {coldkey_address}"
                        + f" | Date for rectification: {coldkey_warning}",
                        color="red",
                    )

            # Validator only
            if is_validator:
                # Update weights
                validator_weights = retrieve_public_file(
                    self.storage_client, bucket_name, IA_VALIDATOR_WEIGHT_FILES
                )

                if "human_reward_model" in validator_weights:
                    # NOTE: Scaling factor for the human reward model
                    #
                    # The human reward model updates the rewards for all
                    # neurons (256 on mainnet) in each step, while the
                    # other reward models only update rewards for a subset
                    # of neurons (e.g., 12) per step.
                    #
                    # To avoid rewards being updated out of sync,
                    # we scale down the human rewards in each step.
                    #
                    # The scaling factor is calculated as the total number
                    # of neurons divided by the number of neurons updated
                    # per step,
                    #
                    # Then multiplied by an adjustment factor (1.5) to account
                    # for potential duplicate neuron selections during a full
                    # epoch.
                    #
                    # The adjustment factor of 1.5 was determined empirically
                    # based on the observed number of times UIDs received
                    # duplicate calls in a full epoch on the mainnet.
                    adjustment_factor: float = 1.5
                    total_number_of_neurons: int = self.metagraph.n.item()

                    self.human_voting_weight = validator_weights[
                        "human_reward_model"
                    ] / ((total_number_of_neurons / N_NEURONS) * adjustment_factor)

                if validator_weights:
                    weights_to_add = []
                    for rw_name in self.reward_names:
                        if rw_name in validator_weights:
                            weights_to_add.append(validator_weights[rw_name])

                    logger.info(f"Raw model weights: {weights_to_add}")

                    if weights_to_add:
                        # Normalize weights
                        if sum(weights_to_add) != 1:
                            weights_to_add = normalize_weights(weights_to_add)
                            logger.info(f"Normalized model weights: {weights_to_add}")

                        self.reward_weights = torch.tensor(
                            weights_to_add, dtype=torch.float32
                        ).to(self.device)
                        logger.info(
                            f"Retrieved the latest validator weights: {self.reward_weights}"
                        )

                    # self.reward_weights = torch.tensor(
                    # [v for k, v in validator_weights.items() if "manual" not in k],
                    # dtype=torch.float32,
                    # ).to(self.device)

                # Update settings
                validator_settings: dict = retrieve_public_file(
                    self.storage_client,
                    bucket_name,
                    IA_VALIDATOR_SETTINGS_FILE,
                )

                if validator_settings:
                    self.request_frequency = validator_settings.get(
                        "request_frequency", self.request_frequency
                    )

                    self.query_timeout = validator_settings.get(
                        "query_timeout", self.query_timeout
                    )

                    self.async_timeout = validator_settings.get(
                        "async_timeout", self.async_timeout
                    )

                    self.epoch_length = validator_settings.get(
                        "epoch_length", self.epoch_length
                    )

                    logger.info(
                        f"Retrieved the latest validator settings: {validator_settings}"
                    )

        except Exception as e:
            logger.info(
                f"An error occurred trying to update settings from the cloud: {e}."
            )

    # Clean up the wandb runs and cache folders
    if self.background_steps == 1 or self.background_steps % 180 == 0:
        logger.info("Trying to clean wandb directoy...")
        wandb_path = WANDB_VALIDATOR_PATH if is_validator else WANDB_MINER_PATH
        try:
            if os.path.exists(wandb_path):
                # Write a condition to skip this if there are no runs to clean
                # os.path.basename(path).split("run-")[1].split("-")[0], "%Y%m%d_%H%M%S"
                runs = [
                    x
                    for x in os.listdir(f"{wandb_path}/wandb")
                    if "run-" in x and not "latest-run" in x
                ]
                if len(runs) > 0:
                    subprocess.call(
                        f"cd {wandb_path} && echo 'y' | wandb sync --clean --clean-old-hours 3",
                        shell=True,
                    )
                    logger.info("Cleaned all synced wandb runs.")
                    subprocess.Popen(["wandb artifact cache cleanup 5GB"], shell=True)
                    logger.info("Cleaned all wandb cache data > 5GB.")

                # Catch any runs that the stock wandb function doesn't
                runs = [
                    x
                    for x in os.listdir(f"{wandb_path}/wandb")
                    if "run-" in x and not "latest-run" in x
                ]

                # Leave the most recent 3 runs
                try:
                    if len(runs) > 3:
                        # Sort runs
                        runs = sorted(
                            runs,
                            key=lambda x: datetime.strptime(
                                x.split("run-")[-1].rsplit("-")[0], "%Y%m%d_%H%M%S"
                            ),
                        )
                        for run in runs[:-3]:
                            shutil.rmtree(f"{wandb_path}/wandb/{run}")

                        logger.info("Finished cleaning out old runs...")
                except Exception as e:
                    logger.warning(f"Failed to manually delete old wandb runs: {e}")

            else:
                logger.warning(f"The path {wandb_path} doesn't exist yet.")
        except Exception as e:
            logger.error(
                f"An error occurred trying to clean wandb artifacts and runs: {e}."
            )

    # Attempt to init wandb if it wasn't sucessfully originally
    if (self.background_steps % 5 == 0) and is_validator and not self.wandb_loaded:
        try:
            init_wandb(self)
            logger.info("Loaded wandb")
            self.wandb_loaded = True
        except Exception:
            self.wandb_loaded = False
            logger.error("Unable to load wandb. Retrying in 5 minutes.")
            logger.error(f"wandb loading error: {traceback.format_exc()}")

    self.background_steps += 1


def normalize_weights(weights):
    sum_weights = float(sum(weights))
    normalizer = 1 / sum_weights
    weights = [weight * normalizer for weight in weights]
    if sum(weights) < 1:
        diff = 1 - sum(weights)
        weights[0] += diff

    return weights


def retrieve_public_file(client, bucket_name, source_name):
    file = None
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_name)
        try:
            file = blob.download_as_text()
            file = json.loads(file)
            logger.info(
                f"Successfully downloaded {source_name} " + f"from {bucket_name}"
            )
        except Exception as e:
            logger.info(
                f"Failed to download {source_name} from " + f"{bucket_name}: {e}"
            )

    except Exception as e:
        logger.info(f"An error occurred downloading from Google Cloud: {e}")

    return file


def clean_nsfw_from_prompt(prompt):
    for word in NSFW_WORDS:
        if re.search(r"\b{}\b".format(word), prompt):
            prompt = re.sub(r"\b{}\b".format(word), "", prompt).strip()
            logger.warning(f"Removed NSFW word {word.strip()} from prompt...")

    return prompt
