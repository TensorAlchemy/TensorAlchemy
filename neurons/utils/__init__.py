import os
import sys
import json
import shutil
import traceback
import subprocess
import multiprocessing

from datetime import datetime
from threading import Timer

import torch

import _thread

from loguru import logger
from google.cloud import storage

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
    N_NEURONS,
    WANDB_MINER_PATH,
    WANDB_VALIDATOR_PATH,
)
from neurons.utils.log import colored_log

from neurons.validator.utils.wandb import init_wandb
from neurons.validator.rewards.types import (
    RewardModelType,
)


# Background Loop
class BackgroundTimer(Timer):
    def run(self):
        self.function(*self.args, **self.kwargs)
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class MultiprocessBackgroundTimer(multiprocessing.Process):
    def __init__(self, interval, function, args=None, kwargs=None):
        super().__init__()
        self.interval = interval
        self.function = function
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.finished = multiprocessing.Event()

    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)

    def cancel(self):
        self.finished.set()


def get_coldkey_for_hotkey(self, hotkey):
    """
    Look up the coldkey of the caller.
    """
    if hotkey in self.metagraph.hotkeys:
        index = self.metagraph.hotkeys.index(hotkey)
        return self.metagraph.coldkeys[index]
    return None


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
                    reward_names = [
                        RewardModelType.IMAGE,
                        # TODO: RewardModelType.SIMILARITY,
                    ]

                    for rw_name in reward_names:
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
            logger.error(
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
                    if "run-" in x and "latest-run" not in x
                ]
                if len(runs) > 0:
                    subprocess.call(
                        f"cd {wandb_path} && echo 'y' | wandb sync --clean --clean-old-hours 3",
                        shell=True,
                    )
                    logger.info("Cleaned all synced wandb runs.")
                    subprocess.Popen(
                        ["wandb artifact cache cleanup 5GB"],
                        shell=True,
                    )
                    logger.info("Cleaned all wandb cache data > 5GB.")

                # Catch any runs that the stock wandb function doesn't
                runs = [
                    x
                    for x in os.listdir(f"{wandb_path}/wandb")
                    if "run-" in x and "latest-run" not in x
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
