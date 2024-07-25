import os
import sys
import inspect
import asyncio
import traceback
import multiprocessing

from threading import Timer

import torch

import _thread

from loguru import logger
from google.cloud import storage

from neurons.constants import (
    IA_MINER_BLACKLIST,
    IA_MINER_WARNINGLIST,
    IA_MINER_WHITELIST,
    IA_VALIDATOR_BLACKLIST,
    IA_VALIDATOR_WEIGHT_FILES,
    IA_VALIDATOR_WHITELIST,
    N_NEURONS,
)
from neurons.utils.common import is_validator
from neurons.utils.gcloud import retrieve_public_file

from neurons.validator.scoring.models.types import (
    RewardModelType,
)
from neurons.utils.log import configure_logging


# Background Loop
class BackgroundTimer(Timer):
    def run(self):
        configure_logging()
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
        configure_logging()

        logger.info(f"{self.function.__name__} started")

        while not self.finished.is_set():
            try:
                if inspect.iscoroutinefunction(self.function):
                    asyncio.run(self.function(*self.args, **self.kwargs))
                else:
                    self.function(*self.args, **self.kwargs)

                self.finished.wait(self.interval)

            except Exception as e:
                logger.error(traceback.format_exc())

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
    whitelist_type = (
        IA_VALIDATOR_WHITELIST if is_validator else IA_MINER_WHITELIST
    )
    blacklist_type = (
        IA_VALIDATOR_BLACKLIST if is_validator else IA_MINER_BLACKLIST
    )
    warninglist_type = IA_MINER_WARNINGLIST

    # Terminate the miner / validator after deregistration
    if self.background_steps % 5 == 0 and self.background_steps > 1:
        try:
            self.metagraph.sync(subtensor=self.subtensor)
            if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
                logger.info(
                    f">>> {neuron_type} has deregistered... terminating."
                )
                try:
                    _thread.interrupt_main()
                except Exception as e:
                    logger.info(
                        f"An error occurred trying to terminate the main thread: {e}."
                    )
                try:
                    os.exit(0)
                except Exception as e:
                    logger.error(
                        f"An error occurred trying to use os._exit(): {e}."
                    )
                sys.exit(0)
        except Exception as e:
            logger.error(
                f">>> An unexpected error occurred syncing the metagraph: {e}"
            )

    # Update the whitelists and blacklists
    if self.background_steps % 5 == 0:
        try:
            # Create client if needed
            if not self.storage_client:
                self.storage_client = storage.Client.create_anonymous_client()
                logger.info("Created anonymous storage client.")

            # Update the blacklists
            blacklist_for_neuron = retrieve_public_file(
                self.storage_client, blacklist_type
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
                self.storage_client, whitelist_type
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
                self.storage_client, warninglist_type
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
                if (
                    self.wallet.hotkey.ss58_address
                    in self.hotkey_warninglist.keys()
                ):
                    hotkey_address: str = self.hotkey_warninglist[
                        self.wallet.hotkey.ss58_address
                    ][0]
                    hotkey_warning: str = self.hotkey_warninglist[
                        self.wallet.hotkey.ss58_address
                    ][1]

                    logger.info(
                        f"This hotkey is on the warning list: {hotkey_address}"
                        + f" | Date for rectification: {hotkey_warning}",
                    )

                coldkey = get_coldkey_for_hotkey(
                    self, self.wallet.hotkey.ss58_address
                )
                if coldkey in self.coldkey_warninglist.keys():
                    coldkey_address: str = self.coldkey_warninglist[coldkey][0]
                    coldkey_warning: str = self.coldkey_warninglist[coldkey][1]
                    logger.info(
                        f"This coldkey is on the warning list: {coldkey_address}"
                        + f" | Date for rectification: {coldkey_warning}",
                    )

            # Validator only
            if is_validator:
                # Update weights
                validator_weights = retrieve_public_file(
                    self.storage_client, IA_VALIDATOR_WEIGHT_FILES
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
                    ] / (
                        (total_number_of_neurons / N_NEURONS)
                        * adjustment_factor
                    )

                if validator_weights:
                    weights_to_add = []
                    reward_names = [
                        RewardModelType.IMAGE,
                    ]

                    for rw_name in reward_names:
                        if rw_name in validator_weights:
                            weights_to_add.append(validator_weights[rw_name])

                    logger.info(f"Raw model weights: {weights_to_add}")

                    if weights_to_add:
                        # Normalize weights
                        if sum(weights_to_add) != 1:
                            weights_to_add = normalize_weights(weights_to_add)
                            logger.info(
                                f"Normalized model weights: {weights_to_add}"
                            )

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

        except Exception as e:
            logger.error(
                f"An error occurred trying to update settings from the cloud: {e}."
            )

    self.background_steps += 1


def normalize_weights(weights):
    sum_weights = float(sum(weights))
    normalizer = 1 / sum_weights
    weights = [weight * normalizer for weight in weights]
    if sum(weights) < 1:
        diff = 1 - sum(weights)
        weights[0] += diff

    return weights
