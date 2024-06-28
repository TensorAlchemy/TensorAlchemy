import os
import sys
import argparse
import asyncio
import copy
import random
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import bittensor as bt
import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
from loguru import logger
from neurons.protocol import ImageGeneration, IsAlive, ModelType
from neurons.utils import (
    BackgroundTimer,
    background_loop,
    get_defaults,
)
from neurons.constants import VPERMIT_TAO
from neurons.utils.nsfw import clean_nsfw_from_prompt
from utils import (
    colored_log,
    do_logs,
    get_caller_stake,
    get_coldkey_for_hotkey,
    nsfw_image_filter,
    sh,
)

from neurons.miners.StableMiner.base import ModelConfig, TaskType

from wandb_utils import WandbUtils


class BaseMiner(ABC):
    def __init__(self) -> None:
        # Parse the config
        self.config: bt.config = self.get_config()

        self.wandb: Optional[WandbUtils] = None

        if self.config.logging.debug:
            bt.debug()
            logger.info("Enabling debug mode...")

        # Output the config
        colored_log("Outputting miner config:", color="green")
        colored_log(f"{self.config}", color="green")

        # Build args
        self.t2i_args: Dict[str, Any] = self.get_t2i_args()
        self.i2i_args: Dict[str, Any] = self.get_i2i_args()

        # Init blacklists and whitelists
        self.hotkey_blacklist: set = set()
        self.coldkey_blacklist: set = set()
        self.coldkey_whitelist: set = set(
            ["5F1FFTkJYyceVGE4DCVN5SxfEQQGJNJQ9CVFVZ3KpihXLxYo"]
        )
        self.hotkey_whitelist: set = set(
            ["5C5PXHeYLV5fAx31HkosfCkv8ark3QjbABbjEusiD3HXH2Ta"]
        )

        self.storage_client: Any = None

        # Initialise event dict
        self.event: Dict[str, Any] = {}
        self.mapping: Dict[str, Dict] = {}

        # Establish subtensor connection
        logger.info("Establishing subtensor connection")
        self.subtensor: bt.subtensor = bt.subtensor(config=self.config)

        # Create the metagraph
        self.metagraph: bt.metagraph = self.subtensor.metagraph(
            netuid=self.config.netuid
        )

        # Configure the wallet
        self.wallet: bt.wallet = bt.wallet(config=self.config)

        # Wait until the miner is registered
        self.loop_until_registered()

        # Defaults
        self.stats: Dict[str, Any] = get_defaults(self)

        # Set up transform function
        self.transform: transforms.Compose = transforms.Compose(
            [transforms.PILToTensor()]
        )

        # Start the wandb logging thread if both project
        # and entity have been provided
        if all(
            [
                self.config.wandb.project,
                self.config.wandb.entity,
                self.config.wandb.api_key,
            ]
        ):
            self.wandb = WandbUtils(
                self,
                self.metagraph,
                self.config,
                self.wallet,
                self.event,
            )

        # Start the generic background loop
        self.background_steps: int = 1
        self.background_timer: BackgroundTimer = BackgroundTimer(
            300,
            background_loop,
            [self, False],
        )
        self.background_timer.daemon = True
        self.background_timer.start()

        # Init history dict
        self.request_dict: Dict[str, Dict[str, Union[List[float], int]]] = {}

    def start_axon(self) -> None:
        # Serve the axon
        colored_log(
            f"Serving axon on port {self.config.axon.port}.",
            color="green",
        )
        try:
            self.axon: bt.axon = (
                bt.axon(
                    wallet=self.wallet,
                    ip=bt.utils.networking.get_external_ip(),
                    external_ip=self.config.axon.get("external_ip")
                    or bt.utils.networking.get_external_ip(),
                    config=self.config,
                )
                .attach(
                    forward_fn=self.is_alive,
                    blacklist_fn=self.blacklist_is_alive,
                    priority_fn=self.priority_is_alive,
                )
                .attach(
                    forward_fn=self.generate_image,
                    blacklist_fn=self.blacklist_image_generation,
                    priority_fn=self.priority_image_generation,
                )
                .start()
            )
            colored_log(f"Axon created: {self.axon}", color="green")

            self.subtensor.serve_axon(axon=self.axon, netuid=self.config.netuid)
        except Exception as e:
            logger.error(f"Failed to start axon: {e}")
            raise

    @abstractmethod
    def get_t2i_args(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_i2i_args(self) -> Dict[str, Any]:
        pass

    def get_config(self) -> bt.config:
        argp = argparse.ArgumentParser(description="Miner Configs")

        # Add any args from the parent class
        self.add_args(argp)

        argp.add_argument("--netuid", type=int, default=1)
        argp.add_argument("--wandb.project", type=str, default="")
        argp.add_argument("--wandb.entity", type=str, default="")
        argp.add_argument("--wandb.api_key", type=str, default="")
        argp.add_argument("--miner.device", type=str, default="cuda:0")
        argp.add_argument("--miner.optimize", action="store_true")

        seed = random.randint(0, 100_000_000_000)
        argp.add_argument("--miner.seed", type=int, default=seed)

        argp.add_argument(
            "--miner.custom_model",
            type=str,
            default="stabilityai/stable-diffusion-xl-base-1.0",
        )

        argp.add_argument(
            "--miner.alchemy_model",
            type=str,
            default="stabilityai/stable-diffusion-xl-base-1.0",
        )

        bt.subtensor.add_args(argp)
        bt.logging.add_args(argp)
        bt.wallet.add_args(argp)
        bt.axon.add_args(argp)

        config = bt.config(argp)

        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
                "miner",
            )
        )
        # Ensure the directory for logging exists
        if not os.path.exists(config.full_path):
            os.makedirs(config.full_path, exist_ok=True)

        return config

    @abstractmethod
    def add_args(self, argp: argparse.ArgumentParser) -> None:
        pass

    def loop_until_registered(self) -> None:
        while True:
            try:
                self.miner_index: Optional[int] = self.get_miner_index()
                if self.miner_index is not None:
                    logger.info(
                        f"Miner {self.config.wallet.hotkey} is registered with uid "
                        f"{self.metagraph.uids[self.miner_index]}"
                    )
                    break

                logger.warning(
                    f"Miner {self.config.wallet.hotkey} is not registered. "
                    "Sleeping for 120 seconds..."
                )
                time.sleep(120)
                self.metagraph.sync(subtensor=self.subtensor)
            except Exception as e:
                logger.error(f"Error in loop_until_registered: {e}")
                time.sleep(120)

    def get_miner_info(self) -> Dict[str, Union[int, float]]:
        try:
            return {
                "block": self.metagraph.block.item(),
                "stake": self.metagraph.stake[self.miner_index].item(),
                "trust": self.metagraph.trust[self.miner_index].item(),
                "consensus": self.metagraph.consensus[self.miner_index].item(),
                "incentive": self.metagraph.incentive[self.miner_index].item(),
                "emissions": self.metagraph.emission[self.miner_index].item(),
            }
        except Exception as e:
            logger.error(f"Error in get_miner_info: {e}")
            return {}

    def get_miner_index(self) -> Optional[int]:
        try:
            return self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        except ValueError:
            return None

    def check_still_registered(self) -> bool:
        return self.get_miner_index() is not None

    def get_incentive(self) -> float:
        if self.miner_index is not None:
            return self.metagraph.I[self.miner_index].item() * 100_000
        return 0.0

    def get_trust(self) -> float:
        if self.miner_index is not None:
            return self.metagraph.T[self.miner_index].item() * 100
        return 0.0

    def get_consensus(self) -> float:
        if self.miner_index is not None:
            return self.metagraph.C[self.miner_index].item() * 100_000
        return 0.0

    async def is_alive(self, synapse: IsAlive) -> IsAlive:
        logger.info("IsAlive")
        synapse.completion = "True"
        return synapse

    async def generate_image(self, synapse: ImageGeneration) -> ImageGeneration:
        """
        Image generation logic shared between both text-to-image and image-to-image
        """

        # Misc
        timeout: float = synapse.timeout
        self.stats["total_requests"] += 1
        start_time: float = time.perf_counter()

        # Set up args
        model_type: str = ModelType.CUSTOM
        if synapse.model_type is not None:
            model_type = synapse.model_type

        try:
            model_config: ModelConfig = self.get_model_config(
                model_type,
                synapse.generation_type.upper(),
            )
        except ValueError as e:
            logger.error(f"Error getting model config: {e}")
            return synapse

        local_args: Dict[str, Any] = copy.deepcopy(model_config.args)

        try:
            local_args["prompt"] = [clean_nsfw_from_prompt(synapse.prompt)]
            local_args["width"] = synapse.width
            local_args["height"] = synapse.height
            local_args["num_images_per_prompt"] = synapse.num_images_per_prompt
            local_args["guidance_scale"] = synapse.guidance_scale

            if synapse.negative_prompt:
                local_args["negative_prompt"] = [synapse.negative_prompt]

            local_args["num_inference_steps"] = getattr(
                synapse, "steps", local_args.get("num_inference_steps", 50)
            )
        except AttributeError as e:
            logger.error(f"Error setting up local args: {e}")

        # Get the model
        model = model_config.model

        if synapse.generation_type.upper() == TaskType.IMAGE_TO_IMAGE:
            try:
                local_args["image"] = T.transforms.ToPILImage()(
                    bt.Tensor.deserialize(synapse.prompt_image)
                )
            except Exception as e:
                logger.error(f"Error processing image for image-to-image: {e}")
                return synapse

        # Output logs
        do_logs(self, synapse, local_args)

        # Generate images & serialize
        for attempt in range(3):
            try:
                seed: int = synapse.seed
                local_args["generator"] = [
                    torch.Generator(device=self.config.miner.device).manual_seed(seed)
                ]
                images = model(**local_args).images

                synapse.images = [
                    bt.Tensor.serialize(self.transform(image)) for image in images
                ]
                colored_log(
                    f"{sh('Generating')} -> Successful image generation after"
                    f" {attempt+1} attempt(s).",
                    color="cyan",
                )
                break
            except Exception as e:
                logger.error(
                    f"Error in attempt number {attempt+1} to generate an image:"
                    f" {e}... sleeping for 5 seconds..."
                )
                await asyncio.sleep(5)
                if attempt == 2:
                    images = []
                    synapse.images = []
                    logger.info(
                        f"Failed to generate any images after" f" {attempt+1} attempts."
                    )

        # Count timeouts
        if time.perf_counter() - start_time > timeout:
            self.stats["timeouts"] += 1

        # Log NSFW images
        try:
            if any(nsfw_image_filter(self, images)):
                logger.info(f"An image was flagged as NSFW: discarding image.")
                self.stats["nsfw_count"] += 1
                synapse.images = []
        except Exception as e:
            logger.error(f"Error in NSFW filtering: {e}")

        # Log to wandb
        try:
            if self.wandb:
                # Store the images and prompts for uploading to wandb
                self.wandb._add_images(synapse)

                # Log to Wandb
                self.wandb._log()

        except Exception as e:
            logger.error(f"Error trying to log events to wandb: {e}")

        # Log time to generate image
        generation_time: float = time.perf_counter() - start_time
        self.stats["generation_time"] += generation_time

        average_time: float = (
            self.stats["generation_time"] / self.stats["total_requests"]
        )
        colored_log(
            f"{sh('Time')} -> {generation_time:.2f}s "
            f"| Average: {average_time:.2f}s",
            color="yellow",
        )
        return synapse

    def _base_priority(self, synapse: Union[IsAlive, ImageGeneration]) -> float:
        # If hotkey or coldkey is whitelisted
        # and not found on the metagraph, give a priority of 5,000
        # Caller hotkey
        caller_hotkey: str = synapse.axon.hotkey

        try:
            # Retrieve the coldkey of the caller
            caller_coldkey: str = get_coldkey_for_hotkey(self, caller_hotkey)

            priority: float = 0.0

            if (
                caller_coldkey in self.coldkey_whitelist
                or caller_hotkey in self.hotkey_whitelist
            ):
                priority = 25000.0
                logger.info(
                    f"Setting the priority of whitelisted key {caller_hotkey} to {priority}"
                )

            try:
                caller_uid: int = self.metagraph.hotkeys.index(synapse.axon.hotkey)
                priority = max(priority, float(self.metagraph.S[caller_uid]))
                logger.info(
                    f"Prioritizing key {synapse.axon.hotkey} with value: {priority}."
                )
            except ValueError:
                logger.warning(f"Hotkey {synapse.axon.hotkey} not found in metagraph")

            return priority
        except Exception as e:
            logger.error(f"Error in _base_priority: {e}")
            return 0.0

    def _base_blacklist(
        self,
        synapse: Union[IsAlive, ImageGeneration],
        vpermit_tao_limit: float = VPERMIT_TAO,
        rate_limit: float = 1.0,
    ) -> Tuple[bool, str]:
        try:
            # Get the name of the synapse
            synapse_type: str = type(synapse).__name__

            # Caller hotkey
            caller_hotkey: str = synapse.axon.hotkey

            # Retrieve the coldkey of the caller
            caller_coldkey: str = get_coldkey_for_hotkey(self, caller_hotkey)

            # Retrieve the stake of the caller
            caller_stake: Optional[float] = get_caller_stake(self, synapse)

            # Count the request frequencies
            exceeded_rate_limit: bool = False
            if synapse_type == "ImageGeneration":
                # Apply a rate limit from the same caller
                if caller_hotkey in self.request_dict:
                    now: float = time.perf_counter()

                    # The difference in seconds between
                    # the current request and the previous one
                    delta: float = now - self.request_dict[caller_hotkey]["history"][-1]

                    # E.g., 0.3 < 1.0
                    if delta < rate_limit:
                        # Count number of rate limited
                        # calls from caller's hotkey
                        self.request_dict[caller_hotkey]["rate_limited_count"] += 1
                        exceeded_rate_limit = True

                    # Store the data
                    self.request_dict[caller_hotkey]["history"].append(now)
                    self.request_dict[caller_hotkey]["delta"].append(delta)
                    self.request_dict[caller_hotkey]["count"] += 1

                else:
                    # For the first request, initialize the dictionary
                    self.request_dict[caller_hotkey] = {
                        "history": [time.perf_counter()],
                        "delta": [0.0],
                        "count": 0,
                        "rate_limited_count": 0,
                    }

            # Allow through any whitelisted keys unconditionally
            # Note that blocking these keys
            # will result in a ban from the network
            if caller_coldkey in self.coldkey_whitelist:
                colored_log(
                    f"Whitelisting coldkey's {synapse_type} request from {caller_hotkey}.",
                    color="green",
                )
                return False, "Whitelisted coldkey recognized."

            if caller_hotkey in self.hotkey_whitelist:
                colored_log(
                    f"Whitelisting hotkey's {synapse_type} request from {caller_hotkey}.",
                    color="green",
                )
                return False, "Whitelisted hotkey recognized."

            # Reject request if rate limit was exceeded
            # and key wasn't whitelisted
            if exceeded_rate_limit:
                colored_log(
                    f"Blacklisted a {synapse_type} request from {caller_hotkey}. "
                    f"Rate limit ({rate_limit:.2f}) exceeded. Delta: {delta:.2f}s.",
                    color="red",
                )
                return (
                    True,
                    f"Blacklisted a {synapse_type} request from {caller_hotkey}. "
                    f"Rate limit ({rate_limit:.2f}) exceeded. Delta: {delta:.2f}s.",
                )

            # Blacklist requests from validators that aren't registered
            if caller_stake is None:
                colored_log(
                    f"Blacklisted a non-registered hotkey's {synapse_type} "
                    f"request from {caller_hotkey}.",
                    color="red",
                )
                return (
                    True,
                    f"Blacklisted a non-registered hotkey's {synapse_type} "
                    f"request from {caller_hotkey}.",
                )

            # Check that the caller has sufficient stake
            if caller_stake < vpermit_tao_limit:
                return (
                    True,
                    f"Blacklisted a {synapse_type} request from {caller_hotkey} "
                    f"due to low stake: {caller_stake:.2f} < {vpermit_tao_limit}",
                )

            logger.info(f"Allowing recognized hotkey {caller_hotkey}")
            return False, "Hotkey recognized"

        except Exception as e:
            logger.error(f"Error in blacklist: {traceback.format_exc()}")
            return True, f"Error in blacklist: {str(e)}"

    def blacklist_is_alive(self, synapse: IsAlive) -> Tuple[bool, str]:
        return self._base_blacklist(synapse)

    def blacklist_image_generation(self, synapse: ImageGeneration) -> Tuple[bool, str]:
        return self._base_blacklist(synapse)

    def priority_is_alive(self, synapse: IsAlive) -> float:
        return self._base_priority(synapse)

    def priority_image_generation(self, synapse: ImageGeneration) -> float:
        return self._base_priority(synapse)

    def loop(self) -> None:
        colored_log("Starting miner loop.", color="green")
        step: int = 0
        while True:
            try:
                # Check the miner is still registered
                is_registered: bool = self.check_still_registered()

                if not is_registered:
                    colored_log("The miner is not currently registered.", color="red")
                    time.sleep(120)

                    # Ensure the metagraph is synced
                    # before the next registration check
                    self.metagraph.sync(subtensor=self.subtensor)
                    continue

                # Output current statistics and set weights
                if step % 5 == 0:
                    # Output metrics
                    log: str = (
                        f"Step: {step} | "
                        f"Block: {self.metagraph.block.item()} | "
                        f"Stake: {self.metagraph.S[self.miner_index]:.2f} | "
                        f"Rank: {self.metagraph.R[self.miner_index]:.2f} | "
                        f"Trust: {self.metagraph.T[self.miner_index]:.2f} | "
                        f"Consensus: {self.metagraph.C[self.miner_index]:.2f} | "
                        f"Incentive: {self.metagraph.I[self.miner_index]:.2f} | "
                        f"Emission: {self.metagraph.E[self.miner_index]:.2f}"
                    )
                    colored_log(log, color="green")

                    # Show the top 10 requestors by calls along
                    # with their delta Hotkey, count, delta, rate limited count
                    top_requestors: List[Tuple[str, int, List[float], int]] = [
                        (k, v["count"], v["delta"], v["rate_limited_count"])
                        for k, v in self.request_dict.items()
                    ]

                    # Retrieve total number of requests
                    total_requests_counted: int = sum([x[1] for x in top_requestors])

                    try:
                        # Sort by count
                        top_requestors = sorted(
                            top_requestors, key=lambda x: x[1], reverse=True
                        )[:10]

                        if len(top_requestors) > 0:
                            formatted_str: str = "\n".join(
                                [
                                    f"Hotkey: {x[0]}, "
                                    f"Count: {x[1]} ({((x[1] / total_requests_counted)*100) if total_requests_counted > 0 else 0:.2f}%), "
                                    f"Average delta: {sum(x[2]) / len(x[2]) if len(x[2]) > 0 else 0:.2f}, "
                                    f"Rate limited count: {x[3]}"
                                    for x in top_requestors
                                ]
                            )
                            formatted_str = f"{formatted_str}"

                            colored_log(
                                f"{sh('Top Callers')} -> Metrics\n{formatted_str}",
                                color="cyan",
                            )
                    except Exception as e:
                        logger.error(f"Error processing top requestors: {e}")

                step += 1
                time.sleep(60)

            # If someone intentionally stops the miner,
            # it'll safely terminate operations.
            except KeyboardInterrupt:
                self.axon.stop()
                logger.success("Miner killed by keyboard interrupt.")
                sys.exit(0)

            # In case of unforeseen errors,
            # the miner will log the error and continue operations.
            except Exception:
                logger.error(f"Unexpected error: {traceback.format_exc()}")
                continue
