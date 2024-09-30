import sys
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from multiprocessing import Manager, Event

from loguru import logger

from neurons.common.saas.utils import saas_show_dashboard_url
from neurons.constants import VPERMIT_TAO
from neurons.protocol import ImageGeneration, IsAlive, ModelType

from neurons.config import get_config, get_wallet, get_metagraph, get_subtensor
from neurons.utils import BackgroundTimer, background_loop
from neurons.utils.defaults import Stats, get_defaults
from neurons.utils.log import sh
from neurons.miners.StableMiner.utils import (
    get_caller_stake,
    get_coldkey_for_hotkey,
)

import bittensor as bt


class BaseMiner(ABC):
    def __init__(self) -> None:
        # Start the batch streaming background loop
        manager = Manager()
        self.should_quit: Event = manager.Event()

        if get_config().logging.debug:
            bt.debug()
            logger.info("Enabling debug mode...")

        self.hotkey_blacklist: set = set()
        self.coldkey_blacklist: set = set()
        self.coldkey_whitelist: set = set(
            ["5F1FFTkJYyceVGE4DCVN5SxfEQQGJNJQ9CVFVZ3KpihXLxYo"]
        )
        self.hotkey_whitelist: set = set(
            ["5C5PXHeYLV5fAx31HkosfCkv8ark3QjbABbjEusiD3HXH2Ta"]
        )

        saas_show_dashboard_url()

        self.initialize_components()
        self.request_dict: Dict[str, Dict[str, Union[List[float], int]]] = {}

    def initialize_components(self) -> None:
        self.initialize_event_dict()
        self.initialize_subtensor_connection()
        self.initialize_metagraph()
        self.initialize_wallet()
        self.loop_until_registered()
        self.initialize_defaults()
        self.start_background_loop()

    def initialize_event_dict(self) -> None:
        self.event: Dict[str, Any] = {}
        self.mapping: Dict[str, Dict] = {}

    def initialize_subtensor_connection(self) -> None:
        get_subtensor()

    def initialize_metagraph(self) -> None:
        get_metagraph()

    def initialize_wallet(self) -> None:
        get_wallet()

    def initialize_defaults(self) -> None:
        self.stats: Stats = get_defaults()

    def start_background_loop(self) -> None:
        self.background_steps: int = 1
        self.background_timer: BackgroundTimer = BackgroundTimer(
            300,
            background_loop,
            [self.should_quit],
        )
        self.background_timer.daemon = True
        self.background_timer.start()

    def start_axon(self) -> None:
        logger.info(f"Serving axon on port {get_config().axon.port}.")
        self.create_axon()
        self.register_axon()

    def create_axon(self) -> None:
        try:
            self.axon: bt.axon = (
                bt.axon(
                    wallet=get_wallet(),
                    ip=bt.utils.networking.get_external_ip(),
                    external_ip=get_config().axon.get("external_ip")
                    or bt.utils.networking.get_external_ip(),
                    config=get_config(),
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
            logger.info(f"Axon created: {self.axon}", color="green")
        except Exception as e:
            logger.error(f"Failed to create axon: {e}")
            raise

    def register_axon(self) -> None:
        try:
            get_subtensor().serve_axon(
                axon=self.axon, netuid=get_config().netuid
            )
        except Exception as e:
            logger.error(f"Failed to register axon: {e}")
            raise

    def loop_until_registered(self) -> None:
        while True:
            try:
                if self.is_miner_registered():
                    break
                self.handle_unregistered_miner()
            except Exception as e:
                logger.error(f"Error in loop_until_registered: {e}")
                time.sleep(120)

    def is_miner_registered(self) -> bool:
        self.miner_index = self.get_miner_index()
        if self.miner_index is not None:
            logger.info(
                f"Miner {get_wallet().hotkey} is registered with uid "
                f"{get_metagraph().uids[self.miner_index]}"
            )
            return True
        return False

    def handle_unregistered_miner(self) -> None:
        logger.warning(
            f"Miner {get_wallet().hotkey} is not registered. "
            "Sleeping for 120 seconds..."
        )
        time.sleep(120)
        get_metagraph().sync(subtensor=get_subtensor())

    def get_miner_info(self) -> Dict[str, Union[int, float]]:
        metagraph: bt.metagraph = get_metagraph()

        try:
            return {
                "block": metagraph.block.item(),
                "stake": metagraph.stake[self.miner_index].item(),
                "trust": metagraph.trust[self.miner_index].item(),
                "consensus": metagraph.consensus[self.miner_index].item(),
                "incentive": metagraph.incentive[self.miner_index].item(),
                "emissions": metagraph.emission[self.miner_index].item(),
            }
        except Exception as e:
            logger.error(f"Error in get_miner_info: {e}")
            return {}

    def get_miner_index(self) -> Optional[int]:
        try:
            return get_metagraph().hotkeys.index(
                get_wallet().hotkey.ss58_address
            )
        except ValueError:
            return None

    def check_still_registered(self) -> bool:
        return self.get_miner_index() is not None

    def get_incentive(self) -> float:
        if self.miner_index is not None:
            return get_metagraph().I[self.miner_index].item() * 100_000
        return 0.0

    def get_trust(self) -> float:
        if self.miner_index is not None:
            return get_metagraph().T[self.miner_index].item() * 100
        return 0.0

    def get_consensus(self) -> float:
        if self.miner_index is not None:
            return get_metagraph().C[self.miner_index].item() * 100_000
        return 0.0

    async def is_alive(self, synapse: IsAlive) -> IsAlive:
        logger.info("IsAlive")
        synapse.completion = "True"
        return synapse

    @abstractmethod
    def get_model_config(self, model_type: ModelType, task_type: str) -> Any:
        pass

    async def generate_image(self, synapse: ImageGeneration) -> ImageGeneration:
        """
        Generic image generation logic
        """
        timeout: float = synapse.timeout
        self.stats.total_requests += 1
        start_time: float = time.perf_counter()

        model_type: str = synapse.model_type or ModelType.CUSTOM

        try:
            model_config = self.get_model_config(
                model_type,
                synapse.generation_type.upper(),
            )
        except ValueError as e:
            logger.error(f"Error getting model config: {e}")
            return synapse

        images = await self._attempt_generate_images(synapse, model_config)

        if len(images) == 0:
            logger.info(f"Failed to generate any images after {3} attempts.")

        if time.perf_counter() - start_time > timeout:
            self.stats.timeouts += 1

        self._log_generation_time(start_time)

        synapse.images = images
        return synapse

    @abstractmethod
    async def _attempt_generate_images(
        self, synapse: ImageGeneration, model_config: Any
    ) -> List[str]:
        pass

    def _log_generation_time(self, start_time: float) -> None:
        generation_time: float = time.perf_counter() - start_time
        self.stats.generation_time += generation_time
        average_time: float = (
            self.stats.generation_time / self.stats.total_requests
        )
        logger.info(
            f"{sh('Time')} -> {generation_time:.2f}s | Average: {average_time:.2f}s",
        )

    def is_whitelisted(
        self, caller_hotkey: str = None, caller_coldkey: str = None
    ) -> bool:
        if caller_hotkey and self._is_in_whitelist(caller_hotkey):
            return True

        if caller_hotkey:
            caller_coldkey = get_coldkey_for_hotkey(caller_hotkey)
            if self._is_in_whitelist(caller_coldkey):
                return True

        if caller_coldkey and self._is_in_whitelist(caller_coldkey):
            return True

        return False

    def _is_in_whitelist(self, key: str) -> bool:
        return key in self.hotkey_whitelist or key in self.coldkey_whitelist

    def _base_priority(self, synapse: Union[IsAlive, ImageGeneration]) -> float:
        caller_hotkey: str = synapse.dendrite.hotkey

        try:
            priority: float = 0.0

            if self.is_whitelisted(caller_hotkey=caller_hotkey):
                priority = 25000.0
                logger.info(
                    "Setting the priority of whitelisted key"
                    + f" {caller_hotkey} to {priority}"
                )

            try:
                caller_uid: int = get_metagraph().hotkeys.index(
                    synapse.dendrite.hotkey,
                )
                priority = max(priority, float(get_metagraph().S[caller_uid]))
                logger.info(
                    f"Prioritizing key {synapse.dendrite.hotkey}"
                    + f" with value: {priority}."
                )
            except ValueError:
                logger.warning(
                    f"Hotkey {synapse.dendrite.hotkey}"
                    + f" not found in metagraph"
                )

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
            synapse_type: str = type(synapse).__name__
            caller_hotkey: str = synapse.dendrite.hotkey
            caller_coldkey: str = get_coldkey_for_hotkey(caller_hotkey)
            caller_stake: Optional[float] = get_caller_stake(synapse)

            exceeded_rate_limit: bool = False
            if synapse_type == "ImageGeneration":
                if caller_hotkey in self.request_dict:
                    now: float = time.perf_counter()
                    delta: float = (
                        now - self.request_dict[caller_hotkey]["history"][-1]
                    )

                    if delta < rate_limit:
                        self.request_dict[caller_hotkey][
                            "rate_limited_count"
                        ] += 1
                        exceeded_rate_limit = True

                    self.request_dict[caller_hotkey]["history"].append(now)
                    self.request_dict[caller_hotkey]["delta"].append(delta)
                    self.request_dict[caller_hotkey]["count"] += 1

                else:
                    self.request_dict[caller_hotkey] = {
                        "history": [time.perf_counter()],
                        "delta": [0.0],
                        "count": 0,
                        "rate_limited_count": 0,
                    }

            if self.is_whitelisted(caller_coldkey=caller_coldkey):
                logger.info(
                    f"Whitelisting coldkey's {synapse_type}"
                    + f" request from {caller_hotkey}.",
                )
                return False, "Whitelisted coldkey recognized."

            if self.is_whitelisted(caller_hotkey=caller_hotkey):
                logger.info(
                    f"Whitelisting hotkey's {synapse_type}"
                    + f" request from {caller_hotkey}.",
                )
                return False, "Whitelisted hotkey recognized."

            if exceeded_rate_limit:
                logger.info(
                    f"Blacklisted a {synapse_type} request from {caller_hotkey}. "
                    f"Rate limit ({rate_limit:.2f}) exceeded. Delta: {delta:.2f}s.",
                )
                return (
                    True,
                    f"Blacklisted a {synapse_type} request from {caller_hotkey}. "
                    f"Rate limit ({rate_limit:.2f}) exceeded. Delta: {delta:.2f}s.",
                )

            if caller_stake is None:
                logger.info(
                    f"Blacklisted a non-registered hotkey's {synapse_type} "
                    f"request from {caller_hotkey}.",
                )
                return (
                    True,
                    f"Blacklisted a non-registered hotkey's {synapse_type} "
                    f"request from {caller_hotkey}.",
                )

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

    def blacklist_image_generation(
        self, synapse: ImageGeneration
    ) -> Tuple[bool, str]:
        return self._base_blacklist(synapse)

    def priority_is_alive(self, synapse: IsAlive) -> float:
        return self._base_priority(synapse)

    def priority_image_generation(self, synapse: ImageGeneration) -> float:
        return self._base_priority(synapse)

    def loop(self) -> None:
        logger.info("Starting miner loop.", color="green")
        step: int = 0
        while not self.should_quit.is_set():
            try:
                # Check the miner is still registered
                is_registered: bool = self.check_still_registered()

                metagraph: bt.metagraph = get_metagraph()

                if not is_registered:
                    logger.info(
                        "The miner is not currently registered.", color="red"
                    )
                    time.sleep(120)

                    # Ensure the metagraph is synced
                    # before the next registration check
                    metagraph.sync(subtensor=get_subtensor())
                    continue

                # Output current statistics and set weights
                if step % 5 == 0:
                    # Output metrics
                    log: str = (
                        f"Step: {step} | "
                        f"Block: {metagraph.block.item()} | "
                        f"Stake: {metagraph.S[self.miner_index]:.2f} | "
                        f"Rank: {metagraph.R[self.miner_index]:.2f} | "
                        f"Trust: {metagraph.T[self.miner_index]:.2f} | "
                        f"Consensus: {metagraph.C[self.miner_index]:.2f} | "
                        f"Incentive: {metagraph.I[self.miner_index]:.2f} | "
                        f"Emission: {metagraph.E[self.miner_index]:.2f}"
                    )
                    logger.info(log, color="green")

                    # Show the top 10 requestors by calls along
                    # with their delta Hotkey, count, delta, rate limited count
                    top_requestors: List[Tuple[str, int, List[float], int]] = [
                        (k, v["count"], v["delta"], v["rate_limited_count"])
                        for k, v in self.request_dict.items()
                    ]

                    # Retrieve total number of requests
                    total_requests_counted: int = sum(
                        [x[1] for x in top_requestors]
                    )

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

                            logger.info(
                                f"{sh('Top Callers')} -> Metrics\n{formatted_str}",
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
