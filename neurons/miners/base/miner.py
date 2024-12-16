import sys
import time
import traceback
from abc import ABC, abstractmethod
from multiprocessing import Event, Manager
from typing import Optional, Tuple

import torch
from loguru import logger
from neurons.common.saas.utils import saas_show_dashboard_url
from neurons.config import get_config, get_metagraph, get_subtensor, get_wallet
from neurons.config.lists import get_blacklist
from neurons.constants import VPERMIT_TAO
from neurons.miners.base.models import MinerState, RequestStats
from neurons.update_checker import safely_check_for_updates
from neurons.utils import (
    BackgroundTimer,
    background_loop,
    get_coldkey_for_hotkey,
    get_stake_for_hotkey,
)
from neurons.utils.log import sh

import bittensor as bt
from bittensor.utils.networking import get_external_ip


class BaseMiner(ABC):
    """
    Abstract base class for miners. Handles core network functionality and state management.
    Individual miners should inherit from this and implement specific synapse handling.
    """

    state: MinerState

    def __init__(self, **kwargs) -> None:
        # Core state management
        self.should_quit: Event = Manager().Event()

        if not self.state:
            self.state = MinerState()

        # Initialize logging
        if get_config().logging.debug:
            bt.debug()
            logger.info("Enabling debug mode...")

        # Setup and display dashboard
        saas_show_dashboard_url()

        # Initialize core components
        self.initialize_components()

        # Initialize implementation specific components
        self.initialize_implementation()

        # Start axon server
        self.start()

        # Start main loop
        self.loop()

    def initialize_components(self) -> None:
        """Initialize core miner components"""
        logger.info("Beginning core component initialization...")
        self.initialize_subtensor()
        self.initialize_wallet()
        self.initialize_metagraph()
        self.loop_until_registered()
        self.start_background_loop()

    @abstractmethod
    def initialize_implementation(self) -> None:
        """Initialize implementation specific components"""

    @abstractmethod
    def create_attachments(
        self,
    ) -> None:
        """Create all attachments to synapse callbacks"""

    def initialize_subtensor(self) -> None:
        get_subtensor()
        logger.info("Initializing subtensor connection...")

    def initialize_metagraph(self) -> None:
        get_metagraph()
        logger.info("Initializing and syncing metagraph...")

    def initialize_wallet(self) -> None:
        get_wallet()
        logger.info("Initializing wallet connection...")

    def start_background_loop(self) -> None:
        """Start background monitoring loop"""
        logger.info("Starting background monitoring loop...")
        self.background_timer: BackgroundTimer = BackgroundTimer(
            300,
            background_loop,
            [self.should_quit],
        )
        self.background_timer.daemon = True
        self.background_timer.start()

    def start(self) -> None:
        """Start the axon server with implementation specific forward functions"""
        logger.info(f"Serving axon on port {get_config().axon.port}.")
        self.create_axon()
        self.register_axon()

    def create_axon(self) -> None:
        try:
            logger.info("Creating axon server configuration...")
            axon_config: Optional[bt.AxonInfo] = get_config().axon

            assert axon_config is not None, "Axon configuration was not defined"

            self.axon = bt.axon(
                wallet=get_wallet(),
                ip=get_external_ip(),
                external_ip=axon_config.get("external_ip") or get_external_ip(),
                config=get_config(),
            )

            self.create_attachments()
            self.axon.start()
            logger.info(f"Axon created: {self.axon}")
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
        """Wait until miner is registered on the network"""
        while True:
            try:
                if self.is_miner_registered():
                    break
                self.handle_unregistered_miner()
            except Exception as e:
                logger.error(f"Error in loop_until_registered: {e}")
                time.sleep(120)

    def is_miner_registered(self) -> bool:
        """Check if miner is registered on network"""
        logger.info("Checking miner registration status...")
        self.state.metrics.miner_index = self.get_miner_index()
        if self.state.metrics.miner_index is not None:
            hotkey: str = get_wallet().hotkey.ss58_address
            logger.info(
                f"Miner {hotkey} registered with uid "
                f"{get_metagraph().uids[self.state.metrics.miner_index]}"
            )
            return True
        return False

    def handle_unregistered_miner(self) -> None:
        """Handle unregistered miner state"""
        hotkey: str = get_wallet().hotkey.ss58_address

        logger.warning(
            f"Miner {hotkey} not registered. " "Sleeping for 120 seconds..."
        )
        time.sleep(120)
        get_metagraph().sync(subtensor=get_subtensor())

    def get_miner_index(self) -> Optional[int]:
        """Get miner's index in the network"""
        try:
            return get_metagraph().hotkeys.index(
                get_wallet().hotkey.ss58_address
            )
        except ValueError:
            return None

    def check_still_registered(self) -> bool:
        """Check if miner is still registered"""
        return self.get_miner_index() is not None

    async def _base_blacklist(self, synapse: bt.Synapse) -> Tuple[bool, str]:
        """Base blacklist implementation that can be used by child classes"""
        logger.debug(f"Running blacklist checks for synapse type {type(synapse).__name__}...")
        vpermit_tao_limit: float = VPERMIT_TAO
        rate_limit: float = 1.0

        try:
            synapse_type: str = type(synapse).__name__
            caller_hotkey: str = synapse.dendrite.hotkey

            # Check rate limiting
            if self._check_rate_limit(synapse_type, caller_hotkey, rate_limit):
                return True, f"Rate limit ({rate_limit:.2f}) exceeded."

            # Get current blacklists
            hotkey_blacklist, coldkey_blacklist = await get_blacklist()

            # Check hotkey blacklist
            if caller_hotkey in hotkey_blacklist:
                return True, "Hotkey is blacklisted"

            # Check coldkey blacklist
            coldkey = get_coldkey_for_hotkey(caller_hotkey)
            if coldkey in coldkey_blacklist:
                return True, "Coldkey is blacklisted"

            # Check registration and stake
            caller_stake = get_stake_for_hotkey(caller_hotkey)
            if caller_stake is None:
                return True, "Non-registered hotkey"

            if caller_stake < vpermit_tao_limit:
                return (
                    True,
                    f"Low stake: {caller_stake:.2f} < {vpermit_tao_limit}",
                )

            logger.info(f"Allowing recognized hotkey {caller_hotkey}")
            return False, "Hotkey recognized"

        except Exception as e:
            logger.error(f"Error in blacklist: {traceback.format_exc()}")
            return True, f"Error in blacklist: {str(e)}"

    def _check_rate_limit(
        self, synapse_type: str, caller_hotkey: str, rate_limit: float
    ) -> bool:
        """Check if caller has exceeded rate limit"""
        logger.debug(f"Checking rate limits for hotkey {caller_hotkey} on {synapse_type}...")
        if synapse_type not in ["IsAlive"]:
            if caller_hotkey in self.state.request_stats:
                now = time.perf_counter()
                last_request = self.state.request_stats[
                    caller_hotkey
                ].last_request_time
                delta = now - last_request

                if delta < rate_limit:
                    self.state.request_stats[
                        caller_hotkey
                    ].rate_limited_count += 1
                    return True

                self.state.request_stats[caller_hotkey].history.append(now)
                self.state.request_stats[caller_hotkey].delta.append(delta)
                self.state.request_stats[caller_hotkey].count += 1
                self.state.request_stats[caller_hotkey].last_request_time = now

            else:
                self.state.request_stats[caller_hotkey] = RequestStats()
        return False

    def update_check(self) -> None:
        """Check for miner updates"""
        if self.state.metrics.step % 4 != 0:
            return

        if safely_check_for_updates():
            if get_config().alchemy.auto_update:
                logger.info("Update detected, initiating shutdown...")
                self.should_quit.set()
            else:
                logger.warning(
                    "New version available but auto-update disabled. "
                    "Please update manually."
                )

    def loop(self) -> None:
        """Main miner loop"""
        logger.info("Starting miner loop.")
        step = 0

        while not self.should_quit.is_set():
            try:
                # Check for updates
                step += 1
                logger.debug(f"Main loop step {step}")
                self.update_check()

                # Check registration
                is_registered: bool = self.check_still_registered()
                if not is_registered:
                    logger.info("Miner not registered")
                    time.sleep(120)
                    get_metagraph().sync(subtensor=get_subtensor())
                    continue

                # Output metrics every 5 steps
                if self.state.metrics.step % 5 == 0:
                    self._log_metrics()
                    self._log_top_requestors()

                self.state.metrics.step += 1
                time.sleep(60)

            except KeyboardInterrupt:
                self.axon.stop()
                logger.success("Miner killed by keyboard interrupt")
                sys.exit(0)

            except Exception:
                logger.error(f"Error in miner loop: {traceback.format_exc()}")
                continue

    def _log_metrics(self) -> None:
        """Log miner metrics"""
        metagraph = get_metagraph()
        miner_index = self.state.metrics.miner_index

        if miner_index is None:
            return

        log = (
            f"Step: {self.state.metrics.step} | "
            f"Block: {metagraph.block.item()} | "
            f"Stake: {metagraph.S[miner_index]:.2f} | "
            f"Rank: {metagraph.R[miner_index]:.2f} | "
            f"Trust: {metagraph.T[miner_index]:.2f} | "
            f"Consensus: {metagraph.C[miner_index]:.2f} | "
            f"Incentive: {metagraph.I[miner_index]:.2f} | "
            f"Emission: {metagraph.E[miner_index]:.2f}"
        )
        logger.info(log, color="green")

    def _log_top_requestors(self) -> None:
        """Log top requestor statistics"""
        try:
            top_requestors = [
                (k, v.count, v.delta, v.rate_limited_count)
                for k, v in self.state.request_stats.items()
            ]

            total_requests = sum([x[1] for x in top_requestors])

            top_requestors = sorted(
                top_requestors, key=lambda x: x[1], reverse=True
            )[:10]

            if len(top_requestors) > 0:
                formatted_str = "\n".join(
                    [
                        f"Hotkey: {x[0]}, "
                        f"Count: {x[1]} ({((x[1] / total_requests)*100) if total_requests > 0 else 0:.2f}%), "
                        f"Average delta: {sum(x[2]) / len(x[2]) if len(x[2]) > 0 else 0:.2f}, "
                        f"Rate limited count: {x[3]}"
                        for x in top_requestors
                    ]
                )
                logger.info(f"{sh('Top Callers')} -> Metrics\n{formatted_str}")
        except Exception as e:
            logger.error(f"Error processing top requestors: {e}")

    def log_gpu_memory_usage(self, stage: str) -> None:
        """Log GPU memory usage stats"""
        try:
            allocated = torch.cuda.memory_allocated() / 1024**2
            max_allocated = torch.cuda.max_memory_allocated() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            free = total - allocated

            logger.info(f"GPU memory allocated {stage}: {allocated:.2f} MB")
            logger.info(
                f"Max GPU memory allocated {stage}: {max_allocated:.2f} MB"
            )
            logger.info(f"Total GPU memory: {total:.2f} MB")
            logger.info(f"Free GPU memory: {free:.2f} MB")
        except Exception as e:
            logger.error(f"Failed to log GPU memory usage {stage}: {str(e)}")
