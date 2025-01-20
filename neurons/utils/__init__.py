import asyncio
import inspect
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from multiprocessing import Event
from threading import Timer
from typing import Any, Callable, Dict, List, Optional

import bittensor as bt
from loguru import logger

from neurons.config.clients import get_metagraph, get_subtensor, get_wallet
from neurons.config.lists import get_warninglist
from neurons.utils.log import configure_logging


class TaskWrapper:
    """Wraps task execution with error handling and recovery logic"""

    def __init__(self, function: Callable, timeout: int = 300):
        self.function = function
        self.timeout = timeout
        self.name = function.__name__
        self._error_count = 0
        self._last_success = time.time()
        self.MAX_ERRORS = 3
        self.ERROR_RESET_TIME = 300  # 5 minutes

    async def execute(self, *args, **kwargs) -> Any:
        try:
            if inspect.iscoroutinefunction(self.function):
                result = await self._run_async(*args, **kwargs)
            else:
                result = await self._run_sync(*args, **kwargs)

            self._handle_success()
            return result

        except Exception as e:
            self._handle_error(e)
            return None

    async def _run_async(self, *args, **kwargs) -> Any:
        try:
            return await asyncio.wait_for(
                self.function(*args, **kwargs), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Task {self.name} timed out after {self.timeout}s"
            )

    async def _run_sync(self, *args, **kwargs) -> Any:
        def wrapper():
            return self.function(*args, **kwargs)

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                return await loop.run_in_executor(executor, wrapper)
            except Exception as e:
                raise type(e)(f"Task {self.name} failed: {str(e)}")

    def _handle_success(self):
        self._error_count = 0
        self._last_success = time.time()

    def _handle_error(self, error: Exception):
        current_time = time.time()
        if current_time - self._last_success > self.ERROR_RESET_TIME:
            self._error_count = 0

        self._error_count += 1

        if self._error_count >= self.MAX_ERRORS:
            logger.error(
                f"Task {self.name} failed {self.MAX_ERRORS} times in succession. Latest error: {error}"
            )
        else:
            logger.warning(f"Task {self.name} failed: {error}")


class BackgroundTimer(Timer):
    """Thread-based timer with improved error handling"""

    def __init__(
        self,
        interval: float,
        function: Callable,
        args: Optional[List] = None,
        kwargs: Optional[Dict] = None,
        timeout: int = 300,
    ):
        super().__init__(interval, function, args or [], kwargs or {})
        self.wrapper = TaskWrapper(function, timeout)
        self._health_check = True
        self.daemon = (
            True  # Allow the program to exit even if the timer is running
        )

    def __str__(self) -> str:
        return self.wrapper.name

    def run(self):
        """Main thread loop with error handling"""
        configure_logging()
        logger.info(f"[thread] {self} started")

        # Initial run
        self._run_cycle()

        # Subsequent runs
        while not self.finished.wait(self.interval):
            self._run_cycle()

    def _run_cycle(self):
        """Single execution cycle with error handling"""
        # Create and run a new event loop for this thread
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)

            loop.run_until_complete(
                self.wrapper.execute(*self.args, **self.kwargs)
            )
            self._health_check = True

        except Exception as e:
            logger.error(f"Error in {self}: {str(e)}")
            logger.debug(traceback.format_exc())
            self._health_check = False
        finally:
            loop.close()

    def is_healthy(self) -> bool:
        """Check if the thread is healthy"""
        return self._health_check


class MultiprocessTimer(multiprocessing.Process):
    """Process-based timer with improved error handling and recovery"""

    def __init__(
        self,
        interval: float,
        function: Callable,
        args: Optional[List] = None,
        kwargs: Optional[Dict] = None,
        timeout: int = 300,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        super().__init__()
        self.interval = interval
        self.wrapper = TaskWrapper(function, timeout)
        self.args = args or []
        self.kwargs = kwargs or {}
        self.finished = multiprocessing.Event()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._health_check = multiprocessing.Value("i", 1)

    def __str__(self) -> str:
        return self.wrapper.name

    def run(self):
        """Main process loop with error handling and health monitoring"""
        self._configure_process()
        logger.info(f"[process] {self} started")

        while not self.finished.is_set():
            try:
                asyncio.run(self._run_cycle())
            except Exception as e:
                self._handle_fatal_error(e)

    async def _run_cycle(self):
        """Single execution cycle with timeout and error handling"""
        try:
            await self.wrapper.execute(*self.args, **self.kwargs)
            self._health_check.value = 1

            # Wait for the next interval or until cancelled
            await asyncio.sleep(self.interval)

        except (EOFError, BrokenPipeError, ConnectionError) as e:
            logger.warning(f"Communication error in {self}: {str(e)}")
            await self._handle_communication_error()

        except Exception as e:
            logger.error(f"Unexpected error in {self}: {str(e)}")
            logger.debug(traceback.format_exc())
            await asyncio.sleep(self.retry_delay)

    async def _handle_communication_error(self):
        """Handle multiprocessing communication errors"""
        self._health_check.value = 0
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                # Attempt to re-establish communication
                if self.finished.is_set():
                    return

                await asyncio.sleep(self.retry_delay * (retry_count + 1))
                retry_count += 1

            except Exception as e:
                logger.error(f"Failed to recover {self}: {str(e)}")
                if retry_count >= self.max_retries - 1:
                    raise

    def _handle_fatal_error(self, error: Exception):
        """Handle errors that require process termination"""
        logger.critical(f"Fatal error in {self}: {str(error)}")
        logger.debug(traceback.format_exc())
        self.cancel()

    def _configure_process(self):
        """Configure process-specific settings"""
        try:
            multiprocessing.current_process().name = f"BGTimer-{self}"
            configure_logging()
        except Exception as e:
            logger.error(f"Failed to configure {self}: {str(e)}")

    def cancel(self):
        """Safely cancel the background timer"""
        logger.info(f"[process] cancelling {self}")
        self.finished.set()

    def is_healthy(self) -> bool:
        """Check if the process is healthy"""
        return bool(self._health_check.value)


def get_coldkey_for_hotkey(hotkey: str) -> Optional[str]:
    """
    Look up the coldkey of the caller.
    """
    metagraph: bt.metagraph = get_metagraph()

    if hotkey in metagraph.hotkeys:
        index = metagraph.hotkeys.index(hotkey)
        return metagraph.coldkeys[index]
    return None


def get_stake_for_hotkey(hotkey: str) -> float:
    """
    Look up the coldkey of the caller.
    """
    metagraph: bt.metagraph = get_metagraph()

    if hotkey in metagraph.hotkeys:
        index = metagraph.hotkeys.index(hotkey)
        return metagraph.S[index]

    return 0.0


background_steps: int = 0


def background_loop(should_quit: Event) -> None:
    """
    Handles terminating the miner after deregistration and
    updating the blacklist and whitelist.
    """
    from neurons.constants import IS_CI_ENV

    if IS_CI_ENV:
        return

    global background_steps
    background_steps += 1

    # Terminate the miner / validator after deregistration
    if background_steps % 5 != 0:
        return

    my_hotkey: str = get_wallet().hotkey.ss58_address
    hotkeys, _coldkeys = asyncio.run(get_warninglist())

    try:
        if my_hotkey in hotkeys.keys():
            hotkey_warning: str = hotkeys[my_hotkey][1]

            logger.info(
                f"This hotkey is on the warning list: {my_hotkey}"
                + f" | Date for rectification: {hotkey_warning}",
            )

        get_metagraph().sync(subtensor=get_subtensor())
        if get_wallet().hotkey.ss58_address not in get_metagraph().hotkeys:
            logger.info(">>> Axon has deregistered... terminating.")
            try:
                should_quit.set()
            except Exception as e:
                logger.info(
                    f"An error occurred trying to terminate the main thread: {e}."
                )

            sys.exit(0)

    except Exception as e:
        logger.error(
            f">>> An unexpected error occurred syncing the metagraph: {e}"
        )


def normalize_weights(weights):
    sum_weights = float(sum(weights))
    normalizer = 1 / sum_weights
    weights = [weight * normalizer for weight in weights]
    if sum(weights) < 1:
        diff = 1 - sum(weights)
        weights[0] += diff

    return weights
