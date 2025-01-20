import asyncio
import copy
import inspect
import os
import queue
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta
from math import ceil
from threading import Event
from multiprocessing import Manager, Process, Queue, set_start_method
from threading import Thread
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from neurons.common.saas.utils import saas_show_dashboard_url
from neurons.config import (
    get_backend_client,
    get_config,
    get_device,
    get_metagraph,
    get_subtensor,
    get_wallet,
    validator_run_id,
)
from neurons.exceptions import StakeBelowThreshold
from neurons.protocol import (
    ImageGenerationTaskModel,
    ModelType,
    denormalize_image_model,
)
from neurons.update_checker import safely_check_for_updates
from neurons.utils import BackgroundTimer, MultiprocessTimer, background_loop
from neurons.utils.common import log_dependencies
from neurons.utils.defaults import get_defaults
from neurons.utils.log import configure_logging
from neurons.validator.backend.client import TensorAlchemyBackendClient
from neurons.validator.backend.models import TaskState
from neurons.validator.config import update_validator_settings
from neurons.validator.forward import run_step
from neurons.validator.schemas import Batch, ScoresUploadRequest
from neurons.validator.utils import (
    generate_random_prompt_gpt,
    is_hotkey_registered,
    select_uids,
    ttl_get_block,
)
from neurons.validator.utils.openai import check_prompt_for_nsfw
from neurons.validator.utils.state import load_ma_scores, save_ma_scores
from neurons.validator.utils.version import get_validator_version
from neurons.validator.weights import (
    SetWeightsTask,
    set_weights_loop,
    tensor_to_list,
)

import bittensor as bt
from bittensor.utils.networking import get_external_ip

# Set the start method for multiprocessing
set_start_method("spawn", force=True)

# Define a type alias for our thread-like objects
ThreadLike = Union[Thread, Process]

upload_images_loop_suspension_end_time = None


def is_valid_current_directory() -> bool:
    # NOTE: We use Alchemy for support
    #       of the old repository name ImageAlchemy
    #       otherwise normally this would be TensorAlchemy
    if "Alchemy" in os.getcwd():
        return True

    return False


async def upload_image(
    backend_client: TensorAlchemyBackendClient,
    batches_upload_queue: Queue,
) -> None:
    queue_size: int = batches_upload_queue.qsize()
    if queue_size > 0:
        logger.info(f"{queue_size} batches in queue")

    batch: Batch = batches_upload_queue.get(block=False)
    logger.info(
        #
        f"uploading ({len(batch.computes)}) computes "
        + f"for batch {batch.batch_id} ..."
    )
    await backend_client.post_batch(batch)


async def upload_images_loop(
    _should_quit: Event,
    batches_upload_queue: Queue,
) -> None:
    global upload_images_loop_suspension_end_time
    if (
        upload_images_loop_suspension_end_time
        and datetime.now() < upload_images_loop_suspension_end_time
    ):
        logger.info(
            f"Skipping uploads until {upload_images_loop_suspension_end_time}"
        )
        return

    # Send new batches to the Human Validation Bot
    try:
        backend_client: TensorAlchemyBackendClient = get_backend_client()
        await asyncio.gather(
            *[
                upload_image(backend_client, batches_upload_queue)
                for _i in range(32)
            ]
        )

    except queue.Empty:
        return
    except StakeBelowThreshold as e:
        logger.error(
            f"Exception occurred: {str(e)}. Suspending uploads for 2 hours."
        )
        upload_images_loop_suspension_end_time = datetime.now() + timedelta(
            hours=2
        )
    except Exception as e:
        logger.info(
            "An error occurred trying to submit a batch: "
            + f"{e}\n{traceback.format_exc()}"
        )


async def upload_scores_loop(
    _should_quit: Event,
    scores_upload_queue: Queue,
) -> None:
    backend_client = get_backend_client()
    queue_size: int = scores_upload_queue.qsize()
    if queue_size > 0:
        logger.info(f"{queue_size} scores data items in queue")

    try:
        scores_upload_request: ScoresUploadRequest = scores_upload_queue.get(
            block=False
        )
    except queue.Empty:
        return

    await backend_client.upload_scores(scores_upload_request)


class StableValidator:
    def loop_until_registered(self):
        index = None
        while True:
            try:
                index = self.metagraph.hotkeys.index(
                    self.wallet.hotkey.ss58_address
                )
            except Exception:
                pass

            if index is not None:
                logger.info(
                    f"Validator {self.config.wallet.hotkey} is registered with uid: "
                    + str(self.metagraph.uids[index]),
                )
                break
            logger.warning(
                f"Validator {self.config.wallet.hotkey} is not registered. "
                + "Sleeping for 120 seconds...",
            )
            time.sleep(120)
            self.metagraph.sync(subtensor=self.subtensor)

    def __init__(self):
        # Init config
        self.config = get_config()

        bt.logging(
            config=self.config,
            debug=self.config.debug,
            trace=self.config.trace,
            logging_dir=self.config.alchemy.full_path,
        )

        configure_logging()
        log_dependencies()

        # Init device.
        self.device = get_device(torch.device(get_device()))

        self.backend_client = TensorAlchemyBackendClient()

        self.prompt_generation_failures = 0

        # Init subtensor
        self.subtensor = get_subtensor()
        logger.info(f"Loaded subtensor: {self.subtensor}")

        # Init wallet.
        self.wallet = get_wallet()
        self.wallet.create_if_non_existent()

        # Dendrite pool for querying the network during training.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        logger.info(f"Loaded dendrite pool: {self.dendrite}")

        # Init metagraph.
        self.metagraph: bt.metagraph = get_metagraph(sync=False)

        # Sync metagraph with subtensor.
        self.metagraph.sync(subtensor=self.subtensor)

        # Keep track of latest active miners
        self.active_uids: List[int] = []

        if "mock" not in self.config.wallet.name:
            # Wait until the miner is registered
            self.loop_until_registered()

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info("Loaded metagraph")

        # Convert metagraph[x] to a PyTorch tensor if it's a NumPy array
        for key in ["stake", "uids"]:
            if isinstance(getattr(self.metagraph, key), np.ndarray):
                setattr(
                    self.metagraph,
                    key,
                    torch.from_numpy(getattr(self.metagraph, key)).float(),
                )

        # Each validator gets a unique identity (UID)
        # in the network for differentiation.
        self.my_subnet_uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
        )
        validator_version = get_validator_version()
        logger.info(
            f"Running validator (version={validator_version})"
            + f" on uid: {self.my_subnet_uid}"
        )

        # Init weights
        self.weights = torch.ones_like(
            self.metagraph.uids, dtype=torch.float32
        ).to(self.device)

        # Init prev_block and step
        self.prev_block = ttl_get_block()
        self.step = 0

        # Init sync with the network. Updates the metagraph.
        self.resync_metagraph()

        # Now load the moving average scores
        # or initialize them from the current metagraph incentives
        self.moving_average_scores = load_ma_scores()

        # Serve axon to enable external connections.
        self.serve_axon()

        # Init the event loop
        self.loop = asyncio.get_event_loop()

        # Init sync with the network. Updates the metagraph.
        asyncio.run(self.sync())

        # Init stats
        self.stats = get_defaults()

        # Get vali index
        self.validator_index = self.get_validator_index()

        # Start the generic background loop
        self.storage_client = None
        self.background_steps = 1

        # Start the batch streaming background loop
        manager = Manager()

        self.should_quit: Event = manager.Event()
        self.set_weights_queue: Queue = manager.Queue(maxsize=128)
        self.batches_upload_queue: Queue = manager.Queue(maxsize=2048)
        self.scores_upload_queue: Queue = manager.Queue(maxsize=2048)

        self.model_type = ModelType.CUSTOM

        self.background_loop: BackgroundTimer = None
        self.set_weights_process: MultiprocessTimer = None
        self.upload_images_process: MultiprocessTimer = None
        self.upload_scores_process: MultiprocessTimer = None

        saas_show_dashboard_url()

        # Start all background threads
        self.start_threads(True)

    def start_thread(self, thread: ThreadLike, is_startup: bool = True) -> None:
        if thread.is_alive():
            return

        thread.start()
        if is_startup:
            logger.info(f"Started {thread}")
        else:
            logger.error(f"{thread} had segfault, restarted")

    def graceful_shutdown(self) -> None:
        """Gracefully shutdown background processes and logging"""
        logger.info("Initiating graceful shutdown...")

        # Signal threads to stop
        self.should_quit.set()

        # Remove all loguru handlers except stderr
        logger.remove()
        logger.add(sys.stderr, level="ERROR")

        # Stop processes with timeout
        processes = [
            self.background_loop,
            self.upload_images_process,
            self.upload_scores_process,
            self.set_weights_process,
        ]

        for process in processes:
            if process and process.is_alive():
                try:
                    process.cancel()
                    process.join(timeout=5)
                except:
                    pass

        logger.info("Shutdown complete")

    def update_check(self) -> None:
        if self.step % 4 != 0:
            return

        if safely_check_for_updates():
            if self.config.alchemy.auto_update:
                logger.info("Update detected, initiating shutdown...")
                self.should_quit.set()
            else:
                logger.warning(
                    "New version available but auto-update is disabled. "
                    "Please update manually."
                )

    def start_threads(self, is_startup: bool = False) -> None:
        logger.info(f"[start_threads] is_startup={is_startup}")
        thread_configs: List[Tuple[str, object, float, callable, list]] = [
            (
                "background_loop",
                BackgroundTimer,
                60,
                background_loop,
                [self.should_quit],
            ),
            (
                "upload_images_process",
                MultiprocessTimer,
                0.5,
                upload_images_loop,
                [self.should_quit, self.batches_upload_queue],
            ),
            (
                "upload_scores_process",
                MultiprocessTimer,
                1.0,
                upload_scores_loop,
                [self.should_quit, self.scores_upload_queue],
            ),
            (
                "set_weights_process",
                MultiprocessTimer,
                1.0,
                set_weights_loop,
                [self.should_quit, self.set_weights_queue],
            ),
        ]

        for (
            attr_name,
            thread_class,
            interval,
            target_func,
            args,
        ) in thread_configs:
            thread = getattr(self, attr_name)
            if thread and thread.is_alive():
                continue

            new_thread = thread_class(interval, target_func, args)

            if attr_name == "background_loop":
                new_thread.daemon = True

            setattr(self, attr_name, new_thread)
            self.start_thread(new_thread, is_startup)

    async def reload_settings(self) -> None:
        # Update settings from google cloud
        await update_validator_settings()

    async def get_image_generation_task(
        self,
        timeout: int = 30,
    ) -> ImageGenerationTaskModel | None:
        """
        Fetch new image generation task from backend or generate new one
        Returns task or None if task cannot be generated
        """
        # NOTE: Will wait for around 60 seconds
        #       trying to get a task from the user
        # before going on and creating a synthetic task
        task: Optional[ImageGenerationTaskModel] = None
        try:
            task = await self.backend_client.poll_task(timeout=timeout)
        # Allow validator to just skip this step if they like
        except KeyboardInterrupt:
            pass

        # No organic task found
        if task is None:
            self.model_type = ModelType.CUSTOM
            prompt = await generate_random_prompt_gpt()
            if not prompt:
                logger.error("failed to generate prompt for synthetic task")
                return None
            # NOTE: Generate synthetic request
            return denormalize_image_model(
                id=str(uuid.uuid4()),
                image_count=1,
                task_type="TEXT_TO_IMAGE",
                guidance_scale=7.5,
                negative_prompt=None,
                prompt=prompt,
                seed=-1,
                steps=30,
                width=1024,
                height=1024,
            )

        is_bad_prompt = await check_prompt_for_nsfw(task.prompt)

        if is_bad_prompt:
            try:
                logger.warning(
                    #
                    "Prompt was marked as NSFW and rejected:"
                    + task.task_id
                )
                await self.backend_client.update_task_state(
                    task.task_id,
                    TaskState.REJECTED,
                )
            except Exception as e:
                logger.info(
                    f"Failed to post {task.task_id} to the"
                    + f" {TaskState.REJECTED.value} endpoint: {e}"
                )
            return None

        return task

    async def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            try:
                self.set_weights_queue.put_nowait(
                    SetWeightsTask(
                        epoch=ttl_get_block(),
                        hotkeys=copy.deepcopy(get_metagraph().hotkeys),
                        weights=tensor_to_list(self.moving_average_scores),
                    )
                )
            except queue.Full:
                logger.error("Cannot add weights setting task, queue is full!")

            logger.info(
                f"Added a weight setting task to the queue"
                f" (current size={self.set_weights_queue.qsize()})"
            )

            self.prev_block = ttl_get_block()

    def get_validator_index(self):
        """
        Retrieve the given miner's index in the metagraph.
        """
        index = None
        try:
            index = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address,
            )
        except ValueError:
            pass
        return index

    def get_validator_info(self):
        return {
            "block": self.metagraph.block.item(),
            "stake": self.metagraph.stake[self.validator_index],
            "rank": self.metagraph.ranks[self.validator_index],
            "vtrust": self.metagraph.validator_trust[self.validator_index],
            "dividends": self.metagraph.dividends[self.validator_index],
            "emissions": self.metagraph.emission[self.validator_index],
        }

    def resync_metagraph(self, **kwargs):
        """
        Resyncs the metagraph and updates
        the hotkeys and moving averages based on the new metagraph.

        Args:
            **kwargs: Additional keyword arguments to pass to metagraph.sync()
        """
        metagraph: bt.metagraph = get_metagraph()
        previous_hotkeys: List[str] = metagraph.hotkeys

        # Sync the metagraph
        metagraph.sync(subtensor=get_subtensor(), **kwargs)

        # Check if the metagraph axon info has changed
        if previous_hotkeys == metagraph.hotkeys:
            logger.debug(
                #
                "No changes in metagraph hotkeys, "
                + "skipping resync"
            )
            return

        logger.info(
            "Metagraph updated, re-syncing hotkeys, "
            "dendrite pool and moving averages"
        )

        # Update the size of the moving average scores
        new_moving_averages = torch.zeros(metagraph.n, device=get_device())

        # Create a mapping of old hotkeys to their scores
        old_hotkey_scores = {
            hotkey: score
            for hotkey, score in zip(
                previous_hotkeys, self.moving_average_scores
            )
        }

        # Update moving averages and handle replaced hotkeys
        for uid, new_hotkey in enumerate(metagraph.hotkeys):
            if new_hotkey in old_hotkey_scores:
                new_moving_averages[uid] = old_hotkey_scores[new_hotkey]

        # Update instance variables
        self.moving_average_scores = new_moving_averages

    def check_registered(self):
        # --- Check for registration.
        if not is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            logger.error(
                f"Wallet: {self.wallet} is not registered on netuid"
                + str(self.config.netuid)
                + ". Please register the hotkey before trying again"
            )
            sys.exit(1)

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed
        since the last checkpoint to sync.
        """
        return (
            ttl_get_block() - self.metagraph.last_update[self.uid]
        ) > self.config.alchemy.epoch_length

    def should_set_weights(self) -> bool:
        # Check if all moving_averages_scores are 0s or 1s
        ma_scores = self.moving_average_scores
        ma_scores_sum = sum(ma_scores)

        if ma_scores_sum == len(ma_scores) or ma_scores_sum == 0:
            logger.info(
                "All moving average scores are either 0s or 1s. "
                + "Not setting weights."
            )
            return False

        # Check if enough epoch blocks have elapsed since the last epoch
        current_block = ttl_get_block()
        blocks_elapsed = current_block % self.prev_block
        logger.debug(
            f"Current block: {current_block},"
            + f" Blocks elapsed: {blocks_elapsed}"
        )

        epoch_length: int = self.config.alchemy.epoch_length

        should_set = blocks_elapsed >= epoch_length

        # Calculate and log the approximate time until next weight set
        if not should_set:
            blocks_until_next_set = epoch_length - blocks_elapsed
            # Assuming an average block time of 12 seconds
            seconds_until_next_set = blocks_until_next_set * 12
            minutes_until_next_set = ceil(seconds_until_next_set / 60)
            logger.info(
                "Next weight set in approximately "
                f"{minutes_until_next_set} minutes "
                f"({blocks_until_next_set} blocks)"
            )

        logger.info(f"Should set weights: {should_set}")

        return should_set

    def serve_axon(self):
        """Serve axon to enable external connections."""

        logger.info("serving ip to chain...")
        try:
            self.axon = bt.axon(
                wallet=self.wallet,
                ip=get_external_ip(),
                external_ip=get_external_ip(),
                config=self.config,
            )

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                logger.info(
                    f"Running validator {self.axon} "
                    + f"on network: {self.config.subtensor.chain_endpoint} "
                    + f"with netuid: {self.config.netuid}"
                )
            except Exception as e:
                logger.error(f"Failed to serve Axon with exception: {e}")

        except Exception as e:
            logger.error(
                f"Failed to create Axon initialize with exception: {e}"
            )

    async def run(self):
        logger.info("Starting validator loop.")
        self.step = 0

        try:
            while not self.should_quit.is_set():
                try:
                    logger.info(
                        f"Started new validator run ({validator_run_id.get()})."
                    )

                    if await self.pre_step():
                        if await self.mid_step():
                            await self.post_step()

                    self.step += 1

                except KeyboardInterrupt:
                    logger.success(
                        "Keyboard interrupt detected. Exiting validator."
                    )
                    break
                except Exception:
                    logger.error(traceback.format_exc())
                    await asyncio.sleep(5)

        finally:
            self.graceful_shutdown()

    async def pre_step(self):
        try:
            self.task = await self.get_image_generation_task()
            if not self.task:
                logger.warning(
                    "Image generation task was not generated successfully."
                )
                return False

            return True
        except Exception:
            logger.error(traceback.format_exc())
            await asyncio.sleep(10)
            return False

    async def mid_step(self):
        try:
            selected_uids: torch.Tensor = await select_uids(count=12)
            if selected_uids.numel() == 0:
                logger.info("No active miners found, retrying in 20 seconds...")
                await asyncio.sleep(20)
                return False

            axons = [self.metagraph.axons[uid] for uid in selected_uids]

            await run_step(
                validator=self,
                task=self.task,
                axons=axons,
                uids=selected_uids,
                model_type=self.model_type,
                stats=self.stats,
            )
            return True
        except Exception:
            logger.error(f"Mid-step failed: {traceback.format_exc()}")
            return False

    async def post_step(self):
        for method in [
            self.sync,
            self.reload_settings,
            self.start_threads,
            self.update_check,
            saas_show_dashboard_url,
            lambda: save_ma_scores(self.moving_average_scores),
        ]:
            try:
                logger.info(f"Running post step: {method.__name__}")
                if inspect.iscoroutinefunction(method):
                    await method()
                else:
                    method()
            except Exception:
                logger.error(
                    f"{method.__name__} failed: " + traceback.format_exc()
                )
