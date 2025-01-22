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
from multiprocessing import Manager, Queue, set_start_method
from threading import Event
from typing import List, Optional, Sequence, Tuple

import bittensor as bt
import numpy as np
import torch
from bittensor.utils.networking import get_external_ip
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
    generate_random_prompt,
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

# Set the start method for multiprocessing
set_start_method("spawn", force=True)

# Define a type alias for our thread-like objects
ThreadLike = MultiprocessTimer | BackgroundTimer | None

upload_images_loop_suspension_end_time = None


def wait_for_registration(
    metagraph: bt.metagraph,
    wallet: bt.Wallet,
    subtensor: bt.subtensor,
) -> None:
    """Wait until wallet is registered in the metagraph"""
    while True:
        try:
            hotkey_ss58 = wallet.hotkey.ss58_address
            index = metagraph.hotkeys.index(hotkey_ss58)
            logger.info(
                f"Validator {hotkey_ss58} is registered with uid: {metagraph.uids[index]}"
            )
            return
        except ValueError:
            logger.warning(
                f"Validator {wallet.hotkey.ss58_address} is not registered. Sleeping for 120 seconds..."
            )
            time.sleep(120)
            metagraph.sync(subtensor=subtensor)


def validate_registration(netuid: int, hotkey_ss58: str) -> None:
    """Validate that a hotkey is registered, exit if not"""
    if not is_hotkey_registered(netuid=netuid, hotkey_ss58=hotkey_ss58):
        logger.error(
            f"Wallet: {hotkey_ss58} is not registered on netuid {netuid}. "
            "Please register the hotkey before trying again"
        )
        sys.exit(1)


def should_sync_metagraph(
    last_update: int,
    epoch_length: int,
) -> bool:
    """Check if enough epoch blocks have elapsed since last checkpoint"""
    return (ttl_get_block() - last_update) > epoch_length


def evaluate_weight_setting(
    moving_average_scores: torch.Tensor,
    current_block: int,
    prev_block: int,
    epoch_length: int,
) -> bool:
    """Determine if weights should be set based on scores and timing"""
    # Check if all scores are 0s or 1s
    ma_scores_sum = sum(moving_average_scores)
    if ma_scores_sum == len(moving_average_scores) or ma_scores_sum == 0:
        logger.info(
            "All moving average scores are either 0s or 1s. Not setting weights."
        )
        return False

    # Check block timing
    blocks_elapsed = current_block % prev_block
    logger.debug(
        f"Current block: {current_block}, Blocks elapsed: {blocks_elapsed}"
    )

    should_set = blocks_elapsed >= epoch_length

    if not should_set:
        blocks_until_next_set = epoch_length - blocks_elapsed
        seconds_until_next_set = blocks_until_next_set * 12
        minutes_until_next_set = ceil(seconds_until_next_set / 60)
        logger.info(
            f"Next weight set in approximately {minutes_until_next_set} minutes "
            f"({blocks_until_next_set} blocks)"
        )

    logger.info(f"Should set weights: {should_set}")
    return should_set


def serve_network_axon(
    wallet: bt.Wallet,
    config: bt.config,
    subtensor: bt.subtensor,
) -> bt.axon:
    """Create and serve axon for network connections"""
    logger.info("serving ip to chain...")
    try:
        axon = bt.axon(
            wallet=wallet,
            ip=get_external_ip(),
            external_ip=get_external_ip(),
            config=config,
        )

        try:
            subtensor.serve_axon(netuid=config.netuid, axon=axon)
            logger.info(
                f"Running validator {axon} on network: {config.subtensor.chain_endpoint} "
                f"with netuid: {config.netuid}"
            )
            return axon
        except Exception as e:
            logger.error(f"Failed to serve Axon with exception: {e}")
            raise

    except Exception as e:
        logger.error(f"Failed to create Axon initialize with exception: {e}")
        raise


def start_background_thread(
    thread_class: type,
    interval: float,
    target_func: callable,
    args: list,
    is_startup: bool = True,
    make_daemon: bool = False,
) -> ThreadLike:
    """Create and start a background thread with given parameters"""
    new_thread = thread_class(interval, target_func, args)
    if make_daemon:
        new_thread.daemon = True

    new_thread.start()
    if is_startup:
        logger.info(f"Started {new_thread}")
    else:
        logger.error(f"{thread_class.__name__} had segfault, restarted")

    return new_thread


async def handle_task_rejection(
    backend_client: TensorAlchemyBackendClient, task_id: str
) -> None:
    """Handle rejection of a task marked as NSFW"""
    try:
        logger.warning("Prompt was marked as NSFW and rejected:" + task_id)
        await backend_client.update_task_state(task_id, TaskState.REJECTED)
    except Exception as e:
        logger.info(
            f"Failed to post {task_id} to the {TaskState.REJECTED.value} endpoint: {e}"
        )


def create_synthetic_task(prompt: str) -> ImageGenerationTaskModel:
    """Create a synthetic image generation task"""
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


def get_thread_configs(
    should_quit: Event, queues: dict
) -> List[Tuple[str, type, float, callable, list]]:
    """Get configuration for background threads"""
    return [
        (
            "background_loop",
            BackgroundTimer,
            60,
            background_loop,
            [should_quit],
        ),
        (
            "upload_images_process",
            MultiprocessTimer,
            0.5,
            upload_images_loop,
            [should_quit, queues["batches"]],
        ),
        (
            "upload_scores_process",
            MultiprocessTimer,
            1.0,
            upload_scores_loop,
            [should_quit, queues["scores"]],
        ),
        (
            "set_weights_process",
            MultiprocessTimer,
            1.0,
            set_weights_loop,
            [should_quit, queues["weights"]],
        ),
    ]


def initialize_queues(manager: Manager) -> dict:
    """Initialize multiprocessing queues"""
    return {
        "weights": manager.Queue(maxsize=128),
        "batches": manager.Queue(maxsize=2048),
        "scores": manager.Queue(maxsize=2048),
    }


def convert_numpy_to_torch(
    metagraph: bt.metagraph, device: torch.device
) -> None:
    """Convert numpy arrays in metagraph to torch tensors"""
    for key in ["stake", "uids"]:
        if isinstance(getattr(metagraph, key), np.ndarray):
            setattr(
                metagraph,
                key,
                torch.from_numpy(getattr(metagraph, key)).float().to(device),
            )


def queue_weight_setting_task(
    active_queue: Queue,
    moving_average_scores: torch.Tensor,
) -> bool:
    """Try to queue a weight setting task"""
    try:
        active_queue.put_nowait(
            SetWeightsTask(
                epoch=ttl_get_block(),
                hotkeys=copy.deepcopy(get_metagraph().hotkeys),
                weights=tensor_to_list(moving_average_scores),
            )
        )
        logger.info(
            f"Added weight setting task to queue (size={active_queue.qsize()})"
        )
        return True
    except queue.Full:
        logger.error("Cannot add weights setting task, queue is full!")
        return False


async def execute_post_step_methods(
    methods: list,
    moving_average_scores: torch.Tensor = None,
):
    """Execute a list of post-step methods safely"""
    for method in methods:
        name = method.__name__ if hasattr(method, "__name__") else str(method)

        try:
            logger.info(f"Running post step: {name}")
            if inspect.iscoroutinefunction(method):
                await method()
            else:
                (
                    method()
                    if not moving_average_scores
                    else method(moving_average_scores)
                )
        except Exception:
            logger.error(f"{name} failed: " + traceback.format_exc())


def drain_queue(q: Queue) -> None:
    """Safely drain a queue"""
    try:
        while not q.empty():
            try:
                q.get_nowait()
            except:
                break
    except Exception as e:
        logger.debug(f"Error draining queue: {e}")


def shutdown_processes(processes: Sequence[ThreadLike]) -> None:
    """Gracefully shutdown background processes"""
    for process in processes:
        if process and process.is_alive():
            try:
                process.cancel()
                process.join(timeout=2)
            except Exception as e:
                logger.debug(f"Error stopping process {process}: {e}")


def cleanup_queues(queues: List[Queue]) -> None:
    """Drain and close multiprocessing queues"""
    for q in queues:
        try:
            drain_queue(q)
            q.close()
            q.join_thread()
        except Exception as e:
            logger.debug(f"Error closing queue: {e}")


def cleanup_logging_handlers() -> None:
    """Remove all logging handlers and set up minimal stderr logging"""
    try:
        # Remove default handler (ID 0) and any other handlers
        logger.remove()

    except Exception as e:
        logger.error(f"Error during logging cleanup: {e}")

    # Set up minimal stderr logging
    try:
        logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])
    except Exception as e:
        logger.error(f"Error configuring final stderr handler: {e}")


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
        wait_for_registration(self.metagraph, self.wallet, self.subtensor)

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

        # Convert metagraph arrays to torch tensors
        convert_numpy_to_torch(self.metagraph, self.device)

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

        # Initialize manager and queues
        manager = Manager()
        self.should_quit: Event = manager.Event()
        queues = initialize_queues(manager)
        self.set_weights_queue = queues["weights"]
        self.batches_upload_queue = queues["batches"]
        self.scores_upload_queue = queues["scores"]

        self.model_type = ModelType.CUSTOM

        self.background_loop: Optional[BackgroundTimer] = None
        self.set_weights_process: Optional[MultiprocessTimer] = None
        self.upload_images_process: Optional[MultiprocessTimer] = None
        self.upload_scores_process: Optional[MultiprocessTimer] = None

        saas_show_dashboard_url()

        # Start all background threads
        self.start_threads(True)

    def start_thread(self, thread: ThreadLike, is_startup: bool = True) -> None:
        """Start thread if not already running"""
        if not thread or not thread.is_alive():
            thread.start()

            if is_startup:
                logger.info(f"Started {thread}")
            else:
                logger.warning(
                    f"{thread.__class__.__name__} was not running, restarted"
                )

    def graceful_shutdown(self) -> None:
        """Gracefully shutdown background processes and logging"""
        logger.info("Initiating graceful shutdown...")

        # Signal threads to stop
        self.should_quit.set()

        # Stop processes in order
        shutdown_processes(
            [
                self.background_loop,
                self.upload_images_process,
                self.upload_scores_process,
                self.set_weights_process,
            ]
        )

        # Clean up queues
        cleanup_queues(
            [
                self.set_weights_queue,
                self.batches_upload_queue,
                self.scores_upload_queue,
            ]
        )

        # Clean up logging
        cleanup_logging_handlers()

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
        queues = {
            "weights": self.set_weights_queue,
            "batches": self.batches_upload_queue,
            "scores": self.scores_upload_queue,
        }
        thread_configs = get_thread_configs(self.should_quit, queues)

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
            prompt = await generate_random_prompt()
            if not prompt:
                logger.error("failed to generate prompt for synthetic task")
                return None
            # NOTE: Generate synthetic request
            return create_synthetic_task(prompt)

        is_bad_prompt = await check_prompt_for_nsfw(task.prompt)

        if is_bad_prompt:
            await handle_task_rejection(self.backend_client, task.task_id)
            return None

        return task

    async def sync(self):
        """Synchronize network state for validator"""
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            if queue_weight_setting_task(
                self.set_weights_queue, self.moving_average_scores
            ):
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
        validate_registration(
            self.config.netuid,
            self.wallet.hotkey.ss58_address,
        )

    def should_sync_metagraph(self):
        return should_sync_metagraph(
            self.metagraph.last_update[self.uid],
            self.config.alchemy.epoch_length,
        )

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
        self.axon = serve_network_axon(
            wallet=self.wallet, config=self.config, subtensor=self.subtensor
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
        """Execute post-step operations"""
        await execute_post_step_methods(
            [
                self.sync,
                self.reload_settings,
                self.start_threads,
                self.update_check,
                saas_show_dashboard_url,
                lambda: save_ma_scores(self.moving_average_scores),
            ]
        )
