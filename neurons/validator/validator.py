import asyncio
import copy
import os
import sys
import time
import traceback
import uuid
from asyncio import Queue

import bittensor as bt
import sentry_sdk
import torch
import wandb
from loguru import logger
import sentry_sdk

from neurons.constants import (
    DEV_URL,
    N_NEURONS,
    PROD_URL,
    VALIDATOR_SENTRY_DSN,
)

from neurons.protocol import (
    denormalize_image_model,
    ImageGenerationTaskModel,
    ModelType,
)
from neurons.utils import (
    BackgroundTimer,
    background_loop,
    colored_log,
    get_defaults,
)
from neurons.validator.config import (
    check_config,
    add_args,
    get_device,
    get_config,
    get_metagraph,
)
from neurons.validator.backend.client import TensorAlchemyBackendClient
from neurons.validator.backend.models import TaskState
from neurons.validator.forward import run_step
from neurons.validator.rewards.models.blacklist import BlacklistFilter
from neurons.validator.rewards.models.similarity import ModelSimilarityRewardModel
from neurons.validator.rewards.models.human import HumanValidationRewardModel
from neurons.validator.rewards.models.image_reward import ImageRewardModel
from neurons.validator.rewards.models.nsfw import NSFWRewardModel
from neurons.validator.rewards.reward import BaseRewardModel, RewardProcessor
from neurons.validator.schemas import Batch
from neurons.validator.services.openai.service import get_openai_service
from neurons.validator.utils import (
    generate_random_prompt_gpt,
    get_device_name,
    get_random_uids,
    init_wandb,
    reinit_wandb,
    ttl_get_block,
    get_validator_version,
)
from neurons.validator.weights import set_weights


def is_valid_current_directory() -> bool:
    # NOTE: We use Alchemy for support
    #       of the old repository name ImageAlchemy
    #       otherwise normally this would be TensorAlchemy
    if "Alchemy" in os.getcwd():
        return True

    return False


class StableValidator:
    def loop_until_registered(self):
        index = None
        while True:
            try:
                index = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
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

        environment: str = "production"
        if self.config.subtensor.network == "test":
            environment = "local"

        sentry_sdk.init(
            environment=environment,
            dsn=VALIDATOR_SENTRY_DSN,
        )

        bt.logging(
            config=self.config,
            logging_dir=self.config.alchemy.full_path,
            debug=self.config.debug,
            trace=self.config.trace,
        )

        # Init device.
        self.device = get_device(torch.device(self.config.alchemy.device))

        self.corcel_api_key = os.environ.get("CORCEL_API_KEY")

        # Init external API services
        self.openai_service = get_openai_service()

        self.backend_client = TensorAlchemyBackendClient()

        wandb.login(anonymous="must")

        self.prompt_generation_failures = 0

        # Init subtensor
        self.subtensor = bt.subtensor(config=self.config)
        logger.info(f"Loaded subtensor: {self.subtensor}")

        try:
            sentry_sdk.set_context(
                "bittensor", {"network": str(self.subtensor.network)}
            )
            sentry_sdk.set_context(
                "cuda_device", {"name": get_device_name(self.device)}
            )
        except Exception:
            logger.error("Failed to set sentry context")

        self.api_url = DEV_URL if self.subtensor.network == "test" else PROD_URL
        if self.config.alchemy.force_prod:
            self.api_url = PROD_URL

        logger.info(f"Using server {self.api_url}")

        # Init wallet.
        self.wallet = bt.wallet(config=self.config)
        self.wallet.create_if_non_existent()

        # Dendrite pool for querying the network during training.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        logger.info(f"Loaded dendrite pool: {self.dendrite}")

        # Init metagraph.
        self.metagraph = get_metagraph(
            netuid=self.config.netuid,
            # Make sure not to sync without passing subtensor
            network=self.subtensor.network,
            sync=False,
        )

        # Sync metagraph with subtensor.
        self.metagraph.sync(subtensor=self.subtensor)
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        if "mock" not in self.config.wallet.name:
            # Wait until the miner is registered
            self.loop_until_registered()

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info("Loaded metagraph")

        self.scores = torch.zeros_like(self.metagraph.stake, dtype=torch.float32)

        # Init Weights.
        self.moving_average_scores = torch.zeros((self.metagraph.n)).to(self.device)

        # Each validator gets a unique identity (UID)
        # in the network for differentiation.
        self.my_subnet_uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
        )
        validator_version = get_validator_version()
        logger.info(
            f"Running validator (version={validator_version}) on uid: {self.my_subnet_uid}"
        )

        # Init weights
        self.weights = torch.ones_like(self.metagraph.uids, dtype=torch.float32).to(
            self.device
        )

        # Init prev_block and step
        self.prev_block = ttl_get_block(self)
        self.step = 0

        # Set validator variables
        self.request_frequency = 35
        self.query_timeout = 20
        self.async_timeout = 1.2
        self.epoch_length = 300

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        self.serve_axon()

        # Init the event loop
        self.loop = asyncio.get_event_loop()

        # Init wandb.
        try:
            init_wandb(self)
            logger.info("Loaded wandb")
            self.wandb_loaded = True
        except Exception:
            self.wandb_loaded = False
            logger.error("Unable to load wandb. Retrying in 5 minnutes.")
            logger.error(f"wandb loading error: {traceback.format_exc()}")

        # Init blacklists and whitelists
        self.hotkey_blacklist = set()
        self.coldkey_blacklist = set()
        self.hotkey_whitelist = set()
        self.coldkey_whitelist = set()

        # Init IsAlive counter
        self.isalive_threshold = 8
        self.isalive_dict = {i: 0 for i in range(self.metagraph.n.item())}

        # Init stats
        self.stats = get_defaults(self)

        # Get vali index
        self.validator_index = self.get_validator_index()

        # Start the batch streaming background loop
        # self.batches = {}
        self.batches_upload_queue = Queue()

        # Start the generic background loop
        self.storage_client = None
        self.background_steps = 1
        self.background_timer = BackgroundTimer(
            60,
            background_loop,
            [self, True],
        )
        self.background_timer.daemon = True
        self.background_timer.start()

        # Create a Dict for storing miner query history
        try:
            self.miner_query_history_duration = {
                self.metagraph.axons[uid].hotkey: float("inf")
                for uid in range(self.metagraph.n.item())
            }
        except Exception:
            pass
        try:
            self.miner_query_history_count = {
                self.metagraph.axons[uid].hotkey: 0
                for uid in range(self.metagraph.n.item())
            }
        except Exception:
            pass
        try:
            self.miner_query_history_fail_count = {
                self.metagraph.axons[uid].hotkey: 0
                for uid in range(self.metagraph.n.item())
            }
        except Exception:
            pass

        self.model_type = ModelType.CUSTOM

    async def run(self):
        # Main Validation Loop
        logger.info("Starting validator loop.")
        # Load Previous Sates
        self.load_state()
        self.step = 0
        while True:
            try:
                # Get a random number of uids
                uids = await get_random_uids(self, self.dendrite, k=N_NEURONS)
                uids = uids.to(self.device)

                axons = [self.metagraph.axons[uid] for uid in uids]

                task = await self.get_image_generation_task()
                if task is None:
                    logger.warning(
                        "image generation task was not generated successfully."
                    )

                    # Prevent loop from forming if the task
                    # error occurs on the first step
                    if self.step == 0:
                        self.step += 1

                    continue

                # Text to Image Run
                await run_step(
                    validator=self,
                    task=task,
                    axons=axons,
                    uids=uids,
                    model_type=self.model_type,
                    stats=self.stats,
                )
                # Re-sync with the network. Updates the metagraph.
                try:
                    self.sync()
                except Exception as e:
                    logger.error(
                        "An unexpected error occurred"
                        + f" trying to sync the metagraph: {e}"
                    )

                # Save Previous Sates
                self.save_state()

                # End the current step and prepare for the next iteration.
                self.step += 1

                # Assuming each step is 3 minutes restart wandb run ever
                # 3 hours to avoid overloading a validators storage space
                if self.step % 360 == 0 and self.step != 0:
                    logger.info("Re-initializing wandb run...")
                    try:
                        reinit_wandb(self)
                        self.wandb_loaded = True
                    except Exception as e:
                        logger.info(
                            f"An unexpected error occurred reinitializing wandb: {e}"
                        )
                        self.wandb_loaded = False

            # If we encounter an unexpected error, log it for debugging.
            except Exception as e:
                logger.error(traceback.format_exc())
                sentry_sdk.capture_exception(e)

            # If the user interrupts the program, gracefully exit.
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt detected. Exiting validator.")
                sys.exit(1)

    async def get_image_generation_task(
        self, timeout=60
    ) -> ImageGenerationTaskModel | None:
        """
        Fetch new image generation task from backend or generate new one
        Returns task or None if task cannot be generated
        """
        # NOTE: Will wait for around 60 seconds
        #       trying to get a task from the user
        # before going on and creating a synthetic task

        logger.info(
            f"polling backend for incoming image generation task ({timeout}s) ..."
        )

        task = await self.backend_client.poll_task(timeout=timeout)

        # No organic task found
        if task is None:
            self.model_type = ModelType.CUSTOM
            prompt = await generate_random_prompt_gpt(self)
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
                steps=50,
                width=1024,
                height=1024,
            )

        is_bad_prompt = await self.openai_service.check_prompt_for_nsfw(task.prompt)

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
            asyncio.run(set_weights(self))
            self.prev_block = ttl_get_block(self)

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

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys
        and moving averages based on the new metagraph."""

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        logger.info(
            "Metagraph updated, re-syncing hotkeys,"
            + " dendrite pool and moving averages"
        )

        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros((self.metagraph.n)).to(self.device)
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average
            for uid in self.metagraph.n.item():
                if uid not in self.isalive_dict:
                    self.isalive_dict[uid] = 0

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
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
            ttl_get_block(self) - self.metagraph.last_update[self.uid]
        ) > self.epoch_length

    def should_set_weights(self) -> bool:
        # Check if all moving_averages_socres are the 0s or 1s
        ma_scores = self.moving_average_scores
        ma_scores_sum = sum(ma_scores)
        if any([ma_scores_sum == len(ma_scores), ma_scores_sum == 0]):
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        return (ttl_get_block(self) % self.prev_block) >= self.epoch_length

    def save_state(self):
        r"""Save hotkeys, neuron model and moving average scores to filesystem."""
        logger.info("save_state()")
        try:
            neuron_state_dict = {
                "neuron_weights": self.moving_average_scores.to("cpu").tolist(),
            }
            torch.save(
                neuron_state_dict, f"{self.config.alchemy.full_path}/model.torch"
            )
            colored_log(
                f"Saved model {self.config.alchemy.full_path}/model.torch",
                color="blue",
            )
        except Exception as e:
            logger.error(f"Failed to save model with error: {e}")

        # empty cache
        torch.cuda.empty_cache()

    def load_state(self):
        r"""Load hotkeys and moving average scores from filesystem."""
        logger.info("load_state()")
        try:
            state_dict = torch.load(f"{self.config.alchemy.full_path}/model.torch")
            neuron_weights = torch.tensor(state_dict["neuron_weights"])

            has_nans = torch.isnan(neuron_weights).any()
            has_infs = torch.isinf(neuron_weights).any()

            if has_nans:
                logger.info(f"Nans found in the model state: {has_nans}")

            if has_infs:
                logger.info(f"Infs found in the model state: {has_infs}")

            # Check to ensure that the size of the neruon
            # weights matches the metagraph size.
            if neuron_weights.shape != (self.metagraph.n,):
                logger.warning(
                    f"Neuron weights shape {neuron_weights.shape} "
                    + f"does not match metagraph n {self.metagraph.n}"
                    "Populating new moving_averaged_scores IDs with zeros"
                )
                self.moving_average_scores[: len(neuron_weights)] = neuron_weights.to(
                    self.device
                )
                # self.update_hotkeys()

            # Check for nans in saved state dict
            elif not any([has_nans, has_infs]):
                self.moving_average_scores = neuron_weights.to(self.device)
                logger.info(f"MA scores: {self.moving_average_scores}")
                # self.update_hotkeys()
            else:
                logger.info("Loaded MA scores from scratch.")

            # Zero out any negative scores
            for i, average in enumerate(self.moving_average_scores):
                if average < 0:
                    self.moving_average_scores[i] = 0

            colored_log(
                f"Reloaded model {self.config.alchemy.full_path}/model.torch",
                color="blue",
            )

        except Exception as e:
            logger.error(f"Failed to load model with error: {e}")

    def serve_axon(self):
        """Serve axon to enable external connections."""

        logger.info("serving ip to chain...")
        try:
            self.axon = bt.axon(
                wallet=self.wallet,
                ip=bt.utils.networking.get_external_ip(),
                external_ip=bt.utils.networking.get_external_ip(),
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
            logger.error(f"Failed to create Axon initialize with exception: {e}")
