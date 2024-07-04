import copy
import os
from threading import Timer

import torch
from loguru import logger
from neurons.constants import WANDB_MINER_PATH
from neurons.utils.image import image_b64_to_pil
from neurons.utils.log import colored_log

import bittensor as bt
import wandb


# Wandb functions
class WandbTimer(Timer):
    def run(self):
        self.function(*self.args, **self.kwargs)
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class WandbUtils:
    def __init__(
        self,
        miner,
        metagraph,
        config,
        wallet,
        event,
    ):
        self.miner = miner
        self.metagraph = metagraph
        self.config = config
        self.wallet = wallet
        self.wandb = None
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.event = event
        self.timer = WandbTimer(600, self._loop, [])
        self.timer.start()

    def _loop(self):
        if not self.wandb:
            self._start_run()

    def _start_run(self):
        if self.wandb:
            self._stop_run()

        logger.info(
            f"Wandb starting run with project {self.config.wandb.project} "
            + f"and entity {self.config.wandb.entity}."
        )

        # Start new run
        config = copy.deepcopy(self.config)
        config["model"] = self.config.model

        tags = [
            self.wallet.hotkey.ss58_address,
            f"netuid_{self.metagraph.netuid}",
        ]

        if not os.path.exists(WANDB_MINER_PATH):
            os.makedirs(WANDB_MINER_PATH, exist_ok=True)

        wandb.login(anonymous="never", key=self.config.wandb.api_key)

        self.wandb = wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            config=config,
            tags=tags,
            dir=WANDB_MINER_PATH,
        )

        # Take the first two random words
        # plus the name of the wallet, hotkey name and uid
        self.wandb.name = (
            "-".join(self.wandb.name.split("-")[:2])
            + f"-{self.wallet.name}-{self.wallet.hotkey_str}-{self.uid}"
        )
        colored_log(f"Started new run: {self.wandb.name}", "c")

    def _add_images(self, synapse, file_type="jpg"):
        # Store the images and prompts for uploading to wandb

        self.event.update(
            {
                "images": [
                    wandb.Image(
                        image_b64_to_pil(image),
                        caption=synapse.prompt,
                        file_type=file_type,
                    )
                    if image != []
                    else wandb.Image(
                        torch.full([3, 1024, 1024], 255, dtype=torch.float),
                        caption=synapse.prompt,
                        file_type=file_type,
                    )
                    for image in synapse.images
                ],
            }
        )

    def _stop_run(self):
        self.wandb.finish()

    def _log(self):
        # Log incentive, trust, emissions, total requests, timeouts
        self.event.update(self.miner.get_miner_info())
        self.event.update(
            {
                "total_requests": self.miner.stats.total_requests,
                "timeouts": self.miner.stats.timeouts,
            }
        )
        self.wandb.log(self.event)
