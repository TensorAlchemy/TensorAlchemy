# Utils for checkpointing and saving the model.
import copy
import os

import pandas as pd
import wandb
from loguru import logger


from neurons.constants import (
    WANDB_VALIDATOR_PATH,
)
from neurons.validator.rewards.models import get_reward_models
from neurons.validator.utils.version import get_validator_version


def init_wandb(validator: "StableValidator", reinit=False):
    """Starts a new wandb run."""
    tags = [
        validator.wallet.hotkey.ss58_address,
        get_validator_version(),
        f"netuid_{validator.metagraph.netuid}",
    ]

    if validator.config.mock:
        tags.append("mock")

    wandb_config = {
        key: copy.deepcopy(validator.config.get(key, None))
        for key in ("neuron", "alchemy", "reward", "netuid", "wandb")
    }
    wandb_config["alchemy"].pop("full_path", None)

    if not os.path.exists(WANDB_VALIDATOR_PATH):
        os.makedirs(WANDB_VALIDATOR_PATH, exist_ok=True)

    project = "ImageAlchemyTest"

    if validator.config.netuid == 26:
        project = "ImageAlchemy"

    validator.wandb = wandb.init(
        anonymous="allow",
        reinit=reinit,
        project=project,
        entity="tensoralchemists",
        config=wandb_config,
        dir=WANDB_VALIDATOR_PATH,
        tags=tags,
    )
    logger.success(f"Started a new wandb run called {validator.wandb.name}.")


def reinit_wandb(self):
    """Reinitializes wandb, rolling over the run."""
    if self.wandb:
        try:
            self.wandb.finish()
        except Exception:
            pass
    init_wandb(self, reinit=True)


def get_promptdb_backup(netuid, prompt=[], limit=1):
    api = wandb.Api()
    project = "ImageAlchemy" if netuid == 26 else "ImageAlchemyTest"
    runs = api.runs(f"tensoralchemists/{project}")

    for run in runs:
        if len(prompt) >= limit:
            break
        if run.historyLineCount >= 100:
            history = run.history()
            if ("prompt" not in history.columns) or ("prompt" not in history.columns):
                continue
            for i in range(0, len(history) - 1, 2):
                if len(prompt) >= limit:
                    break

                if (
                    pd.isna(history.loc[i, "prompt"])
                    or (history.loc[i, "prompt"] is None)
                    or (i == len(history))
                    or (history.loc[i + 1, "prompt"] is None)
                    or pd.isna(history.loc[i + 1, "prompt"])
                ):
                    continue

                prompt = (
                    history.loc[i, "prompt"],
                    history.loc[i + 1, "prompt"],
                )

                if prompt in prompt:
                    continue

                prompt.append(prompt)

    return prompt
