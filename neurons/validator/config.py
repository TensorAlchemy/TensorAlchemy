import os
import argparse
from typing import Dict, Optional

import torch
import bittensor as bt
from loguru import logger

from neurons.constants import (
    IS_TEST,
    EVENTS_RETENTION_SIZE,
)


def get_default_device() -> torch.device:
    if IS_TEST:
        logger.info("Using CPU for test environment (CI)")
        return torch.device("cpu:0")

    return torch.device("cuda:0")


def check_config(to_check: bt.config):
    """Checks/validates the config namespace object."""
    bt.logging.check_config(to_check)
    # bt.wallet.check_config(config)
    # bt.subtensor.check_config(config)

    if to_check.mock:
        to_check.neuron.mock_reward_models = True
        to_check.neuron.mock_gating_model = True
        to_check.neuron.mock_dataset = True
        to_check.wallet._mock = True

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            to_check.logging.logging_dir,
            to_check.wallet.name,
            to_check.wallet.hotkey,
            to_check.netuid,
            to_check.alchemy.name,
        )
    )
    to_check.alchemy.full_path = os.path.expanduser(full_path)
    if not os.path.exists(to_check.alchemy.full_path):
        os.makedirs(to_check.alchemy.full_path, exist_ok=True)

    # Add custom event logger for the events.
    logger.level("EVENTS", no=38, icon="📝")
    logger.add(
        to_check.alchemy.full_path + "/" + "completions.log",
        rotation=EVENTS_RETENTION_SIZE,
        serialize=True,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        level="EVENTS",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    )


def add_args(parser):
    # Netuid Arg
    parser.add_argument(
        "--netuid",
        type=int,
        help="Network netuid",
        default=26,
    )

    parser.add_argument(
        "--alchemy.name",
        type=str,
        help="Trials for this validator go in validator.root"
        + " / (wallet_cold - wallet_hot) / validator.name.",
        default="image_alchemy_validator",
    )
    parser.add_argument(
        "--alchemy.device",
        type=str,
        default=get_default_device(),
        help="Device to run the validator on.",
    )
    parser.add_argument(
        "--alchemy.force_prod",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--alchemy.streamlit_port",
        type=int,
        help="Port number for streamlit app",
        default=None,
    )

    # Add arguments for validator settings (downloaded)
    parser.add_argument(
        "--alchemy.request_frequency",
        type=int,
        default=35,
        help="Request frequency for the validator",
    )
    parser.add_argument(
        "--alchemy.query_timeout",
        type=float,
        default=20,
        help="Query timeout for the validator",
    )
    parser.add_argument(
        "--alchemy.async_timeout",
        type=float,
        default=1.2,
        help="Async timeout for the validator",
    )
    parser.add_argument(
        "--alchemy.epoch_length",
        type=int,
        default=100,
        help="Epoch length for the validator",
    )


config: bt.config = None
device: torch.device = None
metagraph: bt.metagraph = None
backend_client: "TensorAlchemyBackendClient" = None


def update_validator_settings(validator_settings: Dict) -> bt.config:
    global config

    if not validator_settings:
        logger.error("Failed to update validator settings")
        return config

    config.alchemy.request_frequency = int(
        validator_settings.get(
            "request_frequency",
            config.request_frequency,
        )
    )
    config.alchemy.query_timeout = float(
        validator_settings.get(
            "query_timeout",
            config.query_timeout,
        )
    )
    config.alchemy.async_timeout = int(
        validator_settings.get(
            "async_timeout",
            config.async_timeout,
        )
    )
    config.alchemy.epoch_length = int(
        validator_settings.get(
            "epoch_length",
            config.epoch_length,
        )
    )
    logger.info(
        #
        "Retrieved the latest validator settings: "
        + validator_settings,
    )

    return config


def get_config():
    global config
    if config:
        return config

    parser = argparse.ArgumentParser()

    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)

    # Add default arguments
    add_args(parser)

    config = bt.config(parser)
    check_config(config)

    return config


def get_metagraph(netuid: int = 25, network: str = "test", **kwargs) -> bt.metagraph:
    global metagraph
    if not metagraph:
        metagraph = bt.metagraph(
            netuid=netuid,
            network=network,
            **kwargs,
        )

    return metagraph


def get_backend_client() -> "TensorAlchemyBackendClient":
    global backend_client
    if not backend_client:
        from neurons.validator.backend.client import TensorAlchemyBackendClient

        backend_client = TensorAlchemyBackendClient()

    return backend_client


def get_device(new_device: Optional[torch.device] = None) -> torch.device:
    global device
    if not device:
        if new_device is None:
            device = get_default_device()

        else:
            device = new_device

    return device
