"""
Configuration parsing and management utilities for the Alchemy project.
"""

import argparse
import os
from typing import Dict

import bittensor as bt
from loguru import logger

from neurons.config.constants import AlchemyHost
from neurons.config.device import get_default_device
from neurons.utils.settings import download_validator_settings


def add_args(parser: argparse.ArgumentParser) -> None:
    """
    Add Alchemy-specific arguments to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add arguments
    """

    # NOTE: Items here are used only for adding to config
    #       they will be overridden by downloaded JSON
    parser.add_argument(
        "--alchemy.auto_update",
        action="store_true",
        default=False,
        help="Enable automatic updates when new versions are detected",
    )
    parser.add_argument(
        "--alchemy.delta_factor",
        type=float,
        default=0.5,
        help="Factor to adjust the delta effect in moving average calculations",
    )
    parser.add_argument(
        "--alchemy.ma_decay",
        type=float,
        default=0.000001,
        help="How much do the moving averages decay each step?",
    )
    parser.add_argument(
        "--alchemy.ma_decay_cycles",
        type=int,
        default=12,
        help="How many cycles until ma_decay takes affect?",
    )
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
        default=2.0,
        help="Async timeout for the validator",
    )
    parser.add_argument(
        "--alchemy.epoch_length",
        type=int,
        default=100,
        help="Epoch length for the validator",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        help="Network netuid",
        default=26,
    )
    parser.add_argument(
        "--alchemy.name",
        type=str,
        help="Validator name",
        default="tensor_alchemy_validator",
    )
    parser.add_argument(
        "--alchemy.debug",
        type=bool,
        default=False,
        help="Enable debug logging",
    )
    parser.add_argument(
        "--alchemy.device",
        type=str,
        default=get_default_device(),
        help="Device to run the validator on",
    )
    parser.add_argument(
        "--alchemy.host",
        type=AlchemyHost,
        choices=list(AlchemyHost),
        help="Choose the Alchemy host",
    )


def check_config(to_check: bt.config) -> None:
    """
    Check and validate the configuration object.

    Args:
        to_check (bt.config): The configuration object to check.
    """
    bt.logging.check_config(to_check)

    full_path = os.path.expanduser(
        to_check.logging.logging_dir
        + f"/{to_check.wallet.name}"
        + f"/{to_check.wallet.hotkey}"
        + f"/netuid{to_check.netuid}"
        + f"/{to_check.alchemy.name}"
    )
    to_check.alchemy.full_path = os.path.expanduser(full_path)
    os.makedirs(to_check.alchemy.full_path, exist_ok=True)


def get_config() -> bt.config:
    """
    Get the global configuration object.

    Returns:
        bt.config: The global configuration object.
    """
    global config
    if config is None:
        parser = argparse.ArgumentParser()
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.axon.add_args(parser)
        add_args(parser)
        config = bt.config(parser)
        check_config(config)

    return config


async def update_validator_settings() -> None:
    """
    Update the validator settings in the global configuration.

    Args:
        validator_settings (Dict): New validator settings to apply.

    Returns:
        bt.config: The updated global configuration object.
    """
    global config
    validator_settings: Dict = await download_validator_settings()

    if not validator_settings:
        logger.error("Failed to update validator settings")
        return config

    config.alchemy.ma_decay = float(
        validator_settings.get(
            "ma_decay",
            config.ma_decay,
        )
    )
    config.alchemy.ma_decay_cycles = int(
        validator_settings.get(
            "ma_decay_cycles",
            config.ma_decay_cycles,
        )
    )
    config.alchemy.delta_factor = float(
        validator_settings.get(
            "delta_factor",
            config.delta_factor,
        )
    )
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
    config.alchemy.async_timeout = float(
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
        f"Retrieved the latest validator settings: {validator_settings}"
    )


config: bt.config = None
