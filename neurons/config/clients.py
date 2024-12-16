"""
Client management utilities for the Alchemy project.
"""

import os
from typing import Optional

from loguru import logger
from neurons.config.parser import get_config
from openai import AsyncOpenAI

import bittensor as bt

wallet: Optional[bt.wallet] = None
dendrite: Optional[bt.dendrite] = None
metagraph: Optional[bt.metagraph] = None
subtensor: Optional[bt.subtensor] = None
openai_client: Optional[AsyncOpenAI] = None
backend_client: Optional["TensorAlchemyBackendClient"] = None


class MissingApiKeyError(ValueError):
    pass


def get_corcel_api_key(required: bool = False) -> str:
    """
    Get the Corcel API key from environment variables.

    Returns:
        str: The Corcel API key.
    """
    to_return: str = os.environ.get("CORCEL_API_KEY", "")

    if required and not len(to_return):
        raise MissingApiKeyError("Please set CORCEL_API_KEY")

    return to_return


def get_openai_api_key(required: bool = False) -> str:
    """
    Get the Openai API key from environment variables.

    Returns:
        str: The Openai API key.
    """
    to_return: str = os.environ.get("OPENAI_API_KEY", "")

    if required and not len(to_return):
        raise MissingApiKeyError("Please set OPENAI_API_KEY")

    return to_return


def get_openai_client(nocache: bool = False, **kwargs) -> AsyncOpenAI:
    """
    Get or create the global OpenAI client.

    Args:
        nocache (Optional[Bool]): Forced reinit of subtensor.

    Returns:
        AsyncOpenAI: The global OpenAI client.
    """
    global openai_client

    if openai_client is None or nocache:
        openai_client = AsyncOpenAI(
            api_key=get_openai_api_key(required=True),
            **kwargs,
        )

    return openai_client


def get_wallet(nocache: bool = False, **kwargs) -> bt.wallet:
    """
    Get or create the global wallet.

    Args:
        nocache (Optional[Bool]): Forced reinit of subtensor.

    Returns:
        bt.wallet: The global wallet.
    """
    global wallet

    if wallet is None or nocache:
        wallet = bt.wallet(config=get_config(), **kwargs)

    return wallet


def get_dendrite(nocache: bool = False, **kwargs) -> bt.dendrite:
    """
    Get or create the global dendrite.

    Args:
        nocache (Optional[Bool]): Forced reinit of subtensor.

    Returns:
        bt.dendrite: The global dendrite.
    """
    global dendrite

    if dendrite is None or nocache:
        dendrite = bt.dendrite(wallet=get_wallet(), **kwargs)

    return dendrite


def get_subtensor(nocache: bool = False, **kwargs) -> bt.subtensor:
    """
    Get or create the global subtensor.

    Args:
        nocache (Optional[Bool]): Forced reinit of subtensor.

    Returns:
        bt.subtensor: The global subtensor.
    """
    global subtensor

    from neurons.constants import IS_CI_ENV

    if IS_CI_ENV:
        raise NotImplementedError("get_subtensor() must be mocked in CI tests")

    if subtensor is None or nocache:
        subtensor = bt.subtensor(config=get_config(), **kwargs)

    return subtensor


def get_metagraph(nocache: bool = False, **kwargs) -> bt.metagraph:
    """
    Get or create the global metagraph.

    Args:
        nocache (Optional[Bool]): Forced reinit of subtensor.
        **kwargs: Additional arguments to pass to the metagraph constructor.

    Returns:
        bt.metagraph: The global metagraph.
    """
    global metagraph

    from neurons.constants import IS_CI_ENV

    if IS_CI_ENV:
        raise NotImplementedError("get_metagraph() must be mocked in CI tests")

    if metagraph is None or nocache:
        config = get_config()
        netuid: int = config.netuid or 26
        network: str = get_subtensor().chain_endpoint or "finney"

        logger.info(f"Creating connection to metagraph: {netuid=}: {network=}")

        metagraph = bt.metagraph(
            netuid=netuid,
            network=network,
            **kwargs,
        )

    return metagraph


def get_backend_client(nocache: bool = False) -> "TensorAlchemyBackendClient":
    """
    Get or create the global TensorAlchemyBackendClient.

    Returns:
        TensorAlchemyBackendClient: The global TensorAlchemyBackendClient.
    """
    global backend_client

    if backend_client is None or nocache:
        from neurons.validator.backend.client import TensorAlchemyBackendClient

        backend_client = TensorAlchemyBackendClient()

    return backend_client


# TODO: Remove
c = get_config()
s = get_subtensor()
m = get_metagraph()
