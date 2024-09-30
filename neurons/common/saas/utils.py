import base64
import json
import time
import urllib.parse

import bittensor as bt

from substrateinterface import Keypair

from neurons import constants
from neurons.config import get_config, AlchemyHost
from neurons.config.utils import is_testnet
from neurons.utils.common import is_validator


def get_dashboard_base_url():
    if is_testnet():
        if get_config().alchemy.host == AlchemyHost.TESTNET:
            return constants.SAAS_DASHBOARD_TESTNET_URL
        return constants.SAAS_DASHBOARD_DEVELOP_URL

    return constants.SAAS_DASHBOARD_MAINNET_URL


def saas_generate_dashboard_url(wallet: bt.wallet):
    """
    Generates a dashboard URL with a signed token for authentication.

    Args:
    wallet: The wallet object containing the hotkey.

    Returns:
    str: The dashboard URL with the authentication token.
    """
    payload = {
        "exp": int(time.time()) + constants.SAAS_DASHBOARD_URL_EXPIRATION_TIME,
        "iat": int(time.time()),
        "hotkey": wallet.hotkey.ss58_address,
    }

    message = json.dumps(payload)

    signature = wallet.hotkey.sign(message.encode())

    signature_base64 = base64.b64encode(signature).decode()

    token_data = {"payload": payload, "signature": signature_base64}

    token = base64.b64encode(json.dumps(token_data).encode()).decode()

    neuron_type = "validator" if is_validator() else "miner"
    query_params = {"type": neuron_type, "token": token}

    dashboard_url = (
        f"{get_dashboard_base_url()}?{urllib.parse.urlencode(query_params)}"
    )

    return dashboard_url
