from neurons.config import get_config


def is_testnet():
    return get_config().netuid == 25
