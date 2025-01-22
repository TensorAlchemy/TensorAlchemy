def is_testnet() -> bool:
    from neurons.config.parser import get_config

    return get_config().netuid == 25
