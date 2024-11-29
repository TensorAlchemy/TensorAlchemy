def get_miner_version() -> str:
    """Returns version of miner (i.e. 1.0.1)"""
    import neurons.miners

    return neurons.miners.__version__


def get_miner_spec_version() -> int:
    """Returns numeric representation of miner's version (i.e. 10001)"""
    import neurons.miners

    return neurons.miners.__spec_version__
