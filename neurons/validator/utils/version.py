def get_validator_version() -> str:
    """Returns version of validator (i.e. 1.0.1)"""
    import neurons.validator

    return neurons.validator.__version__


def get_validator_spec_version() -> int:
    """Returns numeric representation of validator's version (i.e. 10001)"""
    import neurons.validator

    return neurons.validator.__spec_version__
