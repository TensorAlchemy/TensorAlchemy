import asyncio
import logging
import pathlib
import sys
import warnings

# Suppress the eth_utils network warnings
# "does not have a valid ChainId."
# NOTE: It's not our bug, it's upstream
# TODO: Remove after updating bittensor
warnings.simplefilter("ignore")
logging.getLogger().setLevel(logging.WARNING)

# Use the older torch style for now


if __name__ == "__main__":
    # Add the base repository to the path so the validator can access it
    file_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
    if file_path not in sys.path:
        sys.path.append(file_path)

    from neurons.update_checker import safely_check_for_updates
    from neurons.utils.log import configure_logging, log_banner

    configure_logging()
    safely_check_for_updates()

    log_banner(
        "Starting Validator",
        width=30,
    )

    # Import StableValidator after fixing paths
    from validator import StableValidator

    asyncio.run(StableValidator().run())
