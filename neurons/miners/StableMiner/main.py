import os
import pathlib
import sys
import warnings
from loguru import logger
from neurons.utils.log import configure_logging
from neurons.miners.StableMiner.miner import StableMiner

# Suppress the eth_utils network warnings
warnings.simplefilter("ignore")

# Use the older torch style for now
os.environ["USE_TORCH"] = "1"


def setup_paths():
    try:
        logger.info("Setting up paths...")
        file_path = pathlib.Path(__file__).parents[4].resolve()
        logger.info(f"Resolved file path: {file_path}")

        if file_path.exists():
            if str(file_path) not in sys.path:
                sys.path.append(str(file_path))
                logger.success(f"Added path to sys.path: {file_path}")
        else:
            raise FileNotFoundError(f"Path does not exist: {file_path}")

        logger.info("Path setup completed successfully")
    except Exception as e:
        logger.error(f"Error in setup_paths: {str(e)}")
        raise


def main():
    try:
        logger.info("Starting main initialization...")

        logger.info("Setting up paths...")
        setup_paths()
        logger.success("Paths setup complete")

        logger.info("Configuring logging...")
        configure_logging()
        logger.success("Logging configured")

        logger.info("Initializing Stable Miner...")
        miner = StableMiner()
        logger.success("Stable Miner initialized successfully")

        # Keep the process running
        logger.info("Entering main loop...")
        while True:
            try:
                miner.start()
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                raise

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
