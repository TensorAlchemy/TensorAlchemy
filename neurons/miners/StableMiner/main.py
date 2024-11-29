import os
import sys
import pathlib
import warnings
from loguru import logger

from neurons.utils.log import configure_logging
from neurons.miners.StableMiner.miner import StableMiner

# Suppress warnings and use torch style
warnings.simplefilter("ignore")
os.environ["USE_TORCH"] = "1"


def setup_environment():
    """Setup paths and logging"""
    try:
        # Add project root to path
        root_path = pathlib.Path(__file__).parents[4].resolve()
        if root_path.exists() and str(root_path) not in sys.path:
            sys.path.append(str(root_path))

        # Configure logging
        configure_logging()

    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        raise


def main():
    """Main entry point"""
    try:
        setup_environment()

        logger.info("Initializing Stable Miner...")
        miner = StableMiner()

        logger.info("Starting miner service...")
        while True:
            try:
                miner.start()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Runtime error: {e}")
                raise

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
