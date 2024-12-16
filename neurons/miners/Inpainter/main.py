import os
import pathlib
import sys
import warnings

from loguru import logger
from neurons.miners.Inpainter.miner import InpaintMiner
from neurons.utils.log import configure_logging

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

        logger.info("Initializing Inpaint Miner...")
        miner = InpaintMiner()

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
