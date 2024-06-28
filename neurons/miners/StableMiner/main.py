import pathlib
import sys
import warnings
from typing import List

# Suppress the eth_utils network warnings
# "does not have a valid ChainId."
# NOTE: It's not our bug, it's upstream
# TODO: Remove after updating bittensor
warnings.simplefilter("ignore")

if __name__ == "__main__":
    try:
        # Add the base repository to the path so the miner can access it
        file_path: str = str(
            pathlib.Path(__file__).parent.parent.parent.parent.resolve(),
        )
        if file_path not in sys.path:
            sys.path.append(file_path)

        # Import StableMiner after fixing path
        from miner import StableMiner

        # Start the miner
        StableMiner()
    except ImportError as e:
        print(f"Error importing StableMiner: {e}")
        print("Please ensure all required packages are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
