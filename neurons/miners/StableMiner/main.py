import pathlib
import sys

if __name__ == "__main__":
    # Add the base repository to the path so the miner can access it
    file_path = str(
        pathlib.Path(__file__).parent.parent.parent.parent.resolve(),
    )
    if file_path not in sys.path:
        sys.path.append(file_path)

    # Import StableMiner after fixing path
    from miner import StableMiner

    # Start the miner
    StableMiner()
