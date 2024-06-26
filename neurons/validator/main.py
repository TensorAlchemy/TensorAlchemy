import asyncio
import pathlib
import sys

from loguru import logger

REPO_URL = "TensorAlchemy/TensorAlchemy"

if __name__ == "__main__":
    # Add the base repository to the path so the validator can access it
    file_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
    if file_path not in sys.path:
        sys.path.append(file_path)

    current_folder = str(pathlib.Path(__file__).parent.resolve())

    from neurons.update_checker import check_for_updates

    try:
        check_for_updates(current_folder, REPO_URL)
    except Exception as error:
        logger.warning(
            "Failed to check remote for updates: " + str(error),
        )

    # Import StableValidator after fixing paths
    from validator import StableValidator

    asyncio.run(StableValidator().run())
