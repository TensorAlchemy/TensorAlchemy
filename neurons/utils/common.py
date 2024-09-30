import os
import sys
import pathlib
import subprocess

from loguru import logger


project_root = str(pathlib.Path(__file__).parent.parent.parent.resolve())


def is_validator() -> bool:
    main_module = sys.modules["__main__"]
    main_file = os.path.abspath(main_module.__file__)
    return "neurons/validator" in main_file


def log_dependencies() -> None:
    # Log dependencies versions specified in requirements.txt
    requirements_path = os.path.join(project_root, "requirements.txt")

    try:
        with open(requirements_path, "r") as req_file:
            required_packages = [
                line.strip().split("==")[0]
                for line in req_file
                if line.strip() and not line.startswith("#")
            ]

        installed_packages = subprocess.getoutput("pip freeze").split("\n")

        dependencies = []
        for package in installed_packages:
            name = package.split("==")[0]
            # Make sure bittensor dependency is logged
            if name in required_packages or "bittensor" in name:
                dependencies.append(package)

        dependencies_str = " ".join(dependencies)
        logger.info(f"dependencies: {dependencies_str}")
    except Exception as e:
        logger.error(f"error logger dependencies: {str(e)}")


def show_warning_message(msg: str, window_width=70, header="WARNING"):

    def create_line(content):
        return f"* {content:<{window_width-4}} *"

    line = "*" * window_width
    empty_line = create_line("")

    msg_lines = msg.split("\n")
    if any([len(line) > window_width for line in msg_lines]):
        warning_lines = [line.strip() for line in msg_lines]
    else:
        warning_lines = [create_line(line.strip()) for line in msg_lines]

    message = "\n".join(
        [line, empty_line, create_line(header), empty_line]
        + warning_lines
        + [empty_line, line]
    )

    logger.warning(f"\n{message}")
