import os
import sys
import pathlib
import subprocess
from typing import List

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


from typing import List
import logging


def show_boxed_message(
    msg: str,
    window_width: int = 90,
    show_edges: bool = True,
    header: str = "WARNING",
    message_type: str = "warning",
) -> None:
    valid_message_types = ["error", "success", "warning", "info"]
    if message_type not in valid_message_types:
        raise ValueError(
            #
            "Invalid message_type. Must be one of "
            + valid_message_types
        )

    def create_line(content: str = " ") -> str:
        if not show_edges:
            return f"{content}\n"

        return f"* {content:<{window_width-4}} *\n"

    def format_lines(content: str) -> List[str]:
        return [create_line(part) for part in content.split("\n")]

    message = "".join(
        [
            "*" * window_width + "\n",
            create_line(),
            *format_lines(header),
            create_line(),
            *format_lines(msg),
            create_line(),
            "*" * window_width + "\n",
        ]
    )

    getattr(logger, message_type)(f"\n{message}")
