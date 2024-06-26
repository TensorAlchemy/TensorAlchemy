from loguru import logger


def sh(message: str):
    return f"{message: <12}"


# Utility function for coloring logs
def colored_log(
    message: str,
    color: str = "white",
    level: str = "INFO",
) -> None:
    logger.opt(colors=True).log(level, f"<bold><{color}>{message}</{color}></bold>")
