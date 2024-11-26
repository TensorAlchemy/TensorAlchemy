import os
import sys
import json
import logging
from multiprocessing import Queue
from typing import Any
from logging.handlers import QueueHandler, QueueListener

import bittensor as bt
import logging_loki
from loguru import logger
from PIL.Image import Image as ImageType

from neurons import constants


def log_banner(message, width=80, max_url_width=None):
    def wrap_text(text, width):
        # Don't wrap URLs/lines that should stay intact
        if text.startswith(("http://", "https://")):
            return [text]

        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word)
            if current_length + word_length + 1 <= width:
                current_line.append(word)
                current_length += word_length + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length

        if current_line:
            lines.append(" ".join(current_line))
        return lines

    # Split input into lines and wrap each line
    content_width = width - 4  # Account for borders and padding
    lines = []
    for line in message.split("\n"):
        if line.strip():
            lines.extend(wrap_text(line, content_width))
        else:
            lines.append("")

    # Create the banner
    print(f"╔{'═' * (width-1)}╗")
    for line in lines:
        if line.startswith(("http://", "https://")):
            # Don't center URLs, just pad with spaces
            print(f"║ {line}{' ' * (width-3-len(line))}")
        else:
            print(f"║ {line:^{width-3}} ║")
    print(f"╚{'═' * (width-1)}╝")


LOKI_VALIDATOR_APP_NAME = "tensoralchemy-validator"
LOKI_MINER_APP_NAME = "tensoralchemy-miner"

TA_LOGGER_DISABLED: bool = os.getenv("TA_LOGGER_DISABLED") == "true"


def image_to_str(image: Any) -> str:
    if isinstance(image, str):
        return f"base64(**bytes:<{len(image)}>**)"

    if isinstance(image, bt.Tensor):
        return f"bt.Tensor({image.shape})"

    if hasattr(image, "shape"):
        return f"shaped({image.shape})"

    if isinstance(image, ImageType):
        return f"PIL.Image({image.width}, {image.height})"

    return f"UNKNOWN IMAGE TYPE {type(image)}"


def sh(message: str):
    return f"{message: <12}"


def summarize_rewards(reward_tensor: "torch.Tensor") -> str:
    non_zero = reward_tensor[reward_tensor != 0]
    if len(non_zero) == 0:
        return "All zeros"

    return (
        f"Non-zero: {len(non_zero)}/{len(reward_tensor)}, "
        f"Mean: {non_zero.mean():.4f}, "
        f"Max: {non_zero.max():.4f}, "
        f"Min non-zero: {non_zero.min():.4f}"
    )


def get_subtensor_network_from_netuid(netuid: int) -> str:
    return {25: "testnet", 26: "finney"}.get(netuid, "")


def configure_loki_logger():
    from neurons.config import get_config, validator_run_id
    from neurons.miners.StableMiner.utils.version import (
        get_miner_version,
        get_miner_spec_version,
    )
    from neurons.validator.utils.version import (
        get_validator_version,
        get_validator_spec_version,
    )
    from neurons.utils.common import is_validator

    """Configure sending logs to loki server"""

    if constants.is_test():
        # Don't use loki for test runs
        return

    class LogHandler(logging_loki.LokiHandler):
        def handleError(self, record):
            self.emitter.close()
            # When Loki endpoint giving error for some reason,
            # parent .handleError starts spamming error trace for each failure
            # so we are disabling this default behaviour
            # super().handleError(record)

    class CustomLokiLoggingHandler(QueueHandler):
        def __init__(self, queue: Queue, **kwargs):
            super().__init__(queue)
            self.handler = LogHandler(**kwargs)  # noqa: WPS110
            self.listener = QueueListener(self.queue, self.handler)
            self.listener.start()

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            try:
                # Extract real message noisy msg line emitted by bittensor
                # might exist better solution here
                msg = "".join(record.getMessage().split(" - ")[1:])
            except Exception:
                msg = record.getMessage()

            try:
                netuid = get_config().netuid
            except:
                netuid = ""

            try:
                hotkey = bt.wallet(config=get_config()).hotkey.ss58_address
            except Exception:
                hotkey = ""

            if is_validator():
                version = get_validator_version()
                spec_version = get_validator_spec_version()
                run_id = validator_run_id.get()
            else:
                version = get_miner_version()
                spec_version = get_miner_spec_version()
                run_id = None

            log_record = {
                "level": record.levelname.lower(),
                "module": record.module,
                "func_name": record.funcName,
                "thread": record.threadName,
                "run_id": run_id,
                "netuid": netuid,
                "subnet": get_subtensor_network_from_netuid(netuid),
                "hotkey": hotkey,
                "message": msg,
                "filename": record.filename,
                "lineno": record.lineno,
                "time": self.formatTime(record, self.datefmt),
                "version": version,
                "spec_version": spec_version,
            }
            return json.dumps(log_record)

    application_name = (
        LOKI_VALIDATOR_APP_NAME if is_validator() else LOKI_MINER_APP_NAME
    )
    # Use LokiQueueHandler to upload logs in background
    loki_handler = CustomLokiLoggingHandler(
        Queue(-1),
        url="https://loki.tensoralchemy.ai/loki/api/v1/push",
        tags={"application": application_name},
        auth=("tensoralchemy-loki", "tPaaDGH0lG"),
        version="1",
    )

    # Send logs to loki as JSON
    loki_handler.setFormatter(JSONFormatter())

    logger.add(loki_handler)


def create_bittensor_logging_wrapper(log_func):
    def bt_log(*args, **kwargs):
        msg = kwargs.get("msg", None)
        prefix = kwargs.get("prefix", None)
        suffix = kwargs.get("suffix", None)

        full_message: str = ""

        if len(args) > 0:
            full_message = str(args[0])

        elif msg is not None:
            full_message = str(msg)

        if prefix:
            full_message = f"{prefix}: {full_message}"

        if suffix:
            full_message = f"{full_message}: {suffix}"

        return log_func(full_message, *args, **kwargs)

    return bt_log


def patch_bt_logging():
    bt.logging.info = create_bittensor_logging_wrapper(logger.info)
    bt.logging.warning = create_bittensor_logging_wrapper(logger.warning)
    bt.logging.error = create_bittensor_logging_wrapper(logger.error)
    bt.logging.debug = create_bittensor_logging_wrapper(logger.debug)
    bt.logging.trace = create_bittensor_logging_wrapper(logger.trace)


def configure_logging():
    if TA_LOGGER_DISABLED:
        logger.warning("TensorAlchemy logger is disabled")
        return

    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    )
    loki_logger_enabled = "--alchemy.disable_loki_logging" not in sys.argv
    if loki_logger_enabled:
        configure_loki_logger()

    patch_bt_logging()
    logger.info("TensorAlchemy logger is enabled")
