import os
import logging
from rich.logging import RichHandler


def setup_logger(name):
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler()
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.propagate = False
    return logger
