# SPDX-License-Identifier: Apache-2.0
# Standard
import logging
import os

# Third Party
from rich.logging import RichHandler


def setup_logger(name, log_dir=None, log_filename="sdg_hub.log"):
    """
    Set up a logger with optional file logging.

    Parameters
    ----------
    name : str
        Logger name.
    log_dir : str, optional
        Directory to save log files. If None, logs are not saved to file.
    log_filename : str, optional
        Name of the log file (default: "sdg_hub.log").

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Suppress litellm logs to reduce noise
    litellm_log_level = os.getenv("LITELLM_LOG_LEVEL", "WARNING").upper()
    logging.getLogger("litellm").setLevel(litellm_log_level)
    logging.getLogger("litellm.proxy").setLevel(litellm_log_level)
    logging.getLogger("litellm.router").setLevel(litellm_log_level)

    # Prevent duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        # Rich console handler
        rich_handler = RichHandler()
        rich_handler.setLevel(log_level)
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")
        rich_handler.setFormatter(formatter)
        logger.addHandler(rich_handler)

        # Optional file handler
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, log_filename)
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="[%Y-%m-%d %H:%M:%S]",
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    # logger.info(f"Logger setup complete: {name}")
    return logger
