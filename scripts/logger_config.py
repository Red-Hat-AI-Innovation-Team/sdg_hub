# SPDX-License-Identifier: Apache-2.0
# Standard
import logging

# Third Party
from rich.logging import RichHandler


def setup_logger(name):
    # Set up the logger
    logging.basicConfig(
        level=logging.ERROR,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    logger = logging.getLogger(name)
    return logger
