# SPDX-License-Identifier: Apache-2.0
"""
Logger configuration module for SDG Hub.

This module provides centralized logging configuration using the Rich library
for enhanced console output. It sets up a consistent logging format across
the application with configurable log levels through environment variables.
"""

# Standard
import os
import logging

# Third Party
from rich.logging import RichHandler


def setup_logger(name: str) -> logging.Logger:
    """
    Set up and configure a logger instance with Rich formatting.
    
    This function:
    1. Reads the log level from environment variable LOG_LEVEL (defaults to "INFO")
    2. Configures the root logger with Rich formatting
    3. Creates and returns a named logger instance
    
    The logger uses Rich's enhanced console output features for:
    - Syntax highlighting
    - Better formatting of log messages
    - Improved readability of stack traces
    - Color-coded log levels
    
    Args:
        name (str): Name of the logger to create. Typically __name__ of the
            calling module.
    
    Returns:
        logging.Logger: Configured logger instance with Rich formatting
    
    Environment Variables:
        LOG_LEVEL (str): Sets the logging level. Can be one of:
            - DEBUG
            - INFO
            - WARNING
            - ERROR
            - CRITICAL
            Defaults to "INFO" if not set.
    """
    # Get log level from environment variable, default to INFO
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Configure the root logger with Rich formatting
    logging.basicConfig(
        level=log_level,
        format="%(message)s",  # Rich handles the formatting
        datefmt="[%X]",        # Time format for log entries
        handlers=[RichHandler()],  # Use Rich's enhanced console output
    )
    
    # Create and return a named logger instance
    logger = logging.getLogger(name)
    return logger
