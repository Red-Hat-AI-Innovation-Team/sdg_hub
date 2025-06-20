# SPDX-License-Identifier: Apache-2.0
"""Configuration validation utilities for SDG Hub.

This module provides functions to validate configuration files used by blocks,
ensuring they meet the required schema and contain all necessary fields.
"""

# Standard
from typing import Any, Dict, List

# Local
from ..logger_config import setup_logger

logger = setup_logger(__name__)


def validate_prompt_config_schema(
    config: Dict[str, Any], config_path: str
) -> tuple[bool, List[str]]:
    """Validate that a prompt configuration file has the required schema fields.

    For prompt template configs, 'system' and 'generation' are required fields.
    Other fields like 'introduction', 'principles', 'examples', 'start_tags', 'end_tags' are optional.

    Parameters
    ----------
    config : Dict[str, Any]
        The loaded configuration dictionary.
    config_path : str
        The path to the configuration file (for error reporting).

    Returns
    -------
    tuple[bool, List[str]]
        A tuple containing:
        - bool: True if schema is valid, False otherwise
        - List[str]: List of validation error messages (empty if valid)
    """
    required_fields = ["system", "generation"]
    errors = []

    # Ensure config is a dictionary
    if not isinstance(config, dict):
        errors.append(f"Configuration must be a dictionary, got {type(config).__name__}")
        return False, errors
    
    # Check for missing required fields
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        errors.append(f"Missing required fields: {missing_fields}")

    # Check for empty or null required fields and validate they are strings
    for field in required_fields:
        if field in config:
            value = config[field]
            if value is None:
                errors.append(f"Required field '{field}' is null")
            elif not isinstance(value, str):
                errors.append(f"Required field '{field}' must be a string, got {type(value).__name__}")
            elif not value.strip():
                errors.append(f"Required field '{field}' is empty")

    # Check optional string fields are strings when present
    string_fields = ["introduction", "principles", "examples"]
    for field in string_fields:
        if field in config:
            value = config[field]
            if value is not None and not isinstance(value, str):
                errors.append(f"Field '{field}' must be a string, got {type(value).__name__}")

    # Check start_tags and end_tags are lists of strings when present
    tag_fields = ["start_tags", "end_tags"]
    for field in tag_fields:
        if field in config:
            value = config[field]
            if value is not None:
                if not isinstance(value, list):
                    errors.append(f"Field '{field}' must be a list, got {type(value).__name__}")
                else:
                    for i, tag in enumerate(value):
                        if not isinstance(tag, str):
                            errors.append(f"Field '{field}[{i}]' must be a string, got {type(tag).__name__}")

    # Log validation results
    if errors:
        for error in errors:
            logger.error(f"Config validation failed for {config_path}: {error}")
        return False, errors

    logger.debug(f"Config validation passed for {config_path}")
    return True, []
