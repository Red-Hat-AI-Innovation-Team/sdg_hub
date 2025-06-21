# SPDX-License-Identifier: Apache-2.0
"""Configuration validation utilities for SDG Hub.

This module provides functions to validate configuration files used by blocks,
ensuring they meet the required schema and contain all necessary fields.
"""

# Standard
from typing import Any, Dict, List, Set, Union

# Third Party
from jinja2 import Environment, meta

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
        errors.append(
            f"Configuration must be a dictionary, got {type(config).__name__}"
        )
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
                errors.append(
                    f"Required field '{field}' must be a string, got {type(value).__name__}"
                )
            elif not value.strip():
                errors.append(f"Required field '{field}' is empty")

    # Check optional string fields are strings when present
    string_fields = ["introduction", "principles", "examples"]
    for field in string_fields:
        if field in config:
            value = config[field]
            if value is not None and not isinstance(value, str):
                errors.append(
                    f"Field '{field}' must be a string, got {type(value).__name__}"
                )

    # Check start_tags and end_tags are lists of strings when present
    tag_fields = ["start_tags", "end_tags"]
    for field in tag_fields:
        if field in config:
            value = config[field]
            if value is not None:
                if not isinstance(value, list):
                    errors.append(
                        f"Field '{field}' must be a list, got {type(value).__name__}"
                    )
                else:
                    for i, tag in enumerate(value):
                        if not isinstance(tag, str):
                            errors.append(
                                f"Field '{field}[{i}]' must be a string, got {type(tag).__name__}"
                            )

    # Log validation results
    if errors:
        for error in errors:
            logger.error(f"Config validation failed for {config_path}: {error}")
        return False, errors

    logger.debug(f"Config validation passed for {config_path}")
    return True, []


def validate_jinja_template_variables(
    config_content: Dict[str, Any], available_columns: Set[str]
) -> List[str]:
    """Validate that all Jinja template variables in config are available in dataset columns.

    Parameters
    ----------
    config_content : Dict[str, Any]
        The loaded YAML configuration containing template fields.
    available_columns : Set[str]
        Set of column names available in the dataset.

    Returns
    -------
    List[str]
        List of missing column names referenced in the templates.
        Empty list if all variables are available.

    Raises
    ------
    Exception
        If any template cannot be parsed by Jinja2.
    """
    # Fields that may contain Jinja templates
    template_fields = ["system", "introduction", "principles", "examples", "generation"]
    all_missing_vars = set()

    try:
        env = Environment()

        for field in template_fields:
            if field in config_content and config_content[field]:
                template_content = config_content[field]
                if isinstance(template_content, str):
                    ast = env.parse(template_content)
                    vars_found = meta.find_undeclared_variables(ast)

                    for var in vars_found:
                        if var not in available_columns:
                            all_missing_vars.add(var)

        return list(all_missing_vars)
    except Exception as e:
        logger.error(f"Failed to parse Jinja template: {e}")
        raise


def validate_block_column_requirements(
    block_name: str,
    block_type_name: str,
    config: Dict[str, Any],
    available_columns: Set[str],
) -> List[str]:
    """Validate that a block has all required columns available.

    Parameters
    ----------
    block_name : str
        Name of the block instance for error reporting.
    block_type_name : str
        Name of the block type/class.
    config : Dict[str, Any]
        Block configuration dictionary.
    available_columns : Set[str]
        Set of column names available in the dataset.

    Returns
    -------
    List[str]
        List of validation error messages. Empty if all requirements are met.
    """
    # Define column requirements for each block type
    BLOCK_COLUMN_REQUIREMENTS = {
        "FilterByValueBlock": {"required_fields": ["filter_column"]},
        "SelectorBlock": {
            "required_fields": ["choice_col"],
            "map_fields": {
                "choice_map": "values"
            },  # values of choice_map dict must be columns
        },
        "CombineColumnsBlock": {
            "list_fields": ["columns"]  # all items in 'columns' list must be columns
        },
        "SamplePopulatorBlock": {"required_fields": ["column_name"]},
        "FlattenColumnsBlock": {"list_fields": ["var_cols"]},
        "DuplicateColumns": {
            "map_fields": {"columns_map": "keys"}  # keys of columns_map must be columns
        },
        "RenameColumns": {"map_fields": {"columns_map": "keys"}},
        "SetToMajorityValue": {"required_fields": ["col_name"]},
        "ConditionalLLMBlock": {"required_fields": ["selector_column_name"]},
    }

    errors = []

    # Get requirements for this block type
    requirements = BLOCK_COLUMN_REQUIREMENTS.get(block_type_name, {})

    # Check required single column fields
    for field in requirements.get("required_fields", []):
        col = config.get(field)
        if col and col not in available_columns:
            errors.append(f"[{block_name}] Missing {field}: '{col}'")

    # Check list fields (where all items must be columns)
    for field in requirements.get("list_fields", []):
        cols = config.get(field, [])
        for col in cols:
            if col not in available_columns:
                errors.append(f"[{block_name}] Missing column in {field}: '{col}'")

    # Check map fields (where keys or values must be columns)
    for field, check_type in requirements.get("map_fields", {}).items():
        field_map = config.get(field, {})
        if check_type == "keys":
            for col in field_map.keys():
                if col not in available_columns:
                    errors.append(
                        f"[{block_name}] Missing source column in {field}: '{col}'"
                    )
        elif check_type == "values":
            for choice_val, col in field_map.items():
                if col not in available_columns:
                    errors.append(
                        f"[{block_name}] {field}['{choice_val}'] references missing column: '{col}'"
                    )

    return errors
