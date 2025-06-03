# SPDX-License-Identifier: Apache-2.0
"""Base block implementation for the SDG Hub system.

This module provides the abstract base class for all blocks in the system,
including functionality for template validation and configuration management.
Blocks are the fundamental building blocks of the SDG Hub pipeline, each
responsible for a specific data processing or transformation task.

Registered Block Name: "Block"

The Block class provides common functionality for:
- Template validation using Jinja2 with detailed error reporting
- Configuration loading from YAML files with error handling
- Basic block registration and identification
- Standardized interface for block operations
- Common utilities for data processing and transformation

Key Features:
- Abstract base class that enforces a consistent interface across all blocks
- Built-in support for Jinja2 template validation with custom error handling
- YAML configuration management with robust error handling
- Integration with the BlockRegistry for block discovery and management
- Logging support for debugging and monitoring
- Type hints and comprehensive documentation

The Block class serves as the foundation for all data processing components in the
SDG Hub system, ensuring consistent behavior and interface across different block types.
"""

# Standard
from abc import ABC
from collections import ChainMap
from typing import Any, Dict, Optional

# Third Party
from jinja2 import Template, UndefinedError
import yaml

# Local
from ..registry import BlockRegistry
from ..logger_config import setup_logger

logger = setup_logger(__name__)


@BlockRegistry.register("Block")
class Block(ABC):
    """Base abstract class for all blocks in the system.

    This class provides common functionality for block validation and configuration loading.
    All specific block implementations should inherit from this class.

    The Block class serves as the foundation for all data processing components in the
    SDG Hub system. It provides standardized methods for:
    - Template validation to ensure all required variables are present
    - Configuration loading from YAML files
    - Basic block identification and registration

    Parameters
    ----------
    block_name : str
        A unique identifier for this block instance.
    """

    def __init__(self, block_name: str) -> None:
        """Initialize a new Block instance.

        Parameters
        ----------
        block_name : str
            A unique identifier for this block instance.
        """
        self.block_name = block_name

    @staticmethod
    def _validate(prompt_template: Template, input_dict: Dict[str, Any]) -> bool:
        """Validate the input data for this block.

        This method validates whether all required variables in the Jinja template
        are provided in the input_dict. It uses a custom dictionary class to raise
        KeyError for missing variables, which is then caught to determine validity.

        Parameters
        ----------
        prompt_template : Template
            The Jinja2 template object to validate against.
        input_dict : Dict[str, Any]
            A dictionary of input values to check against the template.

        Returns
        -------
        bool
            True if the input data is valid (i.e., no missing variables), False otherwise.
        """

        class Default(dict):
            """Custom dictionary that raises KeyError for missing keys.
            
            This is used to detect missing template variables during validation.
            """
            def __missing__(self, key: str) -> None:
                raise KeyError(key)

        try:
            # Try rendering the template with the input_dict
            # ChainMap ensures input_dict values take precedence over Default
            prompt_template.render(ChainMap(input_dict, Default()))
            return True
        except UndefinedError as e:
            logger.error(f"Missing key: {e}")
            return False

    def _load_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """Load the configuration file for this block.

        This method reads and parses a YAML configuration file, handling various
        potential errors that might occur during the process.

        Parameters
        ----------
        config_path : str
            The path to the configuration file to load.

        Returns
        -------
        Optional[Dict[str, Any]]
            The loaded configuration as a dictionary, or None if loading fails.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist at the specified path.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                try:
                    return yaml.safe_load(config_file)
                except yaml.YAMLError as e:
                    logger.error(f"Error parsing YAML from {config_path}: {e}")
                    return None
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading config file {config_path}: {e}")
            return None
