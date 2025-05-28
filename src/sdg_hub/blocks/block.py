# SPDX-License-Identifier: Apache-2.0
"""Base block implementation for the SDG Hub system.

This module provides the abstract base class for all blocks in the system,
including functionality for template validation and configuration management.
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
    """
    
    def __init__(self, block_name: str) -> None:
        self.block_name = block_name

    @staticmethod
    def _validate(prompt_template: Template, input_dict: Dict[str, Any]) -> bool:
        """Validate the input data for this block.

        This method validates whether all required variables in the Jinja template are provided in the input_dict.

        Parameters
        ----------
        prompt_template : Template
            The Jinja2 template object.
        input_dict : Dict[str, Any]
            A dictionary of input values to check against the template.

        Returns
        -------
        bool
            True if the input data is valid (i.e., no missing variables), False otherwise.
        """

        class Default(dict):
            def __missing__(self, key: str) -> None:
                raise KeyError(key)

        try:
            # Try rendering the template with the input_dict
            prompt_template.render(ChainMap(input_dict, Default()))
            return True
        except UndefinedError as e:
            logger.error(f"Missing key: {e}")
            return False

    def _load_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """Load the configuration file for this block.

        Parameters
        ----------
        config_path : str
            The path to the configuration file.

        Returns
        -------
        Optional[Dict[str, Any]]
            The loaded configuration. Returns None if file cannot be read or parsed.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
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
