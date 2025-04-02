# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC
from collections import ChainMap
from typing import Any, Dict, Union

# Third Party
from jinja2 import Template, UndefinedError
import yaml

# Local
from ..registry import BlockRegistry
from ..logger_config import setup_logger

logger = setup_logger(__name__)


@BlockRegistry.register("Block")
class Block(ABC):
    def __init__(self, block_name: str) -> None:
        self.block_name = block_name

    @staticmethod
    def _validate(prompt_template: Template, input_dict: Dict[str, Any]) -> bool:
        """
        Validate the input data for this block. This method validates whether all required
        variables in the Jinja template are provided in the input_dict.

        :param prompt_template: The Jinja2 template object.
        :param input_dict: A dictionary of input values to check against the template.
        :return: True if the input data is valid (i.e., no missing variables), False otherwise.
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

    def _load_config(self, config_path: str) -> Union[Dict[str, Any], None]:
        """
        Load the configuration file for this block.

        :param config_path: The path to the configuration file.
        :return: The loaded configuration.
        """
        with open(config_path, "r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)
