# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC
from importlib import resources
from typing import Optional
import operator
import os

# Third Party
import yaml

# Local
from .registry import BlockRegistry, PromptRegistry
from . import prompts
from . import blocks


OPERATOR_MAP = {
    "operator.eq": operator.eq,
    "operator.ge": operator.ge,
    "operator.contains": operator.contains,
}

CONVERT_DTYPE_MAP = {
    "float": float,
    "int": int,
}


class Flow(ABC):
    def __init__(
        self,
        llm_client,
        num_samples_to_generate: Optional[int] = None,
    ) -> None:
        self.llm_client = llm_client
        self.num_samples_to_generate = num_samples_to_generate
        self.base_path = str(resources.files(__package__))
        self.registered_blocks = BlockRegistry.get_registry()

    def get_flow_from_file(self, yaml_path: str) -> list:
        yaml_path_relative_to_base = os.path.join(self.base_path, yaml_path)
        if os.path.isfile(yaml_path_relative_to_base):
            yaml_path = yaml_path_relative_to_base

        try:
            with open(yaml_path, "r", encoding="utf-8") as yaml_file:
                flow = yaml.safe_load(yaml_file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File not found: {yaml_path}") from exc

        # update config with class instances
        for block in flow:
            # check if theres an llm block in the flow
            if "LLM" in block["block_type"]:
                block["block_config"]["client"] = self.llm_client
                # model_id and prompt templates
                # try to get a template using the model_id, but if model_prompt_template is provided, use that
                if block["block_config"].get("model_prompt", None) is None:
                    # try to find a match in the registry
                    matched_prompt = next(
                        (
                            key
                            for key in PromptRegistry.get_registry()
                            if key in block["block_config"]["model_id"]
                        ),
                        None,
                    )
                    if matched_prompt is not None:
                        block["block_config"]["model_prompt"] = matched_prompt
                    else:
                        raise KeyError(
                            f"Prompt not found in registry: {block['block_config']['model_id']}"
                        )

                if self.num_samples_to_generate is not None:
                    block["num_samples"] = self.num_samples_to_generate

            # update block type to llm class instance
            try:
                block["block_type"] = self.registered_blocks[block["block_type"]]
            except KeyError as exc:
                raise KeyError(
                    f"Block not found in registry: {block['block_type']}"
                ) from exc

            # update config path to absolute path
            if "config_path" in block["block_config"]:
                config_path_relative_to_base = os.path.join(
                    self.base_path, block["block_config"]["config_path"]
                )
                if os.path.isfile(config_path_relative_to_base):
                    block["block_config"]["config_path"] = config_path_relative_to_base

            # update config paths to absolute paths - this might be a list or a dict
            if "config_paths" in block["block_config"]:
                if isinstance(block["block_config"]["config_paths"], dict):
                    for key, path in block["block_config"]["config_paths"].items():
                        config_path_relative_to_base = os.path.join(
                            self.base_path, path
                        )
                        if os.path.isfile(config_path_relative_to_base):
                            block["block_config"]["config_paths"][key] = (
                                config_path_relative_to_base
                            )

                if isinstance(block["block_config"]["config_paths"], list):
                    for i, path in enumerate(block["block_config"]["config_paths"]):
                        config_path_relative_to_base = os.path.join(
                            self.base_path, path
                        )
                        if os.path.isfile(config_path_relative_to_base):
                            block["block_config"]["config_paths"][i] = (
                                config_path_relative_to_base
                            )

            if "operation" in block["block_config"]:
                block["block_config"]["operation"] = OPERATOR_MAP[
                    block["block_config"]["operation"]
                ]

            if "convert_dtype" in block["block_config"]:
                block["block_config"]["convert_dtype"] = CONVERT_DTYPE_MAP[
                    block["block_config"]["convert_dtype"]
                ]

        return flow
