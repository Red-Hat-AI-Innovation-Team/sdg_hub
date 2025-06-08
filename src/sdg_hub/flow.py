"""
Flow module for managing data generation pipelines.

This module provides the core Flow class that handles both configuration loading and execution
of data generation blocks. The Flow class serves as the main interface for defining and running
data generation pipelines, supporting both direct usage with SDG and backward compatibility
through the deprecated Pipeline class.

Example:
    >>> flow = Flow(llm_client)
    >>> flow = flow.get_flow_from_file("path/to/flow.yaml")
    >>> dataset = flow.generate(input_dataset)

Note:
    This module is part of the SDG Hub package and is designed to work in conjunction
    with the SDG class for distributed data generation.
"""

# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC
from importlib import resources
from typing import Optional, List, Dict, Any, Callable
import operator
import os

# Third Party
import yaml
from datasets import Dataset
from datasets.data_files import EmptyDatasetError

# Local
from .blocks import *  # needed to register blocks
from .prompts import *  # needed to register prompts
from .registry import BlockRegistry, PromptRegistry
from .logger_config import setup_logger


logger = setup_logger(__name__)

OPERATOR_MAP: Dict[str, Callable] = {
    "operator.eq": operator.eq,
    "operator.ge": operator.ge,
    "operator.le": operator.le,
    "operator.gt": operator.gt,
    "operator.lt": operator.lt,
    "operator.ne": operator.ne,
    "operator.contains": operator.contains,
}

CONVERT_DTYPE_MAP: Dict[str, Callable] = {
    "float": float,
    "int": int,
}


class Flow(ABC):
    """A class representing a data generation flow.

    This class handles both configuration loading and execution of data generation
    blocks. It can be used directly with SDG or through the deprecated Pipeline class.
    """

    def __init__(
        self,
        llm_client: Any,
        num_samples_to_generate: Optional[int] = None,
    ) -> None:
        """
        Initialize the Flow class.

        Parameters
        ----------
        llm_client : Any
            The LLM client to use for generation.
        num_samples_to_generate : Optional[int], optional
            Number of samples to generate, by default None

        Attributes
        ----------
        llm_client : Any
            The LLM client instance.
        base_path : str
            Base path for resource files.
        registered_blocks : Dict[str, Any]
            Registry of available blocks.
        chained_blocks : Optional[List[Dict[str, Any]]]
            List of block configurations.
        num_samples_to_generate : Optional[int]
            Number of samples to generate.

        """
        self.llm_client = llm_client
        self.base_path = str(resources.files(__package__))
        self.registered_blocks = BlockRegistry.get_registry()
        self.chained_blocks = None  # Will be set by get_flow_from_file
        self.num_samples_to_generate = num_samples_to_generate

    def _getFilePath(self, dirs: List[str], filename: str) -> str:
        """Find a named configuration file.

        Files are checked in the following order:
            1. Absolute path is always used
            2. Checked relative to the directories in "dirs"
            3. Relative to the current directory

        Parameters
        ----------
        dirs : List[str]
            Directories in which to search for the file.
        filename : str
            The path to the configuration file.

        Returns
        -------
        str
            Selected file path.
        """
        if os.path.isabs(filename):
            return filename
        for d in dirs:
            full_file_path = os.path.join(d, filename)
            if os.path.isfile(full_file_path):
                return full_file_path
        # If not found above then return the path unchanged i.e.
        # assume the path is relative to the current directory
        return filename

    def _drop_duplicates(self, dataset: Dataset, cols: List[str]) -> Dataset:
        """Drop duplicates from the dataset based on the columns provided.

        Parameters
        ----------
        dataset : Dataset
            The input dataset.
        cols : List[str]
            Columns to consider for duplicate detection.

        Returns
        -------
        Dataset
            Dataset with duplicates removed.
        """
        df = dataset.to_pandas()
        df = df.drop_duplicates(subset=cols).reset_index(drop=True)
        return Dataset.from_pandas(df)

    def generate(self, dataset: Dataset) -> Dataset:
        """Generate the dataset by running the pipeline steps.

        Parameters
        ----------
        dataset : Dataset
            The input dataset to process.

        Returns
        -------
        Dataset
            The processed dataset.

        Raises
        ------
        ValueError
            If Flow has not been initialized with blocks.
        EmptyDatasetError
            If a block produces an empty dataset.
        """
        if self.chained_blocks is None:
            raise ValueError(
                "Flow has not been initialized with blocks. "
                "Call get_flow_from_file() first. "
                "Or pass a list of blocks to the Flow constructor."
            )

        for block_prop in self.chained_blocks:
            block_type = block_prop["block_type"]
            block_config = block_prop["block_config"]
            drop_columns = block_prop.get("drop_columns", [])
            gen_kwargs = block_prop.get("gen_kwargs", {})
            drop_duplicates_cols = block_prop.get("drop_duplicates", False)
            block = block_type(**block_config)

            logger.debug("------------------------------------\n")
            logger.debug("Running block: %s", block_config["block_name"])
            logger.debug("Input dataset: %s", dataset)

            dataset = block.generate(dataset, **gen_kwargs)

            if len(dataset) == 0:
                raise EmptyDatasetError(
                    f"Pipeline stopped: "
                    f"Empty dataset after running block: "
                    f"{block_config['block_name']}"
                )

            drop_columns_in_ds = [e for e in drop_columns if e in dataset.column_names]
            if drop_columns:
                dataset = dataset.remove_columns(drop_columns_in_ds)

            if drop_duplicates_cols:
                dataset = self._drop_duplicates(dataset, cols=drop_duplicates_cols)

            logger.debug("Output dataset: %s", dataset)
            logger.debug("------------------------------------\n\n")

        return dataset

    def get_flow_from_file(self, yaml_path: str) -> "Flow":
        """Load and initialize flow configuration from a YAML file.

        Parameters
        ----------
        yaml_path : str
            Path to the YAML configuration file.

        Returns
        -------
        Flow
            Self with initialized chained_blocks.

        Raises
        ------
        FileNotFoundError
            If the YAML file cannot be found.
        KeyError
            If a required block or prompt is not found in the registry.
        """
        yaml_path_relative_to_base = os.path.join(self.base_path, yaml_path)
        if os.path.isfile(yaml_path_relative_to_base):
            yaml_path = yaml_path_relative_to_base
        yaml_dir = os.path.dirname(yaml_path)

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
                block["block_config"]["config_path"] = self._getFilePath(
                    [yaml_dir, self.base_path], block["block_config"]["config_path"]
                )

            # update config paths to absolute paths - this might be a list or a dict
            if "config_paths" in block["block_config"]:
                if isinstance(block["block_config"]["config_paths"], dict):
                    for key, path in block["block_config"]["config_paths"].items():
                        block["block_config"]["config_paths"][key] = self._getFilePath(
                            [yaml_dir, self.base_path], path
                        )

                elif isinstance(block["block_config"]["config_paths"], list):
                    for i, path in enumerate(block["block_config"]["config_paths"]):
                        block["block_config"]["config_paths"][i] = self._getFilePath(
                            [yaml_dir, self.base_path], path
                        )

            if "operation" in block["block_config"]:
                block["block_config"]["operation"] = OPERATOR_MAP[
                    block["block_config"]["operation"]
                ]

            if "convert_dtype" in block["block_config"]:
                block["block_config"]["convert_dtype"] = CONVERT_DTYPE_MAP[
                    block["block_config"]["convert_dtype"]
                ]

        # Store the chained blocks and return self
        self.chained_blocks = flow
        return self
