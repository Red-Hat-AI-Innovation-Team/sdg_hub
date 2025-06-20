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
from typing import Any, Callable, Dict, List, Optional
import operator
import os

# Third Party
from datasets import Dataset
from datasets.data_files import EmptyDatasetError
from jinja2 import Environment, meta
from rich.console import Console
from rich.table import Table
import yaml

# Local
from .blocks import *  # needed to register blocks
from .logger_config import setup_logger
from .prompts import *  # needed to register prompts
from .registry import BlockRegistry, PromptRegistry
from .utils.config_validation import validate_prompt_config_schema
from .utils.validation_result import ValidationResult

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
        log_level: Optional[str] = None,
    ) -> None:
        """
        Initialize the Flow class.

        Parameters
        ----------
        llm_client : Any
            The LLM client to use for generation.
        num_samples_to_generate : Optional[int], optional
            Number of samples to generate, by default None
        log_level : Optional[str], optional
            Logging verbosity level, by default None

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

        # Logging verbosity level
        self.log_level = log_level or os.getenv("SDG_HUB_LOG_LEVEL", "normal").lower()
        self.console = Console() if self.log_level in ["verbose", "debug"] else None

    def _log_block_info(
        self, index: int, total: int, name: str, ds: Dataset, stage: str
    ) -> None:
        if self.log_level in ["verbose", "debug"] and self.console:
            table = Table(
                title=f"{stage} Block {index + 1}/{total}: {name}", show_header=True
            )
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="magenta")
            table.add_row("Rows", str(len(ds)))
            table.add_row("Columns", ", ".join(ds.column_names))
            self.console.print(table)

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

        for i, block_prop in enumerate(self.chained_blocks):
            block_type = block_prop["block_type"]
            block_config = block_prop["block_config"]
            drop_columns = block_prop.get("drop_columns", [])
            gen_kwargs = block_prop.get("gen_kwargs", {})
            drop_duplicates_cols = block_prop.get("drop_duplicates", False)
            block = block_type(**block_config)

            name = block_config.get("block_name", f"block_{i}")

            # Logging: always show basic progress unless in quiet mode
            if self.log_level in ["normal", "verbose", "debug"]:
                logger.info(
                    f"🔄 Running block {i + 1}/{len(self.chained_blocks)}: {name}"
                )

            # Log dataset shape before block (verbose/debug)
            self._log_block_info(i, len(self.chained_blocks), name, dataset, "Input")

            if self.log_level == "debug":
                logger.debug(f"Input dataset (truncated): {dataset}")

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

            # Log dataset shape after block (verbose/debug)
            self._log_block_info(i, len(self.chained_blocks), name, dataset, "Output")

            if self.log_level == "debug":
                logger.debug(f"Output dataset (truncated): {dataset}")

        return dataset

    def validate_config_files(self) -> "ValidationResult":
        """
        Validate all configuration file paths referenced in the flow blocks.

        This method checks that all config files specified via `config_path` or `config_paths`
        in each block:
            - Exist on the filesystem
            - Are readable by the current process
            - Are valid YAML files (optional format check)

        Returns
        -------
        ValidationResult
            An object indicating whether all config files passed validation, along with a list
            of error messages for any missing, unreadable, or invalid YAML files.

        Notes
        -----
        This method is automatically called at the end of `get_flow_from_file()` to ensure
        early detection of misconfigured blocks.
        """
        errors = []

        def check_file(path: str, context: str):
            if not os.path.isfile(path):
                errors.append(f"[{context}] File does not exist: {path}")
            else:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        config_data = yaml.safe_load(f)
                        _, validation_errors = validate_prompt_config_schema(config_data, path)

                        if validation_errors:
                            errors.extend(validation_errors)

                except PermissionError:
                    errors.append(f"[{context}] File is not readable: {path}")
                except yaml.YAMLError as e:
                    errors.append(f"[{context}] YAML load failed: {path} ({e})")

        for i, block in enumerate(self.chained_blocks or []):
            block_name = block["block_config"].get("block_name", f"block_{i}")

            config_path = block["block_config"].get("config_path")
            if config_path:
                check_file(config_path, f"{block_name}.config_path")

            config_paths = block["block_config"].get("config_paths")
            if isinstance(config_paths, list):
                for idx, path in enumerate(config_paths):
                    check_file(path, f"{block_name}.config_paths[{idx}]")
            elif isinstance(config_paths, dict):
                for key, path in config_paths.items():
                    check_file(path, f"{block_name}.config_paths['{key}']")

        return ValidationResult(valid=(len(errors) == 0), errors=errors)

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

        # Validate config files
        result = self.validate_config_files()
        if not result.valid:
            raise ValueError("Invalid config files:\n\n".join(result.errors))

        return self

    def validate_flow(self, dataset: Dataset) -> "ValidationResult":
        """
        Validate that all required dataset columns are present before executing the flow.

        This includes:
        - Columns referenced in Jinja templates for LLM blocks
        - Columns required by specific utility blocks (e.g. filter_column, choice_col, etc.)

        Parameters
        ----------
        dataset : Dataset
            The input dataset to validate against.

        Returns
        -------
        ValidationResult
            Whether the dataset has all required columns, and which ones are missing.
        """
        errors = []
        all_columns = set(dataset.column_names)

        for i, block in enumerate(self.chained_blocks or []):
            name = block["block_config"].get("block_name", f"block_{i}")
            block_type = block["block_type"]
            config = block["block_config"]

            # LLM Block: parse Jinja vars
            cls_name = block_type.__name__ if isinstance(block_type, type) else block_type.__class__.__name__
            logger.info(f"Validating block: {name} ({cls_name})")
            if "LLM" in cls_name:
                config_path = config.get("config_path")
                if config_path and os.path.isfile(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        env = Environment()
                        ast = env.parse(content)
                        vars_found = meta.find_undeclared_variables(ast)
                        for var in vars_found:
                            if var not in all_columns:
                                errors.append(f"[{name}] Missing column for prompt var: '{var}'")

            # FilterByValueBlock
            if "FilterByValueBlock" in str(block_type):
                col = config.get("filter_column")
                if col and col not in all_columns:
                    errors.append(f"[{name}] Missing filter_column: '{col}'")

            # SelectorBlock
            if "SelectorBlock" in str(block_type):
                col = config.get("choice_col")
                if col and col not in all_columns:
                    errors.append(f"[{name}] Missing choice_col: '{col}'")

                choice_map = config.get("choice_map", {})
                for col in choice_map.values():
                    if col not in all_columns:
                        errors.append(f"[{name}] choice_map references missing column: '{col}'")

            # CombineColumnsBlock
            if "CombineColumnsBlock" in str(block_type):
                cols = config.get("columns", [])
                for col in cols:
                    if col not in all_columns:
                        errors.append(f"[{name}] CombineColumnsBlock requires column: '{col}'")

        return ValidationResult(valid=(len(errors) == 0), errors=errors)