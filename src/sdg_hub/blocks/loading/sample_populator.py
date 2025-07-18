# SPDX-License-Identifier: Apache-2.0
"""Sample populator block for enriching datasets with configuration data.

This module provides a block for populating datasets with data from configuration files,
allowing for dynamic enrichment of samples based on key-value lookups.
"""

# Standard
from typing import Any, Dict, List, Optional
import os

# Third Party
from datasets import Dataset
from pydantic import Field, field_validator
import yaml

# Local
from ...logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "SamplePopulatorBlock",
    "loading",
    "Populates datasets with data from configuration files based on key-value lookups",
)
class SamplePopulatorBlock(BaseBlock):
    """Block for populating dataset with data from configuration files.

    This block reads data from one or more configuration files and populates a
    dataset with the data. The data is stored in a dictionary, with the keys
    being the names of the configuration files.

    The input_cols should specify the column to use as the lookup key.
    The output_cols can be specified to control which columns are added, or left
    empty to add all columns from the configuration files.

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Input column name(s). Must specify exactly one column for lookup.
    output_cols : Union[str, List[str], None]
        Output column specification. If None, all config data is added.
    config_paths : List[str]
        List of paths to configuration files to load.
    post_fix : str
        Suffix to append to configuration filenames.
    """

    config_paths: List[str] = Field(
        ..., description="List of paths to configuration files to load"
    )
    post_fix: str = Field(
        default="", description="Suffix to append to configuration filenames"
    )

    # Internal fields for loaded configurations
    configs: Optional[Dict[str, Any]] = Field(
        None, description="Loaded configuration data", exclude=True
    )

    @field_validator("input_cols", mode="after")
    @classmethod
    def validate_input_cols_single(cls, v):
        """Validate that exactly one input column is specified."""
        if not v or len(v) != 1:
            raise ValueError("SamplePopulatorBlock requires exactly one input column")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Initialize derived attributes after Pydantic validation."""
        super().model_post_init(__context) if hasattr(super(), "model_post_init") else None
        
        # Load configuration files
        self.configs = {}
        for config_path in self.config_paths:
            if self.post_fix:
                config_name = config_path.replace(".yaml", f"_{self.post_fix}.yaml")
            else:
                config_name = config_path
            
            config_key = os.path.basename(config_path).split(".")[0]
            self.configs[config_key] = self._load_config(config_name)
        
        # Set derived attributes
        self.column_name = self.input_cols[0]  # Use first (and only) input column
        
        # Validate schema consistency across configurations
        valid_configs = {k: v for k, v in self.configs.items() if v is not None}
        if len(valid_configs) >= 2:
            # Get all keys from all configs
            all_keys = set()
            config_keys = {}
            for config_name, config_data in valid_configs.items():
                keys = set(config_data.keys())
                config_keys[config_name] = keys
                all_keys.update(keys)
            
            # Check for missing keys in each config
            warnings = []
            for config_name, keys in config_keys.items():
                missing_keys = all_keys - keys
                if missing_keys:
                    warnings.append(f"Config '{config_name}' is missing keys: {sorted(missing_keys)}")
            
            if warnings:
                logger.warning(
                    f"Schema inconsistencies detected in {self.block_name} configs:\n" +
                    "\n".join(f"  - {w}" for w in warnings) +
                    "\nThis may cause dataset schema conflicts. Consider standardizing your config files."
                )

    def _load_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """Load a configuration file.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.

        Returns
        -------
        Optional[Dict[str, Any]]
            Loaded configuration data or None if loading fails.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            return None
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading config {config_path}: {e}")
            return None

    def _validate_custom(self, dataset: Dataset) -> None:
        """Validate that all lookup keys in the dataset have corresponding configs.

        Parameters
        ----------
        dataset : Dataset
            Input dataset to validate.

        Raises
        ------
        ValueError
            If any lookup keys are missing from the loaded configurations.
        """
        # Get unique lookup keys from the dataset
        unique_keys = set(dataset[self.column_name])
        
        # Check which keys don't have corresponding configs
        missing_configs = []
        for key in unique_keys:
            if key not in self.configs:
                missing_configs.append(key)
            elif self.configs[key] is None:
                missing_configs.append(f"{key} (config failed to load)")
        
        if missing_configs:
            available_configs = [k for k, v in self.configs.items() if v is not None]
            raise ValueError(
                f"Missing configurations for lookup keys: {missing_configs}. "
                f"Available configs: {available_configs}. "
                f"Please ensure all values in column '{self.column_name}' have corresponding YAML files."
            )

    def _generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new sample by populating it with configuration data.

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample to populate with configuration data.

        Returns
        -------
        Dict[str, Any]
            Sample populated with configuration data.

        Raises
        ------
        KeyError
            If the lookup key is not found in the configuration.
        TypeError
            If the configuration data is None (failed to load).
        """
        lookup_key = sample[self.column_name]
        
        if lookup_key not in self.configs:
            raise KeyError(f"Lookup key '{lookup_key}' not found in configurations")
        
        config_data = self.configs[lookup_key]
        if config_data is None:
            raise TypeError(f"Configuration data for '{lookup_key}' is None (failed to load)")
        
        # Merge configuration data with sample
        return {**sample, **config_data}

    def generate(self, samples: Dataset) -> Dataset:
        """Generate a new dataset with populated configuration data.

        Parameters
        ----------
        samples : Dataset
            Input dataset to populate with configuration data.

        Returns
        -------
        Dataset
            Dataset populated with configuration data.
        """
        # Use map for processing
        samples = samples.map(self._generate)
        return samples