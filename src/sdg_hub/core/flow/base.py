# SPDX-License-Identifier: Apache-2.0
"""Pydantic-based Flow class for managing data generation pipelines."""

# Standard
from typing import Any, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
import datasets

# Third Party
import pandas as pd

# Local
from ..blocks.base import BaseBlock
from ..utils.logger_config import setup_logger
from .metadata import DatasetRequirements, FlowMetadata

logger = setup_logger(__name__)


class Flow(BaseModel):
    """Pydantic-based flow for chaining data generation blocks.

    A Flow represents a complete data generation pipeline with proper validation,
    metadata tracking, and execution capabilities. All configuration is validated
    using Pydantic models for type safety and better error messages.

    Attributes
    ----------
    blocks : List[BaseBlock]
        Ordered list of blocks to execute in the flow.
    metadata : FlowMetadata
        Flow metadata including name, version, author, etc.
    """

    blocks: list[BaseBlock] = Field(
        default_factory=list,
        description="Ordered list of blocks to execute in the flow",
    )
    metadata: FlowMetadata = Field(
        description="Flow metadata including name, version, author, etc."
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Private attributes (not serialized)
    _model_config_set: bool = PrivateAttr(default=False)
    _agent_config_set: bool = PrivateAttr(default=False)
    _block_metrics: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    @field_validator("blocks")
    @classmethod
    def validate_blocks(cls, v: list[BaseBlock]) -> list[BaseBlock]:
        """Validate that all blocks are BaseBlock instances."""
        if not v:
            return v

        for i, block in enumerate(v):
            if not isinstance(block, BaseBlock):
                raise ValueError(
                    f"Block at index {i} is not a BaseBlock instance: {type(block)}"
                )

        return v

    @model_validator(mode="after")
    def validate_block_names_unique(self) -> "Flow":
        """Ensure all block names are unique within the flow."""
        if not self.blocks:
            return self

        seen_names = set()
        for i, block in enumerate(self.blocks):
            if block.block_name in seen_names:
                raise ValueError(
                    f"Duplicate block name '{block.block_name}' at index {i}. "
                    f"All block names must be unique within a flow."
                )
            seen_names.add(block.block_name)

        return self

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Flow":
        """Load flow from YAML configuration file."""
        from .serialization import load_flow_from_yaml

        return load_flow_from_yaml(cls, yaml_path)

    @staticmethod
    def _convert_to_dataframe(
        dataset: Union[pd.DataFrame, datasets.Dataset],
    ) -> tuple[pd.DataFrame, bool]:
        """Convert datasets.Dataset to pd.DataFrame if needed. Returns (df, was_dataset)."""
        from .execution import convert_to_dataframe

        return convert_to_dataframe(dataset)

    @staticmethod
    def _convert_from_dataframe(
        df: pd.DataFrame, should_convert: bool
    ) -> Union[pd.DataFrame, datasets.Dataset]:
        """Convert pd.DataFrame back to datasets.Dataset if should_convert is True."""
        from .execution import convert_from_dataframe

        return convert_from_dataframe(df, should_convert)

    def generate(
        self,
        dataset: Union[pd.DataFrame, datasets.Dataset],
        runtime_params: Optional[dict[str, dict[str, Any]]] = None,
        checkpoint_dir: Optional[str] = None,
        save_freq: Optional[int] = None,
        log_dir: Optional[str] = None,
        max_concurrency: Optional[int] = None,
    ) -> Union[pd.DataFrame, datasets.Dataset]:
        """Execute the flow blocks in sequence to generate data.

        For flows with LLM blocks, set_model_config() must be called first.

        See execution.execute_flow for full parameter documentation.
        """
        from .execution import execute_flow

        return execute_flow(
            self,
            dataset,
            runtime_params,
            checkpoint_dir,
            save_freq,
            log_dir,
            max_concurrency,
        )

    def set_model_config(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        blocks: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Configure model settings for LLM blocks in this flow (in-place).

        Auto-detects LLM blocks and applies configuration. See model_config.set_model_config
        for full parameter documentation.
        """
        from .model_config import set_model_config

        set_model_config(self, model, api_base, api_key, blocks, **kwargs)

    def _detect_llm_blocks(self) -> list[str]:
        """Detect blocks with block_type='llm'. Returns list of block names."""
        from .model_config import detect_llm_blocks

        return detect_llm_blocks(self)

    def is_model_config_required(self) -> bool:
        """Check if model configuration is required (True if flow has LLM blocks)."""
        from .model_config import is_model_config_required

        return is_model_config_required(self)

    def is_model_config_set(self) -> bool:
        """Check if model configuration has been set or is not required."""
        from .model_config import is_model_config_set

        return is_model_config_set(self)

    def reset_model_config(self) -> None:
        """Reset model configuration flag (useful for testing or reconfiguration).

        After calling this, set_model_config() must be called again before generate().
        """
        from .model_config import reset_model_config

        reset_model_config(self)

    def get_default_model(self) -> Optional[str]:
        """Get the default recommended model for this flow, or None if unspecified."""
        from .model_config import get_default_model

        return get_default_model(self)

    def get_model_recommendations(self) -> dict[str, Any]:
        """Get model recommendations dict with 'default', 'compatible', 'experimental' keys."""
        from .model_config import get_model_recommendations

        return get_model_recommendations(self)

    def set_agent_config(
        self,
        agent_framework: Optional[str] = None,
        agent_url: Optional[str] = None,
        agent_api_key: Optional[str] = None,
        blocks: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Configure agent settings for agent blocks in this flow (in-place).

        Auto-detects agent blocks and applies configuration. See agent_config.set_agent_config
        for full parameter documentation.
        """
        from .agent_config import set_agent_config

        set_agent_config(
            self, agent_framework, agent_url, agent_api_key, blocks, **kwargs
        )

    def _detect_agent_blocks(self) -> list[str]:
        """Detect blocks with block_type='agent'. Returns list of block names."""
        from .agent_config import detect_agent_blocks

        return detect_agent_blocks(self)

    def is_agent_config_required(self) -> bool:
        """Check if agent configuration is required (True if flow has agent blocks)."""
        from .agent_config import is_agent_config_required

        return is_agent_config_required(self)

    def is_agent_config_set(self) -> bool:
        """Check if agent configuration has been set or is not required."""
        from .agent_config import is_agent_config_set

        return is_agent_config_set(self)

    def reset_agent_config(self) -> None:
        """Reset agent configuration flag (useful for testing or reconfiguration).

        After calling this, set_agent_config() must be called again before generate().
        """
        from .agent_config import reset_agent_config

        reset_agent_config(self)

    def validate_dataset(
        self, dataset: Union[pd.DataFrame, datasets.Dataset]
    ) -> list[str]:
        """Validate dataset against flow requirements. Returns list of error messages."""
        from .execution import validate_flow_dataset

        return validate_flow_dataset(self, dataset)

    def dry_run(
        self,
        dataset: Union[pd.DataFrame, datasets.Dataset],
        sample_size: int = 2,
        runtime_params: Optional[dict[str, dict[str, Any]]] = None,
        max_concurrency: Optional[int] = None,
        enable_time_estimation: bool = False,
    ) -> dict[str, Any]:
        """Perform a dry run of the flow with a subset of data.

        See execution.run_dry_run for full parameter documentation.
        """
        from .execution import run_dry_run

        return run_dry_run(
            self,
            dataset,
            sample_size,
            runtime_params,
            max_concurrency,
            enable_time_estimation,
        )

    def add_block(self, block: BaseBlock) -> "Flow":
        """Add a block to the flow, returning a new Flow instance."""
        if not isinstance(block, BaseBlock):
            raise ValueError(f"Block must be a BaseBlock instance, got: {type(block)}")

        # Check for name conflicts
        existing_names = {b.block_name for b in self.blocks}
        if block.block_name in existing_names:
            raise ValueError(
                f"Block name '{block.block_name}' already exists in flow. "
                f"Block names must be unique."
            )

        # Create new flow with added block
        new_blocks = self.blocks + [block]

        return Flow(blocks=new_blocks, metadata=self.metadata)

    def get_info(self) -> dict[str, Any]:
        """Get information about the flow."""
        from .display import get_flow_info

        return get_flow_info(self)

    def get_dataset_requirements(self) -> Optional[DatasetRequirements]:
        """Get the dataset requirements for this flow, or None if not defined."""
        from .display import get_dataset_requirements

        return get_dataset_requirements(self)

    def get_dataset_schema(self) -> pd.DataFrame:
        """Get an empty DataFrame with the correct schema for this flow."""
        from .display import get_dataset_schema

        return get_dataset_schema(self)

    def print_info(self) -> None:
        """Print an interactive summary of the Flow in the console using rich."""
        from .display import print_flow_info

        print_flow_info(self)

    def to_yaml(self, output_path: str) -> None:
        """Save flow configuration to YAML file.

        Note: This creates a basic YAML structure. For exact reproduction
        of original YAML, save the original file separately.
        """
        from .serialization import save_flow_to_yaml

        save_flow_to_yaml(self, output_path)

    def __len__(self) -> int:
        """Number of blocks in the flow."""
        return len(self.blocks)

    def __repr__(self) -> str:
        """String representation of the flow."""
        return (
            f"Flow(name='{self.metadata.name}', "
            f"version='{self.metadata.version}', "
            f"blocks={len(self.blocks)})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        block_names = [block.block_name for block in self.blocks]
        return (
            f"Flow '{self.metadata.name}' v{self.metadata.version}\n"
            f"Blocks: {' -> '.join(block_names) if block_names else 'None'}\n"
            f"Author: {self.metadata.author or 'Unknown'}\n"
            f"Description: {self.metadata.description or 'No description'}"
        )
