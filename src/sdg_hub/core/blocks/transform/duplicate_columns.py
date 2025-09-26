# SPDX-License-Identifier: Apache-2.0
"""Duplicate columns block for dataset column duplication operations.

This module provides a block for duplicating existing columns with new names
according to a mapping specification.
"""

# Standard
from typing import Any

# Third Party
from datasets import Dataset
from pydantic import field_validator

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "DuplicateColumnsBlock",
    "transform",
    "Duplicates existing columns with new names according to a mapping specification",
)
class DuplicateColumnsBlock(BaseBlock):
    """Block for duplicating existing columns with new names.

    This block creates copies of existing columns with new names according to a mapping specification.
    The mapping is provided through input_cols as a dictionary.

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : Dict[str, str]
        Dictionary mapping existing column names to new column names.
        Keys are existing column names, values are new column names.
    """

    @field_validator("input_cols", mode="after")
    @classmethod
    def validate_input_cols(cls, v):
        """Validate that input_cols is a non-empty dict."""
        if not v:
            raise ValueError("input_cols cannot be empty")
        if not isinstance(v, dict):
            raise ValueError(
                "input_cols must be a dictionary mapping existing column names to new column names"
            )
        return v

    def model_post_init(self, __context: Any) -> None:
        """Initialize derived attributes after Pydantic validation."""
        super().model_post_init(__context) if hasattr(
            super(), "model_post_init"
        ) else None

        # Set output_cols to the new column names being created
        if self.output_cols is None:
            self.output_cols = list(self.input_cols.values())

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate a dataset with duplicated columns.

        Parameters
        ----------
        samples : Dataset
            Input dataset to duplicate columns from.

        Returns
        -------
        Dataset
            Dataset with additional duplicated columns.
        """
        # Create a copy to avoid modifying the original
        result = samples

        # Duplicate each column as specified in the mapping
        for source_col, target_col in self.input_cols.items():
            if source_col not in result.column_names:
                raise ValueError(f"Source column '{source_col}' not found in dataset")

            result = result.add_column(target_col, result[source_col])

        return result
