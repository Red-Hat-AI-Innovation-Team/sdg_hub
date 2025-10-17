# SPDX-License-Identifier: Apache-2.0
"""Text concatenation block for dataset column combination operations.

This module provides a block for combining multiple columns into a single column
using a specified separator.
"""

# Standard
from typing import Any

from pydantic import Field, field_validator

# Third Party
import pandas as pd

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "TextConcatBlock",
    "transform",
    "Combines multiple columns into a single column using a specified separator",
)
class TextConcatBlock(BaseBlock):
    """Block for combining multiple columns into a single column.

    This block concatenates values from multiple columns into a single output column,
    using a specified separator between values.

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : list[str]
        List of column names to combine.
    output_cols : list[str]
        List containing the single output column name.
    separator : str
        String to use as separator between combined values.
    """

    separator: str = Field(
        default="\n\n", description="Separator to use between combined values"
    )

    @field_validator("input_cols", mode="after")
    @classmethod
    def validate_input_cols(cls, v):
        """Validate that input_cols is a non-empty list."""
        if not v:
            raise ValueError("input_cols cannot be empty")
        if not isinstance(v, list):
            raise ValueError("input_cols must be a list of column names")
        return v

    @field_validator("output_cols", mode="after")
    @classmethod
    def validate_output_cols(cls, v):
        """Validate that exactly one output column is specified."""
        if not v or len(v) != 1:
            raise ValueError("TextConcatBlock requires exactly one output column")
        return v

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate a dataset with combined columns.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset to process.

        Returns
        -------
        pd.DataFrame
            Dataset with combined values stored in output column.
        """
        if not self.output_cols:
            raise ValueError("output_cols must be specified")

        output_col = self.output_cols[0]

        # Validate that all input columns exist in the dataset
        for col in self.input_cols:
            if col not in samples.columns:
                raise ValueError(f"Input column '{col}' not found in sample")

        # Create a copy to avoid modifying the input
        result = samples.copy()

        # Combine columns using vectorized string operations
        # Convert all input columns to strings and concatenate with separator
        result[output_col] = (
            result[self.input_cols].astype(str).agg(self.separator.join, axis=1)
        )

        return result
