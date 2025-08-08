# SPDX-License-Identifier: Apache-2.0
"""Melt columns block for wide-to-long format transformation.

This module provides a block for transforming wide dataset format into long format
by melting specified columns into rows.
"""

# Standard
from typing import Any

# Third Party
from datasets import Dataset
from pydantic import field_validator

# Local
from ...utils.error_handling import MissingColumnError
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "MeltColumnsBlock",
    "transform",
    "Transforms wide dataset format into long format by melting columns into rows",
)
class MeltColumnsBlock(BaseBlock):
    """Block for flattening multiple columns into a long format.

    This block transforms a wide dataset format into a long format by melting
    specified columns into rows, creating new variable and value columns.

    The input_cols should contain the columns to be melted (variable columns).
    The output_cols must specify exactly two columns: [value_column, variable_column].
    Any other columns in the dataset will be treated as ID columns and preserved.

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str], Dict[str, Any], None]
        Columns to be melted into rows (variable columns).
    output_cols : Union[str, List[str], Dict[str, Any], None]
        Output column specification. Must specify exactly two columns: [value_column, variable_column].
    """

    @field_validator("input_cols", mode="after")
    @classmethod
    def validate_input_cols(cls, v):
        """Validate that input_cols is not empty."""
        if not v:
            raise ValueError("input_cols cannot be empty")
        return v

    @field_validator("output_cols", mode="after")
    @classmethod
    def validate_output_cols(cls, v):
        """Validate that exactly two output columns are specified."""
        if len(v) != 2:
            raise ValueError(
                f"MeltColumnsBlock expects exactly two output columns (value, variable), got {len(v)}: {v}"
            )
        return v

    def model_post_init(self, __context: Any) -> None:
        """Initialize derived attributes after Pydantic validation."""
        super().model_post_init(__context) if hasattr(
            super(), "model_post_init"
        ) else None

        # Derive value and variable column names from output_cols
        self.value_name = self.output_cols[0]  # First output column is value
        self.var_name = self.output_cols[1]  # Second output column is variable

        # input_cols contains the columns to be melted (what was var_cols)
        self.var_cols = (
            self.input_cols if isinstance(self.input_cols, list) else [self.input_cols]
        )

    def _validate_custom(self, samples: Dataset) -> None:
        """Validate that required columns exist in the dataset.

        Parameters
        ----------
        samples : Dataset
            Input dataset to validate.

        Raises
        ------
        MissingColumnError
            If required columns are missing from the dataset.
        """
        # Check that all var_cols exist in the dataset
        missing_cols = list(set(self.var_cols) - set(samples.column_names))
        if missing_cols:
            raise MissingColumnError(
                block_name=self.block_name,
                missing_columns=missing_cols,
                available_columns=samples.column_names,
            )

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate a flattened dataset in long format.

        Parameters
        ----------
        samples : Dataset
            Input dataset to flatten.

        Returns
        -------
        Dataset
            Flattened dataset in long format with new variable and value columns.
        """
        # Use the original simple logic - just adapted to use derived attributes
        df = samples.to_pandas()
        id_cols = [col for col in samples.column_names if col not in self.var_cols]
        flatten_df = df.melt(
            id_vars=id_cols,
            value_vars=self.var_cols,
            value_name=self.value_name,
            var_name=self.var_name,
        )
        return Dataset.from_pandas(flatten_df)
