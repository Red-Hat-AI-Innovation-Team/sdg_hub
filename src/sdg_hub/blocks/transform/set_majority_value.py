# SPDX-License-Identifier: Apache-2.0
"""Set to majority value block for column value standardization.

This module provides a block for setting all values in a column to the most
frequent value found in that column.
"""

# Standard
from typing import Any

# Third Party
from datasets import Dataset
from pydantic import field_validator

# Local
from ...logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "SetToMajorityValue",
    "transform",
    "Sets all values in a column to the most frequent (majority) value",
)
class SetToMajorityValue(BaseBlock):
    """Block for setting all values in a column to the most frequent value.

    This block finds the most common value (mode) in a specified column and
    replaces all values in that column with this majority value.

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Input column name(s). Must specify exactly one column.
    output_cols : Union[str, List[str]]
        Output column specification. Defaults to empty list (modifies existing column).
    """

    @field_validator("input_cols", mode="after")
    @classmethod
    def validate_input_cols_single(cls, v):
        """Validate that exactly one input column is specified."""
        if not v or len(v) != 1:
            raise ValueError("SetToMajorityValue requires exactly one input column")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Initialize derived attributes after Pydantic validation."""
        super().model_post_init(__context) if hasattr(super(), "model_post_init") else None
        
        # SetToMajorityValue modifies existing columns in-place, should not have output_cols
        if self.output_cols is not None and len(self.output_cols) > 0:
            logger.warning(
                f"SetToMajorityValue modifies columns in-place. "
                f"Specified output_cols {self.output_cols} will be ignored."
            )
        self.output_cols = []
        
        # Set derived attributes
        self.col_name = self.input_cols[0]  # Use first (and only) input column


    def generate(self, samples: Dataset) -> Dataset:
        """Generate a dataset with column set to majority value.

        Parameters
        ----------
        samples : Dataset
            Input dataset to process.

        Returns
        -------
        Dataset
            Dataset with specified column set to its majority value.
        """

        # Convert to pandas for mode calculation
        df = samples.to_pandas()

        # Find the majority value for logging
        # Validate column has data
        if df.empty:
            raise ValueError(f"Cannot compute majority value for empty dataset")
        
        mode_values = df[self.col_name].mode()
        if mode_values.empty:
            raise ValueError(f"Column '{self.col_name}' has no valid values to compute majority")
        
        majority_value = mode_values.iloc[0]
        
        # Log the operation
        logger.info(
            f"Setting column '{self.col_name}' to majority value '{majority_value}' for block '{self.block_name}'"
        )

        # Set all values to majority value (original logic)
        df[self.col_name] = majority_value

        # Convert back to dataset
        return Dataset.from_pandas(df)
