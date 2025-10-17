# SPDX-License-Identifier: Apache-2.0
"""Rename columns block for dataset column renaming operations.

This module provides a block for renaming columns in datasets according
to a mapping specification.
"""

# Standard
from typing import Any

from pydantic import field_validator

# Third Party
import pandas as pd

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "RenameColumnsBlock",
    "transform",
    "Renames columns in a dataset according to a mapping specification",
)
class RenameColumnsBlock(BaseBlock):
    """Block for renaming columns in a dataset.

    This block renames columns in a dataset according to a mapping specification.
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
                "input_cols must be a dictionary mapping old column names to new column names"
            )
        return v

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate a dataset with renamed columns.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset to rename columns in.

        Returns
        -------
        pd.DataFrame
            Dataset with renamed columns.

        Raises
        ------
        ValueError
            If attempting to rename to a column name that already exists,
            or if the original column names don't exist in the dataset.
        """
        # Check that all original column names exist in the dataset
        existing_cols = set(samples.columns.tolist())
        original_cols = set(self.input_cols.keys())

        missing_cols = original_cols - existing_cols
        if missing_cols:
            raise ValueError(
                f"Original column names {sorted(missing_cols)} not in the dataset"
            )

        # Check for column name collisions
        # Strict validation: no target column name can be an existing column name
        # This prevents chained/circular renames which can be confusing
        target_cols = set(self.input_cols.values())

        collision = target_cols & existing_cols
        if collision:
            raise ValueError(
                f"Cannot rename to existing column names: {sorted(collision)}. "
                "Target column names must not already exist in the dataset. "
                "Chained renames are not supported."
            )

        # Rename columns using pandas method
        return samples.rename(columns=self.input_cols)
