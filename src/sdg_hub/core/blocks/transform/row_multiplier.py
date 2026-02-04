# SPDX-License-Identifier: Apache-2.0
"""Multiplier block for duplicating dataset rows.

This module provides a block for duplicating each row in a dataset
a configurable number of times.
"""

# Standard
from typing import Any, Optional

from pydantic import Field, field_validator

# Third Party
import pandas as pd

# Local
from ...utils.error_handling import OutputColumnCollisionError
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "RowMultiplierBlock",
    "transform",
    "Duplicates each row in the dataset a configurable number of times",
)
class RowMultiplierBlock(BaseBlock):
    """Block for duplicating dataset rows.

    This block duplicates each row in the dataset a configurable number of times.
    Primary use case: expanding configuration/seed data before LLM processing.

    Attributes
    ----------
    block_name : str
        Name of the block.
    num_samples : int
        Number of times to duplicate each row (must be >= 1).
    add_index_column : bool
        Whether to add a column tracking the duplicate index (0 to num_samples-1).
    index_column_name : str
        Name of the index column (used when add_index_column is True).
    shuffle : bool
        Whether to shuffle output rows after duplication.
    random_seed : Optional[int]
        Seed for reproducible shuffling.
    """

    block_type: str = "transform"

    num_samples: int = Field(
        ..., ge=1, description="Number of times to duplicate each row"
    )
    add_index_column: bool = Field(
        default=False,
        description="Add column tracking duplicate index (0 to num_samples-1)",
    )
    index_column_name: str = Field(
        default="sample_index",
        description="Name of the index column",
    )
    shuffle: bool = Field(
        default=False,
        description="Shuffle output rows after duplication",
    )
    random_seed: Optional[int] = Field(
        default=None,
        description="Seed for reproducible shuffling",
    )

    @field_validator("index_column_name", mode="after")
    @classmethod
    def validate_index_column_name(cls, v: str) -> str:
        """Validate that index_column_name is not empty."""
        if not v or not v.strip():
            raise ValueError("index_column_name cannot be empty")
        return v

    def _validate_custom(self, df: pd.DataFrame) -> None:
        """Validate that index column won't collide with existing columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset to validate.

        Raises
        ------
        OutputColumnCollisionError
            If index column would overwrite an existing column.
        """
        if self.add_index_column and self.index_column_name in df.columns:
            raise OutputColumnCollisionError(
                block_name=self.block_name,
                collision_columns=[self.index_column_name],
                existing_columns=df.columns.tolist(),
            )

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate a dataset with duplicated rows.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset to duplicate.

        Returns
        -------
        pd.DataFrame
            Dataset with each row duplicated num_samples times.
        """
        original_row_count = len(samples)

        # Efficient row duplication using index.repeat()
        result = samples.loc[samples.index.repeat(self.num_samples)].copy()
        result = result.reset_index(drop=True)

        # Add index column if requested
        if self.add_index_column:
            result[self.index_column_name] = (
                list(range(self.num_samples)) * original_row_count
            )

        # Shuffle if requested
        if self.shuffle:
            result = result.sample(frac=1, random_state=self.random_seed).reset_index(
                drop=True
            )

        return result
