# SPDX-License-Identifier: Apache-2.0
"""Multiplier block for duplicating dataset rows.

This module provides a block for duplicating each row in a dataset
a configurable number of times.
"""

# Standard
from typing import Any, Optional

from pydantic import Field

# Third Party
import pandas as pd

# Local
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
    shuffle : bool
        Whether to shuffle output rows after duplication.
    random_seed : Optional[int]
        Seed for reproducible shuffling.
    """

    block_type: str = "transform"

    num_samples: int = Field(
        ..., ge=1, description="Number of times to duplicate each row"
    )
    shuffle: bool = Field(
        default=False,
        description="Shuffle output rows after duplication",
    )
    random_seed: Optional[int] = Field(
        default=None,
        description="Seed for reproducible shuffling",
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

        # Use iloc with RangeIndex to handle duplicate index labels
        result = samples.iloc[
            pd.RangeIndex(original_row_count).repeat(self.num_samples)
        ].reset_index(drop=True)

        # Shuffle if requested
        if self.shuffle:
            result = result.sample(frac=1, random_state=self.random_seed).reset_index(
                drop=True
            )

        return result
