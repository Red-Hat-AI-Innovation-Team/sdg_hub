# SPDX-License-Identifier: Apache-2.0
"""Sampler block for randomly sampling values from list columns.

This module provides a block for sampling a specified number of values
from list or set columns in each row of a dataset.
"""

# Standard
from typing import Any, Optional

from pydantic import Field, field_validator
import numpy as np

# Third Party
import pandas as pd

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "SamplerBlock",
    "transform",
    "Randomly samples n values from a list column and outputs to a new column",
)
class SamplerBlock(BaseBlock):
    """Block for randomly sampling values from list columns.

    This block samples a specified number of values from each row's list/set
    and outputs the sampled values to a new column.

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : list[str]
        Single input column containing lists/sets to sample from.
    output_cols : list[str]
        Single output column for sampled values.
    num_samples : int
        Number of values to sample from each list.
    random_seed : int, optional
        Random seed for reproducibility.
    """

    block_type: str = "transform"

    num_samples: int = Field(
        default=5, description="Number of values to randomly sample from each list"
    )
    random_seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility"
    )

    @field_validator("input_cols", mode="after")
    @classmethod
    def validate_input_cols(cls, v: list[str]) -> list[str]:
        """Validate that exactly one input column is specified."""
        if not v or len(v) != 1:
            raise ValueError("SamplerBlock requires exactly one input column")
        return v

    @field_validator("output_cols", mode="after")
    @classmethod
    def validate_output_cols(cls, v: list[str]) -> list[str]:
        """Validate that exactly one output column is specified."""
        if not v or len(v) != 1:
            raise ValueError("SamplerBlock requires exactly one output column")
        return v

    def _sample_values(self, values: Any, rng: np.random.Generator) -> list[Any]:
        """Sample values from a list or set.

        Parameters
        ----------
        values : Any
            The list, set, or other iterable to sample from.
        rng : np.random.Generator
            Random number generator for sampling.

        Returns
        -------
        list[Any]
            Sampled values as a list.
        """
        if values is None:
            return []

        # Convert to list if it's a set or other iterable
        if isinstance(values, set):
            values = list(values)
        elif not isinstance(values, (list, np.ndarray)):
            try:
                values = list(values)
            except TypeError:
                return []

        if len(values) == 0:
            return []

        n = min(self.num_samples, len(values))
        indices = rng.choice(len(values), size=n, replace=False)
        return [values[i] for i in indices]

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate a dataset with sampled values.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset to process.

        Returns
        -------
        pd.DataFrame
            Dataset with sampled values in output column.
        """
        input_col = self.input_cols[0]
        output_col = self.output_cols[0]

        result = samples.copy()

        # Create random number generator
        rng = np.random.default_rng(self.random_seed)

        result[output_col] = result[input_col].apply(
            lambda x: self._sample_values(x, rng)
        )

        return result
