# SPDX-License-Identifier: Apache-2.0
"""Deprecated SetToMajorityValue for backwards compatibility.

This module provides a deprecated wrapper around UniformColumnValueSetter
to maintain backwards compatibility with existing code and configurations.
"""

# Standard
from typing import Any
import warnings

# Third Party
from datasets import Dataset

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry
from ..transform import UniformColumnValueSetter

logger = setup_logger(__name__)


@BlockRegistry.register(
    "SetToMajorityValue",
    "deprecated",
    "DEPRECATED: Use UniformColumnValueSetter with reduction_strategy='mode' instead. Sets all values in a column to the most frequent value",
)
class SetToMajorityValue(BaseBlock):
    """DEPRECATED: Block for setting all values in a column to the most frequent value.

    This block is deprecated and maintained only for backwards compatibility.
    Please use UniformColumnValueSetter with reduction_strategy='mode' instead.

    This block finds the most common value (mode) in a specified column and
    replaces all values in that column with this majority value.
    """

    def __init__(
        self,
        block_name: str,
        col_name: str,
    ) -> None:
        """Initialize the deprecated SetToMajorityValue.

        Parameters
        ----------
        block_name : str
            Name of the block.
        col_name : str
            Name of the column to set to majority value.
        """
        # Issue deprecation warning
        warnings.warn(
            "SetToMajorityValue is deprecated and will be removed in a future version. "
            "Please use UniformColumnValueSetter with reduction_strategy='mode' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Map old signature to new signature
        super().__init__(
            block_name=block_name,
            input_cols=[col_name],
            output_cols=[],
        )

        # Create the new block instance with mapped parameters
        self._new_block = UniformColumnValueSetter(
            block_name=block_name,
            input_cols=[col_name],
            reduction_strategy="mode",
        )

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate dataset with column set to majority value using UniformColumnValueSetter.

        Parameters
        ----------
        samples : Dataset
            The input dataset to process.

        Returns
        -------
        Dataset
            The dataset with specified column set to its majority value.
        """
        return self._new_block.generate(samples, **kwargs)
