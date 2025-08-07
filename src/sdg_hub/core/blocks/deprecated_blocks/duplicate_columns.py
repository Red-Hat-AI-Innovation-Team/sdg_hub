# SPDX-License-Identifier: Apache-2.0
"""Deprecated DuplicateColumns for backwards compatibility.

This module provides a deprecated wrapper around DuplicateColumnsBlock
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
from ..transform import DuplicateColumnsBlock

logger = setup_logger(__name__)


@BlockRegistry.register(
    "DuplicateColumns",
    "deprecated",
    "DEPRECATED: Use DuplicateColumnsBlock instead. Duplicates existing columns with new names according to a mapping dictionary",
)
class DuplicateColumns(BaseBlock):
    """DEPRECATED: Block for duplicating existing columns with new names.

    This block is deprecated and maintained only for backwards compatibility.
    Please use DuplicateColumnsBlock instead.

    This block creates copies of existing columns with new names as specified
    in the columns mapping dictionary.
    """

    def __init__(
        self,
        block_name: str,
        columns_map: dict[str, str],
    ) -> None:
        """Initialize the deprecated DuplicateColumns.

        Parameters
        ----------
        block_name : str
            Name of the block.
        columns_map : Dict[str, str]
            Dictionary mapping existing column names to new column names.
            Keys are existing column names, values are new column names.
        """
        # Issue deprecation warning
        warnings.warn(
            "DuplicateColumns is deprecated and will be removed in a future version. "
            "Please use DuplicateColumnsBlock instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Map old signature to new signature
        super().__init__(
            block_name=block_name,
            input_cols=columns_map,
            output_cols=list(columns_map.values()),
        )

        # Create the new block instance with mapped parameters
        self._new_block = DuplicateColumnsBlock(
            block_name=block_name,
            input_cols=columns_map,
        )

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate dataset with duplicated columns using the new DuplicateColumnsBlock.

        Parameters
        ----------
        samples : Dataset
            The input dataset to duplicate columns from.

        Returns
        -------
        Dataset
            The dataset with additional duplicated columns.
        """
        return self._new_block.generate(samples, **kwargs)
