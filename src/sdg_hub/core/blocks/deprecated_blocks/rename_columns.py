# SPDX-License-Identifier: Apache-2.0
"""Deprecated RenameColumns for backwards compatibility.

This module provides a deprecated wrapper around RenameColumnsBlock
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
from ..transform import RenameColumnsBlock

logger = setup_logger(__name__)


@BlockRegistry.register(
    "RenameColumns",
    "deprecated",
    "DEPRECATED: Use RenameColumnsBlock instead. Renames columns in a dataset according to a mapping dictionary",
)
class RenameColumns(BaseBlock):
    """DEPRECATED: Block for renaming columns in a dataset.

    This block is deprecated and maintained only for backwards compatibility.
    Please use RenameColumnsBlock instead.

    This block renames columns in a dataset according to a mapping dictionary,
    where keys are existing column names and values are new column names.
    """

    def __init__(
        self,
        block_name: str,
        columns_map: dict[str, str],
    ) -> None:
        """Initialize the deprecated RenameColumns.

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
            "RenameColumns is deprecated and will be removed in a future version. "
            "Please use RenameColumnsBlock instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Map old signature to new signature
        super().__init__(
            block_name=block_name,
            input_cols=columns_map,
            output_cols=[],
        )

        # Create the new block instance with mapped parameters
        self._new_block = RenameColumnsBlock(
            block_name=block_name,
            input_cols=columns_map,
        )

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate dataset with renamed columns using the new RenameColumnsBlock.

        Parameters
        ----------
        samples : Dataset
            The input dataset to rename columns in.

        Returns
        -------
        Dataset
            The dataset with renamed columns.
        """
        return self._new_block.generate(samples, **kwargs)
