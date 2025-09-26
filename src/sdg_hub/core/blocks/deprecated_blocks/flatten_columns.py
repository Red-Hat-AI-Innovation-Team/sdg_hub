# SPDX-License-Identifier: Apache-2.0
"""Deprecated FlattenColumnsBlock for backwards compatibility.

This module provides a deprecated wrapper around MeltColumnsBlock
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
from ..transform import MeltColumnsBlock

logger = setup_logger(__name__)


@BlockRegistry.register(
    "FlattenColumnsBlock",
    "deprecated",
    "DEPRECATED: Use MeltColumnsBlock instead. Transforms wide dataset format into long format by melting columns into rows",
)
class FlattenColumnsBlock(BaseBlock):
    """DEPRECATED: Block for flattening multiple columns into a long format.

    This block is deprecated and maintained only for backwards compatibility.
    Please use MeltColumnsBlock instead.

    This block transforms a wide dataset format into a long format by melting
    specified columns into rows, creating new variable and value columns.
    """

    def __init__(
        self,
        block_name: str,
        var_cols: list[str],
        value_name: str,
        var_name: str,
    ) -> None:
        """Initialize the deprecated FlattenColumnsBlock.

        Parameters
        ----------
        block_name : str
            Name of the block.
        var_cols : List[str]
            List of column names to be melted into rows.
        value_name : str
            Name of the new column that will contain the values.
        var_name : str
            Name of the new column that will contain the variable names.
        """
        # Issue deprecation warning
        warnings.warn(
            "FlattenColumnsBlock is deprecated and will be removed in a future version. "
            "Please use MeltColumnsBlock instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Map old signature to new signature
        super().__init__(
            block_name=block_name,
            input_cols=var_cols,
            output_cols=[value_name, var_name],
        )

        # Create the new block instance with mapped parameters
        self._new_block = MeltColumnsBlock(
            block_name=block_name,
            input_cols=var_cols,
            output_cols=[value_name, var_name],
        )

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate flattened dataset using the new MeltColumnsBlock.

        Parameters
        ----------
        samples : Dataset
            The input dataset to flatten.

        Returns
        -------
        Dataset
            The flattened dataset in long format.
        """
        return self._new_block.generate(samples, **kwargs)
