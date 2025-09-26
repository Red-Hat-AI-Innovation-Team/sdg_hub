# SPDX-License-Identifier: Apache-2.0
"""DEPRECATED: CombineColumnsBlock for backward compatibility.

This module provides a deprecated wrapper for the old CombineColumnsBlock interface.
Use transform.CombineColumnsBlock instead.
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
from ..transform.text_concat import TextConcatBlock

logger = setup_logger(__name__)


@BlockRegistry.register(
    "CombineColumnsBlock",
    "deprecated",
    "DEPRECATED: Use TextConcatBlock instead. Combines multiple columns into a single column using a separator",
)
class CombineColumnsBlock(BaseBlock):
    r"""DEPRECATED: Combine multiple columns into a single column using a separator.

    .. deprecated::
        Use `sdg_hub.blocks.transform.CombineColumnsBlock` instead.
        This class will be removed in a future version.

    This block concatenates values from multiple columns into a single output column,
    using a specified separator between values.

    Parameters
    ----------
    block_name : str
        Name of the block.
    columns : List[str]
        List of column names to combine.
    output_col : str
        Name of the column to store combined values.
    separator : str, optional
        String to use as separator between combined values, by default "\\n\\n".
    **batch_kwargs : Dict[str, Any]
        Additional keyword arguments for batch processing.
    """

    def __init__(
        self,
        block_name: str,
        columns: list[str],
        output_col: str,
        separator: str = "\n\n",
        **batch_kwargs: dict[str, Any],
    ) -> None:
        warnings.warn(
            "CombineColumnsBlock is deprecated. Use sdg_hub.blocks.transform.TextConcatBlock instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Initialize with dummy values for BaseBlock validation
        super().__init__(
            block_name=block_name, input_cols=columns, output_cols=[output_col]
        )

        # Create the new implementation
        self._impl = TextConcatBlock(
            block_name=block_name,
            input_cols=columns,
            output_cols=[output_col],
            separator=separator,
        )

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate a dataset with combined columns.

        Parameters
        ----------
        samples : Dataset
            Input dataset to process.

        Returns
        -------
        Dataset
            Dataset with combined values stored in output column.
        """
        return self._impl.generate(samples)
