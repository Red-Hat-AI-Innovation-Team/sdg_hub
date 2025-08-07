# SPDX-License-Identifier: Apache-2.0
"""Deprecated FilterByValueBlock for backwards compatibility.

This module provides a deprecated wrapper around ColumnValueFilterBlock
to maintain backwards compatibility with existing code and configurations.
"""

# Standard
from typing import Any, Callable, Optional, Union
import warnings

# Third Party
from datasets import Dataset

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..filtering import ColumnValueFilterBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "FilterByValueBlock",
    "deprecated",
    "DEPRECATED: Use ColumnValueFilterBlock instead. Filters datasets based on column values using various comparison operations",
)
class FilterByValueBlock(BaseBlock):
    """DEPRECATED: A block for filtering datasets based on column values.

    This block is deprecated and maintained only for backwards compatibility.
    Please use ColumnValueFilterBlock instead.

    This block allows filtering of datasets using various operations (e.g., equals, contains)
    on specified column values, with optional data type conversion.
    """

    def __init__(
        self,
        block_name: str,
        filter_column: str,
        filter_value: Union[Any, list[Any]],
        operation: Callable[[Any, Any], bool],
        convert_dtype: Optional[Union[type[float], type[int]]] = None,
        **batch_kwargs: dict[str, Any],
    ) -> None:
        """Initialize the deprecated FilterByValueBlock.

        Parameters
        ----------
        block_name : str
            Name of the block.
        filter_column : str
            Column name to filter on.
        filter_value : Union[Any, list[Any]]
            The value(s) to filter by.
        operation : Callable[[Any, Any], bool]
            A binary operator from the operator module.
        convert_dtype : Optional[Union[type[float], type[int]]], optional
            Type to convert the filter column to.
        **batch_kwargs : dict[str, Any]
            Additional batch processing arguments.
        """
        # Issue deprecation warning
        warnings.warn(
            "FilterByValueBlock is deprecated and will be removed in a future version. "
            "Please use ColumnValueFilterBlock instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Map old signature to new signature
        super().__init__(
            block_name=block_name,
            input_cols=[filter_column],
            output_cols=[],
        )

        # Create the new block instance with mapped parameters
        self._new_block = ColumnValueFilterBlock(
            block_name=block_name,
            input_cols=[filter_column],
            output_cols=[],
            filter_value=filter_value,
            operation=operation,
            convert_dtype=convert_dtype,
        )

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate filtered dataset using the new ColumnValueFilterBlock.

        Parameters
        ----------
        samples : Dataset
            The input dataset to filter.

        Returns
        -------
        Dataset
            The filtered dataset.
        """
        return self._new_block.generate(samples, **kwargs)
