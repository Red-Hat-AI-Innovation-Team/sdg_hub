# SPDX-License-Identifier: Apache-2.0
"""Filter by value block for dataset filtering operations.

This module provides a block for filtering datasets based on column values
using various operations with optional data type conversion.
"""

# Standard
from typing import Any, Callable, Dict, List, Optional, Type, Union
import operator

# Third Party
from datasets import Dataset

# Local
from ...logger_config import setup_logger
from ..registry import BlockRegistry
from ..base import BaseBlock

logger = setup_logger(__name__)


@BlockRegistry.register(
    "FilterByValueBlock",
    "filtering",
    "Filters datasets based on column values using various comparison operations",
)
class FilterByValueBlock(BaseBlock):
    """A block for filtering datasets based on column values.

    This block allows filtering of datasets using various operations (e.g., equals, contains)
    on specified column values, with optional data type conversion.

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Input column name(s). The first column will be used for filtering.
    filter_value : Union[Any, List[Any]]
        The value(s) to filter by.
    operation : Callable[[Any, Any], bool]
        A binary operator from the operator module (e.g., operator.eq, operator.contains)
        that takes two arguments and returns a boolean.
    convert_dtype : Optional[Union[Type[float], Type[int]]], optional
        Type to convert the filter column to. Can be either float or int.
        If None, no conversion is performed.

    Raises
    ------
    ValueError
        If the operation is not from the operator module.
    """

    def __init__(
        self,
        block_name: str,
        input_cols: Union[str, List[str]],
        filter_value: Union[Any, List[Any]],
        operation: Callable[[Any, Any], bool],
        convert_dtype: Optional[Union[Type[float], Type[int]]] = None,
    ) -> None:
        """Initialize a new FilterByValueBlock instance."""
        # Initialize BaseBlock - filtering doesn't create new columns, so output_cols=None
        super().__init__(
            block_name=block_name,
            input_cols=input_cols,
            output_cols=None,
        )

        # Validate that we have at least one input column
        if len(self.input_cols) == 0:
            raise ValueError("FilterByValueBlock requires at least one input column")

        # Validate that operation is from operator module
        if operation.__module__ != "_operator":
            logger.error("Invalid operation: %s", operation)
            raise ValueError("Operation must be from operator module")

        self.value = filter_value if isinstance(filter_value, list) else [filter_value]
        self.column_name = self.input_cols[0]  # Use first input column for filtering
        self.operation = operation
        self.convert_dtype = convert_dtype

    def _convert_dtype(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert the data type of the filter column.

        Parameters
        ----------
        sample : Dict[str, Any]
            The sample dictionary containing the column to convert.

        Returns
        -------
        Dict[str, Any]
            The sample with converted column value.
        """
        try:
            sample[self.column_name] = self.convert_dtype(sample[self.column_name])
        except ValueError as e:
            logger.error(
                "Error converting dtype: %s, filling with None to be filtered later", e
            )
            sample[self.column_name] = None
        return sample

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate filtered dataset based on specified conditions.

        Parameters
        ----------
        samples : Dataset
            The input dataset to filter.

        Returns
        -------
        Dataset
            The filtered dataset.
        """
        if self.convert_dtype:
            samples = samples.map(self._convert_dtype)

        samples = samples.filter(
            lambda x: x[self.column_name] is not None,
        )

        # Apply filter operation
        samples = samples.filter(
            lambda x: any(
                self.operation(x[self.column_name], value) for value in self.value
            )
        )

        return samples
