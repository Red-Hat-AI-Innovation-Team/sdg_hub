# SPDX-License-Identifier: Apache-2.0
"""Filter by value block for dataset filtering operations.

This module provides a block for filtering datasets based on column values
using various operations with optional data type conversion.
"""

# Standard
from typing import Any, Optional, Union
import operator

# Third Party
from datasets import Dataset
from pydantic import Field, field_validator

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)

# Supported operations mapping
OPERATION_MAP = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "contains": operator.contains,
    "in": lambda x, y: x in y,  # Reverse contains for "x in y" semantics
}

# Supported data types mapping
DTYPE_MAP = {
    "float": float,
    "int": int,
}


@BlockRegistry.register(
    "ColumnValueFilterBlock",
    "filtering",
    "Filters datasets based on column values using various comparison operations",
)
class ColumnValueFilterBlock(BaseBlock):
    """A block for filtering datasets based on column values.

    This block allows filtering of datasets using various operations (e.g., equals, contains)
    on specified column values, with optional data type conversion.

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Input column name(s). The first column will be used for filtering.
    filter_value : Union[Any, List[Any]]
        The value(s) to filter by.
    operation : str
        A string representing the binary operation to perform (e.g., "eq", "contains", "gt").
        Supported operations: "eq", "ne", "lt", "le", "gt", "ge", "contains", "in".
    convert_dtype : Optional[str], optional
        String representation of type to convert the filter column to. Can be "float" or "int".
        If None, no conversion is performed.
    """

    filter_value: Union[Any, list[Any]] = Field(
        ..., description="The value(s) to filter by"
    )
    operation: str = Field(
        ...,
        description="String name of binary operator for comparison (e.g., 'eq', 'contains')",
    )
    convert_dtype: Optional[str] = Field(
        None,
        description="String name of type to convert filter column to ('float' or 'int')",
    )

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v):
        """Validate that operation is a supported operation string."""
        if v not in OPERATION_MAP:
            raise ValueError(
                f"Unsupported operation '{v}'. Supported operations: {list(OPERATION_MAP.keys())}"
            )
        return v

    @field_validator("convert_dtype")
    @classmethod
    def validate_convert_dtype(cls, v):
        """Validate that convert_dtype is a supported type string."""
        if v is not None and v not in DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype '{v}'. Supported dtypes: {list(DTYPE_MAP.keys())}"
            )
        return v

    @field_validator("input_cols", mode="after")
    @classmethod
    def validate_input_cols_not_empty(cls, v):
        """Validate that we have at least one input column."""
        if not v or len(v) == 0:
            raise ValueError(
                "ColumnValueFilterBlock requires at least one input column"
            )
        return v

    def model_post_init(self, __context: Any) -> None:
        """Initialize derived attributes after Pydantic validation."""
        super().model_post_init(__context) if hasattr(
            super(), "model_post_init"
        ) else None

        # Ensure output_cols is empty list for filtering operations (doesn't create new columns)
        if self.output_cols is None:
            self.output_cols = []

        # Set derived attributes
        self.filter_value = (
            self.filter_value
            if isinstance(self.filter_value, list)
            else [self.filter_value]
        )
        self.column_name = self.input_cols[0]  # Use first input column for filtering

        # Convert string operation to actual callable
        self._operation_func = OPERATION_MAP[self.operation]

        # Convert string dtype to actual type if specified
        self._convert_dtype_func = (
            DTYPE_MAP[self.convert_dtype] if self.convert_dtype else None
        )

    def _convert_dtype(self, sample: dict[str, Any]) -> dict[str, Any]:
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
            sample[self.column_name] = self._convert_dtype_func(
                sample[self.column_name]
            )
        except ValueError as e:
            logger.error(
                "Error converting dtype: %s, filling with None to be filtered later", e
            )
            sample[self.column_name] = None
        return sample

    def generate(self, samples: Dataset, **_kwargs: Any) -> Dataset:
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
        if self._convert_dtype_func:
            samples = samples.map(self._convert_dtype)

        samples = samples.filter(
            lambda x: x[self.column_name] is not None,
        )

        # Apply filter operation
        samples = samples.filter(
            lambda x: any(
                self._operation_func(x[self.column_name], value)
                for value in self.filter_value
            )
        )

        return samples
