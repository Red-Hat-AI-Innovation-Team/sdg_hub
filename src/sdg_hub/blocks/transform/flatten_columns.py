# SPDX-License-Identifier: Apache-2.0
"""Flatten columns block for wide-to-long format transformation.

This module provides a block for transforming wide dataset format into long format
by melting specified columns into rows.
"""

# Standard
from typing import Any, List, Optional, Union

# Third Party
from datasets import Dataset

# Local
from ...logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry
from ...utils.error_handling import MissingColumnError

logger = setup_logger(__name__)


@BlockRegistry.register(
    "FlattenColumnsBlock",
    "transform",
    "Transforms wide dataset format into long format by melting columns into rows",
)
class FlattenColumnsBlock(BaseBlock):
    """Block for flattening multiple columns into a long format.

    This block transforms a wide dataset format into a long format by melting
    specified columns into rows, creating new variable and value columns.

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
    input_cols : Optional[Union[str, List[str]]], optional
        Input column specification. If provided, var_cols must be subset.
    output_cols : Optional[Union[str, List[str]]], optional
        Output column specification. Defaults to [value_name, var_name].
    """

    def __init__(
        self,
        block_name: str,
        var_cols: List[str],
        value_name: str,
        var_name: str,
        input_cols: Optional[Union[str, List[str]]] = None,
        output_cols: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> None:
        # Handle backward compatibility - old style constructor
        if input_cols is None and output_cols is None:
            # Legacy mode - derive columns automatically
            input_cols = var_cols
            output_cols = [value_name, var_name]

        super().__init__(
            block_name=block_name,
            input_cols=input_cols,
            output_cols=output_cols,
            **kwargs,
        )

        self.var_cols = var_cols
        self.value_name = value_name
        self.var_name = var_name

        # Validate var_cols are subset of input_cols if both specified
        if isinstance(self.input_cols, list) and self.var_cols:
            missing_cols = set(self.var_cols) - set(self.input_cols)
            if missing_cols:
                logger.warning(
                    f"Variable columns {missing_cols} not found in input_cols {self.input_cols}"
                )

    def _validate(self, samples: Dataset) -> Dataset:
        """Validate that required columns exist in the dataset.

        Parameters
        ----------
        samples : Dataset
            Input dataset to validate.

        Returns
        -------
        Dataset
            Validated dataset.

        Raises
        ------
        MissingColumnError
            If required columns are missing from the dataset.
        """
        # Check that all var_cols exist in the dataset
        missing_cols = list(set(self.var_cols) - set(samples.column_names))
        if missing_cols:
            raise MissingColumnError(
                block_name=self.block_name,
                missing_columns=missing_cols,
                available_columns=samples.column_names,
            )

        return samples

    def generate(self, samples: Dataset) -> Dataset:
        """Generate a flattened dataset in long format.

        Parameters
        ----------
        samples : Dataset
            Input dataset to flatten.

        Returns
        -------
        Dataset
            Flattened dataset in long format with new variable and value columns.
        """
        # Validate input
        samples = self._validate(samples)

        # Log the operation
        logger.info(
            f"Flattening {len(self.var_cols)} columns into long format for block '{self.block_name}'",
            extra={
                "block_name": self.block_name,
                "variable_columns": self.var_cols,
                "value_column": self.value_name,
                "variable_name_column": self.var_name,
                "input_rows": len(samples),
            },
        )

        # Convert to pandas for melting operation
        df = samples.to_pandas()
        id_cols = [col for col in samples.column_names if col not in self.var_cols]

        # Perform the melt operation
        flatten_df = df.melt(
            id_vars=id_cols,
            value_vars=self.var_cols,
            value_name=self.value_name,
            var_name=self.var_name,
        )

        # Convert back to dataset
        result = Dataset.from_pandas(flatten_df)

        # Log completion
        logger.info(
            f"Successfully flattened dataset for block '{self.block_name}'",
            extra={
                "block_name": self.block_name,
                "output_rows": len(result),
                "new_columns": [self.value_name, self.var_name],
                "expansion_factor": len(result) / len(samples)
                if len(samples) > 0
                else 0,
            },
        )

        return result
