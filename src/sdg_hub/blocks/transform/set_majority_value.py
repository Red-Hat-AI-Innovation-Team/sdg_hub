# SPDX-License-Identifier: Apache-2.0
"""Set to majority value block for column value standardization.

This module provides a block for setting all values in a column to the most
frequent value found in that column.
"""

# Standard
from typing import Any, List, Optional, Union

# Third Party
from datasets import Dataset

# Local
from ...logger_config import setup_logger
from ..base import BaseBlock
from ...registry import BlockRegistry
from ...utils.error_handling import MissingColumnError

logger = setup_logger(__name__)


@BlockRegistry.register(
    "SetToMajorityValue", 
    "transform",
    "Sets all values in a column to the most frequent (majority) value",
)
class SetToMajorityValue(BaseBlock):
    """Block for setting all values in a column to the most frequent value.

    This block finds the most common value (mode) in a specified column and
    replaces all values in that column with this majority value.

    Parameters
    ----------
    block_name : str
        Name of the block.
    col_name : str
        Name of the column to set to majority value.
    input_cols : Optional[Union[str, List[str]]], optional
        Input column specification. If provided, col_name must be included.
    output_cols : Optional[Union[str, List[str]]], optional  
        Output column specification. Defaults to same as input column.
    """

    def __init__(
        self,
        block_name: str,
        col_name: str,
        input_cols: Optional[Union[str, List[str]]] = None,
        output_cols: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> None:
        # Handle backward compatibility - old style constructor
        if input_cols is None and output_cols is None:
            # Legacy mode - derive columns automatically
            input_cols = [col_name]
            output_cols = [col_name]
        
        super().__init__(
            block_name=block_name,
            input_cols=input_cols,
            output_cols=output_cols,
            **kwargs
        )
        
        self.col_name = col_name
        
        # Validate that col_name is in input_cols if specified
        if isinstance(self.input_cols, list) and self.col_name not in self.input_cols:
            logger.warning(
                f"Column '{self.col_name}' not found in input_cols {self.input_cols}"
            )

    def _validate(self, samples: Dataset) -> Dataset:
        """Validate that the required column exists in the dataset.
        
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
            If the required column is missing from the dataset.
        """
        if self.col_name not in samples.column_names:
            raise MissingColumnError(
                block_name=self.block_name,
                missing_columns=[self.col_name],
                available_columns=samples.column_names
            )
        
        return samples

    def generate(self, samples: Dataset) -> Dataset:
        """Generate a dataset with column set to majority value.

        Parameters
        ----------
        samples : Dataset
            Input dataset to process.

        Returns
        -------
        Dataset
            Dataset with specified column set to its majority value.
        """
        # Validate input
        samples = self._validate(samples)
        
        # Convert to pandas for mode calculation
        df = samples.to_pandas()
        
        # Find the majority value (mode)
        mode_series = df[self.col_name].mode()
        if len(mode_series) == 0:
            logger.warning(f"No mode found for column '{self.col_name}', keeping original values")
            return samples
        
        majority_value = mode_series[0]
        original_unique_values = df[self.col_name].nunique()
        
        # Log the operation
        logger.info(
            f"Setting column '{self.col_name}' to majority value for block '{self.block_name}'",
            extra={
                "block_name": self.block_name,
                "column_name": self.col_name,
                "majority_value": str(majority_value),
                "original_unique_values": original_unique_values,
                "total_rows": len(df),
            }
        )
        
        # Set all values to majority value
        df[self.col_name] = majority_value
        
        # Convert back to dataset
        result = Dataset.from_pandas(df)
        
        # Log completion
        logger.info(
            f"Successfully set column to majority value for block '{self.block_name}'",
            extra={
                "block_name": self.block_name,
                "column_name": self.col_name,
                "new_value": str(majority_value),
                "rows_affected": len(result),
            }
        )
        
        return result