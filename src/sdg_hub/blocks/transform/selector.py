# SPDX-License-Identifier: Apache-2.0
"""Selector block for column value selection and mapping.

This module provides a block for selecting and mapping values from one column
to another based on a choice column's value.
"""

# Standard
from typing import Any, Dict, List, Optional, Union

# Third Party
from datasets import Dataset

# Local
from ...logger_config import setup_logger
from ..base import BaseBlock
from ...registry import BlockRegistry
from ...utils.error_handling import MissingColumnError

logger = setup_logger(__name__)


@BlockRegistry.register(
    "SelectorBlock",
    "transform", 
    "Selects and maps values from one column to another based on choice mapping",
)
class SelectorBlock(BaseBlock):
    """Block for selecting and mapping values from one column to another.

    This block uses a mapping dictionary to select values from one column and
    store them in a new output column based on a choice column's value.

    Parameters
    ----------
    block_name : str
        Name of the block.
    choice_map : Dict[str, str]
        Dictionary mapping choice values to column names.
    choice_col : str
        Name of the column containing choice values.
    output_col : str
        Name of the column to store selected values.
    input_cols : Optional[Union[str, List[str]]], optional
        Input column specification. Should include choice_col and mapped columns.
    output_cols : Optional[Union[str, List[str]]], optional
        Output column specification. Defaults to output_col.
    num_procs : int, optional
        Number of processes for parallel processing, by default 8.
    **batch_kwargs : Dict[str, Any]
        Additional keyword arguments for batch processing.
    """

    def __init__(
        self,
        block_name: str,
        choice_map: Dict[str, str],
        choice_col: str,
        output_col: str,
        input_cols: Optional[Union[str, List[str]]] = None,
        output_cols: Optional[Union[str, List[str]]] = None,
        num_procs: int = 8,
        **batch_kwargs: Dict[str, Any],
    ) -> None:
        # Handle backward compatibility - old style constructor
        if input_cols is None and output_cols is None:
            # Legacy mode - derive columns automatically
            mapped_cols = list(choice_map.values())
            input_cols = [choice_col] + mapped_cols
            output_cols = [output_col]
        
        super().__init__(
            block_name=block_name,
            input_cols=input_cols,
            output_cols=output_cols,
            **batch_kwargs
        )
        
        self.choice_map = choice_map
        self.choice_col = choice_col
        self.output_col = output_col
        self.num_procs = num_procs
        
        # Validate that choice_col and mapped columns are in input_cols if specified
        if isinstance(self.input_cols, list):
            if self.choice_col not in self.input_cols:
                logger.warning(
                    f"Choice column '{self.choice_col}' not found in input_cols {self.input_cols}"
                )
            
            missing_mapped_cols = set(choice_map.values()) - set(self.input_cols)
            if missing_mapped_cols:
                logger.warning(
                    f"Mapped columns {missing_mapped_cols} not found in input_cols {self.input_cols}"
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
        # Check that choice_col exists
        if self.choice_col not in samples.column_names:
            raise MissingColumnError(
                block_name=self.block_name,
                missing_columns=[self.choice_col],
                available_columns=samples.column_names
            )
        
        # Check that all mapped columns exist
        mapped_cols = list(self.choice_map.values())
        missing_cols = list(set(mapped_cols) - set(samples.column_names))
        if missing_cols:
            raise MissingColumnError(
                block_name=self.block_name,
                missing_columns=missing_cols,
                available_columns=samples.column_names
            )
        
        return samples

    def _generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new sample by selecting values based on choice mapping.

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample to process.

        Returns
        -------
        Dict[str, Any]
            Sample with selected values stored in output column.
        """
        choice_value = sample[self.choice_col]
        
        # Check if choice value exists in mapping
        if choice_value not in self.choice_map:
            logger.warning(
                f"Choice value '{choice_value}' not found in choice_map. "
                f"Available choices: {list(self.choice_map.keys())}"
            )
            sample[self.output_col] = None
        else:
            # Get the column name to select from
            source_col = self.choice_map[choice_value]
            sample[self.output_col] = sample[source_col]
        
        return sample

    def generate(self, samples: Dataset) -> Dataset:
        """Generate a new dataset with selected values.

        Parameters
        ----------
        samples : Dataset
            Input dataset to process.

        Returns
        -------
        Dataset
            Dataset with selected values stored in output column.
        """
        # Validate input
        samples = self._validate(samples)
        
        # Log the operation
        unique_choices = set(samples[self.choice_col])
        mapped_choices = set(self.choice_map.keys())
        
        logger.info(
            f"Selecting values based on choice mapping for block '{self.block_name}'",
            extra={
                "block_name": self.block_name,
                "choice_column": self.choice_col,
                "output_column": self.output_col,
                "choice_mappings": len(self.choice_map),
                "unique_choices_in_data": len(unique_choices),
                "unmapped_choices": len(unique_choices - mapped_choices),
            }
        )
        
        # Apply the mapping
        result = samples.map(self._generate, num_proc=self.num_procs)
        
        # Log completion
        logger.info(
            f"Successfully applied choice mapping for block '{self.block_name}'",
            extra={
                "block_name": self.block_name,
                "rows_processed": len(result),
                "output_column": self.output_col,
                "mapping_coverage": len(mapped_choices & unique_choices) / len(unique_choices) if unique_choices else 0,
            }
        )
        
        return result