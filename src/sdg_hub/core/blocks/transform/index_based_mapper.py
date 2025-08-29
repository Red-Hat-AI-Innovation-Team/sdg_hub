# SPDX-License-Identifier: Apache-2.0
"""Selector block for column value selection and mapping.

This module provides a block for selecting and mapping values from one column
to another based on a choice column's value.
"""

# Standard
from typing import Any

# Third Party
from datasets import Dataset
from pydantic import Field, field_validator, model_validator

# Local
from ...utils.error_handling import MissingColumnError
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "IndexBasedMapperBlock",
    "transform",
    "Maps values from source columns to output columns based on choice columns using shared mapping",
)
class IndexBasedMapperBlock(BaseBlock):
    """Block for mapping values from source columns to output columns based on choice columns.

    This block uses a shared mapping dictionary to select values from source columns and
    store them in output columns based on corresponding choice columns' values.
    The choice_cols and output_cols must have the same length - choice_cols[i] determines
    the value for output_cols[i].

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str], Dict[str, Any], None]
        Input column specification. Should include choice columns and mapped columns.
    output_cols : Union[str, List[str], Dict[str, Any], None]
        Output column specification. Must have same length as choice_cols.
    choice_map : Dict[str, str]
        Dictionary mapping choice values to source column names.
    choice_cols : List[str]
        List of column names containing choice values. Must have same length as output_cols.
    """

    choice_map: dict[str, str] = Field(
        ..., description="Dictionary mapping choice values to column names"
    )
    choice_cols: list[str] = Field(
        ..., description="List of column names containing choice values"
    )

    @field_validator("choice_map")
    @classmethod
    def validate_choice_map(cls, v):
        """Validate that choice_map is not empty."""
        if not v:
            raise ValueError("choice_map cannot be empty")
        return v

    @field_validator("choice_cols")
    @classmethod
    def validate_choice_cols_not_empty(cls, v):
        """Validate that choice_cols is not empty."""
        if not v:
            raise ValueError("choice_cols cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_input_output_consistency(self):
        """Validate that choice_cols and output_cols have same length and consistency."""
        # Validate equal lengths
        if len(self.choice_cols) != len(self.output_cols):
            raise ValueError(
                f"choice_cols and output_cols must have same length. "
                f"Got choice_cols: {len(self.choice_cols)}, output_cols: {len(self.output_cols)}"
            )

        if isinstance(self.input_cols, list):
            # Check that all choice_cols are in input_cols
            missing_choice_cols = set(self.choice_cols) - set(self.input_cols)
            if missing_choice_cols:
                logger.warning(
                    f"Choice columns {missing_choice_cols} not found in input_cols {self.input_cols}"
                )

            # Check that all mapped columns are in input_cols
            missing_mapped_cols = set(self.choice_map.values()) - set(self.input_cols)
            if missing_mapped_cols:
                logger.warning(
                    f"Mapped columns {missing_mapped_cols} not found in input_cols {self.input_cols}"
                )

        return self

    def model_post_init(self, __context: Any) -> None:
        """Initialize derived attributes after Pydantic validation."""
        # Create mapping from choice_col to output_col for easy access
        self.choice_to_output_map = dict(zip(self.choice_cols, self.output_cols))

    def _validate_custom(self, samples: Dataset) -> None:
        """Validate that required columns exist in the dataset.

        Parameters
        ----------
        samples : Dataset
            Input dataset to validate.

        Raises
        ------
        MissingColumnError
            If required columns are missing from the dataset.
        ValueError
            If choice values in data are not found in choice_map.
        """
        # Check that all choice_cols exist
        missing_choice_cols = [
            col for col in self.choice_cols if col not in samples.column_names
        ]
        if missing_choice_cols:
            raise MissingColumnError(
                block_name=self.block_name,
                missing_columns=missing_choice_cols,
                available_columns=samples.column_names,
            )

        # Check that all mapped columns exist
        mapped_cols = list(self.choice_map.values())
        missing_cols = list(set(mapped_cols) - set(samples.column_names))
        if missing_cols:
            raise MissingColumnError(
                block_name=self.block_name,
                missing_columns=missing_cols,
                available_columns=samples.column_names,
            )

        # Check that all choice values in all choice columns have corresponding mappings
        all_unique_choices = set()
        for choice_col in self.choice_cols:
            all_unique_choices.update(samples[choice_col])

        mapped_choices = set(self.choice_map.keys())
        unmapped_choices = all_unique_choices - mapped_choices

        if unmapped_choices:
            raise ValueError(
                f"Choice values {sorted(unmapped_choices)} not found in choice_map for block '{self.block_name}'. "
                f"Available choices in mapping: {sorted(mapped_choices)}"
            )

    def _generate(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Generate a new sample by selecting values based on choice mapping.

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample to process.

        Returns
        -------
        Dict[str, Any]
            Sample with selected values stored in corresponding output columns.
        """
        for choice_col, output_col in self.choice_to_output_map.items():
            choice_value = sample[choice_col]
            source_col = self.choice_map[
                choice_value
            ]  # Safe since validated in _validate_custom
            sample[output_col] = sample[source_col]
        return sample

    def generate(self, samples: Dataset, **kwargs) -> Dataset:
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
        # Log the operation
        all_unique_choices = set()
        for choice_col in self.choice_cols:
            all_unique_choices.update(samples[choice_col])
        mapped_choices = set(self.choice_map.keys())

        logger.info(
            f"Mapping values based on choice columns for block '{self.block_name}'",
            extra={
                "block_name": self.block_name,
                "choice_columns": self.choice_cols,
                "output_columns": self.output_cols,
                "choice_mappings": len(self.choice_map),
                "unique_choices_in_data": len(all_unique_choices),
                "unmapped_choices": len(all_unique_choices - mapped_choices),
            },
        )

        # Apply the mapping
        result = samples.map(self._generate)

        # Log completion
        logger.info(
            f"Successfully applied choice mapping for block '{self.block_name}'",
            extra={
                "block_name": self.block_name,
                "rows_processed": len(result),
                "output_columns": self.output_cols,
                "mapping_coverage": len(mapped_choices & all_unique_choices)
                / len(all_unique_choices)
                if all_unique_choices
                else 0,
            },
        )

        return result
