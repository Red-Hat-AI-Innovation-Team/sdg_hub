# SPDX-License-Identifier: Apache-2.0
"""JSON structure block for combining multiple columns into a structured JSON object.

This module provides a block for combining multiple columns into a single column
containing a structured JSON object with specified field names.
"""

# Standard
from typing import Any, Dict
import json

from pydantic import Field, field_validator

# Third Party
import pandas as pd

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "JSONStructureBlock",
    "transform",
    "Combines multiple columns into a single column containing a structured JSON object",
)
class JSONStructureBlock(BaseBlock):
    """Block for combining multiple columns into a structured JSON object.

    This block takes values from multiple input columns and combines them into a single
    output column containing a JSON object. The JSON field names match the input column names.

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : List[str]
        List of input column names to include in the JSON object.
        Column names become the JSON field names.
    output_cols : List[str]
        List containing the single output column name.
    ensure_json_serializable : bool
        Whether to ensure all values are JSON serializable (default True).
    pretty_print : bool
        Whether to format JSON with indentation (default False).
    """

    ensure_json_serializable: bool = Field(
        default=True, description="Whether to ensure all values are JSON serializable"
    )
    pretty_print: bool = Field(
        default=False, description="Whether to format JSON with indentation"
    )

    @field_validator("output_cols", mode="after")
    @classmethod
    def validate_output_cols(cls, v):
        """Validate that exactly one output column is specified."""
        if not v or len(v) != 1:
            raise ValueError("JSONStructureBlock requires exactly one output column")
        return v

    def _make_json_serializable(self, value: Any) -> Any:
        """Convert value to JSON serializable format."""
        if value is None:
            return None

        # Handle basic types that are already JSON serializable
        if isinstance(value, (str, int, float, bool)):
            return value

        # Handle lists
        if isinstance(value, (list, tuple)):
            return [self._make_json_serializable(item) for item in value]

        # Handle dictionaries
        if isinstance(value, dict):
            return {k: self._make_json_serializable(v) for k, v in value.items()}

        # Convert other types to string
        return str(value)

    def _get_field_mapping(self) -> Dict[str, str]:
        """Get the mapping of JSON field names to input column names."""
        # Use column names as JSON field names (standard SDG Hub pattern)
        if isinstance(self.input_cols, list):
            return {col: col for col in self.input_cols}

        raise ValueError("input_cols must be a list of column names")

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate a dataset with JSON structured output.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset to process.

        Returns
        -------
        pd.DataFrame
            Dataset with JSON structured output in the specified column.
        """
        if not self.output_cols:
            raise ValueError("output_cols must be specified")

        output_col = self.output_cols[0]
        field_mapping = self._get_field_mapping()

        def _create_json_structure(row):
            """Create JSON structure from input columns."""
            json_obj = {}

            # Build the JSON object using the field mapping
            for json_field, col_name in field_mapping.items():
                if col_name not in row.index:
                    logger.warning(f"Input column '{col_name}' not found in row")
                    json_obj[json_field] = None
                else:
                    value = row[col_name]
                    if self.ensure_json_serializable:
                        value = self._make_json_serializable(value)
                    json_obj[json_field] = value

            # Convert to JSON string
            try:
                if self.pretty_print:
                    json_string = json.dumps(json_obj, indent=2, ensure_ascii=False)
                else:
                    json_string = json.dumps(json_obj, ensure_ascii=False)
                return json_string
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to serialize JSON object: {e}")
                return "{}"

        # Create a copy to avoid modifying the input
        result = samples.copy()

        # Apply the JSON structuring to all samples
        result[output_col] = result.apply(_create_json_structure, axis=1)

        return result
