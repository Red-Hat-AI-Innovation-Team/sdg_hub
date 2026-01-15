# SPDX-License-Identifier: Apache-2.0
"""Dict field extractor block for extracting fields from nested dictionaries.

This module provides a block for extracting fields from nested dictionary structures
using dot notation paths with array indexing support.
"""

# Standard
from typing import Any, Optional
import re

from pydantic import Field, field_validator

# Third Party
import pandas as pd

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "DictFieldExtractorBlock",
    "transform",
    "Extracts fields from nested dictionaries using dot notation paths",
)
class DictFieldExtractorBlock(BaseBlock):
    """Block for extracting fields from nested dictionary structures.

    This block extracts values from nested dictionaries using dot notation paths
    with array indexing support. It's particularly useful for parsing complex API
    responses like those from Langflow, LangGraph, or other agent frameworks.

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Input column name containing dict or list[dict]. Must be exactly one column.
    output_cols : List[str]
        Output column names for extracted fields.
    field_mapping : dict[str, str]
        Maps output column names to field paths in dot notation.
        Example: {"message": "artifacts.message", "steps": "results.content_blocks[0]"}
    default_values : Optional[dict[str, Any]], optional
        Default values for each output column when field is missing, by default None.
    strict_mode : bool, optional
        If True, raise error when field not found. If False, use default or None, by default False.

    Examples
    --------
    >>> # Extract fields from Langflow response
    >>> block = DictFieldExtractorBlock(
    ...     block_name="extract_langflow",
    ...     input_cols="agent_response",
    ...     output_cols=["message_text", "steps"],
    ...     field_mapping={
    ...         "message_text": "outputs[0].outputs[0].artifacts.message",
    ...         "steps": "outputs[0].outputs[0].results.message.content_blocks"
    ...     },
    ...     default_values={"message_text": "", "steps": []},
    ...     strict_mode=False
    ... )
    """

    field_mapping: dict[str, str] = Field(
        ...,
        description="Maps output column names to field paths. "
        "Example: {'message': 'artifacts.message', 'steps': 'results.content_blocks[0]'}",
    )
    default_values: Optional[dict[str, Any]] = Field(
        default=None,
        description="Default values for each output column when field is missing",
    )
    strict_mode: bool = Field(
        default=False,
        description="If True, raise error when field not found. If False, use default or None",
    )

    @field_validator("field_mapping")
    @classmethod
    def validate_field_mapping(cls, v):
        """Validate that field_mapping is a non-empty dict."""
        if not v:
            raise ValueError("field_mapping must be a non-empty dictionary")
        if not isinstance(v, dict):
            raise ValueError("field_mapping must be a dictionary")
        return v

    def _validate_custom(self, dataset: pd.DataFrame) -> None:
        """Validate DictFieldExtractorBlock specific requirements.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to validate.

        Raises
        ------
        ValueError
            If DictFieldExtractorBlock requirements are not met.
        """
        # Validate exactly one input column
        if len(self.input_cols) != 1:
            raise ValueError(
                f"DictFieldExtractorBlock expects exactly one input column, "
                f"got {len(self.input_cols)}: {self.input_cols}"
            )

        # Validate field_mapping keys match output_cols
        mapping_keys = set(self.field_mapping.keys())
        output_keys = set(self.output_cols)
        if mapping_keys != output_keys:
            raise ValueError(
                f"field_mapping keys {mapping_keys} must match output_cols {output_keys}"
            )

        # Validate default_values keys (if provided) match output_cols
        if self.default_values:
            default_keys = set(self.default_values.keys())
            if default_keys != output_keys:
                raise ValueError(
                    f"default_values keys {default_keys} must match output_cols {output_keys}"
                )

        # Sample validation: check first row is dict or list
        if len(dataset) > 0:
            first_value = dataset.iloc[0][self.input_cols[0]]
            if not isinstance(first_value, (dict, list)):
                raise ValueError(
                    f"Input column '{self.input_cols[0]}' must contain dict or list[dict], "
                    f"got {type(first_value)}"
                )

    def _parse_path_segment(self, segment: str) -> tuple[str, Optional[int]]:
        """Parse a path segment to extract key and optional array index.

        Parameters
        ----------
        segment : str
            Path segment like "key" or "key[0]"

        Returns
        -------
        tuple[str, Optional[int]]
            Tuple of (key, index) where index is None if no bracket notation

        Examples
        --------
        >>> _parse_path_segment("items")
        ("items", None)
        >>> _parse_path_segment("items[0]")
        ("items", 0)
        """
        # Match pattern like "key[123]"
        match = re.match(r"^([^\[]+)\[(\d+)\]$", segment)
        if match:
            key = match.group(1)
            index = int(match.group(2))
            return key, index
        return segment, None

    def _extract_field(self, data: dict, field_path: str) -> Any:
        """Extract field from nested dict using dot notation with array indexing.

        Parameters
        ----------
        data : dict
            Dictionary to extract from
        field_path : str
            Dot notation path like "level1.level2[0].level3"

        Returns
        -------
        Any
            Extracted value

        Raises
        ------
        KeyError
            If key not found in dict
        IndexError
            If index out of range in list
        TypeError
            If trying to index non-list or access non-dict

        Examples
        --------
        >>> data = {"a": {"b": [{"c": "value"}]}}
        >>> _extract_field(data, "a.b[0].c")
        "value"
        """
        # Split by dots to get path segments
        segments = field_path.split(".")
        current = data

        for segment in segments:
            # Parse segment for array indexing
            key, index = self._parse_path_segment(segment)

            # Navigate to key
            if isinstance(current, dict):
                if key not in current:
                    available_keys = list(current.keys())
                    raise KeyError(
                        f"Key '{key}' not found in path '{field_path}'. "
                        f"Available keys at this level: {available_keys}"
                    )
                current = current[key]
            else:
                raise TypeError(
                    f"Cannot access key '{key}' on non-dict type {type(current)}"
                )

            # Apply array indexing if specified
            if index is not None:
                if isinstance(current, list):
                    current = current[index]
                else:
                    raise TypeError(
                        f"Cannot index '{key}' of type {type(current)} with [{index}]"
                    )

        return current

    def _extract_fields_from_dict(self, data: dict, sample: dict) -> dict:
        """Extract all mapped fields from a single dict.

        Parameters
        ----------
        data : dict
            Dictionary to extract from
        sample : dict
            Original sample row data

        Returns
        -------
        dict
            Dictionary with original sample data plus extracted fields
        """
        extracted = {}

        for output_col, field_path in self.field_mapping.items():
            try:
                value = self._extract_field(data, field_path)
                extracted[output_col] = value
                logger.debug(
                    f"Extracted '{output_col}' from path '{field_path}': "
                    f"{type(value).__name__} with length {len(value) if hasattr(value, '__len__') else 'N/A'}"
                )
            except (KeyError, IndexError, TypeError) as e:
                if self.strict_mode:
                    raise ValueError(
                        f"Failed to extract field '{field_path}' for output column '{output_col}': {e}"
                    ) from e

                # Use default value or None
                default_value = (
                    self.default_values.get(output_col)
                    if self.default_values
                    else None
                )
                extracted[output_col] = default_value
                logger.warning(
                    f"Field '{field_path}' not found for '{output_col}', "
                    f"using default: {default_value}. Error: {e}"
                )

        return {**sample, **extracted}

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate dataset with extracted fields.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset to process.
        **kwargs : Any
            Additional keyword arguments (unused).

        Returns
        -------
        pd.DataFrame
            Dataset with extracted fields added as new columns.
        """
        logger.debug(f"Extracting fields for {len(samples)} samples")
        if len(samples) == 0:
            logger.warning("No samples to process, returning empty dataset")
            return pd.DataFrame()

        input_column = self.input_cols[0]

        # Convert DataFrame to list of dicts for better performance
        samples_list = samples.to_dict("records")
        new_data: list[dict] = []

        for idx, sample in enumerate(samples_list):
            raw_data = sample[input_column]

            # Handle dict input
            if isinstance(raw_data, dict):
                result_row = self._extract_fields_from_dict(raw_data, sample)
                new_data.append(result_row)

            # Handle list of dicts - process each dict separately
            elif isinstance(raw_data, list):
                if not raw_data:
                    logger.warning(
                        f"Sample {idx}: Input column '{input_column}' contains empty list"
                    )
                    continue

                for item_idx, item in enumerate(raw_data):
                    if isinstance(item, dict):
                        result_row = self._extract_fields_from_dict(item, sample)
                        new_data.append(result_row)
                    else:
                        logger.warning(
                            f"Sample {idx}, item {item_idx}: Expected dict, got {type(item)}"
                        )

            else:
                logger.warning(
                    f"Sample {idx}: Input column '{input_column}' contains invalid data type: {type(raw_data)}. "
                    f"Expected dict or list[dict]"
                )

        if not new_data:
            logger.warning("No valid data extracted, returning empty dataset")
            return pd.DataFrame()

        result_df = pd.DataFrame(new_data)
        logger.info(
            f"Successfully extracted fields for {len(result_df)} rows from {len(samples)} input samples"
        )
        return result_df
