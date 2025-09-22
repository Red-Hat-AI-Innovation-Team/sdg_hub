# SPDX-License-Identifier: Apache-2.0
"""LLM parser block for extracting fields from LLM response objects.

This module provides the LLMParserBlock for extracting specific fields
(content, reasoning_content, tool_calls) from chat completion response objects.
"""

# Standard
from typing import Any

# Third Party
from datasets import Dataset
from pydantic import Field, model_validator

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "LLMParserBlock",
    "llm",
    "Extracts specified fields from LLM response objects",
)
class LLMParserBlock(BaseBlock):
    """Block for extracting fields from LLM response objects.

    This block extracts specified fields from chat completion response objects.
    It expects exactly one input column containing response objects (dict or list of dicts).

    Attributes
    ----------
    block_name : str
        Unique identifier for this block instance.
    input_cols : Union[str, List[str], Dict[str, Any], None]
        Input column name(s) containing LLM response objects. Must specify exactly one column.
    output_cols : Union[str, List[str], Dict[str, Any], None]
        Output column name(s) for extracted fields.
    extract_content : bool
        Whether to extract 'content' field from responses.
    extract_reasoning_content : bool
        Whether to extract 'reasoning_content' field from responses.
    extract_tool_calls : bool
        Whether to extract 'tool_calls' field from responses.
    expand_lists : bool
        Whether to expand list inputs into individual rows (True) or preserve lists (False).
        Default is True for backward compatibility.
    field_prefix : str
        Prefix to add to output field names. Default is empty string (no prefix).
        Example: 'llm_' results in 'llm_content', 'llm_reasoning_content', 'llm_tool_calls'.
    """

    extract_content: bool = Field(
        default=True,
        description="Whether to extract 'content' field from responses.",
    )
    extract_reasoning_content: bool = Field(
        default=False,
        description="Whether to extract 'reasoning_content' field from responses.",
    )
    extract_tool_calls: bool = Field(
        default=False,
        description="Whether to extract 'tool_calls' field from responses.",
    )
    expand_lists: bool = Field(
        default=True,
        description="Whether to expand list inputs into individual rows (True) or preserve lists (False).",
    )
    field_prefix: str = Field(
        default="",
        description="Prefix to add to output field names (e.g., 'llm_' results in 'llm_content', 'llm_reasoning_content').",
    )

    @model_validator(mode="after")
    def validate_extraction_configuration(self):
        """Validate that at least one extraction field is enabled and pre-compute field names."""
        if not any(
            [
                self.extract_content,
                self.extract_reasoning_content,
                self.extract_tool_calls,
            ]
        ):
            raise ValueError(
                "LLMParserBlock requires at least one extraction field to be enabled: "
                "extract_content, extract_reasoning_content, or extract_tool_calls"
            )

        # Pre-compute prefixed field names for efficiency
        prefix = self.field_prefix
        if prefix == "":
            prefix = self.block_name + "_"
        self._content_field = f"{prefix}content"
        self._reasoning_content_field = f"{prefix}reasoning_content"
        self._tool_calls_field = f"{prefix}tool_calls"

        # Advertise output columns for standard collision checks
        self.output_cols = self._get_output_columns()

        return self

    def _validate_custom(self, dataset: Dataset) -> None:
        """Validate LLMParserBlock specific requirements.

        Parameters
        ----------
        dataset : Dataset
            The dataset to validate.

        Raises
        ------
        ValueError
            If LLMParserBlock requirements are not met.
        """
        # Validate that we have exactly one input column
        if len(self.input_cols) == 0:
            raise ValueError("LLMParserBlock expects at least one input column")
        if len(self.input_cols) > 1:
            logger.warning(
                f"LLMParserBlock expects exactly one input column, but got {len(self.input_cols)}. "
                f"Using the first column: {self.input_cols[0]}"
            )

    def _extract_fields_from_response(self, response: dict) -> dict[str, Any]:
        """Extract specified fields from a single response object.

        Parameters
        ----------
        response : dict
            Response object from chat completion API

        Returns
        -------
        dict[str, Any]
            Dictionary with extracted fields using prefixed field names

        Raises
        ------
        ValueError
            If none of the requested fields are found in the response
        """
        extracted = {}
        missing_fields = []

        if self.extract_content:
            if "content" not in response:
                missing_fields.append("content")
            else:
                if response["content"] is None:
                    ## skip this field
                    logger.warning("Content field is None, using empty string instead")
                    extracted[self._content_field] = ""
                else:
                    extracted[self._content_field] = response["content"]

        if self.extract_reasoning_content:
            if "reasoning_content" not in response:
                missing_fields.append("reasoning_content")
            else:
                if response["reasoning_content"] is None:
                    ## skip this field
                    logger.warning(
                        "Reasoning content field is None, using empty string instead"
                    )
                    extracted[self._reasoning_content_field] = ""
                else:
                    extracted[self._reasoning_content_field] = response[
                        "reasoning_content"
                    ]

        if self.extract_tool_calls:
            if "tool_calls" not in response:
                missing_fields.append("tool_calls")
            else:
                if response["tool_calls"] is None:
                    ## skip this field
                    logger.warning("Tool calls field is None, using empty list instead")
                    extracted[self._tool_calls_field] = []
                else:
                    extracted[self._tool_calls_field] = response["tool_calls"]

        if missing_fields:
            logger.warning(
                f"Requested fields {missing_fields} not found in response. Available keys: {list(response.keys())}"
            )

        if not extracted:
            raise ValueError(
                f"No requested fields found in response. Available keys: {list(response.keys())}"
            )
        return extracted

    def _get_output_columns(self) -> list[str]:
        """Get the list of output columns based on extraction settings."""
        columns = []
        if self.extract_content:
            columns.append(self._content_field)
        if self.extract_reasoning_content:
            columns.append(self._reasoning_content_field)
        if self.extract_tool_calls:
            columns.append(self._tool_calls_field)
        return columns

    def _generate(self, sample: dict) -> list[dict]:
        input_column = self.input_cols[0]
        raw_output = sample[input_column]

        # Handle list inputs (e.g., from LLMChatBlock with n > 1)
        if isinstance(raw_output, list):
            return self._process_list_input(sample, raw_output, input_column)

        # Handle single dict input
        elif isinstance(raw_output, dict):
            return self._process_single_input(sample, raw_output)

        else:
            logger.warning(
                f"Input column '{input_column}' contains invalid data type: {type(raw_output)}. "
                f"Expected dict or list[dict]"
            )
            return []

    def _process_list_input(
        self, sample: dict, raw_output: list, input_column: str
    ) -> list[dict]:
        """Process list of response objects."""
        if not raw_output:
            logger.warning(f"Input column '{input_column}' contains empty list")
            return []

        if not self.expand_lists:
            # Preserve list structure - collect all extracted fields as lists
            return self._process_list_preserve_structure(
                sample, raw_output, input_column
            )
        else:
            # Expand lists - create individual rows for each response
            return self._process_list_expand_rows(sample, raw_output, input_column)

    def _process_list_preserve_structure(
        self, sample: dict, raw_output: list, input_column: str
    ) -> list[dict]:
        """Process list input while preserving list structure."""
        output_columns = self._get_output_columns()
        all_extracted = {col: [] for col in output_columns}
        valid_responses = 0

        for i, response in enumerate(raw_output):
            if not isinstance(response, dict):
                logger.warning(
                    f"List item {i} in column '{input_column}' is not a dict"
                )
                continue

            try:
                extracted = self._extract_fields_from_response(response)
                valid_responses += 1
                for col in output_columns:
                    if col in extracted:
                        all_extracted[col].append(extracted[col])
            except ValueError as e:
                logger.warning(f"Failed to extract fields from list item {i}: {e}")
                continue

        if valid_responses == 0:
            raise ValueError(
                f"No valid responses found in list input for column '{input_column}'"
            )

        # Return single row with lists as values
        return [{**sample, **all_extracted}]

    def _process_list_expand_rows(
        self, sample: dict, raw_output: list, input_column: str
    ) -> list[dict]:
        """Process list input by expanding into individual rows."""
        all_results = []

        for i, response in enumerate(raw_output):
            if not isinstance(response, dict):
                logger.warning(
                    f"List item {i} in column '{input_column}' is not a dict"
                )
                continue

            try:
                extracted = self._extract_fields_from_response(response)
                # Create a row for this response
                result_row = {**sample, **extracted}
                all_results.append(result_row)
            except ValueError as e:
                logger.warning(f"Failed to extract fields from list item {i}: {e}")
                continue

        if not all_results:
            raise ValueError(
                f"No valid responses found in list input for column '{input_column}'"
            )

        return all_results

    def _process_single_input(self, sample: dict, raw_output: dict) -> list[dict]:
        """Process single response object."""
        # _extract_fields_from_response now raises ValueError if no fields found
        extracted = self._extract_fields_from_response(raw_output)
        return [{**sample, **extracted}]

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        logger.debug(f"Extracting fields from {len(samples)} samples")
        if len(samples) == 0:
            logger.warning("No samples to process, returning empty dataset")
            return Dataset.from_list([])

        new_data = []
        for sample in samples:
            new_data.extend(self._generate(sample))
        return Dataset.from_list(new_data)
