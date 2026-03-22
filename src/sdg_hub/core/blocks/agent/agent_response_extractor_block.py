# SPDX-License-Identifier: Apache-2.0
"""Agent response extractor block for extracting fields from agent framework responses.

This module provides the AgentResponseExtractorBlock for extracting text content
and other fields from agent framework response objects (e.g., Langflow responses).
"""

from typing import Any, cast

from pydantic import Field, model_validator
import pandas as pd

from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "AgentResponseExtractorBlock",
    "agent",
    "Extracts text content from agent framework responses",
)
class AgentResponseExtractorBlock(BaseBlock):
    """Block for extracting fields from agent framework response objects.

    This block extracts text content from agent framework responses.
    It expects exactly one input column containing response objects (dict or list of dicts).

    Attributes
    ----------
    block_name : str
        Unique identifier for this block instance.
    agent_framework : str
        Agent framework whose response format to parse (e.g., 'langflow').
    input_cols : Union[str, List[str], Dict[str, Any], None]
        Input column name(s) containing agent response objects. Must specify exactly one column.
    output_cols : Union[str, List[str], Dict[str, Any], None]
        Output column name(s) for extracted fields.
    extract_text : bool
        Whether to extract text content from responses.
    extract_session_id : bool
        Whether to extract session_id from responses.
    expand_lists : bool
        Whether to expand list inputs into individual rows (True) or preserve lists (False).
        Default is True for backward compatibility.
    field_prefix : str
        Prefix to add to output field names. Default is empty string (uses block_name).
        Example: 'agent_' results in 'agent_text', 'agent_session_id'.

    Example
    -------
    >>> block = AgentResponseExtractorBlock(
    ...     block_name="langflow_extractor",
    ...     agent_framework="langflow",
    ...     input_cols="agent_response",
    ...     extract_text=True,
    ... )
    >>> result = block.generate(dataset)
    """

    _flow_requires_jsonl_tmp: bool = True

    block_type: str = "agent_util"

    agent_framework: str = Field(
        ...,
        description="Agent framework whose response format to parse (e.g., 'langflow')",
    )
    extract_text: bool = Field(
        default=True,
        description="Whether to extract text content from responses.",
    )
    extract_session_id: bool = Field(
        default=False,
        description="Whether to extract session_id from responses.",
    )
    extract_tool_trace: bool = Field(
        default=False,
        description=(
            "Whether to extract the full tool call trace from agent responses. "
            "For Langflow, this extracts the content_blocks 'Agent Steps' array "
            "containing structured tool_use entries (name, tool_input, output) "
            "and text entries (input/output messages)."
        ),
    )
    expand_lists: bool = Field(
        default=True,
        description="Whether to expand list inputs into individual rows (True) or preserve lists (False).",
    )
    field_prefix: str = Field(
        default="",
        description="Prefix to add to output field names (e.g., 'agent_' results in 'agent_text').",
    )

    @model_validator(mode="after")
    def validate_extraction_configuration(self):
        """Validate that at least one extraction field is enabled and pre-compute field names."""
        if not any(
            [self.extract_text, self.extract_session_id, self.extract_tool_trace]
        ):
            raise ValueError(
                "AgentResponseExtractorBlock requires at least one extraction field to be enabled: "
                "extract_text, extract_session_id, or extract_tool_trace"
            )

        # Validate agent_framework
        supported_frameworks = ["langflow"]
        if self.agent_framework not in supported_frameworks:
            raise ValueError(
                f"Unsupported agent_framework: '{self.agent_framework}'. "
                f"Supported frameworks: {supported_frameworks}"
            )

        # Pre-compute prefixed field names for efficiency
        prefix = self.field_prefix
        if prefix == "":
            prefix = self.block_name + "_"
        self._text_field = f"{prefix}text"
        self._session_id_field = f"{prefix}session_id"
        self._tool_trace_field = f"{prefix}tool_trace"

        # Advertise output columns for standard collision checks
        self.output_cols = self._get_output_columns()

        return self

    def _validate_custom(self, dataset: pd.DataFrame) -> None:
        """Validate AgentResponseExtractorBlock specific requirements.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to validate.

        Raises
        ------
        ValueError
            If AgentResponseExtractorBlock requirements are not met.
        """
        input_cols = cast(list[str], self.input_cols)
        if len(input_cols) == 0:
            raise ValueError(
                "AgentResponseExtractorBlock expects at least one input column"
            )
        if len(input_cols) > 1:
            logger.warning(
                f"AgentResponseExtractorBlock expects exactly one input column, but got {len(input_cols)}. "
                f"Using the first column: {input_cols[0]}"
            )

    def _extract_langflow_fields(self, response: dict) -> dict[str, Any]:
        """Extract fields from a Langflow response object.

        Langflow response structure:
        {
            "session_id": "...",
            "outputs": [
                {
                    "outputs": [
                        {
                            "results": {
                                "message": {
                                    "text": "..."
                                }
                            }
                        }
                    ]
                }
            ]
        }

        Parameters
        ----------
        response : dict
            Response object from Langflow API.

        Returns
        -------
        dict[str, Any]
            Dictionary with extracted fields using prefixed field names.

        Raises
        ------
        ValueError
            If none of the requested fields are found in the response.
        """
        extracted: dict[str, Any] = {}
        missing_fields: list[str] = []

        if self.extract_text:
            try:
                text = response["outputs"][0]["outputs"][0]["results"]["message"][
                    "text"
                ]
                if text is None:
                    logger.warning("Text field is None, using empty string instead")
                    extracted[self._text_field] = ""
                else:
                    extracted[self._text_field] = text
            except (KeyError, IndexError, TypeError):
                missing_fields.append("text")

        if self.extract_session_id:
            if "session_id" not in response:
                missing_fields.append("session_id")
            else:
                if response["session_id"] is None:
                    logger.warning(
                        "Session ID field is None, using empty string instead"
                    )
                    extracted[self._session_id_field] = ""
                else:
                    extracted[self._session_id_field] = response["session_id"]

        if self.extract_tool_trace:
            tool_trace = self._extract_langflow_tool_trace(response)
            if tool_trace is not None:
                extracted[self._tool_trace_field] = tool_trace
            else:
                missing_fields.append("tool_trace")

        if missing_fields:
            logger.warning(
                f"Requested fields {missing_fields} not found in response. "
                f"Available keys: {list(response.keys())}"
            )

        if not extracted:
            raise ValueError(
                f"No requested fields found in response. Available keys: {list(response.keys())}"
            )
        return extracted

    def _extract_langflow_tool_trace(
        self, response: dict
    ) -> list[dict[str, Any]] | None:
        """Extract the full tool call trace from Langflow content_blocks.

        Langflow agent responses include an 'Agent Steps' content block with
        structured entries for each step: user input (text), tool calls
        (tool_use with name, tool_input, output), and final output (text).

        Parameters
        ----------
        response : dict
            Raw Langflow API response.

        Returns
        -------
        list[dict] or None
            List of step dicts from content_blocks, or None if not found.
        """
        # Try both paths where content_blocks can appear
        for path_fn in [
            lambda r: r["outputs"][0]["outputs"][0]["results"]["message"]["data"][
                "content_blocks"
            ],
            lambda r: r["outputs"][0]["outputs"][0]["results"]["message"][
                "content_blocks"
            ],
        ]:
            try:
                content_blocks = path_fn(response)
                if not content_blocks:
                    continue
                # Find the "Agent Steps" block
                for block in content_blocks:
                    contents = block.get("contents")
                    if contents and isinstance(contents, list):
                        return contents
            except (KeyError, IndexError, TypeError):
                continue

        logger.warning("No content_blocks with tool trace found in Langflow response")
        return None

    def _extract_fields_from_response(self, response: dict) -> dict[str, Any]:
        """Extract specified fields from a single response object.

        Parameters
        ----------
        response : dict
            Response object from agent framework.

        Returns
        -------
        dict[str, Any]
            Dictionary with extracted fields using prefixed field names.

        Raises
        ------
        ValueError
            If none of the requested fields are found in the response.
        """
        if self.agent_framework == "langflow":
            return self._extract_langflow_fields(response)
        else:
            raise ValueError(f"Unsupported agent_framework: '{self.agent_framework}'")

    def _get_output_columns(self) -> list[str]:
        """Get the list of output columns based on extraction settings."""
        columns = []
        if self.extract_text:
            columns.append(self._text_field)
        if self.extract_session_id:
            columns.append(self._session_id_field)
        if self.extract_tool_trace:
            columns.append(self._tool_trace_field)
        return columns

    def _generate(self, sample: dict) -> list[dict]:
        input_cols = cast(list[str], self.input_cols)
        input_column = input_cols[0]
        raw_output = sample[input_column]

        # Handle list inputs (e.g., from batch agent responses)
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
            return self._process_list_preserve_structure(
                sample, raw_output, input_column
            )
        else:
            return self._process_list_expand_rows(sample, raw_output, input_column)

    def _process_list_preserve_structure(
        self, sample: dict, raw_output: list, input_column: str
    ) -> list[dict]:
        """Process list input while preserving list structure."""
        output_columns = self._get_output_columns()
        all_extracted: dict[str, list[Any]] = {col: [] for col in output_columns}
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
        extracted = self._extract_fields_from_response(raw_output)
        return [{**sample, **extracted}]

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        logger.debug(f"Extracting fields from {len(samples)} samples")
        if len(samples) == 0:
            logger.warning("No samples to process, returning empty dataset")
            return pd.DataFrame()

        new_data = []
        samples_list = samples.to_dict("records")

        for sample in samples_list:
            new_data.extend(self._generate(sample))

        return pd.DataFrame(new_data)
