# SPDX-License-Identifier: Apache-2.0
"""Message formatter block for converting tool traces to structured conversations.

This module wraps the ``tool_trace_to_messages()`` utility so that
training-data formatting can be expressed as a block inside a flow YAML.
"""

# Standard
from typing import Any, cast

from pydantic import field_validator

# Third Party
import pandas as pd

# Local
from ...utils.logger_config import setup_logger
from ...utils.message_formatter import tool_trace_to_messages
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "MessageFormatterBlock",
    "transform",
    "Converts tool traces and tool lists into structured tool-calling conversations",
)
class MessageFormatterBlock(BaseBlock):
    """Block for formatting tool traces into structured training conversations.

    This block applies ``tool_trace_to_messages()`` row-by-row, reading a
    tool-trace column and a tool-list column and producing a messages column
    suitable for fine-tuning tool-use models.

    The block expects ``input_cols`` as a **dict** with two required keys:

    * ``tool_trace`` -- the DataFrame column containing the tool trace
      (list of dicts from ``AgentResponseExtractorBlock``).
    * ``tool_list`` -- the DataFrame column containing the tool schemas
      (list of dicts with ``name``, ``description``, ``inputSchema``).

    ``output_cols`` must be a single-element list giving the name of the
    output column that will hold the formatted message list.

    Example YAML usage::

        - block_type: MessageFormatterBlock
          block_config:
            block_name: format_training_data
            input_cols:
              tool_trace: extract_agent_text_tool_trace
              tool_list: tool_list
            output_cols: [messages]

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : dict
        Mapping with keys ``tool_trace`` and ``tool_list`` pointing to
        DataFrame column names.
    output_cols : list[str]
        Single-element list with the output column name.
    """

    block_type: str = "transform"

    @field_validator("input_cols", mode="after")
    @classmethod
    def validate_input_cols(cls, v):
        """Validate that input_cols is a dict with the required keys."""
        if not isinstance(v, dict):
            raise ValueError(
                "MessageFormatterBlock requires input_cols to be a dict "
                "with keys 'tool_trace' and 'tool_list'"
            )
        missing = {"tool_trace", "tool_list"} - set(v.keys())
        if missing:
            raise ValueError(
                f"MessageFormatterBlock input_cols missing required keys: "
                f"{', '.join(sorted(missing))}"
            )
        return v

    @field_validator("output_cols", mode="after")
    @classmethod
    def validate_output_cols(cls, v):
        """Validate that exactly one output column is specified."""
        if not v or len(v) != 1:
            raise ValueError("MessageFormatterBlock requires exactly one output column")
        return v

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Check that the two input columns exist in the DataFrame."""
        input_cols = cast(dict, self.input_cols)
        available = df.columns.tolist()
        missing = [col for col in input_cols.values() if col not in available]
        if missing:
            from ...utils.error_handling import MissingColumnError

            raise MissingColumnError(
                block_name=self.block_name,
                missing_columns=missing,
                available_columns=available,
            )

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Apply ``tool_trace_to_messages`` to each row.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset.  Must contain the columns referenced by
            ``input_cols["tool_trace"]`` and ``input_cols["tool_list"]``.

        Returns
        -------
        pd.DataFrame
            Dataset with the new output column containing structured
            message lists.
        """
        input_cols = cast(dict, self.input_cols)
        output_cols = cast(list[str], self.output_cols)
        output_col = output_cols[0]

        trace_col = input_cols["tool_trace"]
        tools_col = input_cols["tool_list"]

        result = samples.copy()
        result[output_col] = result.apply(
            lambda row: tool_trace_to_messages(row[trace_col], row[tools_col]),
            axis=1,
        )
        return result
