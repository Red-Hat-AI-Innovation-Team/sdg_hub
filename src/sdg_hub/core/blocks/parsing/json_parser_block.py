# SPDX-License-Identifier: Apache-2.0
"""JSON parser block for extracting JSON fields into separate columns.

This module provides a block for parsing JSON from text (including embedded JSON)
and expanding the fields into separate columns.
"""

# Standard
from typing import Any, Optional, cast
import json
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
    "JSONParserBlock",
    "parsing",
    "Parses JSON from text and expands fields into separate columns",
)
class JSONParserBlock(BaseBlock):
    """Block for parsing JSON from text and expanding fields into columns.

    This block takes a column containing JSON strings (or text with embedded JSON),
    parses the JSON, and expands the fields into separate columns. Useful for
    processing LLM responses that return JSON-formatted data.

    Attributes
    ----------
    block_name : str
        Name of the block.
    input_cols : List[str]
        Single input column containing JSON text to parse.
    output_cols : List[str]
        Optional list of specific fields to extract. If empty, all JSON fields
        are extracted as columns.
    field_prefix : str
        Optional prefix to add to extracted column names.
    fix_trailing_commas : bool
        Whether to fix trailing commas in JSON (common LLM output issue).
    extract_embedded : bool
        Whether to extract JSON embedded in surrounding text.
    drop_input : bool
        Whether to drop the input column after extraction.
    """

    block_type: str = "parsing"

    field_prefix: str = Field(
        default="",
        description="Optional prefix to add to extracted column names",
    )
    fix_trailing_commas: bool = Field(
        default=True,
        description="Whether to fix trailing commas in JSON (common LLM output issue)",
    )
    extract_embedded: bool = Field(
        default=True,
        description="Whether to extract JSON embedded in surrounding text",
    )
    drop_input: bool = Field(
        default=False,
        description="Whether to drop the input column after extraction",
    )

    @field_validator("input_cols", mode="after")
    @classmethod
    def validate_input_cols(cls, v: list[str]) -> list[str]:
        """Validate that exactly one input column is specified."""
        if not v or len(v) != 1:
            raise ValueError("JSONParserBlock requires exactly one input column")
        return v

    def _fix_json_string(self, json_str: str) -> str:
        """Fix common JSON issues like trailing commas.

        Parameters
        ----------
        json_str : str
            The JSON string to fix.

        Returns
        -------
        str
            Fixed JSON string.
        """
        if self.fix_trailing_commas:
            # Fix trailing commas before } or ]
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*]", "]", json_str)
        return json_str

    def _extract_or_recover(
        self, text: str, open_char: str, close_char: str
    ) -> Optional[str]:
        """Extract delimited JSON, recovering truncated input if needed.

        Finds the first ``open_char`` and last ``close_char``. When the text
        appears truncated (opener present but no matching closer), counts
        unclosed delimiters and appends the required number of closers.
        """
        start = text.find(open_char)
        if start == -1:
            return None

        end = text.rfind(close_char)
        if end != -1 and end > start:
            return text[start : end + 1]

        unclosed = 0
        for ch in text[start:]:
            if ch == open_char:
                unclosed += 1
            elif ch == close_char:
                unclosed -= 1

        if unclosed <= 0:
            return None

        logger.warning(
            f"JSON appears truncated (missing {unclosed} closing "
            f"'{close_char}'). Attempting recovery."
        )
        return text[start:].rstrip() + close_char * unclosed

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text, handling embedded JSON.

        Parameters
        ----------
        text : str
            The text that may contain JSON.

        Returns
        -------
        Optional[str]
            The extracted JSON string, or None if not found.
        """
        if not text:
            return None

        if not self.extract_embedded:
            return text.strip()

        result = self._extract_or_recover(text, "{", "}")
        if result is not None:
            return result

        return self._extract_or_recover(text, "[", "]")

    @staticmethod
    def _normalize_parsed(parsed: Any) -> dict[str, Any]:
        """Normalize a parsed JSON value into a dict.

        Dicts pass through unchanged; lists are wrapped as ``{"items": ...}``;
        scalars are wrapped as ``{"value": ...}``.
        """
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
        return {"value": parsed}

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Parse JSON from text.

        Parameters
        ----------
        text : str
            The text containing JSON.

        Returns
        -------
        dict[str, Any]
            Parsed JSON as a dictionary. Returns empty dict on failure.
        """
        if not isinstance(text, str) or not text:
            return {}

        json_str = self._extract_json(text)
        if not json_str:
            logger.warning("No JSON found in input text")
            return {}

        json_str = self._fix_json_string(json_str)

        try:
            parsed = json.loads(json_str, strict=False)
            return self._normalize_parsed(parsed)
        except json.JSONDecodeError as e:
            logger.warning(
                f"JSON parse error at position {e.pos}: {e.msg}. "
                f"Problematic area: ...{json_str[max(0, e.pos - 30) : e.pos + 30]}..."
            )
            return {}

    def _filter_output_columns(self, parsed_df: pd.DataFrame) -> pd.DataFrame:
        """Filter parsed DataFrame to only the requested output columns.

        Logs warnings for missing columns. If none of the requested columns
        are found, all parsed columns are kept as a fallback.

        Parameters
        ----------
        parsed_df : pd.DataFrame
            DataFrame of parsed JSON fields.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame.
        """
        if not self.output_cols:
            return parsed_df

        existing_cols = [col for col in self.output_cols if col in parsed_df.columns]
        missing_cols = [col for col in self.output_cols if col not in parsed_df.columns]

        if missing_cols:
            logger.warning(
                f"Requested columns not found in JSON: {missing_cols}. "
                f"Available columns: {list(parsed_df.columns)}"
            )

        if existing_cols:
            return parsed_df[existing_cols]

        logger.warning(
            "None of the requested output columns found in JSON. Keeping all fields."
        )
        return parsed_df

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate a dataset with JSON fields expanded into columns.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset to process.

        Returns
        -------
        pd.DataFrame
            Dataset with JSON fields expanded into separate columns.
        """
        input_col = cast(list[str], self.input_cols)[0]
        result = samples.copy()

        # Parse JSON from each row and expand into columns
        parsed_series = result[input_col].apply(self._parse_json)
        parsed_df = parsed_series.apply(pd.Series)

        # Remove phantom '0' column created when all rows return empty dicts
        if 0 in parsed_df.columns:
            parsed_df = parsed_df.drop(columns=[0])

        parsed_df = self._filter_output_columns(parsed_df)

        if self.field_prefix:
            parsed_df = parsed_df.add_prefix(self.field_prefix)

        if self.drop_input:
            result = result.drop(columns=[input_col])

        return pd.concat([result, parsed_df], axis=1)
