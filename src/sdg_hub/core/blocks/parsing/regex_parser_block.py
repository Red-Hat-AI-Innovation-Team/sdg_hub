# SPDX-License-Identifier: Apache-2.0
"""Regex-based text parser block."""

from itertools import chain
from typing import Any, Optional
import re

from pydantic import Field
import pandas as pd

from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "RegexParserBlock",
    "parsing",
    "Parses text content using regex patterns",
)
class RegexParserBlock(BaseBlock):
    """Block for parsing text content using regex patterns."""

    _flow_requires_jsonl_tmp: bool = True
    block_type: str = "parser"

    parsing_pattern: str = Field(..., description="Regex pattern with capture groups")
    parser_cleanup_tags: Optional[list[str]] = Field(
        default=None, description="Tags to remove from extracted content"
    )

    def _validate_custom(self, dataset: pd.DataFrame) -> None:
        if len(self.input_cols) != 1:
            raise ValueError("RegexParserBlock requires exactly one input column")

    def _clean(self, value: str) -> str:
        for tag in self.parser_cleanup_tags or []:
            value = value.replace(tag, "")
        return value

    def _parse_row(self, sample: dict) -> list[dict]:
        text = sample[self.input_cols[0]]
        if not isinstance(text, str) or not text:
            return []

        matches = re.findall(self.parsing_pattern, text, re.DOTALL)
        if not matches:
            return []

        # Handle tuple matches (multiple capture groups) vs single matches
        if matches and isinstance(matches[0], tuple):
            return [
                {
                    **sample,
                    **{
                        col: self._clean(val.strip())
                        for col, val in zip(self.output_cols, match)
                    },
                }
                for match in matches
            ]
        else:
            return [
                {**sample, self.output_cols[0]: self._clean(match.strip())}
                for match in matches
            ]

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if samples.empty:
            return pd.DataFrame()
        rows = list(
            chain.from_iterable(map(self._parse_row, samples.to_dict("records")))
        )
        return pd.DataFrame(rows) if rows else pd.DataFrame()
