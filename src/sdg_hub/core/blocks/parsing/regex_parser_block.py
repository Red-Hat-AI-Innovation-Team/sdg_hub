# SPDX-License-Identifier: Apache-2.0
"""Regex-based text parser block."""

from typing import cast
import re

import pandas as pd

from ...utils.logger_config import setup_logger
from ..registry import BlockRegistry
from .base_text_parser_block import BaseTextParserBlock

logger = setup_logger(__name__)


@BlockRegistry.register(
    "RegexParserBlock",
    "parsing",
    "Parses text content using regex patterns",
)
class RegexParserBlock(BaseTextParserBlock):
    """Block for parsing text content using regex patterns."""

    parsing_pattern: str

    def _validate_custom(self, dataset: pd.DataFrame) -> None:
        input_cols = cast(list[str], self.input_cols)
        if len(input_cols) != 1:
            raise ValueError("RegexParserBlock requires exactly one input column")

    def _parse_single_text(self, sample: dict, text: str) -> list[dict]:
        output_cols = cast(list[str], self.output_cols)
        matches = re.findall(self.parsing_pattern, text, re.DOTALL)
        if not matches:
            return []

        if isinstance(matches[0], tuple):
            return [
                {
                    **sample,
                    **{
                        col: self._clean(val.strip())
                        for col, val in zip(output_cols, match)
                    },
                }
                for match in matches
            ]
        else:
            return [
                {**sample, output_cols[0]: self._clean(match.strip())}
                for match in matches
            ]
