# SPDX-License-Identifier: Apache-2.0
"""Deprecated TextParserBlock for backwards compatibility."""

from itertools import chain
from typing import Any, Optional, cast
import re
import warnings

from pydantic import Field, field_validator, model_validator
import pandas as pd

from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "TextParserBlock",
    "parsing",
    "DEPRECATED: Use TagParserBlock or RegexParserBlock",
)
class TextParserBlock(BaseBlock):
    """Deprecated. Use TagParserBlock or RegexParserBlock instead."""

    _flow_requires_jsonl_tmp: bool = True
    block_type: str = "parser"

    start_tags: list[str] = Field(default_factory=list)
    end_tags: list[str] = Field(default_factory=list)
    parsing_pattern: Optional[str] = Field(default=None)
    parser_cleanup_tags: Optional[list[str]] = Field(default=None)

    def __init__(self, **kwargs):
        warnings.warn(
            "TextParserBlock is deprecated. Use TagParserBlock or RegexParserBlock.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)

    @field_validator("start_tags", "end_tags", mode="before")
    @classmethod
    def normalize_tags(cls, v):
        if v is None:
            return []
        return [v] if isinstance(v, str) else v

    @model_validator(mode="after")
    def validate_config(self):
        has_tags = bool(self.start_tags) or bool(self.end_tags)
        if not self.parsing_pattern and not has_tags:
            raise ValueError("Requires parsing_pattern or start_tags/end_tags")
        if has_tags and len(self.start_tags) != len(self.end_tags):
            raise ValueError("start_tags and end_tags must have same length")
        return self

    def _validate_custom(self, dataset: pd.DataFrame) -> None:
        input_cols = cast(list[str], self.input_cols)
        output_cols = cast(list[str], self.output_cols)
        if len(input_cols) != 1:
            raise ValueError("TextParserBlock requires exactly one input column")
        if self.start_tags and len(self.start_tags) != len(output_cols):
            raise ValueError(
                "When using tag-based parsing, the number of tag pairs must match output_cols. "
                f"Got {len(self.start_tags)} tag pairs and {len(output_cols)} output columns"
            )

    def _clean(self, value: str) -> str:
        for tag in self.parser_cleanup_tags or []:
            value = value.replace(tag, "")
        return value

    def _extract_tags(self, text: str, start: str, end: str) -> list[str]:
        if not text:
            return []
        if not start and not end:
            return [text.strip()] if text.strip() else []

        pattern = ""
        if start:
            pattern += re.escape(start)
        pattern += r"(.*?)"
        if end:
            pattern += re.escape(end)
        elif start:
            pattern += "$"

        return [m.strip() for m in re.findall(pattern, text, re.DOTALL) if m.strip()]

    def _parse_single_text(self, sample: dict, text: str) -> list[dict]:
        output_cols = cast(list[str], self.output_cols)

        if self.parsing_pattern:
            matches = re.findall(self.parsing_pattern, text, re.DOTALL)
            if not matches:
                return []
            if isinstance(matches[0], tuple):
                return [
                    {
                        **sample,
                        **{
                            col: self._clean(val.strip())
                            for col, val in zip(output_cols, m)
                        },
                    }
                    for m in matches
                ]
            return [{**sample, output_cols[0]: self._clean(m.strip())} for m in matches]
        else:
            parsed = {
                col: [self._clean(v) for v in self._extract_tags(text, start, end)]
                for col, start, end in zip(output_cols, self.start_tags, self.end_tags)
            }
            if not any(parsed.values()):
                return []
            return [
                {**sample, **dict(zip(output_cols, values))}
                for values in zip(*(parsed[col] for col in output_cols))
            ]

    def _parse_row(self, sample: dict) -> list[dict]:
        input_cols = cast(list[str], self.input_cols)
        output_cols = cast(list[str], self.output_cols)
        text = sample[input_cols[0]]

        if isinstance(text, list):
            if not text:
                logger.warning(f"Input column '{input_cols[0]}' contains empty list")
                return []
            all_parsed: dict[str, list[str]] = {col: [] for col in output_cols}
            valid = 0
            for item in text:
                if not isinstance(item, str) or not item:
                    continue
                rows = self._parse_single_text(sample, item)
                if rows:
                    valid += 1
                    for row in rows:
                        for col in output_cols:
                            if col in row:
                                all_parsed[col].append(row[col])
            if valid == 0:
                return []
            return [{**sample, **all_parsed}]

        if not isinstance(text, str) or not text:
            return []

        return self._parse_single_text(sample, text)

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if samples.empty:
            return pd.DataFrame()
        rows = list(
            chain.from_iterable(map(self._parse_row, samples.to_dict("records")))
        )
        return pd.DataFrame(rows) if rows else pd.DataFrame()
