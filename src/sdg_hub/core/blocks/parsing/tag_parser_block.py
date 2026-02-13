# SPDX-License-Identifier: Apache-2.0
"""Tag-based text parser block."""

from itertools import chain
from typing import Any, Optional, cast
import re

from pydantic import Field, field_validator, model_validator
import pandas as pd

from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "TagParserBlock",
    "parsing",
    "Parses text content using start/end tags",
)
class TagParserBlock(BaseBlock):
    """Block for parsing text content using start/end tags."""

    _flow_requires_jsonl_tmp: bool = True
    block_type: str = "parser"

    start_tags: list[str] = Field(..., description="Start tags for extraction")
    end_tags: list[str] = Field(..., description="End tags for extraction")
    parser_cleanup_tags: Optional[list[str]] = Field(
        default=None, description="Tags to remove from extracted content"
    )

    @field_validator("start_tags", "end_tags", mode="before")
    @classmethod
    def normalize_tags(cls, v):
        if v is None:
            return []
        return [v] if isinstance(v, str) else v

    @model_validator(mode="after")
    def validate_tags(self):
        if len(self.start_tags) != len(self.end_tags):
            raise ValueError(
                f"start_tags and end_tags must have same length. "
                f"Got {len(self.start_tags)} and {len(self.end_tags)}"
            )
        return self

    def _validate_custom(self, dataset: pd.DataFrame) -> None:
        input_cols = cast(list[str], self.input_cols)
        output_cols = cast(list[str], self.output_cols)
        if len(input_cols) != 1:
            raise ValueError("TagParserBlock requires exactly one input column")
        if len(self.start_tags) != len(output_cols):
            raise ValueError(
                f"Number of tag pairs ({len(self.start_tags)}) must match "
                f"output_cols ({len(output_cols)})"
            )

    def _extract(self, text: str, start: str, end: str) -> list[str]:
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

    def _clean(self, value: str) -> str:
        for tag in self.parser_cleanup_tags or []:
            value = value.replace(tag, "")
        return value

    def _parse_single_text(self, sample: dict, text: str) -> list[dict]:
        output_cols = cast(list[str], self.output_cols)
        parsed = {
            col: [self._clean(v) for v in self._extract(text, start, end)]
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
