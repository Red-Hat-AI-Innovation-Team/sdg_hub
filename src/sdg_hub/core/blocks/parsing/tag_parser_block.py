# SPDX-License-Identifier: Apache-2.0
"""Tag-based text parser block."""

from itertools import chain
from typing import Any, Optional
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
        if len(self.input_cols) != 1:
            raise ValueError("TagParserBlock requires exactly one input column")
        if len(self.start_tags) != len(self.output_cols):
            raise ValueError(
                f"Number of tag pairs ({len(self.start_tags)}) must match "
                f"output_cols ({len(self.output_cols)})"
            )

    def _extract(self, text: str, start: str, end: str) -> list[str]:
        if not text:
            return []
        pattern = re.escape(start) + r"(.*?)" + re.escape(end)
        return [m.strip() for m in re.findall(pattern, text, re.DOTALL)]

    def _clean(self, value: str) -> str:
        for tag in self.parser_cleanup_tags or []:
            value = value.replace(tag, "")
        return value

    def _parse_row(self, sample: dict) -> list[dict]:
        text = sample[self.input_cols[0]]
        if not isinstance(text, str) or not text:
            return []

        parsed = {
            col: [self._clean(v) for v in self._extract(text, start, end)]
            for col, start, end in zip(self.output_cols, self.start_tags, self.end_tags)
        }

        if not any(parsed.values()):
            return []

        max_len = max(len(v) for v in parsed.values())
        return [
            {
                **sample,
                **{
                    col: parsed[col][i] if i < len(parsed[col]) else ""
                    for col in self.output_cols
                },
            }
            for i in range(max_len)
        ]

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if samples.empty:
            return pd.DataFrame()
        rows = list(
            chain.from_iterable(map(self._parse_row, samples.to_dict("records")))
        )
        return pd.DataFrame(rows) if rows else pd.DataFrame()
