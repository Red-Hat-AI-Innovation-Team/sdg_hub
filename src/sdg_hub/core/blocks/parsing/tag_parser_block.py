# SPDX-License-Identifier: Apache-2.0
"""Tag-based text parser block."""

from typing import cast
import re

from pydantic import Field, field_validator, model_validator
import pandas as pd

from ...utils.logger_config import setup_logger
from ..registry import BlockRegistry
from .base_text_parser_block import BaseTextParserBlock

logger = setup_logger(__name__)


@BlockRegistry.register(
    "TagParserBlock",
    "parsing",
    "Parses text content using start/end tags",
)
class TagParserBlock(BaseTextParserBlock):
    """Block for parsing text content using start/end tags."""

    start_tags: list[str] = Field(..., description="Start tags for extraction")
    end_tags: list[str] = Field(..., description="End tags for extraction")

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
