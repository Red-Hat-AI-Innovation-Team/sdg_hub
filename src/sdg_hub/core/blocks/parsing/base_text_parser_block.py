# SPDX-License-Identifier: Apache-2.0
"""Shared base class for tag-based and regex-based text parser blocks."""

from abc import abstractmethod
from itertools import chain
from typing import Any, Optional, cast

from pydantic import Field
import pandas as pd

from ...utils.logger_config import setup_logger
from ..base import BaseBlock

logger = setup_logger(__name__)


class BaseTextParserBlock(BaseBlock):
    """Base class for text parser blocks that parse individual text inputs.

    Subclasses must implement ``_parse_single_text`` to define how a single
    text string is parsed into rows of column values.
    """

    _flow_requires_jsonl_tmp: bool = True
    block_type: str = "parser"

    parser_cleanup_tags: Optional[list[str]] = Field(
        default=None, description="Tags to remove from extracted content"
    )

    def _clean(self, value: str) -> str:
        for tag in self.parser_cleanup_tags or []:
            value = value.replace(tag, "")
        return value

    @abstractmethod
    def _parse_single_text(self, sample: dict, text: str) -> list[dict]: ...

    def _accumulate_parsed_rows(
        self, rows: list[dict], all_parsed: dict[str, list[str]]
    ) -> None:
        output_cols = cast(list[str], self.output_cols)
        for row in rows:
            for col in output_cols:
                if col in row:
                    all_parsed[col].append(row[col])

    def _parse_list_input(self, sample: dict, items: list) -> list[dict]:
        output_cols = cast(list[str], self.output_cols)
        all_parsed: dict[str, list[str]] = {col: [] for col in output_cols}
        valid = 0
        for item in items:
            if not isinstance(item, str) or not item:
                continue
            rows = self._parse_single_text(sample, item)
            if rows:
                valid += 1
                self._accumulate_parsed_rows(rows, all_parsed)
        if valid == 0:
            return []
        return [{**sample, **all_parsed}]

    def _parse_row(self, sample: dict) -> list[dict]:
        input_cols = cast(list[str], self.input_cols)
        text = sample[input_cols[0]]

        if isinstance(text, list):
            if not text:
                logger.warning(f"Input column '{input_cols[0]}' contains empty list")
                return []
            return self._parse_list_input(sample, text)

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
