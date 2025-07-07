# SPDX-License-Identifier: Apache-2.0
"""Text parser block for parsing and post-processing LLM outputs.

This module provides the TextParserBlock for handling output parsing using
start/end tags, custom regex patterns, and cleanup operations.
"""

# Standard
import re
from typing import Any, Dict, List, Optional, Union

# Third Party
from datasets import Dataset

# Local
from ...logger_config import setup_logger
from ...registry import BlockRegistry
from ..block import Block

logger = setup_logger(__name__)


@BlockRegistry.register("TextParserBlock")
class TextParserBlock(Block):
    """Block for parsing and post-processing LLM outputs.

    This block handles output parsing using start/end tags, custom regex patterns,
    and cleanup operations. It duplicates the parsing functionality from LLMBlock.

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Input column name(s) containing raw LLM output.
    output_cols : Union[str, List[str]]
        Output column name(s) for parsed results.
    start_tags : List[str], optional
        List of start tags for tag-based parsing. Default is [].
    end_tags : List[str], optional
        List of end tags for tag-based parsing. Default is [].
    parsing_pattern : Optional[str], optional
        Regex pattern for custom parsing. Default is None.
    parser_cleanup_tags : Optional[List[str]], optional
        List of tags to clean from parsed output. Default is None.
    """

    def __init__(
        self,
        block_name: str,
        input_cols: Union[str, List[str]],
        output_cols: Union[str, List[str]],
        start_tags: Optional[List[str]] = None,
        end_tags: Optional[List[str]] = None,
        parsing_pattern: Optional[str] = None,
        parser_cleanup_tags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(block_name)
        self.input_cols = [input_cols] if isinstance(input_cols, str) else input_cols
        self.output_cols = (
            [output_cols] if isinstance(output_cols, str) else output_cols
        )
        self.start_tags = start_tags or []
        self.end_tags = end_tags or []
        self.parsing_pattern = parsing_pattern
        self.parser_cleanup_tags = parser_cleanup_tags

        # Validate the block configuration
        if len(self.input_cols) == 0:
            raise ValueError("TextParserBlock expects at least one input column")
        elif len(self.input_cols) > 1:
            logger.warning(
                f"TextParserBlock expects exactly one input column, but got {len(self.input_cols)}. "
                f"Using the first column: {self.input_cols[0]}"
            )

    def _extract_matches(
        self, text: str, start_tag: Optional[str], end_tag: Optional[str]
    ) -> List[str]:
        if not text:
            return []
        if not start_tag and not end_tag:
            return [text.strip()]

        pattern = ""
        if start_tag:
            pattern += re.escape(start_tag)
        pattern += r"(.*?)"
        if end_tag:
            pattern += re.escape(end_tag)
        elif start_tag:
            pattern += "$"

        return [match.strip() for match in re.findall(pattern, text, re.DOTALL)]

    def _parse(self, generated_string: str) -> dict[str, list[str]]:
        if self.parsing_pattern is not None:
            return self._parse_with_regex(generated_string)
        return self._parse_with_tags(generated_string)

    def _parse_with_regex(self, generated_string: str) -> dict[str, list[str]]:
        """Parse using regex pattern."""
        if self.parsing_pattern is None:
            raise ValueError("parsing_pattern is required for regex parsing")
        pattern = re.compile(self.parsing_pattern, re.DOTALL)
        all_matches = pattern.findall(generated_string)
        matches: dict[str, list[str]] = {
            column_name: [] for column_name in self.output_cols
        }

        if all_matches and isinstance(all_matches[0], tuple):
            return self._process_tuple_matches(all_matches, matches)
        return self._process_single_matches(all_matches, matches)

    def _parse_with_tags(self, generated_string: str) -> dict[str, list[str]]:
        """Parse using start/end tags."""
        matches: dict[str, list[str]] = {
            column_name: [] for column_name in self.output_cols
        }

        for start_tag, end_tag, output_col in zip(
            self.start_tags, self.end_tags, self.output_cols
        ):
            matches[output_col] = self._extract_matches(
                generated_string, start_tag, end_tag
            )
        return matches

    def _process_tuple_matches(
        self, all_matches: list, matches: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Process regex matches that are tuples."""
        for match in all_matches:
            for column_name, value in zip(self.output_cols, match):
                value = self._clean_value(value.strip())
                matches[column_name].append(value)
        return matches

    def _process_single_matches(
        self, all_matches: list, matches: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Process regex matches that are single values."""
        cleaned_matches = [self._clean_value(match.strip()) for match in all_matches]
        matches[self.output_cols[0]] = cleaned_matches
        return matches

    def _clean_value(self, value: str) -> str:
        """Clean value by removing cleanup tags."""
        if self.parser_cleanup_tags:
            for clean_tag in self.parser_cleanup_tags:
                value = value.replace(clean_tag, "")
        return value

    def _generate(self, sample: dict) -> List[dict]:
        input_column = self.input_cols[0]
        if input_column not in sample:
            logger.warning(
                f"Input column '{input_column}' not found in sample: {sample}"
            )
            return []

        raw_output = sample[input_column]
        parsed_outputs = self._parse(raw_output)

        if not parsed_outputs or not any(
            len(value) > 0 for value in parsed_outputs.values()
        ):
            return []

        result = []
        max_length = max(len(value) for value in parsed_outputs.values())
        for values in zip(*(lst[:max_length] for lst in parsed_outputs.values())):
            result.append({**sample, **dict(zip(parsed_outputs.keys(), values))})
        return result

    def generate(self, samples: Dataset) -> Dataset:
        logger.debug(f"Parsing outputs for {len(samples)} samples")
        if len(samples) == 0:
            logger.warning("No samples to parse, returning empty dataset")
            return Dataset.from_list([])

        new_data = []
        for sample in samples:
            new_data.extend(self._generate(sample))
        return Dataset.from_list(new_data)
