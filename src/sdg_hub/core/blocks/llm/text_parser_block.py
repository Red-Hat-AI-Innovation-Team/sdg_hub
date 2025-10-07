# SPDX-License-Identifier: Apache-2.0
"""Text parser block for parsing and post-processing text content.

This module provides the TextParserBlock for handling text parsing using
start/end tags, custom regex patterns, and cleanup operations.
"""

# Standard
from typing import Any, Optional
import re

# Third Party
from datasets import Dataset
from pydantic import Field, field_validator, model_validator

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "TextParserBlock",
    "llm",
    "Parses and post-processes text content using tags or regex patterns",
)
class TextParserBlock(BaseBlock):
    """Block for parsing and post-processing text content.

    This block handles text parsing using start/end tags, custom regex patterns,
    and cleanup operations. It expects exactly one input column containing text content
    as either a string or a list of strings.

    Attributes
    ----------
    block_name : str
        Unique identifier for this block instance.
    input_cols : Union[str, List[str], Dict[str, Any], None]
        Input column name(s) containing text content (str or List[str]). Must specify exactly one column.
    output_cols : Union[str, List[str], Dict[str, Any], None]
        Output column name(s) for parsed results.
    start_tags : List[str]
        List of start tags for tag-based parsing.
    end_tags : List[str]
        List of end tags for tag-based parsing.
    parsing_pattern : Optional[str]
        Regex pattern for custom parsing.
    parser_cleanup_tags : Optional[List[str]]
        List of tags to clean from parsed output.
    expand_lists : bool
        Whether to expand list inputs into individual rows (True) or preserve lists (False).
        Default is True for backward compatibility.
    """

    start_tags: list[str] = Field(
        default_factory=list, description="List of start tags for tag-based parsing"
    )
    end_tags: list[str] = Field(
        default_factory=list, description="List of end tags for tag-based parsing"
    )
    parsing_pattern: Optional[str] = Field(
        default=None, description="Regex pattern for custom parsing"
    )
    parser_cleanup_tags: Optional[list[str]] = Field(
        default=None, description="List of tags to clean from parsed output"
    )

    @field_validator("start_tags", "end_tags", mode="before")
    @classmethod
    def normalize_tags(cls, v):
        """Normalize tag lists to ensure they are always lists."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return v
        raise ValueError(f"Tags must be a string, list, or None, got {type(v)}")

    @field_validator("parser_cleanup_tags", mode="before")
    @classmethod
    def normalize_cleanup_tags(cls, v):
        """Normalize cleanup tags to ensure they are always lists when not None."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return v
        raise ValueError(f"Cleanup tags must be a string, list, or None, got {type(v)}")

    @model_validator(mode="after")
    def validate_parsing_configuration(self):
        """Validate that parsing configuration is consistent."""
        # Validate that at least one parsing method is configured
        has_regex = self.parsing_pattern is not None
        has_tags = bool(self.start_tags) or bool(self.end_tags)

        if not has_regex and not has_tags:
            raise ValueError(
                "TextParserBlock requires at least one parsing method: "
                "either 'parsing_pattern' (regex) or 'start_tags'/'end_tags' (tag-based parsing)"
            )

        # Validate tag parsing configuration
        if has_tags:
            if len(self.start_tags) != len(self.end_tags):
                raise ValueError(
                    f"start_tags and end_tags must have the same length. "
                    f"Got {len(self.start_tags)} start_tags and {len(self.end_tags)} end_tags"
                )

            # We can't validate against output_cols here since they might not be normalized yet
            # This validation will be moved to _validate_custom

        return self

    def _validate_custom(self, dataset: Dataset) -> None:
        """Validate TextParserBlock specific requirements.

        Parameters
        ----------
        dataset : Dataset
            The dataset to validate.

        Raises
        ------
        ValueError
            If TextParserBlock requirements are not met.
        """
        # Validate that we have exactly one input column
        if len(self.input_cols) == 0:
            raise ValueError("TextParserBlock expects at least one input column")
        if len(self.input_cols) > 1:
            logger.warning(
                f"TextParserBlock expects exactly one input column, but got {len(self.input_cols)}. "
                f"Using the first column: {self.input_cols[0]}"
            )

        # Validate tag parsing against output columns (can only be done after model creation)
        has_tags = bool(self.start_tags) or bool(self.end_tags)
        if has_tags and len(self.start_tags) != len(self.output_cols):
            raise ValueError(
                f"When using tag-based parsing, the number of tag pairs must match output_cols. "
                f"Got {len(self.start_tags)} tag pairs and {len(self.output_cols)} output columns"
            )

    def _extract_matches(
        self, text: str, start_tag: Optional[str], end_tag: Optional[str]
    ) -> list[str]:
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

        logger.debug(
            f"Regex parsing found {len(all_matches)} matches with pattern: {self.parsing_pattern}"
        )

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
            extracted = self._extract_matches(generated_string, start_tag, end_tag)
            matches[output_col] = extracted
            logger.debug(
                f"Tag parsing for '{output_col}' with tags '{start_tag}'/'{end_tag}' found {len(extracted)} matches"
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

    def _generate(self, sample: dict) -> list[dict]:
        input_column = self.input_cols[0]
        raw_output = sample[input_column]

        # Handle list inputs (e.g., multiple text strings to process)
        if isinstance(raw_output, list):
            if not raw_output:
                logger.warning(f"Input column '{input_column}' contains empty list")
                return []

            # Parse each text string in the list and collect results as lists
            all_parsed_outputs = {col: [] for col in self.output_cols}
            valid_responses = 0

            for i, message in enumerate(raw_output):
                # Ensure each item in the list is a string
                if not isinstance(message, str):
                    logger.warning(
                        f"List item {i} in column '{input_column}' is not a string: {type(message)}. "
                        f"Expected List[str], skipping this item."
                    )
                    continue

                if not message:
                    logger.warning(f"List item {i} in column '{input_column}' is empty")
                    continue

                parsed_outputs = self._parse(message)

                if not parsed_outputs or not any(
                    len(value) > 0 for value in parsed_outputs.values()
                ):
                    logger.warning(
                        f"Failed to parse content from list item {i}. Text length: {len(message)}, "
                        f"parsing method: {'regex' if self.parsing_pattern else 'tags'}"
                    )
                    continue

                valid_responses += 1
                # Collect all parsed values for each column as lists
                for col in self.output_cols:
                    all_parsed_outputs[col].extend(parsed_outputs.get(col, []))

            if valid_responses == 0:
                return []

            # Return single row with lists as values
            return [{**sample, **all_parsed_outputs}]
        # Handle string inputs
        elif isinstance(raw_output, str):
            if not raw_output:
                logger.warning(f"Input column '{input_column}' contains empty string")
                return []

            parsed_outputs = self._parse(raw_output)

            if not parsed_outputs or not any(
                len(value) > 0 for value in parsed_outputs.values()
            ):
                logger.warning(
                    f"Failed to parse any content from input. Text length: {len(raw_output)}, "
                    f"parsing method: {'regex' if self.parsing_pattern else 'tags'}"
                )
                return []

            result = []
            max_length = max(len(value) for value in parsed_outputs.values())
            for values in zip(*(lst[:max_length] for lst in parsed_outputs.values())):
                result_row = {**sample, **dict(zip(parsed_outputs.keys(), values))}
                result.append(result_row)

            return result

        else:
            logger.warning(
                f"Input column '{input_column}' contains invalid data type: {type(raw_output)}. "
                f"Expected str or List[str]"
            )
            return []

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        logger.debug(f"Parsing outputs for {len(samples)} samples")
        if len(samples) == 0:
            logger.warning("No samples to parse, returning empty dataset")
            return Dataset.from_list([])

        new_data = []
        for sample in samples:
            new_data.extend(self._generate(sample))
        return Dataset.from_list(new_data)
