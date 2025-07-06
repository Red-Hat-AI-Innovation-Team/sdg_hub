# SPDX-License-Identifier: Apache-2.0
"""LLM chat completion utilities for input formatting and output parsing.

This module provides blocks for handling LLM chat completions, including:
- PromptBuilderBlock: Format prompts into structured chat messages or plain text
- TextParserBlock: Parse and post-process LLM outputs
"""

# Standard
import re
from typing import Any, Dict, List, Optional, Union

# Third Party
from datasets import Dataset
from jinja2 import Template

# Local
from ..logger_config import setup_logger
from ..registry import BlockRegistry
from .block import Block

logger = setup_logger(__name__)


@BlockRegistry.register("PromptBuilderBlock")
class PromptBuilderBlock(Block):
    """Block for formatting prompts into structured chat messages or plain text.

    This block takes input from dataset columns, applies a Jinja template from a YAML config,
    and outputs either structured chat messages or formatted text.

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str], Dict[str, str]]
        Input column specification:
        - str: Single column name
        - List[str]: List of column names
        - Dict[str, str]: Mapping from template variables to dataset column names
    output_cols : str
        Name of the output column where formatted content will be saved.
    prompt_config_path : str
        Path to YAML file containing the Jinja template configuration.
    format_as_messages : bool, optional
        Whether to format output as chat messages (default True).
        If True, outputs List[Dict[str, str]] with 'role' and 'content' keys.
        If False, outputs the rendered template as a plain string.
    default_role : str, optional
        Default role for messages when format_as_messages=True, by default "user".
    """

    def __init__(
        self,
        block_name: str,
        input_cols: Union[str, List[str], Dict[str, str]],
        output_cols: str,
        prompt_config_path: str,
        format_as_messages: bool = True,
        default_role: str = "user",
    ) -> None:
        super().__init__(block_name)
        self.input_cols = input_cols
        # Process input column specifications
        self._process_input_cols()
        self.output_cols = (
            output_cols if isinstance(output_cols, str) else output_cols[0]
        )

        self.prompt_config_path = prompt_config_path
        self.format_as_messages = format_as_messages
        self.default_role = default_role

        # Load prompt configuration
        self.prompt_config = self._load_config(prompt_config_path)

        # Build the prompt structure from config
        prompt_struct = """{introduction}\n{principles}\n{examples}\n{generation}"""

        # Filter out None values and create the template
        filtered_config = {
            k: (v if v is not None else "") for k, v in self.prompt_config.items()
        }
        self.prompt_template = Template(prompt_struct.format(**filtered_config))

    def _process_input_cols(self) -> None:
        """Process input column specifications into standardized format."""
        if isinstance(self.input_cols, str):
            self.input_col_map = {self.input_cols: self.input_cols}
        elif isinstance(self.input_cols, list):
            self.input_col_map = {col: col for col in self.input_cols}
        elif isinstance(self.input_cols, dict):
            self.input_col_map = self.input_cols
        else:
            raise ValueError("input_cols must be str, list, or dict")

    def _resolve_template_vars(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve template variables from dataset columns based on input_col_map.

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample from dataset.

        Returns
        -------
        Dict[str, Any]
            Template variables mapped from dataset columns.
        """
        template_vars = {}
        for template_var, dataset_col in self.input_col_map.items():
            if dataset_col in sample:
                template_vars[template_var] = sample[dataset_col]
            else:
                logger.warning(f"Dataset column '{dataset_col}' not found in sample")
        return template_vars

    def _validate_message_role(self, role: str) -> str:
        """Validate and normalize message role.

        Parameters
        ----------
        role : str
            The role to validate.

        Returns
        -------
        str
            Validated role, defaults to 'user' if invalid.
        """
        valid_roles = {"system", "user", "assistant", "tool"}
        if role.lower() in valid_roles:
            return role.lower()
        else:
            logger.warning(f"Invalid role '{role}', defaulting to 'user'")
            return "user"

    def _format_message_content(self, content: Any) -> Union[str, List[Dict[str, Any]]]:
        """Format message content for OpenAI API.

        Parameters
        ----------
        content : Any
            Content to format (string, dict, or list).

        Returns
        -------
        Union[str, List[Dict[str, Any]]]
            Formatted content compatible with OpenAI API.
        """
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, dict):
            # Handle structured content blocks (e.g., images, tool calls)
            return [content]
        elif isinstance(content, list):
            # Handle multiple content blocks
            formatted_blocks = []
            for block in content:
                if isinstance(block, dict):
                    formatted_blocks.append(block)
                elif isinstance(block, str) and block.strip():
                    formatted_blocks.append({"type": "text", "text": block.strip()})
            return formatted_blocks if formatted_blocks else ""
        else:
            # Convert other types to string
            return str(content).strip()

    def _create_openai_message(
        self, role: str, content: Any
    ) -> Optional[Dict[str, Any]]:
        """Create a single OpenAI message with proper validation.

        Parameters
        ----------
        role : str
            Message role (system, user, assistant, tool).
        content : Any
            Message content.

        Returns
        -------
        Optional[Dict[str, Any]]
            OpenAI message dict or None if content is empty.
        """
        validated_role = self._validate_message_role(role)
        formatted_content = self._format_message_content(content)

        # Skip empty content
        if not formatted_content:
            return None

        message = {"role": validated_role, "content": formatted_content}

        # Add additional fields for specific roles if needed
        if validated_role == "tool":
            # Tool messages require tool_call_id (would need to be provided)
            logger.warning(
                "Tool messages require tool_call_id - this may cause API errors"
            )

        return message

    def _convert_to_openai_messages(
        self,
        rendered_content: str,
        template_vars: Dict[str, Any],
        system_message: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convert rendered content to OpenAI Messages format with robust handling.

        This function creates a properly formatted list of OpenAI messages with:
        - Validation of roles and content
        - Support for complex content blocks
        - Proper handling of system messages
        - Error handling and logging

        Parameters
        ----------
        rendered_content : str
            The rendered template content.
        template_vars : Dict[str, Any]
            Template variables used for rendering.
        system_message : Optional[str], optional
            Optional system message to prepend.

        Returns
        -------
        List[Dict[str, Any]]
            List of OpenAI-compatible message dictionaries.
        """
        messages = []

        # Handle system message from config or parameter
        system_content = system_message or self.prompt_config.get("system", "")
        if system_content and system_content.strip():
            try:
                # Render system message with template variables if it contains template syntax
                if "{{" in system_content or "{%" in system_content:
                    system_template = Template(system_content)
                    rendered_system = system_template.render(template_vars)
                else:
                    rendered_system = system_content

                system_msg = self._create_openai_message("system", rendered_system)
                if system_msg:
                    messages.append(system_msg)
            except Exception as e:
                logger.warning(f"Failed to render system message: {e}")

        # Handle main content
        if rendered_content and rendered_content.strip():
            main_msg = self._create_openai_message(self.default_role, rendered_content)
            if main_msg:
                messages.append(main_msg)

        # Validate final message list
        if not messages:
            logger.warning("No valid messages generated")

        return messages

    def _generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Generate formatted output for a single sample.

        1. Resolve columns needed for prompt templating
        2. Populate template with values to get formatted string
        3. Convert to OpenAI Messages format if required

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample from dataset.

        Returns
        -------
        Dict[str, Any]
            Sample with formatted output added to specified output column.
        """
        try:
            # Step 1: Resolve template variables from dataset columns
            template_vars = self._resolve_template_vars(sample)

            # Step 2: Validate and render the template
            if self._validate(self.prompt_template, template_vars):
                rendered_content = self.prompt_template.render(template_vars).strip()

                if rendered_content:
                    # Step 3: Format output based on format_as_messages setting
                    if self.format_as_messages:
                        # Convert to OpenAI Messages format with robust handling
                        formatted_output = self._convert_to_openai_messages(
                            rendered_content, template_vars
                        )
                    else:
                        # Keep as plain string
                        formatted_output = rendered_content

                    sample[self.output_cols] = formatted_output
                else:
                    logger.warning(f"Sample produced no content: {sample}")
                    sample[self.output_cols] = [] if self.format_as_messages else ""
            else:
                logger.warning(f"Sample failed template validation: {sample}")
                sample[self.output_cols] = [] if self.format_as_messages else ""

        except Exception as e:
            logger.warning(f"Failed to format sample: {sample}. Error: {e}")
            sample[self.output_cols] = [] if self.format_as_messages else ""

        return sample

    def generate(self, samples: Dataset) -> Dataset:
        """Generate formatted output for all samples using dataset map.

        Parameters
        ----------
        samples : Dataset
            Input dataset containing samples to be formatted.
        **gen_kwargs : Dict[str, Any]
            Additional keyword arguments (unused in this block).

        Returns
        -------
        Dataset
            Dataset with the formatted output added to the specified column.
        """
        logger.debug("Formatting prompts for {} samples".format(len(samples)))

        # Use dataset map for efficient processing
        formatted_dataset = samples.map(self._generate)

        logger.debug(f"Successfully formatted {len(formatted_dataset)} samples")
        return formatted_dataset


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
