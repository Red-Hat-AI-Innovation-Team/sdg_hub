# SPDX-License-Identifier: Apache-2.0
"""Prompt builder block for formatting prompts into structured chat messages or plain text.

This module provides the PromptBuilderBlock for handling LLM prompt formatting,
including conversion to OpenAI Messages format and template rendering.
"""

# Standard
from typing import Any, Dict, List, Optional, Union

# Third Party
from datasets import Dataset
from jinja2 import Template, meta
import yaml

# Local
from ...logger_config import setup_logger
from ..registry import BlockRegistry
from ...utils.error_handling import TemplateValidationError
from ..base import BaseBlock

logger = setup_logger(__name__)


@BlockRegistry.register(
    "PromptBuilderBlock",
    "llm",
    "Formats prompts into structured chat messages or plain text using Jinja templates",
)
class PromptBuilderBlock(BaseBlock):
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
        **kwargs: Any,
    ) -> None:
        super().__init__(
            block_name=block_name,
            input_cols=input_cols,
            output_cols=output_cols,
            **kwargs,
        )

        # Validate exactly one output column
        if len(self.output_cols) != 1:
            raise ValueError(
                f"PromptBuilderBlock expects exactly one output column, got {len(self.output_cols)}: {self.output_cols}"
            )

        # Process input column specifications
        self._process_input_cols()

        self.prompt_config_path = prompt_config_path
        self.format_as_messages = format_as_messages
        self.default_role = default_role

        # Load prompt configuration
        self.prompt_config = self._load_config(prompt_config_path)
        if self.prompt_config is None:
            raise ValueError(
                f"Failed to load prompt configuration from {prompt_config_path}"
            )

        # Build the prompt structure from config
        prompt_struct = """{introduction}\n{principles}\n{examples}\n{generation}"""

        # Filter out None values and create the template
        filtered_config = {
            k: (v if v is not None else "") for k, v in self.prompt_config.items()
        }
        self.prompt_template = Template(prompt_struct.format(**filtered_config))

    def _load_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """Load the configuration file for this block."""
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                try:
                    return yaml.safe_load(config_file)
                except yaml.YAMLError as e:
                    logger.error(f"Error parsing YAML from {config_path}: {e}")
                    return None
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading config file {config_path}: {e}")
            return None

    def _validate_custom(self, dataset: Dataset) -> None:
        if len(dataset) > 0:
            # Get required variables directly from template AST
            ast = self.prompt_template.environment.parse(self.prompt_template.source)
            required_vars = meta.find_undeclared_variables(ast)

            sample = dataset[0]
            template_vars = self._resolve_template_vars(sample)
            missing_vars = required_vars - set(template_vars.keys())

            if missing_vars:
                raise TemplateValidationError(
                    block_name=self.block_name,
                    missing_variables=list(missing_vars),
                    available_variables=list(template_vars.keys()),
                )

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

            # Step 2: Render the template (validation is handled by _validate_custom)
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

                # Use the single output column (validated to be exactly one)
                output_col = self.output_cols[0]
                sample[output_col] = formatted_output
            else:
                logger.warning(f"Sample produced no content: {sample}")
                output_col = self.output_cols[0]
                sample[output_col] = [] if self.format_as_messages else ""

        except Exception as e:
            logger.warning(f"Failed to format sample: {sample}. Error: {e}")
            output_col = self.output_cols[0]
            sample[output_col] = [] if self.format_as_messages else ""

        return sample

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
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
