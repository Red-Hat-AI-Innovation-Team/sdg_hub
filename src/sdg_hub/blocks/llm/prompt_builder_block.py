# SPDX-License-Identifier: Apache-2.0
"""Prompt builder block for formatting prompts into structured chat messages or plain text.

This module provides the PromptBuilderBlock for handling LLM prompt formatting,
including conversion to OpenAI Messages format and template rendering.
"""

# Standard
from typing import Any, Dict, List, Literal, Optional

# Third Party
from datasets import Dataset
from jinja2 import Template, meta
from pydantic import BaseModel, Field, field_validator
import yaml

# Local
from ...logger_config import setup_logger
from ...utils.error_handling import TemplateValidationError
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


class ChatMessage(BaseModel):
    """Pydantic model for chat messages with proper validation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensure content is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()


class MessageTemplate(BaseModel):
    """Template for a chat message with Jinja2 template."""

    role: Literal["system", "user", "assistant", "tool"]
    content_template: Template

    model_config = {"arbitrary_types_allowed": True}


@BlockRegistry.register(
    "PromptBuilderBlock",
    "llm",
    "Formats prompts into structured chat messages or plain text using Jinja templates",
)
class PromptBuilderBlock(BaseBlock):
    """Block for formatting prompts into structured chat messages or plain text.

    This block takes input from dataset columns, applies Jinja templates from a YAML config
    containing a list of messages, and outputs either structured chat messages or formatted text.

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
        Path to YAML file containing list of message objects with 'role' and 'content' fields.
    format_as_messages : bool, optional
        Whether to format output as chat messages (default True).
        If True, outputs List[Dict[str, str]] with 'role' and 'content' keys.
        If False, outputs concatenated string with role prefixes.
    """

    prompt_config_path: str = Field(
        ..., description="Path to YAML file containing the Jinja template configuration"
    )
    format_as_messages: bool = Field(
        True, description="Whether to format output as chat messages"
    )

    # Internal fields for loaded config and templates
    prompt_config: Optional[List[Dict[str, Any]]] = Field(
        None, description="Loaded prompt configuration", exclude=True
    )
    message_templates: Optional[List[MessageTemplate]] = Field(
        None, description="Compiled message templates", exclude=True
    )

    @field_validator("output_cols", mode="after")
    @classmethod
    def validate_single_output_col(cls, v):
        """Validate that exactly one output column is specified."""
        if len(v) != 1:
            raise ValueError(
                f"PromptBuilderBlock expects exactly one output column, got {len(v)}: {v}"
            )
        return v

    def model_post_init(self, __context: Any) -> None:
        """Initialize the block after Pydantic validation."""
        # Load prompt configuration
        self.prompt_config = self._load_config(self.prompt_config_path)
        if self.prompt_config is None:
            raise ValueError(
                f"Failed to load prompt configuration from {self.prompt_config_path}"
            )

        # Compile message templates
        self._compile_message_templates()

    def _load_config(self, config_path: str) -> Optional[List[Dict[str, Any]]]:
        """Load the configuration file for this block."""
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                try:
                    config = yaml.safe_load(config_file)
                    if not isinstance(config, list):
                        raise ValueError(
                            "Template config must be a list of message objects"
                        )
                    return config
                except yaml.YAMLError as e:
                    logger.error(f"Error parsing YAML from {config_path}: {e}")
                    return None
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading config file {config_path}: {e}")
            return None

    def _compile_message_templates(self) -> None:
        """Compile Jinja templates for each message in the config."""
        self.message_templates = []

        if not self.prompt_config:
            raise ValueError("Prompt configuration cannot be empty")

        for i, message in enumerate(self.prompt_config):
            if "role" not in message or "content" not in message:
                raise ValueError(
                    f"Message {i} must have 'role' and 'content' fields. Got: {message.keys()}"
                )

            try:
                # Validate role using Pydantic's Literal validation
                message_template = MessageTemplate(
                    role=message["role"], content_template=Template(message["content"])
                )
                self.message_templates.append(message_template)
            except Exception as e:
                raise ValueError(
                    f"Failed to compile template for message {i}: {e}"
                ) from e

        # Validate that there's at least one user message
        user_messages = [msg for msg in self.message_templates if msg.role == "user"]
        if not user_messages:
            raise ValueError(
                "Template must contain at least one message with role='user' for proper conversation flow."
            )

        # Validate that the final message has role="user" for proper chat completion
        if self.message_templates and self.message_templates[-1].role != "user":
            raise ValueError(
                f"The final message must have role='user' for proper chat completion. "
                f"Got role='{self.message_templates[-1].role}' for the last message."
            )

    def _validate_custom(self, dataset: Dataset) -> None:
        if len(dataset) > 0:
            # Get required variables from all message templates
            required_vars = set()
            for msg_template in self.message_templates:
                template = msg_template.content_template
                ast = template.environment.parse(template.source)
                required_vars.update(meta.find_undeclared_variables(ast))

            sample = dataset[0]
            template_vars = self._resolve_template_vars(sample)
            missing_vars = required_vars - set(template_vars.keys())

            if missing_vars:
                raise TemplateValidationError(
                    block_name=self.block_name,
                    missing_variables=list(missing_vars),
                    available_variables=list(template_vars.keys()),
                )

    def _resolve_template_vars(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve template variables from dataset columns based on input_cols.

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

        if isinstance(self.input_cols, dict):
            # Map template variables to dataset columns
            for template_var, dataset_col in self.input_cols.items():
                if dataset_col in sample:
                    template_vars[template_var] = sample[dataset_col]
                else:
                    logger.warning(
                        f"Dataset column '{dataset_col}' not found in sample"
                    )
        else:
            # Use column names directly as template variables
            for col in self.input_cols:
                if col in sample:
                    template_vars[col] = sample[col]
                else:
                    logger.warning(f"Dataset column '{col}' not found in sample")

        return template_vars

    def _generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Generate formatted output for a single sample.

        1. Resolve columns needed for prompt templating
        2. Render each message template with the variables
        3. Format as messages or concatenated string based on format_as_messages

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample from dataset.

        Returns
        -------
        Dict[str, Any]
            Sample with formatted output added to specified output column.
        """
        output_col = self.output_cols[0]

        try:
            # Step 1: Resolve template variables from dataset columns
            template_vars = self._resolve_template_vars(sample)

            # Step 2: Render each message template
            rendered_messages = []
            for i, msg_template in enumerate(self.message_templates):
                try:
                    rendered_content = msg_template.content_template.render(
                        template_vars
                    ).strip()
                    if rendered_content:  # Only add non-empty messages
                        # Use Pydantic model for validation
                        chat_message = ChatMessage(
                            role=msg_template.role, content=rendered_content
                        )
                        rendered_messages.append(chat_message)
                except Exception as e:
                    logger.warning(f"Failed to render message {i}: {e}")
                    continue

            # Step 3: Format output based on format_as_messages setting
            if not rendered_messages:
                logger.warning(f"No valid messages generated for sample: {sample}")
                sample[output_col] = [] if self.format_as_messages else ""
            elif self.format_as_messages:
                # Convert to dict format for serialization
                sample[output_col] = [msg.model_dump() for msg in rendered_messages]
            else:
                # Concatenate all messages into a single string
                sample[output_col] = "\n\n".join(
                    [f"{msg.role}: {msg.content}" for msg in rendered_messages]
                )

        except Exception as e:
            logger.error(f"Failed to format sample: {e}")
            sample[output_col] = [] if self.format_as_messages else ""

        return sample

    def generate(self, samples: Dataset, **_kwargs: Any) -> Dataset:
        """Generate formatted output for all samples using dataset map.

        Parameters
        ----------
        samples : Dataset
            Input dataset containing samples to be formatted.
        **kwargs : Dict[str, Any]
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
