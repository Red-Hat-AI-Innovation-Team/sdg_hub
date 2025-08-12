# SPDX-License-Identifier: Apache-2.0
"""Prompt builder block for formatting prompts into structured chat messages or plain text.

This module provides the PromptBuilderBlock for handling LLM prompt formatting,
including conversion to OpenAI Messages format and template rendering.
"""

# Standard
from typing import Any, Literal, Optional

# Third Party
from datasets import Dataset
from jinja2 import Template, meta
from pydantic import BaseModel, Field, field_validator
import yaml

# Local
from ...utils.error_handling import TemplateValidationError
from ...utils.logger_config import setup_logger
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
    """Template for a chat message with Jinja2 template and original source."""

    role: Literal["system", "user", "assistant", "tool"]
    content_template: Template
    original_source: str

    model_config = {"arbitrary_types_allowed": True}


class PromptTemplateConfig:
    """Self-contained class for loading and validating YAML prompt configurations."""

    def __init__(self, config_path: str):
        """Initialize with path to YAML config file."""
        self.config_path = config_path
        self.message_templates: list[MessageTemplate] = []
        self._load_and_validate()

    def _load_and_validate(self) -> None:
        """Load YAML config and validate format."""
        try:
            with open(self.config_path, encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)

                if not isinstance(config, list):
                    raise ValueError(
                        "Template config must be a list of message objects"
                    )

                if not config:
                    raise ValueError("Prompt configuration cannot be empty")

                self._compile_templates(config)
                self._validate_message_flow()

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML from {self.config_path}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error reading config file {self.config_path}: {e}"
            )
            raise

    def _compile_templates(self, config: list[dict[str, Any]]) -> None:
        """Compile Jinja templates for each message in the config."""
        for i, message in enumerate(config):
            if "role" not in message or "content" not in message:
                raise ValueError(
                    f"Message {i} must have 'role' and 'content' fields. Got: {message.keys()}"
                )

            try:
                content_source = message["content"]
                message_template = MessageTemplate(
                    role=message["role"],
                    content_template=Template(content_source),
                    original_source=content_source,
                )
                self.message_templates.append(message_template)
            except Exception as e:
                raise ValueError(
                    f"Failed to compile template for message {i}: {e}"
                ) from e

    def _validate_message_flow(self) -> None:
        """Validate that message flow is appropriate for chat completion."""
        user_messages = [msg for msg in self.message_templates if msg.role == "user"]
        if not user_messages:
            raise ValueError(
                "Template must contain at least one message with role='user' for proper conversation flow."
            )

        if self.message_templates and self.message_templates[-1].role != "user":
            raise ValueError(
                f"The final message must have role='user' for proper chat completion. "
                f"Got role='{self.message_templates[-1].role}' for the last message."
            )

    def get_message_templates(self) -> list[MessageTemplate]:
        """Return the compiled message templates."""
        return self.message_templates


class PromptRenderer:
    """Handles rendering of message templates with variable substitution."""

    def __init__(self, message_templates: list[MessageTemplate]):
        """Initialize with a list of message templates."""
        self.message_templates = message_templates

    def get_required_variables(self) -> set:
        """Extract all required variables from message templates."""
        required_vars = set()
        for msg_template in self.message_templates:
            # Parse the original source to find undeclared variables
            # Use the template's existing environment to ensure consistency
            ast = msg_template.content_template.environment.parse(
                msg_template.original_source
            )
            required_vars.update(meta.find_undeclared_variables(ast))
        return required_vars

    def resolve_template_vars(
        self, sample: dict[str, Any], input_cols
    ) -> dict[str, Any]:
        """Resolve template variables from dataset columns based on input_cols.

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample from dataset.
        input_cols : Union[str, List[str], Dict[str, str]]
            Input column specification - now maps dataset columns to template variables.

        Returns
        -------
        Dict[str, Any]
            Template variables mapped from dataset columns.
        """
        template_vars = {}

        if isinstance(input_cols, dict):
            # Map dataset columns to template variables
            for dataset_col, template_var in input_cols.items():
                if dataset_col in sample:
                    template_vars[template_var] = sample[dataset_col]
                else:
                    logger.warning(
                        f"Dataset column '{dataset_col}' not found in sample"
                    )
        else:
            # Use column names directly as template variables
            for col in input_cols:
                if col in sample:
                    template_vars[col] = sample[col]
                else:
                    logger.warning(f"Dataset column '{col}' not found in sample")

        return template_vars

    def render_messages(self, template_vars: dict[str, Any]) -> list[ChatMessage]:
        """Render all message templates with the given variables.

        Parameters
        ----------
        template_vars : Dict[str, Any]
            Variables to substitute in templates.

        Returns
        -------
        List[ChatMessage]
            List of rendered and validated chat messages.
        """
        rendered_messages = []

        for i, msg_template in enumerate(self.message_templates):
            try:
                rendered_content = msg_template.content_template.render(
                    template_vars
                ).strip()
                if rendered_content:  # Only add non-empty messages
                    chat_message = ChatMessage(
                        role=msg_template.role, content=rendered_content
                    )
                    rendered_messages.append(chat_message)
            except Exception as e:
                logger.warning(f"Failed to render message {i}: {e}")
                continue

        return rendered_messages


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
        - Dict[str, str]: Mapping from dataset column names to template variables
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

    # Internal fields for configuration and renderer
    prompt_template_config: Optional[PromptTemplateConfig] = Field(
        None, description="Loaded prompt template configuration", exclude=True
    )
    prompt_renderer: Optional[PromptRenderer] = Field(
        None, description="Prompt renderer instance", exclude=True
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
        # Load and validate prompt configuration
        self.prompt_template_config = PromptTemplateConfig(self.prompt_config_path)

        # Initialize prompt renderer
        message_templates = self.prompt_template_config.get_message_templates()
        self.prompt_renderer = PromptRenderer(message_templates)

    def _validate_custom(self, dataset: Dataset) -> None:
        if len(dataset) > 0:
            # Get required variables from all message templates
            required_vars = self.prompt_renderer.get_required_variables()

            sample = dataset[0]
            template_vars = self.prompt_renderer.resolve_template_vars(
                sample, self.input_cols
            )
            missing_vars = required_vars - set(template_vars.keys())

            if missing_vars:
                raise TemplateValidationError(
                    block_name=self.block_name,
                    missing_variables=list(missing_vars),
                    available_variables=list(template_vars.keys()),
                )

    def _generate(self, sample: dict[str, Any]) -> dict[str, Any]:
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
            template_vars = self.prompt_renderer.resolve_template_vars(
                sample, self.input_cols
            )

            # Step 2: Render messages using the prompt renderer
            rendered_messages = self.prompt_renderer.render_messages(template_vars)

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
        logger.debug(f"Formatting prompts for {len(samples)} samples")

        # Use dataset map for efficient processing
        formatted_dataset = samples.map(self._generate)

        logger.debug(f"Successfully formatted {len(formatted_dataset)} samples")
        return formatted_dataset
