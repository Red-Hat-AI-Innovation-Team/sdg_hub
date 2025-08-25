# SPDX-License-Identifier: Apache-2.0
"""Composite block for question verification and quality assessment.

This module provides the VerifyQuestionBlock that encapsulates the complete
question verification workflow, combining prompt building, LLM chat, text parsing,
and filtering into a single block for simplified configuration.

The block uses dynamic parameter forwarding to support all parameters from its
internal blocks (LLMChatBlock, PromptBuilderBlock, TextParserBlock, ColumnValueFilterBlock)
without requiring manual parameter declarations.
"""

# Standard
from typing import Any, Optional

# Third Party
from datasets import Dataset
from pydantic import ConfigDict, Field, field_validator

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..filtering.column_value_filter import ColumnValueFilterBlock
from ..llm.llm_chat_block import LLMChatBlock
from ..llm.prompt_builder_block import PromptBuilderBlock
from ..llm.text_parser_block import TextParserBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "VerifyQuestionBlock",
    "evaluation",
    "Composite block for question verification and quality assessment",
)
class VerifyQuestionBlock(BaseBlock):
    """Composite block for question verification workflow.

    This block combines four separate blocks into a single cohesive verification block:
    1. PromptBuilderBlock - builds verification prompt from question
    2. LLMChatBlock - generates question quality assessment using LLM
    3. TextParserBlock - parses explanation and rating from raw output
    4. ColumnValueFilterBlock - filters based on verification rating

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : List[str]
        Input columns: ["question"]
    output_cols : List[str]
        Output columns: ["verification_explanation", "verification_rating"]
    **kwargs : Any
        All parameters from internal blocks are supported with automatic forwarding:

        - **LLMChatBlock parameters**: model, api_base, api_key, temperature, max_tokens,
          top_p, frequency_penalty, presence_penalty, stop, seed, response_format, stream,
          n, logprobs, top_logprobs, user, extra_headers, extra_body, timeout, max_retries,
          async_mode, provider_specific, and more.

        - **PromptBuilderBlock parameters**: prompt_config_path (REQUIRED), format_as_messages (default: True).

        - **TextParserBlock parameters**: start_tags (default: ["[Start of Explanation]", "[Start of Rating]"]),
          end_tags (default: ["[End of Explanation]", "[End of Rating]"]), parsing_pattern,
          parser_cleanup_tags, expand_lists.

        - **ColumnValueFilterBlock parameters**: filter_value (default: 1.0), operation (default: "ge"),
          convert_dtype (default: "float").

        See the respective block documentation for complete parameter details.
    """

    model_config = ConfigDict(
        extra="allow"  # Allow extra fields for dynamic parameter forwarding
    )

    # No composite-specific configuration - all parameters are forwarded dynamically

    # Store parameters for internal blocks
    llm_params: dict[str, Any] = Field(default_factory=dict, exclude=True)
    prompt_params: dict[str, Any] = Field(default_factory=dict, exclude=True)
    parser_params: dict[str, Any] = Field(default_factory=dict, exclude=True)
    filter_params: dict[str, Any] = Field(default_factory=dict, exclude=True)

    # Internal blocks - excluded from serialization
    prompt_builder: Optional[PromptBuilderBlock] = Field(default=None, exclude=True)
    llm_chat: Optional[LLMChatBlock] = Field(default=None, exclude=True)
    text_parser: Optional[TextParserBlock] = Field(default=None, exclude=True)
    filter_block: Optional[ColumnValueFilterBlock] = Field(default=None, exclude=True)

    @field_validator("input_cols")
    @classmethod
    def validate_input_cols(cls, v):
        """Validate that input columns are exactly ["question"]."""
        expected = ["question"]
        if v != expected:
            raise ValueError(
                f"VerifyQuestionBlock expects input_cols={expected}, got {v}"
            )
        return v

    @field_validator("output_cols")
    @classmethod
    def validate_output_cols(cls, v):
        """Validate that output columns are exactly ["verification_explanation", "verification_rating"]."""
        expected = [
            "verification_explanation",
            "verification_rating",
        ]
        if v != expected:
            raise ValueError(
                f"VerifyQuestionBlock expects output_cols={expected}, got {v}"
            )
        return v

    def __init__(self, **kwargs):
        """Initialize with dynamic parameter forwarding."""
        # No composite-specific parameters - everything is forwarded dynamically
        composite_params = {}

        # Forward parameters to appropriate internal blocks
        llm_params = {k: v for k, v in kwargs.items() if k in LLMChatBlock.model_fields}
        prompt_params = {
            k: v for k, v in kwargs.items() if k in PromptBuilderBlock.model_fields
        }
        parser_params = {
            k: v for k, v in kwargs.items() if k in TextParserBlock.model_fields
        }
        filter_params = {
            k: v for k, v in kwargs.items() if k in ColumnValueFilterBlock.model_fields
        }

        # Keep only BaseBlock fields for super().__init__
        base_params = {k: v for k, v in kwargs.items() if k in BaseBlock.model_fields}
        base_params.update(composite_params)
        base_params["llm_params"] = llm_params
        base_params["prompt_params"] = prompt_params
        base_params["parser_params"] = parser_params
        base_params["filter_params"] = filter_params

        # Initialize parent with all valid parameters
        super().__init__(**base_params)

        # Create internal blocks with forwarded parameters
        self._create_internal_blocks()

        # Log initialization only when model is configured
        model = self.llm_params.get("model")
        if model:
            logger.info(
                f"Initialized VerifyQuestionBlock '{self.block_name}' with model '{model}'",
                extra={
                    "block_name": self.block_name,
                    "model": model,
                    "async_mode": self.llm_params.get("async_mode", True),
                    "filter_value": self.filter_params.get("filter_value", 1.0),
                },
            )

    def _create_internal_blocks(self) -> None:
        """Create and configure the internal blocks using dynamic parameter forwarding."""
        # 1. PromptBuilderBlock
        prompt_kwargs = {
            **self.prompt_params,  # Forward all prompt parameters dynamically
            "block_name": f"{self.block_name}_prompt_builder",
            "input_cols": ["question"],
            "output_cols": ["verify_question_prompt"],
            "prompt_config_path": self.prompt_params.get("prompt_config_path"),
            "format_as_messages": self.prompt_params.get("format_as_messages", True),
        }
        self.prompt_builder = PromptBuilderBlock(**prompt_kwargs)

        # 2. LLMChatBlock
        llm_kwargs = {
            **self.llm_params,  # Forward all LLM parameters dynamically
            "block_name": f"{self.block_name}_llm_chat",
            "input_cols": ["verify_question_prompt"],
            "output_cols": ["raw_verify_question"],
        }
        self.llm_chat = LLMChatBlock(**llm_kwargs)

        # 3. TextParserBlock
        text_parser_kwargs = {
            **self.parser_params,  # Forward all parser parameters dynamically
            "block_name": f"{self.block_name}_text_parser",
            "input_cols": ["raw_verify_question"],
            "output_cols": ["verification_explanation", "verification_rating"],
            "start_tags": self.parser_params.get(
                "start_tags", ["[Start of Explanation]", "[Start of Rating]"]
            ),
            "end_tags": self.parser_params.get(
                "end_tags", ["[End of Explanation]", "[End of Rating]"]
            ),
        }
        self.text_parser = TextParserBlock(**text_parser_kwargs)

        # 4. ColumnValueFilterBlock
        filter_kwargs = {
            **self.filter_params,  # Forward all filter parameters dynamically
            "block_name": f"{self.block_name}_filter",
            "input_cols": ["verification_rating"],
            "output_cols": [],  # Filter blocks don't create new columns
            "filter_value": self.filter_params.get("filter_value", 1.0),
            "operation": self.filter_params.get("operation", "ge"),
            "convert_dtype": self.filter_params.get("convert_dtype", "float"),
        }
        self.filter_block = ColumnValueFilterBlock(**filter_kwargs)

    def _reinitialize_client_manager(self) -> None:
        """Reinitialize the internal LLM chat block's client manager.

        This should be called after model configuration changes to ensure
        the internal LLM chat block uses the updated model configuration.
        """
        if self.llm_chat and hasattr(self.llm_chat, "_reinitialize_client_manager"):
            # Update the internal LLM chat block's model config from stored params
            for key in ["model", "api_base", "api_key"]:
                if key in self.llm_params:
                    setattr(self.llm_chat, key, self.llm_params[key])
            # Reinitialize its client manager
            self.llm_chat._reinitialize_client_manager()

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate question verification for all samples.

        This method chains the four internal blocks in sequence:
        1. Build question verification prompts
        2. Generate LLM responses
        3. Parse explanation and rating
        4. Filter based on rating

        Parameters
        ----------
        samples : Dataset
            Input dataset containing 'question' column.
        **kwargs : Any
            Additional keyword arguments passed to internal blocks.

        Returns
        -------
        Dataset
            Dataset with question verification results and filtering applied.

        Raises
        ------
        BlockValidationError
            If model is not configured before calling generate().
        """
        # Validate that model is configured
        model = self.llm_params.get("model")
        if not model:
            # Local
            from ...utils.error_handling import BlockValidationError

            raise BlockValidationError(
                f"Model not configured for block '{self.block_name}'. "
                f"Call flow.set_model_config() before generating."
            )
        logger.info(
            f"Starting question verification for {len(samples)} samples",
            extra={
                "block_name": self.block_name,
                "model": model,
                "batch_size": len(samples),
            },
        )

        current_dataset = samples

        try:
            # Step 1: Build prompts
            logger.debug("Step 1: Building question verification prompts")
            current_dataset = self.prompt_builder(current_dataset, **kwargs)

            # Step 2: Generate LLM responses
            logger.debug("Step 2: Generating LLM responses")
            current_dataset = self.llm_chat(current_dataset, **kwargs)

            # Step 3: Parse responses
            logger.debug("Step 3: Parsing question verification responses")
            current_dataset = self.text_parser(current_dataset, **kwargs)

            # Step 4: Filter based on rating
            logger.debug("Step 4: Filtering based on verification rating")
            original_count = len(current_dataset)
            current_dataset = self.filter_block(current_dataset, **kwargs)
            filtered_count = len(current_dataset)

            logger.info(
                f"Question verification completed: {original_count} → {filtered_count} samples "
                f"(filtered {original_count - filtered_count} samples)",
                extra={
                    "block_name": self.block_name,
                    "original_count": original_count,
                    "filtered_count": filtered_count,
                    "filter_rate": (original_count - filtered_count) / original_count
                    if original_count > 0
                    else 0,
                },
            )

            return current_dataset

        except Exception as e:
            logger.error(
                f"Error during question verification: {e}",
                extra={
                    "block_name": self.block_name,
                    "model": model,
                    "error": str(e),
                },
            )
            raise

    def get_internal_blocks_info(self) -> dict[str, Any]:
        """Get information about the internal blocks.

        Returns
        -------
        Dict[str, Any]
            Information about each internal block.
        """
        return {
            "prompt_builder": self.prompt_builder.get_info()
            if self.prompt_builder
            else None,
            "llm_chat": self.llm_chat.get_info() if self.llm_chat else None,
            "text_parser": self.text_parser.get_info() if self.text_parser else None,
            "filter": self.filter_block.get_info() if self.filter_block else None,
        }

    def __repr__(self) -> str:
        """String representation of the block."""
        model = self.llm_params.get("model", "None")
        filter_value = self.filter_params.get("filter_value", 1.0)
        return (
            f"VerifyQuestionBlock(name='{self.block_name}', "
            f"model='{model}', filter_value='{filter_value}')"
        )
