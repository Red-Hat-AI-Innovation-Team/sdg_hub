# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper for faithfulness evaluation using 4 composed blocks.

This module provides a simple, lightweight wrapper that composes:
- PromptBuilderBlock: builds evaluation prompts
- LLMChatBlock: generates LLM responses
- TextParserBlock: parses structured output
- ColumnValueFilterBlock: filters based on judgment

The wrapper exposes minimal LLM interface for flow detection while
delegating all functionality to the internal blocks.
"""

# Standard
from typing import Any, Optional

# Third Party
from datasets import Dataset
from pydantic import ConfigDict, Field, field_validator

# Local
from ...utils.error_handling import BlockValidationError
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..filtering.column_value_filter import ColumnValueFilterBlock
from ..llm.llm_chat_block import LLMChatBlock
from ..llm.prompt_builder_block import PromptBuilderBlock
from ..llm.text_parser_block import TextParserBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "EvaluateFaithfulnessBlock",
    "evaluation",
    "Thin wrapper composing 4 blocks for faithfulness evaluation",
)
class EvaluateFaithfulnessBlock(BaseBlock):
    """Thin wrapper for faithfulness evaluation using composed blocks.

    Composes PromptBuilderBlock + LLMChatBlock + TextParserBlock + ColumnValueFilterBlock
    into a single evaluation pipeline with smart parameter routing.

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : List[str]
        Input columns: ["document", "response"]
    output_cols : List[str]
        Output columns: ["faithfulness_explanation", "faithfulness_judgment"]
    model : Optional[str]
        LLM model identifier.
    api_base : Optional[str]
        API base URL.
    api_key : Optional[str]
        API key.
    prompt_config_path : str
        Path to YAML prompt template file (required).
    **kwargs : Any
        All other parameters are automatically routed to appropriate internal blocks
        based on each block's accepted parameters. This includes all LLM parameters
        (temperature, max_tokens, extra_body, extra_headers, etc.), text parser
        parameters, and filter parameters.
    """

    model_config = ConfigDict(
        extra="allow"
    )  # Allow extra fields for dynamic forwarding

    # --- Core configuration ---
    prompt_config_path: str = Field(
        ...,
        description="Path to YAML file containing the faithfulness evaluation prompt template",
    )

    # --- LLM interface (for flow detection) ---
    model: Optional[str] = Field(None, description="LLM model identifier")
    api_base: Optional[str] = Field(None, description="API base URL")
    api_key: Optional[str] = Field(None, description="API key")

    # --- Filter configuration ---
    filter_value: str = Field(
        "YES", description="Value to filter on for faithfulness judgment"
    )
    operation: str = Field("eq", description="Filter operation")
    convert_dtype: Optional[str] = Field(
        None, description="Data type conversion for filter column"
    )

    # --- Parser configuration ---
    start_tags: list[str] = Field(
        ["[Start of Explanation]", "[Start of Answer]"],
        description="Start tags for parsing explanation and judgment",
    )
    end_tags: list[str] = Field(
        ["[End of Explanation]", "[End of Answer]"],
        description="End tags for parsing explanation and judgment",
    )
    parsing_pattern: Optional[str] = Field(
        None,
        description="Regex pattern for custom parsing. If provided, takes precedence over tag-based parsing",
    )

    # --- Internal blocks (composition) ---
    prompt_builder: PromptBuilderBlock = Field(None, exclude=True)  # type: ignore
    llm_chat: LLMChatBlock = Field(None, exclude=True)  # type: ignore
    text_parser: TextParserBlock = Field(None, exclude=True)  # type: ignore
    filter_block: ColumnValueFilterBlock = Field(None, exclude=True)  # type: ignore

    @field_validator("input_cols")
    @classmethod
    def validate_input_cols(cls, v):
        """Validate input columns."""
        if v != ["document", "response"]:
            raise ValueError(
                f"EvaluateFaithfulnessBlock expects input_cols ['document', 'response'], got {v}"
            )
        return v

    @field_validator("output_cols")
    @classmethod
    def validate_output_cols(cls, v):
        """Validate output columns."""
        expected = ["faithfulness_explanation", "faithfulness_judgment"]
        if v != expected:
            raise ValueError(
                f"EvaluateFaithfulnessBlock expects output_cols {expected}, got {v}"
            )
        return v

    def __init__(self, **kwargs):
        """Initialize with smart parameter routing."""
        super().__init__(**kwargs)
        self._create_internal_blocks(**kwargs)

        # Log initialization if model is configured
        if self.model:
            logger.info(
                f"Initialized EvaluateFaithfulnessBlock '{self.block_name}' with model '{self.model}'"
            )

    def _extract_params(self, kwargs: dict, block_class) -> dict:
        """Extract parameters for specific block class based on its model_fields."""
        # Exclude parameters that are handled by this wrapper's structure
        wrapper_params = {
            "block_name",
            "input_cols",
            "output_cols",
        }

        # Extract parameters that the target block accepts
        params = {
            k: v
            for k, v in kwargs.items()
            if k in block_class.model_fields and k not in wrapper_params
        }

        # Also include declared fields from this composite block that the target block accepts
        for field_name in self.__class__.model_fields:
            if (
                field_name in block_class.model_fields
                and field_name not in wrapper_params
            ):
                field_value = getattr(self, field_name)
                if field_value is not None:  # Only forward non-None values
                    params[field_name] = field_value

        return params

    def _create_internal_blocks(self, **kwargs):
        """Create internal blocks with parameter routing."""
        # Route parameters to appropriate blocks
        prompt_params = self._extract_params(kwargs, PromptBuilderBlock)
        llm_params = self._extract_params(kwargs, LLMChatBlock)
        parser_params = self._extract_params(kwargs, TextParserBlock)
        filter_params = self._extract_params(kwargs, ColumnValueFilterBlock)

        self.prompt_builder = PromptBuilderBlock(
            block_name=f"{self.block_name}_prompt_builder",
            input_cols=["document", "response"],
            output_cols=["eval_faithfulness_prompt"],
            **prompt_params,
        )

        # Create LLM chat block with dynamic LLM parameter forwarding
        llm_config = {
            "block_name": f"{self.block_name}_llm_chat",
            "input_cols": ["eval_faithfulness_prompt"],
            "output_cols": ["raw_eval_faithfulness"],
            **llm_params,
        }

        # Only add LLM parameters if they are provided
        if self.model is not None:
            llm_config["model"] = self.model
        if self.api_base is not None:
            llm_config["api_base"] = self.api_base
        if self.api_key is not None:
            llm_config["api_key"] = self.api_key

        self.llm_chat = LLMChatBlock(**llm_config)

        # Create text parser
        self.text_parser = TextParserBlock(
            block_name=f"{self.block_name}_text_parser",
            input_cols=["raw_eval_faithfulness"],
            output_cols=["faithfulness_explanation", "faithfulness_judgment"],
            **parser_params,
        )

        self.filter_block = ColumnValueFilterBlock(
            block_name=f"{self.block_name}_filter",
            input_cols=["faithfulness_judgment"],
            output_cols=[],  # Filter doesn't create new columns
            **filter_params,
        )

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Execute the 4-block faithfulness evaluation pipeline.

        Parameters
        ----------
        samples : Dataset
            Input dataset with 'document' and 'response' columns.
        **kwargs : Any
            Additional arguments passed to internal blocks.

        Returns
        -------
        Dataset
            Filtered dataset with faithfulness evaluation results.
        """
        # Validate model is configured
        if not self.model:
            raise BlockValidationError(
                f"Model not configured for block '{self.block_name}'. "
                f"Call flow.set_model_config() before generating."
            )

        logger.info(
            f"Starting faithfulness evaluation for {len(samples)} samples",
            extra={"block_name": self.block_name, "model": self.model},
        )

        try:
            # Execute 4-block pipeline with validation delegation
            result = self.prompt_builder(samples, **kwargs)
            result = self.llm_chat(result, **kwargs)
            result = self.text_parser(result, **kwargs)
            result = self.filter_block(result, **kwargs)

            logger.info(
                f"Faithfulness evaluation completed: {len(samples)} → {len(result)} samples",
                extra={"block_name": self.block_name},
            )

            return result

        except Exception as e:
            logger.error(
                f"Error during faithfulness evaluation: {e}",
                extra={"block_name": self.block_name, "error": str(e)},
            )
            raise

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to appropriate internal block."""
        # Check each internal block to see which one has this parameter
        for block_attr, block_class in [
            ("prompt_builder", PromptBuilderBlock),
            ("llm_chat", LLMChatBlock),
            ("text_parser", TextParserBlock),
            ("filter_block", ColumnValueFilterBlock),
        ]:
            if hasattr(self, block_attr) and name in block_class.model_fields:
                internal_block = getattr(self, block_attr)
                if internal_block is not None:
                    return getattr(internal_block, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle dynamic parameter updates from flow.set_model_config()."""
        super().__setattr__(name, value)

        # Forward to appropriate internal blocks
        for block_attr, block_class in [
            ("prompt_builder", PromptBuilderBlock),
            ("llm_chat", LLMChatBlock),
            ("text_parser", TextParserBlock),
            ("filter_block", ColumnValueFilterBlock),
        ]:
            if hasattr(self, block_attr) and name in block_class.model_fields:
                setattr(getattr(self, block_attr), name, value)

    def _reinitialize_client_manager(self) -> None:
        """Reinitialize internal LLM block's client manager."""
        if hasattr(self.llm_chat, "_reinitialize_client_manager"):
            self.llm_chat._reinitialize_client_manager()

    def get_internal_blocks_info(self) -> dict[str, Any]:
        """Get information about internal blocks."""
        return {
            "prompt_builder": self.prompt_builder.get_info(),
            "llm_chat": self.llm_chat.get_info(),
            "text_parser": self.text_parser.get_info(),
            "filter": self.filter_block.get_info(),
        }

    def __repr__(self) -> str:
        """String representation of the block."""
        filter_value = (
            getattr(self.filter_block, "filter_value", "YES")
            if hasattr(self, "filter_block")
            else "YES"
        )
        return (
            f"EvaluateFaithfulnessBlock(name='{self.block_name}', "
            f"model='{self.model}', filter_value='{filter_value}')"
        )
