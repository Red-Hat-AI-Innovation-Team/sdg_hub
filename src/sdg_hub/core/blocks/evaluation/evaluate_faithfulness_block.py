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
from ..llm.llm_parser_block import LLMParserBlock
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
    llm_parser: LLMParserBlock = Field(None, exclude=True)  # type: ignore
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

    def _get_wrapper_params(self) -> set[str]:
        """Get parameters that are handled by this wrapper's structure."""
        return {"block_name", "input_cols", "output_cols"}

    def _add_composite_fields(
        self, params: dict, block_class, wrapper_params: set[str]
    ) -> dict:
        """Add declared fields from this composite block to parameters."""
        for field_name in self.__class__.model_fields:
            if field_name in wrapper_params:
                continue

            # For LLMChatBlock, add all non-wrapper fields
            # For other blocks, only add fields they accept
            if block_class == LLMChatBlock or field_name in block_class.model_fields:
                field_value = getattr(self, field_name)
                if field_value is not None:  # Only forward non-None values
                    params[field_name] = field_value
        return params

    def _extract_params(
        self, kwargs: dict, block_class, remove_params: list[str] = []
    ) -> dict:
        """Extract parameters for specific block class based on its model_fields."""
        wrapper_params = self._get_wrapper_params()

        # For LLMChatBlock (with extra="allow"), forward all parameters except wrapper params
        if block_class == LLMChatBlock:
            params = {k: v for k, v in kwargs.items() if k not in wrapper_params}
            params = self._add_composite_fields(params, block_class, wrapper_params)
            params = {k: v for k, v in params.items() if k not in remove_params}
        else:
            # For other blocks, only forward parameters they accept
            params = {
                k: v
                for k, v in kwargs.items()
                if k in block_class.model_fields and k not in wrapper_params
            }
            params = self._add_composite_fields(params, block_class, wrapper_params)

        return params

    def _create_internal_blocks(self, **kwargs):
        """Create internal blocks with parameter routing."""
        # Route parameters to appropriate blocks
        prompt_params = self._extract_params(kwargs, PromptBuilderBlock)
        llm_params = self._extract_params(kwargs, LLMChatBlock)
        llm_parser_params = self._extract_params(kwargs, LLMParserBlock)
        parser_params = self._extract_params(kwargs, TextParserBlock)
        filter_params = self._extract_params(kwargs, ColumnValueFilterBlock)
        remove_params = (
            set(prompt_params.keys())
            | set(parser_params.keys())
            | set(filter_params.keys())
        )
        llm_params = self._extract_params(kwargs, LLMChatBlock, remove_params)

        self.prompt_builder = PromptBuilderBlock(
            block_name=f"{self.block_name}_prompt_builder",
            input_cols=["document", "response"],
            output_cols=["eval_faithfulness_prompt"],
            **prompt_params,
        )

        # Create LLM chat block with parameter forwarding
        self.llm_chat = LLMChatBlock(
            block_name=f"{self.block_name}_llm_chat",
            input_cols=["eval_faithfulness_prompt"],
            output_cols=["raw_eval_faithfulness"],
            **llm_params,
        )

        # Create LLM parser block
        self.llm_parser = LLMParserBlock(
            block_name=f"{self.block_name}_llm_parser",
            input_cols=["raw_eval_faithfulness"],
            **llm_parser_params,
        )

        print(
            f"{self.llm_parser.field_prefix if self.llm_parser.field_prefix!='' else self.llm_parser.block_name}_content"
        )

        # Create text parser
        self.text_parser = TextParserBlock(
            block_name=f"{self.block_name}_text_parser",
            input_cols=[
                f"{self.llm_parser.field_prefix if self.llm_parser.field_prefix!='' else self.llm_parser.block_name}_content"
            ],
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
            # Filter override kwargs - only pass runtime kwargs, not composite fields
            prompt_params = {
                k: v for k, v in kwargs.items() if k in self.prompt_builder.model_fields
            }
            parser_params = {
                k: v for k, v in kwargs.items() if k in self.text_parser.model_fields
            }
            filter_params = {
                k: v for k, v in kwargs.items() if k in self.filter_block.model_fields
            }
            llm_parser_params = {
                k: v for k, v in kwargs.items() if k in self.llm_parser.model_fields
            }
            non_llm_params = (
                set(prompt_params.keys())
                | set(parser_params.keys())
                | set(filter_params.keys())
                | set(llm_parser_params.keys())
            )
            llm_params = {k: v for k, v in kwargs.items() if k not in non_llm_params}

            # Execute 4-block pipeline with validation delegation
            result = self.prompt_builder(samples, **prompt_params)
            result = self.llm_chat(result, **llm_params)
            result = self.llm_parser(result, **llm_parser_params)
            result = self.text_parser(result, **parser_params)
            result = self.filter_block(result, **filter_params)

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
        # Try LLMChatBlock first since it accepts any parameters via extra="allow"
        if hasattr(self, "llm_chat") and self.llm_chat is not None:
            # Always try LLMChatBlock first - it will return None for unset attributes
            # due to extra="allow", which makes hasattr() work correctly
            return getattr(self.llm_chat, name, None)

        # Check other internal blocks for their specific model_fields
        for block_attr, block_class in [
            ("prompt_builder", PromptBuilderBlock),
            ("llm_chat", LLMChatBlock),
            ("llm_parser", LLMParserBlock),
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

        # Skip forwarding for internal block attributes and wrapper-specific params
        if name in {
            "prompt_builder",
            "llm_chat",
            "text_parser",
            "filter_block",
            "block_name",
            "input_cols",
            "output_cols",
        }:
            return

        # Forward to LLMChatBlock first since it accepts any parameters via extra="allow"
        if hasattr(self, "llm_chat") and self.llm_chat is not None:
            setattr(self.llm_chat, name, value)

        # Forward to other internal blocks for their specific model_fields
        for block_attr, block_class in [
            ("prompt_builder", PromptBuilderBlock),
            ("llm_chat", LLMChatBlock),
            ("llm_parser", LLMParserBlock),
            ("text_parser", TextParserBlock),
            ("filter_block", ColumnValueFilterBlock),
        ]:
            if hasattr(self, block_attr) and name in block_class.model_fields:
                internal_block = getattr(self, block_attr)
                if internal_block is not None:
                    setattr(internal_block, name, value)

    def get_internal_blocks_info(self) -> dict[str, Any]:
        """Get information about internal blocks."""
        return {
            "prompt_builder": self.prompt_builder.get_info(),
            "llm_chat": self.llm_chat.get_info(),
            "llm_parser": self.llm_parser.get_info(),
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
