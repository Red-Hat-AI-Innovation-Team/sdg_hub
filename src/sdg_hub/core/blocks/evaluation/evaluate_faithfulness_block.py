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
from pydantic import Field, field_validator

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
    prompt_config_path : str
        Path to YAML prompt template file.
    model : Optional[str]
        LLM model identifier (for flow detection).
    api_base : Optional[str]
        API base URL (for flow detection).
    api_key : Optional[str]
        API key (for flow detection).
    filter_value : str, optional
        Value to filter on (default: "YES").
    operation : str, optional
        Filter operation (default: "eq").
    **kwargs : Any
        All other parameters are automatically routed to appropriate internal blocks.
    """

    # --- Block-specific configuration ---
    prompt_config_path: str = Field(..., description="Path to YAML prompt template")
    filter_value: str = Field("YES", description="Filter value for judgment")
    operation: str = Field("eq", description="Filter operation")

    # --- Minimal LLM interface (for flow detection) ---
    model: Optional[str] = Field(None, description="LLM model identifier")
    api_base: Optional[str] = Field(None, description="API base URL")
    api_key: Optional[str] = Field(None, description="API key")
    extra_headers: Optional[dict] = Field(
        None, description="Extra headers for LLM requests"
    )

    # --- Internal blocks (composition) ---
    prompt_builder: PromptBuilderBlock = Field(None, exclude=True)
    llm_chat: LLMChatBlock = Field(None, exclude=True)
    text_parser: TextParserBlock = Field(None, exclude=True)
    filter_block: ColumnValueFilterBlock = Field(None, exclude=True)

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
        """Extract parameters belonging to specific block class."""
        # Exclude parameters we explicitly set for internal blocks
        exclude_params = {
            "block_name",
            "input_cols",
            "output_cols",
            "prompt_config_path",
            "filter_value",
            "operation",
        }
        return {
            k: v
            for k, v in kwargs.items()
            if k in block_class.model_fields and k not in exclude_params
        }

    def _create_internal_blocks(self, **kwargs):
        """Create internal blocks with parameter routing."""
        # Route parameters to appropriate blocks
        prompt_params = self._extract_params(kwargs, PromptBuilderBlock)
        llm_params = self._extract_params(kwargs, LLMChatBlock)
        parser_params = self._extract_params(kwargs, TextParserBlock)
        filter_params = self._extract_params(kwargs, ColumnValueFilterBlock)

        # Create prompt builder
        self.prompt_builder = PromptBuilderBlock(
            block_name=f"{self.block_name}_prompt_builder",
            input_cols=["document", "response"],
            output_cols=["eval_faithfulness_prompt"],
            prompt_config_path=self.prompt_config_path,
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
        if self.extra_headers is not None:
            llm_config["extra_headers"] = self.extra_headers

        self.llm_chat = LLMChatBlock(**llm_config)

        # Create text parser
        self.text_parser = TextParserBlock(
            block_name=f"{self.block_name}_text_parser",
            input_cols=["raw_eval_faithfulness"],
            output_cols=["faithfulness_explanation", "faithfulness_judgment"],
            **parser_params,
        )

        # Create filter block
        self.filter_block = ColumnValueFilterBlock(
            block_name=f"{self.block_name}_filter",
            input_cols=["faithfulness_judgment"],
            output_cols=[],  # Filter doesn't create new columns
            filter_value=self.filter_value,
            operation=self.operation,
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
            from ...utils.error_handling import BlockValidationError

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

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle dynamic parameter updates from flow.set_model_config()."""
        super().__setattr__(name, value)

        # Propagate LLM parameters to internal LLM block
        if name in LLMChatBlock.model_fields and hasattr(self, "llm_chat"):
            setattr(self.llm_chat, name, value)

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
        return (
            f"EvaluateFaithfulnessBlock(name='{self.block_name}', "
            f"model='{self.model}', filter_value='{self.filter_value}')"
        )
