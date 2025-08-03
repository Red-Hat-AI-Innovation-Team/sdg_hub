# SPDX-License-Identifier: Apache-2.0
"""Composite block for relevancy evaluation of question-answer pairs.

This module provides the EvaluateRelevancyBlock that encapsulates the complete
relevancy evaluation workflow, combining prompt building, LLM chat, text parsing,
and filtering into a single block for simplified configuration.
"""

# Standard
from typing import Any, Dict, List, Optional, Union

# Third Party
from datasets import Dataset
from pydantic import ConfigDict, Field, field_validator

# Local
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry
from ..llm.prompt_builder_block import PromptBuilderBlock
from ..llm.llm_chat_block import LLMChatBlock
from ..llm.text_parser_block import TextParserBlock
from ..filtering.column_value_filter import ColumnValueFilterBlock

logger = setup_logger(__name__)


@BlockRegistry.register(
    "EvaluateRelevancyBlock",
    "evaluation",
    "Composite block for relevancy evaluation of question-answer pairs",
)
class EvaluateRelevancyBlock(BaseBlock):
    """Composite block for relevancy evaluation workflow.

    This block combines four separate blocks into a single cohesive evaluation block:
    1. PromptBuilderBlock - builds evaluation prompt from question and response
    2. LLMChatBlock - generates relevancy evaluation using LLM
    3. TextParserBlock - parses feedback and score from raw output
    4. ColumnValueFilterBlock - filters based on relevancy score

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : List[str]
        Input columns: ["question", "response"]
    output_cols : List[str]
        Output columns: ["relevancy_explanation", "relevancy_score"]
    prompt_config_path : str
        Path to YAML file containing the relevancy evaluation prompt template.
    model : str
        Model identifier in LiteLLM format (e.g., "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct")
    api_base : Optional[str]
        Base URL for the API. Required for local models.
    api_key : Optional[str]
        API key for the provider. Falls back to environment variables.
    filter_value : Union[str, int, float], optional
        Value to filter on for relevancy score (default: 2.0)
    operation : str, optional
        Filter operation (default: "eq")
    convert_dtype : Optional[str], optional
        Data type conversion for filter column (default: "float")
    async_mode : bool, optional
        Whether to use async processing (default: True)
    format_as_messages : bool, optional
        Whether to format prompt as messages (default: True)
    start_tags : List[str], optional
        Start tags for parsing (default: ["[Start of Feedback]", "[Start of Score]"])
    end_tags : List[str], optional
        End tags for parsing (default: ["[End of Feedback]", "[End of Score]"])
    parsing_pattern : Optional[str], optional
        Regex pattern for custom parsing. If provided, takes precedence over tag-based parsing.
    parser_cleanup_tags : Optional[List[str]], optional
        List of tags to clean from parsed output.

    ### LLM Generation Parameters ###
    temperature : Optional[float], optional
        Sampling temperature (0.0 to 2.0).
    max_tokens : Optional[int], optional
        Maximum tokens to generate.
    top_p : Optional[float], optional
        Nucleus sampling parameter (0.0 to 1.0).
    frequency_penalty : Optional[float], optional
        Frequency penalty (-2.0 to 2.0).
    presence_penalty : Optional[float], optional
        Presence penalty (-2.0 to 2.0).
    stop : Optional[Union[str, List[str]]], optional
        Stop sequences.
    seed : Optional[int], optional
        Random seed for reproducible outputs.
    response_format : Optional[Dict[str, Any]], optional
        Response format specification (e.g., JSON mode).
    stream : Optional[bool], optional
        Whether to stream responses.
    n : Optional[int], optional
        Number of completions to generate. When n > 1, the output column will contain
        a list of responses for each input sample.
    logprobs : Optional[bool], optional
        Whether to return log probabilities.
    top_logprobs : Optional[int], optional
        Number of top log probabilities to return.
    user : Optional[str], optional
        End-user identifier.
    extra_headers : Optional[Dict[str, str]], optional
        Additional headers to send with requests.
    extra_body : Optional[Dict[str, Any]], optional
        Additional parameters for the request body.
    timeout : float, optional
        Request timeout in seconds (default: 120.0).
    max_retries : int, optional
        Maximum number of retry attempts (default: 6).
    **kwargs : Any
        Additional provider-specific parameters.
    """

    model_config = ConfigDict(extra="forbid")

    # Core configuration
    prompt_config_path: str = Field(
        ...,
        description="Path to YAML file containing the relevancy evaluation prompt template",
    )
    model: str = Field(..., description="Model identifier in LiteLLM format")
    api_base: Optional[str] = Field(None, description="Base URL for the API")
    api_key: Optional[str] = Field(
        None,
        description="API key for the provider. Falls back to environment variables.",
    )

    # Filter configuration
    filter_value: Union[str, int, float] = Field(
        2.0, description="Value to filter on for relevancy score"
    )
    operation: str = Field("eq", description="Filter operation")
    convert_dtype: Optional[str] = Field(
        "float", description="Data type conversion for filter column"
    )

    # Processing configuration
    async_mode: bool = Field(True, description="Whether to use async processing")
    format_as_messages: bool = Field(
        True, description="Whether to format prompt as messages"
    )

    # Parser configuration
    start_tags: List[str] = Field(
        ["[Start of Feedback]", "[Start of Score]"],
        description="Start tags for parsing feedback and score",
    )
    end_tags: List[str] = Field(
        ["[End of Feedback]", "[End of Score]"],
        description="End tags for parsing feedback and score",
    )
    parsing_pattern: Optional[str] = Field(
        None, description="Regex pattern for custom parsing. If provided, takes precedence over tag-based parsing"
    )
    parser_cleanup_tags: Optional[List[str]] = Field(
        None, description="List of tags to clean from parsed output"
    )

    # LLM generation parameters
    temperature: Optional[float] = Field(
        None, description="Sampling temperature (0.0 to 2.0)"
    )
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(
        None, description="Nucleus sampling parameter (0.0 to 1.0)"
    )
    frequency_penalty: Optional[float] = Field(
        None, description="Frequency penalty (-2.0 to 2.0)"
    )
    presence_penalty: Optional[float] = Field(
        None, description="Presence penalty (-2.0 to 2.0)"
    )
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    seed: Optional[int] = Field(
        None, description="Random seed for reproducible outputs"
    )
    response_format: Optional[Dict[str, Any]] = Field(
        None, description="Response format specification (e.g., JSON mode)"
    )
    stream: Optional[bool] = Field(None, description="Whether to stream responses")
    n: Optional[int] = Field(
        None,
        description="Number of completions to generate. When n > 1, the output column will contain a list of responses for each input sample",
    )
    logprobs: Optional[bool] = Field(
        None, description="Whether to return log probabilities"
    )
    top_logprobs: Optional[int] = Field(
        None, description="Number of top log probabilities to return"
    )
    user: Optional[str] = Field(None, description="End-user identifier")
    extra_headers: Optional[Dict[str, str]] = Field(
        None, description="Additional headers to send with requests"
    )
    extra_body: Optional[Dict[str, Any]] = Field(
        None, description="Additional parameters for the request body"
    )
    timeout: float = Field(120.0, description="Request timeout in seconds")
    max_retries: int = Field(6, description="Maximum number of retry attempts")

    # Additional provider-specific parameters
    llm_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific parameters"
    )

    # Internal blocks - excluded from serialization
    prompt_builder: Optional[PromptBuilderBlock] = Field(None, exclude=True)
    llm_chat: Optional[LLMChatBlock] = Field(None, exclude=True)
    text_parser: Optional[TextParserBlock] = Field(None, exclude=True)
    filter_block: Optional[ColumnValueFilterBlock] = Field(None, exclude=True)

    @field_validator("input_cols")
    @classmethod
    def validate_input_cols(cls, v):
        """Validate that input columns are exactly ["question", "response"]."""
        expected = ["question", "response"]
        if v != expected:
            raise ValueError(
                f"EvaluateRelevancyBlock expects input_cols={expected}, got {v}"
            )
        return v

    @field_validator("output_cols")
    @classmethod
    def validate_output_cols(cls, v):
        """Validate that output columns are exactly ["relevancy_explanation", "relevancy_score"]."""
        expected = [
            "relevancy_explanation",
            "relevancy_score",
        ]
        if v != expected:
            raise ValueError(
                f"EvaluateRelevancyBlock expects output_cols={expected}, got {v}"
            )
        return v

    def model_post_init(self, __context: Any) -> None:
        """Initialize the internal blocks after Pydantic validation."""
        super().model_post_init(__context)

        # Create internal blocks
        self._create_internal_blocks()

        logger.info(
            f"Initialized EvaluateRelevancyBlock '{self.block_name}' with model '{self.model}'",
            extra={
                "block_name": self.block_name,
                "model": self.model,
                "async_mode": self.async_mode,
                "filter_value": self.filter_value,
            },
        )

    def _create_internal_blocks(self) -> None:
        """Create and configure the internal blocks."""
        # 1. PromptBuilderBlock
        self.prompt_builder = PromptBuilderBlock(
            block_name=f"{self.block_name}_prompt_builder",
            input_cols=["question", "response"],
            output_cols=["eval_relevancy_prompt"],
            prompt_config_path=self.prompt_config_path,
            format_as_messages=self.format_as_messages,
        )

        # 2. LLMChatBlock
        llm_kwargs = {
            "block_name": f"{self.block_name}_llm_chat",
            "input_cols": ["eval_relevancy_prompt"],
            "output_cols": ["raw_eval_relevancy"],
            "model": self.model,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "async_mode": self.async_mode,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        # Add generation parameters if specified
        if self.temperature is not None:
            llm_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            llm_kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            llm_kwargs["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            llm_kwargs["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            llm_kwargs["presence_penalty"] = self.presence_penalty
        if self.stop is not None:
            llm_kwargs["stop"] = self.stop
        if self.seed is not None:
            llm_kwargs["seed"] = self.seed
        if self.response_format is not None:
            llm_kwargs["response_format"] = self.response_format
        if self.stream is not None:
            llm_kwargs["stream"] = self.stream
        if self.n is not None:
            llm_kwargs["n"] = self.n
        if self.logprobs is not None:
            llm_kwargs["logprobs"] = self.logprobs
        if self.top_logprobs is not None:
            llm_kwargs["top_logprobs"] = self.top_logprobs
        if self.user is not None:
            llm_kwargs["user"] = self.user
        if self.extra_headers is not None:
            llm_kwargs["extra_headers"] = self.extra_headers
        if self.extra_body is not None:
            llm_kwargs["extra_body"] = self.extra_body
        
        # Add any additional kwargs
        llm_kwargs.update(self.llm_kwargs)

        self.llm_chat = LLMChatBlock(**llm_kwargs)

        # 3. TextParserBlock
        text_parser_kwargs = {
            "block_name": f"{self.block_name}_text_parser",
            "input_cols": ["raw_eval_relevancy"],
            "output_cols": ["relevancy_explanation", "relevancy_score"],
            "start_tags": self.start_tags,
            "end_tags": self.end_tags,
        }
        
        # Add optional TextParserBlock parameters if specified
        if self.parsing_pattern is not None:
            text_parser_kwargs["parsing_pattern"] = self.parsing_pattern
        if self.parser_cleanup_tags is not None:
            text_parser_kwargs["parser_cleanup_tags"] = self.parser_cleanup_tags
            
        self.text_parser = TextParserBlock(**text_parser_kwargs)

        # 4. ColumnValueFilterBlock
        filter_kwargs = {
            "block_name": f"{self.block_name}_filter",
            "input_cols": ["relevancy_score"],
            "output_cols": [],  # Filter blocks don't create new columns
            "filter_value": self.filter_value,
            "operation": self.operation,
        }

        if self.convert_dtype is not None:
            filter_kwargs["convert_dtype"] = self.convert_dtype

        self.filter_block = ColumnValueFilterBlock(**filter_kwargs)

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate relevancy evaluation for all samples.

        This method chains the four internal blocks in sequence:
        1. Build relevancy evaluation prompts
        2. Generate LLM responses
        3. Parse explanation and score
        4. Filter based on score

        Parameters
        ----------
        samples : Dataset
            Input dataset containing 'question' and 'response' columns.
        **kwargs : Any
            Additional keyword arguments passed to internal blocks.

        Returns
        -------
        Dataset
            Dataset with relevancy evaluation results and filtering applied.
        """
        logger.info(
            f"Starting relevancy evaluation for {len(samples)} samples",
            extra={
                "block_name": self.block_name,
                "model": self.model,
                "batch_size": len(samples),
            },
        )

        current_dataset = samples

        try:
            # Step 1: Build prompts
            logger.debug(f"Step 1: Building relevancy evaluation prompts")
            current_dataset = self.prompt_builder.generate(current_dataset, **kwargs)

            # Step 2: Generate LLM responses
            logger.debug(f"Step 2: Generating LLM responses")
            current_dataset = self.llm_chat.generate(current_dataset, **kwargs)

            # Step 3: Parse responses
            logger.debug(f"Step 3: Parsing relevancy evaluation responses")
            current_dataset = self.text_parser.generate(current_dataset, **kwargs)

            # Step 4: Filter based on score
            logger.debug(f"Step 4: Filtering based on relevancy score")
            original_count = len(current_dataset)
            current_dataset = self.filter_block.generate(current_dataset, **kwargs)
            filtered_count = len(current_dataset)

            logger.info(
                f"Relevancy evaluation completed: {original_count} → {filtered_count} samples "
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
                f"Error during relevancy evaluation: {e}",
                extra={
                    "block_name": self.block_name,
                    "model": self.model,
                    "error": str(e),
                },
            )
            raise

    def _validate_custom(self, dataset: Dataset) -> None:
        """Custom validation for relevancy evaluation.

        This method validates the entire chain of internal blocks by simulating
        the data flow through each block to ensure they can all process the data correctly.
        """
        # Validate that required columns exist
        required_columns = ["question", "response"]
        missing_columns = [
            col for col in required_columns if col not in dataset.column_names
        ]
        if missing_columns:
            raise ValueError(
                f"EvaluateRelevancyBlock requires columns {required_columns}, "
                f"missing: {missing_columns}"
            )

        # Validate the entire chain of internal blocks
        if not all(
            [self.prompt_builder, self.llm_chat, self.text_parser, self.filter_block]
        ):
            raise ValueError(
                "All internal blocks must be initialized before validation"
            )

        # Simulate data flow through the chain to validate each block
        current_dataset = dataset

        try:
            # 1. Validate PromptBuilderBlock
            logger.debug("Validating prompt builder block")
            self.prompt_builder._validate_custom(current_dataset)

            # Simulate prompt builder output for next validation
            # Add the expected output column temporarily for validation
            if "eval_relevancy_prompt" not in current_dataset.column_names:
                # Create a temporary dataset with the expected column for validation
                temp_data = []
                for sample in current_dataset:
                    temp_sample = dict(sample)
                    temp_sample["eval_relevancy_prompt"] = [
                        {"role": "user", "content": "test"}
                    ]
                    temp_data.append(temp_sample)
                current_dataset = Dataset.from_list(temp_data)

            # 2. Validate LLMChatBlock
            logger.debug("Validating LLM chat block")
            self.llm_chat._validate_custom(current_dataset)

            # Simulate LLM chat output for next validation
            if "raw_eval_relevancy" not in current_dataset.column_names:
                temp_data = []
                for sample in current_dataset:
                    temp_sample = dict(sample)
                    temp_sample["raw_eval_relevancy"] = (
                        "[Start of Feedback]Test feedback[End of Feedback]\n[Start of Score]2.0[End of Score]"
                    )
                    temp_data.append(temp_sample)
                current_dataset = Dataset.from_list(temp_data)

            # 3. Validate TextParserBlock
            logger.debug("Validating text parser block")
            self.text_parser._validate_custom(current_dataset)

            # Simulate text parser output for final validation
            if "relevancy_score" not in current_dataset.column_names:
                temp_data = []
                for sample in current_dataset:
                    temp_sample = dict(sample)
                    temp_sample["relevancy_explanation"] = "Test feedback"
                    temp_sample["relevancy_score"] = "2.0"
                    temp_data.append(temp_sample)
                current_dataset = Dataset.from_list(temp_data)

            # 4. Validate ColumnValueFilterBlock
            logger.debug("Validating filter block")
            self.filter_block._validate_custom(current_dataset)

            logger.debug("All internal blocks validated successfully")

        except Exception as e:
            logger.error(f"Validation failed in internal blocks: {e}")
            raise ValueError(f"Internal block validation failed: {e}") from e

    def get_internal_blocks_info(self) -> Dict[str, Any]:
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
        return (
            f"EvaluateRelevancyBlock(name='{self.block_name}', "
            f"model='{self.model}', filter_value='{self.filter_value}')"
        )