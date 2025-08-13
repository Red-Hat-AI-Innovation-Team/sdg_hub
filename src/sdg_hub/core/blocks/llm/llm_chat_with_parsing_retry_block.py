# SPDX-License-Identifier: Apache-2.0
"""Composite block combining LLM chat and text parsing with retry logic.

This module provides the LLMChatWithParsingRetryBlock that encapsulates the complete
LLM generation and parsing workflow with automatic retry on parsing failures.
"""

# Standard
from typing import Any, Optional, Union

# Third Party
from datasets import Dataset
from pydantic import ConfigDict, Field, field_validator

# Local
from ...utils.logger_config import setup_logger
from ...utils.error_handling import BlockValidationError
from ..base import BaseBlock
from ..registry import BlockRegistry
from .llm_chat_block import LLMChatBlock
from .text_parser_block import TextParserBlock

logger = setup_logger(__name__)


class MaxRetriesExceededError(Exception):
    """Raised when maximum retry attempts are exceeded without achieving target count."""

    def __init__(self, target_count: int, actual_count: int, max_retries: int):
        self.target_count = target_count
        self.actual_count = actual_count
        self.max_retries = max_retries
        super().__init__(
            f"Failed to achieve target count {target_count} after {max_retries} retries. "
            f"Only got {actual_count} successful parses."
        )


@BlockRegistry.register(
    "LLMChatWithParsingRetryBlock",
    "llm",
    "Composite block combining LLM chat and text parsing with automatic retry on parsing failures",
)
class LLMChatWithParsingRetryBlock(BaseBlock):
    """Composite block for LLM generation with parsing retry logic.

    This block combines LLMChatBlock and TextParserBlock into a single cohesive block
    that automatically retries LLM generation when parsing fails, accumulating successful
    results until the target count is reached or max retries exceeded.

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Input column name(s). Should contain the messages list.
    output_cols : Union[str, List[str]]
        Output column name(s) for parsed results.
    model : str
        Model identifier in LiteLLM format.
    api_base : Optional[str]
        Base URL for the API. Required for local models.
    api_key : Optional[str]
        API key for the provider. Falls back to environment variables.
    parsing_max_retries : int, optional
        Maximum number of retry attempts for parsing failures (default: 3).
        This is different from max_retries, which handles LLM network/API failures.

    ### LLM Generation Parameters ###
    async_mode : bool, optional
        Whether to use async processing (default: False).
    timeout : float, optional
        Request timeout in seconds (default: 120.0).
    max_retries : int, optional
        Maximum number of LLM retry attempts for network failures (default: 6).
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
        Number of completions to generate per retry attempt.
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
    provider_specific : Optional[Dict[str, Any]], optional
        Provider-specific parameters.

    ### Text Parser Parameters ###
    start_tags : List[str], optional
        List of start tags for tag-based parsing.
    end_tags : List[str], optional
        List of end tags for tag-based parsing.
    parsing_pattern : Optional[str], optional
        Regex pattern for custom parsing.
    parser_cleanup_tags : Optional[List[str]], optional
        List of tags to clean from parsed output.

    Examples
    --------
    >>> # Basic JSON parsing with retry
    >>> block = LLMChatWithParsingRetryBlock(
    ...     block_name="json_retry_block",
    ...     input_cols="messages",
    ...     output_cols="parsed_json",
    ...     model="openai/gpt-4",
    ...     parsing_max_retries=3,
    ...     parsing_pattern=r'"result":\s*"([^"]*)"',
    ...     n=3
    ... )

    >>> # Tag-based parsing with retry
    >>> block = LLMChatWithParsingRetryBlock(
    ...     block_name="tag_retry_block",
    ...     input_cols="messages",
    ...     output_cols=["explanation", "answer"],
    ...     model="anthropic/claude-3-sonnet-20240229",
    ...     parsing_max_retries=5,
    ...     start_tags=["<explanation>", "<answer>"],
    ...     end_tags=["</explanation>", "</answer>"],
    ...     n=2
    ... )
    """

    model_config = ConfigDict(extra="forbid")

    # Core configuration
    model: Optional[str] = Field(None, description="Model identifier in LiteLLM format")
    api_base: Optional[str] = Field(None, description="Base URL for the API")
    api_key: Optional[str] = Field(
        None,
        description="API key for the provider. Falls back to environment variables.",
    )

    # Retry configuration
    parsing_max_retries: int = Field(
        3, description="Maximum number of retry attempts for parsing failures"
    )

    # LLM configuration
    async_mode: bool = Field(False, description="Whether to use async processing")
    timeout: float = Field(120.0, description="Request timeout in seconds")
    max_retries: int = Field(
        6, description="Maximum number of LLM retry attempts for network failures"
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
    stop: Optional[Union[str, list[str]]] = Field(None, description="Stop sequences")
    seed: Optional[int] = Field(
        None, description="Random seed for reproducible outputs"
    )
    response_format: Optional[dict[str, Any]] = Field(
        None, description="Response format specification"
    )
    stream: Optional[bool] = Field(None, description="Whether to stream responses")
    n: Optional[int] = Field(None, description="Number of completions to generate")
    logprobs: Optional[bool] = Field(
        None, description="Whether to return log probabilities"
    )
    top_logprobs: Optional[int] = Field(
        None, description="Number of top log probabilities to return"
    )
    user: Optional[str] = Field(None, description="End-user identifier")
    extra_headers: Optional[dict[str, str]] = Field(
        None, description="Additional headers"
    )
    extra_body: Optional[dict[str, Any]] = Field(
        None, description="Additional request body parameters"
    )
    provider_specific: Optional[dict[str, Any]] = Field(
        None, description="Provider-specific parameters"
    )

    # Text parser parameters
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

    # Internal blocks - excluded from serialization
    llm_chat: Optional[LLMChatBlock] = Field(None, exclude=True)
    text_parser: Optional[TextParserBlock] = Field(None, exclude=True)

    @field_validator("input_cols")
    @classmethod
    def validate_single_input_col(cls, v):
        """Ensure exactly one input column."""
        if isinstance(v, str):
            return [v]
        if isinstance(v, list) and len(v) == 1:
            return v
        if isinstance(v, list) and len(v) != 1:
            raise ValueError(
                f"LLMChatWithParsingRetryBlock expects exactly one input column, got {len(v)}: {v}"
            )
        raise ValueError(f"Invalid input_cols format: {v}")

    @field_validator("parsing_max_retries")
    @classmethod
    def validate_parsing_max_retries(cls, v):
        """Ensure parsing_max_retries is positive."""
        if v < 1:
            raise ValueError("parsing_max_retries must be at least 1")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Initialize the internal blocks after Pydantic validation."""
        super().model_post_init(__context)

        # Create internal blocks
        self._create_internal_blocks()

        # Log initialization only when model is configured
        if self.model:
            logger.info(
                f"Initialized LLMChatWithParsingRetryBlock '{self.block_name}' with model '{self.model}'",
                extra={
                    "block_name": self.block_name,
                    "model": self.model,
                    "async_mode": self.async_mode,
                    "parsing_max_retries": self.parsing_max_retries,
                },
            )

    def _create_internal_blocks(self) -> None:
        """Create and configure the internal blocks."""
        # 1. LLMChatBlock
        llm_kwargs = {
            "block_name": f"{self.block_name}_llm_chat",
            "input_cols": self.input_cols,
            "output_cols": [f"{self.block_name}_raw_response"],
            "model": self.model,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "async_mode": self.async_mode,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        # Add generation parameters (LLMChatBlock safely handles None values)
        llm_kwargs.update(
            {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "stop": self.stop,
                "seed": self.seed,
                "response_format": self.response_format,
                "stream": self.stream,
                "n": self.n,
                "logprobs": self.logprobs,
                "top_logprobs": self.top_logprobs,
                "user": self.user,
                "extra_headers": self.extra_headers,
                "extra_body": self.extra_body,
                "provider_specific": self.provider_specific,
            }
        )

        self.llm_chat = LLMChatBlock(**llm_kwargs)

        # 2. TextParserBlock
        text_parser_kwargs = {
            "block_name": f"{self.block_name}_text_parser",
            "input_cols": [f"{self.block_name}_raw_response"],
            "output_cols": self.output_cols,
            "start_tags": self.start_tags,
            "end_tags": self.end_tags,
        }

        # Add optional TextParserBlock parameters if specified
        if self.parsing_pattern is not None:
            text_parser_kwargs["parsing_pattern"] = self.parsing_pattern
        if self.parser_cleanup_tags is not None:
            text_parser_kwargs["parser_cleanup_tags"] = self.parser_cleanup_tags

        self.text_parser = TextParserBlock(**text_parser_kwargs)

    def _reinitialize_client_manager(self) -> None:
        """Reinitialize the internal LLM chat block's client manager.

        This should be called after model configuration changes to ensure
        the internal LLM chat block uses the updated model configuration.
        """
        if self.llm_chat and hasattr(self.llm_chat, "_reinitialize_client_manager"):
            # Update the internal LLM chat block's model config
            self.llm_chat.model = self.model
            self.llm_chat.api_base = self.api_base
            self.llm_chat.api_key = self.api_key
            # Reinitialize its client manager
            self.llm_chat._reinitialize_client_manager()

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate responses with parsing retry logic.

        For each input sample, this method:
        1. Generates LLM responses using the configured n parameter
        2. Attempts to parse the responses using TextParserBlock
        3. Counts successful parses and retries if below target
        4. Accumulates results across retry attempts
        5. Returns final dataset with all successful parses

        Parameters
        ----------
        samples : Dataset
            Input dataset containing the messages column.
        **kwargs : Any
            Additional keyword arguments passed to internal blocks.

        Returns
        -------
        Dataset
            Dataset with parsed results from successful generations.

        Raises
        ------
        BlockValidationError
            If model is not configured before calling generate().
        MaxRetriesExceededError
            If target count not reached after max retries for any sample.
        """
        # Validate that model is configured
        if not self.model:
            raise BlockValidationError(
                f"Model not configured for block '{self.block_name}'. "
                f"Call flow.set_model_config() before generating."
            )

        logger.info(
            f"Starting LLM generation with parsing retry for {len(samples)} samples",
            extra={
                "block_name": self.block_name,
                "model": self.model,
                "batch_size": len(samples),
                "parsing_max_retries": self.parsing_max_retries,
            },
        )

        all_results = []

        # Process each sample independently with retry logic
        for sample_idx, sample in enumerate(samples):
            sample_results = []
            total_parsed_count = 0

            # Determine target count for this sample (number of completions requested)
            target = kwargs.get("n", self.n) or 1

            logger.debug(
                f"Processing sample {sample_idx} with target count {target}",
                extra={
                    "block_name": self.block_name,
                    "sample_idx": sample_idx,
                    "target_count": target,
                },
            )

            # Retry loop for this sample
            for attempt in range(self.parsing_max_retries):
                if total_parsed_count >= target:
                    break  # Already reached target

                try:
                    # Generate LLM responses for this sample
                    temp_dataset = Dataset.from_list([sample])
                    llm_result = self.llm_chat.generate(temp_dataset, **kwargs)

                    # Parse the responses
                    parsed_result = self.text_parser.generate(llm_result, **kwargs)

                    # Count successful parses and accumulate results
                    new_parsed_count = len(parsed_result)
                    total_parsed_count += new_parsed_count
                    sample_results.extend(parsed_result)

                    logger.debug(
                        f"Attempt {attempt + 1} for sample {sample_idx}: {new_parsed_count} successful parses "
                        f"(total: {total_parsed_count}/{target})",
                        extra={
                            "block_name": self.block_name,
                            "sample_idx": sample_idx,
                            "attempt": attempt + 1,
                            "new_parses": new_parsed_count,
                            "total_parses": total_parsed_count,
                            "target_count": target,
                        },
                    )

                    if total_parsed_count >= target:
                        logger.debug(
                            f"Target reached for sample {sample_idx} after {attempt + 1} attempts",
                            extra={
                                "block_name": self.block_name,
                                "sample_idx": sample_idx,
                                "attempts": attempt + 1,
                                "final_count": total_parsed_count,
                            },
                        )
                        break

                except Exception as e:
                    logger.warning(
                        f"Error during attempt {attempt + 1} for sample {sample_idx}: {e}",
                        extra={
                            "block_name": self.block_name,
                            "sample_idx": sample_idx,
                            "attempt": attempt + 1,
                            "error": str(e),
                        },
                    )
                    # Continue to next attempt
                    continue

            # Check if we reached the target count
            if total_parsed_count < target:
                raise MaxRetriesExceededError(
                    target_count=target,
                    actual_count=total_parsed_count,
                    max_retries=self.parsing_max_retries,
                )

            # Trim results to exact target count if we exceeded it
            if total_parsed_count > target:
                sample_results = sample_results[:target]
                logger.debug(
                    f"Trimmed sample {sample_idx} results from {total_parsed_count} to {target}",
                    extra={
                        "block_name": self.block_name,
                        "sample_idx": sample_idx,
                        "trimmed_from": total_parsed_count,
                        "trimmed_to": target,
                    },
                )

            # Add this sample's results to final dataset
            all_results.extend(sample_results)

        logger.info(
            f"LLM generation with parsing retry completed: {len(samples)} input samples → {len(all_results)} output rows",
            extra={
                "block_name": self.block_name,
                "input_samples": len(samples),
                "output_rows": len(all_results),
                "model": self.model,
            },
        )

        return Dataset.from_list(all_results)

    def _validate_custom(self, dataset: Dataset) -> None:
        """Custom validation for LLMChatWithParsingRetryBlock.

        This method validates the entire chain of internal blocks by simulating
        the data flow through each block to ensure they can all process the data correctly.
        """
        # Validate that required input column exists
        if len(self.input_cols) != 1:
            raise ValueError(
                f"LLMChatWithParsingRetryBlock expects exactly one input column, got {len(self.input_cols)}"
            )

        input_col = self.input_cols[0]
        if input_col not in dataset.column_names:
            raise ValueError(
                f"Required input column '{input_col}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        # Validate parsing configuration
        has_regex = self.parsing_pattern is not None
        has_tags = bool(self.start_tags) or bool(self.end_tags)

        if not has_regex and not has_tags:
            raise ValueError(
                "LLMChatWithParsingRetryBlock requires at least one parsing method: "
                "either 'parsing_pattern' (regex) or 'start_tags'/'end_tags' (tag-based parsing)"
            )

        # Validate that internal blocks are initialized
        if not all([self.llm_chat, self.text_parser]):
            raise ValueError(
                "All internal blocks must be initialized before validation"
            )

        # Validate internal blocks
        try:
            logger.debug("Validating internal LLM chat block")
            self.llm_chat._validate_custom(dataset)

            # Create temporary dataset with expected LLM output for parser validation
            temp_data = []
            for sample in dataset:
                temp_sample = dict(sample)
                temp_sample[f"{self.block_name}_raw_response"] = "test output"
                temp_data.append(temp_sample)
            temp_dataset = Dataset.from_list(temp_data)

            logger.debug("Validating internal text parser block")
            self.text_parser._validate_custom(temp_dataset)

            logger.debug("All internal blocks validated successfully")

        except Exception as e:
            logger.error(f"Validation failed in internal blocks: {e}")
            raise ValueError(f"Internal block validation failed: {e}") from e

    def get_internal_blocks_info(self) -> dict[str, Any]:
        """Get information about the internal blocks.

        Returns
        -------
        Dict[str, Any]
            Information about each internal block.
        """
        return {
            "llm_chat": self.llm_chat.get_info() if self.llm_chat else None,
            "text_parser": self.text_parser.get_info() if self.text_parser else None,
        }

    def __repr__(self) -> str:
        """String representation of the block."""
        return (
            f"LLMChatWithParsingRetryBlock(name='{self.block_name}', "
            f"model='{self.model}', parsing_max_retries={self.parsing_max_retries})"
        )
