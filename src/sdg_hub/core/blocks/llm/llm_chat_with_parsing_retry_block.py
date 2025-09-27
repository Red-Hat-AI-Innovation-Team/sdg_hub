# SPDX-License-Identifier: Apache-2.0
"""Composite block combining LLM chat and text parsing with retry logic.

This module provides the LLMChatWithParsingRetryBlock that encapsulates the complete
LLM generation and parsing workflow with automatic retry on parsing failures.
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
from ..registry import BlockRegistry
from .llm_chat_block import LLMChatBlock
from .llm_parser_block import LLMParserBlock
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

    **llm_kwargs : Any
        Any LiteLLM completion parameters (model, api_base, api_key, temperature,
        max_tokens, top_p, frequency_penalty, presence_penalty, stop, seed,
        response_format, stream, n, logprobs, top_logprobs, user, extra_headers,
        extra_body, async_mode, timeout, num_retries, etc.).
        See https://docs.litellm.ai/docs/completion/input for full list.

    ### Text Parser Parameters ###
    start_tags : List[str], optional
        List of start tags for tag-based parsing.
    end_tags : List[str], optional
        List of end tags for tag-based parsing.
    parsing_pattern : Optional[str], optional
        Regex pattern for custom parsing.
    parser_cleanup_tags : Optional[List[str]], optional
        List of tags to clean from parsed output.

    ### LLMParserBlock Parameters ###
    extract_content : bool, optional
        Whether to extract 'content' field from responses.
    extract_reasoning_content : bool, optional
        Whether to extract 'reasoning_content' field from responses.
    extract_tool_calls : bool, optional
        Whether to extract 'tool_calls' field from responses.
    expand_lists : bool, optional
        Whether to expand list inputs into individual rows (True) or preserve lists (False).
    field_prefix : Optional[str], optional
        Prefix for the field names in the parsed output.

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

    model_config = ConfigDict(
        extra="allow"
    )  # Allow extra fields for dynamic forwarding

    # --- Composite-specific configuration ---
    parsing_max_retries: int = Field(
        3, description="Maximum number of retry attempts for parsing failures"
    )

    # --- Parser configuration (required for internal TextParserBlock) ---
    start_tags: Optional[list[str]] = Field(
        None, description="Start tags for tag-based parsing"
    )
    end_tags: Optional[list[str]] = Field(
        None, description="End tags for tag-based parsing"
    )
    parsing_pattern: Optional[str] = Field(
        None, description="Regex pattern for custom parsing"
    )
    parser_cleanup_tags: Optional[list[str]] = Field(
        None, description="List of tags to clean from parsed output"
    )

    ### LLMParserBlock Parameters ###
    extract_content: bool = Field(
        default=True, description="Whether to extract 'content' field from responses."
    )
    extract_reasoning_content: bool = Field(
        default=False,
        description="Whether to extract 'reasoning_content' field from responses.",
    )
    extract_tool_calls: bool = Field(
        default=False,
        description="Whether to extract 'tool_calls' field from responses.",
    )
    expand_lists: bool = Field(
        default=True,
        description="Whether to expand list inputs into individual rows (True) or preserve lists (False).",
    )
    field_prefix: Optional[str] = Field(
        default="", description="Prefix for the field names in the parsed output."
    )

    # Internal blocks - excluded from serialization
    llm_chat: Optional[LLMChatBlock] = Field(None, exclude=True)
    text_parser: Optional[TextParserBlock] = Field(None, exclude=True)
    llm_parser: Optional[LLMParserBlock] = Field(None, exclude=True)

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

    def __init__(self, **kwargs):
        """Initialize with dynamic parameter routing."""
        super().__init__(**kwargs)
        self._create_internal_blocks(**kwargs)

        # Log initialization if model is configured
        if self.llm_chat and self.llm_chat.model:
            logger.info(
                f"Initialized LLMChatWithParsingRetryBlock '{self.block_name}' with model '{self.llm_chat.model}'",
                extra={
                    "block_name": self.block_name,
                    "model": self.llm_chat.model,
                    "parsing_max_retries": self.parsing_max_retries,
                },
            )

    def _extract_params(self, kwargs: dict, block_class) -> dict:
        """Extract parameters for specific block class."""
        # Parameters that belong to this wrapper and shouldn't be forwarded
        wrapper_params = {
            "block_name",
            "input_cols",
            "output_cols",
            "parsing_max_retries",
        }

        if block_class == LLMChatBlock:
            # LLMChatBlock accepts any parameters via extra="allow"
            # Forward everything except wrapper-specific and parser-specific params
            parser_specific_params = {
                "start_tags",
                "end_tags",
                "parsing_pattern",
                "parser_cleanup_tags",
            }
            llm_parser_specific_params = {
                "extract_content",
                "extract_reasoning_content",
                "extract_tool_calls",
                "expand_lists",
                "field_prefix",
            }
            excluded_params = (
                wrapper_params | parser_specific_params | llm_parser_specific_params
            )

            # Forward all other kwargs
            params = {k: v for k, v in kwargs.items() if k not in excluded_params}

            # Also forward instance attributes that aren't parser-specific
            for field_name, field_value in self.__dict__.items():
                if (
                    field_name not in excluded_params
                    and not field_name.startswith("_")
                    and field_name not in ["llm_chat", "text_parser", "llm_parser"]
                    and field_value is not None
                ):
                    params[field_name] = field_value

        else:
            # For TextParserBlock, only forward known fields and parser-specific params
            non_llm_chat_params = {
                "start_tags",
                "end_tags",
                "parsing_pattern",
                "parser_cleanup_tags",
                "expand_lists",
                "field_prefix",
                "extract_content",
                "extract_reasoning_content",
                "extract_tool_calls",
            }

            # Forward parser-specific parameters from kwargs
            params = {
                k: v
                for k, v in kwargs.items()
                if k in block_class.model_fields and k not in wrapper_params
            }

            # Forward parser-specific instance attributes
            for field_name in non_llm_chat_params:
                if hasattr(self, field_name):
                    field_value = getattr(self, field_name)
                    if field_value is not None:
                        params[field_name] = field_value

        return params

    def _create_internal_blocks(self, **kwargs):
        """Create internal blocks with parameter routing."""
        # Route parameters to appropriate blocks
        llm_params = self._extract_params(kwargs, LLMChatBlock)
        parser_params = self._extract_params(kwargs, TextParserBlock)
        llm_parser_params = self._extract_params(kwargs, LLMParserBlock)

        # 1. LLMChatBlock
        self.llm_chat = LLMChatBlock(
            block_name=f"{self.block_name}_llm_chat",
            input_cols=self.input_cols,
            output_cols=[f"{self.block_name}_raw_response"],
            **llm_params,
        )

        # 2. LLMParserBlock
        self.llm_parser = LLMParserBlock(
            block_name=f"{self.block_name}_llm_parser",
            input_cols=[f"{self.block_name}_raw_response"],
            **llm_parser_params,
        )

        # 2. TextParserBlock
        self.text_parser = TextParserBlock(
            block_name=f"{self.block_name}_text_parser",
            input_cols=[
                f"{self.llm_parser.field_prefix if self.llm_parser.field_prefix!='' else self.llm_parser.block_name}_content"
            ],
            output_cols=self.output_cols,
            **parser_params,
        )

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to appropriate internal block."""
        # Parser-specific parameters go to text_parser
        parser_params = {
            "start_tags",
            "end_tags",
            "parsing_pattern",
            "parser_cleanup_tags",
        }
        llm_parser_params = {
            "extract_content",
            "extract_reasoning_content",
            "extract_tool_calls",
            "expand_lists",
            "field_prefix",
        }

        if name in parser_params and hasattr(self, "text_parser") and self.text_parser:
            return getattr(self.text_parser, name)

        if (
            name in llm_parser_params
            and hasattr(self, "llm_parser")
            and self.llm_parser
        ):
            return getattr(self.llm_parser, name)

        # Everything else goes to llm_chat (which accepts any parameters via extra="allow")
        if hasattr(self, "llm_chat") and self.llm_chat:
            # Always try LLMChatBlock - it will return None for unset attributes
            # due to extra="allow", which makes hasattr() work correctly
            return getattr(self.llm_chat, name, None)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle dynamic parameter updates from flow.set_model_config()."""
        super().__setattr__(name, value)

        # Don't forward during initialization or for internal attributes
        if not hasattr(self, "llm_chat") or name.startswith("_"):
            return

        # Parser-specific parameters go to text_parser
        parser_params = {
            "start_tags",
            "end_tags",
            "parsing_pattern",
            "parser_cleanup_tags",
        }
        llm_parser_params = {
            "extract_content",
            "extract_reasoning_content",
            "extract_tool_calls",
            "expand_lists",
            "field_prefix",
        }

        if name in parser_params and hasattr(self, "text_parser") and self.text_parser:
            setattr(self.text_parser, name, value)

        if (
            name in llm_parser_params
            and hasattr(self, "llm_parser")
            and self.llm_parser
        ):
            setattr(self.llm_parser, name, value)

        # LLM-related parameters go to llm_chat (which accepts any via extra="allow")
        elif (
            hasattr(self, "llm_chat")
            and self.llm_chat
            and name
            not in {
                "block_name",
                "input_cols",
                "output_cols",
                "parsing_max_retries",
                "llm_chat",
                "llm_parser",
                "text_parser",
            }
        ):
            setattr(self.llm_chat, name, value)

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
        # Validate that model is configured (check internal LLM block)
        if not self.llm_chat or not self.llm_chat.model:
            raise BlockValidationError(
                f"Model not configured for block '{self.block_name}'. "
                f"Call flow.set_model_config() before generating."
            )

        logger.info(
            f"Starting LLM generation with parsing retry for {len(samples)} samples",
            extra={
                "block_name": self.block_name,
                "model": self.llm_chat.model,
                "batch_size": len(samples),
                "parsing_max_retries": self.parsing_max_retries,
            },
        )

        all_results = []

        # Process each sample independently with retry logic
        for sample_idx, sample in enumerate(samples):
            # Determine target count for this sample (number of completions requested)
            target = kwargs.get("n", getattr(self, "n", None)) or 1

            logger.debug(
                f"Processing sample {sample_idx} with target count {target}",
                extra={
                    "block_name": self.block_name,
                    "sample_idx": sample_idx,
                    "target_count": target,
                },
            )

            if self.llm_parser.expand_lists:
                # Current behavior for expand_lists=True: count rows directly
                sample_results = []
                total_parsed_count = 0

                # Retry loop for this sample
                for attempt in range(self.parsing_max_retries):
                    if total_parsed_count >= target:
                        break  # Already reached target

                    try:
                        # Generate LLM responses for this sample
                        temp_dataset = Dataset.from_list([sample])
                        llm_result = self.llm_chat.generate(temp_dataset, **kwargs)
                        llm_parser_result = self.llm_parser.generate(
                            llm_result, **kwargs
                        )

                        # Parse the responses
                        parsed_result = self.text_parser.generate(
                            llm_parser_result, **kwargs
                        )

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

            else:
                # New behavior for expand_lists=False: parse individual responses and accumulate
                accumulated_parsed_items = {col: [] for col in self.output_cols}
                total_parsed_count = 0

                # Retry loop for this sample
                for attempt in range(self.parsing_max_retries):
                    if total_parsed_count >= target:
                        break  # Already reached target

                    try:
                        # Generate LLM responses for this sample
                        temp_dataset = Dataset.from_list([sample])
                        llm_result = self.llm_chat.generate(temp_dataset, **kwargs)
                        llm_parser_result = self.llm_parser.generate(
                            llm_result, **kwargs
                        )
                        # Get the raw responses (should be a list when n > 1)
                        raw_response_col = f"{self.llm_parser.field_prefix if self.llm_parser.field_prefix!='' else self.llm_parser.block_name}_content"
                        raw_responses = llm_parser_result[0][raw_response_col]
                        if not isinstance(raw_responses, list):
                            raw_responses = [raw_responses]

                        # Parse each response individually and accumulate successful ones
                        new_parsed_count = 0
                        for response in raw_responses:
                            if total_parsed_count >= target:
                                break  # Stop if we've reached target

                            # Create temporary dataset with single response for parsing
                            temp_parse_data = [{**sample, raw_response_col: response}]
                            temp_parse_dataset = Dataset.from_list(temp_parse_data)

                            # Force expand_lists=True temporarily to get individual parsed items
                            original_expand_lists = self.llm_parser.expand_lists
                            try:
                                self.llm_parser.expand_lists = (
                                    self.llm_parser.expand_lists
                                )
                                parsed_result = self.text_parser.generate(
                                    temp_parse_dataset, **kwargs
                                )
                            except Exception as parse_e:
                                logger.debug(
                                    f"Failed to parse individual response: {parse_e}"
                                )
                                continue
                            finally:
                                self.llm_parser.expand_lists = original_expand_lists

                            # If parsing was successful, accumulate the results
                            if len(parsed_result) > 0:
                                for parsed_row in parsed_result:
                                    if total_parsed_count >= target:
                                        break

                                    # Only count as successful if ALL output columns are present
                                    if all(
                                        col in parsed_row for col in self.output_cols
                                    ):
                                        for col in self.output_cols:
                                            accumulated_parsed_items[col].append(
                                                parsed_row[col]
                                            )
                                        total_parsed_count += 1
                                        new_parsed_count += 1
                                    # If any column is missing, skip this parsed response entirely

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

                # Create final result row with accumulated lists
                if total_parsed_count > 0:
                    # Trim to exact target count if needed
                    for col in self.output_cols:
                        if len(accumulated_parsed_items[col]) > target:
                            accumulated_parsed_items[col] = accumulated_parsed_items[
                                col
                            ][:target]

                    # Only add the parsed output columns as lists, preserve other columns as-is
                    final_row = {**sample, **accumulated_parsed_items}
                    sample_results = [final_row]
                else:
                    sample_results = []

            # Check if we reached the target count
            if total_parsed_count < target:
                raise MaxRetriesExceededError(
                    target_count=target,
                    actual_count=total_parsed_count,
                    max_retries=self.parsing_max_retries,
                )

            # For expand_lists=True, trim results to exact target count if we exceeded it
            if self.llm_parser.expand_lists and total_parsed_count > target:
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
                "model": self.llm_chat.model,
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
        has_regex = getattr(self, "parsing_pattern", None) is not None
        has_tags = bool(getattr(self, "start_tags", [])) or bool(
            getattr(self, "end_tags", [])
        )

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
            "llm_parser": self.llm_parser.get_info() if self.llm_parser else None,
            "text_parser": self.text_parser.get_info() if self.text_parser else None,
        }

    def __repr__(self) -> str:
        """String representation of the block."""
        model = (
            self.llm_chat.model
            if (self.llm_chat and self.llm_chat.model)
            else "not_configured"
        )
        return (
            f"LLMChatWithParsingRetryBlock(name='{self.block_name}', "
            f"model='{model}', parsing_max_retries={self.parsing_max_retries})"
        )
