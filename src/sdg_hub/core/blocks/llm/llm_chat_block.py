# SPDX-License-Identifier: Apache-2.0
"""Unified LLM chat block supporting all providers via LiteLLM."""

# Standard
from typing import Any, Dict, List, Optional, Union
import asyncio

# Third Party
from datasets import Dataset
from pydantic import Field, field_validator

# Local
from ...utils.logger_config import setup_logger
from ...utils.error_handling import BlockValidationError
from ..base import BaseBlock
from ..registry import BlockRegistry
from .client_manager import LLMClientManager
from .config import LLMConfig

logger = setup_logger(__name__)


@BlockRegistry.register(
    "LLMChatBlock",
    "llm",
    "Unified LLM chat block supporting 100+ providers via LiteLLM",
)
class LLMChatBlock(BaseBlock):
    """Unified LLM chat block supporting all providers via LiteLLM.

    This block replaces OpenAIChatBlock and OpenAIAsyncChatBlock with a single
    implementation that supports 100+ LLM providers through LiteLLM, including:
    - OpenAI (GPT-3.5, GPT-4, etc.)
    - Anthropic (Claude models)
    - Google (Gemini, PaLM)
    - Local models (vLLM, Ollama, etc.)
    - And many more...

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Input column name(s). Should contain the messages list.
    output_cols : Union[str, List[str]]
        Output column name(s) for the response. When n > 1, the column will contain
        a list of responses instead of a single string.
    model : str
        Model identifier in LiteLLM format. Examples:
        - "openai/gpt-4"
        - "anthropic/claude-3-sonnet-20240229"
        - "hosted_vllm/meta-llama/Llama-2-7b-chat-hf"
    api_key : Optional[str], optional
        API key for the provider. Falls back to environment variables.
    api_base : Optional[str], optional
        Base URL for the API. Required for local models.
    async_mode : bool, optional
        Whether to use async processing, by default False.
    timeout : float, optional
        Request timeout in seconds, by default 120.0.
    max_retries : int, optional
        Maximum number of retry attempts, by default 6.

    ### Generation Parameters ###

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
    **kwargs : Any
        Additional provider-specific parameters.

    Examples
    --------
    >>> # OpenAI GPT-4
    >>> block = LLMChatBlock(
    ...     block_name="gpt4_block",
    ...     input_cols="messages",
    ...     output_cols="response",
    ...     model="openai/gpt-4",
    ...     temperature=0.7
    ... )

    >>> # Anthropic Claude
    >>> block = LLMChatBlock(
    ...     block_name="claude_block",
    ...     input_cols="messages",
    ...     output_cols="response",
    ...     model="anthropic/claude-3-sonnet-20240229",
    ...     temperature=0.7
    ... )

    >>> # Local vLLM model
    >>> block = LLMChatBlock(
    ...     block_name="local_llama",
    ...     input_cols="messages",
    ...     output_cols="response",
    ...     model="hosted_vllm/meta-llama/Llama-2-7b-chat-hf",
    ...     api_base="http://localhost:8000/v1",
    ...     temperature=0.7
    ... )

    >>> # Multiple completions (n > 1)
    >>> block = LLMChatBlock(
    ...     block_name="gpt4_multiple",
    ...     input_cols="messages",
    ...     output_cols="responses",  # Will contain lists of strings
    ...     model="openai/gpt-4",
    ...     n=3,  # Generate 3 responses per input
    ...     temperature=0.8
    ... )
    """

    # LLM Configuration
    model: str = Field(..., description="Model identifier in LiteLLM format")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    api_base: Optional[str] = Field(None, description="Base URL for the API")
    async_mode: bool = Field(False, description="Whether to use async processing")
    timeout: float = Field(120.0, description="Request timeout in seconds")
    max_retries: int = Field(6, description="Maximum number of retry attempts")

    # Generation parameters
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
    extra_headers: Optional[Dict[str, str]] = Field(
        None, description="Additional headers"
    )
    extra_body: Optional[Dict[str, Any]] = Field(
        None, description="Additional request body parameters"
    )
    provider_specific: Optional[Dict[str, Any]] = Field(
        None, description="Provider-specific parameters"
    )

    # Exclude from serialization - internal computed field
    client_manager: Optional[Any] = Field(
        None, exclude=True, description="Internal client manager"
    )

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
                f"LLMChatBlock expects exactly one input column, got {len(v)}: {v}"
            )
        raise ValueError(f"Invalid input_cols format: {v}")

    @field_validator("output_cols")
    @classmethod
    def validate_single_output_col(cls, v):
        """Ensure exactly one output column."""
        if isinstance(v, str):
            return [v]
        if isinstance(v, list) and len(v) == 1:
            return v
        if isinstance(v, list) and len(v) != 1:
            raise ValueError(
                f"LLMChatBlock expects exactly one output column, got {len(v)}: {v}"
            )
        raise ValueError(f"Invalid output_cols format: {v}")

    def model_post_init(self, __context) -> None:
        """Initialize after Pydantic validation."""
        super().model_post_init(__context)

        # Convenience properties removed - use self.input_cols[0] and self.output_cols[0] directly

        # Create configuration
        config = LLMConfig(
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            timeout=self.timeout,
            max_retries=self.max_retries,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self.stop,
            seed=self.seed,
            response_format=self.response_format,
            stream=self.stream,
            n=self.n,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
            user=self.user,
            extra_headers=self.extra_headers,
            extra_body=self.extra_body,
            provider_specific=self.provider_specific,
        )

        # Create client manager
        self.client_manager = LLMClientManager(config)

        # Load client immediately
        self.client_manager.load()

        # Log initialization
        logger.info(
            f"Initialized LLMChatBlock '{self.block_name}' with model '{self.model}'",
            extra={
                "block_name": self.block_name,
                "model": self.model,
                "provider": self.client_manager.config.get_provider(),
                "is_local": self.client_manager.config.is_local_model(),
                "async_mode": self.async_mode,
                "generation_params": self.client_manager.config.get_generation_kwargs(),
            },
        )

    def generate(self, samples: Dataset, **override_kwargs: Dict[str, Any]) -> Dataset:
        """Generate responses from the LLM.

        Parameters set at runtime override those set during initialization.
        Supports all LiteLLM parameters for the configured provider.

        Parameters
        ----------
        samples : Dataset
            Input dataset containing the messages column.
        **override_kwargs : Dict[str, Any]
            Runtime parameters that override initialization defaults.
            Valid parameters depend on the provider but typically include:
            temperature, max_tokens, top_p, frequency_penalty, presence_penalty,
            stop, seed, response_format, stream, n, and provider-specific params.

        Returns
        -------
        Dataset
            Dataset with responses added to the output column.
        """
        # Extract messages
        messages_list = samples[self.input_cols[0]]

        # Log generation start
        logger.info(
            f"Starting {'async' if self.async_mode else 'sync'} generation for {len(messages_list)} samples",
            extra={
                "block_name": self.block_name,
                "model": self.model,
                "provider": self.client_manager.config.get_provider(),
                "batch_size": len(messages_list),
                "async_mode": self.async_mode,
                "override_params": override_kwargs,
            },
        )

        # Generate responses
        if self.async_mode:
            responses = asyncio.run(
                self._generate_async(messages_list, **override_kwargs)
            )
        else:
            responses = self._generate_sync(messages_list, **override_kwargs)

        # Log completion
        logger.info(
            f"Generation completed successfully for {len(responses)} samples",
            extra={
                "block_name": self.block_name,
                "model": self.model,
                "provider": self.client_manager.config.get_provider(),
                "batch_size": len(responses),
            },
        )

        # Add responses as new column
        return samples.add_column(self.output_cols[0], responses)

    def _generate_sync(
        self,
        messages_list: List[List[Dict[str, Any]]],
        **override_kwargs: Dict[str, Any],
    ) -> List[Union[str, List[str]]]:
        """Generate responses synchronously.

        Parameters
        ----------
        messages_list : List[List[Dict[str, Any]]]
            List of message lists to process.
        **override_kwargs : Dict[str, Any]
            Runtime parameter overrides.

        Returns
        -------
        List[Union[str, List[str]]]
            List of response strings or lists of response strings (when n > 1).
        """
        responses = []

        for i, messages in enumerate(messages_list):
            try:
                response = self.client_manager.create_completion(
                    messages, **override_kwargs
                )
                responses.append(response)

                # Log progress for large batches
                if (i + 1) % 10 == 0:
                    logger.debug(
                        f"Generated {i + 1}/{len(messages_list)} responses",
                        extra={
                            "block_name": self.block_name,
                            "progress": f"{i + 1}/{len(messages_list)}",
                        },
                    )

            except Exception as e:
                error_msg = self.client_manager.error_handler.format_error_message(
                    e, {"model": self.model, "sample_index": i}
                )
                logger.error(
                    f"Failed to generate response for sample {i}: {error_msg}",
                    extra={
                        "block_name": self.block_name,
                        "sample_index": i,
                        "error": str(e),
                    },
                )
                raise

        return responses

    async def _generate_async(
        self,
        messages_list: List[List[Dict[str, Any]]],
        **override_kwargs: Dict[str, Any],
    ) -> List[Union[str, List[str]]]:
        """Generate responses asynchronously.

        Parameters
        ----------
        messages_list : List[List[Dict[str, Any]]]
            List of message lists to process.
        **override_kwargs : Dict[str, Any]
            Runtime parameter overrides.

        Returns
        -------
        List[Union[str, List[str]]]
            List of response strings or lists of response strings (when n > 1).
        """
        try:
            responses = await self.client_manager.acreate_completions_batch(
                messages_list, **override_kwargs
            )
            return responses

        except Exception as e:
            error_msg = self.client_manager.error_handler.format_error_message(
                e, {"model": self.model}
            )
            logger.error(
                f"Failed to generate async responses: {error_msg}",
                extra={
                    "block_name": self.block_name,
                    "batch_size": len(messages_list),
                    "error": str(e),
                },
            )
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured model.

        Returns
        -------
        Dict[str, Any]
            Model information including provider, capabilities, etc.
        """
        return {
            **self.client_manager.get_model_info(),
            "block_name": self.block_name,
            "input_column": self.input_cols[0],
            "output_column": self.output_cols[0],
            "async_mode": self.async_mode,
        }

    def _validate_custom(self, dataset: Dataset) -> None:
        """Custom validation for LLMChatBlock message format.

        Validates that all samples contain properly formatted messages.

        Parameters
        ----------
        dataset : Dataset
            The dataset to validate.

        Raises
        ------
        BlockValidationError
            If message format validation fails.
        """

        def validate_sample(sample_with_index):
            """Validate a single sample's message format."""
            idx, sample = sample_with_index
            messages = sample[self.input_cols[0]]

            # Validate messages is a list
            if not isinstance(messages, list):
                raise BlockValidationError(
                    f"Messages column '{self.input_cols[0]}' must contain a list, "
                    f"got {type(messages)} in row {idx}",
                    details=f"Block: {self.block_name}, Row: {idx}, Value: {messages}",
                )

            # Validate messages is not empty
            if not messages:
                raise BlockValidationError(
                    f"Messages list is empty in row {idx}",
                    details=f"Block: {self.block_name}, Row: {idx}",
                )

            # Validate each message format
            for msg_idx, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise BlockValidationError(
                        f"Message {msg_idx} in row {idx} must be a dict, got {type(message)}",
                        details=f"Block: {self.block_name}, Row: {idx}, Message: {msg_idx}, Value: {message}",
                    )

                # Validate required fields
                if "role" not in message or message["role"] is None:
                    raise BlockValidationError(
                        f"Message {msg_idx} in row {idx} missing required 'role' field",
                        details=f"Block: {self.block_name}, Row: {idx}, Message: {msg_idx}, Available fields: {list(message.keys())}",
                    )

                if "content" not in message or message["content"] is None:
                    raise BlockValidationError(
                        f"Message {msg_idx} in row {idx} missing required 'content' field",
                        details=f"Block: {self.block_name}, Row: {idx}, Message: {msg_idx}, Available fields: {list(message.keys())}",
                    )

            return True  # Return something for map

        # Use map to validate all samples
        # Add index to each sample for better error reporting
        indexed_samples = [(i, sample) for i, sample in enumerate(dataset)]
        list(map(validate_sample, indexed_samples))

    def __del__(self) -> None:
        """Cleanup when block is destroyed."""
        try:
            if hasattr(self, "client_manager"):
                self.client_manager.unload()
        except Exception:
            # Ignore errors during cleanup to prevent issues during shutdown
            pass

    def __repr__(self) -> str:
        """String representation of the block."""
        return (
            f"LLMChatBlock(name='{self.block_name}', model='{self.model}', "
            f"provider='{self.client_manager.config.get_provider()}', async_mode={self.async_mode})"
        )
