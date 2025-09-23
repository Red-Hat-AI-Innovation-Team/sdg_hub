# SPDX-License-Identifier: Apache-2.0
"""Unified LLM chat block supporting all providers via LiteLLM."""

# Standard
from typing import Any, Optional, Union
import asyncio

# Third Party
from datasets import Dataset
from litellm import acompletion, completion, RateLimitError, APIConnectionError, InternalServerError, ServiceUnavailableError
from pydantic import ConfigDict, Field, field_validator
import time

from ...utils.error_handling import BlockValidationError
from ...utils.logger_config import setup_logger

# Local
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "LLMChatBlock",
    "llm",
    "Unified LLM chat block supporting 100+ providers via LiteLLM",
)
class LLMChatBlock(BaseBlock):
    model_config = ConfigDict(extra="allow")

    """Unified LLM chat block supporting all providers via LiteLLM.

    This block provides a minimal wrapper around LiteLLM's completion API,
    supporting 100+ LLM providers including:
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
    output_cols : Union[dict, List[dict]]
        Output column name(s) for the response.
    model : Optional[str], optional
        Model identifier in LiteLLM format. Can be set later via flow.set_model_config().
        Examples: "openai/gpt-4", "anthropic/claude-3-sonnet-20240229"
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
    **kwargs : Any
        Any LiteLLM completion parameters (temperature, max_tokens, top_p, etc.).
        See https://docs.litellm.ai/docs/completion/input for full list.

    Examples
    --------
    >>> # OpenAI GPT-4 with generation parameters
    >>> block = LLMChatBlock(
    ...     block_name="gpt4_block",
    ...     input_cols="messages",
    ...     output_cols="response",
    ...     model="openai/gpt-4",
    ...     temperature=0.7,
    ...     max_tokens=1000
    ... )

    >>> # Local vLLM model with custom parameters
    >>> block = LLMChatBlock(
    ...     block_name="local_llama",
    ...     input_cols="messages",
    ...     output_cols="response",
    ...     model="hosted_vllm/meta-llama/Llama-2-7b-chat-hf",
    ...     api_base="http://localhost:8000/v1",
    ...     temperature=0.7,
    ...     response_format={"type": "json_object"}
    ... )
    """

    # Essential operational fields (excluded from YAML serialization)
    model: Optional[str] = Field(
        None, exclude=True, description="Model identifier in LiteLLM format"
    )
    api_key: Optional[str] = Field(
        None, exclude=True, description="API key for the provider"
    )
    api_base: Optional[str] = Field(
        None, exclude=True, description="Base URL for the API"
    )
    async_mode: bool = Field(
        False, exclude=True, description="Whether to use async processing"
    )
    timeout: float = Field(
        120.0, exclude=True, description="Request timeout in seconds"
    )
    max_retries: int = Field(
        6, exclude=True, description="Maximum number of retry attempts"
    )

    # All LiteLLM completion parameters can be passed via extra="allow"
    # Common examples: temperature, max_tokens, top_p, frequency_penalty,
    # presence_penalty, stop, seed, response_format, stream, n, logprobs,
    # top_logprobs, user, extra_headers, extra_body, etc.

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

        # Log initialization only when model is configured
        if self.model:
            logger.info(
                f"Initialized LLMChatBlock '{self.block_name}' with model '{self.model}'",
                extra={
                    "block_name": self.block_name,
                    "model": self.model,
                    "async_mode": self.async_mode,
                },
            )

    def _reinitialize_client_manager(self) -> None:
        """Reinitialize client manager (no-op for simplified implementation).

        This method is called by Flow.set_model_config() to reinitialize
        LLM blocks after model configuration changes. Since our simplified
        implementation doesn't use a client manager, this is a no-op.
        """
        pass

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate responses from the LLM.

        Parameters
        ----------
        samples : Dataset
            Input dataset containing the messages column.
        **kwargs : Any
            Runtime parameters that override initialization defaults.
            Supports all LiteLLM completion parameters.

        Returns
        -------
        Dataset
            Dataset with responses added to the output column.

        Raises
        ------
        BlockValidationError
            If model is not configured before calling generate().
        """
        # Validate that model is configured
        if not self.model:
            raise BlockValidationError(
                f"Model not configured for block '{self.block_name}'. "
                f"Call flow.set_model_config() before generating."
            )

        # Extract flow-specific parameters (BaseBlock already handled block field overrides)
        flow_max_concurrency = kwargs.pop("_flow_max_concurrency", None)

        # Build completion kwargs from ALL fields + runtime overrides
        completion_kwargs = self._build_completion_kwargs(**kwargs)

        # Extract messages
        messages_list = samples[self.input_cols[0]]

        # Log generation start
        logger.info(
            f"Starting {'async' if self.async_mode else 'sync'} generation for {len(messages_list)} samples"
            + (
                f" (max_concurrency={flow_max_concurrency})"
                if flow_max_concurrency
                else ""
            ),
            extra={
                "block_name": self.block_name,
                "model": self.model,
                "batch_size": len(messages_list),
                "async_mode": self.async_mode,
                "flow_max_concurrency": flow_max_concurrency,
            },
        )

        # Generate responses
        if self.async_mode:
            try:
                # Check if there's already a running event loop
                loop = asyncio.get_running_loop()
                # Check if nest_asyncio is applied (allows nested asyncio.run)
                nest_asyncio_applied = (
                    hasattr(loop, "_nest_patched")
                    or getattr(asyncio.run, "__module__", "") == "nest_asyncio"
                )

                if nest_asyncio_applied:
                    # nest_asyncio is applied, safe to use asyncio.run
                    responses = asyncio.run(
                        self._generate_async(
                            messages_list, completion_kwargs, flow_max_concurrency
                        )
                    )
                else:
                    # Running inside an event loop without nest_asyncio
                    raise BlockValidationError(
                        f"async_mode=True cannot be used from within a running event loop for '{self.block_name}'. "
                        "Use an async entrypoint, set async_mode=False, or apply nest_asyncio.apply() in notebook environments."
                    )
            except RuntimeError:
                # No running loop; safe to create one
                responses = asyncio.run(
                    self._generate_async(
                        messages_list, completion_kwargs, flow_max_concurrency
                    )
                )
        else:
            responses = self._generate_sync(messages_list, completion_kwargs)

        # Log completion
        logger.info(
            f"Generation completed successfully for {len(responses)} samples",
            extra={
                "block_name": self.block_name,
                "model": self.model,
                "batch_size": len(responses),
            },
        )

        # Add responses as new column
        return samples.add_column(self.output_cols[0], responses)

    def _build_completion_kwargs(self, **overrides) -> dict[str, Any]:
        """Build kwargs for LiteLLM completion call.

        Returns
        -------
        dict[str, Any]
            Kwargs for litellm.completion() or litellm.acompletion().
        """
        # Start with extra fields (temperature, max_tokens, etc.) from extra="allow"
        extra_values = self.model_dump(exclude_unset=True)

        # Remove block-operational fields that shouldn't go to LiteLLM
        block_only_fields = {
            "block_name",
            "input_cols",
            "output_cols",
            "async_mode",
            "max_retries",
        }

        completion_kwargs = {
            k: v for k, v in extra_values.items() if k not in block_only_fields
        }

        # Add essential LiteLLM fields (even though they're excluded from serialization)
        if self.model is not None:
            completion_kwargs["model"] = self.model
        if self.api_key is not None:
            completion_kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            completion_kwargs["api_base"] = self.api_base
        if self.timeout is not None:
            completion_kwargs["timeout"] = self.timeout

        # Apply runtime overrides (from BaseBlock + Flow)
        completion_kwargs.update(overrides)

        return completion_kwargs

    def _completion_with_retry(self, messages, completion_kwargs):
        """Call LiteLLM completion with basic retry logic for rate limits and transient errors."""
        retryable_exceptions = (RateLimitError, APIConnectionError, InternalServerError, ServiceUnavailableError)
        max_retries = self.max_retries

        for attempt in range(max_retries + 1):
            try:
                return completion(messages=messages, **completion_kwargs)
            except retryable_exceptions as e:
                if attempt == max_retries:
                    # Last attempt failed, re-raise the exception
                    raise

                # Wait with exponential backoff
                wait_time = min(2 ** attempt, 60)  # Cap at 60 seconds
                logger.warning(
                    f"Retryable error on attempt {attempt + 1}/{max_retries + 1}: {str(e)}. "
                    f"Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            except Exception as e:
                # Non-retryable error, raise immediately
                raise

    async def _acompletion_with_retry(self, messages, completion_kwargs):
        """Call LiteLLM acompletion with basic retry logic for rate limits and transient errors."""
        retryable_exceptions = (RateLimitError, APIConnectionError, InternalServerError, ServiceUnavailableError)
        max_retries = self.max_retries

        for attempt in range(max_retries + 1):
            try:
                return await acompletion(messages=messages, **completion_kwargs)
            except retryable_exceptions as e:
                if attempt == max_retries:
                    # Last attempt failed, re-raise the exception
                    raise

                # Wait with exponential backoff
                wait_time = min(2 ** attempt, 60)  # Cap at 60 seconds
                logger.warning(
                    f"Retryable error on attempt {attempt + 1}/{max_retries + 1}: {str(e)}. "
                    f"Retrying in {wait_time} seconds..."
                )
                await asyncio.sleep(wait_time)
            except Exception as e:
                # Non-retryable error, raise immediately
                raise

    def _extract_response(self, response) -> Union[dict, list[dict]]:
        """Extract response content from LiteLLM response.

        Parameters
        ----------
        response : Any
            LiteLLM completion response.

        Returns
        -------
        Union[dict, list[dict]]
            Response dict(s) containing 'content' and other fields.
        """
        # Check if n > 1 to determine return type
        if len(response.choices) > 1:
            return [
                {
                    "content": choice.message.content,
                    **getattr(choice.message, "__dict__", {}),
                }
                for choice in response.choices
            ]
        else:
            message = response.choices[0].message
            return {"content": message.content, **getattr(message, "__dict__", {})}

    def _generate_sync(
        self,
        messages_list: list[list[dict[str, Any]]],
        completion_kwargs: dict[str, Any],
    ) -> list[Union[dict, list[dict]]]:
        """Generate responses synchronously.

        Parameters
        ----------
        messages_list : list[list[dict[str, Any]]]
            List of message lists to process.
        completion_kwargs : dict[str, Any]
            Kwargs for LiteLLM completion.

        Returns
        -------
        list[Union[dict, list[dict]]]
            List of responses.
        """
        responses = []

        for i, messages in enumerate(messages_list):
            try:
                response = self._completion_with_retry(messages, completion_kwargs)
                responses.append(self._extract_response(response))

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
                logger.error(
                    f"Failed to generate response for sample {i}: {str(e)}",
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
        messages_list: list[list[dict[str, Any]]],
        completion_kwargs: dict[str, Any],
        flow_max_concurrency: Optional[int] = None,
    ) -> list[Union[dict, list[dict]]]:
        """Generate responses asynchronously.

        Parameters
        ----------
        messages_list : list[list[dict[str, Any]]]
            List of message lists to process.
        completion_kwargs : dict[str, Any]
            Kwargs for LiteLLM acompletion.
        flow_max_concurrency : Optional[int], optional
            Maximum concurrency for async requests.

        Returns
        -------
        list[Union[dict, list[dict]]]
            List of responses.
        """

        async def _create_single(messages):
            """Create a single async completion."""
            response = await self._acompletion_with_retry(messages, completion_kwargs)
            return self._extract_response(response)

        try:
            if flow_max_concurrency is not None:
                # Validate max_concurrency parameter
                if flow_max_concurrency < 1:
                    raise ValueError(
                        f"max_concurrency must be greater than 0, got {flow_max_concurrency}"
                    )

                # Adjust concurrency based on n parameter (number of completions per request)
                effective_concurrency = flow_max_concurrency
                n_value = completion_kwargs.get("n", 1)

                if n_value and n_value > 1:
                    if flow_max_concurrency >= n_value:
                        # Adjust concurrency to account for n completions per request
                        effective_concurrency = flow_max_concurrency // n_value
                        logger.debug(
                            f"Adjusted max_concurrency from {flow_max_concurrency} to {effective_concurrency} "
                            f"for n={n_value} completions per request",
                            extra={
                                "block_name": self.block_name,
                                "original_max_concurrency": flow_max_concurrency,
                                "adjusted_max_concurrency": effective_concurrency,
                                "n_value": n_value,
                            },
                        )
                    else:
                        # Warn when max_concurrency is less than n
                        logger.warning(
                            f"max_concurrency ({flow_max_concurrency}) is less than n ({n_value}). "
                            f"Consider increasing max_concurrency for optimal performance.",
                            extra={
                                "block_name": self.block_name,
                                "max_concurrency": flow_max_concurrency,
                                "n_value": n_value,
                            },
                        )
                        effective_concurrency = flow_max_concurrency

                # Use semaphore for concurrency control
                semaphore = asyncio.Semaphore(effective_concurrency)

                async def _create_with_semaphore(messages):
                    async with semaphore:
                        return await _create_single(messages)

                tasks = [_create_with_semaphore(messages) for messages in messages_list]
            else:
                # No concurrency limit
                tasks = [_create_single(messages) for messages in messages_list]

            responses = await asyncio.gather(*tasks)
            return responses

        except Exception as e:
            logger.error(
                f"Failed to generate async responses: {str(e)}",
                extra={
                    "block_name": self.block_name,
                    "batch_size": len(messages_list),
                    "error": str(e),
                },
            )
            raise

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the configured model.

        Returns
        -------
        dict[str, Any]
            Model information.
        """
        provider = None
        model_name = None
        is_local = False

        if self.model:
            if "/" in self.model:
                provider = self.model.split("/")[0]
                model_name = self.model.split("/", 1)[1]
            else:
                model_name = self.model

            # Check if local model
            local_providers = {"hosted_vllm", "ollama", "local", "vllm"}
            is_local = provider and provider.lower() in local_providers

        return {
            "model": self.model,
            "provider": provider,
            "model_name": model_name,
            "is_local": is_local,
            "api_base": self.api_base,
            "block_name": self.block_name,
            "input_column": self.input_cols[0],
            "output_column": self.output_cols[0],
            "async_mode": self.async_mode,
        }

    def _validate_custom(self, dataset: Dataset) -> None:
        """Custom validation for LLMChatBlock message format.

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

            return True

        # Validate all samples
        indexed_samples = [(i, sample) for i, sample in enumerate(dataset)]
        list(map(validate_sample, indexed_samples))

    def __repr__(self) -> str:
        """String representation of the block."""
        provider = None
        if self.model and "/" in self.model:
            provider = self.model.split("/")[0]

        return (
            f"LLMChatBlock(name='{self.block_name}', model='{self.model}', "
            f"provider='{provider}', async_mode={self.async_mode})"
        )
