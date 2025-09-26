# SPDX-License-Identifier: Apache-2.0
"""Unified LLM chat block supporting all providers via LiteLLM."""

# Standard
from typing import Any, Optional
import asyncio

# Third Party
from datasets import Dataset
from litellm import acompletion, completion
from pydantic import ConfigDict, Field, field_validator
import litellm

from ...utils.error_handling import BlockValidationError
from ...utils.logger_config import setup_logger

# Local
from ..base import BaseBlock
from ..registry import BlockRegistry

litellm.drop_params = True
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
    num_retries : int, optional
        Number of retry attempts (uses LiteLLM's built-in retry mechanism), by default 6.
        Note: For rate limit handling, use LiteLLM's fallbacks parameter instead.
    drop_params : bool, optional
        Whether to drop unsupported parameters to prevent API errors, by default True.
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
    num_retries: int = Field(
        6,
        exclude=True,
        description="Number of retry attempts (uses LiteLLM's built-in retry mechanism)",
    )
    drop_params: bool = Field(
        True, description="Whether to drop unsupported parameters to prevent API errors"
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
                "Initialized LLMChatBlock '%s' with model '%s'",
                self.block_name,
                self.model,
                extra={
                    "block_name": self.block_name,
                    "model": self.model,
                    "async_mode": self.async_mode,
                },
            )

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
            "Starting %s generation for %d samples%s",
            "async" if self.async_mode else "sync",
            len(messages_list),
            (
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
            "Generation completed successfully for %d samples",
            len(responses),
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
        if self.num_retries is not None:
            completion_kwargs["num_retries"] = self.num_retries

        # Apply only non-block-field overrides (flow params + unknown LiteLLM params)
        # BaseBlock already handles block field overrides by modifying instance attributes
        non_block_overrides = {
            k: v for k, v in overrides.items() if k not in self.__class__.model_fields
        }
        completion_kwargs.update(non_block_overrides)

        # Ensure drop_params is set to handle unknown parameters gracefully
        completion_kwargs["drop_params"] = self.drop_params

        return completion_kwargs

    def _message_to_dict(self, message) -> dict[str, Any]:
        """Convert LiteLLM message to dict."""
        return {"content": message.content, **getattr(message, "__dict__", {})}

    def _generate_sync(
        self,
        messages_list: list[list[dict[str, Any]]],
        completion_kwargs: dict[str, Any],
    ) -> list[list[dict[str, Any]]]:
        """Generate responses synchronously.

        Parameters
        ----------
        messages_list : list[list[dict[str, Any]]]
            List of message lists to process.
        completion_kwargs : dict[str, Any]
            Kwargs for LiteLLM completion.

        Returns
        -------
        list[list[dict[str, Any]]]
            List of response lists, each containing LiteLLM completion response dictionaries.
        """
        responses = []

        for i, messages in enumerate(messages_list):
            try:
                response = completion(messages=messages, **completion_kwargs)
                # Extract response based on n parameter
                n_value = completion_kwargs.get("n", 1)
                if n_value > 1:
                    response_data = [
                        self._message_to_dict(choice.message)
                        for choice in response.choices
                    ]
                else:
                    response_data = [self._message_to_dict(response.choices[0].message)]
                responses.append(response_data)

                # Log progress for large batches
                if (i + 1) % 10 == 0:
                    logger.debug(
                        "Generated %d/%d responses",
                        i + 1,
                        len(messages_list),
                        extra={
                            "block_name": self.block_name,
                            "progress": f"{i + 1}/{len(messages_list)}",
                        },
                    )

            except Exception as e:
                logger.error(
                    "Failed to generate response for sample %d: %s",
                    i,
                    str(e),
                    extra={
                        "block_name": self.block_name,
                        "sample_index": i,
                        "error": str(e),
                    },
                )
                raise

        return responses

    async def _make_acompletion(
        self,
        messages: list[dict[str, Any]],
        completion_kwargs: dict[str, Any],
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> list[dict[str, Any]]:
        """Make a single async completion with optional concurrency control.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            Messages for this completion.
        completion_kwargs : dict[str, Any]
            Kwargs for LiteLLM acompletion.
        semaphore : Optional[asyncio.Semaphore], optional
            Semaphore for concurrency control.

        Returns
        -------
        list[dict[str, Any]]
            List of response dictionaries.
        """
        if semaphore:
            async with semaphore:
                response = await acompletion(messages=messages, **completion_kwargs)
        else:
            response = await acompletion(messages=messages, **completion_kwargs)

        # Extract response based on n parameter
        n_value = completion_kwargs.get("n", 1)
        if n_value > 1:
            return [
                self._message_to_dict(choice.message) for choice in response.choices
            ]
        return [self._message_to_dict(response.choices[0].message)]

    async def _generate_async(
        self,
        messages_list: list[list[dict[str, Any]]],
        completion_kwargs: dict[str, Any],
        flow_max_concurrency: Optional[int] = None,
    ) -> list[list[dict[str, Any]]]:
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
        list[list[dict[str, Any]]]
            List of response lists, each containing LiteLLM completion response dictionaries.
        """

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
                            "Adjusted max_concurrency from %d to %d for n=%d completions per request",
                            flow_max_concurrency,
                            effective_concurrency,
                            n_value,
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
                            "max_concurrency (%d) is less than n (%d). Consider increasing max_concurrency for optimal performance.",
                            flow_max_concurrency,
                            n_value,
                            extra={
                                "block_name": self.block_name,
                                "max_concurrency": flow_max_concurrency,
                                "n_value": n_value,
                            },
                        )
                        effective_concurrency = flow_max_concurrency

                # Use semaphore for concurrency control
                semaphore = asyncio.Semaphore(effective_concurrency)
                tasks = [
                    self._make_acompletion(messages, completion_kwargs, semaphore)
                    for messages in messages_list
                ]
            else:
                # No concurrency limit
                tasks = [
                    self._make_acompletion(messages, completion_kwargs)
                    for messages in messages_list
                ]

            responses = await asyncio.gather(*tasks)
            return responses

        except Exception as e:
            logger.error(
                "Failed to generate async responses: %s",
                str(e),
                extra={
                    "block_name": self.block_name,
                    "batch_size": len(messages_list),
                    "error": str(e),
                },
            )
            raise

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
