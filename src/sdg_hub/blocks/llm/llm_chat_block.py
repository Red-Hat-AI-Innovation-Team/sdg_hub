# SPDX-License-Identifier: Apache-2.0
"""Unified LLM chat block supporting all providers via LiteLLM."""

# Standard
from typing import Any, Dict, List, Optional, Union
import asyncio

# Third Party
from datasets import Dataset

# Local
from ...logger_config import setup_logger
from ...registry import BlockRegistry
from ..block import Block
from .client_manager import LLMClientManager
from .config import LLMConfig

logger = setup_logger(__name__)


@BlockRegistry.register("LLMChatBlock")
class LLMChatBlock(Block):
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
        Output column name(s) for the response.
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
        Number of completions to generate.
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
    """

    def __init__(
        self,
        block_name: str,
        input_cols: Union[str, List[str]],
        output_cols: Union[str, List[str]],
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        async_mode: bool = False,
        timeout: float = 120.0,
        max_retries: int = 6,
        # Generation parameters
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stream: Optional[bool] = None,
        n: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(block_name)

        # Validate input/output columns
        self.input_cols = [input_cols] if isinstance(input_cols, str) else input_cols
        self.output_cols = (
            [output_cols] if isinstance(output_cols, str) else output_cols
        )

        if len(self.input_cols) != 1:
            raise ValueError("LLMChatBlock expects exactly one input column")
        if len(self.output_cols) != 1:
            raise ValueError("LLMChatBlock expects exactly one output column")

        self.messages_column = self.input_cols[0]
        self.output_column = self.output_cols[0]
        self.async_mode = async_mode

        # Create configuration
        self.config = LLMConfig(
            model=model,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            seed=seed,
            response_format=response_format,
            stream=stream,
            n=n,
            provider_specific=kwargs if kwargs else None,
        )

        # Create client manager
        self.client_manager = LLMClientManager(self.config)

        # Load client immediately
        self.client_manager.load()

        # Log initialization
        logger.info(
            f"Initialized LLMChatBlock '{block_name}' with model '{model}'",
            extra={
                "block_name": block_name,
                "model": model,
                "provider": self.config.get_provider(),
                "is_local": self.config.is_local_model(),
                "async_mode": async_mode,
                "generation_params": self.config.get_generation_kwargs(),
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
        messages_list = samples[self.messages_column]

        # Log generation start
        logger.info(
            f"Starting {'async' if self.async_mode else 'sync'} generation for {len(messages_list)} samples",
            extra={
                "block_name": self.block_name,
                "model": self.config.model,
                "provider": self.config.get_provider(),
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
                "model": self.config.model,
                "provider": self.config.get_provider(),
                "batch_size": len(responses),
            },
        )

        # Add responses as new column
        return samples.add_column(self.output_column, responses)

    def _generate_sync(
        self,
        messages_list: List[List[Dict[str, Any]]],
        **override_kwargs: Dict[str, Any],
    ) -> List[str]:
        """Generate responses synchronously.

        Parameters
        ----------
        messages_list : List[List[Dict[str, Any]]]
            List of message lists to process.
        **override_kwargs : Dict[str, Any]
            Runtime parameter overrides.

        Returns
        -------
        List[str]
            List of response strings.
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
                    e, {"model": self.config.model, "sample_index": i}
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
    ) -> List[str]:
        """Generate responses asynchronously.

        Parameters
        ----------
        messages_list : List[List[Dict[str, Any]]]
            List of message lists to process.
        **override_kwargs : Dict[str, Any]
            Runtime parameter overrides.

        Returns
        -------
        List[str]
            List of response strings.
        """
        try:
            responses = await self.client_manager.acreate_completions_batch(
                messages_list, **override_kwargs
            )
            return responses

        except Exception as e:
            error_msg = self.client_manager.error_handler.format_error_message(
                e, {"model": self.config.model}
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
            "input_column": self.messages_column,
            "output_column": self.output_column,
            "async_mode": self.async_mode,
        }

    def validate_input(self, sample: Dict[str, Any]) -> bool:
        """Validate that a sample has the required input format.

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample to validate.

        Returns
        -------
        bool
            True if the sample is valid.
        """
        if self.messages_column not in sample:
            logger.warning(
                f"Sample missing required column '{self.messages_column}'",
                extra={"block_name": self.block_name, "sample": sample},
            )
            return False

        messages = sample[self.messages_column]

        if not isinstance(messages, list):
            logger.warning(
                f"Messages column must be a list, got {type(messages)}",
                extra={"block_name": self.block_name, "messages_type": type(messages)},
            )
            return False

        if not messages:
            logger.warning(
                "Messages list is empty", extra={"block_name": self.block_name}
            )
            return False

        # Validate message format
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                logger.warning(
                    f"Message {i} must be a dict, got {type(message)}",
                    extra={"block_name": self.block_name, "message_index": i},
                )
                return False

            if "role" not in message or "content" not in message:
                logger.warning(
                    f"Message {i} must have 'role' and 'content' fields",
                    extra={
                        "block_name": self.block_name,
                        "message_index": i,
                        "message_data": message,
                    },
                )
                return False

        return True

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
            f"LLMChatBlock(name='{self.block_name}', model='{self.config.model}', "
            f"provider='{self.config.get_provider()}', async_mode={self.async_mode})"
        )
