# SPDX-License-Identifier: Apache-2.0
"""OpenAI-specific blocks for text generation.

This module provides blocks for interacting with OpenAI's Chat Completions API.
"""

# Standard
from typing import Any, Dict, List, Optional, Union
import asyncio

# Third Party
from datasets import Dataset
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
import openai

# Local
from ..logger_config import setup_logger
from ..registry import BlockRegistry
from .block import Block

logger = setup_logger(__name__)


@BlockRegistry.register("OpenAIChatBlock")
class OpenAIChatBlock(Block):
    """Block for generating text using OpenAI Chat Completions API.

    This block takes a column containing OpenAI message format and makes
    direct calls to the chat completions endpoint.

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Input column name(s). Should contain the messages list.
    output_cols : Union[str, List[str]]
        Output column name(s) for the response.
    client : openai.OpenAI
        OpenAI client instance.
    model_id : str
        Model ID to use.

    ### Text-relevant OpenAI Chat Completions API parameters ###

    frequency_penalty : Optional[float], optional
        Penalize frequent tokens (-2.0 to 2.0).
    logit_bias : Optional[Dict[str, int]], optional
        Modify likelihood of specified tokens.
    logprobs : Optional[bool], optional
        Whether to return log probabilities.
    max_completion_tokens : Optional[int], optional
        Maximum tokens in completion.
    max_tokens : Optional[int], optional
        Maximum tokens in completion (legacy).
    n : Optional[int], optional
        Number of completions to generate.
    presence_penalty : Optional[float], optional
        Penalize repeated tokens (-2.0 to 2.0).
    response_format : Optional[Dict[str, Any]], optional
        Response format specification (e.g., JSON mode).
    seed : Optional[int], optional
        Seed for deterministic outputs.
    stop : Optional[Union[str, List[str]]], optional
        Stop sequences.
    stream : Optional[bool], optional
        Whether to stream responses.
    temperature : Optional[float], optional
        Sampling temperature (0.0 to 2.0).
    tool_choice : Optional[Union[str, Dict[str, Any]]], optional
        Tool selection strategy.
    tools : Optional[List[Dict[str, Any]]], optional
        Available tools for function calling.
    top_logprobs : Optional[int], optional
        Number of top log probabilities to return.
    top_p : Optional[float], optional
        Nucleus sampling parameter (0.0 to 1.0).
    user : Optional[str], optional
        End-user identifier.
    extra_body : Optional[dict], optional
        Dictionary of additional parameters if supported by inference backend
    """

    def __init__(
        self,
        block_name: str,
        input_cols: Union[str, List[str]],
        output_cols: Union[str, List[str]],
        client: openai.OpenAI,
        model_id: str,
        # Text-relevant OpenAI Chat Completions API parameters
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        extra_body: Optional[dict] = None,
    ) -> None:
        super().__init__(block_name)
        self.input_cols = [input_cols] if isinstance(input_cols, str) else input_cols
        self.output_cols = (
            [output_cols] if isinstance(output_cols, str) else output_cols
        )
        self.client = client
        self.model_id = model_id

        # For this block, we expect exactly one input column (messages) and one output column
        if len(self.input_cols) != 1:
            raise ValueError("OpenAIChatBlock expects exactly one input column")
        if len(self.output_cols) != 1:
            raise ValueError("OpenAIChatBlock expects exactly one output column")

        self.messages_column = self.input_cols[0]
        self.output_column = self.output_cols[0]

        # Store all generation parameters (only non-None values)
        self.gen_kwargs = {}
        params = {
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "max_completion_tokens": max_completion_tokens,
            "max_tokens": max_tokens,
            "n": n,
            "presence_penalty": presence_penalty,
            "response_format": response_format,
            "seed": seed,
            "stop": stop,
            "stream": stream,
            "temperature": temperature,
            "tool_choice": tool_choice,
            "tools": tools,
            "top_logprobs": top_logprobs,
            "top_p": top_p,
            "user": user,
            "extra_body": extra_body,
        }

        # Only include non-None parameters
        for key, value in params.items():
            if value is not None:
                self.gen_kwargs[key] = value

        # Log initialization with model and parameters
        logger.info(
            f"Initialized OpenAIChatBlock '{block_name}' with model '{model_id}'",
            extra={
                "block_name": block_name,
                "model_id": model_id,
                "generation_params": self.gen_kwargs,
            },
        )

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
        ),
    )
    def _create_completion_with_retry(self, **kwargs):
        """Create completion with retry logic."""
        return self.client.chat.completions.create(**kwargs)

    def generate(self, samples: Dataset, **override_kwargs: Dict[str, Any]) -> Dataset:
        """Generate the output from the block.

        Parameters set at runtime override those set during initialization.
        Supports all text-relevant OpenAI Chat Completions API parameters.

        Parameters
        ----------
        samples : Dataset
            Input dataset containing the messages column.
        **override_kwargs : Dict[str, Any]
            Runtime parameters that override initialization defaults.
            Valid parameters: frequency_penalty, logit_bias, logprobs,
            max_completion_tokens, max_tokens, n, presence_penalty,
            response_format, seed, stop, stream, temperature, tool_choice,
            tools, top_logprobs, top_p, user.

        Returns
        -------
        Dataset
            Dataset with the response added to the output column.
        """
        # Define valid parameters for validation
        valid_params = {
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "max_completion_tokens",
            "max_tokens",
            "n",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "stream",
            "temperature",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "user",
            "extra_body",
        }

        # Filter and validate override parameters
        filtered_kwargs = {
            k: v for k, v in override_kwargs.items() if k in valid_params
        }

        # Warn about invalid parameters
        invalid_params = set(override_kwargs.keys()) - valid_params
        if invalid_params:
            logger.warning(f"Ignoring invalid parameters: {invalid_params}")

        # Merge kwargs with priority: runtime > init > defaults
        final_kwargs = {**self.gen_kwargs, **filtered_kwargs}
        final_kwargs["model"] = self.model_id

        # Extract all messages
        messages_list = samples[self.messages_column]

        # Log generation start with model and effective parameters
        logger.info(
            f"Starting generation for {len(messages_list)} samples",
            extra={
                "block_name": self.block_name,
                "model_id": self.model_id,
                "batch_size": len(messages_list),
                "effective_params": {
                    k: v
                    for k, v in final_kwargs.items()
                    if k
                    in [
                        "temperature",
                        "max_tokens",
                        "max_completion_tokens",
                        "top_p",
                        "n",
                        "seed",
                    ]
                },
            },
        )

        # Get all responses
        responses = []
        for messages in messages_list:
            response = self._create_completion_with_retry(
                messages=messages, **final_kwargs
            )
            responses.append(response.choices[0].message.content)

        # Log completion
        logger.info(
            f"Generation completed successfully for {len(responses)} samples",
            extra={
                "block_name": self.block_name,
                "model_id": self.model_id,
                "batch_size": len(responses),
            },
        )

        # Add responses as new column
        return samples.add_column(self.output_column, responses)


@BlockRegistry.register("OpenAIAsyncChatBlock")
class OpenAIAsyncChatBlock(Block):
    """Async block for generating text using OpenAI Chat Completions API.

    This block takes a column containing OpenAI message format and makes
    asynchronous calls to the chat completions endpoint for better performance.

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Input column name(s). Should contain the messages list.
    output_cols : Union[str, List[str]]
        Output column name(s) for the response.
    async_client : openai.AsyncOpenAI
        Async OpenAI client instance.
    model_id : str
        Model ID to use.

    ### Text-relevant OpenAI Chat Completions API parameters ###

    frequency_penalty : Optional[float], optional
        Penalize frequent tokens (-2.0 to 2.0).
    logit_bias : Optional[Dict[str, int]], optional
        Modify likelihood of specified tokens.
    logprobs : Optional[bool], optional
        Whether to return log probabilities.
    max_completion_tokens : Optional[int], optional
        Maximum tokens in completion.
    max_tokens : Optional[int], optional
        Maximum tokens in completion (legacy).
    n : Optional[int], optional
        Number of completions to generate.
    presence_penalty : Optional[float], optional
        Penalize repeated tokens (-2.0 to 2.0).
    response_format : Optional[Dict[str, Any]], optional
        Response format specification (e.g., JSON mode).
    seed : Optional[int], optional
        Seed for deterministic outputs.
    stop : Optional[Union[str, List[str]]], optional
        Stop sequences.
    stream : Optional[bool], optional
        Whether to stream responses.
    temperature : Optional[float], optional
        Sampling temperature (0.0 to 2.0).
    tool_choice : Optional[Union[str, Dict[str, Any]]], optional
        Tool selection strategy.
    tools : Optional[List[Dict[str, Any]]], optional
        Available tools for function calling.
    top_logprobs : Optional[int], optional
        Number of top log probabilities to return.
    top_p : Optional[float], optional
        Nucleus sampling parameter (0.0 to 1.0).
    user : Optional[str], optional
        End-user identifier.
    extra_body : Optional[dict], optional
        Dictionary of additional parameters if supported by inference backend
    """

    def __init__(
        self,
        block_name: str,
        input_cols: Union[str, List[str]],
        output_cols: Union[str, List[str]],
        async_client: openai.AsyncOpenAI,
        model_id: str,
        # Text-relevant OpenAI Chat Completions API parameters
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        extra_body: Optional[dict] = None,
    ) -> None:
        super().__init__(block_name)
        self.input_cols = [input_cols] if isinstance(input_cols, str) else input_cols
        self.output_cols = (
            [output_cols] if isinstance(output_cols, str) else output_cols
        )
        self.async_client = async_client
        self.model_id = model_id

        # For this block, we expect exactly one input column (messages) and one output column
        if len(self.input_cols) != 1:
            raise ValueError("OpenAIAsyncChatBlock expects exactly one input column")
        if len(self.output_cols) != 1:
            raise ValueError("OpenAIAsyncChatBlock expects exactly one output column")

        self.messages_column = self.input_cols[0]
        self.output_column = self.output_cols[0]

        # Store all generation parameters (only non-None values)
        self.gen_kwargs = {}
        params = {
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "max_completion_tokens": max_completion_tokens,
            "max_tokens": max_tokens,
            "n": n,
            "presence_penalty": presence_penalty,
            "response_format": response_format,
            "seed": seed,
            "stop": stop,
            "stream": stream,
            "temperature": temperature,
            "tool_choice": tool_choice,
            "tools": tools,
            "top_logprobs": top_logprobs,
            "top_p": top_p,
            "user": user,
            "extra_body": extra_body,
        }

        # Only include non-None parameters
        for key, value in params.items():
            if value is not None:
                self.gen_kwargs[key] = value

        # Log initialization with model and parameters
        logger.info(
            f"Initialized OpenAIAsyncChatBlock '{block_name}' with model '{model_id}'",
            extra={
                "block_name": block_name,
                "model_id": model_id,
                "generation_params": self.gen_kwargs,
            },
        )

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
        ),
    )
    async def _generate_single(
        self, messages: List[Dict[str, Any]], **final_kwargs: Dict[str, Any]
    ) -> str:
        """Generate a single response asynchronously."""
        response = await self.async_client.chat.completions.create(
            messages=messages, **final_kwargs
        )
        return response.choices[0].message.content

    def generate(self, samples: Dataset, **override_kwargs: Dict[str, Any]) -> Dataset:
        """Generate the output from the block using async calls.

        Parameters set at runtime override those set during initialization.
        Supports all text-relevant OpenAI Chat Completions API parameters.

        Parameters
        ----------
        samples : Dataset
            Input dataset containing the messages column.
        **override_kwargs : Dict[str, Any]
            Runtime parameters that override initialization defaults.
            Valid parameters: frequency_penalty, logit_bias, logprobs,
            max_completion_tokens, max_tokens, n, presence_penalty,
            response_format, seed, stop, stream, temperature, tool_choice,
            tools, top_logprobs, top_p, user.

        Returns
        -------
        Dataset
            Dataset with the response added to the output column.
        """
        # Define valid parameters for validation
        valid_params = {
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "max_completion_tokens",
            "max_tokens",
            "n",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "stream",
            "temperature",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "user",
        }

        # Filter and validate override parameters
        filtered_kwargs = {
            k: v for k, v in override_kwargs.items() if k in valid_params
        }

        # Warn about invalid parameters
        invalid_params = set(override_kwargs.keys()) - valid_params
        if invalid_params:
            logger.warning(f"Ignoring invalid parameters: {invalid_params}")

        # Merge kwargs with priority: runtime > init > defaults
        final_kwargs = {**self.gen_kwargs, **filtered_kwargs}
        final_kwargs["model"] = self.model_id

        # Log generation start with model and effective parameters
        logger.info(
            f"Starting async generation for {len(samples)} samples",
            extra={
                "block_name": self.block_name,
                "model_id": self.model_id,
                "batch_size": len(samples),
                "effective_params": {
                    k: v
                    for k, v in final_kwargs.items()
                    if k
                    in [
                        "temperature",
                        "max_tokens",
                        "max_completion_tokens",
                        "top_p",
                        "n",
                        "seed",
                    ]
                },
            },
        )

        # Run async generation
        return asyncio.run(self._generate_async(samples, final_kwargs))

    async def _generate_async(
        self, samples: Dataset, final_kwargs: Dict[str, Any]
    ) -> Dataset:
        """Internal async method to generate all responses concurrently."""
        # Extract all messages
        messages_list = samples[self.messages_column]

        # Create all tasks
        tasks = [
            self._generate_single(messages, **final_kwargs)
            for messages in messages_list
        ]

        # Execute all tasks concurrently and collect responses
        responses = await asyncio.gather(*tasks)

        # Log completion
        logger.info(
            f"Async generation completed successfully for {len(responses)} samples",
            extra={
                "block_name": self.block_name,
                "model_id": final_kwargs["model"],
                "batch_size": len(responses),
            },
        )

        # Add responses as new column
        return samples.add_column(self.output_column, responses)
