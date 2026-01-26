# SPDX-License-Identifier: Apache-2.0
"""Agent block for external agent framework integration."""

from typing import Any, Optional
import asyncio
import uuid

from pydantic import ConfigDict, Field, field_validator
import pandas as pd

from ...utils.error_handling import BlockValidationError
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "AgentBlock",
    "agent",
    "Agent block with LLM and tools support via external frameworks",
)
class AgentBlock(BaseBlock):
    """Agent block for external agent framework integration.

    This block integrates with external agent frameworks (Langflow, LangGraph, etc.)
    to provide LLM agents with tool/function calling capabilities. It uses framework-specific
    wrappers to normalize different agent APIs into a consistent interface.

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Input column name(s). Should contain the messages list.
    output_cols : Union[dict, List[dict]]
        Output column name(s) for the response.
    agent_framework : str
        Agent framework to use. Currently supported: "langflow".
    agent_url : str
        API endpoint to access the agent.
    agent_api_key : Optional[str], optional
        API key to access the agent, by default None.
    timeout : float, optional
        Request timeout in seconds, by default 120.0.

    Examples
    --------
    >>> # Langflow agent
    >>> block = AgentBlock(
    ...     block_name="langflow_agent",
    ...     input_cols="messages",
    ...     output_cols="response",
    ...     agent_framework="langflow",
    ...     agent_url="http://localhost:7860/api/v1/run/flow-id",
    ...     agent_api_key="your-api-key",
    ...     timeout=60.0
    ... )
    """

    model_config = ConfigDict(extra="allow")

    # Operational fields (excluded from YAML serialization)
    agent_framework: Optional[str] = Field(
        None, exclude=True, description="Agent framework (e.g., 'langflow')"
    )
    agent_url: Optional[str] = Field(
        None, exclude=True, description="Agent API endpoint URL"
    )
    agent_api_key: Optional[str] = Field(
        None, exclude=True, description="API key for agent authentication"
    )
    timeout: float = Field(
        120.0, exclude=True, description="Request timeout in seconds"
    )
    async_mode: bool = Field(
        False, exclude=True, description="Whether to use async processing for agent requests"
    )

    # Private: agent wrapper instance (initialized once)
    _agent_wrapper: Optional[Any] = None

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
                f"AgentBlock expects exactly one input column, got {len(v)}: {v}"
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
                f"AgentBlock expects exactly one output column, got {len(v)}: {v}"
            )
        raise ValueError(f"Invalid output_cols format: {v}")

    @field_validator("agent_framework")
    @classmethod
    def validate_agent_framework(cls, v):
        """Validate agent framework is supported if provided."""
        if v is None:
            return v

        supported = ["langflow"]
        if v not in supported:
            raise ValueError(
                f"Unsupported agent framework: {v}. Supported: {', '.join(supported)}"
            )
        return v

    def model_post_init(self, __context) -> None:
        """Initialize after Pydantic validation."""
        super().model_post_init(__context)

        # Initialize wrapper if required fields are set
        if self.agent_framework and self.agent_url:
            self._agent_wrapper = self._create_agent_wrapper()

            logger.info(
                "Initialized AgentBlock '%s' with framework '%s', timeout=%.1fs, async_mode=%s",
                self.block_name,
                self.agent_framework,
                self.timeout,
                self.async_mode,
                extra={
                    "block_name": self.block_name,
                    "agent_framework": self.agent_framework,
                    "agent_url": self.agent_url,
                    "timeout": self.timeout,
                    "async_mode": self.async_mode,
                },
            )

    def _create_agent_wrapper(self):
        """Create the appropriate agent wrapper based on framework.

        Returns
        -------
        BaseAgentWrapper
            The initialized agent wrapper.

        Raises
        ------
        BlockValidationError
            If the framework is not supported.
        """
        if self.agent_framework == "langflow":
            from .agent_wrapper.langflow_agent_wrapper import LangflowAgentWrapper

            return LangflowAgentWrapper(
                agent_framework=self.agent_framework,
                agent_url=self.agent_url,
                agent_api_key=self.agent_api_key,
                timeout=self.timeout,
            )
        else:
            raise BlockValidationError(
                f"Unsupported agent framework: {self.agent_framework}"
            )

    def _get_current_wrapper(self):
        """Get wrapper with current field values, recreating if needed."""
        if not self.agent_framework or not self.agent_url:
            raise BlockValidationError(
                f"agent_framework and agent_url are required for block '{self.block_name}'"
            )

        if self._agent_wrapper is not None:
            w = self._agent_wrapper
            if (
                w.agent_url == self.agent_url
                and w.agent_api_key == self.agent_api_key
                and w.timeout == self.timeout
            ):
                return w

        return self._create_agent_wrapper()

    def _message_to_dict(self, message: Any) -> dict[str, Any]:
        """Convert message to dict.

        Parameters
        ----------
        message : Any
            Message to convert (dict or object).

        Returns
        -------
        dict[str, Any]
            Message as dictionary.
        """
        if isinstance(message, dict):
            return message
        # Convert object to dict (e.g., if using message objects)
        return {"content": message.content, **getattr(message, "__dict__", {})}

    def _messages_to_dict(self, messages: Any) -> list[dict[str, Any]]:
        """Convert messages to list of dicts.

        Parameters
        ----------
        messages : Any
            Messages in various formats (list, dict, or object).

        Returns
        -------
        list[dict[str, Any]]
            Messages as list of dictionaries.
        """
        if isinstance(messages, list):
            return [self._message_to_dict(msg) for msg in messages]
        if isinstance(messages, dict):
            return [messages]
        return [self._message_to_dict(messages)]

    def _generate_session_id(self) -> str:
        """Generate a unique session ID for each sample.

        Returns
        -------
        str
            A unique UUID string for the session.
        """
        return str(uuid.uuid4())

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate responses from the agent using the wrapper.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset containing the input column with messages.
        **kwargs : Any
            Runtime parameters including _flow_max_concurrency for async mode.

        Returns
        -------
        pd.DataFrame
            Dataset with responses added to the output column.

        Raises
        ------
        BlockValidationError
            If agent wrapper is not initialized or generation fails.
        """
        # Get wrapper with current values (supports runtime overrides)
        wrapper = self._get_current_wrapper()

        # Extract flow-specific parameters
        flow_max_concurrency = kwargs.pop("_flow_max_concurrency", None)

        # Extract messages from DataFrame
        messages_list = samples[self.input_cols[0]].tolist()

        logger.info(
            "Starting %s agent generation for %d samples using %s framework%s",
            "async" if self.async_mode else "sync",
            len(messages_list),
            self.agent_framework,
            (
                f" (max_concurrency={flow_max_concurrency})"
                if flow_max_concurrency
                else ""
            ),
            extra={
                "block_name": self.block_name,
                "agent_framework": self.agent_framework,
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
                            wrapper, messages_list, flow_max_concurrency
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
                    self._generate_async(wrapper, messages_list, flow_max_concurrency)
                )
        else:
            responses = self._generate_sync(wrapper, messages_list)

        logger.info(
            "Agent generation completed successfully for %d samples",
            len(responses),
            extra={
                "block_name": self.block_name,
                "agent_framework": self.agent_framework,
                "batch_size": len(responses),
            },
        )

        # Add responses as new column
        result = samples.copy()
        result[self.output_cols[0]] = responses
        return result

    def _generate_sync(
        self,
        wrapper: Any,
        messages_list: list[Any],
    ) -> list[dict[str, Any]]:
        """Generate responses synchronously.

        Parameters
        ----------
        wrapper : BaseAgentWrapper
            The agent wrapper instance.
        messages_list : list[Any]
            List of messages to process.

        Returns
        -------
        list[dict[str, Any]]
            List of response dictionaries.

        Raises
        ------
        BlockValidationError
            If generation fails for any sample.
        """
        responses = []

        for idx, message in enumerate(messages_list):
            session_id = self._generate_session_id()

            logger.debug(
                "Processing sample %d/%d with session_id=%s",
                idx + 1,
                len(messages_list),
                session_id,
                extra={
                    "block_name": self.block_name,
                    "sample_idx": idx,
                    "session_id": session_id,
                },
            )

            try:
                response = wrapper.generate(
                    self._messages_to_dict(message), session_id
                )
                responses.append(response)

                # Log progress for large batches
                if (idx + 1) % 10 == 0:
                    logger.debug(
                        "Generated %d/%d responses",
                        idx + 1,
                        len(messages_list),
                        extra={
                            "block_name": self.block_name,
                            "progress": f"{idx + 1}/{len(messages_list)}",
                        },
                    )

            except BlockValidationError:
                # Re-raise BlockValidationError as-is (already logged in wrapper)
                raise
            except Exception as e:
                # Unexpected errors
                error_msg = f"Unexpected error for sample {idx + 1}: {str(e)}"
                logger.error(
                    error_msg,
                    extra={
                        "block_name": self.block_name,
                        "sample_idx": idx,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                raise BlockValidationError(error_msg) from e

        return responses

    async def _make_async_completion(
        self,
        wrapper: Any,
        message: Any,
        session_id: str,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> dict[str, Any]:
        """Make a single async agent completion with optional concurrency control.

        Parameters
        ----------
        wrapper : BaseAgentWrapper
            The agent wrapper instance.
        message : Any
            Message for this completion.
        session_id : str
            Session ID for this request.
        semaphore : Optional[asyncio.Semaphore], optional
            Semaphore for concurrency control.

        Returns
        -------
        dict[str, Any]
            Response dictionary.

        Raises
        ------
        BlockValidationError
            If generation fails.
        """
        if semaphore:
            async with semaphore:
                return await wrapper.agenerate(
                    self._messages_to_dict(message), session_id
                )
        else:
            return await wrapper.agenerate(
                self._messages_to_dict(message), session_id
            )

    async def _generate_async(
        self,
        wrapper: Any,
        messages_list: list[Any],
        flow_max_concurrency: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Generate responses asynchronously.

        Parameters
        ----------
        wrapper : BaseAgentWrapper
            The agent wrapper instance.
        messages_list : list[Any]
            List of messages to process.
        flow_max_concurrency : Optional[int], optional
            Maximum concurrency for async requests.

        Returns
        -------
        list[dict[str, Any]]
            List of response dictionaries.

        Raises
        ------
        BlockValidationError
            If generation fails.
        """
        try:
            # Generate session IDs for all messages
            session_ids = [self._generate_session_id() for _ in messages_list]

            if flow_max_concurrency is not None:
                # Validate max_concurrency parameter
                if flow_max_concurrency < 1:
                    raise ValueError(
                        f"max_concurrency must be greater than 0, got {flow_max_concurrency}"
                    )

                logger.debug(
                    "Using semaphore for concurrency control with max_concurrency=%d",
                    flow_max_concurrency,
                    extra={
                        "block_name": self.block_name,
                        "max_concurrency": flow_max_concurrency,
                    },
                )

                # Use semaphore for concurrency control
                semaphore = asyncio.Semaphore(flow_max_concurrency)
                tasks = [
                    self._make_async_completion(wrapper, message, session_id, semaphore)
                    for message, session_id in zip(messages_list, session_ids)
                ]
            else:
                # No concurrency limit
                tasks = [
                    self._make_async_completion(wrapper, message, session_id)
                    for message, session_id in zip(messages_list, session_ids)
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
                    "error_type": type(e).__name__,
                },
            )
            raise
