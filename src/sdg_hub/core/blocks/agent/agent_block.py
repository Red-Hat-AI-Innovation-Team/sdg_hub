# SPDX-License-Identifier: Apache-2.0
"""Agent block for external agent framework integration."""

from typing import Any, Optional
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

    # Required operational fields (excluded from YAML serialization)
    agent_framework: str = Field(
        ..., exclude=True, description="Agent framework (e.g., 'langflow')"
    )
    agent_url: str = Field(..., exclude=True, description="Agent API endpoint URL")
    agent_api_key: Optional[str] = Field(
        None, exclude=True, description="API key for agent authentication"
    )
    timeout: float = Field(
        120.0, exclude=True, description="Request timeout in seconds"
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
        """Validate that agent framework is provided and supported."""
        if not v:
            raise ValueError("agent_framework is required")

        supported = ["langflow"]
        if v not in supported:
            raise ValueError(
                f"Unsupported agent framework: {v}. Supported: {', '.join(supported)}"
            )
        return v

    @field_validator("agent_url")
    @classmethod
    def validate_agent_url(cls, v):
        """Validate that agent URL is provided."""
        if not v:
            raise ValueError("agent_url is required")
        return v

    def model_post_init(self, __context) -> None:
        """Initialize after Pydantic validation."""
        super().model_post_init(__context)

        # Initialize the agent wrapper once
        self._agent_wrapper = self._create_agent_wrapper()

        logger.info(
            "Initialized AgentBlock '%s' with framework '%s', timeout=%.1fs",
            self.block_name,
            self.agent_framework,
            self.timeout,
            extra={
                "block_name": self.block_name,
                "agent_framework": self.agent_framework,
                "agent_url": self.agent_url,
                "timeout": self.timeout,
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
            # This should never happen due to field validation, but keeping for safety
            raise BlockValidationError(
                f"Unsupported agent framework: {self.agent_framework}"
            )

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
            Runtime parameters (currently unused).

        Returns
        -------
        pd.DataFrame
            Dataset with responses added to the output column.

        Raises
        ------
        BlockValidationError
            If agent wrapper is not initialized or generation fails.
        """
        if self._agent_wrapper is None:
            raise BlockValidationError(
                f"Agent wrapper not initialized for block '{self.block_name}'"
            )

        # Extract messages from DataFrame
        messages_list = samples[self.input_cols[0]].tolist()

        logger.info(
            "Starting agent generation for %d samples using %s framework",
            len(messages_list),
            self.agent_framework,
            extra={
                "block_name": self.block_name,
                "agent_framework": self.agent_framework,
                "batch_size": len(messages_list),
            },
        )

        # Generate responses
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
                response = self._agent_wrapper.generate(
                    self._messages_to_dict(message), session_id
                )
                responses.append(response)

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
