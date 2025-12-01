# SPDX-License-Identifier: Apache-2.0
"""Agent block with LLM and tools support."""

# Standard
from typing import Any
import uuid

from pydantic import ConfigDict, Field, field_validator

# Third Party
import pandas as pd

from ...utils.error_handling import BlockValidationError
from ...utils.logger_config import setup_logger

# Local
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "AgentBlock",
    "agent",
    "Agent block with LLM and tools support",
)
class AgentBlock(BaseBlock):
    model_config = ConfigDict(extra="allow")

    """Agent block with LLM and tools support.

    This block provides a minimal wrapper around LLM and tools support.

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Input column name(s). Should contain the messages list.
    output_cols : Union[dict, List[dict]]
        Output column name(s) for the response.
    agent_framework : str
        Agent framework. langflow, openai-agent-sdk, google-adk or custom.
    agent_url: str
        API endpoint to access the agent.
    agent_api_key: str
        API key to access the agent.
    async_mode: bool
        Whether to use async processing, by default False.
    timeout: float
        Timeout for the agent, by default 120.0 seconds.
    **kwargs : Any
        Any additional parameters to pass to the agent through the wrapper.
    
    Examples
    --------
    >>> # Langflow agent wrapper
    >>> block = AgentBlock(
    ...     block_name="langflow_agent",
    ...     input_cols="messages",
    ...     output_cols="response",
    ...     agent_framework="langflow",
    ...     agent_url="http://localhost:7860/api/v1/run/df3584bc-77c1-415f-b16b-58349ba738c6",
    ...     agent_api_key="your-api-key"
    ... )
    """

    # Essential operational fields (excluded from YAML serialization)
    agent_framework: str = Field(
        None, exclude=True, description="Agent framework"
    )
    agent_url: str = Field(
        None, exclude=True, description="API endpoint to access the agent"
    )
    agent_api_key: str = Field(
        None, exclude=True, description="API key to access the agent"
    )
    async_mode: bool = Field(
        False, exclude=True, description="Whether to use async processing"
    )
    timeout: float = Field(
        120.0, exclude=True, description="Timeout for the agent"
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

    def model_post_init(self, __context) -> None:
        """Initialize after Pydantic validation."""
        super().model_post_init(__context)

        # Log initialization only when agent framework is configured
        if self.agent_framework:
            logger.info(
                "Initialized AgentBlock '%s' with agent framework '%s'",
                self.block_name,
                self.agent_framework,
                extra={
                    "block_name": self.block_name,
                    "agent_framework": self.agent_framework,
                    "agent_url": self.agent_url,
                },
            )

    def _message_to_dict(self, message) -> dict[str, Any]:
        """Convert message to dict."""
        # If already a dict, return as-is
        if isinstance(message, dict):
            return message
        # Otherwise, convert from object
        return {"content": message.content, **getattr(message, "__dict__", {})}

    def _messages_to_dict(self, messages: list | dict) -> list[dict[str, Any]]:
        """Convert messages to dict."""
        # If messages is already a list of dicts, return as-is
        if isinstance(messages, list):
            return [self._message_to_dict(msg) for msg in messages]
        # If it's a single dict, wrap in list
        if isinstance(messages, dict):
            return [messages]
        # Otherwise convert to list
        return [self._message_to_dict(messages)]

    def _initialize_agent_wrapper(self):
        """Initialize the appropriate agent wrapper based on framework."""
        if self.agent_framework == "langflow":
            from .agent_wrapper.langflow_agent_wrapper import LangflowAgentWrapper
            return LangflowAgentWrapper(
                agent_framework=self.agent_framework,
                agent_url=self.agent_url,
                agent_api_key=self.agent_api_key
            )
        else:
            raise BlockValidationError(
                f"Unsupported agent framework: {self.agent_framework}. "
                f"Supported frameworks: langflow"
            )

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate responses from the agent using the wrapper.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset containing the input column with messages.
        **kwargs : Any
            Runtime parameters that override initialization defaults.

        Returns
        -------
        pd.DataFrame
            Dataset with responses added to the output column.
        """
        # Initialize agent wrapper
        agent_wrapper = self._initialize_agent_wrapper()

        # Extract messages from pandas DataFrame
        messages_list = samples[self.input_cols[0]].tolist()

        # Log generation start
        logger.info(
            "Starting agent generation for %d samples using %s framework",
            len(messages_list),
            self.agent_framework,
            extra={
                "block_name": self.block_name,
                "agent_framework": self.agent_framework,
                "batch_size": len(messages_list),
                "agent_url": self.agent_url,
            },
        )

        # Generate responses
        responses = []
        for idx, message in enumerate(messages_list):
            session_id = str(uuid.uuid4())

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
                response = agent_wrapper.generate(
                    self._messages_to_dict(message),
                    session_id
                )
                responses.append(response)
            except Exception as e:
                logger.error(
                    "Failed to generate response for sample %d: %s",
                    idx + 1,
                    str(e),
                    extra={
                        "block_name": self.block_name,
                        "sample_idx": idx,
                        "error": str(e),
                    },
                )
                raise

        # Log completion
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