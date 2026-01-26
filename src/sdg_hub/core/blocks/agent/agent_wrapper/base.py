# SPDX-License-Identifier: Apache-2.0
"""Base agent wrapper for external agent framework integration."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ....utils.logger_config import setup_logger

logger = setup_logger(__name__)


class BaseAgentWrapper(ABC):
    """Abstract base class for agent wrappers.

    This class defines the interface that all agent wrappers must implement
    to integrate with different agent frameworks (Langflow, LangGraph, OpenAI Agent SDK, etc.).

    Agent wrappers serve as adapters that normalize different agent framework APIs
    into a consistent interface that AgentBlock can use.

    Attributes
    ----------
    agent_framework : str
        The name of the agent framework (e.g., "langflow", "langgraph").
    agent_url : str
        The API endpoint URL for the agent service.
    agent_api_key : Optional[str]
        The API key for authenticating with the agent service.
    timeout : float
        Request timeout in seconds.
    """

    def __init__(
        self,
        agent_framework: str,
        agent_url: str,
        agent_api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        """Initialize the agent wrapper.

        Parameters
        ----------
        agent_framework : str
            The name of the agent framework.
        agent_url : str
            The API endpoint URL for the agent service.
        agent_api_key : Optional[str], optional
            The API key for authentication, by default None.
        timeout : float, optional
            Request timeout in seconds, by default 120.0.
        """
        self.agent_framework = agent_framework
        self.agent_url = agent_url
        self.agent_api_key = agent_api_key
        self.timeout = timeout

        logger.debug(
            "Initialized %s wrapper with url=%s, timeout=%.1f",
            agent_framework,
            agent_url,
            timeout,
            extra={
                "agent_framework": agent_framework,
                "agent_url": agent_url,
                "timeout": timeout,
            },
        )

    @abstractmethod
    def generate(
        self, messages: list[dict[str, Any]], session_id: str
    ) -> dict[str, Any]:
        """Generate a response from the agent.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            The messages to send to the agent. Format depends on the framework.
        session_id : str
            The session ID to use for conversation continuity.

        Returns
        -------
        dict[str, Any]
            The validated and normalized response from the agent.

        Raises
        ------
        NotImplementedError
            If the subclass doesn't implement this method.
        """

    @abstractmethod
    async def agenerate(
        self, messages: list[dict[str, Any]], session_id: str
    ) -> dict[str, Any]:
        """Generate a response from the agent asynchronously.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            The messages to send to the agent. Format depends on the framework.
        session_id : str
            The session ID to use for conversation continuity.

        Returns
        -------
        dict[str, Any]
            The validated and normalized response from the agent.

        Raises
        ------
        NotImplementedError
            If the subclass doesn't implement this method.
        """

    @abstractmethod
    def validate_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize the agent response.

        This method should check that the response has the expected structure
        and extract/normalize the data as needed for the specific framework.

        Parameters
        ----------
        response : dict[str, Any]
            The raw response from the agent API.

        Returns
        -------
        dict[str, Any]
            The validated and normalized response.

        Raises
        ------
        ValueError
            If the response is invalid or missing required fields.
        NotImplementedError
            If the subclass doesn't implement this method.
        """
