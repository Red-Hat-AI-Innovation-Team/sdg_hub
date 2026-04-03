# SPDX-License-Identifier: Apache-2.0
"""LangGraph agent framework connector."""

from typing import Any

from pydantic import Field

from ...utils.logger_config import setup_logger
from ..exceptions import ConnectorError
from ..registry import ConnectorRegistry
from .base import BaseAgentConnector

logger = setup_logger(__name__)


@ConnectorRegistry.register("langgraph")
class LangGraphConnector(BaseAgentConnector):
    """Connector for LangGraph agent framework.

    LangGraph is a framework for building stateful, multi-actor applications
    with LLMs. This connector communicates with any HTTP endpoint that
    implements the LangGraph Platform API (thread and run management).
    Common deployment options include ``langgraph dev`` for local
    development, the LangGraph Platform for managed hosting, or
    self-hosted setups behind FastAPI / Docker on any cloud provider.

    The connector uses thread-based runs:

    1. Creates a thread via ``POST {base_url}/threads``
    2. Runs the agent via ``POST {base_url}/threads/{thread_id}/runs/wait``

    The ``session_id`` from :class:`AgentBlock` maps to the LangGraph
    ``thread_id``, so rows sharing a session share conversation history.

    Parameters
    ----------
    assistant_id : str
        The assistant ID or graph name to run. Defaults to ``"agent"``,
        which is the standard default for LangGraph deployments.

    Example
    -------
    >>> from sdg_hub.core.connectors import ConnectorConfig, LangGraphConnector
    >>>
    >>> config = ConnectorConfig(
    ...     url="http://localhost:2024",
    ...     api_key="your-api-key",
    ... )
    >>> connector = LangGraphConnector(config=config)
    >>> response = connector.send(
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     session_id="session-123",
    ... )
    """

    assistant_id: str = Field(
        default="agent",
        description="The assistant ID or graph name to run.",
    )

    def _build_headers(self) -> dict[str, str]:
        """Build headers for LangGraph API.

        LangGraph / LangSmith deployments use ``x-api-key`` for authentication.

        Returns
        -------
        dict[str, str]
            HTTP headers.
        """
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        return headers

    def build_request(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> dict[str, Any]:
        """Build LangGraph run request payload.

        Formats messages into the LangGraph input structure with the
        configured ``assistant_id``.

        Parameters
        ----------
        messages : list[dict]
            Messages in standard format.
        session_id : str
            Session identifier (used as thread_id).

        Returns
        -------
        dict
            LangGraph ``/runs/wait`` request payload.
        """
        return {
            "assistant_id": self.assistant_id,
            "input": {"messages": messages},
        }

    def parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Parse LangGraph response.

        LangGraph returns the final graph state as a dict. For chat agents
        this typically contains a ``messages`` list with the full
        conversation history.

        Parameters
        ----------
        response : dict
            Raw response from LangGraph API (final graph state).

        Returns
        -------
        dict
            Validated response dict.

        Raises
        ------
        ConnectorError
            If response is not a valid dict.
        """
        if not isinstance(response, dict):
            raise ConnectorError(
                f"Expected dict response, got {type(response).__name__}"
            )

        return response

    async def _send_async(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> dict[str, Any]:
        """Send request to LangGraph API using thread-based runs.

        Creates a thread and then executes a run on it. The ``session_id``
        is stored as thread metadata for traceability.

        Parameters
        ----------
        messages : list[dict]
            Messages to send to the agent.
        session_id : str
            Session identifier, stored as thread metadata.

        Returns
        -------
        dict
            Parsed response from the agent (final graph state).
        """
        if not self.config.url:
            raise ConnectorError("No URL configured for connector")

        http_client = self._get_http_client()
        headers = self._build_headers()
        base_url = self.config.url.rstrip("/")

        # Step 1: Create a thread
        logger.debug(f"Creating thread at {base_url}/threads")
        thread_response = await http_client.post(
            url=f"{base_url}/threads",
            payload={"metadata": {"session_id": session_id}},
            headers=headers,
        )
        thread_id = thread_response["thread_id"]
        logger.debug(f"Created thread {thread_id}")

        # Step 2: Run agent on the thread
        request = self.build_request(messages, session_id)
        run_url = f"{base_url}/threads/{thread_id}/runs/wait"
        logger.debug(f"Sending run request to {run_url}")
        raw_response = await http_client.post(
            url=run_url,
            payload=request,
            headers=headers,
        )
        logger.debug(f"Received response from {run_url}")

        return self.parse_response(raw_response)
