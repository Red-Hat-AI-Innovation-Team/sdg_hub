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

    Each call creates a new LangGraph thread. The ``session_id`` is
    stored as thread metadata for traceability but does not cause
    thread reuse -- each request starts a fresh conversation.

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
        min_length=1,
        description="The assistant ID or graph name to run.",
    )
    run_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional configuration dict passed in the run payload. "
            "Merged as the 'config' key in the LangGraph /runs/wait request. "
            "Use this to pass runtime parameters to the graph via "
            "'configurable', e.g. ``{'configurable': {'model': 'gpt-4o'}}``."
        ),
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
            Session identifier. Not used in the run payload; included
            for interface compatibility with ``BaseAgentConnector``.

        Returns
        -------
        dict
            LangGraph ``/runs/wait`` request payload.

        Raises
        ------
        ConnectorError
            If messages list is empty.
        """
        if not messages:
            raise ConnectorError(
                "Cannot send empty messages list to LangGraph. "
                "Expected at least one message with role and content."
            )
        payload = {
            "assistant_id": self.assistant_id,
            "input": {"messages": messages},
        }
        if self.run_config:
            payload["config"] = self.run_config
        return payload

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
        if not response:
            raise ConnectorError(
                "LangGraph API returned an empty response. "
                "Verify the agent graph is configured correctly."
            )
        if "messages" not in response:
            logger.warning(
                "LangGraph response has no 'messages' key. "
                f"Available keys: {list(response.keys())}. "
                "Text and tool trace extraction will return None. "
                "This may indicate an API error or misconfigured graph."
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
        try:
            thread_response = await http_client.post(
                url=f"{base_url}/threads",
                payload={"metadata": {"session_id": session_id}},
                headers=headers,
            )
        except Exception as e:
            raise ConnectorError(f"LangGraph thread creation failed: {e}") from e
        thread_id = thread_response.get("thread_id")
        if not thread_id:
            raise ConnectorError(
                f"LangGraph /threads response missing 'thread_id'. "
                f"Response: {thread_response}"
            )
        logger.debug(f"Created thread {thread_id}")

        # Step 2: Run agent on the thread
        request = self.build_request(messages, session_id)
        run_url = f"{base_url}/threads/{thread_id}/runs/wait"
        logger.debug(f"Sending run request to {run_url}")
        try:
            raw_response = await http_client.post(
                url=run_url,
                payload=request,
                headers=headers,
            )
        except Exception as e:
            raise ConnectorError(
                f"LangGraph run execution failed on thread {thread_id}: {e}"
            ) from e
        logger.debug(f"Received response from {run_url}")

        return self.parse_response(raw_response)

    # ------------------------------------------------------------------
    # Response field extraction
    # ------------------------------------------------------------------

    @classmethod
    def extract_text(cls, response: dict[str, Any]) -> str | None:
        """Extract text from the last AI message in LangGraph state.

        Parameters
        ----------
        response : dict
            LangGraph final graph state.

        Returns
        -------
        str or None
            Content of the last AI message, or None if not found.
        """
        messages = response.get("messages")
        if not messages or not isinstance(messages, list):
            return None

        for msg in reversed(messages):
            if not isinstance(msg, dict):
                logger.debug(f"Skipping non-dict message in extract_text: {type(msg)}")
                continue
            role = msg.get("type") or msg.get("role", "")
            if role in ("ai", "assistant"):
                content = msg.get("content")
                if content is None:
                    logger.warning("AI message content is None, using empty string")
                    return ""
                return str(content)

        return None

    @classmethod
    def extract_session_id(cls, response: dict[str, Any]) -> str | None:
        """Extract session ID from a LangGraph response.

        LangGraph uses thread-based state; there is no top-level
        ``session_id`` in the run response.

        Parameters
        ----------
        response : dict
            LangGraph final graph state.

        Returns
        -------
        None
            Always returns None for LangGraph.
        """
        return None

    @classmethod
    def extract_tool_trace(
        cls, response: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        """Extract tool call trace from LangGraph messages.

        Collects AI messages with ``tool_calls`` and tool response
        messages into a structured trace.

        Parameters
        ----------
        response : dict
            LangGraph final graph state.

        Returns
        -------
        list[dict] or None
            List of tool-related message dicts, or None if none found.
        """
        messages = response.get("messages")
        if not messages or not isinstance(messages, list):
            return None

        tool_entries: list[dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                logger.debug(
                    f"Skipping non-dict message in extract_tool_trace: {type(msg)}"
                )
                continue
            role = msg.get("type") or msg.get("role", "")
            if role in ("ai", "assistant") and msg.get("tool_calls"):
                tool_entries.append(
                    {"type": "tool_use", "tool_calls": msg["tool_calls"]}
                )
            elif role == "tool":
                tool_result: dict[str, Any] = {
                    "type": "tool_result",
                    "name": msg.get("name", ""),
                    "content": msg.get("content", ""),
                }
                tc_id = msg.get("tool_call_id") or msg.get("id")
                if tc_id:
                    tool_result["tool_call_id"] = tc_id
                tool_entries.append(tool_result)

        return tool_entries if tool_entries else None
