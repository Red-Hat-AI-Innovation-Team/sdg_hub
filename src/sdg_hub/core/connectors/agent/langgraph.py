# SPDX-License-Identifier: Apache-2.0
"""LangGraph agent framework connector."""

from typing import Any
import json
import uuid

from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentRequest,
    ChatAgentResponse,
)
from mlflow.types.chat import Function, ToolCall
from pydantic import Field

from ...utils.logger_config import setup_logger
from ..exceptions import ConnectorError
from ..registry import ConnectorRegistry
from .base import BaseAgentConnector

logger = setup_logger(__name__)

# Default LangGraph API endpoint URL for local development.
DEFAULT_LANGGRAPH_URL = "http://localhost:2024"

# Mapping from LangGraph message types to standard roles.
_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "tool": "tool",
    "system": "system",
}


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
    ...     url=DEFAULT_LANGGRAPH_URL,
    ...     api_key="your-api-key",
    ... )
    >>> connector = LangGraphConnector(config=config)
    >>> request = ChatAgentRequest(
    ...     messages=[ChatAgentMessage(role="user", content="Hello!")],
    ...     context=ChatContext(conversation_id="session-123"),
    ... )
    >>> response = connector.send(request)
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
        request: ChatAgentRequest,
    ) -> dict[str, Any]:
        """Build LangGraph run request payload.

        Formats messages into the LangGraph input structure with the
        configured ``assistant_id``.

        Parameters
        ----------
        request : ChatAgentRequest
            Standardized agent request.

        Returns
        -------
        dict
            LangGraph ``/runs/wait`` request payload.

        Raises
        ------
        ConnectorError
            If messages list is empty.
        """
        if not request.messages:
            raise ConnectorError(
                "Cannot send empty messages list to LangGraph. "
                "Expected at least one message with role and content."
            )

        # Convert ChatAgentMessage objects to dicts for LangGraph wire format
        messages_as_dicts = []
        for msg in request.messages:
            d: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            if msg.name:
                d["name"] = msg.name
            messages_as_dicts.append(d)

        payload: dict[str, Any] = {
            "assistant_id": self.assistant_id,
            "input": {"messages": messages_as_dicts},
        }
        if self.run_config:
            payload["config"] = self.run_config
        return payload

    def parse_response(self, response: dict[str, Any]) -> ChatAgentResponse:
        """Parse LangGraph response into ChatAgentResponse.

        LangGraph returns the final graph state as a dict. For chat agents
        this typically contains a ``messages`` list with the full
        conversation history.

        Parameters
        ----------
        response : dict
            Raw response from LangGraph API (final graph state).

        Returns
        -------
        ChatAgentResponse
            Standardized agent response.

        Raises
        ------
        ConnectorError
            If response is not a valid dict, is empty, or has no messages.
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

        raw_messages = response.get("messages", [])
        if not raw_messages:
            logger.warning(
                "LangGraph response has no 'messages' key or empty messages. "
                f"Available keys: {list(response.keys())}. "
                "This may indicate an API error or misconfigured graph."
            )

        messages: list[ChatAgentMessage] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                logger.debug(
                    f"Skipping non-dict message in parse_response: {type(msg)}"
                )
                continue

            # Map LangGraph types to standard roles
            raw_role: str = msg.get("type") or msg.get("role") or "user"
            role: str = _ROLE_MAP.get(raw_role, raw_role)

            if role in ("assistant",) and msg.get("tool_calls"):
                # Assistant message with tool calls
                tool_calls = []
                for tc in msg["tool_calls"]:
                    # LangGraph uses "args" for tool arguments
                    args = tc.get("args", tc.get("arguments", {}))
                    if isinstance(args, dict):
                        args_str = json.dumps(args)
                    else:
                        args_str = str(args) if args else "{}"
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", str(uuid.uuid4())),
                            type="function",
                            function=Function(
                                name=tc.get("name", ""),
                                arguments=args_str,
                            ),
                        )
                    )
                messages.append(
                    ChatAgentMessage(
                        role="assistant",
                        content=msg.get("content", "") or "",
                        tool_calls=tool_calls,
                        id=str(uuid.uuid4()),
                    )
                )
            elif role == "tool":
                # Tool result message
                messages.append(
                    ChatAgentMessage(
                        role="tool",
                        name=msg.get("name", ""),
                        content=str(msg.get("content", "")),
                        id=str(uuid.uuid4()),
                        tool_call_id=msg.get("tool_call_id") or str(uuid.uuid4()),
                    )
                )
            else:
                # Human/user/assistant/system messages
                messages.append(
                    ChatAgentMessage(
                        role=role,
                        content=str(msg.get("content", "")),
                        id=str(uuid.uuid4()),
                    )
                )

        if not messages:
            raise ConnectorError(
                "Could not parse any messages from LangGraph response. "
                f"Available keys: {list(response.keys())}"
            )

        return ChatAgentResponse(messages=messages)

    async def _send_async(
        self,
        request: ChatAgentRequest,
    ) -> ChatAgentResponse:
        """Send request to LangGraph API using thread-based runs.

        Creates a thread and then executes a run on it. The ``session_id``
        is stored as thread metadata for traceability.

        Parameters
        ----------
        request : ChatAgentRequest
            Standardized agent request.

        Returns
        -------
        ChatAgentResponse
            Parsed response from the agent (final graph state).
        """
        if not self.config.url:
            raise ConnectorError("No URL configured for connector")

        http_client = self._get_http_client()
        headers = self._build_headers()
        base_url = self.config.url.rstrip("/")

        session_id = (
            request.context.conversation_id if request.context else "default"
        ) or "default"

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
        payload = self.build_request(request)
        run_url = f"{base_url}/threads/{thread_id}/runs/wait"
        logger.debug(f"Sending run request to {run_url}")
        try:
            raw_response = await http_client.post(
                url=run_url,
                payload=payload,
                headers=headers,
            )
        except Exception as e:
            raise ConnectorError(
                f"LangGraph run execution failed on thread {thread_id}: {e}"
            ) from e
        logger.debug(f"Received response from {run_url}")

        return self.parse_response(raw_response)
