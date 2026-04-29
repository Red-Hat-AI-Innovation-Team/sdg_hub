# SPDX-License-Identifier: Apache-2.0
"""Base class for agent framework connectors."""

from abc import abstractmethod
from typing import Any, Optional
import asyncio
import uuid

from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentRequest,
    ChatAgentResponse,
    ChatContext,
)
from mlflow.types.chat import Function, ToolCall
from pydantic import PrivateAttr

from ...utils.logger_config import setup_logger
from ..base import BaseConnector
from ..exceptions import ConnectorError
from ..http import HttpClient

logger = setup_logger(__name__)


class BaseAgentConnector(BaseConnector):
    """Base class for agent framework connectors.

    This class provides a common interface for communicating with
    agent frameworks (Langflow, LangGraph, etc.). It uses an async-first
    pattern where the core logic is implemented once in async, and sync
    is derived automatically.

    Subclasses must implement:
    - build_request: Convert a ChatAgentRequest to framework-specific format
    - parse_response: Convert framework response to ChatAgentResponse

    Example:
        ```python
        class MyAgentConnector(BaseAgentConnector):
            def build_request(self, request):
                return {"input": request.messages[-1].content}

            def parse_response(self, response):
                return ChatAgentResponse(
                    messages=[ChatAgentMessage(
                        role="assistant",
                        content=response["result"],
                        id=str(uuid.uuid4()),
                    )]
                )

        connector = MyAgentConnector(config=ConnectorConfig(url="http://api"))
        request = ChatAgentRequest(
            messages=[ChatAgentMessage(role="user", content="Hello")]
        )
        response = connector.send(request)
        ```
    """

    _http_client: Optional[HttpClient] = PrivateAttr(default=None)

    def _get_http_client(self) -> HttpClient:
        """Get or create the HTTP client."""
        if self._http_client is None:
            self._http_client = HttpClient(
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._http_client

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for requests.

        Override in subclasses for framework-specific headers.

        Returns
        -------
        dict[str, str]
            HTTP headers to include in requests.
        """
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    @abstractmethod
    def build_request(
        self,
        request: ChatAgentRequest,
    ) -> dict[str, Any]:
        """Build framework-specific request payload.

        Parameters
        ----------
        request : ChatAgentRequest
            Standardized agent request with messages and context.

        Returns
        -------
        dict
            Framework-specific request payload.
        """
        pass

    @abstractmethod
    def parse_response(self, response: dict[str, Any]) -> ChatAgentResponse:
        """Parse and validate framework response.

        Parameters
        ----------
        response : dict
            Raw response from the framework.

        Returns
        -------
        ChatAgentResponse
            Standardized agent response.

        Raises
        ------
        ConnectorError
            If the response is invalid or cannot be parsed.
        """
        pass

    async def _send_async(
        self,
        request: ChatAgentRequest,
    ) -> ChatAgentResponse:
        """Core async implementation.

        Parameters
        ----------
        request : ChatAgentRequest
            Standardized agent request.

        Returns
        -------
        ChatAgentResponse
            Parsed response from the agent.
        """
        if not self.config.url:
            raise ConnectorError("No URL configured for connector")

        http_client = self._get_http_client()
        payload = self.build_request(request)
        headers = self._build_headers()

        logger.debug(f"Sending request to {self.config.url}")
        raw_response = await http_client.post(
            url=self.config.url,
            payload=payload,
            headers=headers,
        )
        logger.debug(f"Received response from {self.config.url}")

        return self.parse_response(raw_response)

    def send(
        self,
        request: ChatAgentRequest,
        async_mode: bool = False,
    ):
        """Send a request to the agent.

        Parameters
        ----------
        request : ChatAgentRequest
            Standardized agent request with messages and context.
        async_mode : bool, optional
            If True, returns a coroutine. If False (default), runs synchronously.

        Returns
        -------
        ChatAgentResponse or Coroutine[ChatAgentResponse]
            Response, or coroutine if async_mode=True.
        """
        if async_mode:
            return self._send_async(request)

        # Sync mode: run async code in event loop
        try:
            asyncio.get_running_loop()
            # Already in async context - use thread executor
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._send_async(request),
                )
                return future.result()
        except RuntimeError:
            # No event loop - create one
            return asyncio.run(self._send_async(request))

    async def asend(
        self,
        request: ChatAgentRequest,
    ) -> ChatAgentResponse:
        """Async send - convenience wrapper.

        Parameters
        ----------
        request : ChatAgentRequest
            Standardized agent request.

        Returns
        -------
        ChatAgentResponse
            Response from the agent.
        """
        return await self._send_async(request)

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute a request (BaseConnector interface).

        Parameters
        ----------
        request : dict
            Request containing 'messages' and optionally 'session_id' keys.

        Returns
        -------
        dict
            Response from the agent as a dict.
        """
        messages = request["messages"]
        session_id = request.get("session_id") or str(uuid.uuid4())

        # Convert dict messages to ChatAgentMessage objects
        agent_messages = []
        for msg in messages:
            if isinstance(msg, ChatAgentMessage):
                if not msg.id:
                    msg = msg.model_copy(update={"id": str(uuid.uuid4())})
                agent_messages.append(msg)
            else:
                kwargs: dict[str, Any] = {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                    "id": str(uuid.uuid4()),
                }
                if msg.get("name"):
                    kwargs["name"] = msg["name"]
                if msg.get("tool_call_id"):
                    kwargs["tool_call_id"] = msg["tool_call_id"]
                if msg.get("tool_calls"):
                    kwargs["tool_calls"] = [
                        ToolCall(
                            id=tc.get("id", str(uuid.uuid4())),
                            type=tc.get("type", "function"),
                            function=Function(
                                name=tc.get("function", {}).get("name", ""),
                                arguments=tc.get("function", {}).get("arguments", "{}"),
                            ),
                        )
                        for tc in msg["tool_calls"]
                    ]
                agent_messages.append(ChatAgentMessage(**kwargs))

        agent_request = ChatAgentRequest(
            messages=agent_messages,
            context=ChatContext(conversation_id=session_id),
        )
        response = self.send(agent_request)
        return response.model_dump()
