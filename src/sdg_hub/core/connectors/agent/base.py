# SPDX-License-Identifier: Apache-2.0
"""Base class for agent framework connectors."""

from abc import abstractmethod
from typing import Any, Optional
import asyncio

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
    - build_request: Convert messages to framework-specific format
    - parse_response: Convert framework response to standard format

    Example:
        ```python
        class MyAgentConnector(BaseAgentConnector):
            def build_request(self, messages, session_id):
                return {"input": messages[-1]["content"], "session": session_id}

            def parse_response(self, response):
                return {"output": response["result"]}

        connector = MyAgentConnector(config=ConnectorConfig(url="http://api"))
        response = connector.send(
            [{"role": "user", "content": "Hello"}], "session1"
        )
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
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> dict[str, Any]:
        """Build framework-specific request payload.

        Parameters
        ----------
        messages : list[dict]
            List of messages in standard format:
            [{"role": "user", "content": "Hello"}, ...]
        session_id : str
            Session identifier for conversation tracking.

        Returns
        -------
        dict
            Framework-specific request payload.
        """
        pass

    @abstractmethod
    def parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Parse and validate framework response.

        Parameters
        ----------
        response : dict
            Raw response from the framework.

        Returns
        -------
        dict
            Validated response dict.

        Raises
        ------
        ConnectorError
            If the response is invalid or cannot be parsed.
        """
        pass

    async def _send_async(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> dict[str, Any]:
        """Core async implementation.

        Parameters
        ----------
        messages : list[dict]
            Messages to send to the agent.
        session_id : str
            Session identifier.

        Returns
        -------
        dict
            Parsed response from the agent.
        """
        if not self.config.url:
            raise ConnectorError("No URL configured for connector")

        http_client = self._get_http_client()
        request = self.build_request(messages, session_id)
        headers = self._build_headers()

        logger.debug(f"Sending request to {self.config.url}")
        raw_response = await http_client.post(
            url=self.config.url,
            payload=request,
            headers=headers,
        )
        logger.debug(f"Received response from {self.config.url}")

        return self.parse_response(raw_response)

    def send(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
        async_mode: bool = False,
    ):
        """Send messages to the agent.

        Parameters
        ----------
        messages : list[dict]
            Messages to send, in format:
            [{"role": "user", "content": "Hello"}, ...]
        session_id : str
            Session identifier for conversation tracking.
        async_mode : bool, optional
            If True, returns a coroutine. If False (default), runs synchronously.

        Returns
        -------
        dict or Coroutine[dict]
            Response dict, or coroutine if async_mode=True.
        """
        if async_mode:
            return self._send_async(messages, session_id)

        # Sync mode: run async code in event loop
        try:
            asyncio.get_running_loop()
            # Already in async context - use thread executor
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._send_async(messages, session_id),
                )
                return future.result()
        except RuntimeError:
            # No event loop - create one
            return asyncio.run(self._send_async(messages, session_id))

    async def asend(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> dict[str, Any]:
        """Async send - convenience wrapper.

        Parameters
        ----------
        messages : list[dict]
            Messages to send.
        session_id : str
            Session identifier.

        Returns
        -------
        dict
            Response from the agent.
        """
        return await self._send_async(messages, session_id)

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute a request (BaseConnector interface).

        Parameters
        ----------
        request : dict
            Request containing 'messages' and 'session_id' keys.

        Returns
        -------
        dict
            Response from the agent.
        """
        return self.send(
            messages=request["messages"],
            session_id=request.get("session_id", "default"),
        )

    # ------------------------------------------------------------------
    # Response field extraction class methods
    # ------------------------------------------------------------------
    # These allow AgentResponseExtractorBlock to extract fields from
    # framework-specific responses without instantiating a connector.
    # Subclasses override to provide framework-specific parsing.

    @classmethod
    def extract_text(cls, response: dict[str, Any]) -> str | None:
        """Extract text content from a framework response.

        Parameters
        ----------
        response : dict
            Raw response from the agent framework.

        Returns
        -------
        str or None
            Extracted text, or None if not found.
        """
        return None

    @classmethod
    def extract_session_id(cls, response: dict[str, Any]) -> str | None:
        """Extract session ID from a framework response.

        Parameters
        ----------
        response : dict
            Raw response from the agent framework.

        Returns
        -------
        str or None
            Extracted session ID, or None if not found.
        """
        return None

    @classmethod
    def extract_tool_trace(
        cls, response: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        """Extract tool call trace from a framework response.

        Parameters
        ----------
        response : dict
            Raw response from the agent framework.

        Returns
        -------
        list[dict] or None
            List of tool call step dicts, or None if not found.
        """
        return None
