# SPDX-License-Identifier: Apache-2.0
"""Tests for BaseAgentConnector."""

from typing import Any
from unittest.mock import AsyncMock, patch
import uuid

from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentRequest,
    ChatAgentResponse,
    ChatContext,
)
import pytest

from sdg_hub.core.connectors.agent.base import BaseAgentConnector
from sdg_hub.core.connectors.base import ConnectorConfig
from sdg_hub.core.connectors.exceptions import ConnectorError


class ConcreteAgentConnector(BaseAgentConnector):
    """Concrete implementation for testing."""

    def build_request(self, request: ChatAgentRequest) -> dict:
        return {
            "input": request.messages[-1].content,
            "session_id": (request.context.conversation_id if request.context else ""),
        }

    def parse_response(self, response: dict[str, Any]) -> ChatAgentResponse:
        if not isinstance(response, dict):
            raise ConnectorError(f"Expected dict, got {type(response)}")
        return ChatAgentResponse(
            messages=[
                ChatAgentMessage(
                    role="assistant",
                    content=str(response.get("result", "")),
                    id=str(uuid.uuid4()),
                )
            ]
        )


class TestBaseAgentConnector:
    """Test BaseAgentConnector."""

    def test_build_headers(self):
        """Test header building with and without API key."""
        connector = ConcreteAgentConnector(config=ConnectorConfig(url="http://test"))
        assert connector._build_headers() == {"Content-Type": "application/json"}

        connector = ConcreteAgentConnector(
            config=ConnectorConfig(url="http://test", api_key="secret")
        )
        assert connector._build_headers()["Authorization"] == "Bearer secret"

    def test_send_and_execute(self):
        """Test send and execute methods."""
        connector = ConcreteAgentConnector(config=ConnectorConfig(url="http://test"))

        mock_response = ChatAgentResponse(
            messages=[
                ChatAgentMessage(
                    role="assistant", content="result", id=str(uuid.uuid4())
                )
            ]
        )

        with patch.object(connector, "_send_async", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response

            # Test send with ChatAgentRequest
            request = ChatAgentRequest(
                messages=[ChatAgentMessage(role="user", content="hi")]
            )
            result = connector.send(request)
            assert isinstance(result, ChatAgentResponse)
            assert result.messages[0].content == "result"

            # Test execute uses default session_id
            connector.execute({"messages": [{"role": "user", "content": "hi"}]})

            # Test execute with custom session_id
            connector.execute(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "session_id": "custom",
                }
            )
            call_request = mock.call_args[0][0]
            assert isinstance(call_request, ChatAgentRequest)
            assert call_request.context.conversation_id == "custom"

    @pytest.mark.asyncio
    async def test_send_async_no_url_raises_error(self):
        """Test error when no URL configured."""
        connector = ConcreteAgentConnector(config=ConnectorConfig())
        request = ChatAgentRequest(
            messages=[ChatAgentMessage(role="user", content="hi")]
        )
        with pytest.raises(ConnectorError, match="No URL configured"):
            await connector._send_async(request)

    @pytest.mark.asyncio
    async def test_send_async_full_flow(self):
        """Test _send_async with mocked HTTP client."""
        connector = ConcreteAgentConnector(config=ConnectorConfig(url="http://test"))

        mock_client = AsyncMock()
        mock_client.post.return_value = {"result": "success"}

        with patch.object(connector, "_get_http_client", return_value=mock_client):
            request = ChatAgentRequest(
                messages=[ChatAgentMessage(role="user", content="hello")],
                context=ChatContext(conversation_id="session-1"),
            )
            result = await connector._send_async(request)

        assert isinstance(result, ChatAgentResponse)
        assert result.messages[0].content == "success"
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["url"] == "http://test"
        assert call_kwargs["payload"]["input"] == "hello"
