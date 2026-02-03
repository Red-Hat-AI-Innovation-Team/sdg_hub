# SPDX-License-Identifier: Apache-2.0
"""Tests for BaseAgentConnector."""

from typing import Any
from unittest.mock import AsyncMock, patch

from sdg_hub.core.connectors.agent.base import BaseAgentConnector
from sdg_hub.core.connectors.base import ConnectorConfig
from sdg_hub.core.connectors.exceptions import ConnectorError
import pytest


class ConcreteAgentConnector(BaseAgentConnector):
    """Concrete implementation for testing."""

    def build_request(self, messages: list[dict[str, Any]], session_id: str) -> dict:
        return {"input": messages[-1]["content"], "session_id": session_id}

    def parse_response(self, response: dict[str, Any]) -> dict:
        if not isinstance(response, dict):
            raise ConnectorError(f"Expected dict, got {type(response)}")
        return response


class TestBaseAgentConnector:
    """Test BaseAgentConnector."""

    def test_build_headers(self):
        """Test header building with and without API key."""
        # Without API key
        connector = ConcreteAgentConnector(config=ConnectorConfig(url="http://test"))
        assert connector._build_headers() == {"Content-Type": "application/json"}

        # With API key
        connector = ConcreteAgentConnector(
            config=ConnectorConfig(url="http://test", api_key="secret")
        )
        headers = connector._build_headers()
        assert headers["Authorization"] == "Bearer secret"

    def test_send_sync_and_async(self):
        """Test send in both sync and async modes."""
        connector = ConcreteAgentConnector(config=ConnectorConfig(url="http://test"))

        with patch.object(connector, "_send_async", new_callable=AsyncMock) as mock:
            mock.return_value = {"output": "result"}

            # Sync mode
            result = connector.send([{"role": "user", "content": "hi"}], "s1")
            assert result == {"output": "result"}

    @pytest.mark.asyncio
    async def test_send_async_no_url_raises_error(self):
        """Test error when no URL configured."""
        connector = ConcreteAgentConnector(config=ConnectorConfig())
        with pytest.raises(ConnectorError, match="No URL configured"):
            await connector._send_async([{"role": "user", "content": "hi"}], "s1")

    def test_lazy_http_client_init(self):
        """Test HTTP client is lazily initialized."""
        connector = ConcreteAgentConnector(config=ConnectorConfig(url="http://test"))
        assert connector._http_client is None
        client = connector._get_http_client()
        assert client is not None
        assert connector._get_http_client() is client  # Same instance

    @pytest.mark.asyncio
    async def test_asend(self):
        """Test asend async method."""
        connector = ConcreteAgentConnector(config=ConnectorConfig(url="http://test"))

        with patch.object(connector, "_send_async", new_callable=AsyncMock) as mock:
            mock.return_value = {"output": "async_result"}
            result = await connector.asend([{"role": "user", "content": "hi"}], "s1")
            assert result == {"output": "async_result"}
            mock.assert_called_once()

    def test_execute(self):
        """Test execute method (BaseConnector interface)."""
        connector = ConcreteAgentConnector(config=ConnectorConfig(url="http://test"))

        with patch.object(connector, "_send_async", new_callable=AsyncMock) as mock:
            mock.return_value = {"output": "executed"}
            result = connector.execute(
                {"messages": [{"role": "user", "content": "hi"}]}
            )
            assert result == {"output": "executed"}

    def test_execute_with_session_id(self):
        """Test execute uses session_id from request."""
        connector = ConcreteAgentConnector(config=ConnectorConfig(url="http://test"))

        with patch.object(connector, "_send_async", new_callable=AsyncMock) as mock:
            mock.return_value = {"output": "result"}
            connector.execute(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "session_id": "custom-session",
                }
            )
            # Verify the session_id was passed through
            mock.assert_called_once()
            call_args = mock.call_args
            assert call_args[0][1] == "custom-session"

    def test_send_returns_coroutine_in_async_mode(self):
        """Test send returns coroutine when async_mode=True."""
        connector = ConcreteAgentConnector(config=ConnectorConfig(url="http://test"))
        result = connector.send(
            [{"role": "user", "content": "hi"}], "s1", async_mode=True
        )
        # Should return a coroutine
        import asyncio

        assert asyncio.iscoroutine(result)
        result.close()  # Clean up the coroutine

    @pytest.mark.asyncio
    async def test_send_async_full_flow(self):
        """Test _send_async with mocked HTTP client."""
        connector = ConcreteAgentConnector(config=ConnectorConfig(url="http://test"))

        mock_client = AsyncMock()
        mock_client.post.return_value = {"result": "success"}

        with patch.object(connector, "_get_http_client", return_value=mock_client):
            result = await connector._send_async(
                [{"role": "user", "content": "hello"}], "session-1"
            )

        assert result == {"result": "success"}
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["url"] == "http://test"
        assert call_kwargs["payload"]["input"] == "hello"
        assert call_kwargs["payload"]["session_id"] == "session-1"

    @pytest.mark.asyncio
    async def test_send_sync_from_async_context(self):
        """Test sync send when called from within async context uses ThreadPoolExecutor."""
        connector = ConcreteAgentConnector(config=ConnectorConfig(url="http://test"))

        with patch.object(connector, "_send_async", new_callable=AsyncMock) as mock:
            mock.return_value = {"output": "from_executor"}
            # This is called from within an async context
            result = connector.send([{"role": "user", "content": "hi"}], "s1")
            assert result == {"output": "from_executor"}
