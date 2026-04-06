# SPDX-License-Identifier: Apache-2.0
"""Tests for BaseAgentConnector."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from sdg_hub.core.connectors.agent.base import BaseAgentConnector
from sdg_hub.core.connectors.base import ConnectorConfig
from sdg_hub.core.connectors.exceptions import ConnectorError


class ConcreteAgentConnector(BaseAgentConnector):
    """Concrete implementation for testing."""

    def build_request(self, messages: list[dict[str, Any]], session_id: str) -> dict:
        return {"input": messages[-1]["content"], "session_id": session_id}

    def parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(response, dict):
            raise ConnectorError(f"Expected dict, got {type(response)}")
        return response


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

        with patch.object(connector, "_send_async", new_callable=AsyncMock) as mock:
            mock.return_value = {"output": "result"}

            # Test send
            result = connector.send([{"role": "user", "content": "hi"}], "s1")
            assert result == {"output": "result"}

            # Test execute uses default session_id
            connector.execute({"messages": [{"role": "user", "content": "hi"}]})

            # Test execute with custom session_id
            connector.execute(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "session_id": "custom",
                }
            )
            assert mock.call_args[0][1] == "custom"

    @pytest.mark.asyncio
    async def test_send_async_no_url_raises_error(self):
        """Test error when no URL configured."""
        connector = ConcreteAgentConnector(config=ConnectorConfig())
        with pytest.raises(ConnectorError, match="No URL configured"):
            await connector._send_async([{"role": "user", "content": "hi"}], "s1")

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
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["url"] == "http://test"
        assert call_kwargs["payload"]["input"] == "hello"
