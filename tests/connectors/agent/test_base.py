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
