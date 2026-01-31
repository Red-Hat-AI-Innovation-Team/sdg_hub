# SPDX-License-Identifier: Apache-2.0
"""Tests for BaseAgentConnector."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from sdg_hub.core.connectors.agent.base import BaseAgentConnector
from sdg_hub.core.connectors.base import ConnectorConfig
from sdg_hub.core.connectors.exceptions import ConnectorResponseError


class ConcreteAgentConnector(BaseAgentConnector):
    """Concrete implementation for testing."""

    def build_request(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> dict[str, Any]:
        return {
            "input": messages[-1]["content"],
            "session_id": session_id,
        }

    def parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(response, dict):
            raise ConnectorResponseError(f"Expected dict, got {type(response)}")
        return response


class TestBaseAgentConnector:
    """Test BaseAgentConnector."""

    def test_supports_async_by_default(self):
        """Test that agent connectors support async by default."""
        assert BaseAgentConnector.supports_async is True

    def test_initialize_client_creates_http_client(self):
        """Test that _initialize_client creates HttpClient."""
        config = ConnectorConfig(url="http://localhost:7860", timeout=60.0)
        connector = ConcreteAgentConnector(config=config)

        connector.warm_up()

        assert connector._http_client is not None
        assert connector._http_client.timeout == 60.0

    def test_build_headers_default(self):
        """Test default header building."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = ConcreteAgentConnector(config=config)

        headers = connector._build_headers()

        assert headers == {"Content-Type": "application/json"}

    def test_build_headers_with_api_key(self):
        """Test header building with API key."""
        config = ConnectorConfig(
            url="http://localhost:7860",
            api_key="secret-key",
        )
        connector = ConcreteAgentConnector(config=config)

        headers = connector._build_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer secret-key"

    def test_build_request(self):
        """Test build_request implementation."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = ConcreteAgentConnector(config=config)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        request = connector.build_request(messages, "session-123")

        assert request["input"] == "Hello!"
        assert request["session_id"] == "session-123"

    def test_parse_response_valid(self):
        """Test parse_response with valid response."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = ConcreteAgentConnector(config=config)

        response = {"output": "Hello back!"}
        parsed = connector.parse_response(response)

        assert parsed == {"output": "Hello back!"}

    def test_parse_response_invalid(self):
        """Test parse_response with invalid response."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = ConcreteAgentConnector(config=config)

        with pytest.raises(ConnectorResponseError):
            connector.parse_response(["not", "a", "dict"])

    @pytest.mark.asyncio
    async def test_send_async_mode(self):
        """Test send in async mode."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = ConcreteAgentConnector(config=config)

        with patch.object(connector, "_send_async", new_callable=AsyncMock) as mock:
            mock.return_value = {"output": "async result"}

            coro = connector.send(
                [{"role": "user", "content": "test"}],
                "session-1",
                async_mode=True,
            )
            result = await coro

            assert result == {"output": "async result"}

    def test_send_sync_mode(self):
        """Test send in sync mode."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = ConcreteAgentConnector(config=config)

        with patch.object(connector, "_send_async", new_callable=AsyncMock) as mock:
            mock.return_value = {"output": "sync result"}

            result = connector.send(
                [{"role": "user", "content": "test"}],
                "session-1",
                async_mode=False,
            )

            assert result == {"output": "sync result"}

    @pytest.mark.asyncio
    async def test_asend(self):
        """Test asend convenience method."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = ConcreteAgentConnector(config=config)

        with patch.object(connector, "_send_async", new_callable=AsyncMock) as mock:
            mock.return_value = {"output": "asend result"}

            result = await connector.asend(
                [{"role": "user", "content": "test"}],
                "session-1",
            )

            assert result == {"output": "asend result"}

    def test_execute_interface(self):
        """Test execute method for BaseConnector interface."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = ConcreteAgentConnector(config=config)

        with patch.object(
            ConcreteAgentConnector, "send", return_value={"output": "result"}
        ) as mock_send:
            result = connector.execute({
                "messages": [{"role": "user", "content": "test"}],
                "session_id": "session-1",
            })

            mock_send.assert_called_once_with(
                messages=[{"role": "user", "content": "test"}],
                session_id="session-1",
            )

    def test_execute_with_default_session_id(self):
        """Test execute uses default session_id if not provided."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = ConcreteAgentConnector(config=config)

        with patch.object(
            ConcreteAgentConnector, "send", return_value={"output": "result"}
        ) as mock_send:
            connector.execute({
                "messages": [{"role": "user", "content": "test"}],
            })

            mock_send.assert_called_once_with(
                messages=[{"role": "user", "content": "test"}],
                session_id="default",
            )

    @pytest.mark.asyncio
    async def test_send_async_no_url_raises_error(self):
        """Test that _send_async raises error if no URL configured."""
        config = ConnectorConfig()  # No URL
        connector = ConcreteAgentConnector(config=config)

        with pytest.raises(ConnectorResponseError, match="No URL configured"):
            await connector._send_async(
                [{"role": "user", "content": "test"}],
                "session-1",
            )
