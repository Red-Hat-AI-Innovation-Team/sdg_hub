# SPDX-License-Identifier: Apache-2.0
"""Tests for LangflowConnector."""

import pytest

from sdg_hub.core.connectors.agent.langflow import LangflowConnector
from sdg_hub.core.connectors.base import ConnectorConfig
from sdg_hub.core.connectors.exceptions import ConnectorError
from sdg_hub.core.connectors.registry import ConnectorRegistry


class TestLangflowConnector:
    """Test LangflowConnector."""

    def test_registered_in_registry(self):
        """Test connector is registered."""
        assert ConnectorRegistry.get("langflow") == LangflowConnector

    def test_build_headers(self):
        """Test Langflow uses x-api-key header (not Authorization)."""
        # With API key
        connector = LangflowConnector(
            config=ConnectorConfig(url="http://test", api_key="secret")
        )
        headers = connector._build_headers()
        assert headers["x-api-key"] == "secret"
        assert "Authorization" not in headers

        # Without API key
        connector = LangflowConnector(config=ConnectorConfig(url="http://test"))
        assert connector._build_headers() == {"Content-Type": "application/json"}

    def test_build_request(self):
        """Test request building extracts last user message."""
        connector = LangflowConnector(config=ConnectorConfig(url="http://test"))

        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]
        request = connector.build_request(messages, "session-1")

        assert request == {
            "output_type": "chat",
            "input_type": "chat",
            "input_value": "Second",
            "session_id": "session-1",
        }

        # No user message raises error
        with pytest.raises(ConnectorError, match="No user message"):
            connector.build_request([{"role": "system", "content": "hi"}], "s1")

    def test_parse_response(self):
        """Test response parsing returns raw dict."""
        connector = LangflowConnector(config=ConnectorConfig(url="http://test"))

        # Valid dict passes through
        response = {"outputs": [{"data": "value"}]}
        assert connector.parse_response(response) == response

        # Non-dict raises error
        with pytest.raises(ConnectorError, match="Expected dict"):
            connector.parse_response(["not", "a", "dict"])
