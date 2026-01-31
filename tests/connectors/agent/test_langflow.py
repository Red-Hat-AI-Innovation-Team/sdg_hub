# SPDX-License-Identifier: Apache-2.0
"""Tests for LangflowConnector."""

import pytest

from sdg_hub.core.connectors.agent.langflow import LangflowConnector
from sdg_hub.core.connectors.base import ConnectorConfig
from sdg_hub.core.connectors.exceptions import ConnectorResponseError
from sdg_hub.core.connectors.registry import ConnectorRegistry


class TestLangflowConnector:
    """Test LangflowConnector."""

    def test_registered_in_registry(self):
        """Test that LangflowConnector is registered."""
        connector_class = ConnectorRegistry.get("langflow")
        assert connector_class == LangflowConnector

    def test_supports_async(self):
        """Test that LangflowConnector supports async."""
        assert LangflowConnector.supports_async is True

    def test_build_headers_without_api_key(self):
        """Test headers without API key."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = LangflowConnector(config=config)

        headers = connector._build_headers()

        assert headers == {"Content-Type": "application/json"}

    def test_build_headers_with_api_key(self):
        """Test headers with API key uses x-api-key."""
        config = ConnectorConfig(
            url="http://localhost:7860",
            api_key="my-secret-key",
        )
        connector = LangflowConnector(config=config)

        headers = connector._build_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["x-api-key"] == "my-secret-key"
        assert "Authorization" not in headers  # Langflow uses x-api-key

    def test_build_request(self):
        """Test build_request formats for Langflow API."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = LangflowConnector(config=config)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        request = connector.build_request(messages, "session-abc")

        assert request == {
            "output_type": "chat",
            "input_type": "chat",
            "input_value": "Hello, how are you?",
            "session_id": "session-abc",
        }

    def test_build_request_extracts_last_user_message(self):
        """Test that build_request extracts the last user message."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = LangflowConnector(config=config)

        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
        request = connector.build_request(messages, "session-123")

        assert request["input_value"] == "Second question"

    def test_build_request_no_user_message_raises_error(self):
        """Test that build_request raises error if no user message."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = LangflowConnector(config=config)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "Hello!"},
        ]

        with pytest.raises(ValueError, match="No user message found"):
            connector.build_request(messages, "session-123")

    def test_build_request_empty_content_raises_error(self):
        """Test that build_request raises error for empty content."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = LangflowConnector(config=config)

        messages = [
            {"role": "user", "content": ""},  # Empty content
        ]

        with pytest.raises(ValueError, match="No user message found"):
            connector.build_request(messages, "session-123")

    def test_parse_response_valid_dict(self):
        """Test parse_response with valid dict response."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = LangflowConnector(config=config)

        response = {
            "outputs": [
                {
                    "outputs": [
                        {
                            "results": {
                                "message": {
                                    "text": "Hello back!"
                                }
                            }
                        }
                    ]
                }
            ]
        }

        parsed = connector.parse_response(response)
        assert parsed == response

    def test_parse_response_invalid_type_raises_error(self):
        """Test parse_response raises error for non-dict."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = LangflowConnector(config=config)

        with pytest.raises(ConnectorResponseError) as exc_info:
            connector.parse_response(["not", "a", "dict"])

        assert "Expected dict response" in str(exc_info.value)

    def test_parse_response_preserves_structure(self):
        """Test that parse_response preserves the full response structure."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = LangflowConnector(config=config)

        response = {
            "outputs": [{"key": "value"}],
            "session_id": "session-123",
            "metadata": {"custom": "data"},
        }

        parsed = connector.parse_response(response)

        assert parsed["outputs"] == [{"key": "value"}]
        assert parsed["session_id"] == "session-123"
        assert parsed["metadata"] == {"custom": "data"}

    def test_extract_last_user_message_with_none_content(self):
        """Test extraction skips messages with None content."""
        config = ConnectorConfig(url="http://localhost:7860")
        connector = LangflowConnector(config=config)

        messages = [
            {"role": "user", "content": "Valid message"},
            {"role": "user", "content": None},  # Should be skipped
        ]

        result = connector._extract_last_user_message(messages)
        assert result == "Valid message"
