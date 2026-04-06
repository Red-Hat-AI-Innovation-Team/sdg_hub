# SPDX-License-Identifier: Apache-2.0
"""Tests for LangGraphConnector."""

from unittest.mock import AsyncMock, patch

import pytest

from sdg_hub.core.connectors.agent.langgraph import LangGraphConnector
from sdg_hub.core.connectors.base import ConnectorConfig
from sdg_hub.core.connectors.exceptions import ConnectorError
from sdg_hub.core.connectors.registry import ConnectorRegistry


class TestLangGraphConnector:
    """Test LangGraphConnector."""

    def test_registered_in_registry(self):
        """Test connector is registered."""
        assert ConnectorRegistry.get("langgraph") == LangGraphConnector

    def test_build_headers_with_api_key(self):
        """Test LangGraph uses x-api-key header."""
        connector = LangGraphConnector(
            config=ConnectorConfig(url="http://test", api_key="secret")
        )
        headers = connector._build_headers()
        assert headers["x-api-key"] == "secret"
        assert "Authorization" not in headers

    def test_build_headers_without_api_key(self):
        """Test headers without API key."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))
        assert connector._build_headers() == {"Content-Type": "application/json"}

    def test_build_request(self):
        """Test request building with default assistant_id."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        request = connector.build_request(messages, "session-1")

        assert request == {
            "assistant_id": "agent",
            "input": {"messages": messages},
        }

    def test_build_request_custom_assistant_id(self):
        """Test request building with custom assistant_id."""
        connector = LangGraphConnector(
            config=ConnectorConfig(url="http://test"),
            assistant_id="my-graph",
        )

        messages = [{"role": "user", "content": "Hello"}]
        request = connector.build_request(messages, "session-1")

        assert request["assistant_id"] == "my-graph"

    def test_parse_response_valid(self):
        """Test response parsing returns raw dict."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))

        response = {
            "messages": [
                {"type": "human", "content": "Hello"},
                {"type": "ai", "content": "Hi there!"},
            ]
        }
        assert connector.parse_response(response) == response

    def test_parse_response_non_dict_raises_error(self):
        """Test non-dict response raises error."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))

        with pytest.raises(ConnectorError, match="Expected dict"):
            connector.parse_response(["not", "a", "dict"])

    @pytest.mark.asyncio
    async def test_send_async_no_url_raises_error(self):
        """Test error when no URL configured."""
        connector = LangGraphConnector(config=ConnectorConfig())
        with pytest.raises(ConnectorError, match="No URL configured"):
            await connector._send_async(
                [{"role": "user", "content": "hi"}], "session-1"
            )

    @pytest.mark.asyncio
    async def test_send_async_full_flow(self):
        """Test _send_async creates thread then runs agent."""
        connector = LangGraphConnector(
            config=ConnectorConfig(url="http://localhost:2024")
        )

        mock_client = AsyncMock()
        # First call: POST /threads -> returns thread_id
        # Second call: POST /threads/{id}/runs/wait -> returns graph state
        mock_client.post.side_effect = [
            {"thread_id": "thread-abc-123"},
            {
                "messages": [
                    {"type": "human", "content": "Hello"},
                    {"type": "ai", "content": "Hi there!"},
                ]
            },
        ]

        with patch.object(connector, "_get_http_client", return_value=mock_client):
            result = await connector._send_async(
                [{"role": "user", "content": "Hello"}], "session-1"
            )

        # Verify thread creation call
        thread_call = mock_client.post.call_args_list[0]
        assert thread_call[1]["url"] == "http://localhost:2024/threads"
        assert thread_call[1]["payload"]["metadata"]["session_id"] == "session-1"

        # Verify run call
        run_call = mock_client.post.call_args_list[1]
        assert (
            run_call[1]["url"]
            == "http://localhost:2024/threads/thread-abc-123/runs/wait"
        )
        assert run_call[1]["payload"]["assistant_id"] == "agent"
        assert run_call[1]["payload"]["input"]["messages"] == [
            {"role": "user", "content": "Hello"}
        ]

        # Verify response
        assert result["messages"][1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_send_async_strips_trailing_slash(self):
        """Test that trailing slashes in URL are handled."""
        connector = LangGraphConnector(
            config=ConnectorConfig(url="http://localhost:2024/")
        )

        mock_client = AsyncMock()
        mock_client.post.side_effect = [
            {"thread_id": "thread-1"},
            {"messages": [{"type": "ai", "content": "ok"}]},
        ]

        with patch.object(connector, "_get_http_client", return_value=mock_client):
            await connector._send_async([{"role": "user", "content": "test"}], "s1")

        thread_url = mock_client.post.call_args_list[0][1]["url"]
        assert thread_url == "http://localhost:2024/threads"

    @pytest.mark.asyncio
    async def test_send_async_custom_assistant_id(self):
        """Test _send_async uses custom assistant_id."""
        connector = LangGraphConnector(
            config=ConnectorConfig(url="http://localhost:2024"),
            assistant_id="my-custom-graph",
        )

        mock_client = AsyncMock()
        mock_client.post.side_effect = [
            {"thread_id": "thread-1"},
            {"messages": [{"type": "ai", "content": "response"}]},
        ]

        with patch.object(connector, "_get_http_client", return_value=mock_client):
            await connector._send_async([{"role": "user", "content": "test"}], "s1")

        run_payload = mock_client.post.call_args_list[1][1]["payload"]
        assert run_payload["assistant_id"] == "my-custom-graph"


class TestLangGraphExtractText:
    """Test LangGraphConnector.extract_text class method."""

    def test_success(self):
        response = {
            "messages": [
                {"type": "human", "content": "Hello"},
                {"type": "ai", "content": "Hi there!"},
            ]
        }
        assert LangGraphConnector.extract_text(response) == "Hi there!"

    def test_role_key_fallback(self):
        """Test extraction works with 'role' key instead of 'type'."""
        response = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        assert LangGraphConnector.extract_text(response) == "Hi!"

    def test_none_content_returns_empty_string(self):
        response = {"messages": [{"type": "ai", "content": None}]}
        assert LangGraphConnector.extract_text(response) == ""

    def test_no_ai_message_returns_none(self):
        response = {"messages": [{"type": "human", "content": "Hello"}]}
        assert LangGraphConnector.extract_text(response) is None

    def test_no_messages_returns_none(self):
        assert LangGraphConnector.extract_text({}) is None
        assert LangGraphConnector.extract_text({"messages": []}) is None
        assert LangGraphConnector.extract_text({"other": "data"}) is None


class TestLangGraphExtractSessionId:
    """Test LangGraphConnector.extract_session_id class method."""

    def test_always_returns_none(self):
        """LangGraph uses thread-based state, no session_id in response."""
        response = {
            "messages": [{"type": "ai", "content": "Hi"}],
            "session_id": "some-id",
        }
        assert LangGraphConnector.extract_session_id(response) is None


class TestLangGraphExtractToolTrace:
    """Test LangGraphConnector.extract_tool_trace class method."""

    def test_success(self):
        response = {
            "messages": [
                {"type": "human", "content": "What is the weather?"},
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {"name": "get_weather", "args": {"city": "NYC"}, "id": "c1"}
                    ],
                },
                {"type": "tool", "name": "get_weather", "content": '{"temp": 72}'},
                {"type": "ai", "content": "It's 72°F in NYC."},
            ]
        }
        trace = LangGraphConnector.extract_tool_trace(response)
        assert trace is not None
        assert len(trace) == 2
        assert trace[0]["type"] == "tool_use"
        assert trace[0]["tool_calls"][0]["name"] == "get_weather"
        assert trace[1]["type"] == "tool_result"
        assert trace[1]["name"] == "get_weather"

    def test_no_tool_calls_returns_none(self):
        response = {
            "messages": [
                {"type": "human", "content": "Hello"},
                {"type": "ai", "content": "Hi!"},
            ]
        }
        assert LangGraphConnector.extract_tool_trace(response) is None

    def test_no_messages_returns_none(self):
        assert LangGraphConnector.extract_tool_trace({}) is None
