# SPDX-License-Identifier: Apache-2.0
"""Tests for LangGraphConnector."""

from unittest.mock import AsyncMock, patch

from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentRequest,
    ChatAgentResponse,
    ChatContext,
)
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

        request = ChatAgentRequest(
            messages=[
                ChatAgentMessage(role="user", content="Hello"),
                ChatAgentMessage(role="assistant", content="Hi"),
                ChatAgentMessage(role="user", content="What is 2+2?"),
            ]
        )
        payload = connector.build_request(request)

        assert payload == {
            "assistant_id": "agent",
            "input": {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                    {"role": "user", "content": "What is 2+2?"},
                ]
            },
        }

    def test_build_request_custom_assistant_id(self):
        """Test request building with custom assistant_id."""
        connector = LangGraphConnector(
            config=ConnectorConfig(url="http://test"),
            assistant_id="my-graph",
        )

        request = ChatAgentRequest(
            messages=[ChatAgentMessage(role="user", content="Hello")]
        )
        payload = connector.build_request(request)

        assert payload["assistant_id"] == "my-graph"

    def test_build_request_with_run_config(self):
        """Test request includes config when run_config is set."""
        connector = LangGraphConnector(
            config=ConnectorConfig(url="http://test"),
            run_config={"configurable": {"model": "gpt-4o"}},
        )

        request = ChatAgentRequest(
            messages=[ChatAgentMessage(role="user", content="Hello")]
        )
        payload = connector.build_request(request)

        assert payload["assistant_id"] == "agent"
        assert payload["config"] == {"configurable": {"model": "gpt-4o"}}

    def test_build_request_without_run_config(self):
        """Test request omits config key when run_config is empty."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))

        request = ChatAgentRequest(
            messages=[ChatAgentMessage(role="user", content="Hello")]
        )
        payload = connector.build_request(request)

        assert "config" not in payload

    def test_parse_response_valid(self):
        """Test response parsing returns ChatAgentResponse."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))

        response = {
            "messages": [
                {"type": "human", "content": "Hello"},
                {"type": "ai", "content": "Hi there!"},
            ]
        }
        result = connector.parse_response(response)
        assert isinstance(result, ChatAgentResponse)
        assert len(result.messages) == 2
        # "human" maps to "user"
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Hello"
        # "ai" maps to "assistant"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].content == "Hi there!"

    def test_parse_response_with_tool_calls(self):
        """Test parse_response handles messages with tool_calls."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))

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
                {
                    "type": "tool",
                    "name": "get_weather",
                    "content": '{"temp": 72}',
                    "tool_call_id": "c1",
                },
                {"type": "ai", "content": "It's 72F in NYC."},
            ]
        }
        result = connector.parse_response(response)
        assert isinstance(result, ChatAgentResponse)
        assert len(result.messages) == 4

        # Check tool call assistant message
        ai_with_tc = result.messages[1]
        assert ai_with_tc.role == "assistant"
        assert len(ai_with_tc.tool_calls) == 1
        assert ai_with_tc.tool_calls[0].function.name == "get_weather"
        assert ai_with_tc.tool_calls[0].id == "c1"

        # Check tool result message
        tool_msg = result.messages[2]
        assert tool_msg.role == "tool"
        assert tool_msg.name == "get_weather"
        assert tool_msg.content == '{"temp": 72}'

        # Check final assistant message
        assert result.messages[3].role == "assistant"
        assert result.messages[3].content == "It's 72F in NYC."

    def test_parse_response_role_key_fallback(self):
        """Test parse_response works with 'role' key instead of 'type'."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))

        response = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        result = connector.parse_response(response)
        assert isinstance(result, ChatAgentResponse)
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"

    def test_parse_response_all_messages_have_ids(self):
        """Test that all messages have unique ids."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))

        response = {
            "messages": [
                {"type": "human", "content": "Hello"},
                {"type": "ai", "content": "Hi there!"},
            ]
        }
        result = connector.parse_response(response)
        ids = [m.id for m in result.messages]
        assert all(id is not None for id in ids)
        assert len(ids) == len(set(ids))

    def test_parse_response_non_dict_raises_error(self):
        """Test non-dict response raises error."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))

        with pytest.raises(ConnectorError, match="Expected dict"):
            connector.parse_response(["not", "a", "dict"])

    @pytest.mark.asyncio
    async def test_send_async_no_url_raises_error(self):
        """Test error when no URL configured."""
        connector = LangGraphConnector(config=ConnectorConfig())
        request = ChatAgentRequest(
            messages=[ChatAgentMessage(role="user", content="hi")]
        )
        with pytest.raises(ConnectorError, match="No URL configured"):
            await connector._send_async(request)

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
            request = ChatAgentRequest(
                messages=[ChatAgentMessage(role="user", content="Hello")],
                context=ChatContext(conversation_id="session-1"),
            )
            result = await connector._send_async(request)

        assert isinstance(result, ChatAgentResponse)

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
        assert result.messages[-1].content == "Hi there!"

    @pytest.mark.asyncio
    async def test_send_async_default_session_id(self):
        """Test _send_async uses 'default' when no context."""
        connector = LangGraphConnector(
            config=ConnectorConfig(url="http://localhost:2024")
        )

        mock_client = AsyncMock()
        mock_client.post.side_effect = [
            {"thread_id": "thread-1"},
            {"messages": [{"type": "ai", "content": "ok"}]},
        ]

        with patch.object(connector, "_get_http_client", return_value=mock_client):
            request = ChatAgentRequest(
                messages=[ChatAgentMessage(role="user", content="test")]
            )
            await connector._send_async(request)

        thread_call = mock_client.post.call_args_list[0]
        assert thread_call[1]["payload"]["metadata"]["session_id"] == "default"

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
            request = ChatAgentRequest(
                messages=[ChatAgentMessage(role="user", content="test")],
                context=ChatContext(conversation_id="s1"),
            )
            await connector._send_async(request)

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
            request = ChatAgentRequest(
                messages=[ChatAgentMessage(role="user", content="test")],
                context=ChatContext(conversation_id="s1"),
            )
            await connector._send_async(request)

        run_payload = mock_client.post.call_args_list[1][1]["payload"]
        assert run_payload["assistant_id"] == "my-custom-graph"


class TestLangGraphErrorPaths:
    """Test error handling in LangGraphConnector."""

    def test_build_request_empty_messages(self):
        """Test build_request raises on empty messages."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))
        request = ChatAgentRequest(messages=[])
        with pytest.raises(ConnectorError, match="empty messages"):
            connector.build_request(request)

    def test_parse_response_empty_dict(self):
        """Test parse_response raises on empty dict."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))
        with pytest.raises(ConnectorError, match="empty response"):
            connector.parse_response({})

    def test_parse_response_missing_messages_warns(self):
        """Test parse_response raises when no messages can be parsed."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))
        # Response with status but no messages key
        with pytest.raises(ConnectorError, match="Could not parse any messages"):
            connector.parse_response({"status": "ok"})

    def test_assistant_id_empty_string_rejected(self):
        """Test that empty assistant_id is rejected by validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LangGraphConnector(
                config=ConnectorConfig(url="http://test"),
                assistant_id="",
            )

    @pytest.mark.asyncio
    async def test_send_async_missing_thread_id(self):
        """Test _send_async raises when thread response has no thread_id."""
        connector = LangGraphConnector(
            config=ConnectorConfig(url="http://localhost:2024")
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = {"status": "ok"}  # No thread_id

        with patch.object(connector, "_get_http_client", return_value=mock_client):
            request = ChatAgentRequest(
                messages=[ChatAgentMessage(role="user", content="hi")]
            )
            with pytest.raises(ConnectorError, match="missing 'thread_id'"):
                await connector._send_async(request)

    @pytest.mark.asyncio
    async def test_send_async_thread_creation_failure(self):
        """Test _send_async wraps thread creation errors with context."""
        connector = LangGraphConnector(
            config=ConnectorConfig(url="http://localhost:2024")
        )
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("Connection refused")

        with patch.object(connector, "_get_http_client", return_value=mock_client):
            request = ChatAgentRequest(
                messages=[ChatAgentMessage(role="user", content="hi")]
            )
            with pytest.raises(ConnectorError, match="thread creation failed"):
                await connector._send_async(request)

    @pytest.mark.asyncio
    async def test_send_async_run_execution_failure(self):
        """Test _send_async wraps run execution errors with context."""
        connector = LangGraphConnector(
            config=ConnectorConfig(url="http://localhost:2024")
        )
        mock_client = AsyncMock()
        # Thread creation succeeds, run fails
        mock_client.post.side_effect = [
            {"thread_id": "thread-1"},
            Exception("Timeout"),
        ]

        with patch.object(connector, "_get_http_client", return_value=mock_client):
            request = ChatAgentRequest(
                messages=[ChatAgentMessage(role="user", content="hi")]
            )
            with pytest.raises(ConnectorError, match="run execution failed"):
                await connector._send_async(request)


class TestLangGraphParseResponseRoleMapping:
    """Test LangGraph parse_response role mapping and message construction."""

    def test_none_content_handled(self):
        """Test handling of None content in messages."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))

        response = {"messages": [{"type": "ai", "content": None}]}
        result = connector.parse_response(response)
        assert result.messages[0].content == "None"

    def test_no_messages_key_raises_error(self):
        """Test that response with no parseable messages raises error."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))

        with pytest.raises(ConnectorError, match="Could not parse any messages"):
            connector.parse_response({"other": "data"})

    def test_non_dict_messages_skipped(self):
        """Test that non-dict items in messages list are skipped."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))

        response = {
            "messages": [
                "not-a-dict",
                {"type": "ai", "content": "valid"},
            ]
        }
        result = connector.parse_response(response)
        assert len(result.messages) == 1
        assert result.messages[0].content == "valid"

    def test_tool_call_id_fallback(self):
        """Test tool message uses 'id' as fallback for tool_call_id."""
        connector = LangGraphConnector(config=ConnectorConfig(url="http://test"))

        response = {
            "messages": [
                {
                    "type": "tool",
                    "name": "my_tool",
                    "content": "result",
                    "id": "fallback-id",
                }
            ]
        }
        result = connector.parse_response(response)
        assert result.messages[0].tool_call_id == "fallback-id"
