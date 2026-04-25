# SPDX-License-Identifier: Apache-2.0
"""Tests for LangflowConnector."""

import json

from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentRequest,
    ChatAgentResponse,
    ChatContext,
)
import pytest

from sdg_hub.core.connectors.agent.langflow import LangflowConnector
from sdg_hub.core.connectors.base import ConnectorConfig
from sdg_hub.core.connectors.exceptions import ConnectorError
from sdg_hub.core.connectors.registry import ConnectorRegistry


def make_langflow_raw_response(text, session_id="session-123", content_blocks=None):
    """Create a sample raw Langflow API response structure."""
    msg = {"text": text}
    if content_blocks is not None:
        msg["data"] = {"content_blocks": content_blocks}
    return {
        "session_id": session_id,
        "outputs": [{"outputs": [{"results": {"message": msg}}]}],
    }


SAMPLE_CONTENT_BLOCKS = [
    {
        "title": "Agent Steps",
        "contents": [
            {"type": "text", "header": {"title": "Input"}, "text": "find laptops"},
            {
                "type": "tool_use",
                "name": "search",
                "tool_input": {"q": "laptops"},
                "output": {"content": [{"type": "text", "text": '{"results": []}'}]},
            },
            {"type": "text", "header": {"title": "Output"}, "text": "No results."},
        ],
    }
]


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

        request = ChatAgentRequest(
            messages=[
                ChatAgentMessage(role="user", content="First"),
                ChatAgentMessage(role="assistant", content="Reply"),
                ChatAgentMessage(role="user", content="Second"),
            ],
            context=ChatContext(conversation_id="session-1"),
        )
        payload = connector.build_request(request)

        assert payload == {
            "output_type": "chat",
            "input_type": "chat",
            "input_value": "Second",
            "session_id": "session-1",
        }

        # No user message raises error
        request_no_user = ChatAgentRequest(
            messages=[ChatAgentMessage(role="system", content="hi")]
        )
        with pytest.raises(ConnectorError, match="No user message"):
            connector.build_request(request_no_user)

    def test_build_request_no_context(self):
        """Test build_request with no context uses empty session_id."""
        connector = LangflowConnector(config=ConnectorConfig(url="http://test"))

        request = ChatAgentRequest(
            messages=[ChatAgentMessage(role="user", content="Hello")]
        )
        payload = connector.build_request(request)
        assert payload["session_id"] == ""

    def test_parse_response_basic(self):
        """Test parse_response returns ChatAgentResponse with text."""
        connector = LangflowConnector(config=ConnectorConfig(url="http://test"))

        raw = make_langflow_raw_response("Hello world")
        result = connector.parse_response(raw)

        assert isinstance(result, ChatAgentResponse)
        # Last message should be assistant with the text
        assert result.messages[-1].role == "assistant"
        assert result.messages[-1].content == "Hello world"
        # Should have session_id in custom_outputs
        assert result.custom_outputs["session_id"] == "session-123"

    def test_parse_response_non_dict_raises_error(self):
        """Test non-dict response raises error."""
        connector = LangflowConnector(config=ConnectorConfig(url="http://test"))

        with pytest.raises(ConnectorError, match="Expected dict"):
            connector.parse_response(["not", "a", "dict"])

    def test_parse_response_none_text(self, caplog):
        """Test parse_response handles None text."""
        connector = LangflowConnector(config=ConnectorConfig(url="http://test"))

        raw = make_langflow_raw_response(None)
        result = connector.parse_response(raw)

        assert isinstance(result, ChatAgentResponse)
        assert result.messages[-1].content == ""
        assert "Text field is None, using empty string instead" in caplog.text

    def test_parse_response_missing_text_path(self):
        """Test parse_response when text path is missing."""
        connector = LangflowConnector(config=ConnectorConfig(url="http://test"))

        result = connector.parse_response({"outputs": []})
        assert isinstance(result, ChatAgentResponse)
        # Should have a minimal empty assistant message
        assert len(result.messages) >= 1
        assert result.messages[-1].role == "assistant"

    def test_parse_response_session_id_none(self, caplog):
        """Test parse_response handles None session_id."""
        connector = LangflowConnector(config=ConnectorConfig(url="http://test"))

        raw = {"session_id": None, "outputs": []}
        result = connector.parse_response(raw)

        assert result.custom_outputs["session_id"] == ""
        assert "Session ID field is None, using empty string instead" in caplog.text

    def test_parse_response_no_session_id(self):
        """Test parse_response when session_id key is absent."""
        connector = LangflowConnector(config=ConnectorConfig(url="http://test"))

        result = connector.parse_response(
            {"outputs": [{"outputs": [{"results": {"message": {"text": "Hi"}}}]}]}
        )
        assert result.custom_outputs is None

    def test_parse_response_with_tool_trace(self):
        """Test parse_response extracts tool calls from content_blocks."""
        connector = LangflowConnector(config=ConnectorConfig(url="http://test"))

        raw = make_langflow_raw_response("answer", content_blocks=SAMPLE_CONTENT_BLOCKS)
        result = connector.parse_response(raw)

        assert isinstance(result, ChatAgentResponse)
        # Should have tool-related messages plus final assistant message
        tool_call_msgs = [
            m for m in result.messages if m.role == "assistant" and m.tool_calls
        ]
        tool_result_msgs = [m for m in result.messages if m.role == "tool"]

        assert len(tool_call_msgs) == 1
        assert len(tool_result_msgs) == 1
        assert tool_call_msgs[0].tool_calls[0].function.name == "search"
        assert json.loads(tool_call_msgs[0].tool_calls[0].function.arguments) == {
            "q": "laptops"
        }
        assert tool_result_msgs[0].name == "search"

        # Final message should be the text answer
        assert result.messages[-1].role == "assistant"
        assert result.messages[-1].content == "answer"

    def test_parse_response_with_direct_content_blocks(self):
        """Test parse_response with content_blocks directly on message (no data wrapper)."""
        connector = LangflowConnector(config=ConnectorConfig(url="http://test"))

        response = {
            "outputs": [
                {
                    "outputs": [
                        {
                            "results": {
                                "message": {
                                    "text": "answer",
                                    "content_blocks": SAMPLE_CONTENT_BLOCKS,
                                }
                            }
                        }
                    ]
                }
            ]
        }
        result = connector.parse_response(response)

        assert isinstance(result, ChatAgentResponse)
        tool_call_msgs = [
            m for m in result.messages if m.role == "assistant" and m.tool_calls
        ]
        assert len(tool_call_msgs) == 1

    def test_parse_response_all_messages_have_ids(self):
        """Test that all messages in parse_response have unique ids."""
        connector = LangflowConnector(config=ConnectorConfig(url="http://test"))

        raw = make_langflow_raw_response("answer", content_blocks=SAMPLE_CONTENT_BLOCKS)
        result = connector.parse_response(raw)

        ids = [m.id for m in result.messages]
        assert all(id is not None for id in ids)
        assert len(ids) == len(set(ids)), "All message IDs should be unique"

    def test_parse_response_minimal_empty(self):
        """Test parse_response with completely empty response returns minimal message."""
        connector = LangflowConnector(config=ConnectorConfig(url="http://test"))

        result = connector.parse_response({"some": "data"})
        assert isinstance(result, ChatAgentResponse)
        assert len(result.messages) == 1
        assert result.messages[0].role == "assistant"
