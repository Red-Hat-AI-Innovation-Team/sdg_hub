# SPDX-License-Identifier: Apache-2.0
"""Tests for GenericHTTPConnector."""

from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentRequest,
    ChatAgentResponse,
    ChatContext,
)
import pytest

from sdg_hub.core.connectors.agent.generic_http import (
    GenericHTTPConnector,
    _get_nested,
    _set_nested,
)
from sdg_hub.core.connectors.base import ConnectorConfig
from sdg_hub.core.connectors.exceptions import ConnectorError
from sdg_hub.core.connectors.registry import ConnectorRegistry


class TestHelperFunctions:
    """Tests for _set_nested and _get_nested helper functions."""

    def test_set_nested_single_key(self):
        obj = {}
        _set_nested(obj, "message", "hello")
        assert obj == {"message": "hello"}

    def test_set_nested_dotted_path(self):
        obj = {}
        _set_nested(obj, "input.question", "hello")
        assert obj == {"input": {"question": "hello"}}

    def test_set_nested_deep_path(self):
        obj = {}
        _set_nested(obj, "a.b.c.d", "value")
        assert obj == {"a": {"b": {"c": {"d": "value"}}}}

    def test_set_nested_preserves_existing_keys(self):
        obj = {"input": {"other": "keep"}}
        _set_nested(obj, "input.question", "hello")
        assert obj == {"input": {"other": "keep", "question": "hello"}}

    def test_get_nested_single_key(self):
        assert _get_nested({"message": "hello"}, "message") == "hello"

    def test_get_nested_dotted_path(self):
        obj = {"output": {"answer": "world"}}
        assert _get_nested(obj, "output.answer") == "world"

    def test_get_nested_missing_key(self):
        assert _get_nested({"output": {}}, "output.answer") is None

    def test_get_nested_missing_intermediate(self):
        assert _get_nested({}, "a.b.c") is None

    def test_get_nested_non_dict_intermediate(self):
        assert _get_nested({"a": "string"}, "a.b") is None


def _make_request(*contents: str, session_id: str = "") -> ChatAgentRequest:
    """Build a ChatAgentRequest from user message strings."""
    messages = [ChatAgentMessage(role="user", content=c) for c in contents]
    ctx = ChatContext(conversation_id=session_id) if session_id else None
    return ChatAgentRequest(messages=messages, context=ctx)


class TestGenericHTTPConnector:
    """Tests for GenericHTTPConnector."""

    def _make_connector(self, **kwargs):
        """Create a connector with defaults."""
        defaults = {
            "config": ConnectorConfig(url="http://test"),
            "request_message_path": "input.question",
            "response_text_path": "output.answer",
        }
        defaults.update(kwargs)
        return GenericHTTPConnector(**defaults)

    def test_registered_in_registry(self):
        """Test connector is registered."""
        assert ConnectorRegistry.get("generic_http") == GenericHTTPConnector

    def test_build_request_nested_path(self):
        connector = self._make_connector(request_message_path="input.question")
        req = _make_request("How do I get started?")
        payload = connector.build_request(req)

        assert payload == {"input": {"question": "How do I get started?"}}

    def test_build_request_single_key(self):
        connector = self._make_connector(request_message_path="message")
        req = _make_request("Hello")
        payload = connector.build_request(req)

        assert payload == {"message": "Hello"}

    def test_build_request_deep_path(self):
        connector = self._make_connector(
            request_message_path="data.input.text.content",
        )
        req = _make_request("Deep")
        payload = connector.build_request(req)

        assert payload == {"data": {"input": {"text": {"content": "Deep"}}}}

    def test_build_request_extracts_last_user_message(self):
        connector = self._make_connector()
        request = ChatAgentRequest(
            messages=[
                ChatAgentMessage(role="user", content="First"),
                ChatAgentMessage(role="assistant", content="Reply"),
                ChatAgentMessage(role="user", content="Second"),
            ],
        )
        payload = connector.build_request(request)

        assert payload == {"input": {"question": "Second"}}

    def test_build_request_no_user_message_raises(self):
        connector = self._make_connector()
        request = ChatAgentRequest(
            messages=[ChatAgentMessage(role="system", content="hi")],
        )
        with pytest.raises(ConnectorError, match="No user message"):
            connector.build_request(request)

    def test_build_request_with_session_id_path(self):
        connector = self._make_connector(request_session_id_path="session_id")
        req = _make_request("Hello", session_id="session-1")
        payload = connector.build_request(req)

        assert payload == {
            "input": {"question": "Hello"},
            "session_id": "session-1",
        }

    def test_build_request_with_nested_session_id_path(self):
        connector = self._make_connector(
            request_session_id_path="metadata.session.id",
        )
        req = _make_request("Hello", session_id="s-123")
        payload = connector.build_request(req)

        assert payload == {
            "input": {"question": "Hello"},
            "metadata": {"session": {"id": "s-123"}},
        }

    def test_build_request_without_session_id_path(self):
        connector = self._make_connector()
        req = _make_request("Hello", session_id="session-1")
        payload = connector.build_request(req)

        assert "session_id" not in payload
        assert payload == {"input": {"question": "Hello"}}

    def test_build_request_overlapping_paths_identical(self):
        connector = self._make_connector(
            request_message_path="input.query",
            request_session_id_path="input.query",
        )
        req = _make_request("Hello", session_id="session-1")
        with pytest.raises(ConnectorError, match="must not overlap"):
            connector.build_request(req)

    def test_build_request_overlapping_paths_parent_child(self):
        connector = self._make_connector(
            request_message_path="input.query",
            request_session_id_path="input",
        )
        req = _make_request("Hello", session_id="session-1")
        with pytest.raises(ConnectorError, match="must not overlap"):
            connector.build_request(req)

    def test_build_request_overlapping_paths_child_parent(self):
        connector = self._make_connector(
            request_message_path="input",
            request_session_id_path="input.session_id",
        )
        req = _make_request("Hello", session_id="session-1")
        with pytest.raises(ConnectorError, match="must not overlap"):
            connector.build_request(req)

    def test_build_request_non_overlapping_paths(self):
        connector = self._make_connector(
            request_message_path="input.query",
            request_session_id_path="input.session_id",
        )
        req = _make_request("Hello", session_id="s-1")
        payload = connector.build_request(req)

        assert payload == {
            "input": {"query": "Hello", "session_id": "s-1"},
        }

    def test_parse_response_extracts_text(self):
        connector = self._make_connector()
        response = {"output": {"answer": "42"}}
        result = connector.parse_response(response)

        assert isinstance(result, ChatAgentResponse)
        assert result.messages[-1].content == "42"

    def test_parse_response_extracts_text_from_configured_path(self):
        connector = self._make_connector(
            response_text_path="data.reply.content",
        )
        response = {"data": {"reply": {"content": "hello world"}}}
        result = connector.parse_response(response)

        assert result.messages[-1].content == "hello world"

    def test_parse_response_missing_text_path(self):
        connector = self._make_connector()
        response = {"other": "value"}
        result = connector.parse_response(response)

        assert result.messages[-1].content == ""

    def test_parse_response_extracts_session_id(self):
        connector = self._make_connector(
            response_session_id_path="meta.sid",
        )
        response = {"output": {"answer": "hi"}, "meta": {"sid": "s-1"}}
        result = connector.parse_response(response)

        assert result.messages[-1].content == "hi"
        assert result.custom_outputs == {"session_id": "s-1"}

    def test_parse_response_non_dict_raises(self):
        connector = self._make_connector()
        with pytest.raises(ConnectorError, match="Expected dict"):
            connector.parse_response(["not", "a", "dict"])

    def test_build_headers_with_api_key(self):
        connector = self._make_connector(
            config=ConnectorConfig(url="http://test", api_key="secret"),
        )
        headers = connector._build_headers()
        assert headers["Authorization"] == "Bearer secret"

    def test_build_headers_without_api_key(self):
        connector = self._make_connector()
        headers = connector._build_headers()
        assert headers == {"Content-Type": "application/json"}
        assert "Authorization" not in headers

    def test_validation_empty_path_segment(self):
        with pytest.raises(ValueError, match="empty segment"):
            self._make_connector(request_message_path="input..question")

    def test_validation_empty_path_segment_optional_field(self):
        with pytest.raises(ValueError, match="empty segment"):
            self._make_connector(response_session_id_path="meta..sid")

    def test_validation_accepts_none_optional_paths(self):
        connector = self._make_connector(
            request_session_id_path=None,
            response_session_id_path=None,
        )
        assert connector.request_session_id_path is None
        assert connector.response_session_id_path is None

    def test_validation_requires_request_message_path(self):
        with pytest.raises(ValueError):
            GenericHTTPConnector(
                config=ConnectorConfig(url="http://test"),
                response_text_path="output.answer",
            )

    def test_validation_requires_response_text_path(self):
        with pytest.raises(ValueError):
            GenericHTTPConnector(
                config=ConnectorConfig(url="http://test"),
                request_message_path="input.question",
            )

    def test_parse_response_extracts_context_list_as_json(self):
        connector = self._make_connector(
            response_context_path="output.context",
        )
        response = {
            "output": {
                "answer": "hi",
                "context": [{"doc": "a"}, {"doc": "b"}],
            },
        }
        result = connector.parse_response(response)

        assert result.custom_outputs is not None
        assert result.custom_outputs["context"] == '[{"doc": "a"}, {"doc": "b"}]'

    def test_parse_response_extracts_context_dict_as_json(self):
        connector = self._make_connector(
            response_context_path="output.context",
        )
        response = {"output": {"answer": "hi", "context": {"key": "value"}}}
        result = connector.parse_response(response)

        assert result.custom_outputs is not None
        assert result.custom_outputs["context"] == '{"key": "value"}'

    def test_parse_response_extracts_context_string(self):
        connector = self._make_connector(
            response_context_path="output.context",
        )
        response = {"output": {"answer": "hi", "context": "some context"}}
        result = connector.parse_response(response)

        assert result.custom_outputs is not None
        assert result.custom_outputs["context"] == "some context"

    def test_parse_response_context_path_missing_in_response(self):
        connector = self._make_connector(
            response_context_path="output.context",
        )
        response = {"output": {"answer": "hi"}}
        result = connector.parse_response(response)

        assert result.custom_outputs is None

    def test_parse_response_no_context_without_path(self):
        connector = self._make_connector()
        response = {"output": {"answer": "hi", "context": "ignored"}}
        result = connector.parse_response(response)

        assert result.custom_outputs is None
