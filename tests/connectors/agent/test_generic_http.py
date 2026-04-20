# SPDX-License-Identifier: Apache-2.0
"""Tests for GenericHTTPConnector."""

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
        """Test request builds nested JSON from dot-notation path."""
        connector = self._make_connector(
            request_message_path="input.question",
        )
        messages = [{"role": "user", "content": "How do I get started?"}]
        request = connector.build_request(messages, "session-1")

        assert request == {"input": {"question": "How do I get started?"}}

    def test_build_request_single_key(self):
        """Test request with a flat path."""
        connector = self._make_connector(
            request_message_path="message",
        )
        messages = [{"role": "user", "content": "Hello"}]
        request = connector.build_request(messages, "session-1")

        assert request == {"message": "Hello"}

    def test_build_request_deep_path(self):
        """Test request with a deeply nested path."""
        connector = self._make_connector(
            request_message_path="data.input.text.content",
        )
        messages = [{"role": "user", "content": "Deep"}]
        request = connector.build_request(messages, "session-1")

        assert request == {"data": {"input": {"text": {"content": "Deep"}}}}

    def test_build_request_extracts_last_user_message(self):
        """Test that the last user message is extracted."""
        connector = self._make_connector()
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]
        request = connector.build_request(messages, "session-1")

        assert request == {"input": {"question": "Second"}}

    def test_build_request_no_user_message_raises(self):
        """Test error when no user message exists."""
        connector = self._make_connector()
        with pytest.raises(ConnectorError, match="No user message"):
            connector.build_request([{"role": "system", "content": "hi"}], "session-1")

    def test_build_request_with_session_id_path(self):
        """Test session ID is included when path is configured."""
        connector = self._make_connector(
            request_session_id_path="session_id",
        )
        messages = [{"role": "user", "content": "Hello"}]
        request = connector.build_request(messages, "session-1")

        assert request == {
            "input": {"question": "Hello"},
            "session_id": "session-1",
        }

    def test_build_request_with_nested_session_id_path(self):
        """Test session ID at a nested path."""
        connector = self._make_connector(
            request_session_id_path="metadata.session.id",
        )
        messages = [{"role": "user", "content": "Hello"}]
        request = connector.build_request(messages, "s-123")

        assert request == {
            "input": {"question": "Hello"},
            "metadata": {"session": {"id": "s-123"}},
        }

    def test_build_request_without_session_id_path(self):
        """Test session ID is excluded when path is not configured."""
        connector = self._make_connector()
        messages = [{"role": "user", "content": "Hello"}]
        request = connector.build_request(messages, "session-1")

        assert "session_id" not in request
        assert request == {"input": {"question": "Hello"}}

    def test_build_request_overlapping_paths_identical(self):
        """Test error when message and session ID paths are identical."""
        connector = self._make_connector(
            request_message_path="input.query",
            request_session_id_path="input.query",
        )
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(ConnectorError, match="must not overlap"):
            connector.build_request(messages, "session-1")

    def test_build_request_overlapping_paths_parent_child(self):
        """Test error when session ID path is parent of message path."""
        connector = self._make_connector(
            request_message_path="input.query",
            request_session_id_path="input",
        )
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(ConnectorError, match="must not overlap"):
            connector.build_request(messages, "session-1")

    def test_build_request_overlapping_paths_child_parent(self):
        """Test error when message path is parent of session ID path."""
        connector = self._make_connector(
            request_message_path="input",
            request_session_id_path="input.session_id",
        )
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(ConnectorError, match="must not overlap"):
            connector.build_request(messages, "session-1")

    def test_build_request_non_overlapping_paths(self):
        """Test that sibling paths work without error."""
        connector = self._make_connector(
            request_message_path="input.query",
            request_session_id_path="input.session_id",
        )
        messages = [{"role": "user", "content": "Hello"}]
        request = connector.build_request(messages, "s-1")

        assert request == {
            "input": {"query": "Hello", "session_id": "s-1"},
        }

    def test_parse_response_valid_dict(self):
        """Test valid dict passes through with extracted text."""
        connector = self._make_connector()
        response = {"output": {"answer": "42"}}
        parsed = connector.parse_response(response)

        assert parsed["output"]["answer"] == "42"
        assert parsed["_extracted_text"] == "42"

    def test_parse_response_extracts_text_from_configured_path(self):
        """Test parse_response uses response_text_path to extract text."""
        connector = self._make_connector(
            response_text_path="data.reply.content",
        )
        response = {"data": {"reply": {"content": "hello world"}}}
        parsed = connector.parse_response(response)

        assert parsed["_extracted_text"] == "hello world"

    def test_parse_response_missing_text_path(self):
        """Test parse_response does not inject key when path is missing."""
        connector = self._make_connector()
        response = {"other": "value"}
        parsed = connector.parse_response(response)

        assert "_extracted_text" not in parsed

    def test_parse_response_extracts_session_id(self):
        """Test parse_response extracts session ID when path is configured."""
        connector = self._make_connector(
            response_session_id_path="meta.sid",
        )
        response = {"output": {"answer": "hi"}, "meta": {"sid": "s-1"}}
        parsed = connector.parse_response(response)

        assert parsed["_extracted_text"] == "hi"
        assert parsed["_extracted_session_id"] == "s-1"

    def test_parse_response_strips_upstream_reserved_keys(self):
        """Test that upstream _extracted_text/_extracted_session_id are stripped."""
        connector = self._make_connector()
        response = {
            "output": {"answer": "real"},
            "_extracted_text": "spoofed",
            "_extracted_session_id": "spoofed-sid",
        }
        parsed = connector.parse_response(response)

        assert parsed["_extracted_text"] == "real"
        assert "_extracted_session_id" not in parsed

    def test_parse_response_non_dict_raises(self):
        """Test non-dict raises error."""
        connector = self._make_connector()
        with pytest.raises(ConnectorError, match="Expected dict"):
            connector.parse_response(["not", "a", "dict"])

    def test_extract_text_reads_injected_key(self):
        """Test extract_text reads _extracted_text from parse_response."""
        assert (
            GenericHTTPConnector.extract_text({"_extracted_text": "hello"}) == "hello"
        )

    def test_extract_text_returns_none_without_key(self):
        """Test extract_text returns None when _extracted_text is absent."""
        assert GenericHTTPConnector.extract_text({"output": "value"}) is None

    def test_extract_session_id_reads_injected_key(self):
        """Test extract_session_id reads _extracted_session_id."""
        assert (
            GenericHTTPConnector.extract_session_id({"_extracted_session_id": "s-1"})
            == "s-1"
        )

    def test_extract_session_id_returns_none_without_key(self):
        """Test extract_session_id returns None when key is absent."""
        assert GenericHTTPConnector.extract_session_id({"other": "val"}) is None

    def test_extract_tool_trace_returns_none(self):
        """Test extract_tool_trace always returns None."""
        assert GenericHTTPConnector.extract_tool_trace({"any": "thing"}) is None

    def test_build_headers_with_api_key(self):
        """Test default headers include Bearer auth."""
        connector = self._make_connector(
            config=ConnectorConfig(url="http://test", api_key="secret"),
        )
        headers = connector._build_headers()
        assert headers["Authorization"] == "Bearer secret"

    def test_build_headers_without_api_key(self):
        """Test headers without API key."""
        connector = self._make_connector()
        headers = connector._build_headers()
        assert headers == {"Content-Type": "application/json"}
        assert "Authorization" not in headers

    def test_validation_empty_path_segment(self):
        """Test validation rejects paths with empty segments."""
        with pytest.raises(ValueError, match="empty segment"):
            self._make_connector(request_message_path="input..question")

    def test_validation_empty_path_segment_optional_field(self):
        """Test validation rejects optional paths with empty segments."""
        with pytest.raises(ValueError, match="empty segment"):
            self._make_connector(response_session_id_path="meta..sid")

    def test_validation_accepts_none_optional_paths(self):
        """Test validation passes when optional paths are None."""
        connector = self._make_connector(
            request_session_id_path=None,
            response_session_id_path=None,
        )
        assert connector.request_session_id_path is None
        assert connector.response_session_id_path is None

    def test_validation_requires_request_message_path(self):
        """Test that request_message_path is required."""
        with pytest.raises(ValueError):
            GenericHTTPConnector(
                config=ConnectorConfig(url="http://test"),
                response_text_path="output.answer",
            )

    def test_validation_requires_response_text_path(self):
        """Test that response_text_path is required."""
        with pytest.raises(ValueError):
            GenericHTTPConnector(
                config=ConnectorConfig(url="http://test"),
                request_message_path="input.question",
            )
