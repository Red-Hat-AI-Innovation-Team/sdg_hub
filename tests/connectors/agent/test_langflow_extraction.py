# SPDX-License-Identifier: Apache-2.0
"""Tests for LangflowConnector extraction class methods."""

from sdg_hub.core.connectors.agent.base import BaseAgentConnector
from sdg_hub.core.connectors.agent.langflow import LangflowConnector


def make_langflow_response(text, session_id="session-123", content_blocks=None):
    """Create a sample Langflow response structure."""
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


class TestBaseAgentConnectorExtractionDefaults:
    """Test that BaseAgentConnector extraction methods return None by default."""

    def test_extract_text_returns_none(self):
        assert BaseAgentConnector.extract_text({"any": "response"}) is None

    def test_extract_session_id_returns_none(self):
        assert BaseAgentConnector.extract_session_id({"any": "response"}) is None

    def test_extract_tool_trace_returns_none(self):
        assert BaseAgentConnector.extract_tool_trace({"any": "response"}) is None


class TestLangflowExtractText:
    """Test LangflowConnector.extract_text class method."""

    def test_success(self):
        response = make_langflow_response("Hello world")
        assert LangflowConnector.extract_text(response) == "Hello world"

    def test_none_returns_none(self, caplog):
        response = make_langflow_response(None)
        result = LangflowConnector.extract_text(response)
        assert result is None
        assert "Text field is None" in caplog.text

    def test_missing_path_returns_none(self):
        assert LangflowConnector.extract_text({"outputs": []}) is None
        assert LangflowConnector.extract_text({}) is None
        assert LangflowConnector.extract_text({"other": "data"}) is None

    def test_deeply_malformed_response(self):
        response = {"outputs": [{"outputs": [{"results": {}}]}]}
        assert LangflowConnector.extract_text(response) is None


class TestLangflowExtractSessionId:
    """Test LangflowConnector.extract_session_id class method."""

    def test_success(self):
        response = make_langflow_response("text", session_id="abc-123")
        assert LangflowConnector.extract_session_id(response) == "abc-123"

    def test_none_returns_none(self, caplog):
        response = {"session_id": None}
        result = LangflowConnector.extract_session_id(response)
        assert result is None
        assert "Session ID field is None" in caplog.text

    def test_missing_key_returns_none(self):
        assert LangflowConnector.extract_session_id({}) is None
        assert LangflowConnector.extract_session_id({"outputs": []}) is None


class TestLangflowExtractToolTrace:
    """Test LangflowConnector.extract_tool_trace class method."""

    def test_success_via_data_path(self):
        response = make_langflow_response("text", content_blocks=SAMPLE_CONTENT_BLOCKS)
        trace = LangflowConnector.extract_tool_trace(response)
        assert trace is not None
        assert len(trace) == 3
        assert trace[0]["type"] == "text"
        assert trace[1]["type"] == "tool_use"
        assert trace[1]["name"] == "search"

    def test_success_via_direct_path(self):
        """Test extraction when content_blocks is directly on message (no data wrapper)."""
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
        trace = LangflowConnector.extract_tool_trace(response)
        assert trace is not None
        assert len(trace) == 3

    def test_missing_returns_none(self, caplog):
        response = make_langflow_response("text")
        result = LangflowConnector.extract_tool_trace(response)
        assert result is None
        assert "No content_blocks with tool trace found" in caplog.text

    def test_empty_content_blocks_returns_none(self, caplog):
        response = make_langflow_response("text", content_blocks=[])
        result = LangflowConnector.extract_tool_trace(response)
        assert result is None
        assert "No content_blocks with tool trace found" in caplog.text
