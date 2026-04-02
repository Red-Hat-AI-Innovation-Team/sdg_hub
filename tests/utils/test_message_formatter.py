"""Tests for message_formatter utility."""

import json

from sdg_hub.core.utils.message_formatter import tool_trace_to_messages

SAMPLE_TOOL_LIST = [
    {
        "name": "search_products",
        "description": "Search products",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
    {
        "name": "get_details",
        "description": "Get product details",
        "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
    },
]


class TestToolTraceToMessages:
    """Test converting Langflow tool traces to function-calling messages."""

    def test_basic_trace(self):
        """Test a trace with input, one tool call, and output."""
        trace = [
            {"type": "text", "header": {"title": "Input"}, "text": "find laptops"},
            {
                "type": "tool_use",
                "name": "search_products",
                "tool_input": {"query": "laptops"},
                "output": {"content": [{"type": "text", "text": '{"results": []}'}]},
            },
            {"type": "text", "header": {"title": "Output"}, "text": "No results."},
        ]
        msgs = tool_trace_to_messages(trace, SAMPLE_TOOL_LIST)

        assert len(msgs) == 5
        assert msgs[0]["role"] == "system"
        assert msgs[1] == {"role": "user", "content": "find laptops"}
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["function_call"]["name"] == "search_products"
        assert msgs[3]["role"] == "function"
        assert msgs[3]["name"] == "search_products"
        assert msgs[4] == {"role": "assistant", "content": "No results."}

    def test_system_message_declares_all_tools(self):
        """Test that the system message contains all tool schemas."""
        trace = [{"type": "text", "header": {"title": "Input"}, "text": "hi"}]
        msgs = tool_trace_to_messages(trace, SAMPLE_TOOL_LIST)

        content = msgs[0]["content"]
        assert content.startswith("<|im_system|>tool_declare<|im_middle|>")
        assert content.endswith("<|im_end|>")
        tools_json = content.split("<|im_middle|>")[1].split("<|im_end|>")[0]
        tools = json.loads(tools_json)
        assert len(tools) == 2
        assert tools[0]["function"]["name"] == "search_products"

    def test_multi_tool_chain(self):
        """Test a trace with multiple chained tool calls."""
        trace = [
            {"type": "text", "header": {"title": "Input"}, "text": "question"},
            {
                "type": "tool_use",
                "name": "search_products",
                "tool_input": {"query": "x"},
                "output": {"content": [{"type": "text", "text": '{"id": "1"}'}]},
            },
            {
                "type": "tool_use",
                "name": "get_details",
                "tool_input": {"id": "1"},
                "output": {"content": [{"type": "text", "text": '{"name": "X"}'}]},
            },
            {"type": "text", "header": {"title": "Output"}, "text": "done"},
        ]
        msgs = tool_trace_to_messages(trace, SAMPLE_TOOL_LIST)

        assert len(msgs) == 7  # system + user + 2*(call+response) + output
        assert msgs[2]["function_call"]["name"] == "search_products"
        assert msgs[4]["function_call"]["name"] == "get_details"

    def test_string_output(self):
        """Test tool_use with plain string output."""
        trace = [
            {
                "type": "tool_use",
                "name": "search_products",
                "tool_input": {},
                "output": "plain text result",
            },
        ]
        msgs = tool_trace_to_messages(trace, SAMPLE_TOOL_LIST)
        assert msgs[2]["content"] == "plain text result"

    def test_structured_content_fallback(self):
        """Test tool_use with structuredContent (no content array)."""
        trace = [
            {
                "type": "tool_use",
                "name": "search_products",
                "tool_input": {},
                "output": {"structuredContent": {"count": 5}},
            },
        ]
        msgs = tool_trace_to_messages(trace, SAMPLE_TOOL_LIST)
        assert json.loads(msgs[2]["content"]) == {"count": 5}

    def test_intermediate_reasoning(self):
        """Test text steps without Input/Output title become assistant messages."""
        trace = [
            {"type": "text", "header": {"title": "Input"}, "text": "q"},
            {"type": "text", "header": {"title": ""}, "text": "thinking..."},
            {"type": "text", "header": {"title": "Output"}, "text": "done"},
        ]
        msgs = tool_trace_to_messages(trace, SAMPLE_TOOL_LIST)
        assert msgs[2] == {"role": "assistant", "content": "thinking..."}

    def test_none_output(self):
        """Test tool_use with None output."""
        trace = [
            {
                "type": "tool_use",
                "name": "search_products",
                "tool_input": {},
                "output": None,
            },
        ]
        msgs = tool_trace_to_messages(trace, SAMPLE_TOOL_LIST)
        assert msgs[2]["content"] == ""

    # --- Edge cases for malformed traces (issue #646) ---

    def test_missing_type_field(self):
        """A step with no 'type' key is silently skipped."""
        trace = [{"text": "hello"}]
        msgs = tool_trace_to_messages(trace, SAMPLE_TOOL_LIST)
        # Only the system message should be present
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_empty_content_list(self):
        """Empty content list is falsy, so the entire output dict is JSON-serialized."""
        trace = [
            {
                "type": "tool_use",
                "name": "search_products",
                "tool_input": {},
                "output": {"content": []},
            },
        ]
        msgs = tool_trace_to_messages(trace, SAMPLE_TOOL_LIST)
        # Empty content list is falsy, so falls through to json.dumps(output)
        assert json.loads(msgs[2]["content"]) == {"content": []}

    def test_missing_header_field(self):
        """A text step with no 'header' key defaults to assistant message."""
        trace = [{"type": "text", "text": "no header here"}]
        msgs = tool_trace_to_messages(trace, SAMPLE_TOOL_LIST)
        assert len(msgs) == 2
        assert msgs[1] == {"role": "assistant", "content": "no header here"}

    def test_missing_text_in_content_item(self):
        """Content item without 'text' key returns empty string."""
        trace = [
            {
                "type": "tool_use",
                "name": "search_products",
                "tool_input": {},
                "output": {"content": [{"type": "text"}]},
            },
        ]
        msgs = tool_trace_to_messages(trace, SAMPLE_TOOL_LIST)
        assert msgs[2]["content"] == ""

    def test_non_list_content_field(self):
        """Non-list content field falls through to structuredContent path."""
        trace = [
            {
                "type": "tool_use",
                "name": "search_products",
                "tool_input": {},
                "output": {"content": "not a list"},
            },
        ]
        msgs = tool_trace_to_messages(trace, SAMPLE_TOOL_LIST)
        # isinstance("not a list", list) is False, so falls through
        assert json.loads(msgs[2]["content"]) == {"content": "not a list"}

    def test_unknown_type_value(self):
        """A step with an unrecognized type value is silently skipped."""
        trace = [{"type": "image", "url": "http://example.com/img.png"}]
        msgs = tool_trace_to_messages(trace, SAMPLE_TOOL_LIST)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_numeric_tool_output(self):
        """Non-dict, non-string, non-None output is JSON-serialized."""
        trace = [
            {
                "type": "tool_use",
                "name": "search_products",
                "tool_input": {},
                "output": 42,
            },
        ]
        msgs = tool_trace_to_messages(trace, SAMPLE_TOOL_LIST)
        assert msgs[2]["content"] == "42"
