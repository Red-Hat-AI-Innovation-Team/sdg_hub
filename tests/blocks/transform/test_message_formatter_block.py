"""Tests for MessageFormatterBlock."""

import pandas as pd
import pytest

from sdg_hub.core.blocks.transform import MessageFormatterBlock
from sdg_hub.core.utils.error_handling import MissingColumnError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_list():
    """Minimal tool schema list used across tests."""
    return [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        },
    ]


@pytest.fixture
def tool_trace():
    """A simple tool trace with an input, tool call, and output."""
    return [
        {
            "type": "text",
            "header": {"title": "Input"},
            "text": "What is the weather in Paris?",
        },
        {
            "type": "tool_use",
            "name": "get_weather",
            "tool_input": {"location": "Paris"},
            "output": "Sunny, 22C",
        },
        {
            "type": "text",
            "header": {"title": "Output"},
            "text": "The weather in Paris is sunny and 22C.",
        },
    ]


@pytest.fixture
def sample_dataset(tool_trace, tool_list):
    """A two-row DataFrame with trace and tool_list columns."""
    return pd.DataFrame(
        {
            "trace_col": [tool_trace, tool_trace],
            "tools_col": [tool_list, tool_list],
        }
    )


def _make_block(**overrides):
    """Helper to build a MessageFormatterBlock with sensible defaults."""
    defaults = {
        "block_name": "test_formatter",
        "input_cols": {"tool_trace": "trace_col", "tool_list": "tools_col"},
        "output_cols": ["messages"],
    }
    defaults.update(overrides)
    return MessageFormatterBlock(**defaults)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_basic_generate(sample_dataset):
    """The block should produce a messages column with the correct structure."""
    block = _make_block()
    result = block.generate(sample_dataset)

    assert "messages" in result.columns.tolist()
    assert len(result) == 2

    messages = result.iloc[0]["messages"]
    assert isinstance(messages, list)
    # system, user, assistant (tool_call), tool, assistant (output)
    assert len(messages) == 5

    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_system_message_contains_tool_declarations(sample_dataset):
    """The system message should include the tool declarations."""
    block = _make_block()
    result = block.generate(sample_dataset)
    system_msg = result.iloc[0]["messages"][0]
    assert system_msg["role"] == "system"
    assert "get_weather" in system_msg["content"]


def test_user_message_content(sample_dataset):
    """The user message should reflect the Input text from the trace."""
    block = _make_block()
    result = block.generate(sample_dataset)
    user_msg = result.iloc[0]["messages"][1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "What is the weather in Paris?"


def test_tool_call_structure(sample_dataset):
    """The assistant tool_calls message should have the expected shape."""
    block = _make_block()
    result = block.generate(sample_dataset)
    assistant_tool_msg = result.iloc[0]["messages"][2]
    assert assistant_tool_msg["role"] == "assistant"
    assert assistant_tool_msg["content"] is None
    tool_calls = assistant_tool_msg["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "get_weather"


def test_tool_response_content(sample_dataset):
    """The tool message should carry the tool output."""
    block = _make_block()
    result = block.generate(sample_dataset)
    tool_msg = result.iloc[0]["messages"][3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == "Sunny, 22C"
    assert tool_msg["name"] == "get_weather"


def test_final_assistant_answer(sample_dataset):
    """The last message should be the assistant's final answer."""
    block = _make_block()
    result = block.generate(sample_dataset)
    final_msg = result.iloc[0]["messages"][4]
    assert final_msg["role"] == "assistant"
    assert "sunny" in final_msg["content"].lower()


def test_original_columns_preserved(sample_dataset):
    """Input columns should still be present after generation."""
    block = _make_block()
    result = block.generate(sample_dataset)
    assert "trace_col" in result.columns.tolist()
    assert "tools_col" in result.columns.tolist()


def test_does_not_mutate_input(sample_dataset):
    """The block must not modify the original DataFrame."""
    original_cols = sample_dataset.columns.tolist()
    block = _make_block()
    block.generate(sample_dataset)
    assert sample_dataset.columns.tolist() == original_cols
    assert "messages" not in sample_dataset.columns.tolist()


# ---------------------------------------------------------------------------
# Multiple tool calls in a single trace
# ---------------------------------------------------------------------------


def test_multiple_tool_calls(tool_list):
    """Traces with more than one tool_use step should produce multiple pairs."""
    trace = [
        {"type": "text", "header": {"title": "Input"}, "text": "Question?"},
        {
            "type": "tool_use",
            "name": "get_weather",
            "tool_input": {"location": "Paris"},
            "output": "Sunny",
        },
        {
            "type": "tool_use",
            "name": "get_weather",
            "tool_input": {"location": "London"},
            "output": "Rainy",
        },
        {"type": "text", "header": {"title": "Output"}, "text": "Done."},
    ]
    df = pd.DataFrame({"trace_col": [trace], "tools_col": [tool_list]})
    block = _make_block()
    result = block.generate(df)
    messages = result.iloc[0]["messages"]
    # system, user, assistant+tool (x2), assistant output = 7
    assert len(messages) == 7
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_input_cols_must_be_dict():
    """input_cols as a list should be rejected."""
    with pytest.raises(ValueError, match="dict"):
        _make_block(input_cols=["trace_col", "tools_col"])


def test_input_cols_missing_tool_trace_key():
    """Omitting the tool_trace key should raise."""
    with pytest.raises(ValueError, match="tool_trace"):
        _make_block(input_cols={"tool_list": "tools_col"})


def test_input_cols_missing_tool_list_key():
    """Omitting the tool_list key should raise."""
    with pytest.raises(ValueError, match="tool_list"):
        _make_block(input_cols={"tool_trace": "trace_col"})


def test_output_cols_must_have_exactly_one():
    """More than one output column should be rejected."""
    with pytest.raises(ValueError, match="exactly one output column"):
        _make_block(output_cols=["a", "b"])


def test_output_cols_cannot_be_empty():
    """Empty output_cols should be rejected."""
    with pytest.raises(ValueError, match="exactly one output column"):
        _make_block(output_cols=[])


def test_missing_dataframe_column(tool_trace, tool_list):
    """Referencing a column absent from the DataFrame should error."""
    df = pd.DataFrame({"trace_col": [tool_trace], "tools_col": [tool_list]})
    block = _make_block(
        input_cols={"tool_trace": "nonexistent", "tool_list": "tools_col"},
    )
    with pytest.raises(MissingColumnError):
        block(df)
