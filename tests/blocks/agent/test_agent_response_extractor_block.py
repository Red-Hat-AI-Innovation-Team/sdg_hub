# SPDX-License-Identifier: Apache-2.0
"""Tests for AgentResponseExtractorBlock."""

import json
import uuid

from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse
from mlflow.types.chat import Function, ToolCall
import pandas as pd
import pytest

from sdg_hub.core.blocks.agent import AgentResponseExtractorBlock
from sdg_hub.core.blocks.registry import BlockRegistry


def make_standard_response(text, session_id=None, tool_messages=None):
    """Create a standardized ChatAgentResponse.model_dump() dict.

    Parameters
    ----------
    text : str
        Text content for the final assistant message.
    session_id : str, optional
        Session ID to include in custom_outputs.
    tool_messages : list[ChatAgentMessage], optional
        Additional tool-related messages to include before the final assistant message.

    Returns
    -------
    dict
        ChatAgentResponse serialized as dict.
    """
    messages = []
    if tool_messages:
        messages.extend(tool_messages)
    messages.append(
        ChatAgentMessage(role="assistant", content=text, id=str(uuid.uuid4()))
    )
    custom_outputs = {"session_id": session_id} if session_id else None
    return ChatAgentResponse(
        messages=messages, custom_outputs=custom_outputs
    ).model_dump()


class TestAgentResponseExtractorBlockRegistration:
    """Test AgentResponseExtractorBlock registration."""

    def test_registered_in_block_registry(self):
        """Test block is registered."""
        block_class = BlockRegistry._get("AgentResponseExtractorBlock")
        assert block_class == AgentResponseExtractorBlock

    def test_registered_in_agent_category(self):
        """Test block is in agent category."""
        agent_blocks = BlockRegistry.list_blocks(category="agent")
        assert "AgentResponseExtractorBlock" in agent_blocks

    def test_metadata_description(self):
        """Test block metadata description."""
        assert "AgentResponseExtractorBlock" in BlockRegistry._metadata
        assert (
            BlockRegistry._metadata["AgentResponseExtractorBlock"].category == "agent"
        )


class TestAgentResponseExtractorBlockInitialization:
    """Test AgentResponseExtractorBlock initialization."""

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
        )

        assert block.block_name == "test_extractor"
        assert block.input_cols == ["agent_response"]
        assert block.extract_text is True
        assert block.extract_session_id is False
        assert block.expand_lists is True
        assert block.field_prefix == ""

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=True,
            expand_lists=False,
            field_prefix="agent_",
        )

        assert block.extract_text is True
        assert block.extract_session_id is True
        assert block.expand_lists is False
        assert block.field_prefix == "agent_"

    def test_init_no_extraction_fields_enabled(self):
        """Test that initialization fails when no extraction fields are enabled."""
        with pytest.raises(ValueError, match="at least one extraction field"):
            AgentResponseExtractorBlock(
                block_name="test_extractor",
                input_cols="agent_response",
                extract_text=False,
                extract_session_id=False,
            )

    def test_field_name_computation(self):
        """Test that field names are computed correctly."""
        # Test with empty prefix (should use block name)
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            field_prefix="",
        )
        assert block._text_field == "test_extractor_text"
        assert block._session_id_field == "test_extractor_session_id"

        # Test with custom prefix
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            field_prefix="agent_",
        )
        assert block._text_field == "agent_text"
        assert block._session_id_field == "agent_session_id"


class TestAgentResponseExtractorBlockExtraction:
    """Test AgentResponseExtractorBlock extraction from standardized responses."""

    def test_extract_text_only(self):
        """Test extracting only text from standard response."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=False,
        )

        response = make_standard_response("Hello world")
        dataset = pd.DataFrame(
            {"agent_response": [response], "other_col": ["other_value"]}
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert "test_extractor_text" in result.columns.tolist()
        assert result["test_extractor_text"][0] == "Hello world"
        assert result["other_col"][0] == "other_value"

    def test_extract_all_fields(self):
        """Test extracting all fields from standard response."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=True,
        )

        response = make_standard_response("Hello world", "session-abc")
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert len(result) == 1
        assert result["test_extractor_text"][0] == "Hello world"
        assert result["test_extractor_session_id"][0] == "session-abc"

    def test_extract_with_custom_prefix(self):
        """Test extracting with custom field prefix."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            field_prefix="agent_",
        )

        response = make_standard_response("Hello world")
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert len(result) == 1
        assert "agent_text" in result.columns.tolist()
        assert result["agent_text"][0] == "Hello world"

    def test_missing_text_field(self, caplog):
        """Test handling when no assistant message has content."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=True,
        )

        # Response with session_id but empty assistant content
        response = ChatAgentResponse(
            messages=[
                ChatAgentMessage(role="assistant", content="", id=str(uuid.uuid4()))
            ],
            custom_outputs={"session_id": "session-123"},
        ).model_dump()

        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert len(result) == 1
        assert result["test_extractor_session_id"][0] == "session-123"
        # Empty content should still be extracted (the assistant message exists)
        assert result["test_extractor_text"][0] == ""

    def test_missing_session_id_field(self, caplog):
        """Test handling missing session_id in custom_outputs."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=True,
        )

        # Response with text but no session_id in custom_outputs
        response = make_standard_response("Hi")
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert len(result) == 1
        assert result["test_extractor_text"][0] == "Hi"
        assert "test_extractor_session_id" in result.columns
        assert result["test_extractor_session_id"][0] is None
        assert "Requested fields ['session_id'] not found in response" in caplog.text


class TestAgentResponseExtractorBlockListResponsesExpandTrue:
    """Test AgentResponseExtractorBlock with list responses and expand_lists=True."""

    def test_expand_list_responses(self):
        """Test expanding list of responses into individual rows."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=True,
        )

        responses = [
            make_standard_response("Response 1"),
            make_standard_response("Response 2"),
            make_standard_response("Response 3"),
        ]
        dataset = pd.DataFrame(
            {"agent_response": [responses], "other_col": ["original_value"]}
        )

        result = block.generate(dataset)

        assert len(result) == 3
        assert result["test_extractor_text"].tolist() == [
            "Response 1",
            "Response 2",
            "Response 3",
        ]
        assert result["other_col"].tolist() == [
            "original_value",
            "original_value",
            "original_value",
        ]

    def test_expand_multiple_samples(self):
        """Test expanding multiple samples with list responses."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=True,
        )

        dataset = pd.DataFrame(
            {
                "agent_response": [
                    [
                        make_standard_response("Sample 1 Response 1"),
                        make_standard_response("Sample 1 Response 2"),
                    ],
                    [make_standard_response("Sample 2 Response 1")],
                ],
                "sample_id": [1, 2],
            }
        )

        result = block.generate(dataset)

        assert len(result) == 3
        assert result["test_extractor_text"].tolist() == [
            "Sample 1 Response 1",
            "Sample 1 Response 2",
            "Sample 2 Response 1",
        ]
        assert result["sample_id"].tolist() == [1, 1, 2]

    def test_expand_empty_list(self):
        """Test handling empty list responses."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=True,
        )

        dataset = pd.DataFrame({"agent_response": [[]], "other_col": ["value"]})

        result = block.generate(dataset)

        assert len(result) == 0

    def test_expand_list_with_missing_text(self, caplog):
        """Test that rows with missing text get None instead of being dropped."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=True,
        )

        dataset = pd.DataFrame(
            {
                "agent_response": [
                    [
                        make_standard_response("Valid response"),
                        {"messages": []},  # No assistant message → text is None
                        make_standard_response("Another valid response"),
                    ]
                ]
            }
        )

        result = block.generate(dataset)

        assert len(result) == 3
        assert "test_extractor_text" in result.columns
        assert result["test_extractor_text"].tolist() == [
            "Valid response",
            None,
            "Another valid response",
        ]

    def test_expand_all_none_extraction_results(self):
        """Test that columns exist with defaults when all items have no extractable fields."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=True,
        )

        # All items have no extractable fields (no messages key)
        dataset = pd.DataFrame(
            {"agent_response": [[{"other": "data"}, {"other": "data2"}]]}
        )

        result = block.generate(dataset)

        assert len(result) == 2
        assert "test_extractor_text" in result.columns
        assert result["test_extractor_text"].tolist() == [None, None]


class TestAgentResponseExtractorBlockListResponsesExpandFalse:
    """Test AgentResponseExtractorBlock with list responses and expand_lists=False."""

    def test_preserve_list_structure(self):
        """Test preserving list structure in output."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=False,
        )

        responses = [
            make_standard_response("Response 1"),
            make_standard_response("Response 2"),
            make_standard_response("Response 3"),
        ]
        dataset = pd.DataFrame(
            {"agent_response": [responses], "other_col": ["original_value"]}
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert result["test_extractor_text"][0] == [
            "Response 1",
            "Response 2",
            "Response 3",
        ]
        assert result["other_col"][0] == "original_value"

    def test_preserve_multiple_fields(self):
        """Test preserving multiple fields as lists."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=True,
            expand_lists=False,
        )

        dataset = pd.DataFrame(
            {
                "agent_response": [
                    [
                        make_standard_response("Response 1", "session-1"),
                        make_standard_response("Response 2", "session-2"),
                    ]
                ]
            }
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert result["test_extractor_text"][0] == ["Response 1", "Response 2"]
        assert result["test_extractor_session_id"][0] == ["session-1", "session-2"]

    def test_preserve_empty_list(self):
        """Test handling empty list with preserve structure."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=False,
        )

        dataset = pd.DataFrame({"agent_response": [[]], "other_col": ["value"]})

        result = block.generate(dataset)

        assert len(result) == 0

    def test_preserve_all_none_extraction_results(self):
        """Test that columns exist with defaults when all items have no extractable fields."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=False,
        )

        dataset = pd.DataFrame(
            {"agent_response": [[{"other": "data"}, {"other": "data2"}]]}
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert "test_extractor_text" in result.columns
        assert result["test_extractor_text"][0] == [None, None]


class TestAgentResponseExtractorBlockValidation:
    """Test AgentResponseExtractorBlock validation."""

    def test_validation_single_input_column(self):
        """Test validation with single input column."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
        )

        dataset = pd.DataFrame({"agent_response": [make_standard_response("test")]})

        # Should not raise any exception
        block._validate_custom(dataset)

    def test_validation_multiple_input_columns_warning(self, caplog):
        """Test validation warning with multiple input columns."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols=["col1", "col2"],
        )

        dataset = pd.DataFrame(
            {
                "col1": [make_standard_response("test1")],
                "col2": [make_standard_response("test2")],
            }
        )

        block._validate_custom(dataset)

        assert "expects exactly one input column" in caplog.text
        assert "Using the first column" in caplog.text

    def test_validation_no_input_columns(self):
        """Test validation fails with no input columns."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols=[],
        )

        dataset = pd.DataFrame({"other_col": ["value"]})

        with pytest.raises(ValueError, match="expects at least one input column"):
            block._validate_custom(dataset)


class TestAgentResponseExtractorBlockErrorHandling:
    """Test AgentResponseExtractorBlock error handling."""

    def test_invalid_input_type(self, caplog):
        """Test handling invalid input data type."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
        )

        dataset = pd.DataFrame({"agent_response": ["not_a_dict_or_list"]})

        result = block.generate(dataset)

        assert len(result) == 0
        assert "invalid data type" in caplog.text

    def test_empty_dataset(self, caplog):
        """Test handling empty dataset."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
        )

        dataset = pd.DataFrame({"agent_response": []})

        result = block.generate(dataset)

        assert len(result) == 0
        assert "No samples to process" in caplog.text

    def test_no_fields_extracted_produces_default_columns(self):
        """Test that columns exist with defaults even when no fields are extracted."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
        )

        # Response with no extractable fields (no messages key)
        dataset = pd.DataFrame({"agent_response": [{"other_field": "value"}]})

        result = block.generate(dataset)

        assert len(result) == 1
        assert "test_extractor_text" in result.columns
        assert result["test_extractor_text"][0] is None

    def test_empty_content_handled_gracefully(self):
        """Test handling when assistant message has empty content."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
        )

        response = ChatAgentResponse(
            messages=[
                ChatAgentMessage(role="assistant", content="", id=str(uuid.uuid4()))
            ],
        ).model_dump()
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert len(result) == 1
        assert result.iloc[0]["test_extractor_text"] == ""


class TestAgentResponseExtractorBlockIntegration:
    """Test AgentResponseExtractorBlock integration scenarios."""

    def test_integration_agentblock_to_extractor(self):
        """Test integration with typical AgentBlock output format."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
        )

        # Simulate AgentBlock output (model_dump() of ChatAgentResponse)
        dataset = pd.DataFrame(
            {
                "question": ["What is 2+2?"],
                "agent_response": [make_standard_response("The answer is 4.")],
            }
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert "test_extractor_text" in result.columns.tolist()
        assert "question" in result.columns.tolist()
        assert result["test_extractor_text"][0] == "The answer is 4."

    def test_integration_batch_responses(self):
        """Test processing multiple responses (batch scenario)."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=True,
        )

        dataset = pd.DataFrame(
            {
                "question": ["Q1", "Q2", "Q3"],
                "agent_response": [
                    make_standard_response("Answer 1", "session-1"),
                    make_standard_response("Answer 2", "session-2"),
                    make_standard_response("Answer 3", "session-3"),
                ],
            }
        )

        result = block.generate(dataset)

        assert len(result) == 3
        assert result["test_extractor_text"].tolist() == [
            "Answer 1",
            "Answer 2",
            "Answer 3",
        ]
        assert result["test_extractor_session_id"].tolist() == [
            "session-1",
            "session-2",
            "session-3",
        ]


class TestAgentResponseExtractorToolTrace:
    """Test extract_tool_trace feature."""

    def _make_tool_response(self, text="answer"):
        """Create a response with tool calls for testing."""
        tc_id = str(uuid.uuid4())
        tool_messages = [
            ChatAgentMessage(
                role="assistant",
                content="",
                id=str(uuid.uuid4()),
                tool_calls=[
                    ToolCall(
                        id=tc_id,
                        type="function",
                        function=Function(
                            name="search",
                            arguments=json.dumps({"q": "laptops"}),
                        ),
                    )
                ],
            ),
            ChatAgentMessage(
                role="tool",
                name="search",
                content='{"results": []}',
                id=str(uuid.uuid4()),
                tool_call_id=tc_id,
            ),
        ]
        return make_standard_response(text, tool_messages=tool_messages)

    def test_extract_tool_trace(self):
        """Test extracting tool trace from standardized response."""
        block = AgentResponseExtractorBlock(
            block_name="ext",
            input_cols="agent_response",
            extract_text=False,
            extract_tool_trace=True,
        )
        response = self._make_tool_response()
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert "ext_tool_trace" in result.columns
        trace = result["ext_tool_trace"].iloc[0]
        assert len(trace) == 2  # assistant with tool_calls + tool result
        assert trace[0]["role"] == "assistant"
        assert trace[0]["tool_calls"][0]["function"]["name"] == "search"
        assert trace[1]["role"] == "tool"
        assert trace[1]["name"] == "search"

    def test_extract_tool_trace_with_text(self):
        """Test extracting both text and tool trace."""
        block = AgentResponseExtractorBlock(
            block_name="ext",
            input_cols="agent_response",
            extract_text=True,
            extract_tool_trace=True,
        )
        response = self._make_tool_response("the answer")
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert result["ext_text"].iloc[0] == "the answer"
        assert len(result["ext_tool_trace"].iloc[0]) == 2

    def test_extract_tool_trace_missing(self, caplog):
        """Test that tool_trace column exists with empty list when no tool messages exist."""
        block = AgentResponseExtractorBlock(
            block_name="ext",
            input_cols="agent_response",
            extract_text=True,
            extract_tool_trace=True,
        )
        response = make_standard_response("answer")
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert result["ext_text"].iloc[0] == "answer"
        assert "ext_tool_trace" in result.columns
        assert result["ext_tool_trace"].iloc[0] == []
        assert "tool_trace" in caplog.text

    def test_init_tool_trace_only(self):
        """Test initializing with only extract_tool_trace."""
        block = AgentResponseExtractorBlock(
            block_name="ext",
            input_cols="agent_response",
            extract_text=False,
            extract_tool_trace=True,
        )
        assert block.extract_tool_trace is True
        assert "ext_tool_trace" in block.output_cols


class TestColumnExistenceWithAllNoneResults:
    """Test that extraction columns always exist even when all results are None."""

    def test_tool_trace_column_exists_when_all_none(self):
        """tool_trace column must exist with [] when no responses have tool calls."""
        block = AgentResponseExtractorBlock(
            block_name="ext",
            input_cols="agent_response",
            extract_text=True,
            extract_tool_trace=True,
        )
        dataset = pd.DataFrame(
            {
                "agent_response": [
                    make_standard_response("answer1"),
                    make_standard_response("answer2"),
                ]
            }
        )

        result = block.generate(dataset)

        assert "ext_tool_trace" in result.columns
        assert result["ext_tool_trace"].tolist() == [[], []]
        assert result["ext_text"].tolist() == ["answer1", "answer2"]

    def test_session_id_column_exists_when_all_none(self):
        """session_id column must exist with None when no responses have session_id."""
        block = AgentResponseExtractorBlock(
            block_name="ext",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=True,
        )
        dataset = pd.DataFrame(
            {
                "agent_response": [
                    make_standard_response("answer1"),
                    make_standard_response("answer2"),
                ]
            }
        )

        result = block.generate(dataset)

        assert "ext_session_id" in result.columns
        assert result["ext_session_id"].tolist() == [None, None]
        assert result["ext_text"].tolist() == ["answer1", "answer2"]

    def test_text_column_exists_when_all_none(self):
        """text column must exist with None when no responses have text."""
        block = AgentResponseExtractorBlock(
            block_name="ext",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=False,
        )
        dataset = pd.DataFrame(
            {
                "agent_response": [
                    {"messages": []},
                    {"messages": []},
                ]
            }
        )

        result = block.generate(dataset)

        assert "ext_text" in result.columns
        assert result["ext_text"].tolist() == [None, None]

    def test_all_columns_exist_when_all_none(self):
        """All three extraction columns must exist when all extractions return None."""
        block = AgentResponseExtractorBlock(
            block_name="ext",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=True,
            extract_tool_trace=True,
        )
        dataset = pd.DataFrame(
            {
                "agent_response": [
                    {"messages": []},
                ]
            }
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert "ext_text" in result.columns
        assert "ext_session_id" in result.columns
        assert "ext_tool_trace" in result.columns
        assert result["ext_text"][0] is None
        assert result["ext_session_id"][0] is None
        assert result["ext_tool_trace"][0] == []

    def test_tool_trace_column_in_list_preserve_mode(self):
        """tool_trace column must exist in preserve mode when all results are None."""
        block = AgentResponseExtractorBlock(
            block_name="ext",
            input_cols="agent_response",
            extract_text=True,
            extract_tool_trace=True,
            expand_lists=False,
        )
        dataset = pd.DataFrame(
            {
                "agent_response": [
                    [
                        make_standard_response("answer1"),
                        make_standard_response("answer2"),
                    ]
                ]
            }
        )

        result = block.generate(dataset)

        assert "ext_tool_trace" in result.columns
        assert result["ext_tool_trace"][0] == [[], []]
        assert result["ext_text"][0] == ["answer1", "answer2"]
