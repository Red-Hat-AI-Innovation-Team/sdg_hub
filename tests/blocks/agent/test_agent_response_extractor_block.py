# SPDX-License-Identifier: Apache-2.0
"""Tests for AgentResponseExtractorBlock."""

import pandas as pd
import pytest

from sdg_hub.core.blocks.agent import AgentResponseExtractorBlock
from sdg_hub.core.blocks.registry import BlockRegistry


# Sample Langflow response structure
def make_langflow_response(text, session_id="session-123", content_blocks=None):
    """Create a sample Langflow response structure."""
    msg = {"text": text}
    if content_blocks is not None:
        msg["data"] = {"content_blocks": content_blocks}
    return {
        "session_id": session_id,
        "outputs": [{"outputs": [{"results": {"message": msg}}]}],
    }


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
            agent_framework="langflow",
            input_cols="agent_response",
        )

        assert block.block_name == "test_extractor"
        assert block.agent_framework == "langflow"
        assert block.input_cols == ["agent_response"]
        assert block.extract_text is True
        assert block.extract_session_id is False
        assert block.expand_lists is True
        assert block.field_prefix == ""

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
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
                agent_framework="langflow",
                input_cols="agent_response",
                extract_text=False,
                extract_session_id=False,
            )

    def test_init_unsupported_framework(self):
        """Test that initialization fails for unsupported framework."""
        with pytest.raises(ValueError, match="Unsupported agent_framework"):
            AgentResponseExtractorBlock(
                block_name="test_extractor",
                agent_framework="unsupported",
                input_cols="agent_response",
            )

    def test_field_name_computation(self):
        """Test that field names are computed correctly."""
        # Test with empty prefix (should use block name)
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            field_prefix="",
        )
        assert block._text_field == "test_extractor_text"
        assert block._session_id_field == "test_extractor_session_id"

        # Test with custom prefix
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            field_prefix="agent_",
        )
        assert block._text_field == "agent_text"
        assert block._session_id_field == "agent_session_id"


class TestAgentResponseExtractorBlockLangflowExtraction:
    """Test AgentResponseExtractorBlock Langflow extraction."""

    def test_extract_text_only(self):
        """Test extracting only text from Langflow response."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=False,
        )

        response = make_langflow_response("Hello world")
        dataset = pd.DataFrame(
            {"agent_response": [response], "other_col": ["other_value"]}
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert "test_extractor_text" in result.columns.tolist()
        assert result["test_extractor_text"][0] == "Hello world"
        assert result["other_col"][0] == "other_value"

    def test_extract_all_fields(self):
        """Test extracting all fields from Langflow response."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=True,
        )

        response = make_langflow_response("Hello world", "session-abc")
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert len(result) == 1
        assert result["test_extractor_text"][0] == "Hello world"
        assert result["test_extractor_session_id"][0] == "session-abc"

    def test_extract_with_custom_prefix(self):
        """Test extracting with custom field prefix."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            field_prefix="agent_",
        )

        response = make_langflow_response("Hello world")
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert len(result) == 1
        assert "agent_text" in result.columns.tolist()
        assert result["agent_text"][0] == "Hello world"

    def test_missing_text_field(self, caplog):
        """Test handling missing text field with partial extraction."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=True,
        )

        # Response with session_id but malformed outputs (no text)
        dataset = pd.DataFrame(
            {"agent_response": [{"session_id": "session-123", "outputs": []}]}
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert result["test_extractor_session_id"][0] == "session-123"
        assert "test_extractor_text" not in result.columns.tolist()
        assert "Requested fields ['text'] not found in response" in caplog.text

    def test_missing_session_id_field(self, caplog):
        """Test handling missing session_id field."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=True,
        )

        # Response with text but no session_id
        response = {
            "outputs": [{"outputs": [{"results": {"message": {"text": "Hi"}}}]}]
        }
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert len(result) == 1
        assert result["test_extractor_text"][0] == "Hi"
        assert "Requested fields ['session_id'] not found in response" in caplog.text


class TestAgentResponseExtractorBlockListResponsesExpandTrue:
    """Test AgentResponseExtractorBlock with list responses and expand_lists=True."""

    def test_expand_list_responses(self):
        """Test expanding list of responses into individual rows."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=True,
        )

        responses = [
            make_langflow_response("Response 1"),
            make_langflow_response("Response 2"),
            make_langflow_response("Response 3"),
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
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=True,
        )

        dataset = pd.DataFrame(
            {
                "agent_response": [
                    [
                        make_langflow_response("Sample 1 Response 1"),
                        make_langflow_response("Sample 1 Response 2"),
                    ],
                    [make_langflow_response("Sample 2 Response 1")],
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
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=True,
        )

        dataset = pd.DataFrame({"agent_response": [[]], "other_col": ["value"]})

        result = block.generate(dataset)

        assert len(result) == 0

    def test_expand_invalid_list_items(self, caplog):
        """Test handling invalid items in list responses."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=True,
        )

        dataset = pd.DataFrame(
            {
                "agent_response": [
                    [
                        make_langflow_response("Valid response"),
                        {"outputs": []},  # Invalid - missing text path
                        make_langflow_response("Another valid response"),
                    ]
                ]
            }
        )

        result = block.generate(dataset)

        # Only valid responses should be included
        assert len(result) == 2
        assert result["test_extractor_text"].tolist() == [
            "Valid response",
            "Another valid response",
        ]

    def test_expand_all_invalid_list_items(self):
        """Test handling when all items in list are invalid."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=True,
        )

        # All items missing the text path
        dataset = pd.DataFrame({"agent_response": [[{"outputs": []}, {"outputs": []}]]})

        with pytest.raises(ValueError, match="No valid responses found in list input"):
            block.generate(dataset)


class TestAgentResponseExtractorBlockListResponsesExpandFalse:
    """Test AgentResponseExtractorBlock with list responses and expand_lists=False."""

    def test_preserve_list_structure(self):
        """Test preserving list structure in output."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=False,
        )

        responses = [
            make_langflow_response("Response 1"),
            make_langflow_response("Response 2"),
            make_langflow_response("Response 3"),
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
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=True,
            expand_lists=False,
        )

        dataset = pd.DataFrame(
            {
                "agent_response": [
                    [
                        make_langflow_response("Response 1", "session-1"),
                        make_langflow_response("Response 2", "session-2"),
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
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=False,
        )

        dataset = pd.DataFrame({"agent_response": [[]], "other_col": ["value"]})

        result = block.generate(dataset)

        assert len(result) == 0

    def test_preserve_all_invalid_list_items(self):
        """Test handling when all items in list are invalid with preserve structure."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            expand_lists=False,
        )

        dataset = pd.DataFrame({"agent_response": [[{"outputs": []}, {"outputs": []}]]})

        with pytest.raises(ValueError, match="No valid responses found in list input"):
            block.generate(dataset)


class TestAgentResponseExtractorBlockValidation:
    """Test AgentResponseExtractorBlock validation."""

    def test_validation_single_input_column(self):
        """Test validation with single input column."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
        )

        dataset = pd.DataFrame({"agent_response": [make_langflow_response("test")]})

        # Should not raise any exception
        block._validate_custom(dataset)

    def test_validation_multiple_input_columns_warning(self, caplog):
        """Test validation warning with multiple input columns."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols=["col1", "col2"],
        )

        dataset = pd.DataFrame(
            {
                "col1": [make_langflow_response("test1")],
                "col2": [make_langflow_response("test2")],
            }
        )

        block._validate_custom(dataset)

        assert "expects exactly one input column" in caplog.text
        assert "Using the first column" in caplog.text

    def test_validation_no_input_columns(self):
        """Test validation fails with no input columns."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
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
            agent_framework="langflow",
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
            agent_framework="langflow",
            input_cols="agent_response",
        )

        dataset = pd.DataFrame({"agent_response": []})

        result = block.generate(dataset)

        assert len(result) == 0
        assert "No samples to process" in caplog.text

    def test_no_fields_extracted(self):
        """Test handling when no fields can be extracted."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
        )

        # Response with no extractable fields
        dataset = pd.DataFrame({"agent_response": [{"other_field": "value"}]})

        with pytest.raises(ValueError, match="No requested fields found in response"):
            block.generate(dataset)

    def test_none_text_handled_gracefully(self, caplog):
        """Test handling when text field is None."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
        )

        response = {
            "session_id": "abc",
            "outputs": [{"outputs": [{"results": {"message": {"text": None}}}]}],
        }
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert len(result) == 1
        assert result.iloc[0]["test_extractor_text"] == ""
        assert "Text field is None, using empty string instead" in caplog.text

    def test_none_session_id_handled_gracefully(self, caplog):
        """Test handling when session_id field is None."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_session_id=True,
        )

        response = {
            "session_id": None,
            "outputs": [{"outputs": [{"results": {"message": {"text": "Hi"}}}]}],
        }
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert len(result) == 1
        assert result.iloc[0]["test_extractor_session_id"] == ""
        assert "Session ID field is None, using empty string instead" in caplog.text


class TestAgentResponseExtractorBlockIntegration:
    """Test AgentResponseExtractorBlock integration scenarios."""

    def test_integration_agentblock_to_extractor(self):
        """Test integration with typical AgentBlock output format."""
        block = AgentResponseExtractorBlock(
            block_name="test_extractor",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
        )

        # Simulate AgentBlock output
        dataset = pd.DataFrame(
            {
                "question": ["What is 2+2?"],
                "agent_response": [make_langflow_response("The answer is 4.")],
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
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            extract_session_id=True,
        )

        dataset = pd.DataFrame(
            {
                "question": ["Q1", "Q2", "Q3"],
                "agent_response": [
                    make_langflow_response("Answer 1", "session-1"),
                    make_langflow_response("Answer 2", "session-2"),
                    make_langflow_response("Answer 3", "session-3"),
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


class TestAgentResponseExtractorToolTrace:
    """Test extract_tool_trace feature."""

    def test_extract_tool_trace(self):
        """Test extracting tool trace from content_blocks."""
        block = AgentResponseExtractorBlock(
            block_name="ext",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=False,
            extract_tool_trace=True,
        )
        response = make_langflow_response(
            "answer", content_blocks=SAMPLE_CONTENT_BLOCKS
        )
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert "ext_tool_trace" in result.columns
        trace = result["ext_tool_trace"].iloc[0]
        assert len(trace) == 3
        assert trace[0]["type"] == "text"
        assert trace[1]["type"] == "tool_use"
        assert trace[1]["name"] == "search"
        assert trace[2]["type"] == "text"

    def test_extract_tool_trace_with_text(self):
        """Test extracting both text and tool trace."""
        block = AgentResponseExtractorBlock(
            block_name="ext",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            extract_tool_trace=True,
        )
        response = make_langflow_response(
            "answer", content_blocks=SAMPLE_CONTENT_BLOCKS
        )
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert result["ext_text"].iloc[0] == "answer"
        assert len(result["ext_tool_trace"].iloc[0]) == 3

    def test_extract_tool_trace_missing(self, caplog):
        """Test graceful handling when content_blocks is missing."""
        block = AgentResponseExtractorBlock(
            block_name="ext",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=True,
            extract_tool_trace=True,
        )
        response = make_langflow_response("answer")
        dataset = pd.DataFrame({"agent_response": [response]})

        result = block.generate(dataset)

        assert result["ext_text"].iloc[0] == "answer"
        assert "tool_trace" in caplog.text

    def test_init_tool_trace_only(self):
        """Test initializing with only extract_tool_trace."""
        block = AgentResponseExtractorBlock(
            block_name="ext",
            agent_framework="langflow",
            input_cols="agent_response",
            extract_text=False,
            extract_tool_trace=True,
        )
        assert block.extract_tool_trace is True
        assert "ext_tool_trace" in block.output_cols
