# SPDX-License-Identifier: Apache-2.0
"""Tests for LLMResponseExtractorBlock."""

# Third Party
# First Party
import pandas as pd
import pytest

from sdg_hub.core.blocks.llm import LLMResponseExtractorBlock


class TestLLMResponseExtractorBlockInitialization:
    """Test LLMResponseExtractorBlock initialization."""

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
        )

        assert block.block_name == "test_parser"
        assert block.input_cols == ["llm_response"]
        assert block.extract_content is True
        assert block.extract_reasoning_content is False
        assert block.extract_tool_calls is False
        assert block.expand_lists is True
        assert block.field_prefix == ""

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            extract_reasoning_content=True,
            extract_tool_calls=True,
            expand_lists=False,
            field_prefix="llm_",
        )

        assert block.extract_content is True
        assert block.extract_reasoning_content is True
        assert block.extract_tool_calls is True
        assert block.expand_lists is False
        assert block.field_prefix == "llm_"

    def test_init_no_extraction_fields_enabled(self):
        """Test that initialization fails when no extraction fields are enabled."""
        with pytest.raises(ValueError, match="at least one extraction field"):
            LLMResponseExtractorBlock(
                block_name="test_parser",
                input_cols="llm_response",
                extract_content=False,
                extract_reasoning_content=False,
                extract_tool_calls=False,
            )

    def test_field_name_computation(self):
        """Test that field names are computed correctly."""
        # Test with empty prefix (should use block name)
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            field_prefix="",
        )
        assert block._content_field == "test_parser_content"
        assert block._reasoning_content_field == "test_parser_reasoning_content"
        assert block._tool_calls_field == "test_parser_tool_calls"

        # Test with custom prefix
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            field_prefix="llm_",
        )
        assert block._content_field == "llm_content"
        assert block._reasoning_content_field == "llm_reasoning_content"
        assert block._tool_calls_field == "llm_tool_calls"


class TestLLMResponseExtractorBlockSingleResponse:
    """Test LLMResponseExtractorBlock with single response objects."""

    def test_extract_content_only(self):
        """Test extracting only content from single response."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            extract_reasoning_content=False,
            extract_tool_calls=False,
        )

        dataset = pd.DataFrame(
            {"llm_response": [{"content": "Hello world"}], "other_col": ["other_value"]}
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert "test_parser_content" in result.columns.tolist()
        assert result["test_parser_content"][0] == "Hello world"
        assert result["other_col"][0] == "other_value"

    def test_extract_all_fields(self):
        """Test extracting all fields from single response."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            extract_reasoning_content=True,
            extract_tool_calls=True,
        )

        dataset = pd.DataFrame(
            {
                "llm_response": [
                    {
                        "content": "Hello world",
                        "reasoning_content": "I said hello",
                        "tool_calls": [{"name": "test_tool"}],
                    }
                ]
            }
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert result["test_parser_content"][0] == "Hello world"
        assert result["test_parser_reasoning_content"][0] == "I said hello"
        assert result["test_parser_tool_calls"][0] == [{"name": "test_tool"}]

    def test_extract_with_custom_prefix(self):
        """Test extracting with custom field prefix."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            field_prefix="llm_",
        )

        dataset = pd.DataFrame({"llm_response": [{"content": "Hello world"}]})

        result = block.generate(dataset)

        assert len(result) == 1
        assert "llm_content" in result.columns.tolist()
        assert result["llm_content"][0] == "Hello world"

    def test_missing_fields_partial_extraction(self, caplog):
        """Test that missing fields get default values and columns always exist."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            extract_reasoning_content=True,
        )

        dataset = pd.DataFrame(
            {
                "llm_response": [
                    {"content": "Hello world"}
                ]  # Missing reasoning_content
            }
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert result["test_parser_content"][0] == "Hello world"
        assert "test_parser_reasoning_content" in result.columns.tolist()
        assert result["test_parser_reasoning_content"][0] == ""

        # Should log warning about missing field
        assert (
            "Requested fields ['reasoning_content'] not found in response"
            in caplog.text
        )

    def test_multiple_missing_fields_warnings(self, caplog):
        """Test that warnings are logged for multiple missing fields."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            extract_reasoning_content=True,
            extract_tool_calls=True,
        )

        dataset = pd.DataFrame(
            {
                "llm_response": [
                    {"content": "Hello world"}
                ]  # Missing reasoning_content and tool_calls
            }
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert result["test_parser_content"][0] == "Hello world"

        # Should log warnings for both missing fields
        assert (
            "Requested fields ['reasoning_content', 'tool_calls'] not found in response"
            in caplog.text
        )


class TestLLMResponseExtractorBlockListResponsesExpandTrue:
    """Test LLMResponseExtractorBlock with list responses and expand_lists=True."""

    def test_expand_list_responses(self):
        """Test expanding list of responses into individual rows."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=True,
        )

        dataset = pd.DataFrame(
            {
                "llm_response": [
                    [
                        {"content": "Response 1"},
                        {"content": "Response 2"},
                        {"content": "Response 3"},
                    ]
                ],
                "other_col": ["original_value"],
            }
        )

        result = block.generate(dataset)

        assert len(result) == 3
        assert result["test_parser_content"].tolist() == [
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
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=True,
        )

        dataset = pd.DataFrame(
            {
                "llm_response": [
                    [
                        {"content": "Sample 1 Response 1"},
                        {"content": "Sample 1 Response 2"},
                    ],
                    [{"content": "Sample 2 Response 1"}],
                ],
                "sample_id": [1, 2],
            }
        )

        result = block.generate(dataset)

        assert len(result) == 3
        assert result["test_parser_content"].tolist() == [
            "Sample 1 Response 1",
            "Sample 1 Response 2",
            "Sample 2 Response 1",
        ]
        assert result["sample_id"].tolist() == [1, 1, 2]

    def test_expand_empty_list(self):
        """Test handling empty list responses."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=True,
        )

        dataset = pd.DataFrame({"llm_response": [[]], "other_col": ["value"]})

        result = block.generate(dataset)

        assert len(result) == 0

    def test_expand_list_with_missing_content(self, caplog):
        """Test that rows with missing content get default "" instead of being dropped."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=True,
        )

        dataset = pd.DataFrame(
            {
                "llm_response": [
                    [
                        {"content": "Valid response"},
                        {"other_field": "value"},  # Missing content → gets ""
                        {"content": "Another valid response"},
                    ]
                ]
            }
        )

        result = block.generate(dataset)

        assert len(result) == 3
        assert "test_parser_content" in result.columns
        assert result["test_parser_content"].tolist() == [
            "Valid response",
            "",
            "Another valid response",
        ]

    def test_expand_all_none_extraction_results(self):
        """Test that columns exist with defaults when all items have no extractable fields."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=True,
        )

        dataset = pd.DataFrame(
            {"llm_response": [[{"other_field": "value1"}, {"other_field": "value2"}]]}
        )

        result = block.generate(dataset)

        assert len(result) == 2
        assert "test_parser_content" in result.columns
        assert result["test_parser_content"].tolist() == ["", ""]


class TestLLMResponseExtractorBlockListResponsesExpandFalse:
    """Test LLMResponseExtractorBlock with list responses and expand_lists=False."""

    def test_preserve_list_structure(self):
        """Test preserving list structure in output."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=False,
        )

        dataset = pd.DataFrame(
            {
                "llm_response": [
                    [
                        {"content": "Response 1"},
                        {"content": "Response 2"},
                        {"content": "Response 3"},
                    ]
                ],
                "other_col": ["original_value"],
            }
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert result["test_parser_content"][0] == [
            "Response 1",
            "Response 2",
            "Response 3",
        ]
        assert result["other_col"][0] == "original_value"

    def test_preserve_multiple_fields(self):
        """Test preserving multiple fields as lists."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            extract_reasoning_content=True,
            expand_lists=False,
        )

        dataset = pd.DataFrame(
            {
                "llm_response": [
                    [
                        {"content": "Response 1", "reasoning_content": "Reasoning 1"},
                        {"content": "Response 2", "reasoning_content": "Reasoning 2"},
                    ]
                ]
            }
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert result["test_parser_content"][0] == ["Response 1", "Response 2"]
        assert result["test_parser_reasoning_content"][0] == [
            "Reasoning 1",
            "Reasoning 2",
        ]

    def test_preserve_empty_list(self):
        """Test handling empty list with preserve structure."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=False,
        )

        dataset = pd.DataFrame({"llm_response": [[]], "other_col": ["value"]})

        result = block.generate(dataset)

        assert len(result) == 0

    def test_preserve_all_none_extraction_results(self):
        """Test that columns exist with defaults when all items have no extractable fields."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=False,
        )

        dataset = pd.DataFrame(
            {"llm_response": [[{"other_field": "value1"}, {"other_field": "value2"}]]}
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert "test_parser_content" in result.columns
        assert result["test_parser_content"][0] == ["", ""]


class TestLLMResponseExtractorBlockValidation:
    """Test LLMResponseExtractorBlock validation."""

    def test_validation_single_input_column(self):
        """Test validation with single input column."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
        )

        dataset = pd.DataFrame({"llm_response": [{"content": "test"}]})

        # Should not raise any exception
        block._validate_custom(dataset)

    def test_validation_multiple_input_columns_warning(self, caplog):
        """Test validation warning with multiple input columns."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols=["col1", "col2"],
        )

        dataset = pd.DataFrame(
            {"col1": [{"content": "test"}], "col2": [{"content": "test2"}]}
        )

        block._validate_custom(dataset)

        assert "expects exactly one input column" in caplog.text
        assert "Using the first column" in caplog.text

    def test_validation_no_input_columns(self):
        """Test validation fails with no input columns."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols=[],
        )

        dataset = pd.DataFrame({"other_col": ["value"]})

        with pytest.raises(ValueError, match="expects at least one input column"):
            block._validate_custom(dataset)


class TestLLMResponseExtractorBlockErrorHandling:
    """Test LLMResponseExtractorBlock error handling."""

    def test_invalid_input_type(self, caplog):
        """Test handling invalid input data type."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
        )

        dataset = pd.DataFrame({"llm_response": ["not_a_dict_or_list"]})

        result = block.generate(dataset)

        assert len(result) == 0
        assert "invalid data type" in caplog.text

    def test_empty_dataset(self, caplog):
        """Test handling empty dataset."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
        )

        dataset = pd.DataFrame({"llm_response": []})

        result = block.generate(dataset)

        assert len(result) == 0
        assert "No samples to process" in caplog.text

    def test_no_fields_extracted_produces_default_columns(self):
        """Test that columns exist with defaults even when no fields are extracted."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
        )

        dataset = pd.DataFrame(
            {
                "llm_response": [{"other_field": "value"}]  # Missing content field
            }
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert "test_parser_content" in result.columns
        assert result["test_parser_content"][0] == ""

    def test_none_content_handled_gracefully(self, caplog):
        """Test handling when content field is None."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
        )

        dataset = pd.DataFrame(
            {
                "llm_response": [
                    {"content": None, "role": "assistant"}
                ]  # None content field
            }
        )

        result = block.generate(dataset)

        # Should not raise error and should use empty string
        assert len(result) == 1
        assert result.iloc[0]["test_parser_content"] == ""
        assert "Content field is None, using empty string instead" in caplog.text

    def test_none_reasoning_content_handled_gracefully(self, caplog):
        """Test handling when reasoning_content field is None."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_reasoning_content=True,
        )

        dataset = pd.DataFrame(
            {
                "llm_response": [
                    {"reasoning_content": None, "role": "assistant"}
                ]  # None reasoning_content field
            }
        )

        result = block.generate(dataset)

        # Should not raise error and should use empty string
        assert len(result) == 1
        assert result.iloc[0]["test_parser_reasoning_content"] == ""
        assert (
            "Reasoning content field is None, using empty string instead" in caplog.text
        )


class TestLLMResponseExtractorBlockRegistration:
    """Test LLMResponseExtractorBlock registration."""

    def test_llm_response_extractor_block_registered(self):
        """Test that LLMResponseExtractorBlock is properly registered."""
        from sdg_hub.core.blocks.registry import BlockRegistry

        assert "LLMResponseExtractorBlock" in BlockRegistry._metadata
        assert (
            BlockRegistry._metadata["LLMResponseExtractorBlock"].block_class
            == LLMResponseExtractorBlock
        )
        assert BlockRegistry._metadata["LLMResponseExtractorBlock"].category == "llm"


class TestLLMResponseExtractorBlockIntegration:
    """Test LLMResponseExtractorBlock integration scenarios."""

    def test_integration_with_llm_chat_output(self):
        """Test integration with typical LLMChatBlock output format."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
        )

        # Simulate LLMChatBlock output with n=3
        dataset = pd.DataFrame(
            {
                "messages": [["user", "Hello"]],
                "llm_response": [
                    [
                        {"content": "Hello! How can I help you?"},
                        {"content": "Hi there! What can I do for you?"},
                        {"content": "Hello! How may I assist you today?"},
                    ]
                ],
            }
        )

        result = block.generate(dataset)

        assert len(result) == 3
        assert "test_parser_content" in result.columns.tolist()
        assert "messages" in result.columns.tolist()
        assert result["test_parser_content"].tolist() == [
            "Hello! How can I help you?",
            "Hi there! What can I do for you?",
            "Hello! How may I assist you today?",
        ]

    def test_integration_preserve_lists_for_parser(self):
        """Test preserving lists for downstream parser block processing."""
        block = LLMResponseExtractorBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=False,
        )

        dataset = pd.DataFrame(
            {
                "messages": [["user", "Generate 3 responses"]],
                "llm_response": [
                    [
                        {"content": "<answer>Response 1</answer>"},
                        {"content": "<answer>Response 2</answer>"},
                        {"content": "<answer>Response 3</answer>"},
                    ]
                ],
            }
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert isinstance(result["test_parser_content"][0], list)
        assert len(result["test_parser_content"][0]) == 3


class TestColumnExistenceWithAllNoneResults:
    """Test that extraction columns always exist even when all results are None."""

    def test_tool_calls_column_exists_when_all_none(self):
        """tool_calls column must exist with [] when no responses have tool calls."""
        block = LLMResponseExtractorBlock(
            block_name="ext",
            input_cols="llm_response",
            extract_content=True,
            extract_tool_calls=True,
        )
        dataset = pd.DataFrame(
            {
                "llm_response": [
                    {"content": "answer1"},
                    {"content": "answer2"},
                ]
            }
        )

        result = block.generate(dataset)

        assert "ext_tool_calls" in result.columns
        assert result["ext_tool_calls"].tolist() == [[], []]
        assert result["ext_content"].tolist() == ["answer1", "answer2"]

    def test_reasoning_content_column_exists_when_all_none(self):
        """reasoning_content column must exist with "" when no responses have it."""
        block = LLMResponseExtractorBlock(
            block_name="ext",
            input_cols="llm_response",
            extract_content=True,
            extract_reasoning_content=True,
        )
        dataset = pd.DataFrame(
            {
                "llm_response": [
                    {"content": "answer1"},
                    {"content": "answer2"},
                ]
            }
        )

        result = block.generate(dataset)

        assert "ext_reasoning_content" in result.columns
        assert result["ext_reasoning_content"].tolist() == ["", ""]
        assert result["ext_content"].tolist() == ["answer1", "answer2"]

    def test_content_column_exists_when_all_none(self):
        """content column must exist with "" when no responses have content."""
        block = LLMResponseExtractorBlock(
            block_name="ext",
            input_cols="llm_response",
            extract_content=True,
        )
        dataset = pd.DataFrame(
            {
                "llm_response": [
                    {"other_key": "data"},
                    {"other_key": "data2"},
                ]
            }
        )

        result = block.generate(dataset)

        assert "ext_content" in result.columns
        assert result["ext_content"].tolist() == ["", ""]

    def test_all_columns_exist_when_all_none(self):
        """All three extraction columns must exist when all extractions return defaults."""
        block = LLMResponseExtractorBlock(
            block_name="ext",
            input_cols="llm_response",
            extract_content=True,
            extract_reasoning_content=True,
            extract_tool_calls=True,
        )
        dataset = pd.DataFrame(
            {
                "llm_response": [
                    {"other_key": "data"},
                ]
            }
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert "ext_content" in result.columns
        assert "ext_reasoning_content" in result.columns
        assert "ext_tool_calls" in result.columns
        assert result["ext_content"][0] == ""
        assert result["ext_reasoning_content"][0] == ""
        assert result["ext_tool_calls"][0] == []

    def test_tool_calls_column_in_list_preserve_mode(self):
        """tool_calls column must exist in preserve mode when all results have no tool calls."""
        block = LLMResponseExtractorBlock(
            block_name="ext",
            input_cols="llm_response",
            extract_content=True,
            extract_tool_calls=True,
            expand_lists=False,
        )
        dataset = pd.DataFrame(
            {
                "llm_response": [
                    [
                        {"content": "answer1"},
                        {"content": "answer2"},
                    ]
                ]
            }
        )

        result = block.generate(dataset)

        assert "ext_tool_calls" in result.columns
        assert result["ext_tool_calls"][0] == [[], []]
        assert result["ext_content"][0] == ["answer1", "answer2"]

    def test_tool_calls_column_in_list_expand_mode(self):
        """tool_calls column must exist in expand mode when no responses have tool calls."""
        block = LLMResponseExtractorBlock(
            block_name="ext",
            input_cols="llm_response",
            extract_content=True,
            extract_tool_calls=True,
            expand_lists=True,
        )
        dataset = pd.DataFrame(
            {
                "llm_response": [
                    [
                        {"content": "answer1"},
                        {"content": "answer2"},
                    ]
                ]
            }
        )

        result = block.generate(dataset)

        assert len(result) == 2
        assert "ext_tool_calls" in result.columns
        assert result["ext_tool_calls"].tolist() == [[], []]
        assert result["ext_content"].tolist() == ["answer1", "answer2"]
