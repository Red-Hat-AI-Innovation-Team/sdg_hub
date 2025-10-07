# SPDX-License-Identifier: Apache-2.0
"""Tests for LLMParserBlock."""

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks.llm import LLMParserBlock
import pytest


class TestLLMParserBlockInitialization:
    """Test LLMParserBlock initialization."""

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        block = LLMParserBlock(
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
        block = LLMParserBlock(
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
            LLMParserBlock(
                block_name="test_parser",
                input_cols="llm_response",
                extract_content=False,
                extract_reasoning_content=False,
                extract_tool_calls=False,
            )

    def test_field_name_computation(self):
        """Test that field names are computed correctly."""
        # Test with empty prefix (should use block name)
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            field_prefix="",
        )
        assert block._content_field == "test_parser_content"
        assert block._reasoning_content_field == "test_parser_reasoning_content"
        assert block._tool_calls_field == "test_parser_tool_calls"

        # Test with custom prefix
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            field_prefix="llm_",
        )
        assert block._content_field == "llm_content"
        assert block._reasoning_content_field == "llm_reasoning_content"
        assert block._tool_calls_field == "llm_tool_calls"


class TestLLMParserBlockSingleResponse:
    """Test LLMParserBlock with single response objects."""

    def test_extract_content_only(self):
        """Test extracting only content from single response."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            extract_reasoning_content=False,
            extract_tool_calls=False,
        )

        dataset = Dataset.from_dict(
            {"llm_response": [{"content": "Hello world"}], "other_col": ["other_value"]}
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert "test_parser_content" in result.column_names
        assert result["test_parser_content"][0] == "Hello world"
        assert result["other_col"][0] == "other_value"

    def test_extract_all_fields(self):
        """Test extracting all fields from single response."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            extract_reasoning_content=True,
            extract_tool_calls=True,
        )

        dataset = Dataset.from_dict(
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
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            field_prefix="llm_",
        )

        dataset = Dataset.from_dict({"llm_response": [{"content": "Hello world"}]})

        result = block.generate(dataset)

        assert len(result) == 1
        assert "llm_content" in result.column_names
        assert result["llm_content"][0] == "Hello world"

    def test_missing_fields_partial_extraction(self, caplog):
        """Test that partial field extraction works when some fields are missing."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            extract_reasoning_content=True,
        )

        dataset = Dataset.from_dict(
            {
                "llm_response": [
                    {"content": "Hello world"}
                ]  # Missing reasoning_content
            }
        )

        result = block.generate(dataset)

        assert len(result) == 1
        assert result["test_parser_content"][0] == "Hello world"
        # Only content field should be present since reasoning_content was missing
        assert "test_parser_content" in result.column_names
        # reasoning_content column should not be created since no valid values were found
        assert "test_parser_reasoning_content" not in result.column_names

        # Should log warning about missing field
        assert (
            "Requested fields ['reasoning_content'] not found in response"
            in caplog.text
        )

    def test_multiple_missing_fields_warnings(self, caplog):
        """Test that warnings are logged for multiple missing fields."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            extract_reasoning_content=True,
            extract_tool_calls=True,
        )

        dataset = Dataset.from_dict(
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


class TestLLMParserBlockListResponsesExpandTrue:
    """Test LLMParserBlock with list responses and expand_lists=True."""

    def test_expand_list_responses(self):
        """Test expanding list of responses into individual rows."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=True,
        )

        dataset = Dataset.from_dict(
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
        assert result["test_parser_content"] == [
            "Response 1",
            "Response 2",
            "Response 3",
        ]
        assert result["other_col"] == [
            "original_value",
            "original_value",
            "original_value",
        ]

    def test_expand_multiple_samples(self):
        """Test expanding multiple samples with list responses."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=True,
        )

        dataset = Dataset.from_dict(
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
        assert result["test_parser_content"] == [
            "Sample 1 Response 1",
            "Sample 1 Response 2",
            "Sample 2 Response 1",
        ]
        assert result["sample_id"] == [1, 1, 2]

    def test_expand_empty_list(self):
        """Test handling empty list responses."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=True,
        )

        dataset = Dataset.from_dict({"llm_response": [[]], "other_col": ["value"]})

        result = block.generate(dataset)

        assert len(result) == 0

    def test_expand_invalid_list_items(self, caplog):
        """Test handling invalid items in list responses."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=True,
        )

        # Test with separate datasets since PyArrow doesn't handle mixed types well
        # Test with valid dict and dict missing content field
        dataset = Dataset.from_dict(
            {
                "llm_response": [
                    [
                        {"content": "Valid response"},
                        {
                            "other_field": "value"
                        },  # Dict without content field - will be skipped
                        {"content": "Another valid response"},
                    ]
                ]
            }
        )

        result = block.generate(dataset)

        # HuggingFace Dataset fills missing keys with None, so middle item gets content=None
        # which is converted to empty string by the parser
        assert len(result) == 3
        assert result["test_parser_content"] == [
            "Valid response",
            "",
            "Another valid response",
        ]

    def test_expand_all_invalid_list_items(self):
        """Test handling when all items in list are invalid."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=True,
        )

        # All items missing the content field
        dataset = Dataset.from_dict(
            {"llm_response": [[{"other_field": "value1"}, {"other_field": "value2"}]]}
        )

        # Should raise ValueError when no valid responses found
        with pytest.raises(ValueError, match="No valid responses found in list input"):
            block.generate(dataset)


class TestLLMParserBlockListResponsesExpandFalse:
    """Test LLMParserBlock with list responses and expand_lists=False."""

    def test_preserve_list_structure(self):
        """Test preserving list structure in output."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=False,
        )

        dataset = Dataset.from_dict(
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
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            extract_reasoning_content=True,
            expand_lists=False,
        )

        dataset = Dataset.from_dict(
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
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=False,
        )

        dataset = Dataset.from_dict({"llm_response": [[]], "other_col": ["value"]})

        result = block.generate(dataset)

        assert len(result) == 0

    def test_preserve_all_invalid_list_items(self):
        """Test handling when all items in list are invalid with preserve structure."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=False,
        )

        # All items missing the content field
        dataset = Dataset.from_dict(
            {"llm_response": [[{"other_field": "value1"}, {"other_field": "value2"}]]}
        )

        # Should raise ValueError when no valid responses found
        with pytest.raises(ValueError, match="No valid responses found in list input"):
            block.generate(dataset)


class TestLLMParserBlockValidation:
    """Test LLMParserBlock validation."""

    def test_validation_single_input_column(self):
        """Test validation with single input column."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
        )

        dataset = Dataset.from_dict({"llm_response": [{"content": "test"}]})

        # Should not raise any exception
        block._validate_custom(dataset)

    def test_validation_multiple_input_columns_warning(self, caplog):
        """Test validation warning with multiple input columns."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols=["col1", "col2"],
        )

        dataset = Dataset.from_dict(
            {"col1": [{"content": "test"}], "col2": [{"content": "test2"}]}
        )

        block._validate_custom(dataset)

        assert "expects exactly one input column" in caplog.text
        assert "Using the first column" in caplog.text

    def test_validation_no_input_columns(self):
        """Test validation fails with no input columns."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols=[],
        )

        dataset = Dataset.from_dict({"other_col": ["value"]})

        with pytest.raises(ValueError, match="expects at least one input column"):
            block._validate_custom(dataset)


class TestLLMParserBlockErrorHandling:
    """Test LLMParserBlock error handling."""

    def test_invalid_input_type(self, caplog):
        """Test handling invalid input data type."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
        )

        dataset = Dataset.from_dict({"llm_response": ["not_a_dict_or_list"]})

        result = block.generate(dataset)

        assert len(result) == 0
        assert "invalid data type" in caplog.text

    def test_empty_dataset(self, caplog):
        """Test handling empty dataset."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
        )

        dataset = Dataset.from_dict({"llm_response": []})

        result = block.generate(dataset)

        assert len(result) == 0
        assert "No samples to process" in caplog.text

    def test_no_fields_extracted(self):
        """Test handling when no fields can be extracted."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
        )

        dataset = Dataset.from_dict(
            {
                "llm_response": [{"other_field": "value"}]  # Missing content field
            }
        )

        # Should raise ValueError when no requested fields are found
        with pytest.raises(ValueError, match="No requested fields found in response"):
            block.generate(dataset)

    def test_none_content_handled_gracefully(self, caplog):
        """Test handling when content field is None."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
        )

        dataset = Dataset.from_dict(
            {
                "llm_response": [
                    {"content": None, "role": "assistant"}
                ]  # None content field
            }
        )

        result = block.generate(dataset)

        # Should not raise error and should use empty string
        assert len(result) == 1
        assert result[0]["test_parser_content"] == ""
        assert "Content field is None, using empty string instead" in caplog.text

    def test_none_reasoning_content_handled_gracefully(self, caplog):
        """Test handling when reasoning_content field is None."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_reasoning_content=True,
        )

        dataset = Dataset.from_dict(
            {
                "llm_response": [
                    {"reasoning_content": None, "role": "assistant"}
                ]  # None reasoning_content field
            }
        )

        result = block.generate(dataset)

        # Should not raise error and should use empty string
        assert len(result) == 1
        assert result[0]["test_parser_reasoning_content"] == ""
        assert (
            "Reasoning content field is None, using empty string instead" in caplog.text
        )


class TestLLMParserBlockRegistration:
    """Test LLMParserBlock registration."""

    def test_llm_parser_block_registered(self):
        """Test that LLMParserBlock is properly registered."""
        from sdg_hub.core.blocks.registry import BlockRegistry

        assert "LLMParserBlock" in BlockRegistry._metadata
        assert BlockRegistry._metadata["LLMParserBlock"].block_class == LLMParserBlock
        assert BlockRegistry._metadata["LLMParserBlock"].category == "llm"


class TestLLMParserBlockIntegration:
    """Test LLMParserBlock integration scenarios."""

    def test_integration_with_llm_chat_output(self):
        """Test integration with typical LLMChatBlock output format."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
        )

        # Simulate LLMChatBlock output with n=3
        dataset = Dataset.from_dict(
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
        assert all("test_parser_content" in row for row in result)
        assert all("messages" in row for row in result)
        assert result["test_parser_content"] == [
            "Hello! How can I help you?",
            "Hi there! What can I do for you?",
            "Hello! How may I assist you today?",
        ]

    def test_integration_preserve_lists_for_text_parser(self):
        """Test preserving lists for downstream TextParserBlock processing."""
        block = LLMParserBlock(
            block_name="test_parser",
            input_cols="llm_response",
            extract_content=True,
            expand_lists=False,
        )

        dataset = Dataset.from_dict(
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
        # This format is suitable for TextParserBlock to process each item in the list
