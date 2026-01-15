# Standard
from unittest.mock import patch

# First Party
from sdg_hub.core.blocks.transform import DictFieldExtractorBlock

# Third Party
import pandas as pd
import pytest


@pytest.fixture
def simple_dict_data():
    """Create simple nested dict data for testing."""
    return pd.DataFrame(
        [
            {
                "id": 1,
                "data": {
                    "message": "Hello World",
                    "metadata": {"timestamp": "2024-01-15", "user": "alice"},
                },
            },
            {
                "id": 2,
                "data": {
                    "message": "Goodbye World",
                    "metadata": {"timestamp": "2024-01-16", "user": "bob"},
                },
            },
        ]
    )


@pytest.fixture
def langflow_response_data():
    """Create Langflow-like response data for testing."""
    return pd.DataFrame(
        [
            {
                "request_id": "req-1",
                "response": {
                    "outputs": [
                        {
                            "outputs": [
                                {
                                    "artifacts": {
                                        "message": 'Here is a simple TCL code snippet that prints "Hello, World!":\n\n```tcl\nputs "Hello, World!"\n```'
                                    },
                                    "results": {
                                        "message": {
                                            "text": 'Here is a simple TCL code snippet that prints "Hello, World!":\n\n```tcl\nputs "Hello, World!"\n```',
                                            "content_blocks": [
                                                {
                                                    "title": "Agent Steps",
                                                    "contents": [
                                                        {
                                                            "text": "Write TCL code to print 'Hello, World!'",
                                                            "type": "text",
                                                            "duration": 1,
                                                        },
                                                        {
                                                            "text": 'Here is a simple TCL code snippet that prints "Hello, World!":\n\n```tcl\nputs "Hello, World!"\n```',
                                                            "type": "text",
                                                            "duration": 3,
                                                        },
                                                    ],
                                                }
                                            ],
                                        }
                                    },
                                }
                            ]
                        }
                    ]
                },
            }
        ]
    )


@pytest.fixture
def array_indexed_data():
    """Create data with arrays for testing array indexing."""
    return pd.DataFrame(
        [
            {
                "id": 1,
                "data": {
                    "items": [
                        {"name": "item1", "value": 100},
                        {"name": "item2", "value": 200},
                    ]
                },
            }
        ]
    )


def test_simple_field_extraction(simple_dict_data):
    """Test basic field extraction with simple paths."""
    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="data",
        output_cols=["message"],
        field_mapping={"message": "message"},
    )

    result = block(simple_dict_data)

    assert len(result) == 2
    assert "message" in result.columns
    assert result.iloc[0]["message"] == "Hello World"
    assert result.iloc[1]["message"] == "Goodbye World"


def test_nested_field_extraction(simple_dict_data):
    """Test extraction of nested fields using dot notation."""
    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="data",
        output_cols=["timestamp", "user"],
        field_mapping={"timestamp": "metadata.timestamp", "user": "metadata.user"},
    )

    result = block(simple_dict_data)

    assert len(result) == 2
    assert result.iloc[0]["timestamp"] == "2024-01-15"
    assert result.iloc[0]["user"] == "alice"
    assert result.iloc[1]["timestamp"] == "2024-01-16"
    assert result.iloc[1]["user"] == "bob"


def test_array_indexing(array_indexed_data):
    """Test field extraction with array indexing."""
    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="data",
        output_cols=["first_name", "second_value"],
        field_mapping={
            "first_name": "items[0].name",
            "second_value": "items[1].value",
        },
    )

    result = block(array_indexed_data)

    assert len(result) == 1
    assert result.iloc[0]["first_name"] == "item1"
    assert result.iloc[0]["second_value"] == 200


def test_langflow_response_extraction(langflow_response_data):
    """Test extraction from Langflow-like response structure."""
    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="response",
        output_cols=["message_text", "agent_steps"],
        field_mapping={
            "message_text": "outputs[0].outputs[0].artifacts.message",
            "agent_steps": "outputs[0].outputs[0].results.message.content_blocks",
        },
    )

    result = block(langflow_response_data)

    assert len(result) == 1
    assert "message_text" in result.columns
    assert "agent_steps" in result.columns
    assert 'puts "Hello, World!"' in result.iloc[0]["message_text"]
    assert isinstance(result.iloc[0]["agent_steps"], list)
    assert len(result.iloc[0]["agent_steps"]) == 1


def test_langflow_nested_content_extraction(langflow_response_data):
    """Test extraction of deeply nested Langflow content."""
    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="response",
        output_cols=["input_text", "output_text"],
        field_mapping={
            "input_text": "outputs[0].outputs[0].results.message.content_blocks[0].contents[0].text",
            "output_text": "outputs[0].outputs[0].results.message.content_blocks[0].contents[1].text",
        },
    )

    result = block(langflow_response_data)

    assert len(result) == 1
    assert result.iloc[0]["input_text"] == "Write TCL code to print 'Hello, World!'"
    assert 'puts "Hello, World!"' in result.iloc[0]["output_text"]


def test_missing_field_with_default():
    """Test handling of missing fields with default values."""
    data = pd.DataFrame([{"id": 1, "data": {"message": "Hello"}}])

    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="data",
        output_cols=["message", "missing_field"],
        field_mapping={"message": "message", "missing_field": "does.not.exist"},
        default_values={"message": "", "missing_field": "default_value"},
        strict_mode=False,
    )

    result = block(data)

    assert len(result) == 1
    assert result.iloc[0]["message"] == "Hello"
    assert result.iloc[0]["missing_field"] == "default_value"


def test_missing_field_strict_mode():
    """Test that strict mode raises error for missing fields."""
    data = pd.DataFrame([{"id": 1, "data": {"message": "Hello"}}])

    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="data",
        output_cols=["missing_field"],
        field_mapping={"missing_field": "does.not.exist"},
        strict_mode=True,
    )

    with pytest.raises(ValueError, match="Failed to extract field"):
        block(data)


def test_empty_dict():
    """Test handling of empty dict."""
    data = pd.DataFrame([{"id": 1, "data": {}}])

    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="data",
        output_cols=["message"],
        field_mapping={"message": "message"},
        default_values={"message": None},
        strict_mode=False,
    )

    result = block(data)

    assert len(result) == 1
    assert result.iloc[0]["message"] is None


def test_list_of_dicts():
    """Test handling of list of dicts input."""
    data = pd.DataFrame(
        [
            {
                "id": 1,
                "data": [
                    {"message": "First"},
                    {"message": "Second"},
                ],
            }
        ]
    )

    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="data",
        output_cols=["message"],
        field_mapping={"message": "message"},
    )

    result = block(data)

    # Should create two rows, one for each dict in the list
    assert len(result) == 2
    assert result.iloc[0]["message"] == "First"
    assert result.iloc[1]["message"] == "Second"


def test_invalid_input_type():
    """Test handling of invalid input type."""
    data = pd.DataFrame([{"id": 1, "data": "not a dict"}])

    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="data",
        output_cols=["message"],
        field_mapping={"message": "message"},
    )

    # Should validate and raise error during __call__
    with pytest.raises(ValueError, match="must contain dict or list"):
        block(data)


def test_validation_field_mapping_keys_mismatch():
    """Test that validation fails when field_mapping keys don't match output_cols."""
    with pytest.raises(ValueError, match="field_mapping keys.*must match output_cols"):
        block = DictFieldExtractorBlock(
            block_name="test_extractor",
            input_cols="data",
            output_cols=["message", "timestamp"],
            field_mapping={"message": "message"},  # Missing "timestamp"
        )
        # Trigger validation
        data = pd.DataFrame([{"data": {}}])
        block(data)


def test_validation_default_values_keys_mismatch():
    """Test that validation fails when default_values keys don't match output_cols."""
    with pytest.raises(ValueError, match="default_values keys.*must match output_cols"):
        block = DictFieldExtractorBlock(
            block_name="test_extractor",
            input_cols="data",
            output_cols=["message"],
            field_mapping={"message": "message"},
            default_values={"message": "", "extra": "value"},  # Extra key
        )
        # Trigger validation
        data = pd.DataFrame([{"data": {}}])
        block(data)


def test_validation_multiple_input_cols():
    """Test that validation fails with multiple input columns."""
    with pytest.raises(ValueError, match="exactly one input column"):
        block = DictFieldExtractorBlock(
            block_name="test_extractor",
            input_cols=["data1", "data2"],
            output_cols=["message"],
            field_mapping={"message": "message"},
        )
        # Trigger validation
        data = pd.DataFrame([{"data1": {}, "data2": {}}])
        block(data)


def test_validation_empty_field_mapping():
    """Test that empty field_mapping raises validation error."""
    with pytest.raises(ValueError, match="field_mapping must be a non-empty dictionary"):
        DictFieldExtractorBlock(
            block_name="test_extractor",
            input_cols="data",
            output_cols=["message"],
            field_mapping={},
        )


def test_index_out_of_range():
    """Test handling of index out of range in array access."""
    data = pd.DataFrame(
        [{"id": 1, "data": {"items": [{"name": "item1"}]}}]  # Only one item
    )

    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="data",
        output_cols=["second_item"],
        field_mapping={"second_item": "items[5].name"},  # Index out of range
        default_values={"second_item": None},
        strict_mode=False,
    )

    result = block(data)

    assert len(result) == 1
    assert result.iloc[0]["second_item"] is None


def test_none_value_in_path():
    """Test handling of None value in the middle of a path."""
    data = pd.DataFrame([{"id": 1, "data": {"nested": None}}])

    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="data",
        output_cols=["value"],
        field_mapping={"value": "nested.field"},
        default_values={"value": "fallback"},
        strict_mode=False,
    )

    result = block(data)

    assert len(result) == 1
    assert result.iloc[0]["value"] == "fallback"


def test_parse_path_segment():
    """Test the _parse_path_segment helper method."""
    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="data",
        output_cols=["test"],
        field_mapping={"test": "test"},
    )

    # Test simple key
    key, index = block._parse_path_segment("simple_key")
    assert key == "simple_key"
    assert index is None

    # Test key with array index
    key, index = block._parse_path_segment("items[0]")
    assert key == "items"
    assert index == 0

    # Test key with larger index
    key, index = block._parse_path_segment("items[123]")
    assert key == "items"
    assert index == 123


def test_empty_dataset():
    """Test handling of empty dataset."""
    data = pd.DataFrame()

    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="data",
        output_cols=["message"],
        field_mapping={"message": "message"},
    )

    result = block(data)

    assert len(result) == 0


def test_multiple_output_columns():
    """Test extraction of multiple output columns simultaneously."""
    data = pd.DataFrame(
        [
            {
                "id": 1,
                "data": {
                    "field1": "value1",
                    "field2": "value2",
                    "nested": {"field3": "value3"},
                },
            }
        ]
    )

    block = DictFieldExtractorBlock(
        block_name="test_extractor",
        input_cols="data",
        output_cols=["out1", "out2", "out3"],
        field_mapping={
            "out1": "field1",
            "out2": "field2",
            "out3": "nested.field3",
        },
    )

    result = block(data)

    assert len(result) == 1
    assert result.iloc[0]["out1"] == "value1"
    assert result.iloc[0]["out2"] == "value2"
    assert result.iloc[0]["out3"] == "value3"
