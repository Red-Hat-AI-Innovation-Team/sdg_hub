"""Tests for JSONStructureBlock."""

# Standard
import json

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks.transform import JSONStructureBlock
import pytest


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_dict(
        {
            "summary": [
                "This is a short summary",
                "Another brief summary",
                "Final summary",
            ],
            "keywords": [
                ["key1", "key2", "key3"],
                ["word1", "word2"],
                ["term1", "term2", "term3"],
            ],
            "entities": [
                ["Person A", "Company B"],
                ["Location X"],
                ["Organization Y", "Person Z"],
            ],
            "sentiment": ["positive", "neutral", "negative"],
            "confidence": [0.95, 0.73, 0.88],
        }
    )


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for basic testing."""
    return Dataset.from_dict(
        {
            "text": ["Sample text 1", "Sample text 2"],
            "score": [85, 92],
            "category": ["news", "article"],
        }
    )


def test_basic_json_structure_with_list_input_cols(sample_dataset):
    """Test basic JSON structure creation using list of column names."""
    block = JSONStructureBlock(
        block_name="test_json",
        input_cols=["summary", "sentiment"],
        output_cols=["structured_output"],
    )

    result = block.generate(sample_dataset)

    assert "structured_output" in result.column_names

    # Parse the JSON output for first sample
    json_data = json.loads(result[0]["structured_output"])
    assert json_data["summary"] == "This is a short summary"
    assert json_data["sentiment"] == "positive"

    # Check all samples have valid JSON
    for i in range(len(result)):
        parsed = json.loads(result[i]["structured_output"])
        assert "summary" in parsed
        assert "sentiment" in parsed


def test_all_column_types(sample_dataset):
    """Test JSON structure with various data types."""
    block = JSONStructureBlock(
        block_name="test_json",
        input_cols=["summary", "keywords", "entities", "sentiment", "confidence"],
        output_cols=["structured_output"],
    )

    result = block.generate(sample_dataset)

    # Parse the JSON output for first sample
    json_data = json.loads(result[0]["structured_output"])

    # Check string
    assert json_data["summary"] == "This is a short summary"

    # Check list
    assert json_data["keywords"] == ["key1", "key2", "key3"]
    assert json_data["entities"] == ["Person A", "Company B"]

    # Check string again
    assert json_data["sentiment"] == "positive"

    # Check float
    assert json_data["confidence"] == 0.95


def test_pretty_print_option(simple_dataset):
    """Test JSON output with pretty printing enabled."""
    block = JSONStructureBlock(
        block_name="test_json",
        input_cols=["text", "score"],
        output_cols=["structured_output"],
        pretty_print=True,
    )

    result = block.generate(simple_dataset)

    json_output = result[0]["structured_output"]

    # Pretty printed JSON should contain newlines and indentation
    assert "\n" in json_output
    assert "  " in json_output

    # Should still be valid JSON
    parsed = json.loads(json_output)
    assert parsed["text"] == "Sample text 1"
    assert parsed["score"] == 85


def test_missing_columns_handling(simple_dataset):
    """Test behavior when specified columns don't exist in dataset."""
    block = JSONStructureBlock(
        block_name="test_json",
        input_cols=["text", "missing_col"],
        output_cols=["structured_output"],
    )

    result = block.generate(simple_dataset)

    # Should not raise error, but set missing columns to null
    json_data = json.loads(result[0]["structured_output"])
    assert json_data["text"] == "Sample text 1"
    assert json_data["missing_col"] is None


def test_json_serialization_with_complex_objects():
    """Test JSON serialization with non-serializable objects."""
    # Create dataset with complex objects
    dataset = Dataset.from_dict(
        {
            "simple_str": ["text"],
            "simple_list": [["a", "b"]],
            "simple_dict": [{"key": "value"}],
            "none_value": [None],
        }
    )

    block = JSONStructureBlock(
        block_name="test_json",
        input_cols=["simple_str", "simple_list", "simple_dict", "none_value"],
        output_cols=["structured_output"],
    )

    result = block.generate(dataset)

    json_data = json.loads(result[0]["structured_output"])
    assert json_data["simple_str"] == "text"
    assert json_data["simple_list"] == ["a", "b"]
    assert json_data["simple_dict"] == {"key": "value"}
    assert json_data["none_value"] is None


def test_disable_json_serialization_check():
    """Test disabling JSON serialization checks."""
    dataset = Dataset.from_dict(
        {
            "text": ["sample text"],
            "number": [42],
        }
    )

    block = JSONStructureBlock(
        block_name="test_json",
        input_cols=["text", "number"],
        output_cols=["structured_output"],
        ensure_json_serializable=False,
    )

    result = block.generate(dataset)

    # Should still work with basic types
    json_data = json.loads(result[0]["structured_output"])
    assert json_data["text"] == "sample text"
    assert json_data["number"] == 42


def test_invalid_output_cols():
    """Test validation with invalid output_cols."""
    with pytest.raises(
        ValueError, match="JSONStructureBlock requires exactly one output column"
    ):
        JSONStructureBlock(
            block_name="test_json",
            input_cols=["col1", "col2"],
            output_cols=["out1", "out2"],  # Should be exactly one
        )


def test_empty_output_cols():
    """Test validation with empty output_cols."""
    with pytest.raises(
        ValueError, match="JSONStructureBlock requires exactly one output column"
    ):
        JSONStructureBlock(
            block_name="test_json",
            input_cols=["col1", "col2"],
            output_cols=[],  # Should not be empty
        )


def test_empty_dataset():
    """Test behavior with empty dataset."""
    empty_dataset = Dataset.from_dict({})

    block = JSONStructureBlock(
        block_name="test_json",
        input_cols=["col1"],
        output_cols=["structured_output"],
    )

    result = block.generate(empty_dataset)
    assert len(result) == 0


def test_field_mapping_validation_error():
    """Test error when field mapping cannot be determined."""
    # This should not actually happen in normal usage due to Pydantic validation,
    # but let's test the internal method
    block = JSONStructureBlock(
        block_name="test_json",
        input_cols=["col1"],
        output_cols=["structured_output"],
    )

    # Temporarily break input_cols to test error handling
    original_input_cols = block.input_cols
    block.input_cols = "invalid_type"

    with pytest.raises(ValueError, match="input_cols must be a list of column names"):
        block._get_field_mapping()

    # Restore for cleanup
    block.input_cols = original_input_cols


def test_large_dataset_performance(sample_dataset):
    """Test performance with larger datasets."""
    # Create a larger dataset by repeating the sample
    large_data = {}
    for col in sample_dataset.column_names:
        large_data[col] = sample_dataset[col] * 100  # 300 samples total

    large_dataset = Dataset.from_dict(large_data)

    block = JSONStructureBlock(
        block_name="test_json",
        input_cols=["summary", "sentiment"],
        output_cols=["structured_output"],
    )

    result = block.generate(large_dataset)

    assert len(result) == 300
    # Spot check a few samples
    json_data_first = json.loads(result[0]["structured_output"])
    json_data_last = json.loads(result[-1]["structured_output"])

    assert "summary" in json_data_first
    assert "sentiment" in json_data_first
    assert "summary" in json_data_last
    assert "sentiment" in json_data_last


def test_unicode_and_special_characters():
    """Test handling of unicode and special characters."""
    dataset = Dataset.from_dict(
        {
            "text": ["Hello 世界", "Special chars: !@#$%^&*()"],
            "emoji": ["😀😂🔥", "🚀✨💡"],
        }
    )

    block = JSONStructureBlock(
        block_name="test_json",
        input_cols=["text", "emoji"],
        output_cols=["structured_output"],
    )

    result = block.generate(dataset)

    json_data = json.loads(result[0]["structured_output"])
    assert json_data["text"] == "Hello 世界"
    assert json_data["emoji"] == "😀😂🔥"

    json_data = json.loads(result[1]["structured_output"])
    assert json_data["text"] == "Special chars: !@#$%^&*()"
    assert json_data["emoji"] == "🚀✨💡"
