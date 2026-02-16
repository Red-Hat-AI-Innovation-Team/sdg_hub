# SPDX-License-Identifier: Apache-2.0
"""Tests for the JSONParserBlock functionality."""

# Third Party
# First Party
from sdg_hub.core.blocks.parsing import JSONParserBlock
import pandas as pd
import pytest


def test_basic_json_parsing():
    """Test basic JSON parsing and field expansion."""
    data = {
        "id": [1, 2],
        "json_content": [
            '{"name": "Alice", "age": 30}',
            '{"name": "Bob", "age": 25}',
        ],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_parser",
        input_cols=["json_content"],
    )

    result = block.generate(dataset)

    assert len(result) == 2
    assert "name" in result.columns
    assert "age" in result.columns
    assert result["name"].tolist() == ["Alice", "Bob"]
    assert result["age"].tolist() == [30, 25]


def test_embedded_json_extraction():
    """Test extracting JSON embedded in surrounding text."""
    data = {
        "response": [
            'Here is the result: {"prompt": "test prompt", "score": 0.9} end of response',
            'Some prefix {"prompt": "another prompt", "score": 0.8} some suffix',
        ],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_embedded",
        input_cols=["response"],
        extract_embedded=True,
    )

    result = block.generate(dataset)

    assert "prompt" in result.columns
    assert "score" in result.columns
    assert result["prompt"].tolist() == ["test prompt", "another prompt"]
    assert result["score"].tolist() == [0.9, 0.8]


def test_specific_output_columns():
    """Test extracting only specific fields."""
    data = {
        "json_content": [
            '{"name": "Alice", "age": 30, "city": "NYC"}',
        ],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_specific",
        input_cols=["json_content"],
        output_cols=["name", "city"],
    )

    result = block.generate(dataset)

    assert "name" in result.columns
    assert "city" in result.columns
    assert "age" not in result.columns
    assert result["name"].tolist() == ["Alice"]
    assert result["city"].tolist() == ["NYC"]


def test_field_prefix():
    """Test adding prefix to extracted column names."""
    data = {
        "json_content": [
            '{"name": "Alice", "age": 30}',
        ],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_prefix",
        input_cols=["json_content"],
        field_prefix="parsed_",
    )

    result = block.generate(dataset)

    assert "parsed_name" in result.columns
    assert "parsed_age" in result.columns
    assert "name" not in result.columns


def test_fix_trailing_commas():
    """Test fixing trailing commas in JSON."""
    data = {
        "json_content": [
            '{"name": "Alice", "age": 30,}',  # Trailing comma
            '{"items": ["a", "b", "c",]}',  # Trailing comma in array
        ],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_trailing",
        input_cols=["json_content"],
        fix_trailing_commas=True,
    )

    result = block.generate(dataset)

    assert "name" in result.columns
    assert result["name"].iloc[0] == "Alice"
    assert "items" in result.columns


def test_drop_input_column():
    """Test dropping input column after extraction."""
    data = {
        "id": [1],
        "json_content": ['{"name": "Alice"}'],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_drop",
        input_cols=["json_content"],
        drop_input=True,
    )

    result = block.generate(dataset)

    assert "json_content" not in result.columns
    assert "name" in result.columns
    assert "id" in result.columns


def test_keep_input_column():
    """Test keeping input column (default behavior)."""
    data = {
        "json_content": ['{"name": "Alice"}'],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_keep",
        input_cols=["json_content"],
        drop_input=False,
    )

    result = block.generate(dataset)

    assert "json_content" in result.columns
    assert "name" in result.columns


def test_invalid_json():
    """Test handling of invalid JSON."""
    data = {
        "json_content": [
            '{"name": "Alice"}',
            "not valid json at all",
            '{"name": "Bob"}',
        ],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_invalid",
        input_cols=["json_content"],
    )

    result = block.generate(dataset)

    assert len(result) == 3
    assert result["name"].iloc[0] == "Alice"
    # Invalid JSON row should have NaN/empty values
    assert pd.isna(result["name"].iloc[1]) or result["name"].iloc[1] == {}
    assert result["name"].iloc[2] == "Bob"


def test_nan_values_in_input():
    """Test handling of NaN values in input column."""
    import numpy as np

    data = {
        "json_content": [
            '{"name": "Alice"}',
            np.nan,  # NaN value
            '{"name": "Bob"}',
            None,  # None value
        ],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_nan",
        input_cols=["json_content"],
    )

    result = block.generate(dataset)

    assert len(result) == 4
    assert result["name"].iloc[0] == "Alice"
    # NaN and None rows should have NaN/empty values, not raise AttributeError
    assert pd.isna(result["name"].iloc[1]) or result["name"].iloc[1] == {}
    assert result["name"].iloc[2] == "Bob"
    assert pd.isna(result["name"].iloc[3]) or result["name"].iloc[3] == {}


def test_empty_json():
    """Test handling of empty JSON object."""
    data = {
        "json_content": ["{}"],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_empty",
        input_cols=["json_content"],
    )

    result = block.generate(dataset)

    assert len(result) == 1
    # Verify no phantom '0' column is created from empty dicts
    assert 0 not in result.columns


def test_all_empty_json_no_phantom_column():
    """Test that all empty JSON objects don't create phantom column 0."""
    data = {
        "json_content": ["{}", "{}", "{}"],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_all_empty",
        input_cols=["json_content"],
    )

    result = block.generate(dataset)

    assert len(result) == 3
    # Verify no phantom '0' column is created
    assert 0 not in result.columns
    # Should only have the original json_content column
    assert list(result.columns) == ["json_content"]


def test_nested_json():
    """Test handling of nested JSON objects."""
    data = {
        "json_content": [
            '{"user": {"name": "Alice", "age": 30}, "active": true}',
        ],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_nested",
        input_cols=["json_content"],
    )

    result = block.generate(dataset)

    assert "user" in result.columns
    assert "active" in result.columns
    assert result["user"].iloc[0] == {"name": "Alice", "age": 30}
    assert bool(result["active"].iloc[0]) is True


def test_json_array():
    """Test handling of JSON array at root level."""
    data = {
        "json_content": [
            '["item1", "item2", "item3"]',
        ],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_array",
        input_cols=["json_content"],
    )

    result = block.generate(dataset)

    # Arrays are wrapped in {"items": [...]}
    assert "items" in result.columns
    assert result["items"].iloc[0] == ["item1", "item2", "item3"]


def test_validation_requires_one_input_col():
    """Test validation error when not exactly one input column."""
    with pytest.raises(ValueError, match="requires exactly one input column"):
        JSONParserBlock(
            block_name="test",
            input_cols=[],
        )

    with pytest.raises(ValueError, match="requires exactly one input column"):
        JSONParserBlock(
            block_name="test",
            input_cols=["col1", "col2"],
        )


def test_missing_requested_columns_warning():
    """Test warning when requested output columns don't exist in JSON."""
    data = {
        "json_content": [
            '{"name": "Alice"}',
        ],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_missing",
        input_cols=["json_content"],
        output_cols=["name", "nonexistent_field"],
    )

    result = block.generate(dataset)

    # Should still work, extracting what's available
    assert "name" in result.columns
    assert result["name"].iloc[0] == "Alice"


def test_extract_embedded_false():
    """Test with extract_embedded=False (entire text must be JSON)."""
    data = {
        "json_content": [
            '{"name": "Alice"}',
            'prefix {"name": "Bob"} suffix',  # This won't parse without extraction
        ],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_no_extract",
        input_cols=["json_content"],
        extract_embedded=False,
    )

    result = block.generate(dataset)

    assert result["name"].iloc[0] == "Alice"
    # Second row should fail to parse
    assert pd.isna(result["name"].iloc[1]) or result["name"].iloc[1] == {}


def test_preserves_existing_columns():
    """Test that existing columns are preserved."""
    data = {
        "id": [1, 2],
        "category": ["A", "B"],
        "json_content": [
            '{"name": "Alice"}',
            '{"name": "Bob"}',
        ],
    }
    dataset = pd.DataFrame(data)

    block = JSONParserBlock(
        block_name="test_preserve",
        input_cols=["json_content"],
    )

    result = block.generate(dataset)

    assert "id" in result.columns
    assert "category" in result.columns
    assert "name" in result.columns
    assert result["id"].tolist() == [1, 2]
    assert result["category"].tolist() == ["A", "B"]
