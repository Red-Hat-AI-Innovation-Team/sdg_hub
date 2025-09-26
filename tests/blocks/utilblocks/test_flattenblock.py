"""Tests for the FlattenColumnsBlock functionality.

This module contains tests that verify the correct behavior of the FlattenColumnsBlock,
including column melting, value mapping, and edge case handling.
"""

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks.deprecated_blocks import FlattenColumnsBlock
import pytest


def test_flatten_columns_basic():
    """Test basic flattening functionality with simple data."""
    # Create sample data
    data = {
        "id": [1, 2],
        "summary_detailed": ["detailed1", "detailed2"],
        "summary_extractive": ["extractive1", "extractive2"],
        "other_col": ["other1", "other2"],
    }
    dataset = Dataset.from_dict(data)

    # Initialize block
    block = FlattenColumnsBlock(
        block_name="test_flatten",
        var_cols=["summary_detailed", "summary_extractive"],
        value_name="summary",
        var_name="dataset_type",
    )

    # Generate flattened dataset
    result = block.generate(dataset)

    # Verify results
    assert len(result) == 4  # 2 rows * 2 columns to flatten
    assert "id" in result.column_names
    assert "other_col" in result.column_names
    assert "summary" in result.column_names
    assert "dataset_type" in result.column_names

    # Check specific values
    result_dict = result.to_dict()
    assert "detailed1" in result_dict["summary"]
    assert "extractive1" in result_dict["summary"]
    assert "summary_detailed" in result_dict["dataset_type"]
    assert "summary_extractive" in result_dict["dataset_type"]


def test_flatten_columns_with_empty_dataset():
    """Test flattening with an empty dataset."""
    data = {"id": [], "summary_detailed": [], "summary_extractive": [], "other_col": []}
    dataset = Dataset.from_dict(data)

    block = FlattenColumnsBlock(
        block_name="test_flatten_empty",
        var_cols=["summary_detailed", "summary_extractive"],
        value_name="summary",
        var_name="dataset_type",
    )

    result = block.generate(dataset)
    assert len(result) == 0
    assert all(
        col in result.column_names
        for col in ["id", "other_col", "summary", "dataset_type"]
    )


def test_flatten_columns_with_missing_values():
    """Test flattening with missing values in the data."""
    data = {
        "id": [1, 2, 3],
        "summary_detailed": ["detailed1", None, "detailed3"],
        "summary_extractive": ["extractive1", "extractive2", None],
        "other_col": ["other1", "other2", "other3"],
    }
    dataset = Dataset.from_dict(data)

    block = FlattenColumnsBlock(
        block_name="test_flatten_missing",
        var_cols=["summary_detailed", "summary_extractive"],
        value_name="summary",
        var_name="dataset_type",
    )

    result = block.generate(dataset)
    assert len(result) == 6  # 3 rows * 2 columns to flatten
    assert None in result["summary"]


def test_flatten_columns_with_all_columns():
    """Test flattening all columns except id."""
    data = {
        "id": [1, 2],
        "summary_detailed": ["detailed1", "detailed2"],
        "summary_extractive": ["extractive1", "extractive2"],
        "summary_atomic_facts": ["atomic1", "atomic2"],
        "base_document": ["base1", "base2"],
    }
    dataset = Dataset.from_dict(data)

    block = FlattenColumnsBlock(
        block_name="test_flatten_all",
        var_cols=[
            "summary_detailed",
            "summary_extractive",
            "summary_atomic_facts",
            "base_document",
        ],
        value_name="summary",
        var_name="dataset_type",
    )

    result = block.generate(dataset)
    assert len(result) == 8  # 2 rows * 4 columns to flatten
    assert len(result.column_names) == 3  # id, dataset_type, and summary columns
    assert "id" in result.column_names
    assert "summary" in result.column_names
    assert "dataset_type" in result.column_names


def test_flatten_columns_with_invalid_input():
    """Test flattening with invalid input data."""
    data = {"id": [1, 2], "summary_detailed": ["detailed1", "detailed2"]}
    dataset = Dataset.from_dict(data)

    # Test with non-existent column
    block = FlattenColumnsBlock(
        block_name="test_flatten_invalid",
        var_cols=["non_existent_column"],
        value_name="summary",
        var_name="dataset_type",
    )

    with pytest.raises(KeyError):
        block.generate(dataset)


def test_flatten_columns_with_empty_columns():
    """Test flattening with columns containing all None values."""
    data = {
        "id": [1, 2],
        "summary_detailed": [None, None],
        "summary_extractive": ["extractive1", "extractive2"],
        "other_col": ["other1", "other2"],
    }
    dataset = Dataset.from_dict(data)

    block = FlattenColumnsBlock(
        block_name="test_flatten_empty_cols",
        var_cols=["summary_detailed", "summary_extractive"],
        value_name="summary",
        var_name="dataset_type",
    )

    result = block.generate(dataset)
    result_dict = result.to_dict()

    # Verify that None values are preserved
    assert None in result_dict["summary"]
    # Verify that the column with None values still appears in dataset_type
    assert "summary_detailed" in result_dict["dataset_type"]


def test_flatten_columns_with_different_lengths():
    """Test flattening with columns of different lengths."""
    # Create data with different lengths
    data = {
        "id": [1, 2, 3],
        "summary_detailed": ["detailed1", "detailed2"],  # Shorter than id
        "summary_extractive": ["extractive1", "extractive2", "extractive3"],
        "other_col": ["other1", "other2", "other3"],
    }

    # Dataset creation will fail due to different lengths
    with pytest.raises(Exception) as exc_info:
        Dataset.from_dict(data)
    assert "expected length" in str(exc_info.value)


def test_flatten_columns_verify_column_names():
    """Test that value_name and var_name are used correctly in the output."""
    data = {
        "id": [1],
        "summary_detailed": ["detailed1"],
        "summary_extractive": ["extractive1"],
    }
    dataset = Dataset.from_dict(data)

    # Test with custom names
    custom_value_name = "custom_value"
    custom_var_name = "custom_var"

    block = FlattenColumnsBlock(
        block_name="test_flatten_custom_names",
        var_cols=["summary_detailed", "summary_extractive"],
        value_name=custom_value_name,
        var_name=custom_var_name,
    )

    result = block.generate(dataset)

    # Verify custom column names are used
    assert custom_value_name in result.column_names
    assert custom_var_name in result.column_names
    assert "summary" not in result.column_names
    assert "dataset_type" not in result.column_names

    # Verify content is correctly mapped to custom names
    result_dict = result.to_dict()
    assert "detailed1" in result_dict[custom_value_name]
    assert "extractive1" in result_dict[custom_value_name]
    assert "summary_detailed" in result_dict[custom_var_name]
    assert "summary_extractive" in result_dict[custom_var_name]
