"""Tests for the RenameColumns block."""

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks import RenameColumns
import pytest


def test_rename_columns_basic():
    """Test basic column renaming functionality."""
    # Create a sample dataset
    data = {
        "document": ["doc1", "doc2"],
        "summary": ["sum1", "sum2"],
        "other": ["other1", "other2"],
    }
    dataset = Dataset.from_dict(data)

    # Initialize the block with column mapping
    block = RenameColumns(
        block_name="test_rename",
        columns_map={"document": "raw_document", "summary": "document"},
    )

    # Apply the transformation
    result = block.generate(dataset)

    # Verify the results
    assert "raw_document" in result.column_names
    assert "document" in result.column_names
    assert "other" in result.column_names
    assert "summary" not in result.column_names

    # Check values are preserved
    assert result["raw_document"] == ["doc1", "doc2"]
    assert result["document"] == ["sum1", "sum2"]
    assert result["other"] == ["other1", "other2"]


def test_rename_columns_nonexistent_column():
    """Test behavior when trying to rename a non-existent column."""
    data = {"col1": [1, 2]}
    dataset = Dataset.from_dict(data)

    block = RenameColumns(
        block_name="test_nonexistent", columns_map={"nonexistent": "new_col"}
    )

    # Should raise ValueError when trying to rename non-existent column
    with pytest.raises(
        ValueError, match="Original column names {'nonexistent'} not in the dataset"
    ):
        block.generate(dataset)


def test_rename_columns_overwrite():
    """Test behavior when renaming to an existing column name."""
    data = {"col1": [1, 2], "col2": [3, 4]}
    dataset = Dataset.from_dict(data)

    block = RenameColumns(block_name="test_overwrite", columns_map={"col1": "col2"})

    result = block.generate(dataset)

    # Verify that col1's values overwrite col2's values
    assert "col1" not in result.column_names
    assert "col2" in result.column_names
    assert result["col2"] == [1, 2]  # Values from col1


def test_rename_columns_preserve_data():
    """Test that data values are preserved after renaming."""
    data = {"old_name": ["value1", "value2", "value3"], "other_col": [1, 2, 3]}
    dataset = Dataset.from_dict(data)

    block = RenameColumns(
        block_name="test_preserve", columns_map={"old_name": "new_name"}
    )

    result = block.generate(dataset)

    # Check that values are preserved
    assert result["new_name"] == dataset["old_name"]
    assert result["other_col"] == dataset["other_col"]
